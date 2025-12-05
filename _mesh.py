# _mesh.py
import numpy as np
import pyvista as pv

def cube_vertices() -> np.ndarray:
    """기본 큐브 데이터를 생성하여 반환 (렌더링용 포맷)."""
    # 초기 큐브는 pyvista를 이용해 생성하되, 데이터만 추출합니다.
    mesh = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
    # WebGPU 렌더링을 위해 삼각형화 및 데이터 포맷팅
    return _format_mesh_for_render(mesh.triangulate())


def _linear_split_quads(points, faces):
    """
    [Step 1: Multi-linear Subdivision]
    사각형(Quad)을 4개의 작은 사각형으로 분할하고, 
    새 정점 위치를 선형 보간(Linear Interpolation)으로 배치합니다.
    """
    # 엣지 캐시: (v1, v2) -> new_vertex_index
    edge_cache = {}
    new_points_list = list(points)
    new_faces = []

    def get_edge_point(v1, v2):
        # 엣지는 방향이 없으므로 정렬하여 키 생성
        key = tuple(sorted((v1, v2)))
        if key in edge_cache:
            return edge_cache[key]
        
        # 선형 보간 (중점)
        mid_point = (points[v1] + points[v2]) * 0.5
        idx = len(new_points_list)
        new_points_list.append(mid_point)
        edge_cache[key] = idx
        return idx

    # 각 사각형 면(Quad)에 대해 수행
    # faces 데이터는 [4, v0, v1, v2, v3, 4, v0, ...] 형식이므로 파싱 필요
    i = 0
    while i < len(faces):
        n_verts = faces[i] # 4여야 함
        v = faces[i+1 : i+1+n_verts]
        i += n_verts + 1
        
        v0, v1, v2, v3 = v

        # 1. 면의 중심점 (Face Point) 생성 - 선형 평균
        face_point = (points[v0] + points[v1] + points[v2] + points[v3]) * 0.25
        fp_idx = len(new_points_list)
        new_points_list.append(face_point)

        # 2. 엣지 포인트 생성 (Edge Points)
        e0 = get_edge_point(v0, v1)
        e1 = get_edge_point(v1, v2)
        e2 = get_edge_point(v2, v3)
        e3 = get_edge_point(v3, v0)

        # 3. 4개의 새 사각형 생성 (Topology Split)
        # Quad 1
        new_faces.extend([4, v0, e0, fp_idx, e3])
        # Quad 2
        new_faces.extend([4, e0, v1, e1, fp_idx])
        # Quad 3
        new_faces.extend([4, fp_idx, e1, v2, e2])
        # Quad 4
        new_faces.extend([4, e3, fp_idx, e2, v3])

    return np.array(new_points_list), np.array(new_faces)


def _cell_averaging(points, faces):
    """
    [Step 2: Cell Averaging]
    논문의 Eq(1): p_new[v] += centroid / val[v]
    각 정점을 '자신을 포함하는 셀(면)들의 무게중심의 평균'으로 이동시킵니다.
    """
    n_points = len(points)
    new_positions = np.zeros_like(points)
    counts = np.zeros(n_points, dtype=int) # val[v]

    i = 0
    while i < len(faces):
        n_verts = faces[i]
        v_indices = faces[i+1 : i+1+n_verts]
        i += n_verts + 1

        # 현재 셀(면)의 무게 중심 (Centroid) 계산
        cell_verts = points[v_indices]
        centroid = np.mean(cell_verts, axis=0)

        # 이 셀에 포함된 모든 정점 v에 대해 무게 중심 누적
        for v_idx in v_indices:
            new_positions[v_idx] += centroid
            counts[v_idx] += 1

    # 평균 계산 (Eq 1)
    # count가 0인 경우는 없겠지만 안전을 위해 처리
    mask = counts > 0
    new_positions[mask] /= counts[mask, None]
    
    # 업데이트되지 않은 점(고립된 점 등)은 원래 위치 유지
    new_positions[~mask] = points[~mask]

    return new_positions


def _format_mesh_for_render(mesh):
    """PyVista 메쉬를 WebGPU 렌더러가 사용할 수 있는 flat numpy 배열로 변환"""
    # 법선 계산
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True, inplace=False)
    
    pts = mesh.points.astype(np.float32)
    normals = mesh.point_normals.astype(np.float32)
    # [N, 3] -> [M, 3] 삼각형 리스트로 변환
    faces = mesh.faces.reshape(-1, 4)[:, 1:] 
    
    out = []
    for tri in faces:
        for idx in tri:
            out.extend([*pts[idx], *normals[idx]])
    return np.array(out, dtype=np.float32)


def subdivided_cube_vertices(level: int) -> np.ndarray:
    """
    MLCA (Multi-Linear Cell Averaging) 알고리즘 구현체
    """
    if level == 0:
        return cube_vertices()

    # 1. 초기 육면체 메쉬 생성 (Quad Mesh 상태여야 함)
    cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
    
    # PyVista 포맷에서 Numpy 데이터로 추출
    points = cube.points.copy()
    faces = cube.faces.copy() # [4, v0, v1, v2, v3, 4, ...]

    # 2. 지정된 레벨만큼 반복
    for _ in range(level):
        # Step 1: Multi-linear Subdivision (Split) 
        points, faces = _linear_split_quads(points, faces)
        
        # Step 2: Cell Averaging (Smooth) 
        points = _cell_averaging(points, faces)

    # 3. 렌더링을 위해 최종 결과를 PyVista 객체로 다시 포장하여 삼각형화
    # (WebGPU는 Quad를 직접 그리지 못하므로 삼각형 변환 필요)
    # faces 배열 포맷: [N, face_len, v0, v1, ... ] -> PyVista는 이 포맷을 그대로 사용 가능
    result_mesh = pv.PolyData(points, faces)
    
    return _format_mesh_for_render(result_mesh.triangulate())


def create_wireframe_indices(num_vertices: int) -> np.ndarray:
    """삼각형 리스트 데이터에서 와이어프레임(라인 리스트) 인덱스를 생성합니다."""
    num_triangles = num_vertices // 3
    indices = []
    for i in range(num_triangles):
        v0, v1, v2 = i * 3, i * 3 + 1, i * 3 + 2
        indices.extend([v0, v1, v1, v2, v2, v0])  
    return np.array(indices, dtype=np.uint32)