# _mesh.py
import numpy as np
import pyvista as pv


def subdivided_cube_vertices(level: int = 1) -> np.ndarray:
    """
    MLCA 세분화의 1단계(Bi-linear subdivision + Cell Averaging)를 PyVista 기능으로 근사하여 
    세분화된 큐브의 정점 및 법선 데이터를 생성합니다.
    level 매개변수는 무시하고 1회 세분화만 수행합니다.
    """
    
    # 1. 초기 큐브 생성 및 삼각형화
    mesh = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0).triangulate()
    
    # 2. 다중 선형 세분화 근사: Bi-linear Subdivision (level=1)
    # Catmull-Clark의 'split' 단계에 해당
    # PyVista는 육면체 메쉬의 직접적인 Catmull-Clark 볼륨 세분화 필터를 제공하지 않으므로,
    # 선형 세분화(Bi-linear)를 사용하여 토폴로지 분할을 수행합니다.
    mesh_split = mesh.subdivide(1, subfilter="linear") 
    
    # 3. 셀 평균화 근사: Smoothing 
    # MLCA의 'average' 단계에 해당. MeshQuality 기반의 스무딩을 사용합니다.
    # iterations를 1로 설정하여 한 번만 평균화 효과를 줍니다.
    mesh_smoothed = mesh_split.smooth(n_iter=1, relaxation_factor=0.1)

    # 4. 법선 계산
    mesh_smoothed = mesh_smoothed.compute_normals(
        auto_orient_normals=True, consistent_normals=True, inplace=False
    )

    pts = mesh_smoothed.points.astype(np.float32)
    normals = mesh_smoothed.point_normals.astype(np.float32)
    # faces 배열은 [N, v0, v1, v2, ...] 형태로 저장되어 있어, 3개 정점 인덱스만 추출합니다.
    faces = mesh_smoothed.faces.reshape(-1, 4)[:, 1:] 

    out = []
    for tri in faces:
        for idx in tri:
            # [position (3) + normal (3)] 순서로 데이터 구성
            out.extend([*pts[idx], *normals[idx]]) 
    return np.array(out, dtype=np.float32)

def cube_vertices() -> np.ndarray:
    """기본 큐브의 정점 및 법선 데이터를 생성합니다 (36 vertices)."""
    # 6 faces * 2 triangles * 3 verts = 36 vertices, position + normal
    p = [
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, -1, 1), (1, 1, 1), (-1, 1, 1),  # front
        (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (-1, -1, -1), (1, 1, -1), (1, -1, -1),  # back
        (-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, -1, -1), (-1, 1, 1), (-1, 1, -1),  # left
        (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, -1), (1, 1, 1), (1, -1, 1),  # right
        (-1, 1, -1), (-1, 1, 1), (1, 1, 1), (-1, 1, -1), (1, 1, 1), (1, 1, -1),  # top
        (-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, -1), (1, -1, 1), (-1, -1, 1),  # bottom
    ]
    n = [
        (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
        (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1),
        (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0),
        (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),
        (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0),
        (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0),
    ]
    data = []
    for pos, norm in zip(p, n):
        data.extend([pos[0] * 0.5, pos[1] * 0.5, pos[2] * 0.5, norm[0], norm[1], norm[2]])
    return np.array(data, dtype=np.float32)


def subdivided_cube_vertices(level: int = 2) -> np.ndarray:
    """
    PyVista를 사용하여 큐브를 세분화하고, 결과 메쉬의 정점 및 법선 데이터를 생성합니다.
    (loop subdivision을 Catmull-Clark의 근사치로 사용)
    """
    mesh = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0).triangulate()
    # level=2로 루프 세분화 적용
    mesh = mesh.subdivide(level, subfilter="loop") 
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True, inplace=False)

    pts = mesh.points.astype(np.float32)
    normals = mesh.point_normals.astype(np.float32)
    # faces 배열은 [N, v0, v1, v2, ...] 형태로 저장되어 있어, 3개 정점 인덱스만 추출합니다.
    faces = mesh.faces.reshape(-1, 4)[:, 1:] 

    out = []
    for tri in faces:
        for idx in tri:
            # [position (3) + normal (3)] 순서로 데이터 구성
            out.extend([*pts[idx], *normals[idx]]) 
    return np.array(out, dtype=np.float32)


def create_wireframe_indices(num_vertices: int) -> np.ndarray:
    """삼각형 리스트 데이터에서 와이어프레임(라인 리스트) 인덱스를 생성합니다."""
    num_triangles = num_vertices // 3
    indices = []
    for i in range(num_triangles):
        v0, v1, v2 = i * 3, i * 3 + 1, i * 3 + 2
        # 삼각형의 3개 엣지: (v0-v1), (v1-v2), (v2-v0)
        indices.extend([v0, v1, v1, v2, v2, v0])  
    return np.array(indices, dtype=np.uint32)