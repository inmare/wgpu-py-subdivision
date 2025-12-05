import numpy as np
import pyvista as pv

def _linear_split_hexs(points, cells):
    """
    [Step 1: Multi-linear Subdivision for Volume]
    논문의 Section 2.1: d-cube를 2^d (3D의 경우 8개)의 subcubes로 분할합니다.
    """
    edge_cache = {}
    face_cache = {}
    
    new_points_list = list(points)
    new_cells = []

    # Helper: 엣지 중점
    def get_edge_point(v1, v2):
        key = tuple(sorted((v1, v2)))
        if key in edge_cache: return edge_cache[key]
        pt = (points[v1] + points[v2]) * 0.5
        idx = len(new_points_list)
        new_points_list.append(pt)
        edge_cache[key] = idx
        return idx

    # Helper: 면 중점
    def get_face_point(v1, v2, v3, v4):
        key = tuple(sorted((v1, v2, v3, v4)))
        if key in face_cache: return face_cache[key]
        pt = (points[v1] + points[v2] + points[v3] + points[v4]) * 0.25
        idx = len(new_points_list)
        new_points_list.append(pt)
        face_cache[key] = idx
        return idx

    # Helper: 셀 중점
    def get_cell_point(verts):
        pt = np.mean(points[verts], axis=0)
        idx = len(new_points_list)
        new_points_list.append(pt)
        return idx

    i = 0
    while i < len(cells):
        n_verts = cells[i] # Hex는 8
        if n_verts != 8:
            raise ValueError("Only Hexahedral meshes are supported")
        
        ids = cells[i+1 : i+1+8]
        i += 9
        v = list(ids)

        # 1. Points creation
        cp = get_cell_point(ids) # Cell Center
        
        # Face Centers
        f_bot   = get_face_point(v[0], v[1], v[2], v[3])
        f_top   = get_face_point(v[4], v[5], v[6], v[7])
        f_front = get_face_point(v[0], v[1], v[5], v[4])
        f_right = get_face_point(v[1], v[2], v[6], v[5])
        f_back  = get_face_point(v[2], v[3], v[7], v[6])
        f_left  = get_face_point(v[3], v[0], v[4], v[7])

        # Edge Midpoints
        e01 = get_edge_point(v[0], v[1])
        e12 = get_edge_point(v[1], v[2])
        e23 = get_edge_point(v[2], v[3])
        e30 = get_edge_point(v[3], v[0])
        e45 = get_edge_point(v[4], v[5])
        e56 = get_edge_point(v[5], v[6])
        e67 = get_edge_point(v[6], v[7])
        e74 = get_edge_point(v[7], v[4])
        e04 = get_edge_point(v[0], v[4])
        e15 = get_edge_point(v[1], v[5])
        e26 = get_edge_point(v[2], v[6])
        e37 = get_edge_point(v[3], v[7])

        # 2. Construct 8 Sub-Hexahedrons
        # Sub-Hex 0
        new_cells.extend([8, v[0], e01, f_bot, e30, e04, f_front, cp, f_left])
        # Sub-Hex 1
        new_cells.extend([8, e01, v[1], e12, f_bot, f_front, e15, f_right, cp])
        # Sub-Hex 2
        new_cells.extend([8, f_bot, e12, v[2], e23, cp, f_right, e26, f_back])
        # Sub-Hex 3
        new_cells.extend([8, e30, f_bot, e23, v[3], f_left, cp, f_back, e37])
        # Sub-Hex 4
        new_cells.extend([8, e04, f_front, cp, f_left, v[4], e45, f_top, e74])
        # Sub-Hex 5
        new_cells.extend([8, f_front, e15, f_right, cp, e45, v[5], e56, f_top])
        # Sub-Hex 6
        new_cells.extend([8, cp, f_right, e26, f_back, f_top, e56, v[6], e67])
        # Sub-Hex 7
        new_cells.extend([8, f_left, cp, f_back, e37, e74, f_top, e67, v[7]])

    return np.array(new_points_list), np.array(new_cells)


def _cell_averaging_vol(points, cells):
    """
    [Step 2: Cell Averaging for Volume]
    논문의 Eq(1) 적용 (Volume 버전)
    """
    n_points = len(points)
    new_positions = np.zeros_like(points)
    counts = np.zeros(n_points, dtype=int)

    i = 0
    while i < len(cells):
        n_verts = cells[i]
        v_indices = cells[i+1 : i+1+n_verts]
        i += n_verts + 1

        cell_verts = points[v_indices]
        centroid = np.mean(cell_verts, axis=0)

        for v_idx in v_indices:
            new_positions[v_idx] += centroid
            counts[v_idx] += 1

    mask = counts > 0
    new_positions[mask] /= counts[mask, None]
    new_positions[~mask] = points[~mask]

    return new_positions


def _format_volume_mesh_for_render(grid, mode='surface'):
    """
    mode='volume': 내부 구조 확인용 (Shrink 적용, 틈이 보임)
    mode='surface': 외형 확인용 (Shrink 없음, 매끈한 표면)
    """
    if mode == 'volume':
        # 내부를 보기 위해 셀을 수축시킴 (기존 방식)
        processed_grid = grid.shrink(shrink_factor=0.8)
        surface = processed_grid.extract_surface()
    else:
        # 표면만 추출 (Catmull-Clark 같은 부드러운 외형 확인용)
        surface = grid.extract_surface()
    
    # 법선 계산 및 삼각형 변환
    surface = surface.compute_normals(auto_orient_normals=True, consistent_normals=True, inplace=False)
    
    if surface.n_cells == 0:
        return np.array([], dtype=np.float32)
        
    tri_mesh = surface.triangulate()
    
    pts = tri_mesh.points.astype(np.float32)
    normals = tri_mesh.point_normals.astype(np.float32)
    faces = tri_mesh.faces.reshape(-1, 4)[:, 1:]
    
    out = []
    for tri in faces:
        for idx in tri:
            out.extend([*pts[idx], *normals[idx]])
            
    return np.array(out, dtype=np.float32)


def subdivided_volume_grid(level: int) -> np.ndarray:
    """메인 로직: Hexahedron 생성 -> Split -> Smooth -> Render Format"""
    # 초기 단일 Hexahedron
    pts = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
    ], dtype=float)
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7]) 
    
    current_pts = pts
    current_cells = cells

    for _ in range(level):
        current_pts, current_cells = _linear_split_hexs(current_pts, current_cells)
        current_pts = _cell_averaging_vol(current_pts, current_cells)

    cell_type = np.full(len(current_cells) // 9, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(current_cells, cell_type, current_pts)
    
    # [변경점] 여기서 모드를 선택하세요.
    # 'volume': 질문하신 이미지처럼 보임 (내부 쪼개짐 확인)
    # 'surface': 매끈하게 연결된 덩어리로 보임 (둥근 정도 확인 가능)
    return _format_volume_mesh_for_render(grid, mode='volume')

# --- 아래 부분이 기존 _mesh.py 와 호환성을 위해 추가된 부분입니다 ---

# 1. 기존 cube_vertices() 대체 (Level 0 호출)
def cube_vertices() -> np.ndarray:
    return subdivided_volume_grid(0)

# 2. 기존 subdivided_cube_vertices() 대체 (이름 Alias)
subdivided_cube_vertices = subdivided_volume_grid

# 3. 와이어프레임 인덱스 생성 함수 (이전 파일에서 복사됨)
def create_wireframe_indices(num_vertices: int) -> np.ndarray:
    """삼각형 리스트 데이터에서 와이어프레임(라인 리스트) 인덱스를 생성합니다."""
    # num_vertices는 float 데이터 개수가 아니라 점의 개수여야 함 (이미 호출부에서 size//6으로 계산해서 줌)
    num_triangles = num_vertices // 3
    indices = []
    for i in range(num_triangles):
        v0, v1, v2 = i * 3, i * 3 + 1, i * 3 + 2
        # 삼각형의 3개 변을 라인으로 정의
        indices.extend([v0, v1, v1, v2, v2, v0])  
    return np.array(indices, dtype=np.uint32)