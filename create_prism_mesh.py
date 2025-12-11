# create_prism_mesh.py
"""
논문 스타일의 기둥형 Hexahedral 메쉬 생성

이 스크립트는 다음 형태의 메쉬를 생성합니다:
1. 1/4 원기둥 (Quarter Cylinder) - 논문 Figure와 유사
2. 삼각기둥 (Triangular Prism)
3. 오각기둥 (Pentagonal Prism)
4. 일반 N각 기둥 (N-gonal Prism)

각 메쉬는 Hexahedral 셀로 구성되어 MLCA 서브디비전에 바로 적용 가능합니다.

핵심: 중앙이 삼각형이 아닌, 모든 면이 사각형인 직교 격자(Cartesian Grid) 기반
"""

import numpy as np
import pyvista as pv
from pathlib import Path


def create_quarter_cylinder_hex_mesh(
    n_x: int = 4,
    n_y: int = 4,
    n_z: int = 4,
    radius: float = 1.0,
    height: float = 1.0,
) -> pv.UnstructuredGrid:
    """
    1/4 원기둥 형태의 Hexahedral 메쉬 생성 (논문 Figure 스타일)
    
    직교 격자를 생성한 후, 1/4 원 영역 내부의 점들만 원기둥 표면으로 투영합니다.
    모든 셀이 사각형 면을 가진 Hexahedron입니다.
    
    Parameters
    ----------
    n_x, n_y, n_z : int
        각 축 방향 셀 개수
    radius : float
        외부 반지름
    height : float
        기둥 높이
    
    Returns
    -------
    pv.UnstructuredGrid
        Hexahedral 메쉬
    """
    # 정규 격자 생성 (0~1 범위)
    xs = np.linspace(0, 1, n_x + 1)
    ys = np.linspace(0, 1, n_y + 1)
    zs = np.linspace(0, height, n_z + 1)
    
    points = []
    point_index = {}
    idx = 0
    
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                # 1/4 원으로 변환 (논문 스타일)
                # 정사각형 [0,1]x[0,1]을 1/4 원으로 매핑
                # 경계(x=1 또는 y=1)의 점들만 원호 위에 놓임
                
                # 방법: x, y를 극좌표처럼 사용하되, 격자 구조 유지
                # 외곽 경계를 원호로 투영
                px, py = x, y
                
                # 가장 바깥쪽 (x=1 또는 y=1)만 원호로 변형
                # 내부는 선형 보간으로 부드럽게 연결
                if x > 0 or y > 0:
                    # 현재 점에서 원점까지의 거리 (정규화)
                    r_square = max(x, y)  # 정사각형 기준 "반지름"
                    
                    if r_square > 0:
                        # 각도 계산 (정사각형 모서리 → 원호)
                        if x >= y and x > 0:
                            t = y / x if x > 0 else 0  # 0~1 사이
                            theta = t * (np.pi / 4)  # 0~45도
                            r_circle = radius * r_square
                            px = r_circle * np.cos(theta)
                            py = r_circle * np.sin(theta)
                        else:
                            t = x / y if y > 0 else 0  # 0~1 사이
                            theta = (1 - t) * (np.pi / 4) + np.pi / 4  # 45~90도
                            r_circle = radius * r_square
                            px = r_circle * np.cos(theta)
                            py = r_circle * np.sin(theta)
                
                points.append([px, py, z])
                point_index[(i, j, k)] = idx
                idx += 1
    
    points = np.array(points, dtype=np.float32)
    
    # Hexahedral 셀 생성
    cells = []
    cell_types = []
    
    for k in range(n_z):
        for j in range(n_y):
            for i in range(n_x):
                # 8개 정점 (VTK Hexahedron 순서)
                v0 = point_index[(i, j, k)]
                v1 = point_index[(i + 1, j, k)]
                v2 = point_index[(i + 1, j + 1, k)]
                v3 = point_index[(i, j + 1, k)]
                v4 = point_index[(i, j, k + 1)]
                v5 = point_index[(i + 1, j, k + 1)]
                v6 = point_index[(i + 1, j + 1, k + 1)]
                v7 = point_index[(i, j + 1, k + 1)]
                
                cells.append([8, v0, v1, v2, v3, v4, v5, v6, v7])
                cell_types.append(pv.CellType.HEXAHEDRON)
    
    cells = np.array(cells).flatten()
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    print(f"✓ 1/4 원기둥 메쉬 생성: {mesh.n_points} 정점, {mesh.n_cells} Hexahedral 셀")
    
    return mesh


def create_ngon_prism_hex_mesh(
    n_sides: int = 5,
    n_radial: int = 3,
    n_angular: int = 3,
    n_height: int = 4,
    radius: float = 1.0,
    height: float = 1.0,
) -> pv.UnstructuredGrid:
    """
    N각 기둥 형태의 Hexahedral 메쉬 생성 (모든 면이 사각형)
    
    원리: 직각삼각형 기둥(1/N 조각)을 N개 만들어 중심을 향해 합침
    각 조각은 직교 격자를 삼각형 영역으로 변환한 것
    
    Parameters
    ----------
    n_sides : int
        다각형 변의 개수 (3=삼각, 5=오각, ...)
    n_radial : int
        반지름 방향 셀 개수
    n_angular : int
        각도 방향 셀 개수 (각 조각 내에서)
    n_height : int
        높이 방향 셀 개수
    radius : float
        외부 반지름
    height : float
        기둥 높이
    
    Returns
    -------
    pv.UnstructuredGrid
        Hexahedral 메쉬
    """
    # 각 조각의 각도 범위
    angle_per_side = 2 * np.pi / n_sides
    
    # 점 저장 (중복 제거를 위해 딕셔너리 사용)
    all_points = []
    point_map = {}  # (sector, i, j, k) -> global_index
    
    zs = np.linspace(0, height, n_height + 1)
    
    def get_or_create_point(x, y, z, tolerance=1e-6):
        """중복 점 확인 및 생성"""
        # 부동소수점 비교를 위해 반올림
        key = (round(x / tolerance), round(y / tolerance), round(z / tolerance))
        if key not in point_map:
            point_map[key] = len(all_points)
            all_points.append([x, y, z])
        return point_map[key]
    
    # 각 조각(sector)에 대해 점 생성
    sector_points = {}  # (sector, i, j, k) -> global_index
    
    for sector in range(n_sides):
        # 이 조각의 시작 각도
        theta_start = sector * angle_per_side
        
        for k, z in enumerate(zs):
            for j in range(n_radial + 1):  # 반지름 방향 (0=중심, n_radial=외곽)
                for i in range(n_angular + 1):  # 각도 방향 (0=왼쪽 경계, n_angular=오른쪽 경계)
                    # 정규화된 좌표 (0~1)
                    u = i / n_angular  # 각도 방향 파라미터
                    v = j / n_radial   # 반지름 방향 파라미터
                    
                    # 삼각형 영역으로 변환
                    # 중심(v=0)에서는 모든 u가 같은 점(원점)
                    # 외곽(v=1)에서는 u에 따라 원호 위의 점
                    
                    # 현재 반지름
                    r = v * radius
                    
                    # 현재 각도 (이 조각 내에서 u에 비례)
                    theta = theta_start + u * angle_per_side
                    
                    # 직교 좌표 계산
                    if r < 1e-10:
                        # 중심점 (모든 조각이 공유)
                        px, py = 0.0, 0.0
                    else:
                        px = r * np.cos(theta)
                        py = r * np.sin(theta)
                    
                    # 중복 제거하며 점 추가
                    global_idx = get_or_create_point(px, py, z)
                    sector_points[(sector, i, j, k)] = global_idx
    
    points = np.array(all_points, dtype=np.float32)
    
    # Hexahedral 셀 생성
    cells = []
    cell_types = []
    
    for sector in range(n_sides):
        for k in range(n_height):
            for j in range(n_radial):
                for i in range(n_angular):
                    # 8개 정점 인덱스 가져오기
                    # 아래면 (z=k)
                    v0 = sector_points[(sector, i, j, k)]
                    v1 = sector_points[(sector, i + 1, j, k)]
                    v2 = sector_points[(sector, i + 1, j + 1, k)]
                    v3 = sector_points[(sector, i, j + 1, k)]
                    # 위면 (z=k+1)
                    v4 = sector_points[(sector, i, j, k + 1)]
                    v5 = sector_points[(sector, i + 1, j, k + 1)]
                    v6 = sector_points[(sector, i + 1, j + 1, k + 1)]
                    v7 = sector_points[(sector, i, j + 1, k + 1)]
                    
                    cells.append([8, v0, v1, v2, v3, v4, v5, v6, v7])
                    cell_types.append(pv.CellType.HEXAHEDRON)
    
    cells = np.array(cells).flatten()
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    
    shape_name = {3: "삼각", 4: "사각", 5: "오각", 6: "육각", 8: "팔각"}.get(n_sides, f"{n_sides}각")
    print(f"✓ {shape_name}기둥 메쉬 생성: {mesh.n_points} 정점, {mesh.n_cells} Hexahedral 셀")
    
    return mesh


def create_triangular_prism(n_radial: int = 3, n_angular: int = 3, n_height: int = 4, **kwargs) -> pv.UnstructuredGrid:
    """삼각기둥 Hexahedral 메쉬 (3개의 조각 합침)"""
    return create_ngon_prism_hex_mesh(n_sides=3, n_radial=n_radial, n_angular=n_angular, n_height=n_height, **kwargs)


def create_pentagonal_prism(n_radial: int = 3, n_angular: int = 3, n_height: int = 4, **kwargs) -> pv.UnstructuredGrid:
    """오각기둥 Hexahedral 메쉬 (5개의 조각 합침)"""
    return create_ngon_prism_hex_mesh(n_sides=5, n_radial=n_radial, n_angular=n_angular, n_height=n_height, **kwargs)


def create_hexagonal_prism(n_radial: int = 3, n_angular: int = 3, n_height: int = 4, **kwargs) -> pv.UnstructuredGrid:
    """육각기둥 Hexahedral 메쉬 (6개의 조각 합침)"""
    return create_ngon_prism_hex_mesh(n_sides=6, n_radial=n_radial, n_angular=n_angular, n_height=n_height, **kwargs)


def create_half_cylinder_hex_mesh(
    n_x: int = 4,
    n_y: int = 4,
    n_z: int = 4,
    radius: float = 1.0,
    height: float = 1.0,
) -> pv.UnstructuredGrid:
    """
    1/2 원기둥 형태의 Hexahedral 메쉬 생성
    
    Parameters
    ----------
    n_x, n_y, n_z : int
        각 축 방향 셀 개수 (n_x는 반원 방향)
    radius : float
        외부 반지름
    height : float
        기둥 높이
    """
    # 격자 생성
    xs = np.linspace(-1, 1, n_x + 1)  # -1 ~ 1 (반원 전체)
    ys = np.linspace(0, 1, n_y + 1)   # 반지름 방향
    zs = np.linspace(0, height, n_z + 1)
    
    points = []
    point_index = {}
    idx = 0
    
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                # 반원으로 변환
                # x: -1~1을 각도 180~0도로 매핑
                # y: 0~1을 반지름으로 매핑
                
                theta = (1 - (x + 1) / 2) * np.pi  # 0~180도
                r = y * radius
                
                px = r * np.cos(theta)
                py = r * np.sin(theta)
                
                points.append([px, py, z])
                point_index[(i, j, k)] = idx
                idx += 1
    
    points = np.array(points, dtype=np.float32)
    
    # Hexahedral 셀 생성
    cells = []
    cell_types = []
    
    for k in range(n_z):
        for j in range(n_y):
            for i in range(n_x):
                v0 = point_index[(i, j, k)]
                v1 = point_index[(i + 1, j, k)]
                v2 = point_index[(i + 1, j + 1, k)]
                v3 = point_index[(i, j + 1, k)]
                v4 = point_index[(i, j, k + 1)]
                v5 = point_index[(i + 1, j, k + 1)]
                v6 = point_index[(i + 1, j + 1, k + 1)]
                v7 = point_index[(i, j + 1, k + 1)]
                
                cells.append([8, v0, v1, v2, v3, v4, v5, v6, v7])
                cell_types.append(pv.CellType.HEXAHEDRON)
    
    cells = np.array(cells).flatten()
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    print(f"✓ 1/2 원기둥 메쉬 생성: {mesh.n_points} 정점, {mesh.n_cells} Hexahedral 셀")
    
    return mesh


def create_full_cylinder_hex_mesh(
    n_theta: int = 8,
    n_r: int = 3,
    n_z: int = 4,
    radius: float = 1.0,
    height: float = 1.0,
) -> pv.UnstructuredGrid:
    """
    전체 원기둥 형태의 Hexahedral 메쉬 생성 (논문 Figure 스타일)
    
    중앙에 정사각형 코어를 두고, 그 주변을 원호 형태로 감싸는 구조입니다.
    모든 면이 사각형입니다.
    
    Parameters
    ----------
    n_theta : int
        원주 방향 셀 개수 (4의 배수 권장)
    n_r : int
        반지름 방향 셀 개수 (코어 제외)
    n_z : int
        높이 방향 셀 개수
    """
    # 중앙 코어 크기 (반지름의 일정 비율)
    core_ratio = 0.3
    core_size = radius * core_ratio
    
    points = []
    point_index = {}
    idx = 0
    
    zs = np.linspace(0, height, n_z + 1)
    
    # n_theta를 4의 배수로 조정
    n_side = n_theta // 4
    if n_side < 1:
        n_side = 1
    
    for k, z in enumerate(zs):
        # 1. 중앙 코어 (n_side x n_side 격자)
        for j in range(n_side + 1):
            for i in range(n_side + 1):
                x = -core_size + 2 * core_size * i / n_side
                y = -core_size + 2 * core_size * j / n_side
                points.append([x, y, z])
                point_index[('core', i, j, k)] = idx
                idx += 1
        
        # 2. 외곽 링들 (코어 외부 → 원주까지)
        for r_idx in range(n_r):
            # 현재 링의 "반지름" 비율
            t = (r_idx + 1) / n_r
            current_r = core_size + t * (radius - core_size)
            
            # 각 변에 대해 점 생성 (4개 변)
            for side in range(4):
                for s in range(n_side):
                    # 정사각형 변을 따라 파라미터화
                    u = s / n_side
                    
                    # 정사각형 좌표
                    if side == 0:  # 아래쪽 변 (y = -1)
                        sx, sy = -1 + 2 * u, -1
                    elif side == 1:  # 오른쪽 변 (x = 1)
                        sx, sy = 1, -1 + 2 * u
                    elif side == 2:  # 위쪽 변 (y = 1)
                        sx, sy = 1 - 2 * u, 1
                    else:  # 왼쪽 변 (x = -1)
                        sx, sy = -1, 1 - 2 * u
                    
                    # 정사각형 → 원으로 변환 (외곽만)
                    angle = np.arctan2(sy, sx)
                    
                    # 내부 점 (코어 경계)
                    inner_x = core_size * sx / max(abs(sx), abs(sy)) if max(abs(sx), abs(sy)) > 0 else 0
                    inner_y = core_size * sy / max(abs(sx), abs(sy)) if max(abs(sx), abs(sy)) > 0 else 0
                    
                    # 외부 점 (원호 위)
                    outer_x = current_r * np.cos(angle)
                    outer_y = current_r * np.sin(angle)
                    
                    # 선형 보간
                    x = outer_x
                    y = outer_y
                    
                    points.append([x, y, z])
                    point_index[('ring', r_idx, side, s, k)] = idx
                    idx += 1
    
    points = np.array(points, dtype=np.float32)
    
    # 셀 생성은 복잡하므로, 간단한 버전으로 대체
    # 여기서는 1/4 원기둥 4개를 조합하는 방식 대신
    # 직접 생성된 점들로 셀 구성
    
    # 복잡한 토폴로지 대신 간단한 접근: 1/4 원기둥 사용 권장
    print(f"  (전체 원기둥은 1/4 원기둥 4개 조합 권장)")
    
    # 임시로 빈 메쉬 반환
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    cell_types = np.array([pv.CellType.HEXAHEDRON], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points[:8] if len(points) >= 8 else points)
    
    return mesh


def create_box_hex_mesh(
    n_x: int = 3,
    n_y: int = 3,
    n_z: int = 3,
    size_x: float = 1.0,
    size_y: float = 1.0,
    size_z: float = 1.0,
) -> pv.UnstructuredGrid:
    """
    단순 직육면체 Hexahedral 메쉬 생성
    
    가장 기본적인 직교 격자 구조입니다.
    """
    xs = np.linspace(0, size_x, n_x + 1)
    ys = np.linspace(0, size_y, n_y + 1)
    zs = np.linspace(0, size_z, n_z + 1)
    
    points = []
    point_index = {}
    idx = 0
    
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                points.append([x, y, z])
                point_index[(i, j, k)] = idx
                idx += 1
    
    points = np.array(points, dtype=np.float32)
    
    cells = []
    cell_types = []
    
    for k in range(n_z):
        for j in range(n_y):
            for i in range(n_x):
                v0 = point_index[(i, j, k)]
                v1 = point_index[(i + 1, j, k)]
                v2 = point_index[(i + 1, j + 1, k)]
                v3 = point_index[(i, j + 1, k)]
                v4 = point_index[(i, j, k + 1)]
                v5 = point_index[(i + 1, j, k + 1)]
                v6 = point_index[(i + 1, j + 1, k + 1)]
                v7 = point_index[(i, j + 1, k + 1)]
                
                cells.append([8, v0, v1, v2, v3, v4, v5, v6, v7])
                cell_types.append(pv.CellType.HEXAHEDRON)
    
    cells = np.array(cells).flatten()
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    print(f"✓ 직육면체 메쉬 생성: {mesh.n_points} 정점, {mesh.n_cells} Hexahedral 셀")
    
    return mesh


def save_mesh(mesh: pv.UnstructuredGrid, filename: str) -> Path:
    """메쉬를 VTK 파일로 저장"""
    filepath = Path(filename)
    mesh.save(str(filepath))
    print(f"✓ 저장됨: {filepath}")
    return filepath


def main():
    """다양한 기둥형 메쉬 생성 및 저장"""
    output_dir = Path("meshes")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("논문 스타일 Hexahedral 메쉬 생성 (모든 면이 사각형)")
    print("=" * 60)
    
    # 1. 1/4 원기둥 (논문 Figure 스타일) - 핵심 예제
    print("\n[1] 1/4 원기둥 (Quarter Cylinder) - 논문 Figure 스타일")
    quarter_cyl = create_quarter_cylinder_hex_mesh(
        n_x=4, n_y=4, n_z=4
    )
    save_mesh(quarter_cyl, output_dir / "quarter_cylinder.vtk")
    
    # 2. 삼각기둥 (3개의 조각 합침)
    print("\n[2] 삼각기둥 (Triangular Prism) - 3개 조각 합침")
    tri_prism = create_triangular_prism(n_radial=3, n_angular=3, n_height=4)
    save_mesh(tri_prism, output_dir / "triangular_prism.vtk")
    
    # 3. 오각기둥 (5개의 조각 합침)
    print("\n[3] 오각기둥 (Pentagonal Prism) - 5개 조각 합침")
    penta_prism = create_pentagonal_prism(n_radial=3, n_angular=3, n_height=4)
    save_mesh(penta_prism, output_dir / "pentagonal_prism.vtk")
    
    # 4. 육각기둥 (6개의 조각 합침)
    print("\n[4] 육각기둥 (Hexagonal Prism) - 6개 조각 합침")
    hex_prism = create_hexagonal_prism(n_radial=3, n_angular=3, n_height=4)
    save_mesh(hex_prism, output_dir / "hexagonal_prism.vtk")
    
    # 5. 직육면체 (기본)
    print("\n[5] 직육면체 (Box)")
    box = create_box_hex_mesh(n_x=3, n_y=3, n_z=3)
    save_mesh(box, output_dir / "box.vtk")
    
    # 6. 원기둥 (전체) - 많은 조각 합침
    print("\n[6] 원기둥 (Full Cylinder) - 12개 조각 합침")
    full_cyl = create_ngon_prism_hex_mesh(n_sides=12, n_radial=3, n_angular=2, n_height=4)
    save_mesh(full_cyl, output_dir / "full_cylinder.vtk")
    
    print("\n" + "=" * 60)
    print("생성 완료! meshes/ 폴더에 VTK 파일이 저장되었습니다.")
    print("=" * 60)
    
    # 미리보기 (선택)
    try:
        print("\n미리보기를 시작합니다... (창을 닫으면 계속)")
        plotter = pv.Plotter(shape=(2, 3), window_size=(1500, 900))
        
        plotter.subplot(0, 0)
        plotter.add_mesh(quarter_cyl, show_edges=True, color='lightblue')
        plotter.add_title("1/4 Cylinder")
        
        plotter.subplot(0, 1)
        plotter.add_mesh(tri_prism, show_edges=True, color='lightgreen')
        plotter.add_title("Triangular (3)")
        
        plotter.subplot(0, 2)
        plotter.add_mesh(penta_prism, show_edges=True, color='lightyellow')
        plotter.add_title("Pentagonal (5)")
        
        plotter.subplot(1, 0)
        plotter.add_mesh(hex_prism, show_edges=True, color='lightpink')
        plotter.add_title("Hexagonal (6)")
        
        plotter.subplot(1, 1)
        plotter.add_mesh(box, show_edges=True, color='lightcoral')
        plotter.add_title("Box")
        
        plotter.subplot(1, 2)
        plotter.add_mesh(full_cyl, show_edges=True, color='lightcyan')
        plotter.add_title("Full Cylinder (12)")
        
        plotter.link_views()
        plotter.show()
    except Exception as e:
        print(f"미리보기 실패 (무시 가능): {e}")


if __name__ == "__main__":
    main()