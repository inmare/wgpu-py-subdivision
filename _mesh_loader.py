# _mesh_loader.py
"""
VTK/PyVista 파일 로더 및 메쉬 정보 추출 모듈

지원 파일 형식:
- VTK Legacy (.vtk)
- VTK XML (.vtu, .vtp, .vts, .vtr, .vti)
- STL (.stl)
- OBJ (.obj)
- PLY (.ply)
- GMSH (.msh)
- 기타 meshio 지원 형식

사용법:
    from _mesh_loader import load_mesh, get_mesh_info, mesh_to_hexahedral
    
    # 파일에서 메쉬 로드
    mesh = load_mesh("model.vtk")
    
    # 메쉬 정보 출력
    info = get_mesh_info(mesh)
    print(info)
    
    # MLCA용 hexahedral 메쉬로 변환
    hex_mesh = mesh_to_hexahedral(mesh, resolution=10)
"""

import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MeshInfo:
    """메쉬 정보를 담는 데이터 클래스"""
    n_points: int
    n_cells: int
    cell_types: Dict[str, int]
    bounds: Tuple[float, float, float, float, float, float]
    center: Tuple[float, float, float]
    volume: Optional[float]
    surface_area: Optional[float]
    point_data_arrays: list
    cell_data_arrays: list
    mesh_type: str
    is_hexahedral: bool
    is_watertight: bool

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "메쉬 정보 (Mesh Information)",
            "=" * 50,
            f"메쉬 타입: {self.mesh_type}",
            f"정점 수 (Points): {self.n_points:,}",
            f"셀 수 (Cells): {self.n_cells:,}",
            "",
            "셀 타입 분포:",
        ]
        for cell_type, count in self.cell_types.items():
            lines.append(f"  - {cell_type}: {count:,}")
        
        lines.extend([
            "",
            f"경계 (Bounds): ",
            f"  X: [{self.bounds[0]:.4f}, {self.bounds[1]:.4f}]",
            f"  Y: [{self.bounds[2]:.4f}, {self.bounds[3]:.4f}]",
            f"  Z: [{self.bounds[4]:.4f}, {self.bounds[5]:.4f}]",
            f"중심 (Center): ({self.center[0]:.4f}, {self.center[1]:.4f}, {self.center[2]:.4f})",
        ])
        
        if self.volume is not None:
            lines.append(f"부피 (Volume): {self.volume:.6f}")
        if self.surface_area is not None:
            lines.append(f"표면적 (Surface Area): {self.surface_area:.6f}")
        
        lines.extend([
            "",
            f"Hexahedral 메쉬 여부: {'예' if self.is_hexahedral else '아니오'}",
            f"Watertight 여부: {'예' if self.is_watertight else '아니오'}",
        ])
        
        if self.point_data_arrays:
            lines.append(f"\nPoint Data Arrays: {', '.join(self.point_data_arrays)}")
        if self.cell_data_arrays:
            lines.append(f"Cell Data Arrays: {', '.join(self.cell_data_arrays)}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def load_mesh(filepath: Union[str, Path]) -> pv.DataSet:
    """
    다양한 형식의 3D 메쉬 파일을 로드합니다.
    
    Parameters
    ----------
    filepath : str or Path
        메쉬 파일 경로 (.vtk, .vtu, .stl, .obj, .ply 등)
    
    Returns
    -------
    pv.DataSet
        로드된 PyVista 메쉬 객체
    
    Examples
    --------
    >>> mesh = load_mesh("model.vtk")
    >>> mesh = load_mesh("bunny.stl")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    # PyVista의 범용 read 함수 사용
    mesh = pv.read(str(filepath))
    
    print(f"✓ 파일 로드 완료: {filepath.name}")
    print(f"  타입: {type(mesh).__name__}")
    print(f"  정점: {mesh.n_points:,}, 셀: {mesh.n_cells:,}")
    
    return mesh


def load_example_mesh(name: str) -> pv.DataSet:
    """
    PyVista 예제 데이터셋을 로드합니다.
    
    Parameters
    ----------
    name : str
        예제 이름. 지원되는 예제:
        - 'bunny': Stanford Bunny
        - 'cow': 소 모델
        - 'dragon': Stanford Dragon
        - 'armadillo': Armadillo 모델
        - 'teapot': Utah Teapot
        - 'sphere': 구
        - 'cube': 정육면체
        - 'cylinder': 원통
        - 'cone': 원뿔
        - 'torus': 토러스
        - 'hexbeam': Hexahedral beam (MLCA 테스트용)
        - 'notch_stress': FEA 노치 스트레스 (UnstructuredGrid)
    
    Returns
    -------
    pv.DataSet
        PyVista 메쉬 객체
    """
    from pyvista import examples
    
    example_map = {
        # 다운로드 가능한 모델
        'bunny': examples.download_bunny,
        'cow': examples.download_cow,
        'dragon': examples.download_dragon,
        'armadillo': examples.download_armadillo,
        'teapot': examples.download_teapot,
        'notch_stress': examples.download_notch_stress,
        'notch_displacement': examples.download_notch_displacement,
        'hexbeam': examples.download_unstructured_grid,
        
        # 기본 기하 도형 (생성)
        'sphere': lambda: pv.Sphere(radius=0.5, theta_resolution=32, phi_resolution=32),
        'cube': lambda: pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0),
        'cylinder': lambda: pv.Cylinder(radius=0.5, height=1.0, resolution=32),
        'cone': lambda: pv.Cone(radius=0.5, height=1.0, resolution=32),
        'torus': lambda: pv.ParametricTorus(),
    }
    
    name_lower = name.lower()
    if name_lower not in example_map:
        available = ', '.join(sorted(example_map.keys()))
        raise ValueError(f"알 수 없는 예제: '{name}'. 사용 가능한 예제: {available}")
    
    mesh = example_map[name_lower]()
    print(f"✓ 예제 로드 완료: {name}")
    print(f"  타입: {type(mesh).__name__}")
    print(f"  정점: {mesh.n_points:,}, 셀: {mesh.n_cells:,}")
    
    return mesh


def get_mesh_info(mesh: pv.DataSet) -> MeshInfo:
    """
    메쉬의 상세 정보를 추출합니다.
    
    Parameters
    ----------
    mesh : pv.DataSet
        PyVista 메쉬 객체
    
    Returns
    -------
    MeshInfo
        메쉬 정보 데이터 클래스
    """
    # 셀 타입 분석
    cell_types = {}
    if hasattr(mesh, 'celltypes'):
        unique_types, counts = np.unique(mesh.celltypes, return_counts=True)
        cell_type_names = {
            1: "Vertex",
            3: "Line",
            5: "Triangle",
            9: "Quad",
            10: "Tetrahedron",
            12: "Hexahedron",
            13: "Wedge",
            14: "Pyramid",
            42: "Polyhedron",
        }
        for ct, count in zip(unique_types, counts):
            name = cell_type_names.get(ct, f"Type_{ct}")
            cell_types[name] = int(count)
    elif hasattr(mesh, 'faces') and mesh.n_cells > 0:
        cell_types["Polygon"] = mesh.n_cells
    
    # Hexahedral 여부 확인
    is_hexahedral = 'Hexahedron' in cell_types and len(cell_types) == 1
    
    # Watertight 여부 (PolyData인 경우)
    is_watertight = False
    if isinstance(mesh, pv.PolyData):
        try:
            is_watertight = mesh.is_manifold
        except:
            pass
    
    # 부피 및 표면적 계산
    volume = None
    surface_area = None
    try:
        if isinstance(mesh, pv.PolyData):
            surface_area = mesh.area
            if is_watertight:
                volume = mesh.volume
        elif isinstance(mesh, pv.UnstructuredGrid):
            volume = mesh.volume
    except:
        pass
    
    return MeshInfo(
        n_points=mesh.n_points,
        n_cells=mesh.n_cells,
        cell_types=cell_types,
        bounds=mesh.bounds,
        center=tuple(mesh.center),
        volume=volume,
        surface_area=surface_area,
        point_data_arrays=list(mesh.point_data.keys()) if hasattr(mesh, 'point_data') else [],
        cell_data_arrays=list(mesh.cell_data.keys()) if hasattr(mesh, 'cell_data') else [],
        mesh_type=type(mesh).__name__,
        is_hexahedral=is_hexahedral,
        is_watertight=is_watertight,
    )


def surface_to_volume(
    surface_mesh: pv.PolyData,
    resolution: int = 10,
    method: str = 'voxelize'
) -> pv.UnstructuredGrid:
    """
    표면 메쉬(PolyData)를 볼륨 메쉬(UnstructuredGrid)로 변환합니다.
    MLCA 알고리즘에 사용할 수 있는 Hexahedral 메쉬를 생성합니다.
    
    Parameters
    ----------
    surface_mesh : pv.PolyData
        표면 메쉬
    resolution : int
        해상도 (높을수록 더 많은 셀 생성)
    method : str
        변환 방법:
        - 'voxelize': 복셀화 (가장 빠름, 계단 현상 있음)
        - 'tetrahedralize': 사면체화 (Hexahedral 아님, 참고용)
    
    Returns
    -------
    pv.UnstructuredGrid
        볼륨 메쉬
    """
    if not isinstance(surface_mesh, pv.PolyData):
        if hasattr(surface_mesh, 'extract_surface'):
            surface_mesh = surface_mesh.extract_surface()
        else:
            raise TypeError("표면 메쉬(PolyData)가 필요합니다.")
    
    if method == 'voxelize':
        # 경계 박스 기반으로 density 계산
        bounds = surface_mesh.bounds
        max_dim = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        )
        density = max_dim / resolution
        
        # 복셀화
        voxels = pv.voxelize(surface_mesh, density=density, check_surface=False)
        print(f"✓ 복셀화 완료: {voxels.n_cells:,} Hexahedral 셀 생성")
        return voxels
    
    elif method == 'tetrahedralize':
        # 사면체화 (TetGen 필요)
        try:
            tet = surface_mesh.delaunay_3d()
            print(f"✓ 사면체화 완료: {tet.n_cells:,} Tetrahedral 셀 생성")
            print("  ⚠ 주의: MLCA는 Hexahedral 메쉬에서만 동작합니다.")
            return tet
        except Exception as e:
            raise RuntimeError(f"사면체화 실패: {e}")
    
    else:
        raise ValueError(f"알 수 없는 변환 방법: {method}")


def create_hexahedral_grid(
    bounds: Tuple[float, float, float, float, float, float] = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
    dims: Tuple[int, int, int] = (2, 2, 2)
) -> pv.UnstructuredGrid:
    """
    균일한 Hexahedral 그리드를 생성합니다.
    
    Parameters
    ----------
    bounds : tuple
        (xmin, xmax, ymin, ymax, zmin, zmax)
    dims : tuple
        (nx, ny, nz) 각 축의 셀 개수
    
    Returns
    -------
    pv.UnstructuredGrid
        Hexahedral UnstructuredGrid
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = dims
    
    # 점 생성
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    z = np.linspace(zmin, zmax, nz + 1)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Hexahedral 셀 생성
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 8개의 정점 인덱스 (VTK Hexahedron 순서)
                idx = lambda ii, jj, kk: ii * (ny+1) * (nz+1) + jj * (nz+1) + kk
                
                v0 = idx(i, j, k)
                v1 = idx(i+1, j, k)
                v2 = idx(i+1, j+1, k)
                v3 = idx(i, j+1, k)
                v4 = idx(i, j, k+1)
                v5 = idx(i+1, j, k+1)
                v6 = idx(i+1, j+1, k+1)
                v7 = idx(i, j+1, k+1)
                
                cells.extend([8, v0, v1, v2, v3, v4, v5, v6, v7])
    
    cells = np.array(cells)
    cell_types = np.full(nx * ny * nz, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    print(f"✓ Hexahedral 그리드 생성: {grid.n_cells:,} 셀, {grid.n_points:,} 정점")
    
    return grid


def extract_hexahedral_data(mesh: pv.UnstructuredGrid) -> Tuple[np.ndarray, np.ndarray]:
    """
    UnstructuredGrid에서 Hexahedral 셀 데이터를 추출합니다.
    MLCA 알고리즘에 직접 사용할 수 있는 형식으로 반환합니다.
    
    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        Hexahedral 메쉬
    
    Returns
    -------
    points : np.ndarray
        정점 좌표 (N, 3)
    cells : np.ndarray
        셀 정의 배열 [8, v0, v1, ..., v7, 8, v0, ...]
    """
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("UnstructuredGrid가 필요합니다.")
    
    # Hexahedral 셀만 확인
    if hasattr(mesh, 'celltypes'):
        hex_mask = mesh.celltypes == pv.CellType.HEXAHEDRON
        if not np.all(hex_mask):
            n_hex = np.sum(hex_mask)
            n_other = len(hex_mask) - n_hex
            print(f"⚠ 경고: {n_other}개의 non-Hexahedral 셀이 무시됩니다.")
            mesh = mesh.extract_cells(hex_mask)
    
    points = mesh.points.copy().astype(np.float64)
    
    # PyVista의 cells 배열을 MLCA 형식으로 변환
    # PyVista: [n_cells, size0, v0, v1, ..., size1, v0, ...]
    # 우리 형식: [8, v0, v1, ..., v7, 8, v0, ...]
    cells_pv = mesh.cells
    
    # 새로운 형식으로 변환
    cells = []
    offset = 0
    for _ in range(mesh.n_cells):
        n_verts = cells_pv[offset]
        if n_verts != 8:
            raise ValueError(f"Hexahedral이 아닌 셀 발견 (정점 수: {n_verts})")
        cells.extend([8] + list(cells_pv[offset+1:offset+9]))
        offset += n_verts + 1
    
    return points, np.array(cells)


def normalize_mesh(mesh: pv.DataSet, target_size: float = 1.0) -> pv.DataSet:
    """
    메쉬를 원점 중심으로 이동하고 크기를 정규화합니다.
    
    Parameters
    ----------
    mesh : pv.DataSet
        입력 메쉬
    target_size : float
        목표 크기 (가장 긴 축의 길이)
    
    Returns
    -------
    pv.DataSet
        정규화된 메쉬
    """
    # 복사본 생성
    normalized = mesh.copy()
    
    # 중심을 원점으로 이동
    center = np.array(normalized.center)
    normalized.translate(-center, inplace=True)
    
    # 크기 정규화
    bounds = normalized.bounds
    max_dim = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    )
    
    if max_dim > 0:
        scale = target_size / max_dim
        normalized.scale([scale, scale, scale], inplace=True)
    
    print(f"✓ 메쉬 정규화 완료: 크기 {target_size}")
    
    return normalized


# 편의 함수: 파일 확장자 목록
SUPPORTED_EXTENSIONS = [
    '.vtk', '.vtu', '.vtp', '.vts', '.vtr', '.vti',  # VTK 형식
    '.stl', '.obj', '.ply', '.off',                   # 일반 3D 형식
    '.msh', '.inp',                                   # 메쉬 형식
    '.case', '.foam',                                 # CFD 형식
]


def list_supported_formats() -> None:
    """지원되는 파일 형식 목록을 출력합니다."""
    print("지원되는 파일 형식:")
    print("-" * 40)
    formats = {
        "VTK Legacy": [".vtk"],
        "VTK XML": [".vtu", ".vtp", ".vts", ".vtr", ".vti"],
        "Stereolithography": [".stl"],
        "Wavefront OBJ": [".obj"],
        "Stanford PLY": [".ply"],
        "Object File Format": [".off"],
        "Gmsh": [".msh"],
        "Abaqus": [".inp"],
        "EnSight": [".case"],
        "OpenFOAM": [".foam"],
    }
    for name, exts in formats.items():
        print(f"  {name}: {', '.join(exts)}")
    print("-" * 40)
    print("meshio가 설치된 경우 더 많은 형식 지원")


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("VTK/PyVista 메쉬 로더 테스트")
    print("=" * 60)
    
    # 지원 형식 출력
    list_supported_formats()
    print()
    
    # 예제 메쉬 로드 테스트
    print("\n--- 예제 메쉬 로드 테스트 ---")
    mesh = load_example_mesh('cube')
    info = get_mesh_info(mesh)
    print(info)
    
    # Hexahedral 그리드 생성 테스트
    print("\n--- Hexahedral 그리드 생성 테스트 ---")
    hex_grid = create_hexahedral_grid(dims=(3, 3, 3))
    info = get_mesh_info(hex_grid)
    print(info)
    
    # 데이터 추출 테스트
    print("\n--- Hexahedral 데이터 추출 테스트 ---")
    points, cells = extract_hexahedral_data(hex_grid)
    print(f"추출된 점: {len(points)}, 셀 배열 길이: {len(cells)}")
