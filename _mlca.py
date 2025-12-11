# _mlca.py
"""
MLCA (Multi-Linear Cell Averaging) 서브디비전 알고리즘 구현

논문 기반:
"Multi-Linear Cell Averaging for Subdivision of Hexahedral Meshes"
(Texas A&M University)

이 모듈은 일반적인 Hexahedral 메쉬에 MLCA 알고리즘을 적용합니다.
기존의 단순 큐브뿐만 아니라, 외부에서 로드한 VTK 메쉬에도 적용 가능합니다.

사용법:
    from _mlca import MLCASubdivision, subdivide_hexahedral_mesh
    
    # 방법 1: 클래스 사용
    mlca = MLCASubdivision()
    result = mlca.subdivide(mesh, level=2)
    
    # 방법 2: 함수 사용
    result = subdivide_hexahedral_mesh(mesh, level=2)
"""

import numpy as np
import pyvista as pv
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SubdivisionResult:
    """서브디비전 결과를 담는 데이터 클래스"""
    mesh: pv.UnstructuredGrid
    points: np.ndarray
    cells: np.ndarray
    level: int
    original_n_cells: int
    final_n_cells: int
    
    def __str__(self) -> str:
        return (
            f"MLCA Subdivision Result\n"
            f"  Level: {self.level}\n"
            f"  Original cells: {self.original_n_cells:,}\n"
            f"  Final cells: {self.final_n_cells:,}\n"
            f"  Points: {len(self.points):,}"
        )


class MLCASubdivision:
    """
    MLCA (Multi-Linear Cell Averaging) 서브디비전 알고리즘
    
    논문의 알고리즘을 구현합니다:
    1. Multi-linear Subdivision (Split): 각 Hex를 8개의 sub-Hex로 분할
    2. Cell Averaging (Smooth): 각 정점을 인접 셀의 무게중심 평균으로 이동
    
    Attributes
    ----------
    verbose : bool
        진행 상황 출력 여부
    
    Examples
    --------
    >>> mlca = MLCASubdivision()
    >>> result = mlca.subdivide(hex_mesh, level=2)
    >>> result.mesh.plot()
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, message: str):
        """로그 출력"""
        if self.verbose:
            print(message)
    
    def subdivide(
        self, 
        mesh: pv.UnstructuredGrid, 
        level: int = 1,
        smooth_iterations: int = 1
    ) -> SubdivisionResult:
        """
        Hexahedral 메쉬에 MLCA 서브디비전을 적용합니다.
        
        Parameters
        ----------
        mesh : pv.UnstructuredGrid
            입력 Hexahedral 메쉬
        level : int
            서브디비전 레벨 (0 = 원본)
        smooth_iterations : int
            각 레벨에서 Cell Averaging 반복 횟수
        
        Returns
        -------
        SubdivisionResult
            서브디비전 결과
        """
        if level < 0:
            raise ValueError("level은 0 이상이어야 합니다.")
        
        # 원본 데이터 추출
        points, cells = self._extract_hex_data(mesh)
        original_n_cells = self._count_cells(cells)
        
        self._log(f"MLCA Subdivision 시작")
        self._log(f"  입력: {len(points)} 정점, {original_n_cells} 셀")
        self._log(f"  목표 레벨: {level}")
        
        if level == 0:
            result_mesh = mesh.copy()
        else:
            # MLCA 알고리즘 적용
            for lv in range(level):
                self._log(f"\n  Level {lv + 1}/{level} 처리 중...")
                
                # Step 1: Multi-linear Subdivision (Split)
                points, cells = self._linear_split_hexs(points, cells)
                self._log(f"    Split 완료: {len(points)} 정점, {self._count_cells(cells)} 셀")
                
                # Step 2: Cell Averaging (Smooth)
                for i in range(smooth_iterations):
                    points = self._cell_averaging(points, cells)
                    if smooth_iterations > 1:
                        self._log(f"    Smooth {i+1}/{smooth_iterations} 완료")
                    else:
                        self._log(f"    Smooth 완료")
            
            # PyVista UnstructuredGrid로 변환
            result_mesh = self._create_unstructured_grid(points, cells)
        
        final_n_cells = self._count_cells(cells)
        self._log(f"\nMLCA Subdivision 완료!")
        self._log(f"  결과: {len(points)} 정점, {final_n_cells} 셀")
        
        return SubdivisionResult(
            mesh=result_mesh,
            points=points,
            cells=cells,
            level=level,
            original_n_cells=original_n_cells,
            final_n_cells=final_n_cells,
        )
    
    def _extract_hex_data(self, mesh: pv.UnstructuredGrid) -> Tuple[np.ndarray, np.ndarray]:
        """PyVista UnstructuredGrid에서 Hexahedral 데이터 추출"""
        if not isinstance(mesh, pv.UnstructuredGrid):
            raise TypeError("UnstructuredGrid가 필요합니다.")
        
        # Hexahedral 셀만 필터링
        if hasattr(mesh, 'celltypes'):
            hex_mask = mesh.celltypes == pv.CellType.HEXAHEDRON
            if not np.all(hex_mask):
                n_hex = np.sum(hex_mask)
                n_other = len(hex_mask) - n_hex
                self._log(f"⚠ {n_other}개의 non-Hexahedral 셀이 무시됩니다.")
                mesh = mesh.extract_cells(hex_mask)
        
        points = mesh.points.copy().astype(np.float64)
        
        # cells 배열 변환
        cells_pv = mesh.cells
        cells = []
        offset = 0
        for _ in range(mesh.n_cells):
            n_verts = cells_pv[offset]
            if n_verts != 8:
                raise ValueError(f"Hexahedral이 아닌 셀 발견 (정점 수: {n_verts})")
            cells.extend([8] + list(cells_pv[offset+1:offset+9]))
            offset += n_verts + 1
        
        return points, np.array(cells)
    
    def _count_cells(self, cells: np.ndarray) -> int:
        """셀 배열에서 셀 개수 계산"""
        return len(cells) // 9
    
    def _linear_split_hexs(
        self, 
        points: np.ndarray, 
        cells: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        [Step 1: Multi-linear Subdivision]
        논문의 Section 2.1: 각 Hexahedron을 8개의 sub-Hexahedra로 분할
        
        새로운 정점 생성:
        - 각 엣지의 중점 (Edge Points)
        - 각 면의 중심 (Face Points)  
        - 셀의 중심 (Cell Point)
        """
        edge_cache = {}
        face_cache = {}
        
        new_points_list = list(points)
        new_cells = []
        
        def get_edge_point(v1, v2):
            """엣지 중점 생성 (캐싱)"""
            key = tuple(sorted((v1, v2)))
            if key in edge_cache:
                return edge_cache[key]
            pt = (points[v1] + points[v2]) * 0.5
            idx = len(new_points_list)
            new_points_list.append(pt)
            edge_cache[key] = idx
            return idx
        
        def get_face_point(v1, v2, v3, v4):
            """면 중심점 생성 (캐싱)"""
            key = tuple(sorted((v1, v2, v3, v4)))
            if key in face_cache:
                return face_cache[key]
            pt = (points[v1] + points[v2] + points[v3] + points[v4]) * 0.25
            idx = len(new_points_list)
            new_points_list.append(pt)
            face_cache[key] = idx
            return idx
        
        def get_cell_point(verts):
            """셀 중심점 생성"""
            pt = np.mean(points[list(verts)], axis=0)
            idx = len(new_points_list)
            new_points_list.append(pt)
            return idx
        
        # 각 Hexahedron 처리
        i = 0
        while i < len(cells):
            n_verts = cells[i]
            if n_verts != 8:
                raise ValueError(f"Invalid cell: expected 8 vertices, got {n_verts}")
            
            v = list(cells[i+1:i+9])
            i += 9
            
            # 중심점 생성
            cp = get_cell_point(v)
            
            # 면 중심점 (6개 면)
            # VTK Hexahedron 정점 순서:
            #     7-------6
            #    /|      /|
            #   4-------5 |
            #   | 3-----|-2
            #   |/      |/
            #   0-------1
            f_bot   = get_face_point(v[0], v[1], v[2], v[3])  # 아래면
            f_top   = get_face_point(v[4], v[5], v[6], v[7])  # 위면
            f_front = get_face_point(v[0], v[1], v[5], v[4])  # 앞면
            f_right = get_face_point(v[1], v[2], v[6], v[5])  # 오른쪽
            f_back  = get_face_point(v[2], v[3], v[7], v[6])  # 뒷면
            f_left  = get_face_point(v[3], v[0], v[4], v[7])  # 왼쪽
            
            # 엣지 중점 (12개 엣지)
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
            
            # 8개의 Sub-Hexahedra 생성
            # Sub-Hex 0: 원래 v[0] 코너
            new_cells.extend([8, v[0], e01, f_bot, e30, e04, f_front, cp, f_left])
            # Sub-Hex 1: 원래 v[1] 코너
            new_cells.extend([8, e01, v[1], e12, f_bot, f_front, e15, f_right, cp])
            # Sub-Hex 2: 원래 v[2] 코너
            new_cells.extend([8, f_bot, e12, v[2], e23, cp, f_right, e26, f_back])
            # Sub-Hex 3: 원래 v[3] 코너
            new_cells.extend([8, e30, f_bot, e23, v[3], f_left, cp, f_back, e37])
            # Sub-Hex 4: 원래 v[4] 코너
            new_cells.extend([8, e04, f_front, cp, f_left, v[4], e45, f_top, e74])
            # Sub-Hex 5: 원래 v[5] 코너
            new_cells.extend([8, f_front, e15, f_right, cp, e45, v[5], e56, f_top])
            # Sub-Hex 6: 원래 v[6] 코너
            new_cells.extend([8, cp, f_right, e26, f_back, f_top, e56, v[6], e67])
            # Sub-Hex 7: 원래 v[7] 코너
            new_cells.extend([8, f_left, cp, f_back, e37, e74, f_top, e67, v[7]])
        
        return np.array(new_points_list), np.array(new_cells)
    
    def _cell_averaging(
        self, 
        points: np.ndarray, 
        cells: np.ndarray
    ) -> np.ndarray:
        """
        [Step 2: Cell Averaging]
        논문의 Eq(1): 각 정점을 인접 셀들의 무게중심 평균으로 이동
        
        p_new[v] = (1/valence[v]) * Σ centroid(cell)
        """
        n_points = len(points)
        new_positions = np.zeros_like(points)
        counts = np.zeros(n_points, dtype=int)
        
        i = 0
        while i < len(cells):
            n_verts = cells[i]
            v_indices = cells[i+1:i+1+n_verts]
            i += n_verts + 1
            
            # 셀의 무게중심 계산
            cell_verts = points[v_indices]
            centroid = np.mean(cell_verts, axis=0)
            
            # 이 셀에 연결된 모든 정점에 무게중심 누적
            for v_idx in v_indices:
                new_positions[v_idx] += centroid
                counts[v_idx] += 1
        
        # 평균 계산
        mask = counts > 0
        new_positions[mask] /= counts[mask, None]
        
        # 업데이트되지 않은 정점은 원래 위치 유지
        new_positions[~mask] = points[~mask]
        
        return new_positions
    
    def _create_unstructured_grid(
        self, 
        points: np.ndarray, 
        cells: np.ndarray
    ) -> pv.UnstructuredGrid:
        """points와 cells로 PyVista UnstructuredGrid 생성"""
        n_cells = len(cells) // 9
        cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        return pv.UnstructuredGrid(cells, cell_types, points)


def subdivide_hexahedral_mesh(
    mesh: pv.UnstructuredGrid,
    level: int = 1,
    smooth_iterations: int = 1,
    verbose: bool = True
) -> pv.UnstructuredGrid:
    """
    Hexahedral 메쉬에 MLCA 서브디비전을 적용하는 편의 함수
    
    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        입력 Hexahedral 메쉬
    level : int
        서브디비전 레벨
    smooth_iterations : int
        각 레벨에서 스무딩 반복 횟수
    verbose : bool
        진행 상황 출력 여부
    
    Returns
    -------
    pv.UnstructuredGrid
        서브디비전된 메쉬
    
    Examples
    --------
    >>> from _mesh_loader import create_hexahedral_grid
    >>> mesh = create_hexahedral_grid(dims=(2, 2, 2))
    >>> subdivided = subdivide_hexahedral_mesh(mesh, level=2)
    >>> subdivided.plot()
    """
    mlca = MLCASubdivision(verbose=verbose)
    result = mlca.subdivide(mesh, level=level, smooth_iterations=smooth_iterations)
    return result.mesh


def subdivide_from_surface(
    surface: pv.PolyData,
    level: int = 1,
    resolution: int = 5,
    verbose: bool = True
) -> pv.UnstructuredGrid:
    """
    표면 메쉬(PolyData)를 Hexahedral로 변환 후 MLCA 적용
    
    Parameters
    ----------
    surface : pv.PolyData
        입력 표면 메쉬
    level : int
        MLCA 서브디비전 레벨
    resolution : int
        복셀화 해상도
    verbose : bool
        진행 상황 출력 여부
    
    Returns
    -------
    pv.UnstructuredGrid
        서브디비전된 볼륨 메쉬
    """
    from _mesh_loader import surface_to_volume
    
    if verbose:
        print("Step 1: 표면 메쉬를 Hexahedral 볼륨으로 변환")
    
    hex_mesh = surface_to_volume(surface, resolution=resolution, method='voxelize')
    
    if verbose:
        print(f"\nStep 2: MLCA 서브디비전 적용 (Level {level})")
    
    return subdivide_hexahedral_mesh(hex_mesh, level=level, verbose=verbose)


# 렌더링용 변환 함수
def format_for_render(
    mesh: pv.UnstructuredGrid,
    mode: str = 'surface'
) -> np.ndarray:
    """
    UnstructuredGrid를 WebGPU 렌더링용 flat numpy 배열로 변환
    
    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        입력 메쉬
    mode : str
        'surface': 외부 표면만 (부드러운 외형)
        'volume': 셀을 수축하여 내부 구조 표시
    
    Returns
    -------
    np.ndarray
        [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, ...] 형식의 배열
    """
    if mode == 'volume':
        # 내부 구조 확인용 (셀 수축)
        processed = mesh.shrink(shrink_factor=0.8)
        surface = processed.extract_surface()
    else:
        # 표면만 추출
        surface = mesh.extract_surface()
    
    # 법선 계산 및 삼각형화
    surface = surface.compute_normals(
        auto_orient_normals=True, 
        consistent_normals=True, 
        inplace=False
    )
    
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


if __name__ == "__main__":
    print("=" * 60)
    print("MLCA (Multi-Linear Cell Averaging) 알고리즘 테스트")
    print("=" * 60)
    
    # 테스트 1: 단순 Hexahedral 그리드
    print("\n--- 테스트 1: 단순 Hexahedral 그리드 ---")
    from _mesh_loader import create_hexahedral_grid
    
    grid = create_hexahedral_grid(dims=(2, 2, 2))
    print(f"원본: {grid.n_cells} 셀")
    
    mlca = MLCASubdivision()
    result = mlca.subdivide(grid, level=2)
    print(result)
    
    # 테스트 2: 예제 모델 사용
    print("\n--- 테스트 2: Bunny 모델 (표면 → 볼륨 → MLCA) ---")
    try:
        from _mesh_loader import load_example_mesh, surface_to_volume, normalize_mesh
        
        bunny = load_example_mesh('bunny')
        bunny = normalize_mesh(bunny)
        
        # 복셀화
        hex_bunny = surface_to_volume(bunny, resolution=8)
        
        # MLCA 적용
        result = mlca.subdivide(hex_bunny, level=1)
        print(result)
        
    except Exception as e:
        print(f"테스트 2 실패: {e}")
    
    print("\n테스트 완료!")
