# MLCA 서브디비전 프로젝트 - 문제 해결 과정

이 문서는 MLCA(Multi-Linear Cell Averaging) 서브디비전 알고리즘을 wgpu-py와 PyVista로 구현하면서 마주친 문제들과 그 해결 방법을 정리합니다.

---

## 📋 목차

1. [MLCA 알고리즘 구현](#1-mlca-알고리즘-구현)
2. [일반 메쉬 지원 - VTK 파일 로딩](#2-일반-메쉬-지원---vtk-파일-로딩)
3. [N각 기둥 메쉬 생성 - 논문 스타일](#3-n각-기둥-메쉬-생성---논문-스타일)
4. [4개 뷰포트 비교 렌더링](#4-4개-뷰포트-비교-렌더링)
5. [시각화 개선 - Quad 와이어프레임](#5-시각화-개선---quad-와이어프레임)

---

## 1. MLCA 알고리즘 구현

### 문제

논문 [Multi-Linear Cell Averaging for Subdivision of Hexahedral Meshes](https://people.engr.tamu.edu/schaefer/research/hexahedral.pdf)의 알고리즘을 코드로 구현해야 했습니다.

### 해결 과정

MLCA는 2단계로 구성됩니다:

#### Step 1: Multi-linear Split (분할)

각 Hexahedron(8개 정점)을 8개의 sub-Hexahedra(27개 정점)로 분할합니다.

```
원본 Hex (8개 정점)              분할 후 (27개 정점)
       7-------6                      7---e6--6
      /|      /|                     /|  /|  /|
     4-------5 |         →          e7--f5--e5 |
     | 3-----|-2                    | e2-|-f2|-e1
     |/      |/                     |/  |/  |/
     0-------1                      0---e0--1
```

새로운 정점 위치:
- **Edge Point**: 엣지 중점 (12개)
- **Face Point**: 면 중심점 (6개)
- **Cell Point**: 셀 중심점 (1개)

```python
def _linear_split_hexs(self, points, cells):
    # 각 Hexahedron에 대해
    for hex_vertices in cells:
        # 1. 12개 Edge Point 계산
        for i, (v0, v1) in enumerate(HEX_EDGES):
            edge_pt = (points[v0] + points[v1]) / 2
            
        # 2. 6개 Face Point 계산
        for i, face_verts in enumerate(HEX_FACES):
            face_pt = np.mean(points[face_verts], axis=0)
            
        # 3. 1개 Cell Point 계산
        cell_pt = np.mean(points[hex_vertices], axis=0)
        
        # 4. 8개 sub-Hex 생성
        sub_hexs = build_8_sub_hexahedra(...)
```

#### Step 2: Cell Averaging (스무딩)

각 정점을 인접한 셀들의 무게중심 평균으로 이동시킵니다:

$$p_{new}(v) = \frac{1}{N} \sum_{i=1}^{N} \text{centroid}(C_i)$$

```python
def _cell_averaging(self, points, cells):
    new_positions = np.zeros_like(points)
    valence = np.zeros(len(points), dtype=int)
    
    for cell in cells:
        centroid = np.mean(points[cell], axis=0)
        for v in cell:
            new_positions[v] += centroid
            valence[v] += 1
    
    mask = valence > 0
    new_positions[mask] /= valence[mask, np.newaxis]
    return new_positions
```

### 핵심 구현 (`_mlca.py`)

```python
class MLCASubdivision:
    def subdivide(self, mesh, level=1):
        points = mesh.points.copy()
        cells = extract_hex_cells(mesh)
        
        for _ in range(level):
            points, cells = self._linear_split_hexs(points, cells)
            points = self._cell_averaging(points, cells)
        
        return create_unstructured_grid(points, cells)
```

---

## 2. 일반 메쉬 지원 - VTK 파일 로딩

### 문제

초기 구현은 단순 큐브만 지원했습니다. 외부 VTK 파일과 PyVista 예제 모델을 지원해야 했습니다.

### 해결: `_mesh_loader.py` 구현

#### VTK 파일 로딩

```python
def load_mesh(filepath: str) -> Union[pv.PolyData, pv.UnstructuredGrid]:
    """다양한 형식의 메쉬 파일 로드"""
    supported = ['.vtk', '.vtu', '.vtp', '.stl', '.obj', '.ply']
    return pv.read(filepath)
```

#### 표면 메쉬 → Hexahedral 변환

STL, OBJ 같은 표면 메쉬는 MLCA에 직접 사용할 수 없습니다. **복셀화(Voxelization)**를 통해 Hexahedral 볼륨 메쉬로 변환합니다:

```python
def surface_to_volume(mesh: pv.PolyData, resolution: int = 10) -> pv.UnstructuredGrid:
    """표면 메쉬를 Hexahedral 볼륨으로 변환"""
    # 1. 경계 박스 계산
    bounds = mesh.bounds
    
    # 2. 균일 격자 생성
    grid = pv.ImageData(dimensions=(resolution+1,)*3, ...)
    
    # 3. 표면 내부의 셀만 추출
    selected = grid.select_enclosed_points(mesh.extract_surface())
    
    # 4. Hexahedral 메쉬로 변환
    return selected.cast_to_unstructured_grid()
```

#### 메쉬 정보 출력

```python
@dataclass
class MeshInfo:
    mesh_type: str
    n_points: int
    n_cells: int
    cell_types: Dict[str, int]
    is_hexahedral: bool
    # ...
```

---

## 3. N각 기둥 메쉬 생성 - 논문 스타일

### 문제

논문의 Figure처럼 **모든 면이 사각형(Quad)인 N각 기둥**을 만들어야 했습니다.

초기 시도: 중앙에 축을 두고 방사형으로 Hex를 배치 → **중앙 면이 삼각형**이 되는 문제 발생

```
문제가 된 구조:
      ╲ | ╱
       ╲|╱
    ────●────   ← 중앙이 삼각형 모양
       ╱|╲
      ╱ | ╲
```

### 해결: N개의 직각삼각형 기둥 합치기

**핵심 아이디어**: 직각삼각형 단면의 기둥(1/N 조각)을 N개 만들어 중심을 향해 합치면, 모든 면이 사각형인 N각 기둥이 됩니다.

```
삼각형 기둥 (N=3):

      조각1        조각2        조각3          합친 결과
        ╱╲                                    ╱─────╲
       ╱  ╲         + ...     + ...    =    ╱   ●   ╲
      ╱────╲                               ╱─────────╲

각 조각은 직각삼각형 단면 → 모든 면이 Quad
```

#### 구현 (`create_prism_mesh.py`)

```python
def create_ngon_prism_hex_mesh(
    n_sides: int,       # 변의 수 (3=삼각형, 5=오각형, 6=육각형)
    n_radial: int,      # 반경 방향 분할
    n_angular: int,     # 각도 방향 분할 (각 조각 내)
    n_height: int       # 높이 방향 분할
) -> pv.UnstructuredGrid:
    
    # 점 중복 제거를 위한 딕셔너리
    point_dict = {}
    
    def get_or_create_point(coords, tol=1e-9):
        """좌표가 같은 점은 같은 인덱스 반환 (중복 제거)"""
        key = tuple(np.round(coords / tol).astype(int))
        if key not in point_dict:
            point_dict[key] = len(all_points)
            all_points.append(coords)
        return point_dict[key]
    
    # N개의 조각 생성
    for sector_idx in range(n_sides):
        angle_start = sector_idx * (2 * np.pi / n_sides)
        angle_end = (sector_idx + 1) * (2 * np.pi / n_sides)
        
        # 이 조각의 Hexahedra 생성
        for ir in range(n_radial):
            for ia in range(n_angular):
                for ih in range(n_height):
                    # 8개 정점 계산 (중심 공유점은 자동 병합)
                    hex_verts = [get_or_create_point(p) for p in hex_points]
                    all_cells.append(hex_verts)
```

#### 결과물

| 메쉬 | 구성 | 정점 | 셀 |
|------|------|------|-----|
| `triangular_prism.vtk` | 3개 조각 | 140 | 108 |
| `pentagonal_prism.vtk` | 5개 조각 | 230 | 180 |
| `hexagonal_prism.vtk` | 6개 조각 | 275 | 216 |
| `full_cylinder.vtk` | 12개 조각 | 365 | 288 |

---

## 4. 4개 뷰포트 비교 렌더링

### 문제

서브디비전 레벨 0~3을 한 화면에서 비교하고 싶었습니다.

### 해결: `MultiLevelRenderer` 클래스

각 메쉬에 대해 별도의 모델 행렬을 적용하여 2x2 그리드 배치:

```python
class MultiLevelRenderer:
    def draw_frame(self, canvas):
        # 4개 메쉬 위치 (2x2 그리드)
        positions = [
            (-1.5,  1.5, 0.0),  # Level 0 (좌상)
            ( 1.5,  1.5, 0.0),  # Level 1 (우상)
            (-1.5, -1.5, 0.0),  # Level 2 (좌하)
            ( 1.5, -1.5, 0.0),  # Level 3 (우하)
        ]
        
        for i, mesh_info in enumerate(self.meshes):
            pos = positions[i]
            model = translate(*pos) @ rotation_y(t) @ rotation_x(t * 0.5)
            self._write_uniforms(mesh_info["uniform_buffer"], model)
            
            # Solid + Wireframe 렌더링
            render_pass.set_pipeline(self.solid_pipeline)
            render_pass.draw(...)
            
            render_pass.set_pipeline(self.wireframe_pipeline)
            render_pass.draw(...)
```

#### 화면 구성

```
┌─────────────┬─────────────┐
│  Level 0    │  Level 1    │
│  (원본)      │  (8셀)      │
├─────────────┼─────────────┤
│  Level 2    │  Level 3    │
│  (64셀)     │  (512셀)    │
└─────────────┴─────────────┘
```

---

## 5. 시각화 개선 - Quad 와이어프레임

### 문제 1: 삼각형 와이어프레임

WebGPU는 Quad를 직접 지원하지 않아 삼각형으로 분할됩니다. 기존 와이어프레임은 삼각형 엣지를 모두 표시해서 시각적으로 복잡했습니다.

```
문제 (삼각형 엣지):       원하는 결과 (Quad 엣지):
┌───┬───┐                 ┌───────┐
│╲  │  ╱│                 │       │
├───┼───┤       →         ├───────┤
│╱  │  ╲│                 │       │
└───┴───┘                 └───────┘
```

### 해결: `extract_quad_edges()` 함수

삼각형화 전에 표면에서 Quad 엣지만 추출:

```python
def extract_quad_edges(mesh: pv.UnstructuredGrid, shrink_factor: float = 1.0) -> np.ndarray:
    """Hexahedral 메쉬에서 Quad 면의 엣지만 추출"""
    surface = mesh.extract_surface()
    
    edges_set = set()  # 중복 제거용
    edges_list = []
    
    # PyVista faces 배열에서 엣지 추출
    faces_arr = surface.faces
    idx = 0
    while idx < len(faces_arr):
        n_pts = faces_arr[idx]
        cell_pts = faces_arr[idx + 1: idx + 1 + n_pts]
        
        # 면의 엣지들 (순환)
        for j in range(n_pts):
            v0, v1 = cell_pts[j], cell_pts[(j + 1) % n_pts]
            edge = tuple(sorted([v0, v1]))  # 정렬해서 중복 방지
            if edge not in edges_set:
                edges_set.add(edge)
                edges_list.append((v0, v1))
        
        idx += n_pts + 1
    
    # 라인 세그먼트 배열 생성 [x1,y1,z1, x2,y2,z2, ...]
    return np.array([[*points[v0], *points[v1]] for v0, v1 in edges_list])
```

### 문제 2: 셀 구조 확인 어려움

서브디비전이 제대로 적용되었는지 셀 구조를 확인하기 어려웠습니다.

### 해결: `--shrink` 옵션

각 셀을 중심으로 수축시켜 셀 사이에 틈을 만듭니다:

```python
def format_for_render(mesh, mode='surface', shrink_factor=1.0):
    if shrink_factor < 1.0:
        processed = mesh.shrink(shrink_factor=shrink_factor)
    # ...
```

#### 사용법

```bash
# 15% 수축 (각 셀 사이에 틈 생성)
uv run python general_mesh_demo.py --file meshes/box.vtk --shrink 0.85

# 20% 수축
uv run python general_mesh_demo.py --file meshes/hexagonal_prism.vtk --shrink 0.8
```

### 와이어프레임 렌더링 파이프라인

별도의 position-only 버텍스 버퍼를 사용:

```python
# Wireframe 셰이더 (position only)
WIREFRAME_SHADER_SOURCE = """
@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOut {
    let world = u_scene.model * vec4<f32>(position, 1.0);
    out.pos = u_scene.view_proj * world;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.1, 0.1, 0.15, 1.0);  // 어두운 회색
}
"""

# 렌더링
render_pass.set_pipeline(self.wireframe_pipeline)
render_pass.set_vertex_buffer(0, mesh_info["edge_buffer"], ...)
render_pass.draw(mesh_info["edge_vertex_count"], 1, 0, 0)
```

---

## 📊 성능 고려사항

### 셀 개수 증가

| Level | 원본 27셀 기준 | 메모리 사용량 |
|-------|---------------|--------------|
| 0 | 27 | ~10 KB |
| 1 | 216 | ~80 KB |
| 2 | 1,728 | ~640 KB |
| 3 | 13,824 | ~5 MB |
| 4 | 110,592 | ~40 MB ⚠️ |

Level 4 이상은 메모리 사용량이 급격히 증가하므로 주의가 필요합니다.

### 최적화 팁

1. `--max-level 2`로 레벨 제한
2. `--resolution` 값을 낮게 유지 (표면→볼륨 변환 시)
3. 복잡한 메쉬는 먼저 단순화 후 MLCA 적용

---

## 🔧 WebGPU 관련 이슈

### 해결된 버그들

1. **stencil_load_op 누락**: `wgpu.LoadOp.clear` 명시 필요
2. **canvas.request_draw() 호출 필요**: 애니메이션 루프 유지
3. **행렬 column-major 순서**: `reshape(-1, order="F")` 사용

### 라인 두께 제한

WebGPU는 현재 라인 두께를 1픽셀로 제한합니다. 더 두꺼운 라인이 필요하면:
- Line을 Quad로 확장하는 별도 구현 필요
- 또는 geometry shader 사용 (WebGPU 미지원)

---

## 📚 참고 자료

- [MLCA 논문 (PDF)](https://people.engr.tamu.edu/schaefer/research/hexahedral.pdf)
- [PyVista 문서](https://docs.pyvista.org/)
- [wgpu-py GitHub](https://github.com/pygfx/wgpu-py)
- [VTK File Formats](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
