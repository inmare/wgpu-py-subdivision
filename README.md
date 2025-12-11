# MLCA Subdivision with WebGPU

`wgpu-py`와 `PyVista`를 사용하여 **MLCA (Multi-Linear Cell Averaging)** 서브디비전 알고리즘을 구현하고 WebGPU로 실시간 렌더링하는 프로젝트입니다.

참고 논문: [Multi-Linear Cell Averaging for Subdivision of Hexahedral Meshes](https://people.engr.tamu.edu/schaefer/research/hexahedral.pdf)

---

## 🚀 빠른 시작

### 설치

```powershell
# 가상환경 생성 및 의존성 설치
uv venv
uv pip install -e .
```

### 실행 명령어

```powershell
# 1. 큐브 세분화 비교 (기본 예제)
uv run python cube.py

# 2. 논문 스타일 메쉬 생성 (1/4 원기둥, 삼각기둥, 오각기둥 등)
uv run python create_prism_mesh.py

# 3. 생성된 메쉬에 MLCA 적용 (4개 레벨 비교 뷰)
uv run python general_mesh_demo.py --file meshes/quarter_cylinder.vtk
uv run python general_mesh_demo.py --file meshes/triangular_prism.vtk
uv run python general_mesh_demo.py --file meshes/pentagonal_prism.vtk
uv run python general_mesh_demo.py --file meshes/hexagonal_prism.vtk
uv run python general_mesh_demo.py --file meshes/full_cylinder.vtk

# 4. PyVista 예제 모델 사용 (표면→볼륨 변환 후 MLCA 적용)
uv run python general_mesh_demo.py --model bunny --resolution 15
uv run python general_mesh_demo.py --model cow --resolution 12

# 5. 옵션 설명
#   --file <path>       : VTK 파일 경로
#   --model <name>      : PyVista 예제 모델 (bunny, cow, dragon, sphere, cube 등)
#   --resolution <n>    : 복셀화 해상도 (표면 메쉬인 경우, 기본값 8)
#   --max-level <n>     : 최대 세분화 레벨 (0~3, 기본값 3)
#   --mode <mode>       : 렌더링 모드 (surface/volume)
#   --info              : 메쉬 정보만 출력하고 종료
```

### 화면 구성 (4개 뷰포트)

```
┌─────────────┬─────────────┐
│  Level 0    │  Level 1    │
│  (원본)      │  (1단계)    │
├─────────────┼─────────────┤
│  Level 2    │  Level 3    │
│  (2단계)     │  (3단계)    │
└─────────────┴─────────────┘
```

---

## 📚 서브디비전(Subdivision)이란?

### 기본 개념

**서브디비전(Subdivision)**은 거친 메쉬를 점점 더 부드럽게 만드는 알고리즘입니다. 간단히 말해:

1. 각 면(Face)이나 셀(Cell)을 **더 작은 조각으로 분할**하고
2. 정점(Vertex)의 위치를 **부드럽게 조정**합니다

이 과정을 반복하면 각진 정육면체가 점점 구에 가까워지게 됩니다.

```
Level 0 (원본)     Level 1          Level 2          Level 3
    ┌───┐           ┌─┬─┐           ┌┬┬┬┐            부드러운
    │   │    →      ├─┼─┤    →      ├┼┼┼┤     →       구 형태
    └───┘           └─┴─┘           └┴┴┴┘
```

### 전통적인 서브디비전 vs MLCA

| 구분 | 전통적인 서브디비전 (Catmull-Clark 등) | MLCA |
|------|---------------------------------------|------|
| **적용 대상** | 2D 표면 메쉬 (삼각형, 사각형) | 3D 볼륨 메쉬 (육면체/Hexahedron) |
| **결과물** | 빈 껍데기 (표면만 존재) | 속이 찬 볼륨 (내부 구조 있음) |
| **용도** | 캐릭터 모델링, 애니메이션 | 유한요소해석(FEA), 시뮬레이션 |
| **스무딩 방식** | 엣지/면 기반 가중 평균 | **셀(Cell) 무게중심 평균** |

### 왜 MLCA를 사용할까?

전통적인 Catmull-Clark은 **표면 메쉬**에 적합하지만, 구조 해석이나 물리 시뮬레이션에서는 **내부가 채워진 볼륨 메쉬**가 필요합니다. MLCA는 이런 Hexahedral(육면체) 볼륨 메쉬를 부드럽게 만들기 위해 설계되었습니다.

---

## 🔧 MLCA 알고리즘 작동 방식

MLCA는 각 레벨에서 **2단계**를 수행합니다:

### Step 1: Multi-linear Subdivision (분할)

각 Hexahedron(육면체)을 **8개의 sub-Hexahedra**로 분할합니다.

```
원본 Hex (8개 정점)              분할 후 (27개 정점, 8개 sub-Hex)
       7-------6                      7---*---6
      /|      /|                     /|  /|  /|
     4-------5 |         →          *---*---* |
     | 3-----|-2                    | *-|-*-|-*
     |/      |/                     |/  |/  |/
     0-------1                      0---*---1
```

새로운 정점 생성 위치:
- **Edge Point (엣지 중점)**: 12개 엣지의 중점 = 12개
- **Face Point (면 중심)**: 6개 면의 중심 = 6개  
- **Cell Point (셀 중심)**: 육면체의 중심 = 1개

→ 원래 8개 정점 + 19개 새 정점 = **27개 정점**으로 **8개 sub-Hex** 생성

### Step 2: Cell Averaging (스무딩)

각 정점을 **인접 셀들의 무게중심 평균**으로 이동시킵니다.

수식으로 표현하면:
$$p_{new}(v) = \frac{1}{N} \sum_{i=1}^{N} \text{centroid}(C_i)$$

여기서:
- $v$: 정점
- $N$: 정점 $v$를 포함하는 셀의 개수 (Valence)
- $C_i$: 정점 $v$를 포함하는 i번째 셀
- $\text{centroid}(C_i)$: 셀 $C_i$의 무게중심 (8개 정점의 평균 좌표)

```python
# Cell Averaging 핵심 코드
for each cell in mesh:
    centroid = average(cell.vertices)  # 셀의 무게중심
    for vertex in cell.vertices:
        vertex.new_position += centroid
        vertex.count += 1

for each vertex:
    vertex.position = vertex.new_position / vertex.count  # 평균
```

### 레벨별 셀 개수 증가

| Level | 셀 개수 | 비고 |
|-------|--------|------|
| 0 | 1 | 원본 |
| 1 | 8 | 8배 증가 |
| 2 | 64 | 8² |
| 3 | 512 | 8³ |
| n | 8ⁿ | 기하급수적 증가 |

⚠️ Level 4 이상은 메모리 사용량이 급격히 증가하므로 주의!

---

## 📂 VTK 파일 형식과 메쉬 데이터 구조

### VTK란?

**VTK (Visualization Toolkit)**는 3D 데이터를 저장하고 시각화하기 위한 오픈소스 라이브러리입니다. VTK 파일 형식은 3D 메쉬 데이터를 저장하는 표준 형식 중 하나입니다.

### 메쉬의 구성 요소

3D 메쉬는 크게 두 가지로 구성됩니다:

```
1. Points (정점 좌표)
   - 3D 공간의 점 위치 (x, y, z)
   - 예: [(0,0,0), (1,0,0), (1,1,0), ...]

2. Cells (셀/면 정의)  
   - 어떤 정점들이 연결되어 면/볼륨을 이루는지
   - 예: Triangle[0,1,2], Quad[0,1,2,3], Hexahedron[0,1,2,3,4,5,6,7]
```

### PyVista로 VTK 파일 읽기

이 프로젝트에서는 PyVista 라이브러리를 사용해 VTK 파일을 읽습니다:

```python
import pyvista as pv

# 파일 읽기
mesh = pv.read("model.vtk")

# 기본 정보 접근
print(mesh.points)      # 정점 좌표 배열 (N x 3)
print(mesh.n_points)    # 정점 개수
print(mesh.n_cells)     # 셀 개수
print(mesh.cells)       # 셀 정의 배열
print(mesh.celltypes)   # 셀 타입 (Triangle=5, Quad=9, Hex=12 등)
```

### 셀 타입 코드

| 코드 | 타입 | 정점 수 | 설명 |
|------|------|--------|------|
| 5 | Triangle | 3 | 삼각형 (표면) |
| 9 | Quad | 4 | 사각형 (표면) |
| 10 | Tetrahedron | 4 | 사면체 (볼륨) |
| **12** | **Hexahedron** | **8** | **육면체 (볼륨) - MLCA 대상** |
| 13 | Wedge | 6 | 쐐기형 (볼륨) |
| 14 | Pyramid | 5 | 피라미드 (볼륨) |

### Hexahedron 정점 순서 (VTK 표준)

```
       7-------6
      /|      /|
     4-------5 |      정점 순서: 0,1,2,3 (아래면 시계방향)
     | 3-----|-2                4,5,6,7 (위면 시계방향)
     |/      |/
     0-------1
```

### 이 프로젝트에서 사용하는 데이터

```python
# _mesh_loader.py에서 추출하는 정보
from _mesh_loader import load_mesh, get_mesh_info

mesh = load_mesh("model.vtk")
info = get_mesh_info(mesh)

# 사용되는 데이터:
# 1. mesh.points → 정점 좌표 (MLCA의 점 위치 계산에 사용)
# 2. mesh.cells → 셀 연결 정보 (어떤 점들이 하나의 Hex를 이루는지)
# 3. mesh.celltypes → 셀 타입 (Hexahedron인지 확인)
```

### 표면 메쉬 → Hexahedral 변환

STL, OBJ 같은 파일은 **표면 메쉬(PolyData)**만 포함합니다. MLCA는 **볼륨 메쉬(UnstructuredGrid)**가 필요하므로, **복셀화(Voxelization)**를 통해 변환합니다:

```python
from _mesh_loader import surface_to_volume

# 표면 메쉬를 Hexahedral 볼륨으로 변환
hex_mesh = surface_to_volume(surface_mesh, resolution=10)

# 내부적으로 수행되는 과정:
# 1. 메쉬의 경계 박스(Bounding Box) 계산
# 2. 박스를 균일한 격자(Grid)로 분할
# 3. 표면 내부에 있는 셀만 추출 → Hexahedral 메쉬
```

---

## 사전 준비

- Python 3.10 이상
- [uv](https://github.com/astral-sh/uv) 설치
- GPU 드라이버가 최신 버전인지 확인 (DX12/Vulkan 지원)

> 💡 **설치 및 실행 명령어는 상단의 [🚀 빠른 시작](#-빠른-시작) 섹션을 참조하세요.**

## 새로운 기능: 일반 메쉬 지원

### VTK 파일 로드

```python
from _mesh_loader import load_mesh, get_mesh_info

# VTK 파일 로드
mesh = load_mesh("model.vtk")

# 메쉬 정보 출력
info = get_mesh_info(mesh)
print(info)
```

### PyVista 예제 모델 사용

```python
from _mesh_loader import load_example_mesh

# 다양한 예제 모델 사용 가능
bunny = load_example_mesh('bunny')
cow = load_example_mesh('cow')
dragon = load_example_mesh('dragon')
```

### 표면 메쉬 → Hexahedral 변환 → MLCA 적용

```python
from _mesh_loader import load_example_mesh, surface_to_volume, normalize_mesh
from _mlca import subdivide_hexahedral_mesh

# 1. 표면 메쉬 로드
bunny = load_example_mesh('bunny')
bunny = normalize_mesh(bunny)

# 2. Hexahedral 볼륨으로 변환
hex_mesh = surface_to_volume(bunny, resolution=10)

# 3. MLCA 서브디비전 적용
subdivided = subdivide_hexahedral_mesh(hex_mesh, level=2)
```

### 커맨드라인 사용

```powershell
# 논문 스타일 메쉬 생성 후 MLCA 적용 (권장)
uv run python create_prism_mesh.py
uv run python general_mesh_demo.py --file meshes/quarter_cylinder.vtk
uv run python general_mesh_demo.py --file meshes/pentagonal_prism.vtk

# PyVista 예제 모델 사용 (표면→볼륨 변환 필요)
uv run python general_mesh_demo.py --model bunny --resolution 15
uv run python general_mesh_demo.py --model cow --resolution 12

# 내부 구조 보기 (셀 수축)
uv run python general_mesh_demo.py --file meshes/quarter_cylinder.vtk --mode volume

# 메쉬 정보만 확인
uv run python general_mesh_demo.py --file meshes/quarter_cylinder.vtk --info
```

## 파일 설명

### 핵심 모듈
| 파일 | 역할 | 주요 함수/클래스 |
|------|------|-----------------|
| `_mlca.py` | MLCA 서브디비전 알고리즘 | `MLCASubdivision`, `subdivide_hexahedral_mesh()` |
| `_mesh_loader.py` | VTK/PyVista 파일 로더 | `load_mesh()`, `get_mesh_info()`, `surface_to_volume()` |
| `_mesh_volume.py` | 볼륨 메쉬용 유틸리티 | `subdivided_volume_grid()` |
| `_mesh.py` | 표면 메쉬용 유틸리티 | `cube_vertices()`, `subdivided_cube_vertices()` |
| `_math.py` | 3D 그래픽스 행렬 함수 | `perspective()`, `look_at()`, `rotation_y()` |
| `_renderer.py` | WebGPU 렌더링 파이프라인 | `CubeRenderer` |

### 예제 스크립트
| 파일 | 역할 |
|------|------|
| `triangle.py` | 기본 삼각형 렌더링 |
| `cube.py` | 큐브 세분화 레벨 비교 (4개 뷰포트) |
| `create_prism_mesh.py` | 논문 스타일 Hexahedral 메쉬 생성 (1/4 원기둥, 삼각/오각/육각 기둥) |
| `general_mesh_demo.py` | 일반 메쉬에 MLCA 적용 (4개 레벨 비교 뷰) |

### 생성된 메쉬 파일 (`meshes/` 폴더)

`create_prism_mesh.py` 실행 시 생성되는 Hexahedral 메쉬들:

| 파일 | 형태 | 정점 | 셀 | 설명 |
|------|------|------|-----|------|
| `quarter_cylinder.vtk` | 1/4 원기둥 | 125 | 64 | 논문 Figure 스타일 |
| `triangular_prism.vtk` | 삼각기둥 | 140 | 108 | 3개 조각 합침 |
| `pentagonal_prism.vtk` | 오각기둥 | 230 | 180 | 5개 조각 합침 |
| `hexagonal_prism.vtk` | 육각기둥 | 275 | 216 | 6개 조각 합침 |
| `box.vtk` | 직육면체 | 64 | 27 | 기본 정육면체 |
| `full_cylinder.vtk` | 원기둥 | 365 | 288 | 12개 조각 합침 |

> 💡 **원리**: 직각삼각형 기둥(1/N 조각)을 N개 만들어 중심을 향해 합치면,
> 모든 면이 사각형인 N각 기둥이 생성됩니다.

---

## 🗂️ 코드 구조 상세

### 데이터 흐름

```
[VTK 파일 / 예제 모델]
        │
        ▼ pv.read() 또는 load_example_mesh()
┌───────────────────┐
│   PyVista Mesh    │
│  (PolyData 또는   │
│  UnstructuredGrid)│
└───────────────────┘
        │
        ▼ surface_to_volume() (표면인 경우)
┌───────────────────┐
│ Hexahedral Mesh   │
│ (UnstructuredGrid)│
│  - points: (N,3)  │
│  - cells: [8,v0,. │
└───────────────────┘
        │
        ▼ MLCASubdivision.subdivide()
┌───────────────────┐
│ Subdivided Mesh   │
│  (더 많은 정점과  │
│   부드러운 형태)  │
└───────────────────┘
        │
        ▼ format_for_render()
┌───────────────────┐
│ Vertex Buffer     │
│ [pos,norm,pos,...]│
└───────────────────┘
        │
        ▼ WebGPU Render Pipeline
┌───────────────────┐
│   화면에 렌더링   │
└───────────────────┘
```

### 핵심 알고리즘 위치

```python
# _mlca.py

class MLCASubdivision:
    def subdivide(self, mesh, level):
        for _ in range(level):
            # Step 1: Split
            points, cells = self._linear_split_hexs(points, cells)
            
            # Step 2: Smooth  
            points = self._cell_averaging(points, cells)
        
        return result
    
    def _linear_split_hexs(self, points, cells):
        # 각 Hex를 8개로 분할
        # Edge Point, Face Point, Cell Point 생성
        ...
    
    def _cell_averaging(self, points, cells):
        # 각 정점을 인접 셀 무게중심의 평균으로 이동
        # p_new[v] = Σ centroid(C) / N
        ...
```

---

## 지원 파일 형식

| 형식 | 확장자 | 설명 |
|------|--------|------|
| VTK Legacy | `.vtk` | VTK 표준 형식 |
| VTK XML | `.vtu`, `.vtp`, `.vts`, `.vtr`, `.vti` | XML 기반 VTK |
| STL | `.stl` | 3D 프린팅 표준 |
| OBJ | `.obj` | Wavefront 형식 |
| PLY | `.ply` | Stanford 형식 |
| GMSH | `.msh` | 메쉬 생성 도구 |

---

## 📖 학습 자료

### 서브디비전 기초
- [Subdivision Surfaces (Wikipedia)](https://en.wikipedia.org/wiki/Subdivision_surface)
- [Catmull-Clark subdivision surface](https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface)

### MLCA 논문
- [Multi-Linear Cell Averaging for Subdivision of Hexahedral Meshes (PDF)](https://people.engr.tamu.edu/schaefer/research/hexahedral.pdf)

### PyVista/VTK
- [PyVista 공식 문서](https://docs.pyvista.org/)
- [VTK File Formats](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)

### WebGPU
- [wgpu-py 공식 저장소](https://github.com/pygfx/wgpu-py)
- [WebGPU 명세](https://www.w3.org/TR/webgpu/)

---

## 문제 해결

| 문제 | 해결 방법 |
|------|----------|
| 창이 바로 닫힘 | GPU 드라이버 업데이트 또는 다른 전원 모드에서 시도 |
| 어댑터를 못 찾음 | 외장 GPU가 비활성화되지 않았는지 확인 (노트북은 고성능 모드) |
| 백엔드 충돌 | `pip uninstall glfw` 후 `pip install glfw==2.7.*` 재설치 |
| 메모리 부족 | `--resolution`과 `--level` 값을 낮추세요 |
| Import 오류 | `uv pip install -e .`로 패키지 재설치 |

---

## 라이선스

MIT License

