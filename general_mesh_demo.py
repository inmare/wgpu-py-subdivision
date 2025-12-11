# general_mesh_demo.py
"""
일반 메쉬에 MLCA 서브디비전을 적용하는 데모

이 스크립트는 다음을 시연합니다:
1. PyVista 예제 모델 로드 (Bunny, Cow, Dragon 등)
2. 외부 VTK 파일 로드
3. 표면 메쉬 → Hexahedral 볼륨 메쉬 변환
4. MLCA 서브디비전 적용 (레벨 0~3)
5. 4개의 뷰포트에 각 레벨 비교 렌더링

사용법:
    uv run python general_mesh_demo.py
    uv run python general_mesh_demo.py --model bunny
    uv run python general_mesh_demo.py --file model.vtk
"""

import argparse
import numpy as np
import wgpu
from wgpu import gpu
import wgpu.backends.auto
from wgpu.gui.auto import WgpuCanvas, run
import time

# 로컬 모듈
from _mesh_loader import (
    load_mesh,
    load_example_mesh,
    get_mesh_info,
    surface_to_volume,
    normalize_mesh,
)
from _mlca import (
    MLCASubdivision,
    format_for_render,
)
from _math import perspective, look_at, rotation_x, rotation_y, translate


# WebGPU 셰이더 (Solid)
SOLID_SHADER_SOURCE = """
struct VertexOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) world_pos : vec3<f32>,
};

struct SceneUniforms {
    model : mat4x4<f32>,
    view_proj : mat4x4<f32>,
    normal : mat4x4<f32>,
    light_dir : vec3<f32>,
    _pad0 : f32,
    camera_pos : vec3<f32>,
    _pad1 : f32,
};

@group(0) @binding(0) var<uniform> u_scene : SceneUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOut {
    var out : VertexOut;
    let world = u_scene.model * vec4<f32>(position, 1.0);
    out.world_pos = world.xyz;
    out.normal = normalize((u_scene.normal * vec4<f32>(normal, 0.0)).xyz);
    out.pos = u_scene.view_proj * world;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(u_scene.light_dir);
    let v = normalize(u_scene.camera_pos - in.world_pos);
    let h = normalize(l + v);

    let ambient = 0.15;
    let diff = max(dot(n, l), 0.0);
    let spec = pow(max(dot(n, h), 0.0), 32.0);

    let color = vec3<f32>(0.7, 0.75, 0.9);
    let lit = color * (ambient + diff * 0.7) + vec3<f32>(0.4) * spec;
    return vec4<f32>(lit, 1.0);
}
"""

# WebGPU 셰이더 (Wireframe)
WIREFRAME_SHADER_SOURCE = """
struct VertexOut {
    @builtin(position) pos : vec4<f32>,
};

struct SceneUniforms {
    model : mat4x4<f32>,
    view_proj : mat4x4<f32>,
    normal : mat4x4<f32>,
    light_dir : vec3<f32>,
    _pad0 : f32,
    camera_pos : vec3<f32>,
    _pad1 : f32,
};

@group(0) @binding(0) var<uniform> u_scene : SceneUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOut {
    var out : VertexOut;
    let world = u_scene.model * vec4<f32>(position, 1.0);
    out.pos = u_scene.view_proj * world;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);  // 검은색 와이어프레임
}
"""


def create_wireframe_indices(num_vertices: int) -> np.ndarray:
    """삼각형 리스트에서 와이어프레임 인덱스 생성"""
    num_triangles = num_vertices // 3
    indices = []
    for i in range(num_triangles):
        v0, v1, v2 = i * 3, i * 3 + 1, i * 3 + 2
        indices.extend([v0, v1, v1, v2, v2, v0])
    return np.array(indices, dtype=np.uint32)


class MultiLevelRenderer:
    """4개의 세분화 레벨을 동시에 렌더링하는 렌더러"""
    
    UNIFORM_BYTE_SIZE = 256
    DEPTH_FORMAT = wgpu.TextureFormat.depth24plus
    
    def __init__(self, device: wgpu.GPUDevice, texture_format: wgpu.TextureFormat, meshes_data: list):
        """
        Parameters
        ----------
        meshes_data : list
            각 레벨의 vertex_data를 담은 리스트 [(level, vertex_data), ...]
        """
        self.device = device
        self.texture_format = texture_format
        self.start_time = time.perf_counter()
        self.depth_texture = None
        
        # 바인드 그룹 레이아웃
        self.bgl = device.create_bind_group_layout(
            entries=[{
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }]
        )
        
        # 각 레벨별 메쉬 데이터 및 버퍼 생성
        self.meshes = []
        usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        
        for level, vertex_data in meshes_data:
            vertex_buffer = device.create_buffer_with_data(
                data=vertex_data.tobytes(),
                usage=wgpu.BufferUsage.VERTEX
            )
            
            wire_indices = create_wireframe_indices(len(vertex_data) // 6)
            wire_index_buffer = device.create_buffer_with_data(
                data=wire_indices.tobytes(),
                usage=wgpu.BufferUsage.INDEX
            )
            
            uniform_buffer = device.create_buffer(size=self.UNIFORM_BYTE_SIZE, usage=usage)
            
            bind_group = device.create_bind_group(
                layout=self.bgl,
                entries=[{
                    "binding": 0,
                    "resource": {"buffer": uniform_buffer, "offset": 0, "size": self.UNIFORM_BYTE_SIZE}
                }],
            )
            
            self.meshes.append({
                "level": level,
                "vertex_data": vertex_data,
                "vertex_buffer": vertex_buffer,
                "wire_indices": wire_indices,
                "wire_indices_len": len(wire_indices),
                "wire_index_buffer": wire_index_buffer,
                "uniform_buffer": uniform_buffer,
                "bind_group": bind_group,
            })
        
        # 파이프라인 생성
        self.solid_pipeline = self._create_solid_pipeline()
        self.wireframe_pipeline = self._create_wireframe_pipeline()
    
    def _create_vertex_state(self, shader):
        return {
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [{
                "array_stride": 6 * 4,
                "attributes": [
                    {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                    {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1},
                ],
                "step_mode": wgpu.VertexStepMode.vertex,
            }]
        }
    
    def _create_solid_pipeline(self) -> wgpu.GPURenderPipeline:
        shader = self.device.create_shader_module(code=SOLID_SHADER_SOURCE)
        return self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bgl]),
            vertex=self._create_vertex_state(shader),
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": self.DEPTH_FORMAT,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
                "stencil_front": {"compare": wgpu.CompareFunction.always, "fail_op": wgpu.StencilOperation.keep, "depth_fail_op": wgpu.StencilOperation.keep, "pass_op": wgpu.StencilOperation.keep},
                "stencil_back": {"compare": wgpu.CompareFunction.always, "fail_op": wgpu.StencilOperation.keep, "depth_fail_op": wgpu.StencilOperation.keep, "pass_op": wgpu.StencilOperation.keep},
                "stencil_read_mask": 0xFFFFFFFF,
                "stencil_write_mask": 0xFFFFFFFF,
            },
            multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.texture_format}],
            },
        )
    
    def _create_wireframe_pipeline(self) -> wgpu.GPURenderPipeline:
        shader = self.device.create_shader_module(code=WIREFRAME_SHADER_SOURCE)
        return self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bgl]),
            vertex=self._create_vertex_state(shader),
            primitive={
                "topology": wgpu.PrimitiveTopology.line_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": self.DEPTH_FORMAT,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.less_equal,
                "stencil_front": {"compare": wgpu.CompareFunction.always, "fail_op": wgpu.StencilOperation.keep, "depth_fail_op": wgpu.StencilOperation.keep, "pass_op": wgpu.StencilOperation.keep},
                "stencil_back": {"compare": wgpu.CompareFunction.always, "fail_op": wgpu.StencilOperation.keep, "depth_fail_op": wgpu.StencilOperation.keep, "pass_op": wgpu.StencilOperation.keep},
                "stencil_read_mask": 0xFFFFFFFF,
                "stencil_write_mask": 0xFFFFFFFF,
            },
            multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.texture_format}],
            },
        )
    
    def _write_uniforms(self, buffer, width, height, model):
        """유니폼 버퍼에 데이터 쓰기"""
        aspect = width / height
        eye = np.array([0.0, 0.0, 8.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        view = look_at(eye, target, up)
        proj = perspective(np.radians(45), aspect, 0.1, 100.0)
        
        light_dir = np.array([0.3, 0.7, 0.55], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        normal3 = np.linalg.inv(model[:3, :3]).T
        normal = np.eye(4, dtype=np.float32)
        normal[:3, :3] = normal3
        
        model_cm = model.astype(np.float32).reshape(-1, order="F")
        view_proj = proj @ view
        view_proj_cm = view_proj.astype(np.float32).reshape(-1, order="F")
        normal_cm = normal.astype(np.float32).reshape(-1, order="F")
        
        data = np.concatenate([
            model_cm,
            view_proj_cm,
            normal_cm,
            np.append(light_dir, 0.0).astype(np.float32),
            np.append(eye, 0.0).astype(np.float32),
        ])
        
        pad_floats = (self.UNIFORM_BYTE_SIZE // 4) - data.size
        if pad_floats > 0:
            data = np.pad(data, (0, pad_floats), mode="constant")
        
        self.device.queue.write_buffer(buffer, 0, data.tobytes())
    
    def draw_frame(self, canvas: WgpuCanvas):
        current_texture = canvas.get_context().get_current_texture()
        tex_width, tex_height, _ = current_texture.size
        
        if self.depth_texture is None or self.depth_texture.size[0] != tex_width or self.depth_texture.size[1] != tex_height:
            self.depth_texture = self.device.create_texture(
                size=(tex_width, tex_height, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                format=self.DEPTH_FORMAT,
            )
        
        t = time.perf_counter() - self.start_time
        
        # 4개의 메쉬 위치 (2x2 그리드)
        positions = [
            (-1.5,  1.5, 0.0),  # Level 0 (Top-Left)
            ( 1.5,  1.5, 0.0),  # Level 1 (Top-Right)
            (-1.5, -1.5, 0.0),  # Level 2 (Bottom-Left)
            ( 1.5, -1.5, 0.0),  # Level 3 (Bottom-Right)
        ]
        
        # 유니폼 업데이트
        for i, mesh_info in enumerate(self.meshes):
            if i < len(positions):
                pos = positions[i]
                model = translate(*pos) @ rotation_y(t * 0.5) @ rotation_x(t * 0.25)
                self._write_uniforms(mesh_info["uniform_buffer"], tex_width, tex_height, model)
        
        color_view = current_texture.create_view()
        depth_view = self.depth_texture.create_view()
        
        color_attachment = {
            "view": color_view,
            "resolve_target": None,
            "load_op": wgpu.LoadOp.clear,
            "clear_value": (0.08, 0.08, 0.12, 1.0),
            "store_op": wgpu.StoreOp.store,
        }
        
        depth_attachment = {
            "view": depth_view,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_clear_value": 1.0,
            "depth_store_op": wgpu.StoreOp.store,
            "stencil_load_op": wgpu.LoadOp.clear,
            "stencil_store_op": wgpu.StoreOp.discard,
        }
        
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[color_attachment],
            depth_stencil_attachment=depth_attachment,
        )
        
        # 각 메쉬 렌더링
        for mesh_info in self.meshes:
            # Solid 렌더링
            render_pass.set_pipeline(self.solid_pipeline)
            render_pass.set_bind_group(0, mesh_info["bind_group"], [], 0, 999_999)
            render_pass.set_vertex_buffer(0, mesh_info["vertex_buffer"], 0, mesh_info["vertex_buffer"].size)
            n_vertices = len(mesh_info["vertex_data"]) // 6
            render_pass.draw(n_vertices, 1, 0, 0)
            
            # Wireframe 렌더링
            render_pass.set_pipeline(self.wireframe_pipeline)
            render_pass.set_bind_group(0, mesh_info["bind_group"], [], 0, 999_999)
            render_pass.set_vertex_buffer(0, mesh_info["vertex_buffer"], 0, mesh_info["vertex_buffer"].size)
            render_pass.set_index_buffer(
                mesh_info["wire_index_buffer"],
                wgpu.IndexFormat.uint32,
                0,
                mesh_info["wire_index_buffer"].size
            )
            render_pass.draw_indexed(mesh_info["wire_indices_len"], 1, 0, 0, 0)
        
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        canvas.request_draw()


def main():
    parser = argparse.ArgumentParser(description='일반 메쉬에 MLCA 서브디비전 적용 (4개 레벨 비교)')
    parser.add_argument('--model', type=str, default='cube',
                        help='PyVista 예제 모델 (bunny, cow, dragon, armadillo, teapot, sphere, cube, hexbeam)')
    parser.add_argument('--file', type=str, default=None,
                        help='로드할 VTK 파일 경로')
    parser.add_argument('--resolution', type=int, default=8,
                        help='복셀화 해상도 (표면 메쉬인 경우)')
    parser.add_argument('--mode', type=str, default='surface', choices=['surface', 'volume'],
                        help='렌더링 모드 (surface: 외형, volume: 내부 구조)')
    parser.add_argument('--info', action='store_true',
                        help='메쉬 정보만 출력하고 종료')
    parser.add_argument('--max-level', type=int, default=3,
                        help='최대 세분화 레벨 (0~3)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLCA 서브디비전 데모 (4개 레벨 비교)")
    print("=" * 60)
    
    # 1. 메쉬 로드
    if args.file:
        print(f"\n[1] VTK 파일 로드: {args.file}")
        mesh = load_mesh(args.file)
    else:
        print(f"\n[1] 예제 모델 로드: {args.model}")
        mesh = load_example_mesh(args.model)
    
    # 메쉬 정보 출력
    info = get_mesh_info(mesh)
    print(info)
    
    if args.info:
        return
    
    # 2. Hexahedral 메쉬로 변환 (필요한 경우)
    import pyvista as pv
    
    if isinstance(mesh, pv.UnstructuredGrid) and info.is_hexahedral:
        print(f"\n[2] 이미 Hexahedral 메쉬입니다. 변환 불필요.")
        hex_mesh = mesh
    else:
        print(f"\n[2] 표면 메쉬 → Hexahedral 볼륨 변환 (해상도: {args.resolution})")
        
        # 정규화
        mesh = normalize_mesh(mesh)
        
        # PolyData 추출
        if not isinstance(mesh, pv.PolyData):
            if hasattr(mesh, 'extract_surface'):
                mesh = mesh.extract_surface()
            else:
                mesh = pv.wrap(mesh)
        
        # 복셀화
        hex_mesh = surface_to_volume(mesh, resolution=args.resolution)
    
    # 3. 각 레벨별 MLCA 서브디비전 적용
    print(f"\n[3] MLCA 서브디비전 적용 (Level 0~{args.max_level})")
    
    mlca = MLCASubdivision(verbose=False)
    meshes_data = []
    
    for level in range(args.max_level + 1):
        print(f"    Level {level} 처리 중...", end=" ")
        
        if level == 0:
            subdivided_mesh = hex_mesh.copy()
        else:
            result = mlca.subdivide(hex_mesh, level=level)
            subdivided_mesh = result.mesh
        
        # 메쉬 정규화 (원점 중심, 크기 1.0)
        center = np.array(subdivided_mesh.center)
        subdivided_mesh.translate(-center, inplace=True)
        
        bounds = subdivided_mesh.bounds
        max_dim = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        )
        if max_dim > 0:
            scale = 1.8 / max_dim  # 약간 작게 해서 겹치지 않도록
            subdivided_mesh.scale([scale, scale, scale], inplace=True)
        
        # 렌더링 데이터 생성
        vertex_data = format_for_render(subdivided_mesh, mode=args.mode)
        meshes_data.append((level, vertex_data))
        
        print(f"완료 ({len(vertex_data) // 6} 정점)")
    
    # 4. WebGPU 렌더링
    print(f"\n[4] WebGPU 렌더링 시작")
    print("    4개의 뷰: Level 0 (좌상), Level 1 (우상), Level 2 (좌하), Level 3 (우하)")
    print("    창을 닫으면 종료됩니다.")
    
    canvas = WgpuCanvas(title=f"MLCA Demo - {args.model} (Level 0~{args.max_level})")
    adapter = gpu.request_adapter(canvas=canvas, power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("GPU 어댑터를 찾을 수 없습니다.")
    
    device = adapter.request_device()
    context = canvas.get_context()
    texture_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=texture_format)
    
    renderer = MultiLevelRenderer(device, texture_format, meshes_data)
    
    def draw_frame():
        renderer.draw_frame(canvas)
    
    canvas.draw_frame = draw_frame
    run()


if __name__ == "__main__":
    main()
