# _renderer.py

import time
import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas
# 변경 전: from ._math import perspective, look_at, rotation_x, rotation_y, translate
# 변경 후:
from _math import perspective, look_at, rotation_x, rotation_y, translate 

# 변경 전: from ._mesh import cube_vertices, subdivided_cube_vertices, create_wireframe_indices
# 변경 후:
from _mesh_volume import cube_vertices, subdivided_cube_vertices, create_wireframe_indices

# ... (CubeRenderer 클래스 이하 동일)

# WebGPU 셰이더 소스
SOLID_SHADER_SOURCE = """
struct VertexOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) world_pos : vec3<f32>,
};

// Python 코드의 SceneUniforms 구조체와 일치해야 합니다.
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
    // Phong Lighting Model
    let n = normalize(in.normal);
    let l = normalize(u_scene.light_dir);
    let v = normalize(u_scene.camera_pos - in.world_pos);
    let h = normalize(l + v);

    let ambient = 0.12;
    let diff = max(dot(n, l), 0.0);
    let spec = pow(max(dot(n, h), 0.0), 32.0);

    let color = vec3<f32>(0.8, 0.85, 1.0); // 물체 색상
    let lit = color * (ambient + diff) + vec3<f32>(0.3) * spec; // 최종 조명 색상
    return vec4<f32>(lit, 1.0);
}
"""

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

class CubeRenderer:
    UNIFORM_BYTE_SIZE = 256
    DEPTH_FORMAT = wgpu.TextureFormat.depth24plus
    
    def __init__(self, device: wgpu.GPUDevice, texture_format: wgpu.TextureFormat):
        self.device = device
        self.texture_format = texture_format
        self.start_time = time.perf_counter()
        self.depth_texture = None

        # 세분화 레벨 정의
        self.subdivision_levels = [0, 1, 2, 3] # 원본, 1회, 2회, 3회
        
        # 1. 메쉬 데이터 로드 및 버퍼 생성
        self.meshes = []
        for level in self.subdivision_levels:
            if level == 0:
                vertex_data = cube_vertices()
            else:
                vertex_data = subdivided_cube_vertices(level=level)

            vertex_buffer = device.create_buffer_with_data(
                data=vertex_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
            )
            wire_indices = create_wireframe_indices(vertex_data.size // 6)
            wire_index_buffer = device.create_buffer_with_data(
                data=wire_indices.tobytes(), usage=wgpu.BufferUsage.INDEX
            )

            self.meshes.append({
                "level": level,
                "vertex_data": vertex_data,
                "vertex_buffer": vertex_buffer,
                "wire_indices_len": len(wire_indices),
                "wire_index_buffer": wire_index_buffer
            })

        # 2. 4개의 유니폼 버퍼 생성
        usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        self.uniform_buffers = [device.create_buffer(size=self.UNIFORM_BYTE_SIZE, usage=usage) for _ in range(4)]
        
        # 3. 4개의 바인드 그룹 생성
        self.bgl = device.create_bind_group_layout(
            entries=[{
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }]
        )
        self.bind_groups = [self._create_bind_group(buf) for buf in self.uniform_buffers]
        
        # 4. 파이프라인 생성 (이전과 동일)
        self.solid_pipeline = self._create_solid_pipeline()
        self.wireframe_pipeline = self._create_wireframe_pipeline()

    def _create_bind_group(self, buffer: wgpu.GPUBuffer) -> wgpu.GPUBindGroup:
        """단일 유니폼 버퍼를 위한 바인드 그룹을 생성합니다."""
        return self.device.create_bind_group(
            layout=self.bgl,
            entries=[{"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}}],
        )

    def _create_vertex_state(self, shader: wgpu.GPUShaderModule) -> dict:
        """Solid 및 Wireframe 파이프라인에 공통으로 사용되는 Vertex State를 반환합니다."""
        return {
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [{
                "array_stride": 6 * 4, # 3 pos + 3 normal = 6 floats * 4 bytes
                "attributes": [
                    {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0}, # position
                    {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1}, # normal
                ],
                "step_mode": wgpu.VertexStepMode.vertex,
            }]
        }

    def _create_solid_pipeline(self) -> wgpu.GPURenderPipeline:
        """솔리드 렌더링 파이프라인을 생성합니다."""
        shader = self.device.create_shader_module(code=SOLID_SHADER_SOURCE)
        return self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bgl]),
            vertex=self._create_vertex_state(shader),
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none, # 큐브의 모든 면을 렌더링
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
        """와이어프레임 렌더링 파이프라인을 생성합니다."""
        shader = self.device.create_shader_module(code=WIREFRAME_SHADER_SOURCE)
        return self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bgl]),
            vertex=self._create_vertex_state(shader),
            primitive={
                "topology": wgpu.PrimitiveTopology.line_list, # 라인 리스트로 설정
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": self.DEPTH_FORMAT,
                "depth_write_enabled": False, # 와이어프레임은 깊이 버퍼를 쓰지 않음
                "depth_compare": wgpu.CompareFunction.less_equal, # 깊이 테스트는 솔리드 위에 그려지도록
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

    def _write_uniforms(self, buffer: wgpu.GPUBuffer, width: int, height: int, model: np.ndarray) -> None:
        """장면 유니폼 데이터를 계산하여 버퍼에 씁니다."""
        aspect = width / max(height, 1)
        proj = perspective(np.radians(50.0), aspect, 0.1, 100.0)
        
        # 수정됨: 카메라를 뒤로(Z=8.0) 빼고, 중앙(0,0)을 바라보게 조정
        eye = np.array([0.0, 0.0, 8.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        view = look_at(eye, target, up)
        
        # ... (이하 light_dir, normal 계산 등 동일) ...

        light_dir = np.array([0.3, 0.7, 0.55], dtype=np.float32)

        normal3 = np.linalg.inv(model[:3, :3]).T
        normal = np.eye(4, dtype=np.float32)
        normal[:3, :3] = normal3

        model_cm = model.astype(np.float32).reshape(-1, order="F")
        view_proj = proj @ view
        view_proj_cm = view_proj.astype(np.float32).reshape(-1, order="F")
        normal_cm = normal.astype(np.float32).reshape(-1, order="F")

        data = np.concatenate(
            [
                model_cm,
                view_proj_cm,
                normal_cm,
                np.append(light_dir, 0.0).astype(np.float32),
                np.append(eye, 0.0).astype(np.float32),
            ]
        )
        
        pad_floats = (self.UNIFORM_BYTE_SIZE // 4) - data.size
        if pad_floats > 0:
            data = np.pad(data, (0, pad_floats), mode="constant")
            
        self.device.queue.write_buffer(buffer, 0, data.tobytes())


    def draw_frame(self, canvas: WgpuCanvas) -> None:
        """프레임을 렌더링하고 유니폼을 업데이트합니다."""
        current_texture = canvas.get_context().get_current_texture()
        tex_width, tex_height, _ = current_texture.size
        
        # ... (깊이 텍스처 생성 코드 동일) ...
        if self.depth_texture is None or self.depth_texture.size[0] != tex_width or self.depth_texture.size[1] != tex_height:
            self.depth_texture = self.device.create_texture(
                size=(tex_width, tex_height, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                format=self.DEPTH_FORMAT,
            )

        t = time.perf_counter() - self.start_time
        
        # 수정됨: 큐브 간격을 좁혀서(+-1.5) 화면 중앙에 오밀조밀하게 모음
        positions = [
            (-1.5, 1.5, 0.0), # Level 0 (Top-Left)
            ( 1.5, 1.5, 0.0), # Level 1 (Top-Right)
            ( 1.5,-1.5, 0.0), # Level 2 (Bottom-Right)
            (-1.5,-1.5, 0.0), # Level 3 (Bottom-Left)
        ]
        
        # 모델 행렬 업데이트 및 유니폼 버퍼 쓰기
        models = []
        for i, pos in enumerate(positions):
            # 시간에 따른 회전 적용
            model = translate(*pos) @ rotation_y(t * 0.5) @ rotation_x(t * 0.25)
            models.append(model)
            self._write_uniforms(self.uniform_buffers[i], tex_width, tex_height, model)

        view = current_texture.create_view()
        depth_view = self.depth_texture.create_view()
        
        # 로컬 변수로 color_attachment 정의 (이전 에러 수정 유지)
        color_attachment = {
            "view": view,
            "resolve_target": None,
            "load_op": wgpu.LoadOp.clear,
            "clear_value": (0.04, 0.04, 0.06, 1.0),
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
        render_pass = encoder.begin_render_pass(color_attachments=[color_attachment], depth_stencil_attachment=depth_attachment)
        
        # 렌더링 루프 (이전과 동일)
        for i, mesh_info in enumerate(self.meshes):
            bind_group = self.bind_groups[i]
            
            # --- 1. Solid 렌더링 ---
            render_pass.set_pipeline(self.solid_pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 999_999)
            render_pass.set_vertex_buffer(0, mesh_info["vertex_buffer"], 0, mesh_info["vertex_buffer"].size)
            render_pass.draw(mesh_info["vertex_data"].size // 6, 1, 0, 0) 

            # --- 2. Wireframe 렌더링 ---
            render_pass.set_pipeline(self.wireframe_pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 999_999)
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