import wgpu
from wgpu import gpu
import wgpu.backends.auto  # noqa: F401 - auto-select backend
from wgpu.gui.auto import WgpuCanvas, run


SHADER_SOURCE = """
struct VertexOutput {
    @builtin(position) pos : vec4<f32>,
    @location(0) color : vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var out : VertexOutput;
    if (vertex_index == 0u) {
        out.pos = vec4<f32>(0.0, 0.6, 0.0, 1.0);
        out.color = vec3<f32>(1.0, 0.3, 0.3);
    } else if (vertex_index == 1u) {
        out.pos = vec4<f32>(-0.5, -0.4, 0.0, 1.0);
        out.color = vec3<f32>(0.3, 1.0, 0.4);
    } else {
        out.pos = vec4<f32>(0.5, -0.4, 0.0, 1.0);
        out.color = vec3<f32>(0.3, 0.5, 1.0);
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"""


def main() -> None:
    """Render a basic triangle in a desktop window using wgpu-py."""
    canvas = WgpuCanvas(title="wgpu-py triangle")
    adapter = gpu.request_adapter(canvas=canvas, power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("적절한 GPU 어댑터를 찾지 못했습니다.")

    device = adapter.request_device()

    context = canvas.get_context()
    texture_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=texture_format)

    shader = device.create_shader_module(code=SHADER_SOURCE)

    pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[]),
        vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=None,
        multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [{"format": texture_format}],
        },
    )

    def draw_frame() -> None:
        current_texture = context.get_current_texture()
        view = current_texture.create_view()
        color_attachment = {
            "view": view,
            "resolve_target": None,
            "load_op": wgpu.LoadOp.clear,
            "clear_value": (0.07, 0.07, 0.1, 1.0),
            "store_op": wgpu.StoreOp.store,
        }
        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(color_attachments=[color_attachment])
        render_pass.set_pipeline(pipeline)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        device.queue.submit([encoder.finish()])
        current_texture.present()

    canvas.draw_frame = draw_frame
    run()


if __name__ == "__main__":
    main()

