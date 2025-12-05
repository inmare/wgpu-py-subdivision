# cube.py

import wgpu
from wgpu import gpu
import wgpu.backends.auto  # noqa: F401
from wgpu.gui.auto import WgpuCanvas, run

# 변경 전: from ._renderer import CubeRenderer
# 변경 후:
from _renderer import CubeRenderer # (같은 디렉토리에 있다면)


def main() -> None:
    canvas = WgpuCanvas(title="wgpu-py Cubes (Phong + Wireframe)")
    adapter = gpu.request_adapter(canvas=canvas, power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("적절한 GPU 어댑터를 찾지 못했습니다.")

    device = adapter.request_device()

    context = canvas.get_context()
    texture_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=texture_format)
    
    # 렌더링 로직을 캡슐화한 CubeRenderer 인스턴스 생성
    renderer = CubeRenderer(device, texture_format)

    # 캔버스의 draw_frame 콜백 함수를 렌더러의 메서드로 설정
    def draw_frame_callback():
        renderer.draw_frame(canvas)
        
    canvas.draw_frame = draw_frame_callback
    
    print("wgpu-py 렌더링 루프를 시작합니다. 창을 닫으면 종료됩니다.")
    run()


if __name__ == "__main__":
    # 이 스크립트를 독립적으로 실행할 때, 모듈 구조를 고려하여 임포트 오류를 피하기 위해
    # 적절한 환경 설정이나 모듈 임포트 방식을 사용해야 합니다.
    # 예: `python -m main_app` 또는 경로 설정
    main()