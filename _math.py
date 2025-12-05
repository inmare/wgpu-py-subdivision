# _math.py
import numpy as np
from math import cos, sin, tan

# 행렬 생성 함수 (전부 numpy.ndarray 반환)
def perspective(fovy_radians: float, aspect: float, near: float, far: float) -> np.ndarray:
    """원근 투영 행렬을 생성합니다."""
    f = 1.0 / tan(fovy_radians / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """뷰(View) 행렬을 생성합니다."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def rotation_y(angle: float) -> np.ndarray:
    """Y축 회전 행렬을 생성합니다."""
    c, s = cos(angle), sin(angle)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_x(angle: float) -> np.ndarray:
    """X축 회전 행렬을 생성합니다."""
    c, s = cos(angle), sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def translate(x: float, y: float, z: float) -> np.ndarray:
    """이동(Translation) 행렬을 생성합니다."""
    m = np.eye(4, dtype=np.float32)
    # row-major 행렬에 translation을 마지막 열에 기록
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m