import math
from typing import List


def compute_projection_matrix(fov: float, ratio: float, near: float, far: float) -> List[List[float]]:
    projection_matrix = [[0.0] * 4 for _ in range(4)]

    # Compute the parameters for the projection matrix
    top = near * math.tan(math.radians(fov) / 2)
    right = top * ratio

    # Fill in the projection matrix
    projection_matrix[0][0] = near / right
    projection_matrix[1][1] = near / top
    projection_matrix[2][2] = -(far + near) / (far - near)
    projection_matrix[2][3] = -(2 * far * near) / (far - near)
    projection_matrix[3][2] = -1.0

    return projection_matrix


if __name__ == "__main__":
    fov = 60.0  # Field-of-view in degrees
    ratio = 16 / 9  # Window size ratio (width / height)
    near = 0.1  # Distance of the near plane
    far = 100.0  # Distance of the far plane
    projection_matrix = compute_projection_matrix(fov, ratio, near, far)
    matrix_str = '\n'.join(', '.join(str(element)
                           for element in row) for row in projection_matrix)
    print(matrix_str)
