import math
from typing import List
from matrix import create_zero_matrix

# easy and best - https://www.youtube.com/watch?v=EqNcqBdrNyI

# mathematical derivation and explanations
# https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix.html
# https://ogldev.org/www/tutorial12/tutorial12.html
# http://www.songho.ca/opengl/gl_projectionmatrix.html
# https://heinleinsgame.tistory.com/11


def compute_projection_matrix(fov: float, ratio: float, near: float, far: float) -> List[List[float]]:
    # by definition  projection matrix is 4x4 matrix
    projection_matrix = create_zero_matrix(4, 4)

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


def perspective_division(matrix):
    result = create_zero_matrix(4, 4)

    for i in range(4):
        for j in range(4):
            if matrix[i][3] != 0:
                result[i][j] = matrix[i][j] / matrix[i][3]
            else:
                result[i][j] = matrix[i][j]

    return result


def two(fov, ratio, near, far):
    fov_radian = math.radians(fov)
    result = create_zero_matrix(4, 4)
    result[0][0] = ratio * (1 / math.tan(fov_radian / 2))
    result[1][1] = 1 / math.tan(fov_radian / 2)
    result[2][2] = far / (far - near)
    result[2][3] = (-far * near) / (far - near)
    result[3][2] = 1.0

    # Apply perspective division on the resulting matrix
    result = perspective_division(result)

    return result


if __name__ == "__main__":
    fov = 45.0  # Field-of-view in degrees
    ratio = 4 / 3  # Window size ratio (width / height)
    near = 1.0  # Distance of the near plane
    far = 50.0  # Distance of the far plane

    projection_matrix = compute_projection_matrix(fov, ratio, near, far)
    two_matrix = two(fov, ratio, near, far)
    matrix_str = '\n'.join(', '.join(str(element)
                           for element in row) for row in projection_matrix)
    two_str = '\n'.join(', '.join(str(element)
                                  for element in row) for row in two_matrix)
    # print(matrix_str)
    print(matrix_str)
    print(two_str)
