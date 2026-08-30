import math
import random


def _coordinate_pairs(points):
    assert len(points) == 4
    pairs = []
    for point in points:
        assert len(point) == 2
        pairs.append((point[0], point[1]))
    return pairs


def _solve_linear_system(matrix, vector):
    size = len(vector)
    augmented = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        assert abs(augmented[pivot_row][column]) > 1e-12
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(size)]


def _perspective_coeffs(startpoints, endpoints):
    matrix = []
    for endpoint, startpoint in zip(endpoints, startpoints):
        x_coord, y_coord = endpoint
        target_x, target_y = startpoint
        matrix.append([x_coord, y_coord, 1, 0, 0, 0, -target_x * x_coord, -target_x * y_coord])
        matrix.append([0, 0, 0, x_coord, y_coord, 1, -target_y * x_coord, -target_y * y_coord])
    target_vector = [coordinate for point in startpoints for coordinate in point]
    return _solve_linear_system(matrix, target_vector)


def test_returns_canonical_source_corners_and_four_endpoints():
    random.seed(7)
    startpoints, endpoints = get_params(12, 10, 0.5)

    assert _coordinate_pairs(startpoints) == [(0, 0), (11, 0), (11, 9), (0, 9)]
    _coordinate_pairs(endpoints)


def test_endpoint_order_preserves_corner_identity():
    random.seed(19)
    width, height, distortion_scale = 20, 14, 0.5
    _startpoints, endpoints = get_params(width, height, distortion_scale)
    topleft, topright, bottomright, bottomleft = _coordinate_pairs(endpoints)
    horizontal_offset = int(distortion_scale * int(width / 2))
    vertical_offset = int(distortion_scale * int(height / 2))

    assert 0 <= topleft[0] <= horizontal_offset
    assert 0 <= topleft[1] <= vertical_offset
    assert width - horizontal_offset - 1 <= topright[0] <= width - 1
    assert 0 <= topright[1] <= vertical_offset
    assert width - horizontal_offset - 1 <= bottomright[0] <= width - 1
    assert height - vertical_offset - 1 <= bottomright[1] <= height - 1
    assert 0 <= bottomleft[0] <= horizontal_offset
    assert height - vertical_offset - 1 <= bottomleft[1] <= height - 1


def test_zero_distortion_keeps_endpoints_at_image_corners():
    random.seed(3)
    startpoints, endpoints = get_params(9, 7, 0)

    expected = [(0, 0), (8, 0), (8, 6), (0, 6)]
    assert _coordinate_pairs(startpoints) == expected
    assert _coordinate_pairs(endpoints) == expected


def test_points_define_valid_perspective_correspondences():
    random.seed(19)
    startpoints, endpoints = get_params(20, 14, 0.5)
    startpoint_pairs = _coordinate_pairs(startpoints)
    endpoint_pairs = _coordinate_pairs(endpoints)
    a, b, c, d, e, f, g, h = _perspective_coeffs(startpoint_pairs, endpoint_pairs)

    assert all(math.isfinite(value) for value in (a, b, c, d, e, f, g, h))
    for endpoint, startpoint in zip(endpoint_pairs, startpoint_pairs):
        x_coord, y_coord = endpoint
        denominator = g * x_coord + h * y_coord + 1
        assert abs(denominator) > 1e-8
        mapped_x = (a * x_coord + b * y_coord + c) / denominator
        mapped_y = (d * x_coord + e * y_coord + f) / denominator
        assert math.isclose(mapped_x, startpoint[0], abs_tol=1e-4)
        assert math.isclose(mapped_y, startpoint[1], abs_tol=1e-4)


TEST_CASES = [
    test_returns_canonical_source_corners_and_four_endpoints,
    test_endpoint_order_preserves_corner_identity,
    test_zero_distortion_keeps_endpoints_at_image_corners,
    test_points_define_valid_perspective_correspondences,
]
