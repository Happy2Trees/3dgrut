import numpy as np

from threedgrut.datasets.utils import equirectangular_camera_rays


def test_erp_ray_directions_basic():
    W, H = 8, 4
    rays_o, rays_d = equirectangular_camera_rays(W, H)
    assert rays_o.shape == (W * H, 3)
    assert rays_d.shape == (W * H, 3)

    # All directions should be unit-length
    norms = np.linalg.norm(rays_d, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Reshape to (H, W, 3) for indexing by row/col
    rays = rays_d.reshape(H, W, 3)

    # Top row should have positive Y close to +1.0
    top_y = rays[0, :, 1]
    assert np.all(top_y > 0.7)

    # Bottom row should have negative Y close to -1.0
    bottom_y = rays[H - 1, :, 1]
    assert np.all(bottom_y < -0.7)

    # Leftmost and rightmost columns should be near seam (negative Z)
    left_z = rays[:, 0, 2]
    right_z = rays[:, W - 1, 2]
    assert np.all(left_z < 0.0)
    assert np.all(right_z < 0.0)

    # Right half columns should have positive X average, left half negative X average
    left_mean_x = rays[:, : W // 2, 0].mean()
    right_mean_x = rays[:, W // 2 :, 0].mean()
    assert left_mean_x < 0.0 and right_mean_x > 0.0

