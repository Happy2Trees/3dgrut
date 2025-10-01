import math
import sys

import torch


class Cfg:
    class Render:
        particle_radiance_sph_degree = 0
        particle_kernel_degree = 2
        particle_kernel_min_response = 1e-4
        particle_kernel_min_alpha = 1e-5
        particle_kernel_max_alpha = 0.99
        min_transmittance = 1e-4
        enable_hitcounts = False

        class Splat:
            n_rolling_shutter_iterations = 1
            k_buffer_size = 4
            global_z_order = False
            ut_alpha = 1.0
            ut_beta = 2.0
            ut_kappa = 0.0
            ut_in_image_margin_factor = 0.0
            ut_require_all_sigma_points_valid = False
            rect_bounding = True
            tight_opacity_bounding = False
            tile_based_culling = True

        splat = Splat()

    render = Render()


def main():
    assert torch.cuda.is_available(), "CUDA is required for this test"

    # Build+load extension
    from threedgut_tracer.setup_3dgut import setup_3dgut

    setup_3dgut(Cfg())
    import lib3dgut_cc as gut

    W, H = 8, 4
    params = gut.fromEquirectangularCameraModelParameters(
        resolution=[W, H],
        shutter_type=gut.ShutterType.GLOBAL,
    )

    # Directions to test
    dirs = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # +Z -> center
            [1.0, 0.0, 0.0],  # +X -> right of center (0.75W)
            [-1.0, 0.0, 0.0],  # -X -> left of center (0.25W)
            [0.0, 1.0, 0.0],  # +Y -> top (v~0)
            [0.0, -1.0, 0.0],  # -Y -> bottom (v~H)
            [0.0, 0.0, -1.0],  # -Z -> seam (u~W-eps)
        ],
        dtype=torch.float32,
        device="cuda",
    )

    proj, valid = gut._test_project_points(params, [W, H], dirs)
    proj = proj.cpu().numpy()
    valid = valid.cpu().numpy()

    # All projections should be valid
    assert valid.sum() == len(dirs), f"invalid projections: {valid}"

    eps = 1e-2
    # +Z
    assert abs(proj[0, 0] - W * 0.5) < eps and abs(proj[0, 1] - H * 0.5) < eps
    # +X
    assert abs(proj[1, 0] - W * 0.75) < eps and abs(proj[1, 1] - H * 0.5) < eps
    # -X
    assert abs(proj[2, 0] - W * 0.25) < eps and abs(proj[2, 1] - H * 0.5) < eps
    # +Y (top)
    assert abs(proj[3, 1] - 0.0) < eps
    # -Y (bottom)
    assert abs(proj[4, 1] - H) < 1e-1  # allow small clamp epsilon
    # -Z (seam)
    assert proj[5, 0] <= W and proj[5, 0] >= W - 1e-1

    print("ERP projection test passed.")


if __name__ == "__main__":
    main()
