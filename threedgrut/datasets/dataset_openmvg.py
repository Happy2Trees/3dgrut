"""OpenMVG dataset loader with train/test split support.

Parses OpenMVG-style JSONs (views/extrinsics split) and image splits
listed in `train.txt`/`test.txt` under a scene folder.

Expected folder structure under `path`:
  - images/                # image files referenced by views JSON
  - data_views.json        # OpenMVG sfm_data v0.3 (views only)
  - data_extrinsics.json   # OpenMVG sfm_data v0.3 (extrinsics only)
  - train.txt              # list of basenames (without extension) for train
  - test.txt               # list of basenames (without extension) for test

Notes
 - Intrinsics are not provided here (id_intrinsic==0 for all views). We default
   to an equirectangular camera model inferred from image size.
 - Extrinsics follow OpenMVG convention with rotation R (world->camera) and
   camera center C. We build W2C = [R|-RC], then invert to get C2W.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .utils import (
    create_camera_visualization,
    get_center_and_diag,
    equirectangular_camera_rays,
    get_worker_id,
)
from .camera_models import (
    ShutterType,
    EquirectangularCameraModelParameters,
)


class OpenMVGDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    """Dataset for OpenMVG-formatted scenes with explicit train/test splits."""

    def __init__(
        self,
        path: str,
        device: str = "cuda",
        split: str = "train",
        ray_jitter=None,
        camera_override_model: str | None = None,
    ) -> None:
        self.path = path
        self.device = device
        self.split = split
        self.ray_jitter = ray_jitter
        self.camera_override_model = camera_override_model

        # Warn if an unsupported override was provided (we only support ERP here)
        if self.camera_override_model is not None:
            override = str(self.camera_override_model).lower()
            if override not in ("equirectangular", "erp"):  # only ERP supported
                logger.warning(
                    f"OpenMVGDataset: camera_override_model='{self.camera_override_model}' is not supported; "
                    "falling back to Equirectangular"
                )

        # Worker-local GPU cache for intrinsics tensors
        self._worker_gpu_cache: Dict[str, dict] = {}

        # (Re)load scene
        self.reload()

    def reload(self):
        # CPU-side intrinsics cache keyed by intrinsic ID
        self.intrinsics: Dict[int, tuple] = {}

        # Load scene metadata and images list for the selected split
        self._load_views_and_extrinsics()
        self._load_split_lists()
        self._build_samples_for_split()
        self._load_camera_data()

        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()
        self.n_frames = len(self.image_paths)
        self._worker_gpu_cache.clear()

    # ------------- Data loading helpers -------------
    def _views_path(self) -> str:
        return os.path.join(self.path, "data_views.json")

    def _extrinsics_path(self) -> str:
        return os.path.join(self.path, "data_extrinsics.json")

    def _images_dir(self) -> str:
        return os.path.join(self.path, "images")

    def _split_file(self, split: str) -> str:
        return os.path.join(self.path, f"{split}.txt")

    def _load_views_and_extrinsics(self) -> None:
        # Parse JSONs
        with open(self._views_path(), "r") as f:
            dv = json.load(f)
        with open(self._extrinsics_path(), "r") as f:
            de = json.load(f)

        assert (
            dv.get("sfm_data_version") == "0.3"
        ), f"Unexpected sfm_data_version in views: {dv.get('sfm_data_version')}"
        assert (
            de.get("sfm_data_version") == "0.3"
        ), f"Unexpected sfm_data_version in extrinsics: {de.get('sfm_data_version')}"

        # Extract view records
        views = dv.get("views", [])
        self._views: List[dict] = views

        # Map pose-id -> (R, C)
        self._extrinsics: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for e in de.get("extrinsics", []):
            k = int(e["key"])
            R = np.array(e["value"]["rotation"], dtype=np.float64)
            C = np.array(e["value"]["center"], dtype=np.float64)
            self._extrinsics[k] = (R, C)

        # Basic consistency
        id_poses = [int(v["value"]["ptr_wrapper"]["data"]["id_pose"]) for v in views]
        missing = sorted(set(id_poses) - set(self._extrinsics.keys()))
        assert not missing, f"Missing extrinsics for pose IDs: {missing[:5]}"

    def _load_split_lists(self) -> None:
        def read_list(p: str) -> List[str]:
            with open(p, "r") as f:
                return [ln.strip() for ln in f if ln.strip()]

        self._train_list = read_list(self._split_file("train"))
        self._test_list = read_list(self._split_file("test"))

    def _build_samples_for_split(self) -> None:
        # Build filename->view record mapping
        view_by_filename: Dict[str, dict] = {}
        view_by_stem: Dict[str, str] = {}
        for v in self._views:
            data = v["value"]["ptr_wrapper"]["data"]
            filename = data["filename"]
            view_by_filename[filename] = data
            stem = os.path.splitext(os.path.basename(filename))[0]
            # If multiple entries map to the same stem, last one wins; warn once
            if stem in view_by_stem and view_by_stem[stem] != filename:
                logger.warning(
                    f"Multiple views share stem '{stem}': '{view_by_stem[stem]}' and '{filename}'. Using '{filename}'."
                )
            view_by_stem[stem] = filename

        # Choose split list
        name_list = self._train_list if self.split == "train" else self._test_list
        # Accept either bare stems or full filenames (with extension) in split files
        filenames: List[str] = []
        for name in name_list:
            if "." in name:  # looks like a filename with extension
                filenames.append(name)
            else:
                # Resolve by stem to the actual filename recorded in views
                resolved = view_by_stem.get(name)
                if resolved is None:
                    # Try common extensions as a fallback
                    candidates = [f"{name}.jpg", f"{name}.png", f"{name}.jpeg", f"{name}.JPG", f"{name}.PNG"]
                    resolved = next((c for c in candidates if c in view_by_filename), None)
                if resolved is None:
                    logger.warning(
                        f"Split item '{name}' not found in views; skipping."
                    )
                    continue
                filenames.append(resolved)

        # Build aligned lists of image paths, poses, and indices
        poses: List[np.ndarray] = []
        image_paths: List[str] = []
        cam_centers: List[np.ndarray] = []

        for fn in filenames:
            data = view_by_filename.get(fn)
            if data is None:
                logger.warning(f"Skipping {fn}: not found in data_views.json")
                continue

            pose_id = int(data["id_pose"])
            R, C = self._extrinsics[pose_id]
            # W2C = [R|-RC]
            W2C = np.eye(4, dtype=np.float64)
            W2C[:3, :3] = R
            W2C[:3, 3] = (-R @ C.reshape(3, 1)).flatten()
            C2W = np.linalg.inv(W2C).astype(np.float32)

            poses.append(C2W)
            cam_centers.append(C2W[:3, 3].copy())

            # Resolve image path robustly: if the JSON filename already contains a
            # subfolder (e.g., 'images/xxx.jpg' or 'some/subdir/xxx.jpg'), interpret it
            # as relative to scene root; otherwise, assume it's under images/.
            fn_norm = fn.replace("\\", "/")
            if "/" in fn_norm:
                image_path = os.path.join(self.path, fn_norm)
            else:
                image_path = os.path.join(self._images_dir(), fn)
            image_paths.append(image_path)

        assert (
            len(image_paths) > 0
        ), f"No images found for split '{self.split}'. Check split files and views."

        self.poses = np.stack(poses, axis=0)
        self.image_paths = np.array(image_paths, dtype=str)
        self.camera_centers = np.stack(cam_centers, axis=0)

        # Masks (optional)
        self.mask_paths = np.array(
            [os.path.splitext(p)[0] + "_mask.png" for p in self.image_paths], dtype=str
        )

    def _load_camera_data(self) -> None:
        """Prepare CPU-side camera intrinsics and per-pixel rays."""
        # We only support equirectangular for this dataset
        # Determine resolution from first image
        first_img = self.image_paths[0]
        with Image.open(first_img) as img:
            width, height = img.size

        def create_equirect_camera(w: int, h: int):
            params = EquirectangularCameraModelParameters(
                resolution=np.array([w, h], dtype=np.int64),
                shutter_type=ShutterType.GLOBAL,
            )
            rays_o_cam, rays_d_cam = equirectangular_camera_rays(
                w, h, self.ray_jitter
            )
            out_shape = (1, h, w, 3)
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32).reshape(out_shape),
                type(params).__name__,
            )

        # Single intrinsic ID (0) for all images
        self.intrinsics = {0: create_equirect_camera(width, height)}

    # ------------- Dataset protocol methods -------------
    @torch.no_grad()
    def compute_spatial_extents(self):
        camera_origins = torch.FloatTensor(self.poses[:, :, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        # Match ColmapDataset's extent logic for consistency
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1
        return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale

    def get_center(self):
        return self.center

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self):
        return self.cameras_extent

    def get_observer_points(self):
        return self.camera_centers

    def get_poses(self) -> np.ndarray:
        """Return camera-to-world (C2W) 4x4 poses."""
        return self.poses

    def get_intrinsics_idx(self, extr_idx: int):
        # Single intrinsic for all frames
        return 0

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> dict:
        # Load image as uint8 RGB to ensure 3 channels
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            image_data = np.asarray(img)
        assert image_data.dtype == np.uint8, "Image data must be of type uint8"

        output = {
            "data": torch.tensor(image_data).unsqueeze(0),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": 0,
        }

        # Optional mask
        mask_path = self.mask_paths[idx]
        if os.path.exists(mask_path):
            h, w = image_data.shape[:2]
            mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))).reshape(
                1, h, w, 1
            )
            output["mask"] = mask

        return output

    def _lazy_worker_intrinsics_cache(self):
        worker_id = get_worker_id()
        if worker_id not in self._worker_gpu_cache:
            worker_intrinsics = {}
            for intr_id, (
                params_dict,
                rays_ori,
                rays_dir,
                camera_name,
            ) in self.intrinsics.items():
                worker_intrinsics[intr_id] = (
                    params_dict,
                    rays_ori.to(self.device, non_blocking=True),
                    rays_dir.to(self.device, non_blocking=True),
                    camera_name,
                )
            self._worker_gpu_cache[worker_id] = worker_intrinsics
        return self._worker_gpu_cache[worker_id]

    def get_gpu_batch_with_intrinsics(self, batch):
        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        intr = batch["intr"][0].item()

        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        worker_intrinsics = self._lazy_worker_intrinsics_cache()
        camera_params_dict, rays_ori, rays_dir, camera_name = worker_intrinsics[intr]

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            f"intrinsics_{camera_name}": camera_params_dict,
        }

        if "mask" in batch:
            mask = batch["mask"][0].to(self.device, non_blocking=True) / 255.0
            mask = (mask > 0.5).to(torch.float32)
            sample["mask"] = mask

        return Batch(**sample)

    def create_dataset_camera_visualization(self):
        cam_list = []
        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # Camera convention rotation (match ColmapDataset visualization)
            camera_convention_rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            trans_mat_world_to_camera = (
                camera_convention_rot @ trans_mat_world_to_camera
            )

            # Load actual image to get dimensions
            image_data = np.asarray(Image.open(self.image_paths[i_cam]))
            h, w = image_data.shape[:2]
            assert image_data.dtype == np.uint8, "Image data must be of type uint8"
            rgb = image_data.reshape(h, w, 3) / np.float32(255.0)

            # For ERP we can report nominal FOVs
            fov_w = np.deg2rad(360.0)
            fov_h = np.deg2rad(180.0)

            cam_list.append(
                {
                    "ext_mat": trans_mat_world_to_camera,
                    "w": w,
                    "h": h,
                    "fov_w": fov_w,
                    "fov_h": fov_h,
                    "rgb_img": rgb,
                    "split": self.split,
                }
            )

        create_camera_visualization(cam_list)
