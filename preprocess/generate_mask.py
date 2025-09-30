#!/usr/bin/env python3
"""Generate fisheye-style masks for COLMAP datasets.

Given a dataset root (예: ``data/fish``) the script 탐색합니다 ``images`` 혹은
``images_*`` 폴더를 찾아 RGB 이미지마다 ``_mask.png`` 파일을 생성합니다. 기본
마스크는 특정 반경 안쪽(픽셀 단위)을 유지하고, 필요하면 알파 채널과 결합할 수
있습니다.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def discover_image_dirs(root: Path) -> list[Path]:
    """Return directories that likely hold RGBs for COLMAP datasets."""
    if root.is_dir() and root.name.startswith("images"):
        return [root]

    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("images")]
    return sorted(candidates)


def iter_image_files(directory: Path) -> Iterable[Path]:
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            yield entry


def build_mask(
    image: Image.Image,
    *,
    use_alpha: bool,
    alpha_threshold: int,
    radius: Optional[float],
    center: Optional[Tuple[float, float]],
) -> Image.Image:
    """Create a radial mask (optionally AND-ed with the alpha channel)."""

    width, height = image.size
    effective_radius = radius if radius is not None else min(width, height) / 2.0
    if effective_radius <= 0:
        raise ValueError("Radius must be positive.")

    cx, cy = center if center is not None else (width / 2.0, height / 2.0)

    yy, xx = np.ogrid[:height, :width]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask_array = np.where(dist_sq <= effective_radius**2, 255, 0).astype(np.uint8)

    if use_alpha and "A" in image.getbands():
        alpha = image.getchannel("A").convert("L")
        alpha_mask = alpha.point(lambda value: 255 if value >= alpha_threshold else 0)
        alpha_array = np.array(alpha_mask, dtype=np.uint8)
        mask_array = np.minimum(mask_array, alpha_array)

    return Image.fromarray(mask_array)


def generate_masks(
    directory: Path,
    *,
    mask_suffix: str,
    overwrite: bool,
    use_alpha: bool,
    alpha_threshold: int,
    radius: Optional[float],
    center: Optional[Tuple[float, float]],
) -> tuple[int, int, int]:
    image_files = [
        image_path
        for image_path in iter_image_files(directory)
        if not image_path.name.endswith(f"{mask_suffix}.png")
    ]

    total = len(image_files)
    created = 0
    skipped = 0

    for index, image_path in enumerate(image_files, start=1):
        sys.stdout.write(
            f"\r[{directory.name}] {index:4d}/{total:4d} -> {image_path.name}"
        )
        sys.stdout.flush()

        mask_name = f"{image_path.stem}{mask_suffix}.png"
        mask_path = image_path.parent / mask_name

        if mask_path.exists() and not overwrite:
            skipped += 1
            continue

        with Image.open(image_path) as image:
            mask = build_mask(
                image,
                use_alpha=use_alpha,
                alpha_threshold=alpha_threshold,
                radius=radius,
                center=center,
            )
            mask.save(mask_path)
        created += 1

    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return created, skipped, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fisheye-style _mask.png files for COLMAP datasets."
    )
    parser.add_argument("data_root", type=Path, help="데이터셋 루트 경로 (예: data/fish)")
    parser.add_argument(
        "--mask-suffix",
        default="_mask",
        help="확장자 앞에 붙일 접미사 (기본값: _mask)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 마스크를 덮어씁니다.",
    )
    parser.add_argument(
        "--use-alpha",
        action="store_true",
        help="원본 이미지의 알파 채널(있을 경우)을 함께 마스킹합니다.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=128,
        metavar="VALUE",
        help="알파 값이 VALUE 이상이면 유효 영역으로 간주합니다 (기본 128).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        metavar="PIXELS",
        help="fisheye 마스크 반경(px). 미지정 시 짧은 변의 절반을 사용합니다.",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        metavar=("CX", "CY"),
        help="마스크 중심(px). 기본값은 이미지 중심입니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Dataset path not found: {root}")

    image_dirs = discover_image_dirs(root)
    if not image_dirs:
        if any(root.glob("*" + ext) for ext in IMAGE_EXTENSIONS):
            image_dirs = [root]
        else:
            raise SystemExit(
                "Could not find an images folder. Expected directories named 'images' or 'images_*'."
            )

    center = tuple(args.center) if args.center is not None else None

    total_created = 0
    total_skipped = 0
    total_images = 0

    for directory in image_dirs:
        created, skipped, seen = generate_masks(
            directory,
            mask_suffix=args.mask_suffix,
            overwrite=args.overwrite,
            use_alpha=args.use_alpha,
            alpha_threshold=args.alpha_threshold,
            radius=args.radius,
            center=center,
        )
        print(
            f"Processed {seen:4d} files in {directory}: created {created}, skipped {skipped}."
        )
        total_created += created
        total_skipped += skipped
        total_images += seen

    print(
        f"Done. Images: {total_images}, masks created: {total_created}, skipped: {total_skipped}."
    )


if __name__ == "__main__":
    main()
