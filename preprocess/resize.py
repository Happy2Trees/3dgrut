from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image


try:
    Resampling = Image.Resampling
except AttributeError:  # Pillow < 9.1
    class Resampling:
        LANCZOS = Image.LANCZOS
        NEAREST = Image.NEAREST


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize COLMAP image sets by an integer factor.")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset root directory (e.g. data/fish)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        required=True,
        help="Downsampling factor (images will be resized to width/factor, height/factor).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("images"),
        help="Relative or absolute path to the source image directory (default: images).",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <dataset_root>/<source-dir>_<factor> when factor != 1, otherwise <dataset_root>/<source-dir>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist at the destination.",
    )
    return parser.parse_args(argv)


def resolve_rooted_path(root: Path, provided: Path) -> Path:
    if provided.is_absolute():
        return provided
    return (root / provided).resolve()


def pick_dest_dir(root: Path, src_dir: Path, factor: int, explicit_dest: Path | None) -> Path:
    if explicit_dest is not None:
        return resolve_rooted_path(root, explicit_dest)

    suffix = src_dir.name if factor == 1 else f"{src_dir.name}_{factor}"
    return (root / suffix).resolve()


def make_resized_copy(src_path: Path, dest_path: Path, factor: int, resample: Resampling) -> None:
    with Image.open(src_path) as img:
        width, height = img.size
        if width % factor or height % factor:
            raise ValueError(
                f"Image '{src_path}' has resolution {width}x{height} which is not divisible by factor {factor}."
            )
        new_size = (width // factor, height // factor)
        resized = img.resize(new_size, resample=resample)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dest_path)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    if args.factor <= 0:
        raise SystemExit("Downsampling factor must be a positive integer.")

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root '{dataset_root}' does not exist or is not a directory.")

    source_dir = resolve_rooted_path(dataset_root, args.source_dir)
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory '{source_dir}' does not exist or is not a directory.")

    dest_dir = pick_dest_dir(dataset_root, source_dir, args.factor, args.dest_dir)

    if dest_dir == source_dir and not (args.overwrite or args.skip_existing):
        raise SystemExit(
            "Destination directory matches source directory. Use --overwrite for in-place resizing or provide --dest-dir."
        )

    files = sorted(
        [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    if not files:
        raise SystemExit(f"No supported image files found in '{source_dir}'.")

    processed = 0
    skipped = 0
    for src_path in files:
        relative_path = src_path.relative_to(source_dir)
        dest_path = dest_dir / relative_path

        if dest_path.exists() and not args.overwrite:
            if args.skip_existing:
                skipped += 1
                continue
            raise SystemExit(
                f"Destination file '{dest_path}' already exists. Use --overwrite or --skip-existing to proceed."
            )

        resample_method = Resampling.NEAREST if src_path.stem.endswith("_mask") else Resampling.LANCZOS

        make_resized_copy(src_path, dest_path, args.factor, resample_method)
        processed += 1
        print(f"Resized {src_path} -> {dest_path}")

    print(f"Finished. Resized {processed} file(s).", end="")
    if skipped:
        print(f" Skipped {skipped} existing file(s).")
    else:
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
