#!/usr/bin/env python3
"""
Batch Image Restoration
Process all images in a folder using the multi-agent pipeline.
"""
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

from image_restoration import restore_image


def get_image_files(folder: str) -> list:
    """Get all image files from a folder."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    files = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(exts):
            files.append(os.path.join(folder, f))
    return files


async def process_batch(input_folder: str, output_folder: str = "restored/batch"):
    os.makedirs(output_folder, exist_ok=True)

    image_files = get_image_files(input_folder)

    if not image_files:
        print(f"❌ No images found in: {input_folder}")
        print(f"   Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP")
        return

    print("=" * 60)
    print(f"📁 BATCH PROCESSING")
    print(f"   Input folder:  {os.path.abspath(input_folder)}")
    print(f"   Output folder: {os.path.abspath(output_folder)}")
    print(f"   Images found:  {len(image_files)}")
    print("=" * 60)
    print()

    success = 0
    failed = 0

    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        print(f"[{i}/{len(image_files)}] 🖼️  {filename}")

        try:
            restored_path = await restore_image(img_path)

            # Copy to batch output with timestamp
            base = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            output_name = f"{base}_restored{ext}"
            output_path = os.path.join(output_folder, output_name)

            import shutil
            shutil.copy(restored_path, output_path)

            print(f"      ✅ Saved: {output_path}")
            success += 1

        except Exception as e:
            print(f"      ❌ Failed: {e}")
            failed += 1

        print()

    print("=" * 60)
    print(f"📊 BATCH COMPLETE")
    print(f"   Success: {success}/{len(image_files)}")
    print(f"   Failed:  {failed}/{len(image_files)}")
    print(f"   Output:  {os.path.abspath(output_folder)}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_restore.py <input_folder> [output_folder]")
        print("Example: python batch_restore.py ./my_photos/ ./restored_photos/")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "restored/batch"

    if not os.path.isdir(input_folder):
        print(f"❌ Not a directory: {input_folder}")
        sys.exit(1)

    asyncio.run(process_batch(input_folder, output_folder))
