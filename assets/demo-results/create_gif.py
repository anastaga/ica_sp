#!/usr/bin/env python

import os
import glob
from PIL import Image

def create_gif(image_folder, output_path, duration=100, speedup=1.0, max_frames=None):
 
    image_folder = os.path.abspath(image_folder)
    print(f"Searching for images in folder: {image_folder}")

    # Get a sorted list of image file paths (supports png and jpg for flexibility)
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    if not image_paths:
        raise ValueError(f"===>No images found in folder: {image_folder}. Ensure PNG or JPG images exist!")

    print(f"Found {len(image_paths)} images.")

    # Limit the number of frames if max_frames is specified
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        print(f"Using the first {len(image_paths)} frames.")

    # Adjust frame duration based on the speed-up factor
    adjusted_duration = max(1, int(duration / speedup))  # Minimum duration is 1ms

    # Open the first image and use it as the base for the GIF
    with Image.open(image_paths[0]) as base_image:
        frames = []  # To store frames temporarily
        for img_path in image_paths[1:]:
            with Image.open(img_path) as img:
                frames.append(img.convert("RGB"))  # Ensure RGB format

        # Save the GIF
        base_image.save(
            output_path,
            save_all=True,
            append_images=frames,
            duration=adjusted_duration,
            loop=0  # Infinite loop
        )

    print(f"GIF created successfully at: {output_path}")
    print(f"Speed-up factor: {speedup}, Frame duration: {adjusted_duration}ms")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert an image sequence to a GIF with frame limiting options.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing the image sequence.")
    parser.add_argument("output_path", type=str, help="Path to save the output GIF.")
    parser.add_argument("--duration", type=int, default=100, help="Base frame duration in milliseconds (default: 100).")
    parser.add_argument("--speedup", type=float, default=1.0, help="Speed-up factor for the GIF (default: 1.0, no speed-up).")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to include in the GIF (default: None, include all).")
    args = parser.parse_args()

    try:
        create_gif(args.image_folder, args.output_path, args.duration, args.speedup, args.max_frames)
    except Exception as e:
        print(f"Error: {e}")

