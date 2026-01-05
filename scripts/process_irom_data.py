#!/usr/bin/env python3
"""
Combined script to extract latents and create meta info in one step.

This script runs:
1. extract_latent.py - Convert raw IROM data to latent dataset
2. create_meta_info.py - Create meta info for the latent dataset

Usage:
    # For play data:
    python scripts/process_irom_data.py \
        --irom_data_path /path/to/raw/data \
        --output_path /path/to/latent/dataset \
        --dataset_type play

    # For demo data:
    python scripts/process_irom_data.py \
        --irom_data_path /path/to/raw/data \
        --output_path /path/to/latent/dataset \
        --dataset_type demo \
        --skip_start_frames 0
"""

import subprocess
import sys
from argparse import ArgumentParser


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ {description} completed successfully")


def main():
    parser = ArgumentParser(description='Process IROM data: extract latents and create meta info')

    # Required arguments
    parser.add_argument('--dataset_type', type=str, required=True, choices=['demo', 'play'],
                        help='Type of dataset: demo or play')
    parser.add_argument('--irom_data_path', type=str, required=True,
                        help='Path to IROM dataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for processed data')

    # Optional arguments for extract_latent
    parser.add_argument('--svd_path', type=str, default='stable-video-diffusion-img2vid',
                        help='Path to SVD model')
    parser.add_argument('--skip_start_frames', type=int, default=0,
                        help='Number of frames to skip at the start to remove idle frames (demo only)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: only process 5 samples')

    # Additional options
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip latent extraction step (only run create_meta_info)')
    parser.add_argument('--skip-meta', action='store_true',
                        help='Skip meta info creation step (only run extract_latent)')
    parser.add_argument('--distributed', action='store_true',
                        help='Use accelerate launch for distributed processing')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: skip metainfo and put all samples under train/ (no val/)')

    args = parser.parse_args()

    # If test mode is enabled, automatically skip meta info creation
    if args.test:
        args.skip_meta = True

    # Step 1: Extract latents
    if not args.skip_extract:
        if args.distributed:
            extract_cmd = [
                'accelerate', 'launch',
                'scripts/extract_latent.py',
            ]
        else:
            extract_cmd = [
                'python',
                'scripts/extract_latent.py',
            ]

        extract_cmd.extend([
            '--dataset_type', args.dataset_type,
            '--irom_data_path', args.irom_data_path,
            '--output_path', args.output_path,
            '--svd_path', args.svd_path,
            '--skip_start_frames', str(args.skip_start_frames),
        ])

        if args.debug:
            extract_cmd.append('--debug')

        if args.test:
            extract_cmd.append('--test')

        run_command(extract_cmd, "Step 1/2: Extracting latents")
    else:
        print("\n⏭️  Skipping latent extraction step")

    # Step 2: Create meta info
    if not args.skip_meta:
        meta_cmd = [
            'python', 'scripts/create_meta_info.py',
            '--droid_output_path', args.output_path,
        ]

        if args.debug:
            meta_cmd.append('--debug')

        run_command(meta_cmd, "Step 2/2: Creating meta info")
    else:
        print("\n⏭️  Skipping meta info creation step")

    print(f"\n{'='*80}")
    print("✓ All steps completed successfully!")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_path}")
    print(f"  - Videos: {args.output_path}/videos/")
    print(f"  - Latent videos: {args.output_path}/latent_videos/")
    print(f"  - Annotations: {args.output_path}/annotation/")
    print(f"  - Meta info: {args.output_path}/metainfo/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
