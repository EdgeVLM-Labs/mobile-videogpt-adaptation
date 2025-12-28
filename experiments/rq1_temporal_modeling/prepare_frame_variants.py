"""
Prepare frame variants for temporal modeling experiments.
"""
import argparse
import json
from pathlib import Path
import shutil


def create_frame_variant_config(input_json, output_dir, frame_counts, fps_ratios):
    """Create test variants with different frame counts and FPS."""
    with open(input_json, 'r') as f:
        data = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For each combination of frame count and FPS ratio
    for fps_ratio in fps_ratios:
        for frame_count in frame_counts:
            # Create variant config
            variant_name = f"fps{fps_ratio}_frames{frame_count}.json"
            output_file = output_dir / variant_name

            # Add frame configuration to each video entry
            variant_data = []
            for item in data:
                variant_item = item.copy()
                variant_item['num_frames'] = frame_count
                variant_item['fps_ratio'] = fps_ratio
                variant_data.append(variant_item)

            # Save variant
            with open(output_file, 'w') as f:
                json.dump(variant_data, f, indent=2)

            print(f"Created variant: {variant_name} ({len(variant_data)} videos)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input test JSON')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--frame_counts', nargs='+', type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument('--fps_ratios', nargs='+', type=float, default=[1.0, 0.5, 0.25])
    args = parser.parse_args()

    create_frame_variant_config(
        args.input,
        args.output_dir,
        args.frame_counts,
        args.fps_ratios
    )

    print(f"\nTotal variants created: {len(args.frame_counts) * len(args.fps_ratios)}")


if __name__ == '__main__':
    main()
