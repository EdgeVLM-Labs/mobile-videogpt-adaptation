"""
Generate degraded videos using vidaug and ffmpeg.
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import vidaug
try:
    import vidaug.augmentors as va
except ImportError:
    print("Error: vidaug not found. Install from: pip install git+https://github.com/okankop/vidaug")
    sys.exit(1)


def apply_vidaug_degradation(video_path, output_path, degradation_type, level):
    """Apply vidaug degradations."""
    augmentors = {
        'blur': va.GaussianBlur(sigma=level),
        'brightness_add': va.Add(value=int(level)),
        'brightness_mult': va.Multiply(value=level),
        'noise_salt': va.Salt(ratio=int(level)),
        'noise_pepper': va.Pepper(ratio=int(level))
    }

    if degradation_type not in augmentors:
        print(f"Unknown degradation type: {degradation_type}")
        return False

    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return False

    # Apply augmentation
    aug = augmentors[degradation_type]
    degraded_frames = aug(frames)

    # Save video
    height, width = degraded_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for frame in degraded_frames:
        out.write(frame)
    out.release()

    return True


def apply_compression(video_path, output_path, crf):
    """Apply compression using ffmpeg."""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-y',  # Overwrite
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--degradation_types', nargs='+',
                       choices=['blur', 'brightness', 'noise', 'compression'])
    args = parser.parse_args()

    # Load test data
    with open(args.input_json) as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define degradation configurations
    degradations = {
        'blur': [(1.5, 'sigma'), (2.5, 'sigma')],
        'brightness': [(-30, 'add'), (30, 'add')],
        'noise': [(100, 'salt')],
        'compression': [(35, 'crf')]
    }

    # Process each degradation type
    for deg_type in args.degradation_types:
        if deg_type not in degradations:
            continue

        for level, param in degradations[deg_type]:
            deg_name = f"{deg_type}_{level}"
            print(f"Creating {deg_name} degradation...")

            # Create degraded dataset
            degraded_data = []
            for item in data:
                video_path = item.get('video')
                if not video_path or not Path(video_path).exists():
                    continue

                # Generate output filename
                output_filename = f"{deg_name}_{Path(video_path).name}"
                output_path = output_dir / output_filename

                # Apply degradation
                success = False
                if deg_type == 'compression':
                    success = apply_compression(str(video_path), str(output_path), level)
                else:
                    vidaug_type = f"{deg_type}_{param}"
                    success = apply_vidaug_degradation(
                        str(video_path), str(output_path), vidaug_type, level
                    )

                if success:
                    degraded_item = item.copy()
                    degraded_item['video'] = str(output_path)
                    degraded_item['degradation'] = deg_name
                    degraded_data.append(degraded_item)

            # Save degraded dataset JSON
            output_json = output_dir / f"{deg_name}_test.json"
            with open(output_json, 'w') as f:
                json.dump(degraded_data, f, indent=2)

            print(f"  Created {len(degraded_data)} degraded videos -> {output_json}")

    print("\nDegradation complete!")


if __name__ == '__main__':
    main()
