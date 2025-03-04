import os
import numpy as np
import shutil


def balanced_sampling(video_dir, target_frames=7):
    for video_folder in os.listdir(video_dir):
        # Skip hidden files and non-directories
        full_path = os.path.join(video_dir, video_folder)
        if not os.path.isdir(full_path) or video_folder.startswith('.'):
            continue

        frames = []
        # Get only image files (not subdirectories)
        for f in os.listdir(full_path):
            file_path = os.path.join(full_path, f)
            if os.path.isfile(file_path) and (f.lower().endswith(('.png', '.jpg', '.jpeg')) or '.' not in f):
                frames.append(f)

        if len(frames) == 0:
            print(f"No valid frames found in {full_path}")
            continue

        # Handle frame duplication if needed
        if len(frames) < target_frames:
            print(
                f"Duplicating {target_frames - len(frames)} frames in {full_path}")

            try:
                # Extract numeric part from filenames
                numbers = [int(f.split('.')[0])
                           for f in frames if f.split('.')[0].isdigit()]
                current_max = max(numbers) if numbers else 0
            except ValueError:
                print(
                    f"Non-numeric frame names in {full_path}, skipping duplication")
                continue

            # Duplicate random frames to reach target count
            for _ in range(target_frames - len(frames)):
                # Pick a random frame to duplicate
                src_frame = np.random.choice(frames)
                src_path = os.path.join(full_path, src_frame)

                # Generate new filename with sequential numbering
                new_num = current_max + 1
                current_max += 1
                new_name = f"{new_num:03d}"
                if '.' in src_frame:
                    ext = src_frame.split('.')[-1]
                    new_name += f".{ext}"

                # Copy the file
                dst_path = os.path.join(full_path, new_name)
                shutil.copy(src_path, dst_path)
                frames.append(new_name)

        # Now select exactly target_frames
        try:
            selected = np.random.choice(frames, target_frames, replace=False)
        except ValueError:
            print(f"Unexpected error in {full_path} with {len(frames)} frames")
            continue

        # Delete extra frames
        for frame in set(frames) - set(selected):
            os.remove(os.path.join(full_path, frame))


# Apply to real videos only
balanced_sampling('deepfake_dataset/LQ/train/fake/felc0')
