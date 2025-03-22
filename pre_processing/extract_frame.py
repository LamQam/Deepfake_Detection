import cv2
import os
import shutil


def extract_avi_frames(video_path, output_dir, frame_interval=10):
    """
    Extract frames from AVI video and delete original on success
    :param video_path: Path to input video
    :param output_dir: Directory to save frames
    :param frame_interval: Extract every nth frame
    :return: Number of frames extracted, or -1 on error
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return -1

        frame_count = 0
        saved_count = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save at specified interval
            if frame_count % frame_interval == 0:
                # Resize for Xception model
                resized = cv2.resize(frame, (299, 299))

                # Save frame
                output_path = os.path.join(
                    output_dir,
                    f"{saved_count:03d}.jpeg"
                )
                cv2.imwrite(output_path, resized)
                saved_count += 1

            frame_count += 1

        cap.release()

        if saved_count > 0:
            print(f"Extracted {saved_count} frames from {video_path}")
            return saved_count
        else:
            print(f"No frames extracted from {video_path}")
            return -1

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return -1


# Batch processing for all AVI files
input_dir = "deepfake_dataset/LQ/test/fake/mwbt0"
output_base = "deepfake_dataset/LQ/test/fake/mwbt0"


for video_file in os.listdir(input_dir):
    if video_file.endswith(".avi"):
        # Create sentence-specific folder from filename
        sentence_id = video_file.split("-")[0]
        video_path = os.path.join(input_dir, video_file)
        output_dir = os.path.join(output_base, sentence_id)

        # Process and delete original if successful
        saved_count = extract_avi_frames(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=15
        )

        # Only delete if frames were successfully extracted
        if saved_count > 0:
            try:
                os.remove(video_path)
                print(f"Deleted original video: {video_path}")
            except Exception as e:
                print(f"Failed to delete {video_path}: {str(e)}")
