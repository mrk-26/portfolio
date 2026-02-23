import os
import cv2

def extract_kick_frames(video_path, output_dir, start_sec=0, end_sec=None, step=5):
    """
    Extracts specific frames from a video file and saves them as images.
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        start_sec (int): Start time in seconds from which to begin extracting frames.
        end_sec (int, optional): End time in seconds until which to extract frames. Defaults to None, meaning until the end of the video.
        step (int): Interval in seconds between extracted frames.
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_frame = int(start_sec * fps)
    end_frame = int((duration - end_sec) * fps)

    print(f"\nProcessing {os.path.basename(video_path)}")
    print(f"FPS: {fps}, Duration: {duration:.2f}s, Total Frames: {total_frames}")
    print(f"Extracting frames from {start_frame} to {end_frame}, every {step} frames")

    i = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or i >= end_frame:
            break

        if i >= start_frame and (i - start_frame) % step == 0:
            filename = f"frame_{saved:04d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1

        i += 1

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")

if __name__ == "__main__":
    raw_folder = "raw_videos"
    output_base = "kick_frames"

    os.makedirs(output_base, exist_ok=True)


    for video_file in os.listdir(raw_folder):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(raw_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(output_base, video_name)
            extract_kick_frames(video_path, output_folder, start_sec=5, end_sec=3, step=5)
        else:
            print(f"Skipping non-video file: {video_file}") 


    
