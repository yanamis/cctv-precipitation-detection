import os

from moviepy.editor import VideoFileClip
import imageio


def extract_frames(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Pobieranie liczby klatek
    clip = VideoFileClip(input_file)
    frame_count = int(clip.fps * clip.duration)
    clip.reader.close()

    frame_interval = 3
    num_frames = 200

    # Określanie indeksów klatek do wyodrębnienia
    selected_frames_indices = range(0, min(frame_count, num_frames * frame_interval), frame_interval)

    # Wyodrębnianie klatek jako obrazów
    for i in selected_frames_indices:
        frame = clip.get_frame(i / clip.fps)
        frame_path = os.path.join(output_folder, 'klatka_{:04d}.png'.format(i))
        imageio.imsave(frame_path, frame)


def process_videos(files_to_process):
    for video_file, output_folder in files_to_process:
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            extract_frames(video_file, output_folder)


files_to_process = [
    (
        "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\Securicam 2MP 1080P POE IP Bullet Camera "
        "Footage.mp4",
        "C:\\Users\\yanam\\opady_dataset\\Securicam 2MP 1080P POE IP Bullet Camera Footage"
    )
]

process_videos(files_to_process)
