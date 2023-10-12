import os
from moviepy.editor import VideoFileClip
import imageio


def extract_frames(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Pobieranie liczby klatek
    clip = VideoFileClip(input_file)
    frame_count = int(clip.fps * clip.duration)
    clip.reader.close()

    # Wyodrębnianie klatket jako obrazów
    for i in range(frame_count):
        frame = clip.get_frame(i/clip.fps)
        frame_path = os.path.join(output_folder, 'klatka_{:04d}.png'.format(i))
        imageio.imsave(frame_path, frame)


def process_videos(files_to_process):
    for video_file, output_folder in files_to_process:
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            extract_frames(video_file, output_folder)


files_to_process = [
    (
        "C:\\Users\\yanam\\opady_dataset\\folder z filmikami\\rain cctv night mood by saleem security systems.mp4",
        "C:\\Users\\yanam\\opady_dataset\\rain cctv night mood by saleem security systems_000"
    ),
    (
        "C:\\Users\\yanam\\opady_dataset\\folder z filmikami\\Blink Security Camera Shows What Rain Looks Like In Night Vision.mp4",
        "C:\\Users\\yanam\\opady_dataset\\Blink Security Camera Shows What Rain Looks Like In Night Vision_mp4_000"
    ),
    (
        "C:\\Users\\yanam\\opady_dataset\\folder z filmikami\\Heavy Rain ~ Caught on security camera, Kendal. #Shorts.mp4",
        "C:\\Users\\yanam\\opady_dataset\\Heavy Rain ~ Caught on security camera, Kendal. #Shorts_000"
    ),
    (
        "C:\\Users\\yanam\\opady_dataset\\folder z filmikami\\Heavy rain caught on my security Camera.mp4",
        "C:\\Users\\yanam\\opady_dataset\\Heavy rain caught on my security Camera_000"
    ),
    (
        "C:\\Users\\yanam\\opady_dataset\\folder z filmikami\\Crazy rain in the morning- Reolink  security  Camera August 28, 2020.mp4",
        "C:\\Users\\yanam\\opady_dataset\\Crazy rain in the morning- Reolink  security  Camera August 28, 2020_000"
    )
]

process_videos(files_to_process)
