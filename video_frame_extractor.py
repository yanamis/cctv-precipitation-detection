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
    for i in range(200, frame_count):
        frame = clip.get_frame(i/clip.fps)
        frame_path = os.path.join(output_folder, 'klatka_{:04d}.png'.format(i))
        imageio.imsave(frame_path, frame)


def process_videos(files_to_process):
    for video_file, output_folder in files_to_process:
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            extract_frames(video_file, output_folder)


files_to_process = [
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\Rain on my security camera 11-12-22 5_30 pm..mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Rain on my security camera 11-12-22 5_30 pm."
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\Heavy Snowfall, big snowflakes ｜ Free Footage for MOAH Members.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Heavy Snowfall, big snowflakes ｜ Free Footage for MOAH Members"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\GIANT snowflakes falling.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\GIANT snowflakes falling"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\Giant snowflakes fall in the front yard on 27-Mar-11.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Giant snowflakes fall in the front yard on 27-Mar-11"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\My cctv camera shows night footage after rainfall during daytime.why？.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\My cctv camera shows night footage after rainfall during daytime"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\rain night cctv by saleem security systems.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\rain night cctv by saleem security systems"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\cctv night mood by saleem security systems.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\cctv night mood by saleem security systems"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\Hikvision DS-2CD2042WD-I 4MP POE IP Camera Footage - www.cctvtek.co.uk.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Hikvision DS-2CD2042WD-I 4MP POE IP Camera Footage"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\Securicam 2MP 1080P POE IP Bullet Camera Footage.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Securicam 2MP 1080P POE IP Bullet Camera Footage"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\Securicam 1000TVL Varifocal Bullet Camera.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Securicam 1000TVL Varifocal Bullet Camera"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\Securicam AHD 720P Varifocal Dome Camera.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Securicam AHD 720P Varifocal Dome Camera"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\rain cctv night mood by saleem security systems.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\rain cctv night mood by saleem security systems_000"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\Blink Security Camera Shows What Rain Looks Like In Night Vision.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Blink Security Camera Shows What Rain Looks Like In Night Vision_mp4_000"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\Heavy Rain ~ Caught on security camera, Kendal. #Shorts.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Heavy Rain ~ Caught on security camera, Kendal. #Shorts_000"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\Heavy rain caught on my security Camera.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Heavy rain caught on my security Camera_000"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami opady\\Crazy rain in the morning- Reolink  security  Camera August 28, 2020.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Crazy rain in the morning- Reolink  security  Camera August 28, 2020_000"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\4K, 8MP DTV (Digital TV) Security Camera Sunny Day.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\4K, 8MP DTV (Digital TV) Security Camera Sunny Day"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\4MP IP Camera Hikvision DS 2CE1043G0E I Demo Video.mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\4MP IP Camera Hikvision DS 2CE1043G0E I Demo Video"
    # ),
    # (
    #     "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\Another who forgot this is a dead end..mp4",
    #     "C:\\Users\\yanam\\opady_dataset\\Another who forgot this is a dead end."
    # ),
    (
        "C:\\Users\\yanam\\opady_dataset\\folder z filmikami brak opadów\\wasting gas or they forgot where they lived and passed up their driveway..mp4",
        "C:\\Users\\yanam\\opady_dataset\\wasting gas or they forgot where they lived and passed up their driveway."
    ),
]

process_videos(files_to_process)
