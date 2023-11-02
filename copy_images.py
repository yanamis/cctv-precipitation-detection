import os
import shutil


def copy_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter = 1

    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                old_path = os.path.join(subdir, file)
                new_name = os.path.join(output_folder, 'img_{}.jpg'.format(counter))
                shutil.copy(old_path, new_name)
                counter += 1


# Lista folderów wejściowych i odpowiadających im folderów wyjściowych
folders = [
    # ('C:\\Users\\yanam\\opady_dataset\\Rain on my security camera 11-12-22 5_30 pm.',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_my_camera'),
    # ('C:\\Users\\yanam\\opady_dataset\\Heavy Snowfall, big snowflakes ｜ Free Footage for MOAH Members',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_heavy_snow'),
    # ('C:\\Users\\yanam\\opady_dataset\\GIANT snowflakes falling',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_giant'),
    # ('C:\\Users\\yanam\\opady_dataset\\Giant snowflakes fall in the front yard on 27-Mar-11',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_giant_2'),
    # ('C:\\Users\\yanam\\opady_dataset\\My cctv camera shows night footage after rainfall during daytime',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_night_footage'),
    # ('C:\\Users\\yanam\\opady_dataset\\rain night cctv by saleem security systems',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_saleem_2'),
    # ('C:\\Users\\yanam\\opady_dataset\\cctv night mood by saleem security systems',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_saleem'),
    # ('C:\\Users\\yanam\\opady_dataset\\Hikvision DS-2CD2042WD-I 4MP POE IP Camera Footage',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_hikvision'),
    # ('C:\\Users\\yanam\\opady_dataset\\Securicam 2MP 1080P POE IP Bullet Camera Footage',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_securicam'),
    # ('C:\\Users\\yanam\\opady_dataset\\Securicam 1000TVL Varifocal Bullet Camera',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_securicam_2'),
    # ('C:\\Users\\yanam\\opady_dataset\\Securicam AHD 720P Varifocal Dome Camera',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_securicam_3'),
    # ('C:\\Users\\yanam\\opady_dataset\\rain cctv night mood by saleem security systems_000',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_saleem'),
    # ('C:\\Users\\yanam\\opady_dataset\\Blink Security Camera Shows What Rain Looks Like In Night Vision_mp4_000',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_blink'),
    # ('C:\\Users\\yanam\\opady_dataset\\Heavy Rain ~ Caught on security camera, Kendal. #Shorts_000',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_kendal'),
    # ('C:\\Users\\yanam\\opady_dataset\\Heavy rain caught on my security Camera_000',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_heavy'),
    # ('C:\\Users\\yanam\\opady_dataset\\Crazy rain in the morning- Reolink  security  Camera August 28, 2020_000',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_crazy')
    ('C:\\Users\\yanam\\opady_dataset\\4K, 8MP DTV (Digital TV) Security Camera Sunny Day',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_sunny'),
    ('C:\\Users\\yanam\\opady_dataset\\4MP IP Camera Hikvision DS 2CE1043G0E I Demo Video',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_hikvision_2'),
    # ('C:\\Users\\yanam\\opady_dataset\\Another who forgot this is a dead end.',
    #  'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_dead_end'),
    ('C:\\Users\\yanam\\opady_dataset\\wasting gas or they forgot where they lived and passed up their driveway.',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_wasting_gas'),
]

for input_folder, output_folder in folders:
    copy_files(input_folder, output_folder)
