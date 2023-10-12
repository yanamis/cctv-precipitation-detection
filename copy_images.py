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
    ('C:\\Users\\yanam\\opady_dataset\\rain cctv night mood by saleem security systems_000',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_saleem'),
    ('C:\\Users\\yanam\\opady_dataset\\Blink Security Camera Shows What Rain Looks Like In Night Vision_mp4_000',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_blink'),
    ('C:\\Users\\yanam\\opady_dataset\\Heavy Rain ~ Caught on security camera, Kendal. #Shorts_000',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_kendal'),
    ('C:\\Users\\yanam\\opady_dataset\\Heavy rain caught on my security Camera_000',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_heavy'),
    ('C:\\Users\\yanam\\opady_dataset\\Crazy rain in the morning- Reolink  security  Camera August 28, 2020_000',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_crazy')
]

for input_folder, output_folder in folders:
    copy_files(input_folder, output_folder)
