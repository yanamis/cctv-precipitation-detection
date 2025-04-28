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


# Lista folderów wejściowych i wyjściowych
folders = [
    ('C:\\Users\\yanam\\opady_dataset\\Hikvision DS-2CD2042WD-I 4MP POE IP Camera Footage',
     'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_hikvision')
]

for input_folder, output_folder in folders:
    copy_files(input_folder, output_folder)
