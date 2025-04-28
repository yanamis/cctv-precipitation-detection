import os
import shutil


def copy_subfolders(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter = 1

    for subdir, _, files in os.walk(input_folder):
        if 'new' in subdir.lower():
            for file in files:
                old_path = os.path.join(subdir, file)
                new_name = os.path.join(output_folder, 'img_{}.jpg'.format(counter))
                shutil.copy(old_path, new_name)
                counter += 1


input_folder = 'C:\\Users\\yanam\\opady_dataset\\archive'

output_folder = 'C:\\Users\\yanam\\opady_dataset\\new_archive'

# Przeniesienie plików z podfolderów "new" do nowego folderu
copy_subfolders(input_folder, output_folder)
