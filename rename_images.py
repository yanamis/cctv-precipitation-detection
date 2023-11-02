import os

# Ścieżka do głównego folderu z danymi
data_root = 'C:\\Users\\yanam\\opady_dataset\\dataset'

# Lista folderów z danymi
data_directories = [
    # os.path.join(data_root, 'brak_cityscapes'),
    # os.path.join(data_root, 'brak_highway'),
    # os.path.join(data_root, 'brak_istanbul'),
    # os.path.join(data_root, 'brak_nonviolence'),
    # os.path.join(data_root, 'brak_spac'),
    # os.path.join(data_root, 'brak_towncentre'),
    # os.path.join(data_root, 'opady_aau'),
    # os.path.join(data_root, 'opady_blink'),
    # os.path.join(data_root, 'opady_cityscapes'),
    # os.path.join(data_root, 'opady_crazy'),
    # os.path.join(data_root, 'opady_heavy'),
    # os.path.join(data_root, 'opady_kendal'),
    # os.path.join(data_root, 'opady_saleem'),
    # os.path.join(data_root, 'opady_spac')
    # os.path.join(data_root, 'opady_night_footage'),
    # os.path.join(data_root, 'opady_saleem_2'),
    os.path.join(data_root, 'brak_saleem'),
    # os.path.join(data_root, 'brak_hikvision'),
    # os.path.join(data_root, 'brak_securicam'),
    # os.path.join(data_root, 'brak_securicam_2'),
    # os.path.join(data_root, 'brak_securicam_3'),
    # os.path.join(data_root, 'brak_sunny'),
    # os.path.join(data_root, 'brak_hikvision_2'),
    # os.path.join(data_root, 'brak_wasting_gas'),
    # os.path.join(data_root, 'opady_my_camera'),
    # os.path.join(data_root, 'opady_heavy_snow'),
    # os.path.join(data_root, 'opady_giant'),
    # os.path.join(data_root, 'opady_giant_2'),

]

# Tworzenie unikalnej nazwy dla plików w folderach
for directory in data_directories:
    class_files = os.listdir(directory)

    for file in class_files:
        source = os.path.join(directory, file)

        # Tworzenie unikalnej nazwy dla pliku
        new_file_name = "{}_{}".format(os.path.basename(directory), file)

        destination = os.path.join(directory, new_file_name)

        os.rename(source, destination)
