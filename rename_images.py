import os

# Ścieżka do głównego folderu z danymi
data_root = 'C:\\Users\\yanam\\opady_dataset\\dataset'

# Lista folderów z danymi
data_directories = [
    os.path.join(data_root, 'brak_cityscapes'),
    os.path.join(data_root, 'brak_highway'),
    os.path.join(data_root, 'brak_istanbul'),
    os.path.join(data_root, 'brak_nonviolence'),
    os.path.join(data_root, 'brak_spac'),
    os.path.join(data_root, 'brak_towncentre'),
    os.path.join(data_root, 'opady_aau'),
    os.path.join(data_root, 'opady_blink'),
    os.path.join(data_root, 'opady_cityscapes'),
    os.path.join(data_root, 'opady_crazy'),
    os.path.join(data_root, 'opady_heavy'),
    os.path.join(data_root, 'opady_kendal'),
    os.path.join(data_root, 'opady_saleem'),
    os.path.join(data_root, 'opady_spac')
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
