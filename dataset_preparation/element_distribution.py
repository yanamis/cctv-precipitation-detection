import math
import os
import pickle


# Ścieżka do głównego folderu z danymi | Path to the main data folder
data_root = 'data/raw'

# Lista wybranych folderów do testowania | List of selected folders for testing
selected_folders = [
    os.path.join(data_root, 'brak_highway'),
    os.path.join(data_root, 'brak_hikvision'),
    os.path.join(data_root, 'brak_hikvision_2'),
    os.path.join(data_root, 'brak_istanbul'),
    os.path.join(data_root, 'brak_nonviolence'),
    os.path.join(data_root, 'brak_saleem'),
    os.path.join(data_root, 'brak_securicam'),
    os.path.join(data_root, 'brak_securicam_2'),
    os.path.join(data_root, 'brak_securicam_3'),
    os.path.join(data_root, 'brak_sunny'),
    os.path.join(data_root, 'brak_towncentre'),
    os.path.join(data_root, 'brak_wasting_gas'),

    os.path.join(data_root, 'opady_aau'),
    os.path.join(data_root, 'opady_blink'),
    os.path.join(data_root, 'opady_crazy'),
    os.path.join(data_root, 'opady_giant'),
    os.path.join(data_root, 'opady_giant_2'),
    os.path.join(data_root, 'opady_heavy'),
    os.path.join(data_root, 'opady_my_camera'),
    os.path.join(data_root, 'opady_night_footage'),
    os.path.join(data_root, 'opady_saleem_2')
]

# Obliczanie całkowitej liczby elementów | Calculate total number of elements
total_elements = sum(len(os.listdir(folder)) for folder in selected_folders)

# Obliczanie 8% całkowitej liczby elementów | Calculate 8% of total elements
x_percent = int(total_elements * 0.08)

# Liczba folderów w klasie 'opady' i 'brak' | Number of folders in 'opady' and 'brak' classes
num_opady_folders = len([folder for folder in selected_folders if 'opady' in os.path.basename(folder)])
num_brak_folders = len([folder for folder in selected_folders if 'brak' in os.path.basename(folder)])

# Obliczanie liczby elementów do pobrania z każdego folderu | Calculate number of elements to take per folder
elements_per_opady_folder = math.ceil(x_percent / 2 / num_opady_folders)
elements_per_brak_folder = math.ceil(x_percent / 2 / num_brak_folders)

# Inicjalizacja słownika num_test_files_per_directory | Initialize num_test_files_per_directory dictionary
num_test_files_per_directory = {}

for folder in selected_folders:
    folder_name = os.path.basename(folder)
    folder_elements = len(os.listdir(folder))

    if 'opady' in folder_name:
        elements_to_take = min(elements_per_opady_folder, folder_elements)
    else:
        elements_to_take = min(elements_per_brak_folder, folder_elements)

    num_test_files_per_directory[folder_name] = elements_to_take

# Zapisywanie danych do pliku | Saving data to file
data_to_save = {
    'data_root': data_root,
    'selected_folders': selected_folders,
    'num_test_files_per_directory': num_test_files_per_directory
}

# Zapis do dataset_preparation/ | Save into dataset_preparation/
os.makedirs('', exist_ok=True)
with open('dataset_preparation/data_element_distribution.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

