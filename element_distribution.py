import os
import pickle

# Ścieżka do głównego folderu z danymi
data_root = 'C:\\Users\\yanam\\opady_dataset\\dataset'

# Lista wybranych folderów do testowania
selected_folders = [
    os.path.join(data_root, 'brak_highway'),
    # os.path.join(data_root, 'brak_istanbul'),
    os.path.join(data_root, 'brak_nonviolence'),
    os.path.join(data_root, 'brak_towncentre'),
    os.path.join(data_root, 'opady_aau'),
    os.path.join(data_root, 'opady_blink'),
    os.path.join(data_root, 'opady_crazy'),
    os.path.join(data_root, 'opady_heavy'),
    os.path.join(data_root, 'opady_saleem'),
]

# Obliczanie całkowitej liczby elementów we wszystkich wybranych folderach
total_elements = sum(len(os.listdir(folder)) for folder in selected_folders)

# Obliczanie 10% całkowitej liczby elementów
ten_percent = int(total_elements * 0.1)

# Obliczanie ile elementów ma być pobranych z każdego folderu
elements_per_folder = ten_percent // len(selected_folders)

# Rozdzielanie elementów między foldery i zapisywanie w słowniku
num_test_files_per_directory = {}

for folder in selected_folders:
    folder_name = os.path.basename(folder)
    folder_elements = len(os.listdir(folder))

    # Obliczanie ile elementów ma być pobranych z tego folderu
    elements_to_take = min(elements_per_folder, folder_elements)

    num_test_files_per_directory[folder_name] = elements_to_take

# Zapisywanie danych
data_to_save = {
    'data_root': data_root,
    'selected_folders': selected_folders,
    'num_test_files_per_directory': num_test_files_per_directory
}

with open('data_element_distribution.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

