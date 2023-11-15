import os
import pickle
import shutil
import random
import cv2


def process_image(source, destination, img_height, img_width):
    # Wczytanie obrazu
    image = cv2.imread(source)

    # Określenie proporcji
    shorter_side, longer_side = sorted(image.shape[:2])
    aspect_ratio = longer_side / shorter_side

    if aspect_ratio >= 1.5:
        scale_factor = img_height / shorter_side
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        scaled_image = scaled_image[:, :img_width]
    else:
        scale_factor = img_width / longer_side
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        scaled_image = scaled_image[:img_height, :]

    # Zastosowanie filtracji medianowej
    median_filtered_image = cv2.medianBlur(scaled_image, 5)

    # Zapisywanie przeskalowanego, obciętego i przefiltrowanego obrazu
    cv2.imwrite(destination, median_filtered_image)


def split_and_copy_data(directory, files, destination, img_height, img_width):
    # Zastosowanie seed, aby uzyskać takie same wyniki przy każdym uruchomieniu
    random.seed(123)

    # Losowanie unikalnego zestawu plików
    random.shuffle(files)

    # Podział na cztery zbiory
    for i in range(4):
        phase = 'split_' + str(i + 1)

        # Kolejność podziału
        order = [i % 4, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4]

        # Tworzenie folderów dla zbiorów treningowego i walidacyjnego
        for class_label in ['brak', 'opady']:
            os.makedirs(os.path.join(destination, phase, 'train', 'data', class_label), exist_ok=True)
            os.makedirs(os.path.join(destination, phase, 'val', 'data', class_label), exist_ok=True)

        # Liczba plików do skopiowania dla danego podziału
        num_files = num_files_per_directory[os.path.basename(directory)]

        # Kopiowanie plików do odpowiednich zbiorów
        for j, file in enumerate(files[:num_files]):
            source = os.path.join(directory, file)
            class_label = os.path.basename(directory).split('_')[0]
            destination_folder = os.path.join(destination, phase)

            # Ustalona kolejność dla danego podziału
            split_index = order[j % 4]

            # Kopiowanie pliku do odpowiedniego zbioru
            if split_index == 3:
                # Zbiór walidacyjny
                destination_val = os.path.join(destination_folder, 'val', 'data', class_label, file)
                process_image(source, destination_val, img_height, img_width)

                used_files.append(file)
            else:
                # Zbiór treningowy
                destination_train = os.path.join(destination_folder, 'train', 'data', class_label, file)
                process_image(source, destination_train, img_height, img_width)


with open('data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

selected_folders = loaded_data['selected_folders']
num_test_files_per_directory = loaded_data['num_test_files_per_directory']

# Ścieżka do głównego folderu z danymi
data_root = loaded_data['data_root']

# Lista folderów z danymi
data_directories = [
    os.path.join(data_root, 'brak_cityscapes'),
    os.path.join(data_root, 'brak_highway'),
    os.path.join(data_root, 'brak_hikvision'),
    os.path.join(data_root, 'brak_hikvision_2'),
    os.path.join(data_root, 'brak_istanbul'),
    os.path.join(data_root, 'brak_nonviolence'),
    os.path.join(data_root, 'brak_saleem'),
    os.path.join(data_root, 'brak_securicam'),
    os.path.join(data_root, 'brak_securicam_2'),
    os.path.join(data_root, 'brak_securicam_3'),
    os.path.join(data_root, 'brak_spac'),
    os.path.join(data_root, 'brak_sunny'),
    os.path.join(data_root, 'brak_towncentre'),
    os.path.join(data_root, 'brak_wasting_gas'),

    os.path.join(data_root, 'opady_aau'),
    os.path.join(data_root, 'opady_blink'),
    os.path.join(data_root, 'opady_cityscapes'),
    os.path.join(data_root, 'opady_crazy'),
    os.path.join(data_root, 'opady_giant'),
    os.path.join(data_root, 'opady_giant_2'),
    os.path.join(data_root, 'opady_heavy'),
    # os.path.join(data_root, 'opady_heavy_snow'),
    os.path.join(data_root, 'opady_my_camera'),
    os.path.join(data_root, 'opady_night_footage'),
    # os.path.join(data_root, 'opady_saleem'),
    os.path.join(data_root, 'opady_saleem_2'),
    os.path.join(data_root, 'opady_spac')
]

# Tworzenie folderów dla zbiorów treningowego i walidacyjnego
for phase in ['train', 'val']:
    for class_label in ['brak', 'opady']:
        if os.path.exists(os.path.join(data_root, phase, 'data', class_label)):
            shutil.rmtree(os.path.join(data_root, phase, 'data', class_label))
        os.makedirs(os.path.join(data_root, phase, 'data', class_label), exist_ok=True)

# Inicjalizacja num_files_per_directory z domyślnymi wartościami
num_files_per_directory = {folder_name: 0 for folder_name in map(os.path.basename, data_directories)}

# Dostosowanie num_files_per_directory na podstawie selected_folders
for folder in data_directories:
    folder_name = os.path.basename(folder)
    if folder in selected_folders:
        num_files_per_directory[folder_name] = len(os.listdir(folder)) - num_test_files_per_directory[folder_name]


# Ręczne określenie liczby elementów do pobrania z pozostałych folderów
additional_values = {
    'brak_cityscapes': 300,
    'brak_hikvision_2': 249,
    'brak_nonviolence': 250,
    'brak_spac': 300,

    'opady_cityscapes': 300,
    'opady_crazy': 300,
    'opady_giant': 355,
    'opady_heavy': 560,
    # 'opady_heavy_snow': 250,
    'opady_my_camera': 500,
    'opady_night_footage': 195,
    'opady_spac': 300
}

num_files_per_directory.update(additional_values)

# Lista użytych plików
used_files = []

# Rozmiar obrazu
img_height = 300
img_width = 450

# Pobieranie danych i tworzenie podziałów
for directory in data_directories:
    class_label = os.path.basename(directory).split('_')[0]
    class_files = os.listdir(directory)

    split_and_copy_data(directory, class_files, data_root, img_height, img_width)

# Zapisywanie rozmiaru danych
image_dimensions = {
    'img_height': img_height,
    'img_width': img_width
}

with open('image_dimensions.pkl', 'wb') as file:
    pickle.dump(image_dimensions, file)

# Zapisywanie listy użytych plików
with open('used_files.pkl', 'wb') as f:
    pickle.dump(used_files, f)
