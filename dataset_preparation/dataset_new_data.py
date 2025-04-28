import os
import pickle
import shutil
import random
import cv2
import numpy as np


# Funkcja przetwarzająca obraz: skalowanie, przycięcie i filtracja medianowa | Function processing image: scaling, cropping, and median filtering
def process_image(source, destination, img_height, img_width):
    # Wczytanie obrazu | Load image
    image = cv2.imread(source)

    # Określenie proporcji | Determine aspect ratio
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

    # Zastosowanie filtracji medianowej | Apply median filtering
    median_filtered_image = cv2.medianBlur(scaled_image, 5)

    # Zapisywanie przetworzonego obrazu | Save processed image
    cv2.imwrite(destination, median_filtered_image)


# Wczytanie class_names | Loading class names
class_names = np.load('saved_models/class_names.npy')

# Wczytanie danych o rozkładzie elementów | Loading element distribution data
with open('dataset_preparation/data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

data_root = loaded_data['data_root']

# Lista nowych folderów do pobrania | List of selected folders
selected_folders = [
    os.path.join(data_root, 'brak_4k'),
    os.path.join(data_root, 'brak_hikvision_3'),
    os.path.join(data_root, 'brak_sample'),

    os.path.join(data_root, 'opady_heavy_2'),
    os.path.join(data_root, 'opady_saleem'),
    os.path.join(data_root, 'opady_storm')
]

# Liczba plików do testu na folder | Number of test files per folder
num_test_files_per_directory = {
    'brak_4k': 100,
    'brak_hikvision_3': 100,
    'brak_sample': 100,

    'opady_heavy_2': 100,
    'opady_saleem': 100,
    'opady_storm': 100
}

# Wczytanie rozmiarów obrazów | Loading image dimensions
with open('dataset_preparation/image_dimensions.pkl', 'rb') as file:
    loaded_dimensions = pickle.load(file)

img_height = loaded_dimensions['img_height']
img_width = loaded_dimensions['img_width']

# Tworzenie katalogów dla zbioru testowego | Creating directories for test set
for class_label in ['brak', 'opady']:
    if os.path.exists(os.path.join(data_root, 'test', 'data', class_label)):
        shutil.rmtree(os.path.join(data_root, 'test', 'data', class_label))
    os.makedirs(os.path.join(data_root, 'test', 'data', class_label), exist_ok=True)

# Przygotowywanie danych testowych | Preparing test data
for directory in selected_folders:
    # Odczytanie etykiety klasy z nazwy folderu | Get class label from folder name
    class_label = os.path.basename(directory).split('_')[0]
    class_files = os.listdir(directory)

    # Ustalenie seeda dla powtarzalności | Fix random seed for reproducibility
    random.seed(123)

    files_to_copy = random.sample(class_files, num_test_files_per_directory[os.path.basename(directory)])

    for file in files_to_copy:
        source = os.path.join(directory, file)
        destination_folder = os.path.join(data_root, 'test', 'data', class_label)
        destination = os.path.join(destination_folder, file)

        process_image(source, destination, img_height, img_width)

# Liczenie obrazów w zbiorze testowym | Counting images in test set
count_per_phase = {'test': {'brak': 0, 'opady': 0}}

phase = list(count_per_phase.keys())[0]

for class_label in count_per_phase[phase]:
    directory_path = os.path.join(data_root, phase, 'data', class_label)
    count = len(os.listdir(directory_path))
    count_per_phase[phase][class_label] = count

# Wypisanie liczby obrazów | Printing image counts
for class_label, count in count_per_phase[phase].items():
    print('Liczba obrazów w {}/{}: {}'.format(phase, class_label, count))

# Zapisywanie count_per_phase do pliku | Saving count_per_phase to file
with open('dataset_preparation/count_per_phase.pkl', 'wb') as file:
    pickle.dump(count_per_phase, file)
