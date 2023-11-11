import os
import pickle
import shutil
import random
import cv2
import numpy as np


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
    median_filtered_image = cv2.medianBlur(scaled_image, 7)

    # Zapisywanie przeskalowanego, obciętego i przefiltrowanego obrazu
    cv2.imwrite(destination, median_filtered_image)


# Wczytanie danych
class_names = np.load('class_names.npy')

with open('data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

data_root = loaded_data['data_root']

# Lista nowych folderów
selected_folders = [
    os.path.join(data_root, 'brak_4k'),
    os.path.join(data_root, 'brak_hikvision_3'),
    os.path.join(data_root, 'brak_sample'),

    os.path.join(data_root, 'opady_heavy_2'),
    os.path.join(data_root, 'opady_saleem'),
    os.path.join(data_root, 'opady_storm')
]

num_test_files_per_directory = {
    'brak_4k': 100,
    'brak_hikvision_3': 100,
    'brak_sample': 100,

    'opady_heavy_2': 100,
    'opady_saleem': 100,
    'opady_storm': 100
}

# Odczytywanie rozmiaru danych
with open('image_dimensions.pkl', 'rb') as file:
    loaded_dimensions = pickle.load(file)

img_height = loaded_dimensions['img_height']
img_width = loaded_dimensions['img_width']

# Tworzenie katalogu dla zbioru testowego
for class_label in ['brak', 'opady']:
    if os.path.exists(os.path.join(data_root, 'test', 'data', class_label)):
        shutil.rmtree(os.path.join(data_root, 'test', 'data', class_label))
    os.makedirs(os.path.join(data_root, 'test', 'data', class_label), exist_ok=True)

# Pobieranie danych testowych
for directory in selected_folders:
    # Pobranie etykiety z nazwy folderu
    class_label = os.path.basename(directory).split('_')[0]
    class_files = os.listdir(directory)

    # Zastosowanie seed, aby uzyskać takie same wyniki przy każdym uruchomieniu
    random.seed(123)

    # Losowanie unikalnego zestawu plików, wykluczając te z used_files
    files_to_copy = random.sample(class_files, num_test_files_per_directory[os.path.basename(directory)])

    for file in files_to_copy:
        source = os.path.join(directory, file)
        destination_folder = os.path.join(data_root, 'test', 'data', class_label)
        destination = os.path.join(destination_folder, file)

        process_image(source, destination, img_height, img_width)

# Liczebność zbioru testowego
count_per_phase = {'test': {'brak': 0, 'opady': 0}}

phase = list(count_per_phase.keys())[0]

for class_label in count_per_phase[phase]:
    directory_path = os.path.join(data_root, phase, 'data', class_label)
    count = len(os.listdir(directory_path))
    count_per_phase[phase][class_label] = count

for class_label, count in count_per_phase[phase].items():
    print('Liczba obrazów w {}/{}: {}'.format(phase, class_label, count))

# Zapisywanie count_per_phase
with open('count_per_phase.pkl', 'wb') as file:
    pickle.dump(count_per_phase, file)
