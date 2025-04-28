import os
import pickle
import shutil
import random
import cv2
import numpy as np


# Funkcja przetwarzająca obraz: skalowanie, przycięcie, filtracja medianowa | Function processing image: scaling, cropping, median filtering
def process_image(source, destination, img_height, img_width):
    image = cv2.imread(source)

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

    median_filtered_image = cv2.medianBlur(scaled_image, 5)
    cv2.imwrite(destination, median_filtered_image)


# Wczytanie listy użytych plików | Loading list of used files
with open('dataset_preparation/used_files.pkl', 'rb') as f:
    used_files = pickle.load(f)

# Wczytanie danych | Loading dataset information
class_names = np.load('saved_models/class_names.npy')

with open('dataset_preparation/data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

data_root = loaded_data['data_root']
selected_folders = loaded_data['selected_folders']
num_test_files_per_directory = loaded_data['num_test_files_per_directory']

# Wczytanie rozmiaru obrazów | Loading image dimensions
with open('dataset_preparation/image_dimensions.pkl', 'rb') as file:
    loaded_dimensions = pickle.load(file)

img_height = loaded_dimensions['img_height']
img_width = loaded_dimensions['img_width']

# Tworzenie katalogu dla zbioru testowego | Creating directory for test dataset
for class_label in ['brak', 'opady']:
    if os.path.exists(os.path.join(data_root, 'test', 'data', class_label)):
        shutil.rmtree(os.path.join(data_root, 'test', 'data', class_label))
    os.makedirs(os.path.join(data_root, 'test', 'data', class_label), exist_ok=True)

# Przygotowanie danych testowych | Preparing test data
for directory in selected_folders:
    class_label = os.path.basename(directory).split('_')[0]
    class_files = os.listdir(directory)

    random.seed(123)

    files_to_copy = random.sample([f for f in class_files if f not in used_files], num_test_files_per_directory[os.path.basename(directory)])

    for file in files_to_copy:
        source = os.path.join(directory, file)
        destination_folder = os.path.join(data_root, 'test', 'data', class_label)
        destination = os.path.join(destination_folder, file)

        process_image(source, destination, img_height, img_width)

# Liczenie liczby obrazów w zbiorze testowym | Counting number of images in test set
count_per_phase = {'test': {'brak': 0, 'opady': 0}}

phase = list(count_per_phase.keys())[0]

for class_label in count_per_phase[phase]:
    directory_path = os.path.join(data_root, phase, 'data', class_label)
    count = len(os.listdir(directory_path))
    count_per_phase[phase][class_label] = count

# Wyświetlenie statystyk | Printing statistics
for class_label, count in count_per_phase[phase].items():
    print('Liczba obrazów w {}/{}: {}'.format(phase, class_label, count))

# Zapisywanie count_per_phase | Saving count_per_phase
os.makedirs('', exist_ok=True)
with open('dataset_preparation/count_per_phase.pkl', 'wb') as file:
    pickle.dump(count_per_phase, file)
