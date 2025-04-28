import os
import pickle
import shutil
import random
import cv2


# Funkcja przetwarzająca obraz: skalowanie, przycięcie, filtracja medianowa | Function to process image: scaling, cropping, median filtering
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


# Funkcja do podziału i kopiowania danych | Function to split and copy data
def split_and_copy_data(directory, files, destination, img_height, img_width):
    random.seed(123)

    random.shuffle(files)

    for i in range(4):
        phase = 'split_' + str(i + 1)

        order = [i % 4, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4]

        for class_label in ['brak', 'opady']:
            os.makedirs(os.path.join(destination, phase, 'train', 'data', class_label), exist_ok=True)
            os.makedirs(os.path.join(destination, phase, 'val', 'data', class_label), exist_ok=True)

        num_files = num_files_per_directory[os.path.basename(directory)]

        for j, file in enumerate(files[:num_files]):
            source = os.path.join(directory, file)
            class_label = os.path.basename(directory).split('_')[0]
            destination_folder = os.path.join(destination, phase)

            split_index = order[j % 4]

            if split_index == 3:
                destination_val = os.path.join(destination_folder, 'val', 'data', class_label, file)
                process_image(source, destination_val, img_height, img_width)

                used_files.append(file)
            else:
                destination_train = os.path.join(destination_folder, 'train', 'data', class_label, file)
                process_image(source, destination_train, img_height, img_width)


# Wczytywanie informacji o danych | Loading data distribution information
with open('dataset_preparation/data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

selected_folders = loaded_data['selected_folders']
num_test_files_per_directory = loaded_data['num_test_files_per_directory']
data_root = loaded_data['data_root']

# Lista folderów z danymi | List of data directories
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

# Tworzenie folderów na dane | Creating directories for data
for phase in ['train', 'val']:
    for class_label in ['brak', 'opady']:
        if os.path.exists(os.path.join(data_root, phase, 'data', class_label)):
            shutil.rmtree(os.path.join(data_root, phase, 'data', class_label))
        os.makedirs(os.path.join(data_root, phase, 'data', class_label), exist_ok=True)

# Inicjalizacja liczby plików na katalog | Initialize num_files_per_directory
num_files_per_directory = {folder_name: 0 for folder_name in map(os.path.basename, data_directories)}

# Uaktualnianie num_files_per_directory | Updating num_files_per_directory
for folder in data_directories:
    folder_name = os.path.basename(folder)
    if folder in selected_folders:
        num_files_per_directory[folder_name] = len(os.listdir(folder)) - num_test_files_per_directory[folder_name]

# Dodatkowe wartości ręczne | Additional manual values
additional_values = {
    'brak_cityscapes': 300,
    'brak_hikvision_2': 249,
    'brak_nonviolence': 250,
    'brak_spac': 300,

    'opady_cityscapes': 300,
    'opady_crazy': 300,
    'opady_giant': 355,
    'opady_heavy': 560,
    'opady_my_camera': 500,
    'opady_night_footage': 195,
    'opady_spac': 300
}

num_files_per_directory.update(additional_values)

# Lista użytych plików | List of used files
used_files = []

# Rozmiary obrazów | Image dimensions
img_height = 300
img_width = 450

# Przetwarzanie danych | Processing data
for directory in data_directories:
    class_label = os.path.basename(directory).split('_')[0]
    class_files = os.listdir(directory)

    split_and_copy_data(directory, class_files, data_root, img_height, img_width)

# Zapisywanie wymiarów obrazów | Saving image dimensions
image_dimensions = {
    'img_height': img_height,
    'img_width': img_width
}
os.makedirs('', exist_ok=True)
with open('dataset_preparation/image_dimensions.pkl', 'wb') as file:
    pickle.dump(image_dimensions, file)

# Zapisywanie listy użytych plików | Saving used files list
with open('dataset_preparation/used_files.pkl', 'wb') as f:
    pickle.dump(used_files, f)
