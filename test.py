import os
import pickle
import shutil
import random
import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras import layers


def find_misclassified_images(model, test_dataset, class_names):
    misclassified_images = []

    for images, labels in test_dataset:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = labels.numpy()

        for i in range(len(true_labels)):
            if predicted_labels[i] != true_labels[i]:
                image = images[i].numpy() * 255
                misclassified_images.append(
                    (image.astype("uint8"), class_names[predicted_labels[i]], class_names[true_labels[i]]))

    return misclassified_images


def calculate_misclassification_stats(misclassified_images, class_names, count_per_phase):
    misclassified_counts = {class_name: 0 for class_name in class_names}

    for i in range(len(misclassified_images)):
        true_class = misclassified_images[i][2]
        misclassified_counts[true_class] += 1

    total_misclassified = sum(misclassified_counts.values())

    print("Liczba błędnie sklasyfikowanych obrazów: {}".format(total_misclassified))
    print("Statystyki dla każdej klasy:")
    for class_name, count in misclassified_counts.items():
        total_class_samples = count_per_phase['test'][class_name]
        misclassification_rate = (count / total_class_samples) * 100
        print("Klasa {}: {} błędnie sklasyfikowanych obrazów ({:.2f}%)".format(class_name, count, misclassification_rate))


# Wczytanie modelu
model = load_model('model_v2.h5')

# Wczytanie listy użytych plików
with open('used_files.pkl', 'rb') as f:
    used_files = pickle.load(f)

# Wczytanie zmiennych
class_names = np.load('class_names.npy')
data_info = np.load('data_info.npy', allow_pickle=True).item()
data_root = data_info['data_root']
data_directories = data_info['data_directories']

# Tworzenie katalogu dla zbioru testowego
for class_label in ['brak', 'opady']:
    if os.path.exists(os.path.join(data_root, 'test', 'data', class_label)):
        shutil.rmtree(os.path.join(data_root, 'test', 'data', class_label))
    os.makedirs(os.path.join(data_root, 'test', 'data', class_label), exist_ok=True)

# Ręczne określenie liczby elementów dla zbioru testowego
num_files_per_directory = {
    'brak_cityscapes': 0,
    'brak_highway': 0,
    'brak_istanbul': 0,
    'brak_nonviolence': 200,
    'brak_spac': 0,
    'brak_towncentre': 50,
    'opady_aau': 50,
    'opady_blink': 50,
    'opady_cityscapes': 0,
    'opady_crazy': 50,
    'opady_heavy': 50,
    'opady_kendal': 0,
    'opady_saleem': 50,
    'opady_spac': 0
}

# Pobieranie danych testowych
for directory in data_directories:
    class_label = os.path.basename(directory).split('_')[0]  # Pobranie etykiety z nazwy katalogu
    class_files = os.listdir(directory)

    # Zastosowanie seed, aby uzyskać takie same wyniki przy każdym uruchomieniu
    random.seed(123)

    # Losowanie unikalnego zestawu plików, wykluczając te z used_files
    if num_files_per_directory[os.path.basename(directory)] != 0:
        files_to_copy = random.sample([f for f in class_files if f not in used_files], num_files_per_directory[os.path.basename(directory)])

        for file in files_to_copy:
            source = os.path.join(directory, file)
            destination_folder = os.path.join(data_root, 'test', 'data', class_label)
            destination = os.path.join(destination_folder, file)
            shutil.copy(source, destination)

# Liczebność zbioru testowego
count_per_phase = {'test': {'brak': 0, 'opady': 0}}

phase = list(count_per_phase.keys())[0]

for class_label in count_per_phase[phase]:
    directory_path = os.path.join(data_root, phase, 'data', class_label)
    count = len(os.listdir(directory_path))
    count_per_phase[phase][class_label] = count

for class_label, count in count_per_phase[phase].items():
    print('Liczba obrazów w {}/{}: {}'.format(phase, class_label, count))

img_height = 300
img_width = 300

batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_root, 'test', 'data'),
    batch_size=batch_size,
    image_size=(img_height, img_width),
    seed=123,
    interpolation='lanczos3',
    crop_to_aspect_ratio=True,
)

normalization_layer = layers.Rescaling(1. / 255)
normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Ewaluacja na zbiorze testowym
test_loss, test_accuracy = model.evaluate(normalized_test_ds)
print("Test accuracy: {}".format(test_accuracy))

# Sprawdzenie, które obrazy zostały źle sklasyfikowane
misclassified_images = find_misclassified_images(model, normalized_test_ds, class_names)

# Obliczanie liczby błędnie sklasyfikowanych obrazów dla każdej klasy
calculate_misclassification_stats(misclassified_images, class_names, count_per_phase)
