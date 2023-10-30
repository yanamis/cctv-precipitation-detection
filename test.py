import os
import pickle
import shutil
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import load_model
from keras import layers


def find_images_info(model, test_dataset, class_names):
    all_images = []
    misclassified_images = []

    for images, labels in test_dataset:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = labels.numpy()

        for i in range(len(true_labels)):
            file_name = os.path.basename(test_ds.file_paths[i])

            all_images.append(
                (file_name, class_names[predicted_labels[i]], class_names[true_labels[i]]))
            if predicted_labels[i] != true_labels[i]:
                misclassified_images.append(
                    (file_name, class_names[predicted_labels[i]], class_names[true_labels[i]]))

    return all_images, misclassified_images


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
model = load_model('model_v3.h5')

# Wczytanie listy użytych plików
with open('used_files.pkl', 'rb') as f:
    used_files = pickle.load(f)

# Wczytanie danych
class_names = np.load('class_names.npy')

with open('data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

data_root = loaded_data['data_root']
selected_folders = loaded_data['selected_folders']
num_test_files_per_directory = loaded_data['num_test_files_per_directory']

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
    files_to_copy = random.sample([f for f in class_files if f not in used_files], num_test_files_per_directory[os.path.basename(directory)])

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
print("Dokładność testowa: {}".format(test_accuracy))

# Sprawdzenie, które obrazy zostały źle sklasyfikowane / Tworzenie listy wszystkich przewidywanych etykiet
all_images, misclassified_images = find_images_info(model, normalized_test_ds, class_names)

# Obliczanie liczby błędnie sklasyfikowanych obrazów dla każdej klasy
calculate_misclassification_stats(misclassified_images, class_names, count_per_phase)

# Tworzenie DataFrame z informacjami o obrazach
df = pd.DataFrame(all_images, columns=["Plik obrazu", "Etykieta przewidywana", "Etykieta prawdziwa"])

# Zapisanie do pliku Excel
excel_path = "all_images_info.xlsx"
df.to_excel(excel_path, index=False)
