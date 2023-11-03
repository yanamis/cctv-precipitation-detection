import os
import pickle
import shutil
import random
import numpy as np
import cv2

import tensorflow as tf
from keras import layers, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler

import matplotlib.pyplot as plt


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
    median_filtered_image = cv2.medianBlur(scaled_image, 3)

    # Zapisywanie przeskalowanego, obciętego i przefiltrowanego obrazu
    cv2.imwrite(destination, median_filtered_image)


def display_images(train_dataset, class_names):
    plt.figure(figsize=(12, 6))

    def display_subset(dataset):
        for i, (images, labels) in enumerate(dataset.take(1)):
            opady_images = []
            brak_images = []
            for j in range(len(labels)):
                if class_names[labels[j]] == 'opady':
                    opady_images.append((images[j], labels[j]))
                else:
                    brak_images.append((images[j], labels[j]))

            for j in range(4):
                ax = plt.subplot(2, 4, j + 1)
                ax.text(0, -5, 'Class: {}'.format(class_names[opady_images[j][1]]), fontsize=10, color='black')
                ax.imshow(opady_images[j][0].numpy().astype("uint8"))
                ax.axis("off")

                ax = plt.subplot(2, 4, j + 5)
                ax.text(0, -5, 'Class: {}'.format(class_names[brak_images[j][1]]), fontsize=10, color='black')
                ax.imshow(brak_images[j][0].numpy().astype("uint8"))
                ax.axis("off")

    display_subset(train_dataset)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


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
    # os.path.join(data_root, 'brak_nonviolence'),
    os.path.join(data_root, 'brak_saleem'),
    os.path.join(data_root, 'brak_securicam'),
    os.path.join(data_root, 'brak_securicam_2'),
    os.path.join(data_root, 'brak_securicam_3'),
    os.path.join(data_root, 'brak_spac'),
    os.path.join(data_root, 'brak_sunny'),
    os.path.join(data_root, 'brak_towncentre'),

    os.path.join(data_root, 'opady_aau'),
    os.path.join(data_root, 'opady_blink'),
    os.path.join(data_root, 'opady_cityscapes'),
    os.path.join(data_root, 'opady_crazy'),
    os.path.join(data_root, 'opady_giant'),
    os.path.join(data_root, 'opady_giant_2'),
    os.path.join(data_root, 'opady_heavy'),
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
        max_num_files = 300
        num_files_per_directory[folder_name] = min(len(os.listdir(folder)) - num_test_files_per_directory[folder_name],
                                                   max_num_files)


# Ręczne określenie liczby elementów do pobrania z pozostałych folderów
additional_values = {
    'brak_cityscapes': 300,
    'brak_spac': 300,

    'opady_cityscapes': 300,
    # 'opady_giant': 953,
    # 'opady_heavy': 1154,
    # 'opady_night_footage': 400,
    'opady_spac': 300
}

num_files_per_directory.update(additional_values)

# Lista użytych plików
used_files = []

# Rozmiar obrazu
img_height = 300
img_width = 450

# Pobieranie danych
for directory in data_directories:
    # Pobranie etykiety z nazwy folderu
    class_label = os.path.basename(directory).split('_')[0]
    class_files = os.listdir(directory)

    # Zastosowanie seed, aby uzyskać takie same wyniki przy każdym uruchomieniu
    random.seed(123)

    # Losowanie unikalnego zestawu plików
    files_to_copy = random.sample(class_files, num_files_per_directory[os.path.basename(directory)])

    for phase in ['train', 'val']:
        # Podział na zbiór treningowy i walidacyjny
        for file in (files_to_copy[:int(0.75 * len(files_to_copy))] if phase == 'train'
        else files_to_copy[int(0.75 * len(files_to_copy)):]):  # 75% do train, 25% do val
            source = os.path.join(directory, file)
            destination_folder = os.path.join(data_root, phase, 'data', class_label)
            destination = os.path.join(destination_folder, file)

            process_image(source, destination, img_height, img_width)

            used_files.append(file)

# Liczebność poszczególnych zbiorów
count_per_phase = {'train': {'brak': 0, 'opady': 0}, 'val': {'brak': 0, 'opady': 0}}

total_count = 0

for phase in ['train', 'val']:
    for class_label in ['brak', 'opady']:
        directory_path = os.path.join(data_root, phase, 'data', class_label)
        count = len(os.listdir(directory_path))
        count_per_phase[phase][class_label] = count
        total_count += count

for phase, classes in count_per_phase.items():
    for class_label, count in classes.items():
        print('Liczba obrazów w {}/{}: {}'.format(phase, class_label, count))

# Liczba wszystkich elementów
print('Łączna liczba obrazów: {}'.format(total_count))

batch_size = 32

# Generator danych dla zbioru treningowego
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_root, 'train', 'data'),
    batch_size=batch_size,
    image_size=(img_height, img_width),
    seed=123
)

# Generator danych dla zbioru walidacyjnego
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_root, 'val', 'data'),
    batch_size=batch_size,
    image_size=(img_height, img_width),
    seed=123
)

class_names = train_ds.class_names

# Wyświetlenie obrazów dla zbioru treningowego
display_images(train_ds, class_names)

plt.tight_layout()
plt.show()

# Tworzenie warstwy normalizacji
normalization_layer = layers.Rescaling(1. / 255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Definicja warstwy augmentacji
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal",
                                   input_shape=(img_height,
                                                img_width,
                                                3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(550, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation="softmax")
])

optimizer = Adam(learning_rate=0.00001)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)

epochs = 20
# Trenowanie modelu
history = model.fit(
    normalized_train_ds,
    validation_data=normalized_val_ds,
    epochs=epochs,
    callbacks=[early_stopping, lr_scheduler]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

# Zapisywanie modelu
model.save('model_v8.h5')

# Zapisywanie listy użytych plików
with open('used_files.pkl', 'wb') as f:
    pickle.dump(used_files, f)

# Zapisywanie class_names
np.save('class_names.npy', class_names)

# Wykres dokładności
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Dokładność Treningowa')
plt.plot(epochs_range, val_acc, label='Dokładność Walidacyjna')
plt.legend(loc='lower right')
plt.title('Dokładność Treningowa i Walidacyjna')

# Wykres straty
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Strata Treningowa')
plt.plot(epochs_range, val_loss, label='Strata Walidacyjna')
plt.legend(loc='upper right')
plt.title('Strata Treningowa i Walidacyjna')
plt.show()
