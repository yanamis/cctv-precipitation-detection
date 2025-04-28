import os
import pickle
import numpy as np

import tensorflow as tf
from keras import layers, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler

import matplotlib.pyplot as plt


# Funkcja wyświetlająca przykładowe obrazy z datasetu | Function to display example images from the dataset
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


# Funkcja zmieniająca wartość learning rate | Function to adjust learning rate during training
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Wczytywanie danych o rozkładzie elementów | Loading element distribution data
with open('dataset_preparation/data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

selected_folders = loaded_data['selected_folders']
num_test_files_per_directory = loaded_data['num_test_files_per_directory']

# Ścieżka do głównego folderu z danymi | Path to the main data folder
data_root = loaded_data['data_root']

# Wczytywanie rozmiarów obrazów | Loading image dimensions
with open('dataset_preparation/image_dimensions.pkl', 'rb') as file:
    loaded_dimensions = pickle.load(file)

img_height = loaded_dimensions['img_height']
img_width = loaded_dimensions['img_width']

batch_size = 32

# Generowanie danych dla każdego podziału | Generating data for each split
for split_num in range(1, 5):
    split_folder = 'split_' + str(split_num)

    # Generator danych dla zbioru treningowego | Generator for training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_root, split_folder, 'train', 'data'),
        batch_size=batch_size,
        image_size=(img_height, img_width),
        seed=123
    )

    # Generator danych dla zbioru walidacyjnego | Generator for validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_root, split_folder, 'val', 'data'),
        batch_size=batch_size,
        image_size=(img_height, img_width),
        seed=123
    )

    class_names = train_ds.class_names

    # Wyświetlenie przykładowych obrazów | Displaying example images
    display_images(train_ds, class_names)

    plt.tight_layout()
    plt.show()

    # Normalizacja obrazów | Image normalization
    normalization_layer = layers.Rescaling(1. / 255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Augmentacja danych | Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal",
                                   input_shape=(img_height,
                                                img_width,
                                                3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    arch = 3 # Wersja architektury | Architecture version

    # Definicja modelu CNN | Defining CNN model
    model = Sequential([
        data_augmentation,
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
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

    # Trenowanie modelu | Model training
    history = model.fit(
        normalized_train_ds,
        validation_data=normalized_val_ds,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Zapisywanie modelu | Saving the model
    os.makedirs('../saved_models', exist_ok=True)
    model_version = 'arch_' + str(arch) + '_median_5_cross_validation_split_' + str(split_num)
    model.save(os.path.join('../saved_models', 'model_' + model_version + '.h5'))

    # Zapisywanie historii treningu | Saving training history
    with open(os.path.join('saved_models', 'history_' + model_version + '.pkl'), 'wb') as file:
        pickle.dump(history.history, file)

    # Zapisywanie class names | Saving class names
    np.save(os.path.join('../saved_models', 'class_names.npy'), class_names)

    # Przygotowanie wykresów dokładności i straty | Preparing accuracy and loss plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(history.history['accuracy']))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Dokładność treningowa')
    plt.plot(epochs_range, val_acc, label='Dokładność walidacyjna')
    plt.legend(loc='lower right')
    plt.title('Dokładność treningowa i walidacyjna')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Strata treningowa')
    plt.plot(epochs_range, val_loss, label='Strata walidacyjna')
    plt.legend(loc='upper right')
    plt.title('Strata treningowa i walidacyjna')

    # Zapisywanie wykresów | Saving plots
    os.makedirs('../reports/plots', exist_ok=True)
    plt.savefig(os.path.join('../reports/plots',
                             'accuracy_loss_plot_' + model_version + '.png'))
    plt.close()
