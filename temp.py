import os
import pickle
import numpy as np

import tensorflow as tf
from keras import layers, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler

import matplotlib.pyplot as plt


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

# Odczytywanie rozmiaru danych
with open('image_dimensions.pkl', 'rb') as file:
    loaded_dimensions = pickle.load(file)

img_height = loaded_dimensions['img_height']
img_width = loaded_dimensions['img_width']

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

arch = 2
model = Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Conv2D(64, (5, 5), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (5, 5), activation="relu"),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Conv2D(256, (5, 5), activation="relu"),
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

# Wersja modelu
model_version = 'arch_' + str(arch)

# Tworzenie folderu modelu
plots_dir = 'plots_model_' + model_version
os.makedirs(plots_dir, exist_ok=True)

# Zapisywanie modelu
model.save(os.path.join(plots_dir, 'model_' + model_version + '.h5'))

# Zapisywanie historii treningu
with open(os.path.join(plots_dir, 'history.pkl'), 'wb') as file:
    pickle.dump(history.history, file)

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

# Zapisywanie wykresów
plt.savefig(os.path.join(plots_dir, 'accuracy_loss_plot.png'))

# Zapisywanie class_names
np.save('class_names.npy', class_names)
