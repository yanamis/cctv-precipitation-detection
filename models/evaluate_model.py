import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import load_model
from keras import layers

import matplotlib.pyplot as plt


# Funkcja znajdująca błędnie sklasyfikowane obrazy | Function to find misclassified images
def find_images_info(model, test_dataset, class_names):
    all_images = []
    misclassified_images = []
    image_id = 0

    for images, labels in test_dataset:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = labels.numpy()

        for i in range(len(true_labels)):
            image_id += 1

            all_images.append(
                (image_id, class_names[predicted_labels[i]], class_names[true_labels[i]]))
            if predicted_labels[i] != true_labels[i]:
                image = images[i].numpy() * 255
                misclassified_images.append(
                    (image.astype("uint8"), class_names[predicted_labels[i]], class_names[true_labels[i]]))

    return all_images, misclassified_images


# Funkcja obliczająca statystyki błędnych klasyfikacji | Function to calculate misclassification stats
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


# Funkcja wyświetlająca błędnie sklasyfikowane obrazy | Function to display misclassified images
def display_misclassified_images(images, title):
    num_images = len(images)
    num_figures = (num_images - 1) // 12 + 1

    for f in range(num_figures):
        start_idx = f * 12
        end_idx = min((f + 1) * 12, num_images)
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))

        for i, (image, predicted_class, true_class) in enumerate(images[start_idx:end_idx]):
            ax = axs[i // 4, i % 4]
            ax.text(0, -5, 'Predicted: {}'.format(predicted_class), fontsize=10, color='black')
            ax.text(0, 290, 'True: {}'.format(true_class), fontsize=10, color='black')
            ax.imshow(image)
            ax.axis("off")

        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.show()


# Wczytywanie modelu | Loading model
model = load_model('saved_models/model_arch_3_median_5_cross_validation_split_1.h5')

# Wczytywanie class names | Loading class names
class_names = np.load('saved_models/class_names_arch_3_split_1.npy')

# Wczytywanie danych rozkładu elementów | Loading element distribution data
with open('dataset_preparation/data_element_distribution.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

data_root = loaded_data['data_root']

# Wczytywanie rozmiaru obrazów | Loading image dimensions
with open('dataset_preparation/image_dimensions.pkl', 'rb') as file:
    loaded_dimensions = pickle.load(file)

img_height = loaded_dimensions['img_height']
img_width = loaded_dimensions['img_width']

# Wczytywanie liczby próbek dla każdej fazy | Loading count_per_phase
with open('dataset_preparation/count_per_phase.pkl', 'rb') as file:
    count_per_phase = pickle.load(file)

batch_size = 32

# Tworzenie zbioru testowego | Creating test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_root, 'test', 'data'),
    batch_size=batch_size,
    image_size=(img_height, img_width),
    seed=123
)

# Normalizacja obrazów | Normalizing images
normalization_layer = layers.Rescaling(1. / 255)
normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Ewaluacja na zbiorze testowym | Evaluation on test set
test_loss, test_accuracy = model.evaluate(normalized_test_ds)
print("Dokładność testowa: {}".format(test_accuracy))

# Znalezienie błędnie sklasyfikowanych obrazów | Finding misclassified images
all_images, misclassified_images = find_images_info(model, normalized_test_ds, class_names)

# Obliczanie statystyk błędów klasyfikacji | Calculating misclassification stats
calculate_misclassification_stats(misclassified_images, class_names, count_per_phase)

# Wyświetlanie błędnie sklasyfikowanych obrazów | Displaying misclassified images
display_misclassified_images(misclassified_images, "Źle sklasyfikowane obrazy")

# Tworzenie DataFrame z informacjami o obrazach | Creating DataFrame with image information
df = pd.DataFrame(all_images, columns=["ID", "Etykieta przewidywana", "Etykieta prawdziwa"])

# Zapisywanie wyników do pliku Excel | Saving results to Excel file
os.makedirs('../reports/metrics', exist_ok=True)
excel_path = os.path.join('../reports/metrics', 'all_images_info_model_arch_3_median_5_cross_validation_split_1.xlsx')
df.to_excel(excel_path, index=False)
