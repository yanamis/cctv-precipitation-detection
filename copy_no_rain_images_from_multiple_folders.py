import os
import shutil


def copy_images_from_multiple_folders(source_folders, destination_folder, start_index):
    for source_folder in source_folders:
        for folder, _, files in os.walk(source_folder):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    source_path = os.path.join(folder, file)
                    new_index = start_index
                    start_index += 1
                    destination_path = os.path.join(destination_folder, 'img_{}.jpg'.format(new_index))
                    shutil.copyfile(source_path, destination_path)


source_folders = [
    'C:\\Users\\yanam\\opady_dataset\\SPAC-SupplementaryMaterials-master\\Dataset_Testing_Synthetic\\a1_GT',
]

destination_folder = 'C:\\Users\\yanam\\opady_dataset\\dataset\\brak_spac'

# Podać listę folderów źródłowych i ścieżkę do foldera docelowego oraz początkowy indeks
copy_images_from_multiple_folders(source_folders, destination_folder, start_index=1)
