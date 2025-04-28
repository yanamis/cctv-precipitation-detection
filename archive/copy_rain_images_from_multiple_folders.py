from copy_no_rain_images_from_multiple_folders import copy_images_from_multiple_folders

source_folders = [
    'C:\\Users\\yanam\\opady_dataset\\SPAC-SupplementaryMaterials-master\\Dataset_Testing_RealRain\\ra1_Rain'
]

destination_folder = 'C:\\Users\\yanam\\opady_dataset\\dataset\\opady_spac'

copy_images_from_multiple_folders(source_folders, destination_folder, start_index=1)
