import os  # Modul untuk mengelola operasi sistem seperti navigasi direktori
import cv2  # OpenCV untuk pemrosesan gambar
import numpy as np  # NumPy untuk operasi array
from tensorflow.keras.utils import to_categorical  # Mengonversi label menjadi one-hot encoding

# Konstanta untuk ukuran gambar dan path ke dataset
IMAGE_SIZE = (128, 128)
DATASET_PATH = r'E:\Pembelajaran_mesin\Flotion\datasets\train'

def load_dataset(dataset_path):
    images = []  # List untuk menyimpan gambar
    labels = []  # List untuk menyimpan label
    class_names = os.listdir(dataset_path)  # Mendapatkan daftar nama kelas dari direktori
    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}  # Kamus untuk memetakan nama kelas ke indeks numerik

    for class_name in class_names:  # Iterasi melalui setiap kelas di dataset
        class_path = os.path.join(dataset_path, class_name)  # Path ke direktori masing-masing kelas
        for img_name in os.listdir(class_path):  # Iterasi melalui setiap gambar di direktori kelas
            img_path = os.path.join(class_path, img_name)  # Path lengkap ke gambar
            img = cv2.imread(img_path)  # Membaca gambar dari file
            img = cv2.resize(img, IMAGE_SIZE)  # Mengubah ukuran gambar menjadi (128x128)
            images.append(img)  # Menambahkan gambar ke list `images`
            labels.append(class_dict[class_name])  # Menambahkan label ke list `labels`

    images = np.array(images)  # Mengonversi list gambar menjadi array NumPy
    labels = to_categorical(np.array(labels))  # Mengonversi list label menjadi one-hot encoded array
    
    return images, labels, class_names  # Mengembalikan gambar, label, dan nama kelas

if __name__ == "__main__":
    images, labels, class_names = load_dataset(DATASET_PATH)  # Memuat dataset
    np.save('images.npy', images)  # Menyimpan gambar ke file numpy
    np.save('labels.npy', labels)  # Menyimpan label ke file numpy
    np.save('class_names.npy', class_names)  # Menyimpan nama kelas ke file numpy
