import numpy as np  # NumPy untuk operasi array
from sklearn.model_selection import train_test_split  # Fungsi untuk membagi dataset menjadi set pelatihan dan pengujian
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier dari Scikit-learn
from tensorflow.keras.models import Sequential  # Sequential model dari Keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Layer yang diperlukan untuk membangun model CNN
import joblib  # Untuk menyimpan dan memuat model

# Konstanta untuk ukuran gambar
IMAGE_SIZE = (128, 128)

# Memuat dataset
images = np.load('images.npy')  # Memuat gambar dari file numpy
labels = np.load('labels.npy')  # Memuat label dari file numpy
class_names = np.load('class_names.npy', allow_pickle=True)  # Memuat nama kelas dari file numpy

# Membagi dataset menjadi set pelatihan dan pengujian dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Melatih model KNN
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Meratakan gambar menjadi satu dimensi
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Meratakan gambar menjadi satu dimensi
knn = KNeighborsClassifier(n_neighbors=5)  # Membuat model KNN dengan 5 tetangga
knn.fit(X_train_flat, np.argmax(y_train, axis=1))  # Melatih model KNN menggunakan data pelatihan

# Menyimpan model KNN
joblib.dump(knn, 'knn_model.pkl')  # Menyimpan model KNN ke file

# Mendefinisikan model CNN dengan arsitektur yang terdiri dari layer konvolusi, pooling, flattening, dan dense
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),  # Layer konvolusi dengan 32 filter ukuran 3x3
    MaxPooling2D((2, 2)),  # Layer pooling dengan ukuran 2x2
    Conv2D(64, (3, 3), activation='relu'),  # Layer konvolusi dengan 64 filter ukuran 3x3
    MaxPooling2D((2, 2)),  # Layer pooling dengan ukuran 2x2
    Flatten(),  # Layer untuk meratakan output
    Dense(128, activation='relu'),  # Layer dense dengan 128 neuron dan aktivasi ReLU
    Dense(len(class_names), activation='softmax')  # Layer output dengan softmax untuk klasifikasi
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Mengompilasi model dengan optimizer Adam dan loss function categorical crossentropy
cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))  # Melatih model CNN dengan data pelatihan selama 10 epoch

# Menyimpan model CNN
cnn.save('cnn_model.h5')  # Menyimpan model CNN ke file
