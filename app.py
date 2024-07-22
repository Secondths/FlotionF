import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from flask import Flask, request, render_template, redirect, url_for
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load models and class names
knn = joblib.load('knn_model.pkl')
cnn = load_model('cnn_model.h5')
class_names = np.load('class_names.npy', allow_pickle=True)

# Constants
IMAGE_SIZE = (128, 128)
UPLOAD_DIR = './static'

# Flask app
app = Flask(__name__)

def predict_image(img):
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    
    # KNN Prediction
    img_flat = img_array.reshape(1, -1)
    knn_pred = knn.predict(img_flat)
    knn_label = class_names[knn_pred[0]]
    
    # CNN Prediction
    cnn_pred = cnn.predict(img_array)
    cnn_label = class_names[np.argmax(cnn_pred)]
    confidence = np.max(cnn_pred) * 100
    
    confidence = f"{confidence:.2f}"  # Format confidence hingga dua angka di belakang titik desimal
    
    return knn_label, cnn_label, confidence, knn_pred[0], np.argmax(cnn_pred)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(UPLOAD_DIR, 'uploaded_image.jpg')
            file.save(file_path)
            
            img = cv2.imread(file_path)
            knn_label, cnn_label, confidence, knn_pred, cnn_pred = predict_image(img)
            
            save_filename = f'classified_cnn_{cnn_label}.jpg'
            save_path = os.path.join(UPLOAD_DIR, save_filename)
            
            if os.path.exists(save_path):
                os.remove(save_path)
            
            os.rename(file_path, save_path)
            
            true_labels = [cnn_label]
            predicted_labels = [knn_label]
            cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
            
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cm_path = os.path.join(UPLOAD_DIR, 'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            
            return render_template('result.html', knn_label=knn_label, cnn_label=cnn_label, confidence=confidence, img_path=save_filename, cm_path='confusion_matrix.png')
    return render_template('upload.html')

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    app.run(debug=True)
