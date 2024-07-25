import os
import cv2
import numpy as np

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
def prepare_training_data(photo_folder):
    faces = []
    labels = []
    label_map = {}
    label_id = 0
    
    for person_name in os.listdir(photo_folder):
        person_path = os.path.join(photo_folder, person_name)
        if not os.path.isdir(person_path):
            continue
        
        label_map[label_id] = person_name
        
        for photo_name in os.listdir(person_path):
            photo_path = os.path.join(person_path, photo_name)
            image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            face = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in face:
                faces.append(image[y:y+h, x:x+w])
                labels.append(label_id)
        
        label_id += 1
    
    return faces, labels, label_map

# Load photos and prepare training data
photo_folder = 'photos'
faces, labels, label_map = prepare_training_data(photo_folder)

# Train the recognizer
recognizer.train(faces, np.array(labels))
recognizer.save('lbph_face_recognizer.xml')

# Save label map
import joblib
joblib.dump(label_map, 'label_map.pkl')

print("Model training complete and saved.")
