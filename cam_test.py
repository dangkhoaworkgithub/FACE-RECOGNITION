import cv2, os, torch, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from IPython.display import Image, clear_output
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imutils import paths
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from face_recognition import face_locations
from matplotlib import pyplot
import cv2
from time import sleep
from imutils.video import VideoStream
import imutils
import time
import cv2

def _save_pickle(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pickle(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj

id_train = _load_pickle("./id_train.pkl")
id_test = _load_pickle("./id_test.pkl")
embed_faces = _load_pickle("./embed_blob_faces.pkl")
y_labels = _load_pickle("./y_labels.pkl")
faces = _load_pickle("./faces.pkl")

faceResizes = []
for face in faces:
  face_rz = cv2.resize(face, (178, 218))
  faceResizes.append(face_rz)

X = np.stack(faceResizes)
X.shape



ids = np.arange(len(y_labels))

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(np.stack(embed_faces), y_labels, ids, test_size = 0.2, stratify = y_labels, random_state=42)
X_train = np.squeeze(X_train, axis = 1)
X_test = np.squeeze(X_test, axis = 1)
print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))

X_train, X_test = X[id_train], X[id_test]

print(X_train.shape)
print(X_test.shape)

# Set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)

from tensorflow import keras
model = keras.models.load_model('model\model_mlp_au.h5')

def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  return label

def _normalize_image(image, epsilon=0.000001):
  means = np.mean(image.reshape(-1, 3), axis=0)
  stds = np.std(image.reshape(-1, 3), axis=0)
  image_norm = image - means
  image_norm = image_norm/(stds + epsilon)
  return image_norm

# load face detection weights
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load gender detection model (Link model: https://www.kaggle.com/code/hongtrung/gender-classification-vgg16-and-fine-turning)
from tensorflow import keras
model = keras.models.load_model('model\model_mlp_au.h5')

# Read camara
camera = VideoStream(src=0).start()
time.sleep(1.0)

while (True):
    img = camera.read()
    img = imutils.resize(img, width=600)
    # Flip image
    img = cv2.flip(img, 1)
    faces = face_cascade.detectMultiScale(img, 1.2, 10,minSize=(100,100))

    for (x, y, w, h) in faces:
        try:
            face = img[y:y + h, x:x + w]
            face_rz = cv2.resize(face,  (178, 218))
            face_tf = _normalize_image(face_rz)
            face_tf = np.expand_dims(face_tf, axis = 0)
            vec = model.predict(face_tf)
            name = _most_similarity(X_train_vec, vec, y_train)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Name " + name, (x,y+h+30), fontface, fontscale, fontcolor ,2)
        except:
            print("Can't recognize face!")
            
    cv2.imshow("Picture", img)

    # Quit
    key = cv2.waitKey(1)
    if key==ord('q'):
        break

camera.stream.release()
cv2.destroyAllWindows()