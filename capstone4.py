import streamlit as st

import numpy as np
import librosa

from moviepy.editor import VideoFileClip

import torch
from PIL import Image
import cv2
import os
import glob
import sys
from sklearn.metrics.pairwise import cosine_distances
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import pandas as pd

import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from keras.models import load_model

from facenet_models import FacenetModel

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logo1 = st.image("logo.PNG")
st.logo("logo.PNG")

st.title(f"Doorbell Camera Detection")
st.subheader(f"Dhir's Disciples")

video_file = st.file_uploader('Upload a video .mp4 file...', type = ['mp4'])

##############
#model code
def identify_agressive(audio_path):
    paths = []
    paths.append(get_features(audio_path))
    paths = Features.iloc[: ,:-1].values
    scaler = StandardScaler()
    audio_features = scaler.fit_transform(paths)
    audio_features = np.expand_dims(audio_features, axis=2)
    pred = reconstructed_model.predict(audio_features)
    pred = encoder.inverse_transform(pred)
    if "angry" in pred:
        print("Agressive behavior identified. Please review footage: ", audio_path)

model.save("audio_model_weights.keras")
# Define the input video file and output audio file
mp4_file = video_file.name
mp3_file = "sample3.wav"

# Load the video clip
video_clip = VideoFileClip(mp4_file)

# Extract the audio from the video clip
audio_clip = video_clip.audio

# Write the audio to a separate file
audio_clip.write_audiofile(mp3_file)

# Close the video and audio clips
audio_clip.close()
video_clip.close()
reconstructed_model = keras.models.load_model("audio_model_weights.keras")

sys.path.append('/Users/andrew/facenet_models/src')

# three emotion labels
emotion_labels = ['Angry', 'Happy', 'Neutral']

# emotion model
emotion_model = load_model('emotion_detection_model.h5')
audio_model = load_model('audio_model_weights.keras')

def get_boxes(results, min_confidence):
    boxes = results.xyxy[0].cpu().numpy()
    filtered_boxes = boxes[boxes[:, 4] > min_confidence]  #only if it is greater than conf
    return filtered_boxes

# draw the bounding boxes--only for boxes
def draw_boxes(image, boxes, labels, color):
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = labels[int(cls)] if int(cls) < len(labels) else 'Unknown'
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f'{label}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
# necessary transformations for emotion detection model
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # model's expected input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    img = preprocess_input(img)         # Preprocess image as needed
    return img
    
def detect_emotion(img):
    preprocessed_img = preprocess_image(img)
    emotion_predictions = emotion_model.predict(preprocessed_img)
    #max value
    predicted_emotion_index = np.argmax(emotion_predictions[0])
    #make sure it is in the bounds
    if 0 <= predicted_emotion_index < len(emotion_labels):
        emotion_label = emotion_labels[predicted_emotion_index]  #label
    else:
        emotion_label = "Neutral"  # if out of range
    return emotion_label
    
def process_frame(frame):
    if frame is None:
        print("Error: Frame is None")
        return frame
    
    emotion_label = detect_emotion(frame)  # Detect emotion in the frame
    
    # Draw the emotion label on the frame
    cv2.putText(frame, f'Emotion: {emotion_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

#facenet
facenet = FacenetModel()
#load both yolov5 models
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/andrew/yolov5/boxTrain/weights/best.pt')
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/andrew/yolov5/humanTrain/weights/best.pt')
#known faces load

known_faces = {}
known_faces_dir = '/Users/andrew/yolov5/doug/'

for image_path in glob.glob(known_faces_dir + '*.jpg'):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    boxes, probabilities, landmarks = facenet.detect(img_array)
    descriptors = facenet.compute_descriptors(img_array, boxes)
    if descriptors is not None and len(descriptors) > 0:
        name = os.path.basename(image_path).split('.')[0]
        known_faces[name] = descriptors[0]  # Assuming one face per image

unknown_faces_dir = '/Users/andrew/unknown_faces/'
os.makedirs(unknown_faces_dir, exist_ok=True)
unknown_face_counter = 1
unknown_faces = {}

def best_match(face_descriptor, face_db, threshold=0.4):
    min_cos = float('inf')
    min_name = "Unknown"
    for name, known_descriptor in face_db.items():
        distances = cosine_distances([face_descriptor], [known_descriptor])
        distance = np.mean(distances)
        if distance < min_cos:
            min_cos = distance
            min_name = name
    return min_name if min_cos < threshold else "Unknown"

#Video
video_path = '/Users/andrew/yolov5/newVideo1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
#vid props
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# codec video writter

output_path = '/Users/andrew/yolov5/newVideo2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi format
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# counters+frame thresh
package_detected = False
package_counter = 0
frame_threshold = 100

face_detected=False
frames_audio = []
faces = []

my_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
    
    if not ret:
        break
    img_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # detection for both models
    results1 = model1(img_cv)
    results2 = model2(img_cv)
    #get boxes
    boxes1 = get_boxes(results1, 0.48)
    boxes2 = get_boxes(results2, 0.45)
     #is package there?
    detected_package = len(boxes1) > 0
    if detected_package:
        if not package_detected:
            package_message = "You have received a package."
        package_detected = True
        package_counter = 0  #reset the coun
    else:
        if package_detected:
            package_counter += 1
            if package_counter > frame_threshold:
                package_message = "Your package has been retrieved."
                package_detected = False  # reset pack det bool
    
    if package_detected:
        cv2.putText(frame, "You have received a package.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    elif package_counter > frame_threshold:
        cv2.putText(frame, "Your package has been retrieved.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # boudning boxes for mod1
    draw_boxes(frame, boxes1, model1.names, (0, 255, 0))  # Green for model1
    # detect faces
    
    for box in boxes2:
        
        x1, y1, x2, y2, conf, cls = box
        face = img_cv[int(y1):int(y2), int(x1):int(x2)]
        boxes, probabilities, landmarks = facenet.detect(face)
        if boxes is not None and len(boxes) > 0:
            descriptors = facenet.compute_descriptors(face, boxes)
            
            face_detected=True
            if descriptors is not None and len(descriptors) > 0:

                name = best_match(descriptors[0], known_faces)

                if name not in faces:
                    faces.append(name)
                    frames_audio.append([])

                frames_audio[len(faces)-1].append(time)
                
                if name == "Unknown":
                    # Check if this unknown face is already in the unknown_faces dictionary
                    unknown_id = None
                    for u_name, u_descriptor in unknown_faces.items():
                        distances = cosine_distances([descriptors[0]], [u_descriptor])
                        if np.mean(distances) < 0.75:  # Check if this face matches an existing unknown face
                            unknown_id = u_name
                            break
                    if unknown_id is None:
                        # save to local device
                        unknown_face_path = os.path.join(unknown_faces_dir, f'unknown_{unknown_face_counter}.jpg')
                        cv2.imwrite(unknown_face_path, face)
                        
                        # add unkown face to unknown
                        unknown_name = f'unknown_{unknown_face_counter}'
                        unknown_faces[unknown_name] = descriptors[0]
                        unknown_face_counter += 1
                        name = "Unknown"
                    else:
                        name = "Unknown"

                # draw face bounding box and emotion
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red box for face
                (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                text_x = int(x1)
                text_y = int(y1) - 10
                text_x = max(text_x, 0)
                text_y = max(text_y, text_height + 10)
                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                emotion_label = detect_emotion(face)
                cv2.putText(frame, f'Emotion: {emotion_label}', (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)  # Yellow for emotion
  
    out.write(frame)

    #cv2.imshow('Combined Results', frame)
    my_placeholder.image(frame, use_column_width=True)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Define the input video files
mp3_files = ["/Users/andrew/zfinal/test_output_0.wav", "/Users/andrew/zfinal/test_output_1.wav", "/Users/andrew/zfinal/test_output_2.wav","AggresiveAudio.wav"]

# Define the sample rate
sample_rate = 44100

def extract_features(data):
    # Extract various features from the audio data
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(data)), sr=sample_rate).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    
    # Concatenate all features
    features = np.hstack((zcr, chroma_stft, mfcc, rms, mel))
    
    return features

def get_features(path):
    # Load audio data
    data, _ = librosa.load(path, duration=2.5, offset=0.6)
    
    # Extract features
    features = extract_features(data)
    
    # Ensure features have the correct size
    expected_size = 162
    if features.size < expected_size:
        features = np.pad(features, (0, expected_size - features.size), 'constant')
    elif features.size > expected_size:
        features = features[:expected_size]
    
    # Reshape to (1, 162, 1)
    features = np.expand_dims(features, axis=0)  # (1, 162)
    features = np.expand_dims(features, axis=-1)  # (1, 162, 1)
    
    return features

def identify_aggressive(audio_path, model):
    # Get features from audio file
    features = get_features(audio_path)
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    
    # Predict with the model
    pred = model.predict(features)
    
    # Interpret prediction
    # Assuming a binary classification with 'angry' as class 0
    if pred[0][0] <0.01:  # Adjust based on your model's output
        st.text("Aggressive behavior identified. Please review footage:", audio_path)
    else:
         st.text("No aggressive behavior identified.")

# Load the trained model
reconstructed_model = load_model("audio_model_weights.h5",compile=False)

# Process each audio file
for mp3_file in mp3_files:
    identify_aggressive(mp3_file, reconstructed_model)

################

#st.text("Here's what's really going on: ")

#video_file = open("file.mp4", "rb")
#video_bytes = video_file.read()

#st.video(video_bytes)

