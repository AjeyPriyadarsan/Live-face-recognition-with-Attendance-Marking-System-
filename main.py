import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the directory containing images for attendance
image_path = 'ImagesAttendance'
image_list = []
class_names = []
image_files = os.listdir(image_path)
print(image_files)

# Load images and extract class names
for image_file in image_files:
    current_image = cv2.imread(f'{image_path}/{image_file}')
    image_list.append(current_image)
    class_names.append(os.path.splitext(image_file)[0])
print(class_names)

# Function to find face encodings in a list of images
def find_encodings(images):
    encoding_list = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(image)[0]
        encoding_list.append(encoding)
    return encoding_list

# Function to mark attendance
def mark_attendance(name):
    with open('Attendance.csv', 'r+') as file:
        data_list = file.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            file.writelines(f'\n{name},{time_string}')

# Encode known faces
known_encodings = find_encodings(image_list)
print('Encoding Complete')

# Open the webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    # frame = capture_screen()
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        # print(face_distances)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            detected_name = class_names[match_index].upper()
            # print(detected_name)
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, detected_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(detected_name)

    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)
