# Live-face-recognition-with-Attendance-Marking-System-
Live face recognition with Attendance Marking System  - This project uses Python and  OpenCV library  to detect faces with the help of Pre- Trained models and Marks attendance from the live captured videos.


Process involves:

the library openCV uses HAAR cascade classifier which is a ML based object detection techniques to detect faces. 
The library face_recognition is used to measure face distance, face locations, mouth patterns etc..

Initially The project works by collecting video captures from live webcam. 
after capturing, openCV uses sliding window method and thresholds to find faces in the video capture with location as coordinates.The found face is usually in BGR format, we need to convert it into RGB format as the OpenCV only process the RGB format. the Formatted RGB image is then encoded with the help of facial features.(which is actually 128 measurements)

We already created a folder with images of students/persons for model training. we have to process and encode these model images to detect the video captured face. the encoded values of trained model is stored.

After training, we need to test it, for that we use webcam to capture video. Once the model found the face in live video capture, it converts the image from BGR to RGB and then it encodes the image and it will return the encoded value. The model compares the encoded value with training object values. 
The lower the difference, the higher the chance of finding the match.
so, if the difference is low, then it will return the name of the image which is the name of student or person.

with the help of this, we will mark attendance in excel sheet with their names and the time they arrived.
