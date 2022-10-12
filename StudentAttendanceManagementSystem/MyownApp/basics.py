import cv2
import numpy as np
import face_recognition

# load images and convert them into RGB
# we're getting images as pgr,but the library understand it as RGB

# first step :import our image first

imgSeipati = face_recognition.load_image_file('Images/Seipatis_Pic.jpg')
# second step  in step 1 convert image into RGB
imgSeipati = cv2.cvtColor(imgSeipati, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('Images/TestImage4.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# finding faces in our image and their enconding as well
# first detect the face
faceLoc = face_recognition.face_locations(imgSeipati)[0]
# encode face detected
encodeSeipati = face_recognition.face_encodings(imgSeipati)[0]
# See where we detected faces
cv2.rectangle(imgSeipati, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeSeipati], encodeTest)
faceDis = face_recognition.face_distance([encodeSeipati], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results}{round(faceDis[0], 2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX ,1,(0,0,255),2)

cv2.imshow('Seipati', imgSeipati)
cv2.imshow('Seipati Test', imgTest)
cv2.waitKey(0)
