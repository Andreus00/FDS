import numpy as np
import cv2
import dlib


def crop_face(image, rects):
    faces = []
    for rect in rects:
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        w = x2 - x1
        h = y2 - y1
        #add padding to the face
        padding = 0.4
        x1 = int(x1 - w * padding)
        x2 = int(x2 + w * padding)
        y1 = int(y1 - h * padding)
        y2 = int(y2 + h * padding)


        faces.append(image[y1:y2, x1:x2])
    return faces[0]



#capture video from webcam
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)        
    if len(faces) > 0:
        faceWindow = crop_face(frame, faces)
        faceWindow = cv2.resize(faceWindow, (250, 250))
        gray = cv2.cvtColor(faceWindow, cv2.COLOR_BGR2GRAY)
        face = detector(gray)
        if len(face) > 0:
            ld = predictor(gray, face[0])
            for n in range(0, 68):
                x = ld.part(n).x
                y = ld.part(n).y
                cv2.circle(faceWindow, (x, y), 4, (0, 255, 0), -1)

        cv2.imshow("Face", faceWindow )

    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
