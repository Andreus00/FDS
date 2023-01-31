import numpy as np
import cv2
import dlib
import mediapipe as mp
import utils

mp_face_detection = mp.solutions.face_detection


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


def detect_face(img):
    ret = None
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None, False
        # print(results.detections[0].location_data)
        x_min = max(int(results.detections[0].location_data.relative_bounding_box.xmin * img.shape[1]), 0)
        y_min = max(int(results.detections[0].location_data.relative_bounding_box.ymin * img.shape[0]), 0)
        width = int(results.detections[0].location_data.relative_bounding_box.width * img.shape[1])
        height = int(results.detections[0].location_data.relative_bounding_box.height * img.shape[0])
        ret = img[y_min:y_min + height, x_min: x_min + width]
    return ret, True





#capture video from webcam
cap = cv2.VideoCapture(0)

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = detector(gray)   
    face, success = detect_face(frame)
    if success:
        cv2.imshow("Face", face )

    # for face in faces:
    #     landmarks = predictor(gray, face)
    #     for n in range(0, 68):
    #         x = landmarks.part(n).x
    #         y = landmarks.part(n).y
    #         cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    pose, success = utils.get_pose(frame)

    if success:
        for point in pose.landmark:
            x = int(point.x * frame.shape[1])
            y = int(point.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # exit()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
