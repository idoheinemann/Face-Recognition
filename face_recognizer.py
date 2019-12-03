import cv2
import pickle as pkl
import numpy as np
import os
from svm_recognizer import SVMRecognizer
from face_detector import FaceDetector


class FaceRecognizer:
    def __init__(self, dataset, embedder='face_detection_model/openface_nn4.small2.v1.t7'):
        self.embedder = cv2.dnn.readNetFromTorch(embedder)
        self.recognizer = pkl.load(open(os.path.join('classifiers', dataset), 'rb'))

    def get_names_from_rects(self, frame, rects):
        return self.get_names_from_faces(FaceDetector.get_faces_by_rects(frame, rects))

    def get_names_from_faces(self, faces):
        names = []
        for face in faces:
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                              (0, 0, 0), swapRB=True, crop=False)
            self.embedder.setInput(face_blob)
            vec = self.embedder.forward()

            # perform classification to recognize the face
            preds = self.recognizer.classifier.predict_proba(vec)[0]
            j = np.argmax(preds)
            # prob = preds[j]
            name = self.recognizer.labels.classes_[j]
            names.append(name)
        return names
