import cv2
import numpy as np


class FaceDetector:
    def __init__(self, prototxt='face_detection_model/deploy.prototxt',
                 model='face_detection_model/res10_300x300_ssd_iter_140000.caffemodel', confidence=0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.confidence = confidence

    def get_rects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                start_x, start_y, end_x, end_y = box.astype(int)
                faces.append([start_x, start_y, end_x - start_x, end_y - start_y])
        return faces

    def get_faces(self, frame):
        return self.get_faces_by_rects(frame, self.get_rects(frame))

    @staticmethod
    def get_faces_by_rects(frame, rects):
        return list(map(lambda x: frame[x[1]:x[1] + x[3], x[0]:x[0] + x[2]], rects))