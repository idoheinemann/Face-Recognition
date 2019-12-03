from face_detector import FaceDetector
import os
import pickle as pkl
import cv2


class DataGenerator:
    def __init__(self, name, detector=None, embedder='face_detection_model/openface_nn4.small2.v1.t7'):
        self.name = name
        if detector is None:
            detector = FaceDetector()
        self.detector = detector
        self.embedder = cv2.dnn.readNetFromTorch(embedder)
        self.data = []

    def open_window(self):
        cv2.namedWindow(self.name)

    def add_data(self, frame, render=False):
        rects = self.detector.get_rects(frame)
        faces = self.detector.get_faces_by_rects(frame, rects)
        if len(faces) == 0:
            if render:
                cv2.imshow(self.name, frame)
                cv2.waitKey(1)
            return
        face = faces[0]
        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                          (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(face_blob)
        vec = self.embedder.forward()
        self.data.append(vec.flatten())
        if render:
            copy = frame.copy()
            rect = rects[0]
            cv2.rectangle(copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0))
            cv2.imshow(self.name, copy)
            cv2.waitKey(1)

    def close_window(self):
        cv2.destroyWindow(self.name)

    def to_file(self, dataset):
        dir_path = os.path.join('datasets', dataset)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        with open(os.path.join(dir_path, self.name), 'wb') as f:
            pkl.dump(self.data, f)

    @staticmethod
    def cycle(name=None, dataset=None, port=0, max_data=100):
        if name is None:
            name = input("Enter Your Name: ")
        if dataset is None:
            dataset = input("Enter The Dataset Name: ")
        vid = cv2.VideoCapture(port)
        gen = DataGenerator(name)
        ok, frame = vid.read()
        gen.open_window()
        while vid.isOpened() and ok and len(gen.data) <= max_data:
            gen.add_data(frame, render=True)
            ok, frame = vid.read()
        gen.close_window()
        gen.to_file(dataset)
        vid.release()


if __name__ == '__main__':
    DataGenerator.cycle()
