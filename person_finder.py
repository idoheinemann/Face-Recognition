from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from svm_recognizer import SVMRecognizer
import cv2


class PersonFinder:
    def __init__(self, dataset):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer(dataset)
        self.dataset = dataset

    def find_all(self, frame):
        rects = self.detector.get_rects(frame)
        labels = self.recognizer.get_names_from_rects(frame, rects)
        return list(zip(labels, rects))

    def open_window(self):
        cv2.namedWindow(self.dataset)

    def close_window(self):
        cv2.destroyWindow(self.dataset)

    def render(self, frame):
        copy = frame.copy()
        for name, rect in self.find_all(frame):
            y = rect[1] - 10 if rect[1] - 10 > 10 else rect[1] + 10
            cv2.rectangle(copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
            cv2.putText(copy, name, (rect[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow(self.dataset, copy)
        return chr(cv2.waitKey(1) & 0xFF)

    @staticmethod
    def cycle(dataset=None, port=0):
        if dataset is None:
            dataset = input("Enter The Dataset Name: ")
        finder = PersonFinder(dataset)
        vid = cv2.VideoCapture(port)
        ok, frame = vid.read()
        finder.open_window()
        while vid.isOpened() and finder.render(frame) not in "qQ":
            ok, frame = vid.read()
        finder.close_window()
        vid.release()
        
if __name__ == '__main__':
    PersonFinder.cycle()