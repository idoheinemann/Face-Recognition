from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle as pkl
import os
import glob


class SVMRecognizer:
    def __init__(self, dataset):
        self.labels = LabelEncoder()
        self.classifier = SVC(C=1.0, kernel="linear", probability=True)
        self.dataset = dataset

    def train(self, unknown_dataset=None):
        dir_path = os.path.join('datasets', self.dataset)
        data = []
        labels = []
        for fname in glob.glob(os.path.join(dir_path, '*')):
            with open(fname, 'rb') as f:
                read_list = pkl.load(f)
                data += read_list
                labels += [os.path.split(fname)[-1]] * len(read_list)
        if unknown_dataset is not None:
            with open(unknown_dataset, 'rb') as f:
                read_list = pkl.load(f)
                data += read_list
                labels += ['unknown'] * len(read_list)
        enc_labels = self.labels.fit_transform(labels)
        self.classifier.fit(data, enc_labels)

    def to_file(self):
        with open(os.path.join('classifiers', self.dataset), 'wb') as f:
            pkl.dump(self, f)

    def train_and_save(self, unknown_dataset=None):
        self.train(unknown_dataset=unknown_dataset)
        self.to_file()

    @staticmethod
    def cycle(dataset=None, unknown_dataset=None):
        if dataset is None:
            dataset = input("Enter The Dataset Name: ")
        recognizer = SVMRecognizer(dataset)
        recognizer.train_and_save(unknown_dataset=unknown_dataset)

if __name__ == '__main__':
    SVMRecognizer.cycle()
