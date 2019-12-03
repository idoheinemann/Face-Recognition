# Face Recognizer

A simple python implementation of face recognition

# ABOUT THE PROGRAM:

This program was designed to preform face recognition.
which means it can observe an image and detect all the faces, and then label the faces according to the people it knows.

The program is divided to datasets.
each dataset is a unique set of people with whom the program is familiar.
it is important to note that datasets are individual, and do not have access to other datasets.
all datasets are saved in the "datasets" folder.

the user is able to add new people to the database.
after the program is familiar with enough people the user will want to compile it in order to be able to use the data.
compiling a dataset is a process in which the program trains a model according to a dataset (i.e. learns the dataset) and produces a trained model which can be used to recognize the people in the database.
all trained models are saved in the "classifiers" folder.

in order to see the result of the trained models on real data, the user can use the camera to receive a live video stream, and process that stream using the model.
the result will be shown on a displaying screen.

**NOTE:** the program is only tested for python 3.7 and will probably only work with python 3


# HOW TO USE THE PROGRAM:

in the folder "run" there are 4 batch/bash files:
install_requirements -
	this file is used to install all required python 3 libraries
	it can be used by simply running the file
	it only needs to be run once
	it will install all required libraries and then close automatically

#### add_person - 
	this file is used to add a new person to a dataset
	in order to use this file, simply run it
	the program will ask you to input the new person's name and afterwards the dataset to which the new person will be added
	both must be legal file names!
	if no such dataset exists the program will create a new one
	the program will then connect to the camera and record the person until it got 100 shots of his face
	the person must stand in front of the camera until the window which has opened closes, and no one else can be in the frame
	it is recommended that the person will move his face around when the program is recording him, so that it will capture as much of his face as possible
	you can see whether the program is able to capture your face or not by seeing (or not seeing on failure) the blue rectangle in the opened window
	after it got it 100 shots the program will save the data and close automatically

#### compile_database -
	this file is used to compile a dataset and create a trained model
	in order to use this file, simply run it
	once run, the program will ask you to input the name of the dataset to compile
	make sure that the dataset exists and that you have added to it all the people you want in it!
	the program will learn the dataset and save the result
	after doing so, it will close automatically

#### test_compiled_database -
	this file is used to test and demonstrate a compiled dataset by opening a graphical window that displays a live video feed.
	in order to use this file, simply run it
	once run, the program will ask you to input the name of the dataset you wish to test or demonstrate
	make sure the dataset was already compiled!
	after you gave it the dataset name, the program will open a live video feed from the computer's webcam
	it will display the live stream on a window, and draw a rectangle around every face in the current image
	if the person is in the dataset, the program will write his name in the upper-left corner of the rectangle
	in order to close the program, simply press the "Q" button on your keyboard (make sure you are typing in english)
	NOTE: pressing the "x" button on the window will not close it
	NOTE: if a person is not in the dataset, the program will write the name of the most similar person who is in the database, this can be canceled by adding an unknown-person database to the dataset
