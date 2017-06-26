# FaceRecognition
Webcam face recognition using tensorflow an opencv.
The application tries to find faces in the webcam image and match them against images in an id folder using deep neural netwoks.

## Inspiration
Models and training code can be found in the [facenet](https://github.com/davidsandberg/facenet) repository.

## How to
Get model from [facenet](https://github.com/davidsandberg/facenet) and setup your id folder.
The id folder should contain subfolders, each containing images of one person. The subfolders should be named after the person in the folder since this name is used as output when a match is found.