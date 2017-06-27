# FaceRecognition
Webcam face recognition using tensorflow and opencv.
The application tries to find faces in the webcam image and match them against images in an id folder using deep neural networks.

## Inspiration
Models, training code and inspriation can be found in the [facenet](https://github.com/davidsandberg/facenet) repository.
[Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) are used for facial and landmark detection while a [Inception Resnet](https://arxiv.org/abs/1602.07261) is used for ID classification.
A direct link to the pretrained Inception Resnet model can be found [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk).

## How to
Get model from [facenet](https://github.com/davidsandberg/facenet) and setup your id folder.
The id folder should contain subfolders, each containing images of one person. The subfolders should be named after the person in the folder since this name is used as output when a match is found.

Visualization hotkeys:
*    l - toggle facial landmarks
*    b - toggle bounding box
*    i - toggle id

![alt text](https://github.com/habrman/FaceRecognition/blob/master/example.png)