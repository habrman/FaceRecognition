# FaceRecognition
Webcam face recognition using tensorflow and opencv.
The application tries to find faces in the webcam image and match them against images in an id folder using deep neural networks.

## Dependencies
*   OpenCv
*   Scipy
*   Tensorflow
*   Scikit-learn

## Inspiration
Models, training code and inspriation can be found in the [facenet](https://github.com/davidsandberg/facenet) repository.
[Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) are used for facial and landmark detection while an [Inception Resnet](https://arxiv.org/abs/1602.07261) is used for ID classification.
A direct link to the pretrained Inception Resnet model can be found [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk).

## How to
Get the [model from facenet](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) and setup your id folder.
The id folder should contain subfolders, each containing at least one image of one person. The subfolders should be named after the person in the folder since this name is used as output when a match is found.

E.g. id folder named `ids` containing subfolders `Adam` and `Eve`, each containing images of the respective person.

```bash
├── ids
│   ├── Adam
│   │   ├── Adam0.png
│   │   ├── Adam1.png
│   ├── Eve
│   │   ├── Eve0.png
```
Download and unpack the [model](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) to a folder and run `python3 main.py ./folder/model.pb ./ids/` to start the program. Make sure to replace `./folder/model.pb` with the path to the downloaded model.

Visualization hotkeys:
*   l - toggle facial landmarks
*   b - toggle bounding box
*   i - toggle id
*   f - toggle frames per second

![alt text](https://github.com/habrman/FaceRecognition/blob/master/example.png)