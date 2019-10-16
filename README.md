# WebCam Face Recognition using FaceNet 

A simple face recognition implementation using a pre-trained, one-shot learning model - FaceNet. Classification on custom dataset by using the WebCam to perform live face recognition. The inspiration is from the task provided by Digiledge. The code is credited to David Sandberg for his facenet repository.

## Steps to follow:

Clone the repository //Insert link of my repository//. Set Python path as mentioned here. 
1. Create a folder named ~/datasets and insert images into the folder in the following format. 
```
root folder 
    │
    └───before
        │
        └───Person 1
        │   │───IMG1
        │   │───IMG2
        │   │   ....
        └───Person 2
        |   │───IMG1
        |   │───IMG2
        |   |   ....
    │
    └───after
```
Eg.
```
datasets
    │
    └───before
        │
        └───Alice
        │   │───Alice1.jpg
        │   │───Alice2.jpg
        │   │   ....
        └───Bob
        |   │───Bob1.jpg
        |   │───Bob2.jpg
        |   |   ....
    │
    └───after
```
These are the images that will be used to train the classifier and perform face recognition on the dataset by using WebCam. 

2. Align the faces using MTCNN or dllib. 

To extract useful information from the images in terms of bounding boxes around the face, run the script align_dataset_mtcnn.py at lib/src/align/ (sample command is given in the script) to generate the aligned images from the ~/datasets/before/ folder. 

[Before alignment](https://github.com/akshayraghavan21/Face_Recognition_Using_Facenet/blob/master/Static/MVIMG_20191005_195238_0.png)   [After alignment] <img src="https://github.com/akshayraghavan21/Face_Recognition_Using_Facenet/blob/master/Static/MVIMG_20191005_195238_0.png"  width="250" height="250" /> 

3. Download [pre-trained-weight](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) ,extract and keep it in lib/src/ckpt folder (for detailed info about availabe weights: [available-weights](https://github.com/davidsandberg/facenet#pre-trained-models)) 

4. Classifier Training/Testing

  4.1 Train a classifier: Train a Classifier on the aligned images using the script classifier.py at lib/src/ which fill create a classifier pickle file at lib/src/ckpt/ which can be used for detection faces using custom dataset. (Refer to the sample script in the classifier.py file)

  4.2 Test the classifier: Once the training is done, use aligned images, which werent provided while training, to test the classifier. 
    Steps:
      Store new custom aligned images in a new folder at ~/datasets/. This folder path is provided as an argument to the Classify command sample provided in the classifier.py file. 
      
 
5. WebCam Detection of face - 

Run the script to start up your webcam and detect face. 
```
  python lib/src/liveweb.py
```
 
 ## sample result 
 
 ![alt text](https://github.com/akshayraghavan21/Face_Recognition_Using_Facenet/blob/master/Static/Screen%20Shot%202019-10-16%20at%204.16.41%20AM.png)
  


References:

* Deepface paper https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf
* Facenet paper https://arxiv.org/pdf/1503.03832.pdf
* Pre-trained facenet model https://github.com/davidsandberg/facenet

