#Finally run this script to use the webcam to detect faces
#Sample Script - 
#     python liveweb.py
#

import cv2
import numpy as np
import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
# from inception_blocks_v2 import *
import sys
import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections


def main():
    
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = './ckpt/lfw_classifier.pkl'
    FACENET_MODEL_PATH = './ckpt/20180408-102900/20180408-102900.pb'
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    #Load Classfier - a pickle file
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file,encoding="latin1")
        print("Custom Classifier, Successfully loaded") 


    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            #Align the images
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

            people_detected = set()
            person_detected = collections.Counter()
            video_capture = cv2.VideoCapture(0)
            while True:
                ret, frame = video_capture.read()
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))


                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20  

                            if best_class_probabilities > 0.15:
                                name = class_names[best_class_indices[0]]
                            else:
                                name = "Unknown"
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1

                except:
                        pass
                cv2.imshow('Face Recognition',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            video_capture.release()
            cv2.destroyAllWindows()
main()