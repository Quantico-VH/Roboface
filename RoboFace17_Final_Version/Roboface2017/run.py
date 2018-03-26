# -*- coding: utf-8 -*-
# -*- python version: 3.** -*-

"""
Running the trained neural network, normalise image and control RoboFace.
Copyright (C) 2017 Letitia Parcalabescu
Modified by (2018): Viacheslav Honcharenko, Athanasios Raptakis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
import numpy as np
from keras.models import load_model
# from scipy import misc
from scipy.misc import imresize
from skimage.transform import resize, rotate
import h5py
import math
import face
from gtts import gTTS
from pygame import mixer, time
import os, subprocess, signal, psutil
from threading import Thread, Event
from time import sleep, time
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d, gaussian_filter
from nltk.tokenize import sent_tokenize
import string


class Robot:
    def __init__(self):
        self.image

        self.face = face.Face(x_weight=0.8, y_weight=0.2)
        # Set Speed for smoother movement
        self.face.setSpeedAll(100)
        self.face.setSpeedHead(80)
        self.face.neutral()
        self.face.setSpeedLips(127)

        self.talk_flag = Event()
        self.talk_flag.clear()

        self.model = load_model('../face_detection/trained/pretrained_CelebA_normalised0203-05.h5')

        self.video_capt = cv2.VideoCapture(1)  # 0 for built-in webcam, 1 for robot

        if self.video_capt.isOpened():  # try to get the first frame
            self.rval, self.image = self.video_capt.read()
        else:
            self.rval = False
            self.say("Sorry, I cannot see you.")

    ################################################################################
    # Declaration of: Constants
    ################################################################################
    IMAGE_SIZE = (128, 128)
    IOD = 40.0

    ################################################################################
    # End of Declaration: Constants
    ################################################################################


    ################################################################################
    # Declaration of: Class variables
    ################################################################################
    image = None
    normalised_image = None
    talk_flag = None
    face = None
    model = None
    rval = None
    probStream = None
    video_capt = None
    video_window_name = ""

    ################################################################################
    # End of Declaration: Class variables
    ################################################################################

    def __imgCrop(self, _image, cropBox, boxScale=1):
        '''
        Crop an area around the detected face (by OpenCV) in order to feed it into the prediction algorithm (NN).
        '''
        off = 90
        y = max(cropBox[1] - 3 * off, 0)
        x = max(cropBox[0] - 2 * off, 0)

        off = 50
        y = max(cropBox[1] - 3 * off, y)
        x = max(cropBox[0] - 2 * off, x)

        off = 20
        y = max(cropBox[1] - 3 * off, y)
        x = max(cropBox[0] - 2 * off, x)

        cropped = _image[y:cropBox[1] + cropBox[3] + 90, x:cropBox[0] + cropBox[2] + 30]
        dims = cropped.shape

        return cropped, x, y

    def __rotateBound(self, _image, angle, center):
        '''
        Rotates image. Used for image normalisation, so that the inter-ocular line is always horizontal for the NN.
        '''
        (cX, cY) = center
        (h, w) = _image.shape[:2]

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(_image, M, (nW, nH))

    def __normaliseImage(self, image, eyes, xcrop, ycrop):
        '''
        Normalize faces usinginter-ocular distance i.o.d
        '''
        # resite, such that i.o.d is always same
        left_eye = eyes[0] + np.array([xcrop, ycrop, 0, 0])
        right_eye = eyes[1] + np.array([xcrop, ycrop, 0, 0])
        scale = self.IOD / np.linalg.norm(left_eye - right_eye)
        left_eye = scale * left_eye
        right_eye = scale * right_eye
        im = resize(image, (int(scale * image.shape[0]), int(scale * image.shape[1])), mode='edge')

        # rotate to keep inter ocular line horizontal
        diff = np.subtract(left_eye, right_eye)
        angle = math.atan2(diff[0], diff[1])
        im = rotate(im, -angle, center=(left_eye[0], left_eye[1]), preserve_range=True, mode='edge')

        # new resizing for making the image compatible with the trained NN.
        iod = np.linalg.norm(left_eye - right_eye)
        xmin = int(left_eye[0] - 1.6 * iod)
        xmax = int(left_eye[0] + 2 * iod)
        ymin = int(left_eye[1] - 1.3 * iod)
        ymax = int(right_eye[1] + 1.3 * iod)
        xmin = max(0, xmin)
        xmax = min(im.shape[0], xmax)
        ymin = max(0, ymin)
        ymax = min(im.shape[1], ymax)
        im = im[xmin:xmax, ymin:ymax, :]
        try:
            im = resize(im, self.IMAGE_SIZE, mode='edge')
        except:
            return None

        return im

    def detectFace(self):
        # http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

        face_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_smile.xml')

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # for each detected face, detect eyes and smile
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        unaltered_image = self.image.copy()
        eyes = None
        self.normalised_image = None
        for face in faces:
            (x, y, w, h) = face
            # show face bounding box on Webcam Preview
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = self.image[y:y + h, x:x + w]

            # normalise image in order to predict on it
            # croppedImage = imgCrop(image, face, boxScale=1)
            # detect eyes for Inter Oculat Distance
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 2:
                left_eye = eyes[0][0:2] + x
                right_eye = eyes[1][0:2] + y
                eyex = int((left_eye[0] + right_eye[0]) * .5)
                eyey = int((left_eye[1] + right_eye[1]) * .5)
                self.face.moveHead(eyex, eyey)
                # suggestion: skip this frame as prediction, so return None, image
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                if len(eyes) == 2 and np.abs(eyes[0, 1] - eyes[1, 1]) < 10:
                    offset1 = np.sqrt((eyes[0, 2] ** 2 + eyes[0, 3] ** 2)) * 0.5
                    offset2 = np.sqrt((eyes[1, 2] ** 2 + eyes[1, 3] ** 2)) * 0.5
                    real_eyes = eyes + np.array([[x + offset1, y + offset1, 0, 0], [x + offset2, y + offset2, 0, 0]])
                    real_eyes = np.sort(real_eyes, axis=0)
                    cropped_image, xcrop, ycrop = self.__imgCrop(unaltered_image, face)
                    self.normalised_image = self.__normaliseImage(cropped_image, real_eyes, -xcrop, -ycrop)

    def mapAttributes(self, classes):
        '''
        Map the output probabilities to the correpsonding names, like 'smile', etc.
        '''
        with open('../face_detection/wanted_attributes_normalised.txt', 'r') as f:
            attributes = f.read()
        attributes = attributes.strip('\n').split(' ')

        result = []
        for i, cl in enumerate(classes):
            if cl == True:
                result.append(attributes[i])
        return result

    ################################################################################
    # Declaration of: say - Talk - MoveLips
    ################################################################################

    def __undersampled_lip_tragectory(self, phrase, Sleep_Time):
        A = "espeak -z -s 80 -v female5 -w temp_phrase.wav "
        A = A + "'" + phrase + "'"
        os.system(A)
        samplerate, data = wavfile.read('temp_phrase.wav')
        dt = 1 / float(samplerate)
        times = np.arange(len(data)) / float(samplerate)
        N = len(times)
        max_data = maximum_filter1d(data, size=1000)
        max_data = gaussian_filter(max_data, sigma=100)
        max_Amplitude = 10
        Amplitude = max_Amplitude * (max_data / float(np.max(max_data)))
        n = Sleep_Time * samplerate
        Amp = []
        T = []
        i = 0
        while (i * n < N):
            Amp.append(Amplitude[int(i * n)])
            T.append(times[int(i * n)])
            i = i + 1
        Amp = np.array(Amp)
        T = np.array(T)

        return Amp, T

    def __moveLips(self, Sleep_Time, Amplitude):
        i = 0
        while self.talk_flag.isSet() and i < len(Amplitude):
            self.face.moveLips(int(Amplitude[i]))
            sleep(Sleep_Time)
            i = i + 1
        if ~self.talk_flag.isSet():
            self.face.moveLips(0)
            sleep(0.05)

    def __talk(self, phrase):
        A = "espeak -z -s 80 -v female5 "
        A = A + "'" + phrase + "'"
        os.system(A)
        self.talk_flag.clear()

    def say(self, _input):
        phrases = sent_tokenize(_input)
        for phrase in phrases:
            phrase = phrase.replace("'", " ")
            self.talk_flag.set()
            Sleep_Time = 0.05
            Amplitude, Time = self.__undersampled_lip_tragectory(phrase, Sleep_Time)
            thread_movement = Thread(target=self.__moveLips, args=(Sleep_Time, Amplitude))
            thread_talk = Thread(target=self.__talk, args=( phrase, ))
            thread_talk.start()
            thread_movement.start()

            while self.talk_flag.isSet():
                self.detectFace()
                self.probStream = None
                self.refresh_image()


    ################################################################################
    # End of Declaration: say - Talk - MoveLips
    ################################################################################


    def sayDoSomething(self, pred_attr):
        talk = {'Smiling': 'I like it when people smile at me!',
                'Female': 'You are a female, am I right?',
                'Male': 'You are a male, am I right?',
                'Wearing_Earrings': 'You are wearing beautiful earrings today!',
                'Wearing_Lipstick': 'I see you are wearing lipstick today. Pretty!',
                'Blond_Hair': 'Nice blond hair!',
                'Eyeglasses': 'You are wearing eyeglasses!',
                'Brown_Hair': 'You have nice brown hair!',
                'Black_Hair': 'You have nice black hair!',
                'Gray_Hair': 'You must be a wise man, judging by your gray hair!',
                'Wavy_Hair': 'You have nice wavy hair!',
                'Straight_Hair': 'You have nice straight hair.'
                }
        # talk = {'Smiling': 'Es ist so schön, wenn mich Menschen anlächeln!',
        #        'Female': 'Bist du eine Frau, oder?',
        #        'Male': 'Bist du denn ein Mann?',
        #        'Wearing_Earrings': 'Du trägst wunderschöne Ohrringe!',
        #        'Wearing_Lipstick': 'Ich merke, dass du Lippenstift trägst. Schön!',
        #        'Blond_Hair': 'Schöne blonde Haare hast du!',
        #        'Eyeglasses': 'Du trägst Brillen.',
        #        'Brown_Hair': 'Du hast schöne braune Haare!',
        #        'Black_Hair': 'Du hast wuunderschöne schwarze Haare!',
        #        'Gray_Hair': 'Du musst eine sehr weise Person sein, da du graue Haare hast.',
        #        'Wavy_Hair': 'Du hast schöne lockige Haare!',
        #        'Straight_Hair': 'Du hast schöne glatte Haare!'
        #        }

        if 'Smiling' in pred_attr:
            self.face.happy(moveHead=False)
        elif 'Black_Hair' in pred_attr:
            self.face.angry(moveHead=False)
        elif 'Eyeglasses' in pred_attr:
            self.face.unsure(moveHead=False)
        else:
            self.face.neutral(moveHead=False)

        index = np.random.randint(0, len(pred_attr))
        self.say(talk[pred_attr[index]])

    def getProbaStream(self, probs):
        if self.probStream == None:
            self.probStream = probs
        else:
            self.probStream = np.vstack((self.probStream, probs))

    def refresh_image(self):
        cv2.imshow(self.video_window_name, self.image)
        self.rval, self.image = self.video_capt.read()


if __name__ == "__main__":

    robo = Robot()
    robo.video_window_name = "Webcam Preview"
    cv2.namedWindow(robo.video_window_name)
    process = None
    saidNothing = 0

    while robo.rval:
        robo.detectFace()

        # if a face is detected and the normalisation was successful, predict on it
        if robo.normalised_image is not None:
            robo.normalised_image = robo.normalised_image[:, :, ::-1]
            # subtract mean face
            meanFace = np.load('../face_detection/mean_face_normalised.npy')

            X_test = np.expand_dims(robo.normalised_image, axis=0)
            X_test -= meanFace
            classes = robo.model.predict_classes(X_test, batch_size=32, verbose=0)
            probs = robo.model.predict_proba(X_test, batch_size=32, verbose=0)
            # pred_attr = mapAttributes((proba > 0.6)[0])
            # print( proba)
            # print(pred_attr)

            robo.getProbaStream(probs)
            if saidNothing == 0 and robo.probStream.shape[0] < 10:
                saidNothing += 1
                robo.refresh_image()
                key = cv2.waitKey(20)
                if key == 27:  # exit on ESC
                    if process != None:
                        os.kill(process.pid, signal.SIGTERM)
                    robo.say("I'm sorry Dave. I'm afraid I can't do that.")
                    while robo.talk_flag.isSet():
                        sleep(10)
                    break
            elif robo.probStream.shape[0] > 10 and len(robo.probStream.shape) >= 2:
                if process != None:
                    os.kill(process.pid, signal.SIGTERM)
                    process = None
                meanProbs = np.mean(robo.probStream, axis=0)
                pred_attr = robo.mapAttributes(meanProbs > 0.6)

                best = []
                if meanProbs[0] > meanProbs[1] and meanProbs[0] > meanProbs[4] and meanProbs[0] > meanProbs[2]:
                    best.append('Black_Hair')
                elif meanProbs[1] > meanProbs[0] and meanProbs[1] > meanProbs[4] and meanProbs[1] > meanProbs[2]:
                    best.append('Blond_Hair')
                elif meanProbs[2] > meanProbs[0] and meanProbs[2] > meanProbs[1]:
                    best.append('Brown_Hair')
                if meanProbs[9] < meanProbs[10]:
                    best.append('Straight_Hair')
                else:
                    best.append('Wavy_Hair')
                if meanProbs[3] > 0.6:
                    best.append('Eyeglasses')
                if meanProbs[8] > 0.6:
                    best.append('Smiling')
                if meanProbs[11] > 0.2:
                    best.append('Wearing_Earrings')
                if meanProbs[12] > 0.2:
                    best.append('Wearing_Lipstick')
                # if meanProbs[12] > 0.11 and meanProbs[11] > 0.11 and meanProbs[5] < 0.6:
                if meanProbs[5] < 0.25:
                    best.append('Female')
                elif meanProbs[12] < 0.11 and meanProbs[11] < 0.11 and meanProbs[5] > 0.85:
                    best.append('Male')
                print(meanProbs)
                print("BEST", best)

                # end NN stuff

                # postprocessing and reaction step
                robo.sayDoSomething(best)
                saidNothing = 0
        elif saidNothing > 100:
            saidNothing = 0
            robo.face.sad()
            robo.say("Hey, why is no one looking at me? I feel neglected. I feel it. I feel it! I am afraid!")
            if process == None:
                process = subprocess.Popen(['rhythmbox', 'creepyMusic.mp3'])
        else:
            saidNothing += 1

        robo.refresh_image()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            if process != None:
                os.kill(process.pid, signal.SIGTERM)
            robo.say("I'm sorry Dave. I'm afraid I can't do that.")
            while robo.talk_flag.isSet():
                sleep(10)
            break
    cv2.destroyWindow(robo.video_window_name)

    # Black_Hair Blond_Hair Brown_Hair Eyeglasses Gray_Hair Male
    # Mouth_Slightly_Open No_Beard Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Lipstick
