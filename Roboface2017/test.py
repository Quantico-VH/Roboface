"""
Running the trained neural network, normalise image and control RoboFace.
Copyright (C) 2017 Letitia Parcalabescu

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
from time import sleep
import os, subprocess, signal, psutil
import pyttsx3
import time

IMAGE_SIZE = (128, 128)
IOD = 40.0


def onStart(name):
    print ('test')




if __name__ == "__main__":
    amplDict = {'a': 10, 'b': 1, 'c': 1, 'd': 2, 'e': 3, 'f': 1, 'g': 3, 'h': 4, 'i': 4, 'j': 5, 'k': 3, 'l': 2, 'm': 0,
                'n': 1, 'o': 10, 'p': 0, 'q': 2, 'r': 1, 's': 2, 't': 1, 'u': 4, 'v': 2, 'w': 2, 'x': 5, 'y': 6, 'z': 2}
    roboFace = face.Face(x_weight=0.8, y_weight=0.2)
    # roboFace.neutral()
    # sleep(10)
    # roboFace.happy()
    # roboFace.angry()
    # roboFace.unsure()
    # roboFace.sad()
    # roboFace.moveLips(2)
    # sleep(1)
    # roboFace.moveLips(5)
    # sleep(1)
    # roboFace.moveLips(9)
    # sleep(1)

    '''str_val = 'Hewston we have a problem'
    engine = pyttsx3.init()
    engine.setProperty('rate', 50)
    engine.connect('started-word', on_word)
    engine.say(str_val)
    engine.runAndWait()'''

    roboFace.happy()
    sleep(5)
    roboFace.sad()
    sleep(5)
    roboFace.unsure()
    sleep(10)
