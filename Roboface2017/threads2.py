import numpy as np
from threading import Thread, Event
import face
from time import sleep, time
import random
import os
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
import math

roboFace = face.Face(x_weight=0.8, y_weight=0.2)
roboFace.setSpeedLips(127)


def MoveLips(times, Amplitude, flag):
    i = 0
    dt = times[1] - times[0]
    current_amp = 0
    current_idx = 0
    while flag.isSet() and i < len(times):
        delta_amp = current_amp - int(Amplitude[i])
        if math.fabs(delta_amp) > 2:
            roboFace.moveLips(current_amp - delta_amp)
            sleep(dt * (i - current_idx))
            current_amp = int(Amplitude[i])
            current_idx = i
        i = i + 1
    '''for k in range(0,len(times)):
        delta_amp = current_amp - int(Amplitude[k])
        if math.fabs(delta_amp) > 2:
            roboFace.moveLips(current_amp - delta_amp)
            sleep(dt * (k - current_idx))
            current_amp = int(Amplitude[k])
            current_idx = k'''
    if ~flag.isSet():
        roboFace.moveLips(0)
        sleep(0.1)


def Talk(phrase, flag):
    A = "espeak -z -s 120 "
    A = A + "'" + phrase + "'"
    os.system(A)
    flag.clear()


def Speech(phrase):
    flag = Event()
    flag.set()
    A = "espeak -z -s 120 -w temp.wav "
    A = A + "'" + phrase + "'"
    os.system(A)
    samplerate, data = wavfile.read('temp.wav')
    times = np.arange(len(data)) / float(samplerate)
    max_data = maximum_filter1d(data, size=500)
    max_Amplitude = 10
    Amplitude = max_Amplitude * (max_data / float(np.max(max_data)))

    '''plt.figure(1)
    plt.plot(times, data)
    plt.plot(times, max_data, 'r')
    plt.show()
    plt.figure(2)
    plt.plot(times, Amplitude)
    plt.show()'''
    thread_movement = Thread(target=MoveLips, args=(times, Amplitude, flag))
    thread_talk = Thread(target=Talk, args=(phrase, flag))

    thread_talk.start()
    thread_movement.start()
    thread_talk.join()

    thread_movement.join()

    print(np.max(max_data))


T = []
for i in range(2):
    phrases = ["blah blah blah blah.","Hewston we have a problem.", "hiberbolic black hole.", "Do you want to play cat and mouse."]
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
            'Straight_Hair': 'You have nice'
            }
    for phr in talk:
        tstart = time()
        Speech(talk[phr])
        tstop = time()
        T.append(tstop - tstart)
        print("i= {}, Ellapsed Time = {}".format(i, tstop - tstart))

T = np.array(T)
print(np.mean(T))
