import numpy as np
from threading import Thread, Event
import face
from time import sleep, time
import os
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d,gaussian_filter
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

roboFace = face.Face(x_weight=0.8, y_weight=0.2)

##############################################
# define: generate_Lip_Tragectory
##############################################

def generate_Lip_Tragectory(phrase):
    A ="espeak -z -s 80 -v female5 -w test.wav "
    A=A + "'" + phrase + "'"
    #os.system("espeak -z -s 80 -v female5 -w test.wav 'Hey, why no one is looking at me? I feel neglected. I feel it! I am afraid!' ")
    os.system(A)
    samplerate, data = wavfile.read('test.wav')
    dt=1/float(samplerate)
    times = np.arange(len(data))/float(samplerate)
    N=len(times)
    max_data=maximum_filter1d(data,size=1000)
    max_data=gaussian_filter(max_data,sigma=100)
    max_Amplitude=10
    Amplitude=np.round(max_Amplitude*(max_data/float(np.max(max_data))))
    Extrema=argrelextrema(max_data, np.less_equal,order=100)[0]
    Amplitude=Amplitude[Extrema]
    Sleep_Time=[]
    for i in range(len(Extrema)):
        if (i+1<len(Extrema)):
            Sleep_Time.append(dt*(times[Extrema[i+1]]-times[Extrema[i]]))

    plt.figure(1)
    plt.suptitle(phrase)
    plt.subplot(211)
    plt.plot(times,data)
    plt.plot(times,max_data,'r')
    plt.plot(times[Extrema],max_data[Extrema],'y*')
    plt.subplot(212)
    plt.plot(times[Extrema],Amplitude)
    plt.show()


    return np.array(Amplitude),np.array(Sleep_Time),Extrema


############################################
#define:     Say,Talk,MoveLips
############################################

def MoveLips(Amplitude, Sleep_Time, flag):
    roboFace.setSpeedLips(127)
    i = 0
   #while flag.isSet() and i < len(Amplitude):
    while i < len(Amplitude):
        #print(int(Amplitude[i]),Sleep_Time[i])
        roboFace.moveLips(int(Amplitude[i]))       
        sleep(Sleep_Time[i]) 
        i=i+1

    if ~flag.isSet():
        roboFace.moveLips(0)
        sleep(0.05)
    


def Talk(phrase, flag):
    A = "espeak -z -s 80 -v female5 "
    A = A + "'" + phrase + "'"
    os.system(A)
    flag.clear()


def Say(phrase):
    flag = Event()
    flag.set()
    Amplitude,Sleep_Time,Extrema=generate_Lip_Tragectory(phrase)

    thread_movement = Thread(target=MoveLips, args=(Amplitude, Sleep_Time, flag))
    thread_talk = Thread(target=Talk, args=(phrase, flag))
    thread_talk.start()
    thread_movement.start()
    thread_talk.join()
    thread_movement.join()
'''
T = []
for i in range(1):

    phrases= ['I like it when people smile at me!',
            'You are a female, am I right?',
            'You are a male, am I right?',
            'You are wearing beautiful earrings today!',
            'I see you are wearing lipstick today. Pretty!',
            'Nice blond hair!',
            'You are wearing eyeglasses!',
            'You have nice brown hair!',
            'You have nice black hair!',
            'You must be a wise man, judging by your gray hair!',
            'You have nice wavy hair!',
            ]

    for phr in phrases:
        tstart = time()
        Say(phr)
        tstop = time()
        T.append(tstop - tstart)
        print(phr)
        print("i= {}, Ellapsed Time = {}".format(i, tstop - tstart))


T = np.array(T)
print(np.mean(T))
'''
phrase="Hey, why no one is looking at me? I feel neglected. I feel it! I am afraid!"
Say(phrase)
'''

phrase="Hey, why is no one looking at me?"
Say(phrase)
phrase=" I feel neglected."
Say(phrase) 
phrase="I feel it."
Say(phrase)
phrase="I feel it!"
Say(phrase)
phrase="I am afraid!"
Say(phrase)
'''


