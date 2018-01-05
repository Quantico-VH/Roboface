import numpy as np
from threading import Thread, Event
import face
from time import sleep,time
import pyttsx3
import random
import os
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d



roboFace = face.Face(x_weight=0.8, y_weight=0.2)
roboFace.setSpeedLips(60)

def MoveLips(flag):
    i=0
    while flag.isSet():
        roboFace.moveLips(i*10)
        sleep(0.5)
	if i==0:
            i=1
        else:
            i=0


def Talk(engine,phrase,flag):

    engine.say(phrase)
    engine.runAndWait()
    flag.clear()

def Speech(phrase):
    engine=pyttsx3.init()
    engine.setProperty('rate',10)
    engine.setProperty('speed',200)
    engine.setProperty('word_gap',10)
    flag=Event()
    flag.set()
    thread_movement=Thread(target=MoveLips,args=(flag,))
    thread_talk=Thread(target=Talk,args=(engine,phrase,flag))

    thread_talk.start()
    thread_movement.start()
    thread_talk.join()

    thread_movement.join()



T=[]
for i in range(3):
	phrase=["bidirectional, shift register."]
	tstart=time()
	Speech(phrase)
	tstop=time()
	T.append(tstop-tstart)
	print("i= {}, Ellapsed Time = {}".format(i,tstop-tstart))
	sleep(0.01)

T=np.array(T)
print(np.mean(T))





