import numpy as np
from threading import Thread, Event
import face
from time import sleep, time
import os
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d,gaussian_filter
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import string
#download nltk punkt in order to complete nltk set-up
#nltk.download()
roboFace = face.Face(x_weight=0.8, y_weight=0.2)


def Undersampled_Lip_Tragectory(phrase,Sleep_Time):
    A ="espeak -z -s 100 -v female5 -w test.wav "
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
    Amplitude=max_Amplitude*(max_data/float(np.max(max_data)))
    n=Sleep_Time*samplerate
    Amp=[]
    T=[]
    i=0
    while (i*n<=N):
        Amp.append(Amplitude[int(i*n)])
        T.append(times[int(i*n)])
        i=i+1
    Amp=np.array(Amp)
    T=np.array(T)
    '''
    plt.figure(1)
    plt.suptitle(phrase)
    plt.subplot(211)
    plt.plot(times,data)
    plt.plot(times,max_data,'r')
    plt.subplot(212)
    plt.plot(times,Amplitude)
    plt.plot(T,Amp,'r*')
    plt.show()
    '''
   
    return Amp,T






def MoveLips(Sleep_Time, Amplitude, flag):
    roboFace.setSpeedLips(127)
    i=0
    while flag.isSet() and i < len(Amplitude):
        roboFace.moveLips(int(Amplitude[i]))
        sleep(Sleep_Time)
        i = i + 1
    
    if ~flag.isSet():
        roboFace.moveLips(0)
        sleep(0.05)
    


def Talk(phrase, flag):
    A = "espeak -z -s 100 -v female5 "
    A = A + "'" + phrase + "'"
    os.system(A)
    flag.clear()


def Say(text):
    phrases=sent_tokenize(text)
    for phrase in phrases:
            phrase=phrase.replace("'"," ")
	    flag = Event()
	    flag.set()
            Sleep_Time=0.05
	    Amplitude,Time=Undersampled_Lip_Tragectory(phrase,Sleep_Time)
	    

	    thread_movement = Thread(target=MoveLips, args=(Sleep_Time, Amplitude, flag))
	    thread_talk = Thread(target=Talk, args=(phrase, flag))
	    thread_talk.start()
	    thread_movement.start()
	    thread_talk.join()
	    thread_movement.join()

phrases = ['Hello world, I think that I should be the next United States President',
'I like it when people smile at me!',
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
            "I'm sorry Dave. I'm afraid I cant do that.",
            "Hey, why is no one looking at me? I feel neglected. I feel it. I feel it! I am afraid!",
            "I'm sorry Dave. I'm afraid I cant do that.",
            "Hey! I am a great actor! I think that I should be the next StarWars maskot. Why George Lukas hasnt made me a contract yet?",
            "May the force be with you! Good bye!"
            ]


for phr in phrases:
    print(phr)
    Say(phr)



