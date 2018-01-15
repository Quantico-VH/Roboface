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

def MoveLips(times, Amplitude, flag):
    roboFace.setSpeedLips(127)
    i = 0
    dt = times[1] - times[0]
    current_amp = 0
    current_idx = 0
    C=[0]
    while flag.isSet() and i < len(times):
        delta_amp = current_amp - int(Amplitude[i])
        if np.abs(delta_amp) > 2:
            roboFace.moveLips(int(Amplitude[i]))
            sleep(dt * (i - current_idx))
            current_amp = int(Amplitude[i])
            current_idx = i
            C.append(current_idx)
        i = i + 1
    
    if ~flag.isSet():
        roboFace.moveLips(0)
        sleep(0.05)
    


def Talk(phrase, flag):
    A = "espeak -z -s 80 -v female5 "
    A = A + "'" + phrase + "'"
    os.system(A)
    flag.clear()


def Say(text):
    phrases=sent_tokenize(text)
    for phrase in phrases:
            phrase=phrase.replace("'"," ")
	    flag = Event()
	    flag.set()
	    A = "espeak -z -s 80 -v female5 -w temp.wav "
	    A = A + "'" + phrase + "'"
	    os.system(A)
	    samplerate, data = wavfile.read('temp.wav')
	    times = np.arange(len(data)) / float(samplerate)
	    max_data = maximum_filter1d(data, size=1000)
	    max_data = gaussian_filter(max_data,sigma=100)
	    max_Amplitude = 10
	    Amplitude = max_Amplitude * (max_data / float(np.max(max_data)))
	    
            
	    i=0
	    dt = times[1] - times[0]
	    current_amp = 0
	    current_idx = 0
	    C=[0]
	    while flag.isSet() and i < len(times):
		delta_amp = current_amp - int(Amplitude[i])
		
		if np.abs(delta_amp) > 2 :
		    current_amp = int(Amplitude[i])
		    current_idx = i
		    C.append(current_idx)
		i = i + 1

	    plt.figure(1)
	    plt.suptitle(phrase)
	    plt.subplot(211)
	    plt.plot(times, data)
	    plt.plot(times, max_data, 'r')
	    plt.subplot(212)
	    plt.plot(times, Amplitude)
	    plt.plot(times[C],Amplitude[C].astype(int),'r*')
	    plt.show()
	    

	    thread_movement = Thread(target=MoveLips, args=(times, Amplitude, flag))
	    thread_talk = Thread(target=Talk, args=(phrase, flag))
	    thread_talk.start()
	    thread_movement.start()
	    thread_talk.join()
	    thread_movement.join()

T = []
for i in range(1):

    phrases = ['I like it when people smile at me!',
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
            "Hey, why is no one looking at me? I feel neglected. I feel it. I feel it! I am afraid!"
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

Say( "I'm sorry Dave. I'm afraid I cant do that.")

