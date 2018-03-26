
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d,gaussian_filter
from scipy.signal import argrelextrema
from threading import Thread
import os


# In[3]:

def speak(string):
    A="espeak -z -s 80 -v female5 "
    A=A+"'"+string+"'"
    print(A)
    os.system(A)


# In[4]:

os.system("espeak -z -s 80 -v female5 -w test.wav 'Hey, why no one is looking at me? I feel neglected. I feel it! I am afraid!' ")
samplerate, data = wavfile.read('test.wav')
times = np.arange(len(data))/float(samplerate)
N=len(times)
max_data=maximum_filter1d(data,size=1000)
max_data=gaussian_filter(max_data,sigma=100)
max_Amplitude=10
Amplitude=np.round(max_Amplitude*(max_data/np.max(max_data)))

#Extrema1=argrelextrema(max_data, np.greater_equal,order=100)[0]
Extrema2=argrelextrema(max_data, np.less_equal,order=100)[0]

plt.figure(1)
plt.plot(times,data)
plt.plot(times,max_data,'r')
#plt.plot(times[Extrema1],max_data[Extrema1],'g*')
plt.plot(times[Extrema2],max_data[Extrema2],'y*')
plt.show()

plt.figure(2)
plt.plot(times,Amplitude,'.-')
#plt.plot(times[Extrema1],Amplitude[Extrema1],'r*')
plt.plot(times[Extrema2],Amplitude[Extrema2],'r*')

plt.show()
print("T= {} ,N={}".format(1/samplerate,N))
print(Extrema2.shape)


# In[26]:

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
    Amplitude=np.round(max_Amplitude*(max_data/np.max(max_data)))
    Extrema=argrelextrema(max_data, np.less_equal,order=100)[0]
    Amplitude=Amplitude[Extrema]
    Sleep_Time=[]
    for i in range(len(Extrema)):
        if (i+1<len(Extrema)):
            Sleep_Time.append(dt*(times[Extrema[i+1]]-times[Extrema[i]]))

    return np.array(Amplitude),np.array(Sleep_Time),Extrema


# In[28]:

phrase="Hey, why no one is looking at me? I feel neglected. I feel it! I am afraid!"
Amplitude,Sleep_Time,Extrema=generate_Lip_Tragectory(phrase)

plt.figure(1)
plt.plot(times,data)
plt.plot(times,max_data,'r')
plt.plot(times[Extrema],max_data[Extrema],'y*')
plt.show()

plt.figure(2)
plt.plot(times[Extrema],Amplitude,'.-')
plt.show()
print(Extrema.shape)

