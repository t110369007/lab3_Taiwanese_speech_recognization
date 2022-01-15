# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input, Activation, Conv1D, Lambda, Add, Multiply, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
#from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#matplotlib inline
import random
import pickle
import glob
from tqdm import tqdm
import os

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import librosa
from IPython.display import Audio
import csv
#å¤å¶ä»??R


ROOT_dir = "/home/wenyuan-u/ML_dir/lab3/ML@NTUT-2021-Autumn-ASR"
ROOT_DATA = "/home/wenyuan-u/ML_dir/lab3/ML@NTUT-2021-Autumn-ASR/train"



def get_wav_files(wav_path):
     wav_files = []
     for (dirpath, dirnames, filenames) in os.walk(wav_path):
         for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
               #print('wav_file 1=',filename) 
               filename_path = os.path.join(dirpath, filename)
               wav_files.append(filename_path)
     return wav_files


def get_tran_texts(wav_files, tran_path):
     tran_texts = []
     for wav_file in wav_files:
         #print('wav_file=',wav_file)
         basename = os.path.basename(wav_file).split('.')[0]
         x = os.path.splitext(basename)[0]
         tran_file = os.path.join(tran_path, x+ '.txt')
         #print('tran_file=',tran_file)
         if os.path.exists(tran_file) is False:
             return None
         fd = open(tran_file, 'r')
         text = fd.readline()
         tran_texts.append(text.split('\n')[0])
         fd.close()
     return tran_texts
 
    
 
"""
with open(ROOT_dir + "/train-toneless_update.csv" ,newline='',errors='ignore') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
             a = row[0]
             b= row[1]
             f = open(ROOT_DATA+"/"+a+'.txt','w')
             #print(a)
             f.write(b)
             f.close()
"""

wav_files = get_wav_files(ROOT_DATA)

get_tran_texts(wav_files, ROOT_DATA)





text_paths = glob.glob(ROOT_DATA+"/*.txt")
total = len(text_paths)
print(total)

voice_paths = glob.glob(ROOT_DATA+"/*.wav")
total = len(voice_paths)
print(total)



with open(text_paths[0], 'r', encoding='utf8') as fr:
    lines = fr.readlines()
    print(lines)
#å¤å¶ä»??
#?å??æ¬?æ³¨?è¯­?³æ?ä»¶è·¯å¾ï?ä¿ç?ä¸­æ?å¹¶å»?ç©º??
texts = []




for path in text_paths:
    with open(path, 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        line = lines[0].strip('\n').replace(' ', '')
        texts.append(line)
       




print("text=",texts[0])




#å¤å¶ä»??
#MFCC?¹å?ä¿ç?13ç»´ï?å®ä?? è½½è¯­é³?ä»¶å¹¶å»?ä¸¤ç«¯é??³ç??½æ°ï¼ä»¥?å¯è§å?è¯­é³?ä»¶?å½??





mfcc_dim = 13



def load_and_trim(path):
    #print("path=",path)
    audio, sr = librosa.load(path)
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    
    return audio, sr

def visualize(index):
    path = voice_paths[index]
    text = texts[index]
    print('Audio Text:', text)
    
    audio, sr = load_and_trim(path)
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(len(audio)), audio)
    plt.title('Raw Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Audio Amplitude')
    plt.show()
    
    feature = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    print('Shape of MFCC:', feature.shape)
    
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    plt.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_xticks(np.arange(0, 13, 2), minor=False);
    plt.show()
    
    return path

Audio(visualize(0))






features = []
for i in tqdm(range(total)):
    path = voice_paths[i]
    audio, sr = load_and_trim(path)
    features.append(mfcc(audio, sr, numcep=mfcc_dim, nfft=551, highfreq=8000))
    
print(len(features), features[0].shape)
#å¤å¶ä»??
#å°MFCC?¹å?è¿è?å½ä???
samples = random.sample(features, 100)
samples = np.vstack(samples)

mfcc_mean = np.mean(samples, axis=0)
mfcc_std = np.std(samples, axis=0)
print(mfcc_mean)
print(mfcc_std)

features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]
#å¤å¶ä»??

chars = {}
for text in texts:
    for c in text:
        chars[c] = chars.get(c, 0) + 1

chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
chars = [char[0] for char in chars]
print(len(chars), chars[:100])

char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}
#å¤å¶ä»??
#?å?è®­ç??°æ®?æ?è¯æ°?®ï?å®ä?äº§ç??¹æ°?®ç??½æ°
total = len(wav_files)
data_index = np.arange(total)
np.random.shuffle(data_index)
train_size = int(0.85 * total)
test_size = total - train_size
train_index = data_index[:train_size]
test_index = data_index[train_size:]

print("total_size =", total)


print("train_size =", train_size)

print("test_size =", test_size)

print("train_index =", train_index)

print("test_index =", test_index)



X_train = [features[i] for i in train_index]
Y_train = [texts[i] for i in train_index]
X_test = [features[i] for i in test_index]
Y_test = [texts[i] for i in test_index]



