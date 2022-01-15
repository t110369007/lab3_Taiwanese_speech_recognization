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
#å¤å¶ä»??
#? è½½?æ¬?æ³¨è·¯å?å¹¶æ¥??


with open('/home/wenyuan-u/ML_dir/lab3/China/train-toneless_update.csv' ,newline='',errors='ignore') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
             a = row[0]
             b= row[1]
             f = open('/home/wenyuan-u/ML_dir/lab3/China/txt/'+a+'.txt','w')
             f.write(b)
             f.close()







text_paths = glob.glob('./txt/*.txt')
total = len(text_paths)
print(total)

with open(text_paths[0], 'r', encoding='utf8') as fr:
    lines = fr.readlines()
    print(lines)
#å¤å¶ä»??
#?å??æ¬?æ³¨?è¯­?³æ?ä»¶è·¯å¾ï?ä¿ç?ä¸­æ?å¹¶å»?ç©º??
texts = []
paths = []
for path in text_paths:
    with open(path, 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        line = lines[0].strip('\n').replace(' ', '')
        texts.append(line)
        paths.append(path)

print(paths[0], texts[0])
#å¤å¶ä»??
#MFCC?¹å?ä¿ç?13ç»´ï?å®ä?? è½½è¯­é³?ä»¶å¹¶å»?ä¸¤ç«¯é??³ç??½æ°ï¼ä»¥?å¯è§å?è¯­é³?ä»¶?å½??





mfcc_dim = 13

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    
    return audio, sr

def visualize(index):
    path = paths[index]
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
#å¤å¶ä»??
#ç¬¬ä??¡è¯­?³æ?ä»¶å¯¹åºç??å?æ³¢å½¢?MFCC?¹å?å¦ä??ç¤?

#?·å??¨é¨è¯­é³?ä»¶å¯¹å??MFCC?¹å?
features = []
for i in tqdm(range(total)):
    path = paths[i]
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

X_train = [features[i] for i in train_index]
Y_train = [texts[i] for i in train_index]
X_test = [features[i] for i in test_index]
Y_test = [texts[i] for i in test_index]

batch_size = 16
    
def batch_generator(x, y, batch_size=batch_size):  
    offset = 0
    while True:
        offset += batch_size
        
        if offset == batch_size or offset >= len(x):
            data_index = np.arange(len(x))
            np.random.shuffle(data_index)
            x = [x[i] for i in data_index]
            y = [y[i] for i in data_index]
            offset = batch_size
            
        X_data = x[offset - batch_size: offset]
        Y_data = y[offset - batch_size: offset]
        
        X_maxlen = max([X_data[i].shape[0] for i in range(batch_size)])
        Y_maxlen = max([len(Y_data[i]) for i in range(batch_size)])
        
        X_batch = np.zeros([batch_size, X_maxlen, mfcc_dim])
        Y_batch = np.ones([batch_size, Y_maxlen]) * len(char2id)
        X_length = np.zeros([batch_size, 1], dtype='int32')
        Y_length = np.zeros([batch_size, 1], dtype='int32')
        
        for i in range(batch_size):
            X_length[i, 0] = X_data[i].shape[0]
            X_batch[i, :X_length[i, 0], :] = X_data[i]
            
            Y_length[i, 0] = len(Y_data[i])
            Y_batch[i, :Y_length[i, 0]] = [char2id[c] for c in Y_data[i]]
        
        inputs = {'X': X_batch, 'Y': Y_batch, 'X_length': X_length, 'Y_length': Y_length}
        outputs = {'ctc': np.zeros([batch_size])}
        
        yield (inputs, outputs)
#å¤å¶ä»??
#å®ä?è®­ç??æ°?æ¨¡?ç??å¹¶å¼å§è®­ç»?
epochs = 25
num_blocks = 3
filters = 128

X = Input(shape=(None, mfcc_dim,), dtype='float32', name='X')
Y = Input(shape=(None,), dtype='float32', name='Y')
X_length = Input(shape=(1,), dtype='int32', name='X_length')
Y_length = Input(shape=(1,), dtype='int32', name='Y_length')

def conv1d(inputs, filters, kernel_size, dilation_rate):
    return Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None, dilation_rate=dilation_rate)(inputs)

def batchnorm(inputs):
    return BatchNormalization()(inputs)

def activation(inputs, activation):
    return Activation(activation)(inputs)

def res_block(inputs, filters, kernel_size, dilation_rate):
    hf = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'tanh')
    hg = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'sigmoid')
    h0 = Multiply()([hf, hg])
    
    ha = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')
    hs = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')
    
    return Add()([ha, inputs]), hs

h0 = activation(batchnorm(conv1d(X, filters, 1, 1)), 'tanh')
shortcut = []
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        h0, s = res_block(h0, filters, 7, r)
        shortcut.append(s)

h1 = activation(Add()(shortcut), 'relu')
h1 = activation(batchnorm(conv1d(h1, filters, 1, 1)), 'relu')
Y_pred = activation(batchnorm(conv1d(h1, len(char2id) + 1, 1, 1)), 'softmax')
sub_model = Model(inputs=X, outputs=Y_pred)

def calc_ctc_loss(args):
    y, yp, ypl, yl = args
    return K.ctc_batch_cost(y, yp, ypl, yl)

ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, Y_pred, X_length, Y_length])
model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
optimizer = SGD(lr=0.02, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)

checkpointer = ModelCheckpoint(filepath='asr.h5', verbose=0)
lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=0.000)

history = model.fit_generator(
    generator=batch_generator(X_train, Y_train), 
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs, 
    validation_data=batch_generator(X_test, Y_test), 
    validation_steps=len(X_test) // batch_size, 
    callbacks=[checkpointer, lr_decay])
#å¤å¶ä»??
#ä¿å?æ¨¡å??å???
sub_model.save('asr.h5')
with open('dictionary.pkl', 'wb') as fw:
    pickle.dump([char2id, id2char, mfcc_mean, mfcc_std], fw)
#å¤å¶ä»??
#ç»å¶è®­ç?è¿ç?ä¸­ç??å¤±?½æ°?²çº¿
train_loss = history.history['loss']
valid_loss = history.history['val_loss']
plt.plot(np.linspace(1, epochs, epochs), train_loss, label='train')
plt.plot(np.linspace(1, epochs, epochs), valid_loss, label='valid')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
#å¤å¶ä»??

#? è½½æ¨¡å?ï¼é??ºå¯¹è®­ç??å?æµè??ä¸­?è¯­?³è?è¡è???
from keras.models import load_model
import pickle

with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

sub_model = load_model('asr.h5')

def random_predict(x, y):
    index = np.random.randint(len(x))
    feature = x[index]
    text = y[index]
    
    pred = sub_model.predict(np.expand_dims(feature, axis=0))
    pred_ids = K.eval(K.ctc_decode(pred, [feature.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()
    
    print('True transcription:\n-- ', text, '\n')
    print('Predicted transcription:\n-- ' + ''.join([id2char[i] for i in pred_ids]), '\n')

random_predict(X_train, Y_train)
random_predict(X_test, Y_test)



import   librosa, display
import numpy  as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1)

librosa.display.waveshow(y, sr = sr, ax=ax[0])
ax[0].set_title('nutcracker waveform')

S = librosa.feature.melspectrogrm(y=y, sr=sr, n_mels=128, fmax=8000)
S_db = librosa.power_to_db(S, ref=np.max)
librosa.display(S_dB, x_axis='time', y_axis='mel' ,sr=sr, fmax=8000, ax=ax[1])
ax[1].set_title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()




             
             
def get_wav_files(wav_path):
     wav_files = []
     for (dirpath, dirnames, filenames) in os.walk(wav_path):
         for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
               print('wav_file 1=',filename) 
               filename_path = os.path.join(dirpath, filename)
               wav_files.append(filename_path)
     return wav_files


def get_tran_texts(wav_files, tran_path):
     tran_texts = []
     for wav_file in wav_files:
         print('wav_file=',wav_file)
         basename = os.path.basename(wav_file).split('.')[0]
         x = os.path.splitext(basename)[0]
         tran_file = os.path.join(tran_path, x+ '.txt')
         print('tran_file=',tran_file)
         if os.path.exists(tran_file) is False:
             return None
         fd = open(tran_file, 'r')
         text = fd.readline()
         tran_texts.append(text.split('\n')[0])
         fd.close()
     return tran_texts

#ä½è

#?¾æ¥ï¼https://juejin.cn/post/6844903682979430413
#?¥æ?ï¼ç??æ???
#?ä??å?ä½è
