# lab3_Taiwanese_speech_recognization



### 原理
ASR的輸入是語音片段，輸出是對應的文本內容
使用深度神經網路（Deep Neural Networks，DNN）實現ASR的一般流程如下

![image](https://user-images.githubusercontent.com/93765298/149614091-df03b67f-b4c9-475e-9c42-4c21f5d6199f.png)


### WaveNet模型結構如下所示，
主要使用了多層因果空洞卷積（Causal Dilated Convolution）和Skip Connections
![image](https://user-images.githubusercontent.com/93765298/149614101-4991cf64-4a15-4ed2-9b5d-f130c10b9163.png)



### 由於MFCC特徵為一維序列，所以使用Conv1D進行卷積
因果是指，卷積的輸出只和當前位置之前的輸入有關，即不使用未來的特徵，可以理解為將卷積的位置向前偏移

![image](https://user-images.githubusercontent.com/93765298/149614060-eb7fd8c3-0608-49d5-9c8c-9b0c2282c5bf.png)



### 工作環境

librosa

MFCC

numpy

python_speech_features

tensorflow

make_axes_locatable

Ubuntu 20.04

Python 3.7

Keras(conV1D)

scipy.io.wavfile



### Visualization
![1](https://user-images.githubusercontent.com/93765298/149708826-d8301113-f478-4bee-9eaa-3fdb68b58e0e.png)


![2](https://user-images.githubusercontent.com/93765298/149708841-93ac32a2-1673-409e-9363-0f588467cf4d.png)



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


### Create Model
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


### calc_ctc_loss
def calc_ctc_loss(args):
    y, yp, ypl, yl = args
    return K.ctc_batch_cost(y, yp, ypl, yl)

ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, Y_pred, X_length, Y_length])
model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
optimizer = SGD(lr=0.02, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)

### random_predict
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

### get_wav_files
def get_wav_files(wav_path):
     wav_files = []
     for (dirpath, dirnames, filenames) in os.walk(wav_path):
         for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
               filename_path = os_path.join(dirpath, filename)
               wav_files.append(filename.path)
return wav_files

### get_tran_texts
def get_tran_texts(wav_files, tran_path):
     tran_texts = []
     for wav_file in wav_files:
          basename = os.path.basename(wav_file).split('.')[0]
          tran_file = os.path.join(tran_path, basename + '.txt')
          if os.path.exists(tran_file) is False:
             return None
         fd = open(tran_file, 'r')
         text = fd.readline()
         tran_texts.append(text.split('\n')[0])
         fd.close()
return tran_texts
