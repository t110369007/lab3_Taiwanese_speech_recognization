# lab3_Taiwanese_speech_recognization

# random_predict
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

# get_wav_files
def get_wav_files(wav_path):
     wav_files = []
     for (dirpath, dirnames, filenames) in os.walk(wav_path):
         for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
               filename_path = os_path.join(dirpath, filename)
               wav_files.append(filename.path)
return wav_files

# get_tran_texts
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
