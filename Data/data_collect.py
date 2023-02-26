import os
import wave
import numpy as np
from helper import *

class data_collect():
    def __init__(self, data_path, emotion, arg):
        self.data_path = data_path
        self.emotion = emotion
        self.arg = arg

    def read_iemocap_mocap(self):
        data = []
        ids = {}
        sessions = self.arg
        for session in sessions:
            path_to_wav = self.data_path + session + '/dialog/wav/'
            path_to_emotions = self.data_path + session + '/dialog/EmoEvaluation/'
            path_to_transcriptions = self.data_path + session + '/dialog/transcriptions/'
            files2 = os.listdir(path_to_wav)
            files = []

            for f in files2:
                if f.endswith(".wav"):
                    if f[0] == '.':
                        files.append(f[2:-4])
                    else:
                        files.append(f[:-4])
                        
            for f in files:                   
                wav = get_audio(path_to_wav, f + '.wav')
                transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
                emotions = get_emotions(path_to_emotions, f + '.txt')
                sample = split_wav(wav, emotions)

                for ie, e in enumerate(emotions):                
                    e['signal'] = sample[ie]['left']
                    e.pop("left", None)
                    e.pop("right", None)
                    e['transcription'] = transcriptions[e['id']]
                    if e['emotion'] in self.emotion:
                        if e['id'] not in ids:
                            data.append(e)
                            ids[e['id']] = 1

        sort_key = get_field(data, "id")
        return np.array(data)[np.argsort(sort_key)]

    def read_emodb(self):
        gender = self.arg
        emodb_dataset = []
        for file in os.listdir(self.data_path):
            if file.endswith('.wav'):
                file_name = file[:-4]
                label = self.emotion[self.emotion[file[5]]]
                if label == 0 or label == 1 or label == 2 or label == 3:
                    new_sample = {}
                    new_sample['emotion'] = label
                    new_sample['file_name'] = file_name
                    new_sample['speaker_id'] = file_name[:2]
                    new_sample['gender'] = gender[file_name[:2]]

                    wav = wave.open(self.data_path + file, mode="r")
                    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
                    content = wav.readframes(nframes)
                    samples = np.fromstring(content, dtype=np.int16)
                    left = samples[0::nchannels]
                    new_sample['signal'] = left
                    emodb_dataset.append(new_sample)
        
        return emodb_dataset
