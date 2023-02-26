import pickle
import numpy as np
from helper import *
from features import *
from sklearn import preprocessing


class preprocess():
    def __init__(self, data, emotion, frame_rate=16000, norm_name='variance_scale'):
        self.freq = frame_rate
        self.data = data
        self.emotion = np.array(emotion)
        self.norm_name = norm_name
        with open(data, 'rb') as handle:
            self.data = pickle.load(handle)


    def calculate_features(self, frames):
        window_sec = 0.2
        window_n = int(self.freq * window_sec)
        st_f = stFeatureExtraction(frames, self.freq, window_n, window_n / 2)

        if st_f.shape[1] > 2:
            i0 = 1
            i1 = st_f.shape[1] - 1
            if i1 - i0 < 1:
                i1 = i0 + 1
            deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
            for i in range(i0, i1):
                deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
            return deriv_st_f
        elif st_f.shape[1] == 2:
            deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
            deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
            return deriv_st_f
        else:
            deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
            deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
            return deriv_st_f
    

    def normalize(self, data):
        out_norm_data = []
        if (self.norm_name == 'variance_scale'):
            for sample in data:
                Data_scale = preprocessing.scale(sample,axis=0)
                # Data_scale = preprocessing.StandardScaler(with_mean=0, with_std=2).fit_transform(data)
                out_norm_data.append(Data_scale )
        elif (self.norm_name == 'min_max'):
            for sample in data:
                min_max_scaler = preprocessing.MinMaxScaler()
                Data_scale = min_max_scaler.fit_transform(sample)
                out_norm_data.append(Data_scale)

        return np.array(out_norm_data)


    def get_feature(self):
        data_speech = []
        data_label = []
        counter = 0

        for ses_mod in self.data:
            x_head = ses_mod['signal']
            st_features = self.calculate_features(x_head)
            st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
            data_speech.append( st_features.T )
            data_label.append(ses_mod['emotion']) 
            counter+=1
            if(counter%100==0):
                print(counter)
        
        data_speech = np.array(self.normalize(data_speech))
        return np.array(data_speech), np.array(data_label)
