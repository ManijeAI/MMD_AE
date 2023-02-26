import os
import numpy as np
from Model.model import *
from Data.preprocess import *
from Data.data_select import *
from Data.data_collect import *
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split



# Read Data

# _________________ Iemocap data collect (Source) ___________________

code_path = os.path.dirname(os.path.realpath(os.getcwd()))
data_path = os.path.join(code_path, '/data_set/IEMOCAP_full_release/')
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

data = data_collect(data_path, emotions_used, sessions)
mocap_data = data.read_iemocap_mocap()

with open(data_path +'mocap_data_collected.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_preprocess = preprocess('/mocap_data_collected.pickle', emotions_used)
Iemocap, Iemocap_label = data_preprocess.get_feature()

Iemocap_label = label_binarize(Iemocap_label, classes=emotions_used)
Iemocap_label = [np.where(r==1)[0][0] for r in Iemocap_label]
Iemocap_label = np.array(Iemocap_label)

# _________________ emodb data collect (Target) ___________________

data_path = './emodb/wav/'
output_path = './Data/'
emotion_dictionary = {'W': 'anger', 'L': 'bordom', 'E': 'disgust', 'A': 'anxiety', 'F': 'happiness', 'T': 'sadness', 'N': 'neutral'}
emotion_used = {'anger': 0, 'happiness': 1, 'neutral': 2, 'sadness': 3, 'bordom': 4, 'disgust': 5, 'anxiety': 6}
gender_dictionary = {'03': 'M', '08': 'F', '09': 'F', '10': 'M', '11': 'M', '12': 'M', '13': 'F', '14': 'F', '15': 'M', '16': 'F'}

data = data_collect(data_path,  emotion_used, gender_dictionary)
emodb_dataset = data.read_emodb()

with open(output_path+'berlin_data_collected.pickle', 'wb') as handle:
    pickle.dump(emodb_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_preprocess = preprocess('./berlin_data_collected.pickle', emotions_used)
Emodb, Emodb_label = data_preprocess.get_feature()


# ___________________ split data ___________________


# select equal balanced data from source and target datasets and split them into two-fold 

n_sample = 224

select_source = select_data(Iemocap, Iemocap_label, n_sample, val=False)
source_data, source_label, _ = select_source.get_data()

select_Target = select_data(Emodb, Emodb_label, n_sample, val=False)
target_data, target_label, _ = select_source.get_data()

sData_F1, sData_F2, sLabel_F1, sLabel_F2 = train_test_split(source_data, source_label , test_size=0.50, random_state=1, stratify=source_label)
tData_F1, tData_F2, tLabel_F1, tLabel_F2 = train_test_split(target_data, target_label , test_size=0.50, random_state=2, stratify=target_label)

print("source_F1:",Counter(sLabel_F1),"source_F2:",Counter(sLabel_F2))
print("Target_F1:",Counter(tLabel_F1),"Target_F2:",Counter(tLabel_F2))
print("\nIemocap:  ", Counter(source_label),"\t Emodb:  ", Counter(target_label))

# shuffle Data
new_index = np.random.permutation(len(sData_F1))
sData_F1, sLabel_F1 = shuffle_data(sData_F1, sLabel_F1, new_index)
sData_F2, sLabel_F2 = shuffle_data(sData_F2, sLabel_F2, new_index)
tData_F1, tLabel_F1 = shuffle_data(tData_F1, tLabel_F1, new_index)
tData_F2, tLabel_F2 = shuffle_data(tData_F2, tLabel_F2, new_index)


# call model
input_shape = 3400
layers = [512, 128, 34]
model_ae = AE(input_shape, layers=layers, lambda_=0.75, plot=0)

# model fit
print("\n ____\033[1m \033[91m train_AE______***_iemocap(fold_1)__Emodb(fold1)____*** \033[0m_____\n")
En_s_f11, En_t_f11 = model_ae.fit(sData_F1, tData_F1, epochs=100, batch_size=100)
print("\n ____\033[1m \033[91m train_AE______***_iemocap(fold_1)__Emodb(fold2)____*** \033[0m______\n")
En_s_f12, En_t_f12 = model_ae.fit(sData_F1, tData_F2, epochs=100, batch_size=100)
print("\n ____\033[1m \033[91m train_AE______***_iemocap(fold_2)__Emodb(fold1)____*** \033[0m______\n")
En_s_f21, En_t_f21 = model_ae.fit(sData_F2, tData_F1, epochs=100, batch_size=100)
print("\n ____\033[1m \033[91m train_AE______***_iemocap(fold_2)__Emodb(fold2)____*** \033[0m______\n")
En_s_f22, En_t_f22 = model_ae.fit(sData_F2, tData_F2, epochs=100, batch_size=100)


