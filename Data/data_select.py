  # balance sample _ same size class
import numpy as np
from numpy.random import seed


seed(5)

def shuffle_data(Data, Label, index):
  sort_index = np.argsort(Label)
  Data = Data[sort_index]
  Label = Label[sort_index]
  return Data[index], Label[index]


class select_data():
    def __init__(self, x, y, sample_size, val=False):
        self.x = x
        self.y = y
        self.val = val
        self.sample_size = sample_size

    def get_data(self):
        uniq_levels = np.unique(self.y)
        # print(uniq_levels.shape[0])
        sample_size = int(np.ceil(self.sample_size / uniq_levels.shape[0]))
        uniq_counts = {level: sum(self.y == level) for level in uniq_levels}
        print(uniq_counts)
        # find observation index of each class levels
        groupby_levels = {}
        for ii, level in enumerate(uniq_levels):
            obs_idx = [idx for idx, val in enumerate(self.y) if val == level]
            groupby_levels[level] = obs_idx
        # oversampling on observations of each label
        balanced_copy_idx = []

        for gb_level, gb_idx in groupby_levels.items():
            # over_sample_idx = np.random.choice(gb_idx, size=sample_size[gb_level], replace=False).tolist()
            over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=False).tolist()
            balanced_copy_idx+=over_sample_idx
            if self.val==True:
                balanced_copy_idx+=over_sample_idx
        
        data_train = self.x[balanced_copy_idx]
        labels_train = self.y[balanced_copy_idx]
        index = balanced_copy_idx
        if self.val == True:
            self.sample_size*=2

        return data_train[0:self.sample_size],labels_train[0:self.sample_size], index
