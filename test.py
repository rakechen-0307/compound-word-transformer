import os
import pickle
import numpy as np

path_data_root = './dataset/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, 'dictionary.pkl')

# load
dictionary = pickle.load(open(path_dictionary, 'rb'))
event2word, word2event = dictionary
train_data = np.load(path_train_data)

print(train_data['x'][0][:50])