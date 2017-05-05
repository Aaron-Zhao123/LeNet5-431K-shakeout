import numpy as np
import pickle

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
c_val_list = [0.1, 10]
parent_dir = './'
mean_list = []
std_list = []

for c_val in c_val_list:
    save_name = 'cval' + str( int (round(c_val * 100))) + '.pkl'
    with open(parent_dir+ 'weights/' + save_name) as f:
        tmp = pickle.load(f)
    print(tmp[2].shape)
    tmp = tmp[2].flatten()
    mean = np.mean(tmp)
    std = np.std(tmp)
    mean_list.append(mean)
    std_list.append(std)

print(mean_list)
print(std_list)
