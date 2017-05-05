import pickle
import scipy.io as sio

def dump_weights(mat_name, open_file_name):
    with open(open_file_name,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    # print(wc1)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': wc1,
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': wc2,
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': wd1,
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': out
    }

    biases = {
        'cov1': bc1,
        'cov2': bc2,
        'fc1': bd1,
        'fc2': bout
    }
    keys = ['cov1', 'cov2', 'fc1', 'fc2']
    print("try dumping weights")
    sio.savemat(mat_name,
                {'weights':weights})

parent_dir = './test_data/'
file_name_list =[
'cval5.pkl',
'cval10.pkl',
'cval50.pkl',
]
mat_name_list =[
'cvalue05.mat',
'cvalue10.mat',
'cvalue50.mat',
]

for i in range(0,3):
    dump_weights(parent_dir+mat_name_list[i],parent_dir+file_name_list[i])
