import os
import training_shakeout
import sys

acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
retrain = 0
lr = 1e-4
model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
parent_dir = './'
acc = 0
# save_name = 'cov0cov0fc0fc0'+'.pkl'
c_val_list = [0.5, 1, 5, 10]

for c_val in c_val_list:
    print(c_val)
    save_name = 'cval' + str( int (round(c_val * 10))) + '.pkl'
    retrain = 0
    acc = 0
    while (acc < 0.9936 and retrain < 2):
        if (retrain == 1):
            lr = 1e-5
            nopruning = False
        else:
            lr = 1e-4
            nopruning = True
        param = [
        ('-pcov',pcov),
        ('-pcov2',pcov2),
        ('-pfc',pfc),
        ('-pfc2',pfc2),
        ('-m',model_tag),
        ('-lr',lr),
        ('-dropout', 1.),
        ('-train',True),
        ('-weight_file_name', save_name),
        ('-shakeout_c', c_val),
        ('-parent_dir', parent_dir),
        ('-nopruning', nopruning)
        ]
        lr = lr / float(2)
        acc = training_shakeout.main(param)
        retrain = retrain + 1

retrain = 0
lr = 1e-4
model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
acc_list.append(acc)

print (acc_list)

print('accuracy summary: {}'.format(acc_list))
