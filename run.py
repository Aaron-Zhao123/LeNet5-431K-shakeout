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
save_name = 'cov0cov0fc0fc0'+'.pkl'

while (acc < 0.9936 and retrain < 1):
    param = [
    ('-pcov',pcov),
    ('-pcov2',pcov2),
    ('-pfc',pfc),
    ('-pfc2',pfc2),
    ('-m',model_tag),
    ('-lr',lr),
    ('-dropout', 0.5),
    ('-train',True),
    ('-weight_file_name', save_name),
    ('-shakeout_c', 1.),
    ('-parent_dir', parent_dir),
    ('-nopruning', False)
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
