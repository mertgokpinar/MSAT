import os
from os import path
from pickle import TRUE 
import numpy as np
data_root='/media/otonom/cold_storage/radiate'
seq=sorted(os.listdir(data_root))

for i in range (len(seq)):
    seq_name=seq[i]
    if path.exists(os.path.join(data_root,'{}/GPS_IMU_Twist/000001.txt'.format(seq_name))) is True:
        
        label_path=os.path.join(data_root,'{}/GPS_IMU_Twist/000001.txt'.format(seq_name))
        gps = np.genfromtxt(label_path,delimiter=',', dtype=str,usecols=(0,1,2))
        if gps.shape[0]!= 18:
            print('missing gps data',seq_name)

    else: 
        print('no gps data:', seq_name)