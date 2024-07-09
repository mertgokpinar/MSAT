
import numpy as np
import radiate
import os
from pandas import*



def set_origin(fdata):

    min=int(np.min(fdata[:,1]))

    max=int(np.max(fdata[:,1]))+1

    x=[]
    for id in range(min,max):

        i=fdata[fdata[:,1]==id]
       
        
        if  i.shape[0] >= 1:
            

            # print('before:',i)
            # breakpoint()

            i[1::,13]= i[1::,13]-i[0,13]
            i[1::,15]= i[1::,15]-i[0,15]
            i[0,13]=0
            i[0,15]=0

           
            #x.append(i)
            fdata[fdata[:,1]==id]=i

            # print('after:',fdata[fdata[:,1]==id])
            # breakpoint() 


    # x=np.asarray(x).astype('float32')

    return fdata


data_root='/media/otonom/cold_storage/radiate'  #added new data path for origin centered labels -mg
seq_root='/media/otonom/cold_storage/radiate/label'
seq_list=sorted(os.listdir(seq_root))
delimiter=''

for s in range(len(seq_list)):

    seq_name=seq_list[s]
    #seq_name='motorway_2_2.txt'
    label_path = os.path.join(data_root, 'label/{}'.format(seq_name))
    seq=np.genfromtxt(label_path, delimiter=delimiter, dtype=str) 
    seq = seq.astype('float32')

    # print('seq shape',np.shape(seq))
    # print('before:','x-axis:',seq[2,13],'y-axis:',seq[2,15])


    seq_2=set_origin(seq)
    #seq_2 = np.vstack(seq_2)


    # print('seq_2 shape:',np.shape(seq_2))
    # print('after:','x-axis:',seq_2[2,13],'y-axis:',seq_2[2,15])

    np.savetxt(f'{data_root}/labels_origin/{seq_name}', seq_2, fmt='%s')




