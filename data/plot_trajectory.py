from cgi import print_arguments
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree

label_path='/media/otonom/cold_storage/radiate/label/motorway_2_0.txt'
delimiter=''

gt = np.genfromtxt(label_path, delimiter=delimiter, dtype=str)
gt = gt.astype('float32')


for i in range(50,70):

    agent_data=gt[gt[:,1]==i] #finds agent data from id
    print(agent_data.shape)
    
    if agent_data.shape[0]>1:
        agent_id =agent_data[0,2] #takes agent class id 
        if agent_id==1:
            agent_id='bus'
        if agent_id==2:
            agent_id='car'
        if agent_id==3:
            agent_id='van'
        if agent_id==4:
            agent_id='truck'
        if agent_id==5:
            agent_id='motorbike'
        if agent_id==6:
            agent_id='bicycle'


        agent_data=agent_data[:] #only take first 10 frames

        if i==13:        
            print(agent_data)
        


        agent_x=agent_data[:,13]
        agent_y=agent_data[:,15]

        fig=plt.figure()
        ax = fig.add_subplot(211)
        ax.scatter(agent_x,agent_y)
        ax.set(title='agent_trajectory_class_{}'.format(agent_id))
        ax = fig.add_subplot(212)
        ax.set(xlabel='frames {}'.format(agent_data[:,0]))
        ax.plot(agent_x,agent_y)
    
plt.show()

