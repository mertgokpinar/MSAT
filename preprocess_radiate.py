
import numpy as np
import radiate
import os
from pandas import*
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def frame_data(seq,frame,total_frame): #takes seq data, frame  number and total number of frames then returns data as np array. -mg
    fdata=[]
    list=seq.get_annotation_from_id(frame)
   
    for x in range(len(list)):
        arr= np.ones(18) * -1.0
        arr[0]=frame
        arr[1]=list[x]['id']
        if list[x]['class_name']=='bus':
            arr[2]=1
        if list[x]['class_name']=='car':
            arr[2]=2
        if list[x]['class_name']=='van':
            arr[2]=3
        if list[x]['class_name']=='truck':
            arr[2]=4
        if list[x]['class_name']=='motorbike':
            arr[2]=5
        if list[x]['class_name']=='bicycle':
            arr[2]=6
        arr[13]=list[x]['bbox']['position'][0] # x direction
        arr[15]=list[x]['bbox']['position'][1] # y direction
        arr[12]=list[x]['bbox']['position'][2] # width of 2d bbox
        arr[10]=list[x]['bbox']['position'][3] # height of 2d bbox
        arr[16]=0
        arr[17]=1 if frame in range(11,total_frame-30) else 0 # we will not use first 10 frames and last 30 frames as samples in dataset -mg
        fdata.append(arr)
    #if fdata:
        # fdata = [[]]
    fdata=np.asarray(fdata).astype('float64')
    breakpoint()
    return fdata

'''
we are only giving x and y coordinates to our 'data' because radiate dataset only contains 2d bbox,
data[14] and data[11] represents Y direction (which is height) in 3D bbox nuScenes dataset, therefore we left data entries as empty since radiate does not have 3D height data
'''

def extrinsics_from_pose(angle: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a 4x4 extrinsics matrix from the camera pose in world coordinates,
    using the Euler angles in degrees (zyx convension) and translation.

    Args:
        angle (np.ndarray): Euler angles in degrees for rotation around xyz axes.
        translation (np.ndarray): Translation in xyz.

    Returns:
        np.ndarray: 4x4 extrinsics matrix.
    """
    # Camera pose is camera to world but we need world to camera
    # So invert the pose to get the extrinscis.
    rot = R.from_euler('zyx', angle, degrees=False)
    rot_inv = rot.as_matrix().T
    T = np.eye(4) # [4,4]
    T[:3,:3] = rot_inv.astype('float64')
    T[:3,3] = -rot_inv@translation
    return T # [4x4]


def transform_annotation(pixel_coord, angle: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Convert the pixel coordinates in radar frame to world coordinates.
    
    Args:
        pixel_coord (np.ndarray): x,y pixel coordinates in radar frame. [2xn]
        
    Returns:
        world_coord (np.ndarray): xyz coordinates of the point, []
    """
    P_v = 0.175*pixel_coord # [2x1]
   

    P_v = np.append(P_v, np.ones((2, P_v.shape[1])),axis=0) # [4xn], xyz in homogenous coordinates 
    

    T_wv = extrinsics_from_pose(angle, translation) # transformation matrix from vehicle to world frame
    #T_wv = np.eye(4)
    
    P_w = T_wv.dot(P_v)  # point in world frame

   
    return P_w

data_root='/media/otonom/cold_storage/radiate'
seq_root='/media/otonom/cold_storage/radiate/sequences'

seq_list=sorted(os.listdir(seq_root))

for s in range(len(seq_list)):

    cvt_data=[]
    f=[]
    seq_name=seq_list[s] 
    seq=radiate.Sequence(os.path.join(seq_root, seq_name)) #loads radiate sequence 
    sequence_path=os.path.join(seq_root, '{}/Navtech_Cartesian.txt'.format(seq_name)) #loads radar .txt file to return total number of frames
    frame = seq.load_timestamp(sequence_path) #frame information
    total_frame=np.max(frame['frame'][-1])-1 #returns total number of frames
    f=sorted(os.listdir(os.path.join(seq_root, '{}/GPS_IMU_Twist'.format(seq_name)))) #names of frame.txt files within sequence folder as list.


    for i in range (total_frame):

        
    
        label_path=os.path.join(seq_root,'{}/GPS_IMU_Twist/{}'.format(seq_name,f[i])) #gps data path
        
        gps = np.genfromtxt(label_path,delimiter=',', dtype=str,usecols=(0,1,2),encoding= 'unicode_escape') #gps data np.array[18x3]
        data=frame_data(seq,i,total_frame) #frame data np.array[nx18] 


        translation=gps[16].astype('float64')
        angle=gps[17]
        if data.shape[0]!=0:               #not every frame has annotation data.
    
            pixel_coor=np.zeros((2,data.shape[0]))
            pixel_coor[0,:]=data[:,13].astype('float64') #x-axis
            pixel_coor[1,:]=data[:,15].astype('float64') #y-axis
            pixel_coor = pixel_coor + np.array([[-576],[576]])

            # add translation def.
            # print('iter',i)
            # print('total_frames',total_frame)
            # print('data',np.shape(data))
            # print('gps_shape',np.shape(gps))
            # print('angle_shape',angle.shape)
            # print('translation_shape',translation.shape)
            # print('pixel_coor',pixel_coor)
            # print('pixel_shape',pixel_coor.shape)
            # print('old_frame_data_array',data)
            # breakpoint()
            
            world_coor=transform_annotation(pixel_coor,angle,translation)
            data[:,13]=world_coor[0,:] #x-axis
            data[:,15]=world_coor[1,:] #y-axis
            data[:,14]=world_coor[2,:] #z-axis
            data=data.astype(str)
            
            cvt_data.append(data)  #append each frame data to sequence array 
        else:                               #if frame does not have any data add a dummy array instead. 
            arr= np.ones(18) * -1.0
            arr[0]=i    
            arr=arr.astype(str)
            cvt_data.append(arr)
        
        # print('worl_arr_shape',world_coor.shape)
        # print('new_frame_data_array',data)
        # #print('world_coor',world_coor)
        #breakpoint()

    cvt_data = np.vstack(cvt_data)
    np.savetxt(f'{data_root}/label-debug/{seq_name}.txt', cvt_data, fmt='%s')

    #save .txt file in here

'''
!!!!!
Important Reminder: Def frame_data returns frame data nx18 array where n represents number of vehicles 
however def transform_annotation returns 4xn where firts 3 rows represents X,Y,Z axis and columns(n) number of vehicles.
!!!!!
'''

