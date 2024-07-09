from nuscenes.nuscenes import NuScenes

# Specify the path to the nuScenes dataset
data_directory = '/home/mide/Desktop/nuscenes'

# Load the nuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot=data_directory, verbose=True)

# Get the list of scenes
scenes = nusc.scene

# Print information about each scene
for i, scene in enumerate(scenes):
    print(f"Scene {i}: Token: {scene['token']}, Name: {scene['name']}")
    
