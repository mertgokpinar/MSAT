## *MSAT: Multi-Sensory All Weather Trajectory Predictor*

> Due to the ease, safety and latent ecological gains associated with autonomous vehicles (AV’s), researchers and industry have shown tremendous interest in AV. Despite optimistic methods from leading au-
tonomous vehicle (AV) companies about the future of AV technology, current methods face significant limitations. These methods have constraints such as the need for high-definition maps, clear sensor data, and accurate GPS information, which hinder broader implementation and restrict AV technology to small-scale
applications.Predicting the trajectories around an AV is challenging due to sensor imperfections, particularly under adverse environmental and weather conditions, which poses a significant obstacle to their widespread
use. To address this issue, a new deep Learning-based framework called the Multi-Sensory All Weather Trajectory Predictor (MSAT) is proposed, which serves as a robust and complementary trajectory prediction solution under inclement weather conditions. The proposed approach employs an autoencoder-based trajectory
prediction algorithm combined with a novel transformer-based multi-sensor fusion block and multi-weather SensorNet to predict the trajectories of multiple agents in adverse weather conditions. MSAT incorporates sensory data such as camera, lidar, and radar data as environmental factors, which allows for improved predictions under conditions where semantic map information is not available. The proposed framework demonstrates the effectiveness of MSAT through experiments conducted in a variety of challenging scenarios. The proposed
framework achieved low average displacement and final displacement error in predicting the future motions of multi-agents in adverse weather conditions such as rain, fog and snow. Overall, this work is anticipated to bring AVs one step closer to safe and reliable autonomous driving in all-weather conditions.

![MSAT_workflow](https://github.com/mastermert/MSAT/assets/67050456/194e130a-b563-4081-b7d4-ecce05b421b5)

## **Installation:**
Clone This Repository:
```
https://github.com/mertgokpinar/MSAT.git
```
Create a conda environment
```
conda create -n msat python==3.10
```
Activate your conda environment
```
conda activate msat
```
  Install Requirements
```
pip install requirements.txt
```
 Clone Robotcar SDK into MSAT diractory
```bash
cd MSAT
https://github.com/ori-mrg/robotcar-dataset-sdk.git
```

## **Preparing Datasets**
MSAT supports publicly available [Radiate](https://pro.hw.ac.uk/radiate/), [nuScenes](https://www.nuscenes.org/nuscenes) and  [Oxford Radar RobotCar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/) datasets, please follow their instructions to access and download datasets.

**Radiate**

To preprocess raw Radiate dataset, run the following code:

```python
python preprocess_radiate.py 
```

**NuScenes**

You can use the following code for preprocessing NuScenes:

```python
python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>
```

**Oxford RobotCar**

For preprocessing Oxford RobotCar dataset, we used publicly available [Gramme](https://github.com/yasinalm/gramme). Usage instructions given in [here](https://github.com/yasinalm/gramme)

## **Training**

**Single Sensor Training**

First Train Variational autoencoder:
```python
python train.py --cfg pre_train_camera
```
Then Train Trajectory sampler:
```python
python train.py --cfg train_camera
```
**Multi Sensor Training**
	
First Train Sensor Fusion Transformer:
```python
python train_transformer.py  --src1  lidar  --tgt  camera  --exp_name  lidar_camera_large  --cfg  pre_train_lidar_camera
```
Second Train Variational autoencoder:
```python
python train.py --cfg pre_train_lidar_camera
```
Third Train Trajectory sampler:
```python
python train.py --cfg train_lidar_camera
```

## **Testing**

For testing the models, you can pass the trajectory sampler training config file as argument.

```python
python train.py --cfg train_camera # change cfg depending on your training
```
**PreTrained Weights**

We provide weights for the Variational autoencoder, Trajectory Sampler and Sensor Fusion Transfromer, each of can be downloaded from [here](https://drive.google.com/drive/folders/1CNwQekxh8Zp9jDJ6_3YCPJrD7uVQN9Fq?usp=drive_link)

## **Reference**

If you find our work useful in your research or if you use parts of this code, please consider citing:

```bibtex
@misc{mert2024msat,
    title={MSAT: Multi-Sensory All Weather Trajectory Predictor},
    author={Gokpinar, M., Kocyigit M.T., Naseer, A., Almalioglu, Y. and Turan, M.},
    year={2024}
}
```

### **Acknowledgments**
This code build on [Agentformer](https://github.com/Khrylx/AgentFormer) paper can be reached from [here](https://arxiv.org/abs/2103.14023).

This code contains code from the following repositories;

[Gramme](https://github.com/yasinalm/gramme)
[Radiate SDK](https://github.com/marcelsheeny/radiate_sdk)
[RobotCar SDK](https://github.com/ori-mrg/robotcar-dataset-sdk)


