# AA3D: Adversarial Attacks on Embodied Visual Navigation via 3D Adversarial Examples
# Data Preparation
- Download the attack scenarios and model weights and place them in the corresponding directories: https://drive.google.com/file/d/1dhUcv7MvavmHwG4E11L12KK6zkjP1Ui5/view?usp=drive_link
## Environment Setup
Refer to [REVAMP](https://github.com/poloclub/revamp) for environment setup
```bash
# Set up conda environment
conda env create -f environment.yml 

# torch and cuda versions
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
  -f https://download.pytorch.org/whl/torch_stable.html  

# Install pre-built Detectron2
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2==0.6 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```
#Setting up the Environment for Testing Navigation Agents
-Refer to [PEANUT](https://github.com/ajzhai/PEANUT) for environment setup
## Download navigation dataset and model weights, place them in the corresponding directories
```
Physical-Attacks-in-Embodied-Navigation
├── PEANUT/
│   ├── habitat-challenge-data/
│   │   ├── objectgoal_hm3d/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── val_mini/
│   │   └── data/
│   │       └── scene_datasets/
│   │           └── hm3d/
│   │               ├── train/
│   │               └── val/
│   └── test_patch/
│       └── models/
│           ├── pred_model_wts.pth
│           └── mask_rcnn_R_101_cat9.pth
```
## Set up the victim model
- Victim model configuration directory:```configs/model/mask_rcnn_R_101_cat9.yaml```
- Victim model's configuration and weights directory: ```pretrained-models/mask_rcnn_R_101_cat9```
## Set up the attack scenario
- Export the adversarial object as an OBJ file in Blender and place it in the mesh directory.
- Export the scene OBJ and place it in the texture directory.
- Run the optimization script.
  ```
  export CUDA_VISIBLE_DEVICES=0
  python revamp.py
  ```
- Run 1.py to rotate the optimized texture.
# Convert the OBJ format to GLB format for simulation.
```
# Install obj2gltf
conda install -c conda-forge nodejs
npm install -g obj2gltf
conda activate obj
# Use obj2gltf to convert
obj2gltf -i attack-scene-example.obj -o svBbv1Pavdk.basis.glb
```
# Build Docker (requires sudo)
```
cd PEANUT
./build_and_run.sh
```
## After building, enter docker to test navigation agents
```
conda activate habitat
export CUDA_VISIBLE_DEVICES=0
./nav_exp.sh

# If you need to save the current running Docker container as an image
docker commit <container ID or container name> <image name>
```
# Acknowledgments
This project builds upon code from [REVAMP](https://github.com/poloclub/revamp), [PEANUT](https://github.com/ajzhai/PEANUT) and [Physically Attacks](https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav/tree/main?tab=readme-ov-file) We thank the authors of these projects for their amazing work!
