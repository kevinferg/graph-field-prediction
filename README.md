# Topology-Agnostic Graph U-Nets for Scalar Field Prediction on Unstructured Meshes


## Paper Abstract
Machine-learned surrogate models to accelerate lengthy computer simulations are becoming increasingly important as engineers look to streamline the product design cycle. In many cases, these approaches offer the ability to predict relevant quantities throughout a geometry, but place constraints on the form of the input data. In a world of diverse data types, a preferred approach would not restrict the input to a particular structure. In this paper, we propose Topology-Agnostic Graph U-Net (TAG U-Net), a graph convolutional network that can be trained to input any mesh or graph structure and output a prediction of a target scalar field at each node. The model constructs coarsened versions of each input graph and performs a set of convolution and pooling operations to predict the node-wise outputs on the original graph. By training on a diverse set of shapes, the model can make strong predictions, even for shapes unlike those seen during training. A 3-D additive manufacturing dataset is presented, containing Laser Powder Bed Fusion simulation results for thousands of parts. The model is demonstrated on this dataset, and it performs well, predicting both 2-D and 3-D scalar fields with a median $R^2 > 0.85$ on test geometries. 



## Dataset
The dataset is currently available via this [Google Drive link](https://drive.google.com/file/d/1cWVClc2hmC7Zvb24OqYGH0fYzAzErq1a/view?usp=sharing).  

Instructions for how to use it are [here](3d/README.md).



## Usage

PyTorch and PyTorch Geometric are required.
Set up a Python virtual environment that has the necessary requirements with the following:
```
cd graph-field-prediction
python -m venv ./
source ./bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```

- The [2d/](2d/) folder contains code for training models on the 2-D stress prediction dataset.
- The [3d/](3d/) folder contains code for training models on the 3-D Additive Manufacturing z-displacement prediction dataset.