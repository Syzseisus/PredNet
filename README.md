# PredNet

**PredNet** implementation of PyTorch.  
Environments are in `env.yaml`.  

The network could be found in `prednet.py` which importing `convlstmcell.py`.  
Train in `train.ipynb`, and test in `test.ipynb`.  

# Training Data
In folder named **'[KITTI](http://www.cvlibs.net/datasets/kitti/)'**, you can find the samples of datas used to testing this **Predet**.  
Save processed datas in `kitti_data` folder as  
for training : `X_train.hkl`, `sources_train.hkl`  
for validation : `X_val.hkl`, `sources_val.hkl`  
for test : `X_test.hkl`, `sources_test.hkl`  
(you can download them in [here](https://figshare.com/articles/dataset/KITTI_hkl_files/7985684))  
Datas loaded with `kitti_data_load.py` when running `train.ipynb`.  

# Details

Code and models accompanying [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104) by Bill Lotter, Gabriel Kreiman, and David Cox.  

The PredNet is a deep recurrent convolutional neural network that is inspired by the neuroscience concept of predictive coding (Rao and Ballard, 1999; Friston, 2005). Check out example prediction videos [here](https://coxlab.github.io/prednet/).  

![image](https://user-images.githubusercontent.com/83002480/125235880-212c2f80-e31e-11eb-9955-e16c6c794c8c.png)

# reference
1. https://coxlab.github.io/prednet/
2. https://github.com/leido/pytorch-prednet
3. https://github.com/jonizhong/afa_prednet

# notice
The name of files like jupyter_ or colab_ is for indication.  
You have to remove them before use it.  

# Hits
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FSyzseisus%2FPredNet&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
