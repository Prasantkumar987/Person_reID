<h1 align="center"> Pytorch ReID </h1>
<h2 align="center"> Strong, Small, Friendly </h2>


A tiny, friendly, strong baseline code for Person-reID.

- **Strong.** It is consistent with the new baseline result in several top-conference works. I arrived Rank@1=88.24%, mAP=70.68% only with softmax loss. 

- **Small.** With fp16 (supported by Nvidia apex), our baseline could be trained with only 2GB GPU memory.

- **Friendly.** You may use the off-the-shelf options to apply many state-of-the-art tricks in one line.

## Table of contents
* [Features](#features)
* [Some News](#some-news)
* [Trained Model](#trained-model)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset--preparation)
    * [Train](#train)
    * [Test](#test)
    * [Evaluation](#evaluation)

## Features
Now we have supported:
- Running the code on Google Colab with Free GPU. Check [Here](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/colab) (Thanks to @ronghao233)
- [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) (10x Large Synethic Dataset from Market **CVPR 2019 Oral**)
- Circle Loss, Triplet Loss, Contrastive Loss, Sphere Loss, Lifted Loss, Arcface, Cosface  and Instance Loss
- Float16 to save GPU memory based on [apex](https://github.com/NVIDIA/apex)
- Part-based Convolutional Baseline(PCB)
- Multiple Query Evaluation
- Re-Ranking ([GPU Version](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/GPU-Re-Ranking))
- Random Erasing
- ResNet/ResNet-ibn/DenseNet

   
## Trained Model
I re-trained several models, and the results may be different with the original one. Just for a quick reference, you may directly use these models. 
The download link is [Here](https://drive.google.com/drive/folders/1O5QgPfAi8yioNqL5ilve4ctbEMKZUx9b?usp=sharing).

|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| [ResNet-50] | 88.84% | 71.59% |  `python train.py --train_all` |

* More training iterations may lead to better results. 
* Swin costs more GPU memory (11G GPU is needed) to run. 
* The hyper-parameter of [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) `--DG` is not tuned. Better hyper-parameter may lead to better results.


### Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- timm `pip install timm` for Swin-Transformer with Pytorch >1.7.0
- pretrainedmodels via `pip install pretrainedmodels`
- [Optional] apex (for float16) 
- [Optional] [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)

**(Some reports found that updating numpy can arrive the right accuracy. If you only get 50~80 Top1 Accuracy, just try it.)**
We have successfully run the code based on numpy 1.12.1 and 1.13.1 .

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optinal] You may skip it. Install apex from the source
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .

### Dataset & Preparation
Download [Market1501 Dataset](https://drive.google.com/file/d/1-57qJxNz6TpaZ6ZOuUZ6IBfIDlA-Crbz/view?usp=sharing)

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.


### Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

### Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


### Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.


### re-ranking
```bash
python evaluate_rerank.py
```
**It may take more than 10G Memory to run.** So run it on a powerful machine if possible. 

It will output Rank@1, Rank@5, Rank@10 and mAP results.


## Related Repos
1. [Pedestrian Alignment Network](https://github.com/layumi/Pedestrian_Alignment) ![GitHub stars](https://img.shields.io/github/stars/layumi/Pedestrian_Alignment.svg?style=flat&label=Star)
2. [DG-Net](https://github.com/NVlabs/DG-Net) ![GitHub stars](https://img.shields.io/github/stars/NVlabs/DG-Net.svg?style=flat&label=Star)
3. [3D Person re-ID](https://github.com/layumi/person-reid-3d) ![GitHub stars](https://img.shields.io/github/stars/layumi/person-reid-3d.svg?style=flat&label=Star)
