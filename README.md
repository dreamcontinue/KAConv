# KAConv: Kernel Attention Convolutions

## Requirement
- `pytorch 1.1.0+`
- `torchvision`
- `tensorboard 1.14+`
- `numpy`
- `pyyaml`
- `tqdm`
- `pillow`

## Dataset
- `ImageNet-1K`

## Introduction
This is a PyTorch implementation of KAConv. 

KAConv embeds attention weights into convolution kernels, so that the model can dynamically adjust kernel parameters at different spatial positions of convolution kernel according to different inputs. 

## Method
![](fig/KA.pdf)
<p align="center">
  The architecture of KAConv.
</p>
