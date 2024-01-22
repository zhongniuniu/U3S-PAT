# Spirally scanning and self-supervised image reconstruction enable ultra-sparse sampling multispectral photoacoustic tomography

## Contents

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Instructions for Running Code](#instructions-for-running-code)

# 1. Overview

This repository provides the PyTorch code for our papar [Spirally scanning and self-supervised image reconstruction enable ultra-sparse sampling multispectral photoacoustic tomography].

by Yutian Zhong et al.

Multispectral photoacoustic tomography (PAT) is an imaging modality that utilizes the photoacoustic effect to achieve non-invasive and high-contrast imaging of internal tissues. However, the hardware cost and computational demand of a multispectral PAT system consisting of up to thousands of detectors are huge. To address this challenge, we propose an ultra-sparse spiral sampling strategy for multispectral PAT, which we named U3S-PAT. Our strategy employs a sparse ring-shaped transducer that, when switching excitation wavelengths, simultaneously rotates and translates. This creates a spiral scanning pattern with multispectral angle-interlaced sampling. To solve the highly ill-conditioned image reconstruction problem, we propose a self-supervised learning method that is able to introduce structural information shared during spiral scanning. We simulate the proposed U3S-PAT method on a commercial PAT system and conduct in vivo animal experiments to verify its performance. The results show that even with a sparse sampling rate as low as 1/30, our U3S-PAT strategy achieves similar reconstruction and spectral unmixing accuracy as non-spiral dense sampling. Given its ability to dramatically reduce the time required for three-dimensional multispectral scanning, our U3S-PAT strategy has the potential to perform volumetric molecular imaging of dynamic biological activities.

# 2. Installation Guide

Before running this package, users should have `Python`, `PyTorch`, and several python packages installed (`numpy`, `skimage`, `yaml`, `opencv`, `odl`) .


## Package Versions

This code functions with following dependency packages. The versions of software are, specifically:
```
python: 3.7.4
pytorch: 1.4.1
numpy: 1.19.4
skimage: 0.17.2
yaml: 0.1.7
opencv: 3.4.2
odl: 1.0.0.dev0
```


## Package Installment

Users should install all the required packages shown above prior to running the algorithm. Most packages can be installed by running following command in terminal on Linux. To install PyTorch, please refer to their official [website](https://pytorch.org). To install ODL, please refer to their official [website](https://github.com/odlgroup/odl).

```
pip install package-name
```



# 3. Instructions for Running Code

## PAT Reconstruction Experiment

### Step 1: Prior embedding

Represent 3D image by implicit network network. The prior image is provided under [data/rate/prior](./data/rate/prior) folder.

```
python train_image_regression_3d.py --config configs/PAT_image_regression_3d.yaml
```

### Step 2: Network training

Reconstruct PAT image from sparsely sampled transducer elements. The reconstruction target image is provided under [data/rate/DS](./data/rate/DS) folder.

With prior embedding:
```
python train_ultraPAT_3d.py --config configs/PAT_recon_3d.yaml --pretrain
```

Without prior embedding:
```
python train_ultraPAT_3d.py --config configs/PAT_recon_3d.yaml
```

### Step 3: Image inference

Output and save the reconstruted 3D image after training is done at a specified iteration step.

With prior embedding:
```
python test_PAT_recon_3d.py --config configs/PAT_recon_3d.yaml --pretrain --iter 2000
```

Without prior embedding:
```
python test_PAT_recon_3d.py --config configs/PAT_recon_3d.yaml --iter 2000
```

