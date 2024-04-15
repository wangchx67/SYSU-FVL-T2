# The solution of team SYSU-FVL-T2 for [NTIRE 2024 Low Light Enhancement Challenge](https://codalab.lisn.upsaclay.fr/competitions/17640#learn_the_details)

**Hongjun Li***,**Chenxi Wang***, **Haitao Lin**, **Zhi jin**

*: contribute equally

## News:

- 2024.3.21: we are currently leading in the testing phase :grinning: :grinning:！

  ![rank](https://github.com/wangchx67/SYSU-FVL-T2/blob/main/figs/rank.png)
- 2024.4.4： We won the championship!

## Description

This workshop aims to provide an overview of the new trends and advances in those areas. Moreover, it will offer an opportunity for academic and industrial attendees to interact and explore collaborations.

Jointly with NTIRE workshop we have an NTIRE challenge on low light enhancement, that is, the task of make images brighter, clearer, and more visually appealing, without introducing too much noise or distortion. The challenge includes scenarios such as high resolution (2K or 4K), non-uniform illumination, backlighting, extreme darkness, and night scenes. It covers indoor, outdoor, daytime, and nighttime settings. .

The aim is to obtain a network design / solution capable to produce high quality results with the best perceptual quality and similar to the reference ground truth.

**As one of the participating teams. This project document outlines the algorithmic solution adopted by our team.**

## About The Project

We follow the multi-scale network and supervision of [**UHDM**](https://github.com/CVMI-Lab/UHDM) and progressive training strategy of **[MIRNet-v2](https://github.com/swz30/MIRNetv2/)**.

## Installation

You can follow the step of MIRNet-v2 in [here](https://github.com/swz30/MIRNetv2/blob/main/INSTALL.md#installation) or based on follows:

1. Clone our repository

```
git clone https://github.com/wangchx67/SYSU-FVL-T2.git
cd SYSU-FVL-T2
```

2. Make conda environment

```
conda create -n sysuenv python=3.8
conda activate sysuenv
```

3. Install dependencies

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr

```
python setup.py develop --no_cuda_ext
```

### Train

Set the dataset root of the configuration in `./Options/Ntire24UHDLowLight.yml` and then run

```
python basicsr/train.py
```

### Test

For testing your own data, you can run

```
cd Enhancement
python test.py --weights [your pretrained model weights] --input_dir [your input data path] --result_dir [your result saved path] --dataset [your dataset name]
```

We have placed our pre-trained model for this challenge in `Enhancement/pretrained_models/net_g_150000.pth`. **If you just want to run the challenge official input data, you can run** 

```
cd Enhancement
python test.py --input_dir [your input data path]
```

the results will be saved in `Enhancement/results/NtireLL`, also, the final result can be downloaded [here](https://drive.google.com/file/d/1RE_DMzGqqX9bOusmIpKGEj7h_4UF3kWE/view?usp=sharing)

## Acknowledgments

This repo is built based on

- [**UHDM**](https://github.com/CVMI-Lab/UHDM) 
- **[MIRNet-v2](https://github.com/swz30/MIRNetv2/)**

We really appreciate their excellent works!

We also thank the computational sources supported by [Frontier Vision Lab](https://fvl2020.github.io/fvl.github.com/), SUN YAT-SEN University.
