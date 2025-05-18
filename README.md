# FreeGS [AAAI'25]

![Approach](assets/network.png)

This repository contains the official implementation of the paper "Bootstraping Clustering of Gaussians for View-consistent 3D Scene Understanding", an unsupervised semantic-embedded 3DGS framework that achieves view-consistent 3D scene understanding without the need for 2D labels. For more details, please refer to: [[Paper]](https://arxiv.org/abs/2411.19551).

## News
-[25-05-18] We released the main code of FreeGS.

## Overview
- TODO
- Setup
- Training
- Evaluation
- Citation
- Acknowledgments

## TODO
- [x] Release the code for training on the LERF-Mask dataset.
- [ ] Release the evaluation code.
- [ ] Release the code on other datasets.

## Setup
- Clone the repository
```shell
git clone https://github.com/wb014/FreeGS && cd FreeGS
```
- Setup environment
```shell
conda create -n freegs python=3.10
conda activate freegs
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r environment.txt

# install msplat for rasterization
git clone https://github.com/pointrix-project/msplat.git --recursive
cd msplat
pip install .

# install cuml following https://docs.rapids.ai/install/
```
- Download dataset: [LERF-Mask](https://github.com/lkeab/gaussian-grouping).

## Training
- Step 1: Train Gaussians for scene reconstruction based on [mini-splatting](https://github.com/fatPeter/mini-splatting). 

- Step 2: Train the FreeGS:
```shell
python train_freegs.py -s datasets/lerf_mask/${scenename} -m outputs/${scenename} --eval --start_checkpoint outputs/${scenename}/chkpnt30000.pth --exp_name ${expname}
```

## Evaluation
```shell
TODO
```

## Citation
If you find FreeGS helpful, please consider giving this repository a star and citing:
```shell
@inproceedings{zhang2025bootstraping,
  title={Bootstraping clustering of gaussians for view-consistent 3d scene understanding},
  author={Zhang, Wenbo and Zhang, Lu and Hu, Ping and Ma, Liqian and Zhuge, Yunzhi and Lu, Huchuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={10166--10175},
  year={2025}
}
```

## Acknowledgments
We thank [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [LangSplat](https://github.com/minghanqin/LangSplat), [MSplat](https://github.com/pointrix-project/msplat), [Mini-Splatting](https://github.com/fatPeter/mini-splatting), [FeatUp](https://github.com/mhamilton723/FeatUp) for their efforts.