<p align="center">
<h1 align="center"><strong>Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos</strong></h1>
<h3 align="center"> ICLR 2026 </h3>

<p align="center">
    <a href="https://liujf1226.github.io/">Jinfeng Liu</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=KKzKc_8AAAAJ">Lingtong Kong</a><sup>2</sup>,
    <a href="https://openreview.net/profile?id=~Mi_Zhou1">Mi Zhou</a><sup>2</sup>,
    <a href="https://openreview.net/profile?id=~Jinwei_Chen3">Jinwei Chen</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1*</sup>
    <br>
        <sup>1</sup>HKUST,
        <sup>2</sup>vivo Mobile Communication Co., Ltd
</p>

<div align="center">
    <a href='https://arxiv.org/abs/2510.18489'><img src='https://img.shields.io/badge/ArXiv-Paper-b31b1b.svg'></a>  
    <a href='https://liujf1226.github.io/Mono4DGS-HDR/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
    <a href='https://huggingface.co/jinfengliu26/Mono4DGS-HDR/tree/main/datasets'><img src='https://img.shields.io/badge/Preprocessed-Data-blue'></a>
</div>
</p>

<br>

![teaser](https://github.com/user-attachments/assets/0a31a55c-d289-46d5-9b6f-23a71eb0b1cc)

## Demo
Demo videos are available at the [project page](https://liujf1226.github.io/Mono4DGS-HDR/).

## TODO
- [x] Release project page
- [x] Release data and code

## Abstract
> We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demonstrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed.

## Method Overview
![framework](https://github.com/user-attachments/assets/bfb0b520-41c0-4567-a648-2c30ee793b44)

## Setup
### Clone the repo
```shell
git clone https://github.com/LiuJF1226/Mono4DGS-HDR.git --recursive
cd Mono4DGS-HDR
```
### Install dependencies
```shell
conda create -n mono4dgs gcc_linux-64=9 gxx_linux-64=9 python=3.10 cmake=3.14.0 numpy=1.26.4 -y
conda activate mono4dgs
bash install.sh
```

## Data
Download the preprocessed scenes (about 4.3 GB) from this [link](https://huggingface.co/jinfengliu26/Mono4DGS-HDR/tree/main/datasets).

`DATA_PATH` is the folder that directly contains `exp2`, `exp3`, and `syn`:

```text
${DATA_PATH}/
├── exp2/
│   ├── Ninja/
│   ├── sce1/
│   ├── sce2/
│   ├── sce3/
│   ├── sce4/
│   ├── sce5/
│   ├── ThrowingTowel/
│   └── WavingHands/
├── exp3/
│   ├── CheckingEmail/
│   ├── Cleaning/
│   ├── Skateboarder/
│   ├── Dog/
│   ├── sce1/
│   ├── sce2/
│   ├── sce3/
│   └── sce4/
└── syn/
    ├── bridge/
    ├── bridge_2/
    ├── cars/
    ├── fishing/
    ├── hallway/
    ├── students/
    ├── students_2/
    ├── welding/
    └── welding_2/
```

- **`exp2` (Real-Exp-2):** 8 real scenes captured with 2 exposure levels, about 50–60 frames each. With only two exposures, the full sequence is used for training and evaluated on the training frames at the observed exposures.
- **`exp3` (Real-Exp-3):** 8 real scenes captured with 3 exposure levels, about 50–60 frames each. Odd frames are for training and even frames for testing. Test frames are held out and evaluated only at the observed exposures.
- **`syn` (Syn-Exp-3):** 9 synthetic scenes with 3 exposure levels and HDR ground truth, about 100 frames each. Odd frames are for training and even frames for testing, so the test views and timestamps are interpolated. LDR frames are tone-mapped from the HDR ground truth. Test frames also include 2 novel exposure levels.

## Foundation Model Weights
Download the prior checkpoints (about 2.42 GB) from this [link](https://huggingface.co/jinfengliu26/Mono4DGS-HDR/tree/main/prior_weights).

Create a folder at the repository root and put the downloaded files there:

```bash
cd Mono4DGS-HDR
mkdir prior_weights
```

```text
Mono4DGS-HDR/prior_weights/
├── raft-things.pth
├── spaT_final.pth
├── gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
├── bootstapir_checkpoint_v2.pt
├── densetrack3d.pth
├── densetrack2d.pth
└── video_depth_anything_vitl.pth
```

## Training
Precompute depth, flow, tracks, and the initial camera bundle, then train. `train.py` first optimizes camera-space Gaussians and then world-space Gaussians. Evaluation is already included at the end of training. Scene configs are `configs/{split}/{scene}.yaml`. `--gpu` selects the GPU.

```bash
# one scene on GPU 0
python mosca_precompute.py --cfg configs/exp3/Skateboarder.yaml --data_path ${DATA_PATH} --gpu 0
python train.py --cfg configs/exp3/Skateboarder.yaml --data_path ${DATA_PATH} --gpu 0

# another scene on GPU 1
python mosca_precompute.py --cfg configs/syn/fishing.yaml --data_path ${DATA_PATH} --gpu 1
python train.py --cfg configs/syn/fishing.yaml --data_path ${DATA_PATH} --gpu 1
```

Outputs are saved under each scene:

```text
${DATA_PATH}/${split}/${scene}/logs/cam_gs
${DATA_PATH}/${split}/${scene}/logs/world_gs
```

## Acknowledgement
This repo is mainly based on [Splatter a Video](https://github.com/SunYangtian/Splatter_A_Video), [MoSca](https://github.com/JiahuiLei/MoSca), and [GaussHDR](https://github.com/LiuJF1226/GaussHDR). We thank the authors for presenting such excellent works.

## Citation
If you find our work helpful to your research, please cite our paper:
```BibTeX
@inproceedings{liu2025mono4dgshdr,
      title={Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos}, 
      author={Jinfeng Liu and Lingtong Kong and Mi Zhou and Jinwen Chen and Dan Xu},
      booktitle={ICLR},
      year={2026},
}
```
