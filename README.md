<p align="center">
<h1 align="center"><strong>Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos</strong></h1>
<h3 align="center">   </h3>

<p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=-moPItwAAAAJ">Jinfeng Liu</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=KKzKc_8AAAAJ">Lingtong Kong</a><sup>2</sup>,
    <a href="https://openreview.net/profile?id=~Mi_Zhou1">Mi Zhou</a><sup>2</sup>,
    <a href="https://openreview.net/profile?id=~Jinwei_Chen3">Jinwei Chen</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1</sup>
    <br>
        <sup>1</sup>HKUST,
        <sup>2</sup>vivo Mobile Communication Co., Ltd
</p>

<div align="center">
    <a href=''><img src='https://img.shields.io/badge/ArXiv-Paper-b31b1b.svg'></a>  
    <a href='https://liujf1226.github.io/Mono4DGS-HDR/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
    <a href=''><img src='https://img.shields.io/badge/Preprocessed-Data-blue'></a>  
    <!-- <a href='https://drive.google.com/file/d/1uaBfv_9boxl9pl3IMED5WIGcbsZMjUS9/view?usp=drive_link'><img src='https://img.shields.io/badge/Pretrained-Models-orange'></a>  -->
</div>
</p>

<br>

![teaser](https://github.com/user-attachments/assets/0a31a55c-d289-46d5-9b6f-23a71eb0b1cc)
Demo videos are available at the [project page](https://liujf1226.github.io/Mono4DGS-HDR/).

## TODO
- [x] Release project page
- [ ] Release data and code

## Abstract
> We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demonstrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed.

## Method Overview
![framework](https://github.com/user-attachments/assets/bfb0b520-41c0-4567-a648-2c30ee793b44)
