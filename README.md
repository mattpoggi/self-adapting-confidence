# Self-adapting confidence estimation for stereo

Demo code of "Self-adapting confidence estimation for stereo", [Matteo Poggi](https://mattpoggi.github.io/), [Filippo Aleotti](http://filippoaleotti.github.io/website/), [Fabio Tosi](https://vision.disi.unibo.it/~ftosi/), Giulio Zaccaroni and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html), ECCV 2020. 

### License
Copyright (c) 2020 University of Bologna. Patent pending. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode ).

<p align="center"> 
<img src=https://mattpoggi.github.io/assets/img/confidence/poggi2020eccv.gif>
</p>

[[Paper]](https://mattpoggi.github.io/assets/papers/poggi2020eccv.pdf)

## Citation
```
@inproceedings{Poggi_ECCV_2020,
  title     = {Self-adapting confidence estimation for stereo},
  author    = {Poggi, Matteo and
               Aleotti, Filippo and
               Tosi, Fabio and
               Zaccaroni, Giulio and
               Mattoccia, Stefano},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```   

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)
5. [Acknowledgements](#acknowledgements)

## Introduction

Estimating the confidence of disparity maps inferred by a stereo algorithm has become a very relevant task in the years, due to the increasing number of applications leveraging such cue. Although self-supervised learning has recently spread across many computer vision tasks, it has been barely considered in the field of confidence estimation. In this paper, we propose a flexible and lightweight solution enabling self-adapting confidence estimation agnostic to the stereo algorithm or network. Our approach relies on the minimum information available in any stereo setup (i.e., the input stereo pair and the output disparity map) to learn an effective confidence measure. This strategy allows us not only a seamless integration with any stereo system, including consumer and industrial devices equipped with undisclosed stereo perception methods, but also, due to its self-adapting capability, for its out-of-the-box deployment in the field. Exhaustive experimental results with different standard datasets support our claims, showing how our solution is the first-ever enabling online learning of accurate confidence estimation for any stereo system and without any requirement for the end-user.

## Usage

### Requirements

* `Tensorflow 1.x` (tested with Tensorflow 1.8 and python 2.7)
* `progressbar2`
* `numpy`
* `cv2`

### Getting started

Download DrivingStereo sequence and pretrained models

DrivingStereo sequence `2018-10-25-07-37-26`:
[left images](https://drive.google.com/file/d/1yPO5UzEWJRHUav8olyHQQkEWBn-aZizA) - [right images](https://drive.google.com/file/d/1mkGQICQ_uucArzCtiVG5oxJLTOGRPsfT)

Pre-computed disparity maps for sequence `2018-10-25-07-37-26`:
[Census-SGM](https://drive.google.com/file/d/1f3tbFIxdmxrPjvtFpeO7AaXr45cPBKVZ) - [MADNet](https://drive.google.com/file/d/1UnuAQkpidaTqfQiXcHkHKl8H6wsPNbz2) - [GANet](https://drive.google.com/file/d/1-VblYLxOlV1Z91STs2w6VSFfbILBXoVb)

ConfNet weights trained on KITTI:
[Checkpoints](https://drive.google.com/file/d/1O7kFOxoz7D3-q3fkZo_azcbJTht1BvLh)

### Run the demo

Launch the following command to play over the DrivingStereo sequence (or your custom data)

```shell
python main.py --mode [mode] \
               --checkpoint_path [checkpoint_path] \
               --output_path [output_path] \
               --left_dir [left_dir] \
               --right_dir [right_dir] \
               --disp_dir [disp_dir] 
```

Main arguments:
* `--mode`: choose the confidence map you want to generate: `reprojection`, `agreement`, `uniqueness`, `otb`, `otb-online`
* `--checkpoint_path`: path to ConfNet weights (required for `otb` and `otb-online`)
* `--output_path`: path for storing results (one subfolder per image will be created, confidence maps will be saved as `--mode`.png)
* `--left_dir`: path to left images
* `--right_dir`: path to right images
* `--disp_dir`: path to disparity maps

Optional arguments:
* `--color`: also saves confidence maps with cv2 WINTER colormap
* `--cpu`: runs on CPU
* `--mem`: limits GPU memory usage
* `--initial_learning_rate`: learning rate for online adaptation 
* `--p`: list of `P` criterions `t`, `a`, `q`, in case of adaptation at least one is required  
* `--q`: list of `Q` criterions `t`, `a`, `q`, in case of adaptation at least one is required

Custom arguments for running on your data:
* `--image_height`: image height (must be multiple of 16 and larger than your images)
* `--image_width`: image width (must be multiple of 16 and larger than your images)
* `--dataset`: a `.txt` file listing the the name.format of your images (one per line)

In case of custom data, left, right and disparity maps are assumed to have the same name in different folders, disparity maps are assumed to be 16 bit `.png` images.
(e.g., left/001.jpg, right/001.jpg, disp/001.png. The `.txt` file should contain 001.jpg)

### Evaluation

Download DrivingStereo sequence `2018-10-25-07-37-26` [ground truth maps](https://drive.google.com/file/d/1UV109ysB8-kjct-JX5Jv0y6MKyPPC6D_)


```shell
python AUC.py --gt_path [gt_path] \
              --disp_path [disp_path] \
              --conf_path [conf_path] \
              --conf_name [measure]
```

Arguments:
* `--gt_path`: path to ground truth maps
* `--disp_path`: path to disparity maps
* `--conf_path`: path to estimated confidence maps
* `--conf_name`: confidence maps name

For each ground truth map `gt_path/map_name.png`, the script will look for a disparity map `disp_path/map_name.png` and a confidence map `map_name/conf_name.png` and compute average AUC.

Optional arguments:
* `--tau`: tau threshold to identify outliers
* `--intervals`: number of intervals for ROC curve computation
* `--logger`: output file where to store single images AUC

## Results

By running the `AUC.py` script, you should be able to reproduce the following results (Census-SGM):

```
Measure: reprojection   & Avg. bad3: 21.007%    & Opt. AUC: 0.029       & Avg. AUC: 0.179 \\
Measure: agreement      & Avg. bad3: 21.007%    & Opt. AUC: 0.029       & Avg. AUC: 0.106 \\
Measure: uniqueness     & Avg. bad3: 21.007%    & Opt. AUC: 0.029       & Avg. AUC: 0.162 \\
Measure: otb            & Avg. bad3: 21.007%    & Opt. AUC: 0.029       & Avg. AUC: 0.072 \\
Measure: otb-online     & Avg. bad3: 21.007%    & Opt. AUC: 0.029       & Avg. AUC: 0.064 \\
```

## Contacts
m [dot] poggi [at] unibo [dot] it

## Acknowledgements

Most of the code is derived from `LGC-Tensorflow` repository: https://github.com/fabiotosi92/LGC-Tensorflow
