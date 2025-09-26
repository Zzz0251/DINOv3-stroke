# DINOv3-stroke

Code for "Benchmarking DINOv3 for Multi-Task Stroke Analysis on Non-Contrast CT"


## Abstract

We used the frozen DINOv3-vitb as the backbone and employed simple decoders for evaluations on many stroke-related tasks using non-contrast CT scans.


## Features

- Stroke analysis based on NCCT
- Support for multiple stroke-related tasks (segmentation, classification)
- Using frozen pre-trained models and mostly public evaluation benchmarks


## Quick Start

### Prerequisites and Installation

The repo is based on official DINOv3, so downloading DINOv3 repo in path, downlowding official pretrained weights, and buliding a virtual env as the DINOv3 requires are enough.

### Data preprocess

For volume datasets, we extract the slices with exact labels, and organize the image and label in a RGB format: put the image into R-channel, label into G-channel. So the dataset path, you can see in the code that we have only "/path/to/Dataset/train" and "/path/to/Dataset/test".

### Basic use

Once you have assert the path of data and model, you can simplely use one of the code in folder "decoders" to try on your own data. The command is simple too, for example:

```bash
python ./decoders/Seg1_UP_test.py
```

## Acknowledgments

- Built upon DINOv3 by Meta AI
- Thanks to the medical imaging community for open datasets
- Special thanks to clinical collaborators

