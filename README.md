## Consistency Distillation | Training example on MS-COCO

### 1. Set up the environment

```shell
conda create -n cd-train python=3.10 -y 
conda activate cd-train

pip3 install -r requirements.txt
```

If you have enough computational resources (one GPU with 16 GiB is enough) for training, do the following:

### 2.1 Download the data (12 GB)

```shell
. data/download_coco_train2014.sh
```
We provide the training example on the [MS-COCO](https://cocodataset.org/) dataset:
[train2014.tar.gz](https://storage.yandexcloud.net/yandex-research/invertible-cd/train2014.tar.gz) - contains the original COCO2014 train set.

### 3.1 Run the training

```shell
. run_training.sh
```

If you do not have  computational resources but want to generate images, 
we've prepared the distilled model checkpoints (multi boundary version)

### 2.2 Download the checkpoint (3.2 GB)
```shell
. data/download_checkpoint_4steps.sh
```

### 3.2 Go to the ```run_inference.ipynb``` and play!
We expect something like:
<p align="center">
<img src="stuff/imgs.jpg" width="1080px"/>
</p>