# inception-cifar10-rotate

Based on [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).  

## Install

#### Pull Docker image
```
$ docker pull tensorflow/tensorflow:1.7.0-py3
$ docker run -it --name {docker-name} tensorflow/tensorflow:1.7.0-py3 /bin/bash
```

#### Pull GitHub repository
```
~# apt-get update
~# apt-get install git
~# git clone https://github.com/PJunhyuk/inception-cifar10-rotate
```

#### Download dataset
```
~/inception-cifar10-rotate# python download_and_convert_data.py --dataset_name=cifar10 --dataset_dir="/tmp/data/cifar10"
```

## Usage

#### Train
```
~/inception-cifar10-rotate# python train_image_classifier.py --train_dir=/tmp/models/inception_v3 --dataset_name=cifar10 --dataset_split_name=train --dataset_dir=/tmp/data/cifar10 --model_name=inception_v3 --clone_on_cpu=True
```

#### Evaluate
```
~/inception-cifar10-rotate# python eval_image_classifier.py --alsologtostderr -checkpoint_path=/tmp/models/inception_v3/model.ckpt-14457 --dataset_dir=/tmp/data/cifar10 --dataset_name=cifar10 --dataset_split_name=test --model_name=inception_v3 --clone_on_cpu=True
```

#### Download&Unzip pre-trained model
> Official

```
/tmp/models# wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
/tmp/models# tar -xvzf inception_v3_2016_08_28.tar.gz
```

> Self trained (26583 steps)

```
/tmp/models# wget https://www.dropbox.com/s/19ckps2vp0b0t1l/model.ckpt-26583.tar.gz?dl=0
/tmp/models# tar -xvzf model.ckpt-26583.tar.gz
```

#### Fine-tuning
```
> python train_image_classifier.py --train_dir=/tmp/models/pre_inception_v3 --dataset_dir=/tmp/data/cifar10 --dataset_name=cifar10 --dataset_split_name=train --model_name=inception_v3 --checkpoint_path=/tmp/models/inception_v3.ckpt --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --clone_on_cpu=True
```

#### TensorBoard
```
> tensorboard --logdir=/tmp/models/inception_v3
```

#### Copy
> Docker to server

```
$ /docker cp {docker-name}:/tmp/models/inception_v3 .
```

> Server to local

```
$ scp -r {user-name}@{server-ip}:{server-directory} .
```

#### Check weights in trained checkpoint file
> Check `check.py`

###### all tensors

```
from tensorflow.python.tools import inspect_checkpoint
inspect_checkpoint.print_tensors_in_checkpoint_file('/tmp/models/inception_v3/model.ckpt-20', tensor_name='', all_tensors=True, all_tensor_names=True)
```

###### specific tensor

```
from tensorflow.python.tools import inspect_checkpoint
inspect_checkpoint.print_tensors_in_checkpoint_file('/tmp/models/inception_v3/model.ckpt-20', tensor_name='InceptionV3/Conv2d_2a_3x3/BatchNorm/beta/RMSProp_1', all_tensors=False)
```

## Results

#### Total required time

###### only CPU: Intel(R) Xeon(R) CPU E5-2687W v3 @ 3.10GHz
> Train: 6.5xx sec/step  
> Fine-tuning: 1.3xx sec/step

###### only CPU: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
> Train: 25.xxx sec/step  
> Fine-tuning: 6.xxx sec/step  

###### only CPU: Intel(R) Core(TM) i5-6600 CPU @ 3.30GHz
> Train: 41.xxx sec/step  

#### Accuracy

###### inception_v3 - train (14457 steps)
- eval/Accuracy[0.6713]  
- eval/Recall_5[0.9622]  

###### inception_v3 - train (26583 steps)
- eval/Accuracy[0.7976]  
- eval/Recall_5[0.9904]  

## Descriptions
#### Tensorflow-slim
A lightweight high-level API of TensorFlow for defining, training and evaluating complex models.  
[TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).  

#### Image Resize
<img src="/src/src_1.PNG" width="600">  

Resize image size from 32x32x3 -> 299x299x3.  

#### Network structure  

| type | patch size/stride | input size |
|:-:|:-:|:-:|:-:|
| conv        | 3x3/2 | 299x299x3  |
| conv        | 3x3/1 | 149x149x32 |
| conv padded | 3x3/1 | 147x147x32 |
| pool        | 3x3/2 | 147x147x64 |
| conv        | 3x3/1 | 73x73x64   |
| conv        | 3x3/2 | 71x71x80   |
| conv        | 3x3/1 | 35x35x192  |
| 3xInception | -     | 35x35x288  |
| 5xInception | -     | 17x17x768  |
| 2xInception | -     | 8x8x1280   |
| pool        | 8x8   | 8x8x2048   |
| linear      | logits| 1x1x2048   |
| softmax     | classifier | 1x1x1000 |

<img src="/src/src_2.png" width="300">  

> You can check more detail descriptions in paper - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).

You can check codes in [inception_v3.py](https://github.com/PJunhyuk/inception-cifar10-rotate/blob/master/nets/inception_v3.py).  

- `padding` in `slim.arg_scope` can get `VALID` or `SAME`(default). [difference between 'SAME' and 'VALID'](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t).
- Each Inception blocks has several branches, and it combined for net by using `tf.concat` function. Depth of net(combined) is sum of all branches' depth.
  - ex) In `Mixed_5b`, `256 = 64 + 48 + 96 + 32`.
