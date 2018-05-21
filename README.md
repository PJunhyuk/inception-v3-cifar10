# inception-cifar10-rotate

## Using Docker

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

#### Train

```
~/inception-cifar10-rotate# python train_image_classifier.py --train_dir=/tmp/models/inception_v3 --dataset_name=cifar10 --dataset_split_name=train --dataset_dir=/tmp/data/cifar10 --model_name=inception_v3 --clone_on_cpu=True
```

#### Download pre-trained model

```
/tmp/models# wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
```

#### Fine-tuning

```
> python train_image_classifier.py --train_dir=/tmp/models/pre_inception_v3 --dataset_dir=/tmp/data/cifar10 --dataset_name=cifar10 --dataset_split_name=train --model_name=inception_v3 --checkpoint_path=/tmp/models/inception_v3.ckpt --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --clone_on_cpu=True
```

## Using Windows

#### TensorBoard

```
> tensorboard --logdir=/tmp/models/inception_v3
```

## Results

#### Total required time

###### only CPU: Intel(R) Xeon(R) CPU E5-2687W v3 @ 3.10GHz

> Train: 6.5xx sec/step  

###### only CPU: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz

> Train: 25.xxx sec/step  
> Fine-tuning: 6.xxx sec/step  

###### only CPU: Intel(R) Core(TM) i5-6600 CPU @ 3.30GHz

> Train: 41.xxx sec/step  
