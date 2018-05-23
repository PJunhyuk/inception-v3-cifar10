# inception-cifar10-rotate

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
