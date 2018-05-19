# inception-cifar10-rotate

```
> python download_and_convert_data.py --dataset_name=cifar10 --dataset_dir="/tmp/data/cifar10"
```

```
> python train_image_classifier.py --train_dir=/tmp/train_logs --dataset_name=cifar10 --dataset_split_name=train --dataset_dir=/tmp/data/cifar10 --model_name=inception_v3
```

```
> tensorboard --logdir=/tmp/train_logs
```

## Dependences
-
