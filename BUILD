# Description:
#   Contains files for loading, training and evaluating TF-Slim-based models.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "dataset_utils",
    srcs = ["datasets/dataset_utils.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_cifar10",
    srcs = ["datasets/download_and_convert_cifar10.py"],
    deps = [
        ":dataset_utils",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_binary(
    name = "download_and_convert_data",
    srcs = ["download_and_convert_data.py"],
    deps = [
        ":download_and_convert_cifar10",
        # "//tensorflow",
    ],
)

py_binary(
    name = "cifar10",
    srcs = ["datasets/cifar10.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "dataset_factory",
    srcs = ["datasets/dataset_factory.py"],
    deps = [
        ":cifar10",
    ],
)

py_library(
    name = "model_deploy",
    srcs = ["deployment/model_deploy.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "model_deploy_test",
    srcs = ["deployment/model_deploy_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_deploy",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "cifarnet_preprocessing",
    srcs = ["preprocessing/cifarnet_preprocessing.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_preprocessing",
    srcs = ["preprocessing/inception_preprocessing.py"],
    deps = [
        # "//tensorflow",
        # "//tensorflow/python:control_flow_ops",
    ],
)

py_library(
    name = "lenet_preprocessing",
    srcs = ["preprocessing/lenet_preprocessing.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "vgg_preprocessing",
    srcs = ["preprocessing/vgg_preprocessing.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "preprocessing_factory",
    srcs = ["preprocessing/preprocessing_factory.py"],
    deps = [
        ":cifarnet_preprocessing",
        ":inception_preprocessing",
        ":lenet_preprocessing",
        ":vgg_preprocessing",
        # "//tensorflow",
    ],
)

# Typical networks definitions.

py_library(
    name = "nets",
    deps = [
        ":alexnet",
        ":cifarnet",
        ":cyclegan",
        ":inception",
        ":lenet",
        ":mobilenet",
        ":nasnet",
        ":overfeat",
        ":pix2pix",
        ":pnasnet",
        ":resnet_v1",
        ":resnet_v2",
        ":vgg",
    ],
)

py_library(
    name = "alexnet",
    srcs = ["nets/alexnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "alexnet_test",
    size = "medium",
    srcs = ["nets/alexnet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":alexnet",
        # "//tensorflow",
    ],
)

py_library(
    name = "cifarnet",
    srcs = ["nets/cifarnet.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "cyclegan",
    srcs = ["nets/cyclegan.py"],
    deps = [
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "cyclegan_test",
    srcs = ["nets/cyclegan_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":cyclegan",
        # "//tensorflow",
    ],
)

py_library(
    name = "dcgan",
    srcs = ["nets/dcgan.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "dcgan_test",
    srcs = ["nets/dcgan_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":dcgan",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception",
    srcs = ["nets/inception.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_resnet_v2",
        ":inception_v1",
        ":inception_v2",
        ":inception_v3",
        ":inception_v4",
    ],
)

py_library(
    name = "inception_utils",
    srcs = ["nets/inception_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v1",
    srcs = ["nets/inception_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v2",
    srcs = ["nets/inception_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v3",
    srcs = ["nets/inception_v3.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v4",
    srcs = ["nets/inception_v4.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_resnet_v2",
    srcs = ["nets/inception_resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v1_test",
    size = "large",
    srcs = ["nets/inception_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v2_test",
    size = "large",
    srcs = ["nets/inception_v2_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v3_test",
    size = "large",
    srcs = ["nets/inception_v3_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v4_test",
    size = "large",
    srcs = ["nets/inception_v4_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_resnet_v2_test",
    size = "large",
    srcs = ["nets/inception_resnet_v2_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//tensorflow",
    ],
)

py_library(
    name = "lenet",
    srcs = ["nets/lenet.py"],
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet_v1",
    srcs = ["nets/mobilenet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet_v2",
    srcs = glob(["nets/mobilenet/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "mobilenet_v2_test",
    srcs = ["nets/mobilenet/mobilenet_v2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet",
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet",
    deps = [
        ":mobilenet_v1",
        ":mobilenet_v2",
    ],
)

py_test(
    name = "mobilenet_v1_test",
    size = "large",
    srcs = ["nets/mobilenet_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet_v1",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_binary(
    name = "mobilenet_v1_train",
    srcs = ["nets/mobilenet_v1_train.py"],
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_binary(
    name = "mobilenet_v1_eval",
    srcs = ["nets/mobilenet_v1_eval.py"],
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_library(
    name = "nasnet_utils",
    srcs = ["nets/nasnet/nasnet_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "nasnet",
    srcs = ["nets/nasnet/nasnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "nasnet_utils_test",
    size = "medium",
    srcs = ["nets/nasnet/nasnet_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "nasnet_test",
    size = "large",
    srcs = ["nets/nasnet/nasnet_test.py"],
    shard_count = 10,
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet",
        # "//tensorflow",
    ],
)

py_library(
    name = "pnasnet",
    srcs = ["nets/nasnet/pnasnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet",
        ":nasnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "pnasnet_test",
    size = "large",
    srcs = ["nets/nasnet/pnasnet_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":pnasnet",
        # "//tensorflow",
    ],
)

py_library(
    name = "overfeat",
    srcs = ["nets/overfeat.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "overfeat_test",
    size = "medium",
    srcs = ["nets/overfeat_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":overfeat",
        # "//tensorflow",
    ],
)

py_library(
    name = "pix2pix",
    srcs = ["nets/pix2pix.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "pix2pix_test",
    srcs = ["nets/pix2pix_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pix2pix",
        # "//tensorflow",
    ],
)

py_library(
    name = "resnet_utils",
    srcs = ["nets/resnet_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "resnet_v1",
    srcs = ["nets/resnet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "resnet_v1_test",
    size = "medium",
    srcs = ["nets/resnet_v1_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v1",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "resnet_v2",
    srcs = ["nets/resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "resnet_v2_test",
    size = "medium",
    srcs = ["nets/resnet_v2_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v2",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "vgg",
    srcs = ["nets/vgg.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "vgg_test",
    size = "medium",
    srcs = ["nets/vgg_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vgg",
        # "//tensorflow",
    ],
)

py_library(
    name = "nets_factory",
    srcs = ["nets/nets_factory.py"],
    deps = [
        ":nets",
        # "//tensorflow",
    ],
)

py_test(
    name = "nets_factory_test",
    size = "medium",
    srcs = ["nets/nets_factory_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":nets_factory",
        # "//tensorflow",
    ],
)

py_binary(
    name = "train_image_classifier",
    srcs = ["train_image_classifier.py"],
    paropts = ["--compress"],
    deps = [
        ":dataset_factory",
        ":model_deploy",
        ":nets_factory",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_binary(
    name = "eval_image_classifier",
    srcs = ["eval_image_classifier.py"],
    deps = [
        ":dataset_factory",
        ":nets_factory",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_binary(
    name = "export_inference_graph",
    srcs = ["export_inference_graph.py"],
    paropts = ["--compress"],
    deps = [
        ":dataset_factory",
        ":nets_factory",
        # "//tensorflow",
        # "//tensorflow/python:platform",
    ],
)

py_test(
    name = "export_inference_graph_test",
    size = "medium",
    srcs = ["export_inference_graph_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
    ],
    deps = [
        ":export_inference_graph",
        # "//tensorflow",
        # "//tensorflow/python:platform",
    ],
)
