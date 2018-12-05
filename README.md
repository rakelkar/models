# Quantization Friendly MobileNet_v1
(This file has been modified from the source fork, for original content scroll to the end)

This repository contains an implementation of a quantization friendly mobilenet_v1 model based on the reccomendations made in [A Quantization-Friendly Separable Convolution for MobileNets](https://arxiv.org/abs/1803.08607)

The repository is a fork of [Tensorflow Models](https://github.com/tensorflow/models). A modified Tensorflow Slim model [research/slim/nets/mobilenet_v1q](research/slim/nets/mobilenet_v1q.py) has been added. Per the paper this model relu for relu6 and omits batchwise and relu between depthwise and pointwise convolution layers.

## Results
TBD

## Training
### Using an Azure GPU
I checked out an NC12 GPU from Azure. Steps I used to set it up are [here](https://gist.github.com/rakelkar/33ff4b354b735ff3abdd0255163eb028). 

### Get the data
I trained the model on CIFAR10 and ImageNet data. For CIFAR10 the TF slim download script works like a charm, see instructions [here](research/slim/README.md). The instructions should also work for ImageNet data, though you do have to go to the ImageNet site to create an account before you can download. Also the download takes a bit.

### Local folder organization
```
~/train_logs - for training logs, tensorboard points to this folder
~/model_output - to place frozen models
~/repos - somewhere to git clone repos
```

### Running Training
Uses instructions [here](research/slim)

```bash
# replace with wherever you clone repos
export REPO_ROOT=~/repos

# OPTIONAL: also clone TF to get the super useful summarize tool
cd ${REPO_ROOT}/
git clone https://github.com/tensorflow/tensorflow
cd ${REPO_ROOT}/tensorflow
bazel build tensorflow/tools/graph_transforms:summarize_graph

# setup env vars
export DATASET_NAME=cifar10
#export DATASET_NAME=imagenet

# model iteration (udpate the number everytime you train so that you can compare iterations)
export MODEL_ITERATION=${DATASET_NAME}_1

export DATA_DIR=~/data/${DATASET_NAME}
export TRAIN_DIR=~/train_logs/${MODEL_ITERATION}

# get the dataset (only works for flowers or cifar10, for imagenet see readme)
cd ${REPO_ROOT}/models/research/slim/
python download_and_convert_data.py --dataset_name=${DATASET_NAME} --dataset_dir="${DATA_DIR}"

# start tensorboard, if you are using a remote VM via SSH use: ssh -L 16006:127.0.0.1:6006  user@machine
tensorboard --logdir ~/train_logs &

# train! --- and watch tensorboard, press ctrl-C when satisfied with loss
 python train_image_classifier.py \
   --train_dir=${TRAIN_DIR} \
   --dataset_name=${DATASET_NAME} \
   --dataset_split_name=train \
   --dataset_dir=${DATA_DIR} \
   --model_name=mobilenet_v1q

# evaluate the model
python eval_image_classifier.py \
   --alsologtostderr \
   --checkpoint_path=${TRAIN_DIR} \
   --dataset_dir=${DATA_DIR} \
   --dataset_name=${DATASET_NAME} \
   --dataset_split_name=validation \
   --model_name=mobilenet_v1q
   
# export graph for inference (doesnt appear to include weights)
python export_inference_graph.py \
   --model_name mobilenet_v1q \
   --dataset_name ${DATASET_NAME} \
   --output_file ~/model_output/${MODEL_ITERATION}/graph.pb \
   --dataset_dir ${TRAIN_DIR}

# OPTIONAL: figure out output layer (summarize tool built using bazel earlier)
# should report MobilenetV1/Predictions/Reshape_1 as the output_layer
./summarize_graph --in_graph ~/model_output/${MODEL_ITERATION}/graph.pb

# find a checkpoint to use e.g. model.ckpt-77770
ls ${TRAIN_DIR}
export CHECKPOINT_FILE=${TRAIN_DIR}/model.ckpt-77770

# freeze weights into the inference graph
freeze_graph \
    --input_graph ~/model_output/${MODEL_ITERATION}/graph.pb \
    --input_checkpoint ${CHECKPOINT_FILE} \
    --input_binary true \
    --output_graph ~/model_output/${MODEL_ITERATION}/frozen_graph.pb \
    --output_node_names MobilenetV1/Predictions/Reshape_1

# try using the script in TF for poets
cd ${REPO_ROOT}
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd ${REPO_ROOT}/tensorflow-for-poets-2

# download some_image.jpg (look online for an image that you want to inference and wget it)
python -m scripts.label_image \
    --graph ~/model_output/${MODEL_ITERATION}/frozen_graph.pb \
    --image some_image.jpg \
    --output_layer MobilenetV1/Predictions/Reshape_1

# export to view in Tensorboard 
# uses pb_viewer: https://gist.github.com/rakelkar/6a1e45b579c4c8c09725115727f8c6b5
mkdir ~/train_logs/${MODEL_ITERATION}_pb
python pb_viewer.py ~/model_output/${MODEL_ITERATION}/graph.pb ~/train_logs/${MODEL_ITERATION}_pb
```

### Evaluating results

# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)
