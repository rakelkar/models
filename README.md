# Quantization Friendly MobileNet_v1

This repository contains an implementation of a quantization friendly mobilenet_v1 model based on the reccomendations made in [A Quantization-Friendly Separable Convolution for MobileNets](https://arxiv.org/abs/1803.08607)

The repository is a fork of [Tensorflow Models](https://github.com/tensorflow/models). A modified Tensorflow Slim model [research/slim/nets/mobilenet_v1q](research/slim/nets/mobilenet_v1q.py) has been added. Per the paper this model relu for relu6 and omits batchwise and relu between depthwise and pointwise convolution layers.

## Results
TBD

## Training
TBD

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
