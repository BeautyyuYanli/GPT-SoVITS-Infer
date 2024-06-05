# GPT-SoVITS-Infer

This is the inference code of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) that can be developer-friendly.

## Prepare the environment

As we all know, the dependencies of an AI project are always a mess. Here is how I prepare the environment for this project, by conda:

```
conda install python=3.10
conda install pytorch=2.1 torchvision torchaudio pytorch-lightning pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install ffmpeg=6.1.1 -c conda-forge
```

You can also try to prepare the environment with cpu only options, which should work, but I have not tested it yet.

After the environment is ready, you can install the package by pip:

```
pip install GPT-SoVITS
```

I do not add the packages related to torch to the dependencies of GPT-SoVITS-Infer. Check if the environment is ready if things go wrong.

## Usage Example

Check out the [example](example.ipynb) notebook for a quick start.