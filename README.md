<h1 align="center">Defend against adversarial attacks in face recognition models</h1>
<h2 align="center">
  <br>
Final project <br>
Technion's EE <br>
Deep Learning course (046211)
  <br>
  <img src="https://media.tenor.com/B8ra2i-OK9QAAAAC/face-recognition.gif" height="200">
</h1>
  <p align="center">
    Gil Kapel: <a href="https://www.linkedin.com/in/gil-kapel-a960b720a/">LinkdIn</a> , <a href="https://github.com/gil-kapel">GitHub</a>
  <br>
    Shelly Francis: <a href="https://www.linkedin.com/in/shelly-francis-85bb2b217/">LinkdIn</a> , <a href="https://github.com/shellyfra">GitHub</a>
  </p>

## Agenda
- [Face Recognition Attacks](#Face-Recognition-Attacks) - About our project
- [Training and Results](#training-and-results) - our network's training visualizations and results
- [Hyper-parameters](#hyper-parameters) - what are our training's hyperparameters
- [Run our model](#run-our-model) - how to run training jobs and inference with our model and how to load checkpoints

# Face-Recognition-Attacks
- Face recognition is a domain with many uses in real-world applications – ranging from photo tagging to automated border control (ABC).
- Today’s models have high accuracy
- These models could be vulnerable to adversarial attacks
- An attacker can intentionally design features that would confuse the network and even impersonate to someone else.
- We tested some countermeasures against these attacks

![](images/att_attack_id_108.png)



# Training and Results

We used Optuna to pick our hyperparameters for .

Here are our results:

# Run our model

## Dataset
Our dataset is: <a href="https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/code">GTZAN dataset </a>,
Our code uses `torchaudio` dataset to load it. You can set the path to your data directory with the data_dir argument.

## Chekpoints
You should set the ckpt_dir parameter as the father checkpoints directory, and ckpt_file as the file name.
For example, if you set the following parameters as:
`ckpt_dir = "checkpoints"`, `test_name = "my_test.pt"`, `ckpt_dir = "best_ckpt.pt"`,
The full checkpoints file path that will be loaded is: `\checkpoints\my_test\best_ckpt.pt`

## Training Music Genre Classifier
To train our classifier network, run `train_env.py`.
```cmd
python ./train_env.py --test_name run_basic
```
Training job parameters are automatically loaded from the options.json in the project directory.
Changes to the parameters can be applied by changing the `codes\options.json` or running with command line arguments, for example:
```cmd
python ./train_env.py --test_name run_learn_window --learn_window 1
```

## Inference Music Genre Classifier
Run the `test.py` with the `test_name` argument set to the name of the model used for inference.
Setting the `test_name` argument can be done through `options.json` or through command line:
```cmd
python ./test.py --test_name my_test --ckpt_dir checkpoints --ckpt_dir best_ckpt.pt
```

# Hyper-parameters

|Parameter | Type | Description |
|-------|------|---------------|
|test_name| string | your trial's name|
|resume| int | 0 if we start a new training run and 1 if we resume old training|
|ckpt_interval| int | epoch interval to save a new checkpoint |
|tensorboard_dir| string | path to tensorboard log directory |
|data_dir| string | path to dataset directory |
|ckpt_dir| string | path to checkpoint directory |
|ckpt_file| string | path to ckpt file to be loaded |
|learn_window| int | 1 to learn stft window coefficients, 0 not to |
|learn_kernels| int | 1 to learn stft kernels coefficients, 0 not to |
|batch_size| int | size of batch |
|num_workers| int | data loader's parameters: number of workers to pre-fetch the data |
|epoch_num| int | number of total epoches to run |
|learning_rate| int | initial optimizer's learning rate |
|split_parts| int | how many parts to split our original audio file to. can be: 1, 3, 4, 6, 12|
|gamma| int | scheduler's gamma |
|cpu| int | 0 if we want to try and run on gpu, else if we want to run on cpu |
|augmentation | int | 0 if we don't want to use augmentation, else if we do |
|three_widows| int | 0 to use 1 STFT in classifier (greyscale), else for 3 STFT modules in classifier (RGB) |
|optimizer_class| string | optimizer type: "SGD" or "AdamW" |

## Changing hyper-parameters
Parameters are automatically loaded from the options.json in the project directory.
Changes to the parameters can be applied by changing the `options.json`.
We also  implemented argparse library, so you can load your parameters with your IDE's configure or within th command line.
Examples are shown in the Run-our-model section.

# Ada-STFT Module

## How to use our module
```python
import torch
from torch import nn
from models.resnet_dropout import *
import STFT

class Classifier(nn.Module):
    def __init__(self, resnet=resnet18, window="hanning", num_classes=10):
        super(Classifier, self).__init__()
        self.stft = STFT(window=window)
        self.resnet = resnet(num_classes=num_classes)

    def forward(self, x):
        x = self.stft(x)
        x = self.monochrome2RGB(x)
        return self.resnet(x)

    @staticmethod
    def monochrome2RGB(tensor):
        return tensor.repeat(1, 3, 1, 1)
```

## STFT Layer Parameters
|Parameter | Description |
|-------|---------------------|
|nfft| window size of STFT calculation|
|hop_length | STFT hop size, or stride of STFT calculation|
| window | type of window to initialize the STFT window to, one of the windows implemented in scipy.signal|
| sample_rate | sampling rate for audio|
| num_mels | number of mel scale frequencies to use, None for don't use mel frequencies|
| log_base | base of log to apply  to STFT, None for no log|
| learn_window | should window be learned (can be set after layer initialization)|
| learn_kernels | should DFT kernel be learned (can be set after layer initialization)|


# Prerequisites
|Library         | Version |
|--------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`scipy`| `1.7.3`|
|`tqdm`| `4.62.3`|
|`librosa`| `0.8.1`|
|`torch`| `1.10.1`|
|`torchaudio`| `0.10.1`|
|`torchaudio-augmentations`| `0.2.3 (https://github.com/Spijkervet/torchaudio-augmentations)`|
|`tensorboard`| `2.7.0`|

Credits:
* Music Genre Classifier Project for classifier network architecture https://github.com/omercohen7640/MusicGenreClassifier
* Animation by <a href="https://medium.com/@gumgumadvertisingblog">GumGum</a>.
* STFT implemenation https://github.com/diggerdu/pytorch_audio
