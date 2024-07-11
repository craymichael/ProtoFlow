# ProtoFlow: An Invertible Prototypical Neural Network
This repository contains code for the paper, "This Probably Looks _Exactly_ Like That: An Invertible Prototypical
Network," which was accepted to ECCV 2024. The proposed architecture, **ProtoFlow**, represents prototypical
distributions as Gaussians in the latent space of a normalizing flow. The approach enables rich interpretation,
effective uncertainty estimation, and a research path forward for intrinsically interpretable neural networks.

## Abstract

> We combine concept-based neural networks with generative, flow-based classifiers into a novel, intrinsically
explainable, exactly invertible approach to supervised learning. Prototypical neural networks, a type of concept-based
neural network, represent an exciting way forward in realizing human-comprehensible machine learning without concept
annotations, but a human-machine semantic gap continues to haunt current approaches. We find that reliance on indirect
interpretation functions for prototypical explanations imposes a severe limit on prototypes' informative power. From
this, we posit that invertibly learning prototypes as distributions over the latent space provides more robust,
expressive, and interpretable modeling. We propose one such model, called ProtoFlow, by composing a normalizing flow
with Gaussian mixture models. ProtoFlow (1) sets a new state-of-the-art in joint generative and predictive modeling and
(2) achieves predictive performance comparable to existing prototypical neural networks while enabling richer
interpretation.


# Installation

Install the requirements in `requirements.txt` as follows:

```shell
pip install requirements.txt
```

Alternatively, the exact environment that was used in this research can be reproduced using conda. After installing
conda, create a new environment using the provided `environment.yml`:
```shell
conda env create -f environment.yml
```

# Training
To train an instance of ProtoFlow, the `train.py` script should be used. Run `python train.py --help` for usage details.
You will want the DenseFlow pretrained checkpoints to initialize the model, which can be downloaded following the
instructions [here](https://github.com/matejgrcic/DenseFlow?tab=readme-ov-file#model-weights).
For example, ProtoFlow can be trained on CIFAR-10 as follows:
```shell
# Optionally enter your dataset root here
#export DATASET_ROOT='/mnt/data/ml_datasets/'
python train.py --flow_ckpt checkpoints/denseflow/imn32/imagenet32/ \
  --img_size 32 \
  --dataset cifar10 \
  --extra my_test_run \
  -e 10 \
  --batch_steps 32 \
  --batch_size 256 \
  --trainable all \
  --lr 2e-4 \
  --gmm_lr 2e-3 \
  --consistency_loss \
  --protos_per_class 5 \
  --elbo_loss2
```

To run using PyTorch DDP (distributed/parallel training), you can use the following:
```shell
torchrun --nproc_per_node=2 train.py ...
```

# Testing
To train an instance of ProtoFlow, the `test.py` script should be used. Run `python test.py --help` for usage details.
A pre-trained model can be downloaded (see proceeding section) and be evaluated using this script. For example:
```shell
# Optionally enter your dataset root here
#export DATASET_ROOT='/mnt/data/ml_datasets/'
python test.py --resume checkpoints/cifar10/checkpoint.pt \
  --tta \
  --tta_num 5 \
  --num_samples 5 \
  --proto_scores
```

If you run out of GPU VRAM, adjust the `--batch_size`.

# Pre-trained Models

All checkpoints and configurations (including hyperparameters) for trained models are available
[here](https://drive.google.com/drive/folders/1cu4aFc7uQMF4YfY_eipEQX_Pxi20PUOT?usp=sharing).

# License

This repository is distributed under the GNU GPL v2.0 License.

This repository contains code from the following projects:
- [`kmeans_pytorch`](https://github.com/subhadarship/kmeans_pytorch) (MIT License): `./protoflow/kmeans/`
- [`DenseFlow`](https://github.com/matejgrcic/DenseFlow) (GNU GPL v2.0): `./denseflow/` and `./experiments/`
- [`gmm-torch`](https://github.com/ldeecke/gmm-torch) (MIT License): `./protoflow/gmm.py`

# Citation

```text
@inproceedings{protoflowECCV2024,
    author    = {Carmichael, Zachariah and
                 Redgrave, Timothy and
                 Gonzalez Cedre, Daniel and
                 Scheirer, Walter J.},
    title     = {This Probably Looks Exactly Like That: An Invertible Prototypical Network},
    booktitle = {European Conference on Computer Vision},
    year      = {2024},
    publisher = {Springer Nature},
}
```

# obligatory miata
![miat](miata.png "miat")
