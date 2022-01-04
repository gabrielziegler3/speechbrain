"""
This file contains a very simple TDNN module to use for speaker-id.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.

Authors
 * Nauman Dawalatabad 2020
 * Mirco Ravanelli 2020
"""

import torch  # noqa: F401
import speechbrain as sb
import os

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.utils.parameter_transfer import Pretrainer


def get_pretrained_model(pretrained_model_dir, device="cpu"):
    model = ECAPA_TDNN(input_size= 80,
                       channels= [1024, 1024, 1024, 1024, 3072],
                       kernel_sizes= [5, 3, 3, 3, 1],
                       dilations= [1, 2, 3, 4, 1],
                       attention_channels= 128,
                       lin_neurons = 192)
    # Initialization of the pre-trainer
    pretrain = Pretrainer(
        collect_in=pretrained_model_dir,
        loadables={'model': model},
        paths={'model': os.path.join(pretrained_model_dir, "embedding_model.ckpt")}
    )

    # We download the pretrained model from HuggingFace in this case
    pretrain.collect_files()
    pretrain.load_collected(device=device)
    return model.to(device)

