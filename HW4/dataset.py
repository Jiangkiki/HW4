#!/usr/bin/env python
# coding: utf-8

# # Task description
# - Classify the speakers of given features.
# - Main goal: Learn how to use transformer.
# - Baselines:
#   - Easy: Run sample code and know how to use transformer.
#   - Medium: Know how to adjust parameters of transformer.
#   - Hard: Construct [conformer](https://arxiv.org/abs/2005.08100) which is a variety of transformer.

# # Download dataset

# In[ ]:


# From Google drive
# !gdown --id '1T0RPnu-Sg5eIPwQPfYysipfcz81MnsYe' --output Dataset.zip
# !unzip Dataset.zip

# From Dropbox
# If Dropbox is not work. Please use google drive.
# get_ipython().system('wget https://www.dropbox.com/s/vw324newiku0sz0/Dataset.tar.gz.aa?dl=0')
# get_ipython().system('wget https://www.dropbox.com/s/z840g69e7lnkayo/Dataset.tar.gz.ab?dl=0')
# get_ipython().system('wget https://www.dropbox.com/s/hl081e1ggonio81/Dataset.tar.gz.ac?dl=0')
# get_ipython().system('wget https://www.dropbox.com/s/fh3zd8ow668c4th/Dataset.tar.gz.ad?dl=0')
# get_ipython().system('wget https://www.dropbox.com/s/ydzygoy2pv6gw9d/Dataset.tar.gz.ae?dl=0')
# get_ipython().system('cat Dataset.tar.gz.* | tar zxvf -')


# # Data

# ## Dataset
# - Original dataset is [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
# - The [license](https://creativecommons.org/licenses/by/4.0/) and [complete version](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt) of Voxceleb1.
# - We randomly select 600 speakers from Voxceleb1.
# - Then preprocess the raw waveforms into mel-spectrograms.
#
# - Args:
#   - data_dir: The path to the data directory.
#   - metadata_path: The path to the metadata.
#   - segment_len: The length of audio segment for training.
# - The architecture of data directory \\
#   - data directory \\
#   |---- metadata.json \\
#   |---- testdata.json \\
#   |---- mapping.json \\
#   |---- uttr-{random string}.pt \\
#
# - The information in metadata
#   - "n_mels": The dimention of mel-spectrogram.
#   - "speakers": A dictionary.
#     - Key: speaker ids.
#     - value: "feature_path" and "mel_len"
#
#
# For efficiency, we segment the mel-spectrograms into segments in the traing step.

# In[ ]:


import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class myDataset(Dataset):
  def __init__(self, data_dir, segment_len=128):
    self.data_dir = data_dir
    self.segment_len = segment_len

    # Load the mapping from speaker neme to their corresponding id.
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    self.speaker2id = mapping["speaker2id"]

    # Load metadata of training data.
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = json.load(open(metadata_path))["speakers"]

    # Get the total number of speaker.
    self.speaker_num = len(metadata.keys())
    self.data = []
    for speaker in metadata.keys():
      for utterances in metadata[speaker]:
        self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    feat_path, speaker = self.data[index]
    # Load preprocessed mel-spectrogram.
    mel = torch.load(os.path.join(self.data_dir, feat_path))

    # Segmemt mel-spectrogram into "segment_len" frames.
    if len(mel) > self.segment_len:
      # Randomly get the starting point of the segment.
      start = random.randint(0, len(mel) - self.segment_len)
      # Get a segment with "segment_len" frames.
      mel = torch.FloatTensor(mel[start:start+self.segment_len])
    else:
      mel = torch.FloatTensor(mel)
    # Turn the speaker id into long for computing loss later.
    speaker = torch.FloatTensor([speaker]).long()
    return mel, speaker

  def get_speaker_number(self):
    return self.speaker_num
