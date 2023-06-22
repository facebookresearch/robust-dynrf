# Copyright (c) Meta Platforms, Inc. and affiliates.

from .nvidia import NvidiaDataset
from .davis import DavisDataset

dataset_dict = {"nvidia": NvidiaDataset, "davis": DavisDataset}
