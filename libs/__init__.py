

import os
import sys
import heapq
import bisect
import pickle
import json
import math
import statistics as stat
import time
import datetime as dt
import logging
import random
import threading as td
import multiprocessing as mp
import re
import unittest as ut
import platform

#
from warnings import filterwarnings
from operator import itemgetter
from pprint import pprint
from functools import partial, cache, lru_cache, cmp_to_key
from types import SimpleNamespace
from collections import deque, namedtuple, OrderedDict, defaultdict, Counter
from copy import copy, deepcopy
from argparse import ArgumentParser, Namespace
from queue import Queue, SimpleQueue, PriorityQueue
from hashlib import sha256
from typing import (
    Literal, List, Tuple, Dict, Set, Callable, Optional, Union, Any,
    Deque, Sequence, Mapping, Iterable, Iterator, DefaultDict, overload
)
from typing_extensions import TypeAlias
from xml.etree import ElementTree as ET
from contextlib import contextmanager

#
import yaml
import requests
from sortedcontainers import SortedList, SortedDict, SortedSet
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import RandomState
from pandas import DataFrame, Series
#

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from PIL import Image
import cv2 as cv
#
import sklearn
import torch
from torch import Tensor, device as Device
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler as lrs
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import torch.nn.init as init
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset, TensorDataset
import torch.utils.data as tud
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys
import torch.distributed as dist
#
import torchvision.transforms.functional_tensor as tvF_t
import torchvision.transforms.functional_pil as tvF_pil
import torchvision.transforms.functional as tvF
from torchvision.transforms.functional import InterpolationMode, pil_modes_mapping
import torchvision as tv
import torchvision.transforms as tvt
import torchvision.datasets as tvd
from torchvision.utils import make_grid
import torchvision.models as tvm
#
# import pytorch_lightning as pl
# import pytorch_lightning.callbacks as plc
#
from transformers.pipelines import pipeline
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from datasets.load import load_dataset, load_metric
#
from torchmetrics import MeanMetric, Metric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.f_beta import F1Score, FBetaScore
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.average_precision import AveragePrecision
# torchmetrics.functional 使用libs_ml中的. (更快)
#
import gym
from gym import Env
#
from . import utils as libs_utils
from . import ml as libs_ml
from . import alg as libs_alg
#
Number = Union[int, float]
# 以下环境变量需要用户自定义设置, 这里为了自己方便进行导入
TORCH_HOME = os.environ.get("TORCH_HOME", None)
DATASETS_PATH = os.environ.get("DATASETS_PATH", None)
HF_HOME = os.environ.get("HF_HOME", None)
