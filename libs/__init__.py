

import pickle
import json
import warnings
from functools import partial
import math
from types import SimpleNamespace
import os
import heapq
from collections import deque, namedtuple, OrderedDict, defaultdict, Counter
from copy import copy, deepcopy
from typing import List, Tuple, Dict, Set, Callable, Optional, \
    Union, Any, Deque, Sequence, Mapping, Iterable, Iterator
import sys
import time
import datetime
# import random
from tqdm import tqdm
from pprint import pprint
import logging
#
from numpy import ndarray
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
Number = Union[int, float]

#
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from PIL import Image
#
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
#
import torchvision.transforms.functional_tensor as tvF_t
import torchvision.transforms.functional_pil as tvF_pil
import torchvision.transforms.functional as tvF
import torchvision.transforms as tvt
from torchvision.transforms.functional import InterpolationMode, pil_modes_mapping
import torchvision as tv
import torchvision.transforms as tvt
import torchvision.datasets as tvd
from torchvision.utils import make_grid
import torchvision.models as tvm
#
import sklearn
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
#
from transformers.pipelines import pipeline
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from datasets.load import load_dataset, load_metric
#
import gym
from gym import Env
#
from . import utils as libs_utils
from . import ml as libs_ml
# logging.basicConfig(level=logging.INFO, format="[%(levelname)s: %(asctime)s - %(pathname)s:%(lineno)d] \n  %(message)s ")  
logging.basicConfig(level=logging.INFO, format="[%(levelname)s: %(asctime)s - %(filename)s:%(lineno)d] %(message)s ")  