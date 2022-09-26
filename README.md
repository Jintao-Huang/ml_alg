# ML-ALG


## Introduction
1. 这个仓库是一个机器学习算法, 传统算法的集成库, 现在主要是自己在使用. 所在文件夹为`libs/`, 下面将会介绍各个文件的用途.



## 文件功能介绍
1. mini_lightning部分, 现已移置: [https://github.com/ustcml/mini-lightning](https://github.com/ustcml/mini-lightning)
2. `libs/ml/*`: 机器学习算法库
3. `libs/ml/ml_alg/*`: 机器学习中的算法实现(没实用性, 可以用于学习)
   1. `_nn_functional.py`: 实现torch.nn.functional包中的算法.
   2. `_metrics.py`: ml的metrics的torch实现
   3. `_optim_functional.py`: ml中的优化器的实现
4. `libs/utils/*`: 一些工具函数的实现
5. `libs/alg/*`: 传统算法库. (开发中)
6. 其他文件没啥用(特别是`_`开头的文件.). 



## 环境:
1. python>=3.8
2. torch>=1.12
3. torchmetrics==0.9.3
4. mini-lightning==0.1.*

