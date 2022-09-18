# ML-ALG


## Introduction
1. 这个仓库是一个机器学习的集成库, 现在主要是自己在使用. 所在文件夹为`libs/`, 下面将会介绍各个文件的用途.
2. mini-lightning发布版: [https://github.com/ustcml/mini-lightning](https://github.com/ustcml/mini-lightning)



## 文件功能介绍
1. mini_lightning部分, 现已移置: [https://github.com/ustcml/mini-lightning](https://github.com/ustcml/mini-lightning)
2. `libs/ml/utils.py`: ml中的工具函数
3. `libs/ml/visualize.py`: 可视化函数的集成
4. `libs/ml/cv/*`: CV中常用的工具函数
5. `libs/ml/nlp/*`: NLP中常用的工具函数
6. `libs/ml/ml_alg/*`: 机器学习中的算法实现(没啥实用性, 学习用)
   1. `_nn_functional.py`: 实现torch.nn.functional包中的算法.
   2. `_metrics.py`: ml的metrics的torch实现
7. `libs/utils/*`: 一些工具函数的实现
8. `libs/alg/*`: 传统算法库
9. 其他文件没啥用(特别是`_`开头的文件.). 



## 环境:
1. python>=3.8
2. torch>=1.12
3. torchmetrics==0.9.3
4. mini-lightning==0.1.*

