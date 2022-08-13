# ML-ALG




## Introduction
1. 这个仓库是一个机器学习的集成库, 现在主要是自己在使用. 所在文件夹为`libs/`, 下面将会介绍各个文件的用途.
2. 若在使用中出现问题, 请提issue. 




## 文件功能介绍
1. `libs/utils/*`: 一些工具函数的实现
2. `libs/ml/cv/*`: CV中常用的工具函数
3. `libs/ml/nlp/*`: NLP中常用的工具函数
4. `libs/ml/ml_alg/*`: 机器学习中的算法实现(没啥实用性, 学习用)
   1. 其中`_nn_functional.py`实现torch.nn.functional包中的算法. libtorch(c++)版本见: https://github.com/Jintao-Huang/alg_ac/tree/main/ml_libs
5. `libs/ml/lrs.py`: lr_scheduler, warmup的实现
6. `libs/ml/metrics.py`: ml的metrics的torch实现
7. `libs/ml/mini_lightning.py`: mini-lightning
   1. pytorch-lightning的mini版本(更快, 更简洁, 更灵活). 
   2. 支持amp, dp, 梯度累加, warmup, lr_scheduler, 梯度裁剪, tensorboard, 模型和超参数和结果保存等.
   3. examples见 `examples/*`文件夹
8. `libs/ml/visualize.py`: 可视化函数的集成(含tensorboard smoothing算法)
9. `libs/ml/utils.py`: ml中的工具函数
10. `examples/*`: mini-lightning使用的例子. 包括cv, nlp, dqn. 
11. 其他文件没啥用(特别是`_`开头的文件.). 





## 环境:
1. python3.9.12
2. torch==1.12.1

