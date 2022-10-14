# ML-ALG


## Introduction
1. 这个仓库是一个机器学习算法, 传统算法的集成库, 现在主要是自己在使用. 所在文件夹为`libs/`, 下面将会介绍各个文件的用途.



## 文件功能介绍
1. mini_lightning部分, 现已移置: [https://github.com/ustcml/mini-lightning](https://github.com/ustcml/mini-lightning)
2. `libs/ml/*`: 机器学习算法库
3. `libs/ml/_ml_alg/*`: 机器学习中的算法实现(没实用性, 可以用于学习)
   1. `_nn_functional.py`: 实现torch.nn.functional包中的算法.
   2. `_metrics.py`: ml的metrics的torch实现
   3. `_optim_functional.py`: ml中的优化器的实现
4. `libs/utils/*`: 一些工具函数的实现
5. `libs/alg/*`: 传统算法库. (开发中)
   1. 数据结构: 树状数组, 堆, 优先级队列, 可变的优先级队列, 红黑树, 链表, 线段树, Lazy线段树, 字符串哈希, 字典树, 并查集等.
   2. 算法: 图算法(dijkstra, kruskal, prim, dinic, 匈牙利算法, 拓扑排序), 背包问题, math, 单调栈/队列, 大数运算, 字符串匹配KMP算法等.



## 环境:
1. python>=3.8
2. torch>=1.12
3. torchmetrics==0.9.3
4. mini-lightning>=0.1.4

