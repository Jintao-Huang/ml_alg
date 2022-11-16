# ML-ALG


## Introduction
1. 这个仓库是一个机器学习算法, 传统算法的集成库, 现在主要是自己在使用. 所在文件夹为`libs/`, 下面将会介绍各个文件的用途.



## 文件功能介绍
1. mini_lightning部分, 现已移置: [https://github.com/ustcml/mini-lightning](https://github.com/ustcml/mini-lightning)
2. `libs/ml/_ml_alg/*`: 机器学习中的算法实现
   1. `_metrics.py`: ml中的metrics的torch实现. (faster than `torchmetrics.functional`, `sklearn`, 使用torch实现, 支持cuda加速)
      1. 含accuracy, confusion_matrix, precision, recall, f1_score, fbeta_score, PR_curve, AP, roc_curve, AUC, r2_score, cosine_similarity, euclidean_distance, kl_divergence, pearson_corrcoef, spearman_corrcoef.
   2. `_nn_functional.py`: 实现torch.nn.functional包中的算法. (没啥实用性, 用于学习)
      1. 含激活函数, 损失, batch_norm, layer_norm, dropout, linear, conv2d, conv_transpose2d, conv1d, avg_pool2d, max_pool2d, lstm, gru, multi-head attention, interpolation, pad, adaptive_avg_pool2d, adaptive_max_pool2d.
   3. `_ml_alg.py`: 传统ml算法的torch实现 (faster than `sklearn`, 支持cuda加速). (开发中...)
      1. 含归一化方法, LinearRegression, Ridge, LogisticRegression, PCA, KMeans, NearestNeighbors等
   4. `_optim_functional.py`: 优化器的实现. (没啥实用性, 用于学习)
      1. 含sgd, adam, adamw.
   5. `tvt_functional.py`: torchvision.transforms.functional的实现. (没啥实用性, 用于学习)
      1. 含to_tensor, normalize, pad, hflip, vflip, rgb_to_grayscale, crop, center_crop, resize, resized_crop, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, rotate, affine
   6. `_linalg.py`: 线性代数算法. (没啥实用性, 用于学习)
      1. 含pinv, solve, lstsq, cholesky_solve, lu_solve等
   7. `_functional.py`: 一些torch的函数实现. (没啥实用性, 用于学习)
      1. 含logsumexp, softmax, var, cov, corrcoef, bincount, unique_consecutive
3. `libs/alg/*`: 传统算法库
   1. 数据结构: 树状数组, 堆, 优先级队列, 可变的优先级队列, 红黑树, 有序数组, 链表, 线段树, Lazy线段树, 字符串哈希, 字典树, 并查集, Huffman树等.
   2. 算法: 图算法(dijkstra, kruskal, prim, dinic, 匈牙利算法, 拓扑排序), 背包问题, math, 单调栈/队列, 大数运算, 字符串匹配KMP算法, 二分搜索, a_star, 常见动态规划, 常见的其他算法等.
4. `libs/ml/*`: 机器学习算法库
5. `libs/utils/*`: 一些工具函数的实现
6. `tests/*`: 测试算法正确性. (其中部分测试写在源文件中)


## Install
```bash
# 下载仓库到本地, 进入setup.py所在文件夹. 输入以下命令即可(会自动安装依赖, pytorch请手动安装, 避免cuda版本不匹配)
pip install .
```
