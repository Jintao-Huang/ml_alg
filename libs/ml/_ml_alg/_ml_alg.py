# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import torch
import torch.linalg as tl
from torch import Tensor
from typing import Tuple, Optional, Literal
import torch.nn.functional as F
try:
    from ._metrics import pairwise_euclidean_distance, batched_euclidean_distance
except ImportError:
    from libs.ml._ml_alg._metrics import pairwise_euclidean_distance, batched_euclidean_distance

__all__ = [
    "StandardScaler", "MinMaxScaler",
    "data_center", "data_normalize",
    "LinearRegression", "Ridge",
    "PCA", "NearestNeighbors"
]


class StandardScaler:
    """
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """

    def __init__(self, with_std: bool = True) -> None:
        self.with_std = with_std
        #
        self.mean: Optional[Tensor] = None  # [F]
        self.std: Optional[Tensor] = None  # [F]

    def fit(self, X: Tensor) -> "StandardScaler":
        """
        X: shape[N, F]
        """
        self.mean = X.mean(dim=0)
        if self.with_std:
            self.std = X.std(dim=0, unbiased=False)
        return self  # for 链式

    def transform(self, X: Tensor) -> Tensor:
        """
        X: shape[N, F]
        return: [N, F]
        """
        assert self.mean is not None
        #
        res = X - self.mean
        if self.with_std:
            assert self.std is not None
            res.div_(self.std)
        return res


if __name__ == '__main__':
    import mini_lightning as ml
    device = "cpu"

# if __name__ == "__main__":
#     from sklearn.preprocessing import StandardScaler as _StandardScaler

#     X = torch.randn(1000, 100).to(torch.float64)
#     X_np = X.numpy()
#     X = X.to(device)
#     for with_std in [True, False]:
#         Z = ml.test_time(lambda: StandardScaler(with_std=with_std).fit(X).transform(X)).cpu()
#         Z2 = ml.test_time(lambda: _StandardScaler(with_std=with_std).fit(X_np).transform(X_np))
#         Z2 = torch.from_numpy(Z2)
#         print(torch.allclose(Z, Z2))


def data_center(X: Tensor) -> Tuple[Tensor, Tensor]:
    """数据居中化
    return: res, X_mean
    """
    ss = StandardScaler(with_std=False)
    res = ss.fit(X).transform(X)
    assert ss.mean is not None
    return res, ss.mean


def data_normalize(X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """数据归一化->mean=0, std=1
    return: res, X_mean, X_std
    """
    ss = StandardScaler(with_std=True)
    res = ss.fit(X).transform(X)
    assert ss.mean is not None
    assert ss.std is not None
    return res, ss.mean, ss.std


class MinMaxScaler:
    """
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """

    def __init__(self, feature_range: Tuple[int, int] = (0, 1), clip: bool = False) -> None:
        """
        clip: 在transform时, 保证数值在feature_range内. 
        """
        self.feature_range = feature_range
        self.clip = clip
        #
        self.scale: Optional[Tensor] = None  # [F]
        self.bias: Optional[Tensor] = None  # [F]
        self.X_min: Optional[Tensor] = None  # [F]
        self.X_max: Optional[Tensor] = None  # [F]

    def fit(self, X: Tensor) -> "MinMaxScaler":
        """
        X: shape[N, F]
        # 
        res = X*scale+min
        r[0]=X.min*scale+bias
        r[1]=X.max*scale+bias
        """
        self.X_min, _ = X.min(dim=0)
        self.X_max, _ = X.max(dim=0)
        self.scale = (self.feature_range[1] - self.feature_range[0]) / (self.X_max - self.X_min)
        self.bias = self.feature_range[0] - self.X_min * self.scale
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        X: shape[N, F]
        return: shape[N, F]
        """
        assert self.scale is not None
        assert self.bias is not None
        #
        res = X * self.scale
        res.add_(self.bias)
        if self.clip:
            torch.clip_(res, self.feature_range[0], self.feature_range[1])
        return res


# if __name__ == "__main__":
#     from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
#     X = torch.randn(1000, 100).to(torch.float64)
#     X_np = X.numpy()
#     X = X.to(device)
#     for clip in [True, False]:
#         Z = ml.test_time(lambda: MinMaxScaler(clip=clip).fit(X).transform(X)).cpu()
#         Z2 = ml.test_time(lambda: _MinMaxScaler(clip=clip).fit(X_np).transform(X_np))
#         Z2 = torch.from_numpy(Z2)
#         print(torch.allclose(Z, Z2))
#     #
#     for clip in [True, False]:
#         Z = ml.test_time(lambda: MinMaxScaler(feature_range=(-1, 1), clip=clip).fit(X).transform(X)).cpu()
#         Z2 = ml.test_time(lambda: _MinMaxScaler(feature_range=(-1, 1), clip=clip).fit(X_np).transform(X_np))
#         Z2 = torch.from_numpy(Z2)
#         print(torch.allclose(Z, Z2))


class _MinMaxScaler2(StandardScaler):
    """另一种实现. 与StandardScaler的transform对应"""

    def __init__(self, feature_range: Tuple[int, int] = (0, 1)) -> None:
        """
        clip: 在transform时, 保证数值在feature_range内. 
        """
        super().__init__(True)
        self.feature_range = feature_range

    def fit(self, X: Tensor) -> "_MinMaxScaler2":
        """
        X: shape[N, F]
        # 
        res = (X-mean)/std. {这里的mean/std不是真的mean/std.}
        r[0]=(X.min-mean)/std
        r[1]=(X.max-mean)/std
        """
        X_min, _ = X.min(dim=0)
        X_max, _ = X.max(dim=0)
        self.std = (X_max - X_min) / (self.feature_range[1] - self.feature_range[0])
        self.mean = X_min - self.feature_range[0] * self.std
        return self


# if __name__ == "__main__":
#     from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
#     X = torch.randn(1000, 100).to(torch.float64)
#     X_np = X.numpy()
#     X = X.to(device)
#     Z = ml.test_time(lambda: _MinMaxScaler().fit(X_np).transform(X_np))
#     Z2 = ml.test_time(lambda: _MinMaxScaler2().fit(X).transform(X)).cpu()
#     Z = torch.from_numpy(Z)
#     print(torch.allclose(Z, Z2))
#     #
#     Z = ml.test_time(lambda: _MinMaxScaler(feature_range=(-1, 1)).fit(X_np).transform(X_np))
#     Z2 = ml.test_time(lambda: _MinMaxScaler2(feature_range=(-1, 1)).fit(X).transform(X)).cpu()
#     Z = torch.from_numpy(Z)
#     print(torch.allclose(Z, Z2))

class LinearRegression:
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    def __init__(self) -> None:
        self.weight: Optional[Tensor] = None  # [Fout, Fin]
        self.bias: Optional[Tensor] = None  # [Fout]

    @staticmethod
    def _fit_bias(X_mean: Tensor, y_mean: Tensor, weight: Tensor) -> Tensor:
        """return: bias"""
        return y_mean - X_mean @ weight.T

    def fit(self, X: Tensor, y: Tensor) -> "LinearRegression":
        """
        X: shape[N, Fin]. 不支持[N], 请传入[N, 1]
        y: shape[N, Fout]
        闭式解: (X^{T}X)^{-1}X^{T}y
            Ref: Hands-On Machine Learning v2, 4.1. Aurelien Geron.
        """
        X, X_mean = data_center(X)
        y, y_mean = data_center(y)
        self.weight = tl.lstsq(X, y)[0].T
        self.bias = self._fit_bias(X_mean, y_mean, self.weight)
        return self

    def predict(self, X: Tensor) -> Tensor:
        assert self.weight is not None
        assert self.bias is not None
        #
        res = X @ self.weight.T
        res.add_(self.bias)
        return res

# if __name__ == "__main__":
#     print()
#     from sklearn.linear_model import LinearRegression as _LinearRegression
#     X = torch.randn(1000, 100).to(torch.float64)
#     X2 = torch.randn(1000, 100).to(torch.float64)
#     y = torch.randn(1000, 200).to(torch.float64)
#     X_np = X.numpy()
#     X2_np = X2.numpy()
#     y_np = y.numpy()
#     X = X.to(device)
#     X2 = X2.to(device)
#     y = y.to(device)
#     #
#     lr = LinearRegression()
#     lr2 = _LinearRegression()
#     Z = ml.test_time(lambda: lr.fit(X, y).predict(X2)).cpu()
#     Z2 = ml.test_time(lambda: lr2.fit(X_np, y_np).predict(X2_np))
#     Z2 = torch.from_numpy(Z2)
#     print(torch.allclose(Z, Z2))
#     print(torch.allclose(lr.weight.cpu(), torch.from_numpy(lr2.coef_)))
#     print(torch.allclose(lr.bias.cpu(), torch.from_numpy(lr2.intercept_)))


class Ridge(LinearRegression):
    """
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    """

    def __init__(self, alpha: float = 1., solver: Literal["svd", "cholesky"] = "cholesky") -> None:

        self.alpha = alpha
        self.solver = solver
        #
        super().__init__()

    @staticmethod
    def _fit_svd(X: Tensor, y: Tensor, alpha: float) -> Tensor:
        U, s, Vh = tl.svd(X, full_matrices=False)
        # 在sklearn中, 对接近0的数值进行置0处理. 这里不进行处理.
        d = s.div_((s * s).add_(alpha))  # div
        return tl.multi_dot([Vh.T.mul_(d), U.T, y]).T

    @staticmethod
    def _fit_cholesky(X: Tensor, y: Tensor, alpha: float) -> Tensor:
        """
        闭式解: (X^{T}X + alphaI)^{-1}X^{T}y
            Ref: Hands-On Machine Learning v2, 4.5. Aurelien Geron.
        """
        A = X.T @ X
        b = X.T @ y
        A.ravel()[::A.shape[0]+1].add_(alpha)  # 对角线+alpha
        L = tl.cholesky(A)
        # (X.T@X)^{-1}@X.T@y
        return torch.cholesky_solve(b, L).T

    def fit(self, X: Tensor, y: Tensor) -> "Ridge":
        """
        X: [N, Fin]
        y: [N, Fout]
        """
        X, X_mean = data_center(X)
        y, y_mean = data_center(y)
        if self.solver == "cholesky":
            self.weight = self._fit_cholesky(X, y, self.alpha)
        else:
            self.weight = self._fit_svd(X, y, self.alpha)
        self.bias = self._fit_bias(X_mean, y_mean, self.weight)
        return self


# if __name__ == "__main__":
#     print()
#     from sklearn.linear_model import Ridge as _Ridge
#     X = torch.randn(1000, 100).to(torch.float64)
#     X2 = torch.randn(1000, 100).to(torch.float64)
#     y = torch.randn(1000, 200).to(torch.float64)
#     X_np = X.numpy()
#     X2_np = X2.numpy()
#     y_np = y.numpy()
#     X = X.to(device)
#     X2 = X2.to(device)
#     y = y.to(device)
#     #
#     Z = ml.test_time(lambda: Ridge(solver="svd").fit(X, y).predict(X2), 10).cpu()
#     Z2 = ml.test_time(lambda: _Ridge(solver="svd").fit(X_np, y_np).predict(X2_np), 10)
#     Z2 = torch.from_numpy(Z2)
#     print(torch.allclose(Z, Z2))
#     #
#     Z = ml.test_time(lambda: Ridge(solver="cholesky").fit(X, y).predict(X2), 10).cpu()
#     Z2 = ml.test_time(lambda: _Ridge(solver="cholesky").fit(X_np, y_np).predict(X2_np), 10)
#     Z2 = torch.from_numpy(Z2)
#     print(torch.allclose(Z, Z2))


class PCA:
    """
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    def __init__(self, k: int, svd_solver: Literal["full"] = "full") -> None:
        """
        k: 成分个数
        """
        self.k = k
        self.svd_solver = svd_solver
        #
        self.mean: Optional[Tensor] = None  # [F]
        self.singular_values: Optional[Tensor] = None  # [K]
        self.components: Optional[Tensor] = None  # [K, F]

    def _fit_full(self, X: Tensor) -> None:
        _, s, Vt = tl.svd(X, full_matrices=False)  # [N, F], [F], [F, F]
        self.singular_values = s[:self.k]
        self.components = Vt[:self.k]

    def fit(self, X: Tensor) -> "PCA":
        """
        X: [N, F]
        """
        X, self.mean = data_center(X)
        self._fit_full(X)
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        X: [N, F]
        return: [N, K]
        """
        assert self.components is not None
        assert self.mean is not None
        X = X - self.mean
        return X @ self.components.T

    def inverse_transform(self, X: Tensor) -> Tensor:
        """
        X: [N, K]
        return: [N, F]
        """
        assert self.components is not None
        assert self.mean is not None
        return (X @ self.components).add_(self.mean)


# if __name__ == "__main__":
#     from sklearn.decomposition import PCA as _PCA
#     X = torch.randn(1000, 100).to(torch.float64)
#     X2 = torch.randn(1000, 100).to(torch.float64)
#     X_np = X.numpy()
#     X2_np = X2.numpy()
#     X = X.to(device)
#     X2 = X2.to(device)
#     #
#     p1 = PCA(100)
#     p2 = _PCA(100)
#     Z = ml.test_time(lambda: p1.fit(X).transform(X2), 10).cpu()
#     Z2 = ml.test_time(lambda: p2.fit(X_np).transform(X2_np), 10)
#     print(torch.allclose(torch.abs(Z), torch.abs(torch.from_numpy(Z2))))
#     Z = p1.inverse_transform(Z)
#     Z2 = p2.inverse_transform(Z2)
#     print(torch.allclose(p1.singular_values, torch.from_numpy(p2.singular_values_)))
#     print(torch.allclose(torch.abs(p1.components), torch.abs(torch.from_numpy(p2.components_))))
#     print(torch.allclose(Z, torch.from_numpy(Z2)))
#     print(torch.allclose(Z, X2))
#     print(torch.allclose(torch.from_numpy(Z2), X2))


class _LogisticRegression(LinearRegression):
    """并不会得到加速. for study. 开发中
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    def __init__(
        self,
        penalty: Literal["l2", None] = "l2",
        C: float = 1.,
        tol: float = 0.0001,
        max_iter: int = 100,
        solver: Literal["lbfgs"] = 'lbfgs'  #
    ) -> None:
        self.penalty = penalty
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.solver = solver
        #
        super().__init__()
        self.n_classes: Optional[int] = None

    def fit(self, X: Tensor, y: Tensor) -> "_LogisticRegression":
        """
        X: [N, Fin]
        y: [N]. long. e.g. 0,1,2...
        """
        assert y.dtype is torch.long
        raise NotImplementedError
        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        """
        X: [N, Fin]
        return: [N, C]
        """
        logits = super().predict(X)  # [N, C]
        return logits.softmax(dim=1)

    def predict(self, X: Tensor) -> Tensor:
        """
        X: [N, Fin]
        return: [N]. long
        """
        proba = self.predict_proba(X)  # [N, C]
        return proba.argmax(dim=1)


# if __name__ == "__main__":
#     LogisticRegression = _LogisticRegression
#     from sklearn.linear_model import LogisticRegression as _LogisticRegression
#     X = torch.randn(10000, 100).to(torch.float64)
#     y = torch.randint(0, 10, (10000, ), dtype=torch.long)
#     lgr = _LogisticRegression()
#     lgr2 = LogisticRegression()
#     ml.test_time(lambda: lgr.fit(X, y))
#     lgr2.weight = torch.from_numpy(lgr.coef_)
#     lgr2.bias = torch.from_numpy(lgr.intercept_)
#     X2 = torch.randn(10000, 100).to(torch.float64)
#     Z = lgr.predict(X2)
#     Z2 = lgr2.predict(X2)
#     print(torch.allclose(torch.from_numpy(Z), Z2))
#


class _KMeans:
    """并不会得到加速. for study
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    def __init__(
        self,
        k: int = 8,
        init: Literal["random", "k-means++"] = "random",
        n_init: int = 5,
        max_iter=300,
        tol=0.0001,
        algorithm: Literal["lloyd"] = "lloyd"  # 当前只支持
    ) -> None:
        self.k = k
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        #
        self.centers: Optional[Tensor] = None
        self.labels: Optional[Tensor] = None
        self.inertia: Optional[float] = None
        self.n_iter: Optional[int] = None

    @staticmethod
    def _random_init_centers(X: Tensor, k: int) -> Tensor:
        """
        X: [N, F]
        return: [K, F]
        """
        N = X.shape[0]
        idx = torch.randperm(N)[:k]
        return X[idx]

    @staticmethod
    def _predict_labels(X: Tensor, centers: Tensor) -> Tensor:
        """
        X: [N, F]
        centers: [K, F]
        return: [N]
        """
        dist2 = pairwise_euclidean_distance(X, centers, squared=True)
        labels = dist2.argmin(dim=1)
        return labels

    @staticmethod
    def _update_centers(X: Tensor, labels: Tensor, k: int) -> Tensor:
        """
        X: [N, F]
        labels: [N]
        return: [K, F]
        """
        L = F.one_hot(labels, num_classes=k).to(X.dtype)  # [N, K]
        centers = (L.T @ X).div_(L.sum(dim=0)[:, None])  # [K, F]
        return centers

    @staticmethod
    def _compute_inertia(X: Tensor, centers: Tensor, labels: Optional[Tensor], k: int) -> float:
        """
        X: [N, F]
        centers: [K, F]
        labels: [N]
        """
        L = F.one_hot(labels, num_classes=k).to(X.dtype)  # [N, K]
        dist2 = pairwise_euclidean_distance(X, centers, squared=True)  # [N, K]
        return torch.einsum("ij,ij->", L, dist2).item()

    @staticmethod
    def _compute_centers_shift(centers: Tensor, prev_centers: Tensor) -> float:
        """
        centers: [K, F]
        prev_centers: [K, F]
        """
        dist2 = batched_euclidean_distance(prev_centers, centers, squared=True)
        return dist2.sum().item()

    def _fit_single_lloyd(self, X: Tensor, init_centers: Tensor) -> Tuple[Tensor, Tensor, float, int]:
        """
        X: [N, F]
        init_centers: [K, F]
        return: labels:[N], centers:[K,F], inertia:float, n_iter
        """
        centers = init_centers
        prev_centers = centers
        #
        for i in range(self.max_iter):
            # update labels
            labels = self._predict_labels(X, centers)  # [N]
            # update centers
            centers = self._update_centers(X, labels, self.k)
            if self._compute_centers_shift(centers, prev_centers) < self.tol:
                break
            prev_centers = centers

        inertia = self._compute_inertia(X, centers, labels, self.k)
        return labels, centers, inertia, i + 1

    def fit(self, X: Tensor) -> "_KMeans":
        """
        X: [N, F]
        """
        X, X_mean = data_center(X)
        best_res = None  # centers, labels, inertia
        for _ in range(self.n_init):
            if self.init == "random":
                init_centers = self._random_init_centers(X, self.k)
            elif self.init == "k-means++":
                raise NotImplementedError
            else:
                raise ValueError(f"self.init: {self.init}")
            #
            res = self._fit_single_lloyd(X, init_centers)
            if best_res is None or best_res[2] < res[2]:
                best_res = res
        assert best_res is not None
        self.labels, self.centers, self.inertia, self.n_iter = best_res
        self.centers.add_(X_mean)

        return self

    def predict(self, X: Tensor) -> Tensor:
        """
        X: [N, F]
        return: [N]. long
        """
        assert self.centers is not None
        return self._predict_labels(X, self.centers)

    def transform(self, X: Tensor) -> Tensor:
        """
        X: [N, F]
        return: [N, K]
        """
        assert self.centers is not None
        return pairwise_euclidean_distance(X, self.centers)


# if __name__ == "__main__":
#     KMeans = _KMeans
#     from sklearn.cluster import KMeans as _KMeans
#     X = torch.randn(10000, 100).to(torch.float64)
#     km = _KMeans(init="random")
#     km2 = KMeans(init="random", n_init=10)
#     ml.test_time(lambda: km.fit(X))
#     km2.centers = torch.from_numpy(km.cluster_centers_)
#     km2.labels = torch.from_numpy(km.labels_)
#     km2.inertia = km.inertia_
#     X2 = torch.randn(10000, 100).to(torch.float64)
#     Z = ml.test_time(lambda: km.predict(X2), 5)
#     Z2 = ml.test_time(lambda: km2.predict(X2), 5)
#     print(torch.allclose(torch.from_numpy(Z).long(), Z2))
#     Z = ml.test_time(lambda: km.transform(X2), 5)
#     Z2 = ml.test_time(lambda: km2.transform(X2), 5)
#     print(torch.allclose(torch.from_numpy(Z), Z2))
#     #
#     km2 = KMeans(init="random")
#     ml.test_time(lambda: km2.fit(X))
#     print(km2.inertia, km.inertia_)
#     print(km2.n_iter, km.n_iter_)


class NearestNeighbors:
    """
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """

    def __init__(self, alg: Literal["naive"] = "naive") -> None:
        """metric使用欧式距离
            naive的方法在cuda上往往比cpu的kd_tree, ball_tree快很多. 测试见下.
        """
        self.alg = alg

    def fit(self, X: Tensor) -> "NearestNeighbors":
        """
        X: [N, F]
        """
        if self.alg == "naive":
            self.X = X
        return self

    def kneighbors(self, Q: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Q: [Q, F]
        return: Tuple[dist, idx]
            dist: [Q, k]
            idx: [Q, k]
        """
        dist = pairwise_euclidean_distance(Q, self.X)  # [Q, N]
        dist, idx = torch.topk(dist, k, dim=1, largest=False)  # [Q, K]
        return dist, idx


# if __name__ == "__main__":
#     from sklearn.neighbors import NearestNeighbors as _NearestNeighbors

#     X = torch.randn(5000, 500).to(torch.float64)
#     Q = torch.randn(100, 500).to(torch.float64)
#     z = ml.test_time(lambda: _NearestNeighbors(algorithm="kd_tree").fit(X).kneighbors(Q, 100), warm_up=1)
#     z2 = ml.test_time(lambda: NearestNeighbors(alg="naive").fit(X).kneighbors(Q, 100), warm_up=1)
#     print(torch.allclose(torch.from_numpy(z[0]), z2[0]))
#     print(torch.allclose(torch.from_numpy(z[1]), z2[1]))
#     X = X.cuda()
#     Q = Q.cuda()
#     z2 = ml.test_time(lambda: NearestNeighbors(alg="naive").fit(X).kneighbors(Q, 100), warm_up=1, timer=ml.time_synchronize)
#     print(torch.allclose(torch.from_numpy(z[0]), z2[0].cpu()))
#     print(torch.allclose(torch.from_numpy(z[1]), z2[1].cpu()))
