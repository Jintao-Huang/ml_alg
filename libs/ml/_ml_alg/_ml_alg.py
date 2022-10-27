import torch
from torch import Tensor
from typing import Tuple, Optional, Literal

__all__ = [
    "StandardScaler", "MinMaxScaler",
    "data_center", "data_normalize",
    "LinearRegression", "Ridge",
    "PCA"
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
        assert not self.with_std or self.std is not None
        #
        res = X - self.mean
        if self.with_std:
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
        0=X.min*scale+bias
        1=X.max*scale+bias
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


class LinearModel:
    def __init__(self) -> None:
        self.weight: Optional[Tensor] = None  # [Fin, Fout]. 与sklearn不同. sklearn: [Fout, Fin]
        self.bias: Optional[Tensor] = None  # [Fout]

    @staticmethod
    def _fit_bias(X_mean: Tensor, y_mean: Tensor, weight: Tensor) -> Tensor:
        """return: bias"""
        return y_mean - X_mean @ weight

    def predict(self, X: Tensor) -> Tensor:
        assert self.weight is not None
        assert self.bias is not None
        #
        res = X @ self.weight
        res.add_(self.bias)
        return res


class LinearRegression(LinearModel):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    def fit(self, X: Tensor, y: Tensor) -> "LinearRegression":
        """
        X: shape[N, Fin]. 不支持[N], 请传入[N, 1]
        y: shape[N, Fout]
        """
        X, X_mean = data_center(X)
        y, y_mean = data_center(y)
        self.weight, _, _, _ = torch.linalg.lstsq(X, y)
        self.bias = self._fit_bias(X_mean, y_mean, self.weight)
        return self


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
#     print(torch.allclose(lr.weight.cpu(), torch.from_numpy(lr2.coef_.T)))
#     print(torch.allclose(lr.bias.cpu(), torch.from_numpy(lr2.intercept_)))
#


class Ridge(LinearModel):
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
        U, s, Vh = torch.linalg.svd(X, full_matrices=False)
        # 在sklearn中, 对接近0的数值进行置0处理. 这里不进行处理.
        d = s.div_((s * s).add_(alpha))  # div
        return torch.linalg.multi_dot([Vh.T.mul_(d), U.T, y])

    @staticmethod
    def _fit_cholesky(X: Tensor, y: Tensor, alpha: float) -> Tensor:
        A = X.T @ X
        b = X.T @ y
        A.ravel()[::A.shape[0]+1].add_(alpha)  # 对角线+alpha
        L = torch.linalg.cholesky(A)
        # (X.T@X)^{-1}@X.T@y
        return torch.cholesky_solve(b, L)

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
        self.k = k
        self.svd_solver = svd_solver
        #
        self.mean: Optional[Tensor] = None  # [F]
        self.singular_values: Optional[Tensor] = None  # [K]
        self.components: Optional[Tensor] = None  # [K, F]

    def _fit_full(self, X: Tensor) -> None:
        U, s, Vt = torch.linalg.svd(X, full_matrices=False)  # [N, F], [F], [F, F]
        self.singular_values = s[:self.k]
        self.components = Vt[:self.k]  # XX^T

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


class LogisticRegression:
    """
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """


class KMeans:
    """
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

# if __name__ == "__main__":
#     from sklearn.linear_model import LogisticRegression as _LogisticRegression
#     from sklearn.cluster import KMeans as _KMeans
