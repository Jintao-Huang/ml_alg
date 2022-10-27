import torch.linalg as tl
from torch import Tensor
import torch
from typing import Literal, Tuple


__all__ = []

if __name__ == "__main__":
    import mini_lightning as ml


# if __name__ == "__main__":
#     """test eig, eigh, inv, pinv"""
#     A = torch.randn(1000, 1000)
#     L, V = tl.eig(A)
#     print(torch.allclose((V @ torch.diag(L) @ tl.inv(V)).to(torch.float32), A, atol=1e-2))
#     #
#     L, Q = tl.eigh(A, "U")
#     print(torch.allclose(torch.triu(Q @ torch.diag(L) @ Q.T), torch.triu(A), atol=1e-4))
#     print(torch.allclose(torch.triu(Q * L[None] @ Q.T), torch.triu(A), atol=1e-4))


def pinv(
    A: Tensor, hermitian: bool = False,
    driver: Literal[None, "eigh", "svd", "qr"] = None
) -> Tensor:
    """driver默认方法. 
        Ref: https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    A: [N,M]
    return: [M,N]
    """
    if hermitian:
        driver = "eigh" if driver is None else driver
        if driver == "eigh":
            # A=Q diag(L)Qh, A^-1=Q diag(1/L)Qh
            L, Q = tl.eigh(A)
            return Q.div(L[None]) @ Q.T
        else:
            raise ValueError(f"driver: {driver}")
    else:
        # torch使用svd方法.
        driver = "svd" if driver is None else driver
        if driver == "svd":
            # A=U diag(s)Vh, A^-1=V diag(1/s)Uh
            U, s, Vh = tl.svd(A, full_matrices=False)
            return Vh.T.div_(s[None]) @ U.T
        elif driver == "qr":
            # 相比较svd方法: qr更快, 但数值不稳定(误差较大)
            Q, R = tl.qr(A)
            return tl.inv(R) @ Q.T
        else:
            raise ValueError(f"driver: {driver}")


# if __name__ == "__main__":
#     A = torch.randn(100, 10)
#     print(tl.pinv(A).shape)
#     A = torch.randn(1000, 1000)
#     for hermitian in [False, True]:
#         if hermitian:
#             A = torch.triu(A)
#         z = ml.test_time(lambda: tl.pinv(A, hermitian=hermitian), 10)
#         z2 = ml.test_time(lambda: pinv(A, hermitian=hermitian), 10)
#         if not hermitian:
#             z3 = ml.test_time(lambda: pinv2(A), 10)
#             print(torch.allclose(z2, z3, atol=1e-4))  # 数值不稳定.
#         print(torch.allclose(z, z2, atol=1e-6))


def dist(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
    return torch.norm(x - y, p=p)


# if __name__ == "__main__":
#     x = torch.randn(1000, 100)
#     y = torch.randn(1000, 100)
#     for p in [1, 2, 3]:
#         z = ml.test_time(lambda: torch.dist(x, y, p=p), 10)
#         z2 = ml.test_time(lambda: dist(x, y, p=p), 10)
#         print(torch.allclose(z, z2))


def solve(A: Tensor, B: Tensor, *, driver: Literal["lu", "cholesky", "naive"]) -> Tensor:
    """AX=B"""
    if driver == "naive":
        return tl.inv(A) @ B
    elif driver == "lu":  # torch实现
        # Ref: https://pytorch.org/docs/stable/generated/torch.linalg.lu.html
        LU, P = tl.lu_factor(A)
        return torch.lu_solve(B, LU, P)
    elif driver == "cholesky":
        # 需保证A为正定的.
        L = tl.cholesky(A)
        return torch.cholesky_solve(B, L)


# if __name__ == "__main__":
#     A = torch.randn(2000, 1000)
#     A = A.T @ A
#     B = torch.randn(1000, 1000)
#     X = ml.test_time(lambda: solve(A, B, driver="lu"), 10)
#     X2 = ml.test_time(lambda: solve(A, B, driver="cholesky"), 10)
#     X3 = ml.test_time(lambda: solve(A, B, driver="naive"), 10)
#     X4 = ml.test_time(lambda: tl.solve(A, B), 10)
#     print(torch.allclose(X, X4))
#     print(torch.allclose(X2, X4))
#     print(torch.allclose(X3, X4))


def lstsq(X: Tensor, y: Tensor, driver: Literal["qr", "svd"] = "qr") -> Tensor:
    """这里只返回solution. 和tl.lstsq实现不同. 
    note: qr即gels; svd即gelss
    Ref: https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
        driver: 一般情况: qr快于svd. tl.lstsq默认driver使用qr方法
    X: [N,Fin]
    y: [N, Fout]
    """
    if driver == "qr":  # 速度依旧会慢于gels
        # pinv2(X) @ y
        Q, R = tl.qr(X)
        return tl.multi_dot([tl.inv(R), Q.T, y])
    elif driver == "svd":  # 速度与gelss相近会更快.
        # pinv(X) @ y
        U, s, Vh = tl.svd(X, full_matrices=False)
        return tl.multi_dot([Vh.T.div_(s[None]), U.T, y])
    else:
        raise ValueError(f"driver: {driver}")


# if __name__ == "__main__":
#     X = torch.randn(10000, 1000)
#     y = torch.randn(10000, 500)
#     z1 = ml.test_time(lambda: lstsq(X, y, driver="qr"), 10)
#     z2 = ml.test_time(lambda: tl.lstsq(X, y, driver="gels"), 10)[0]
#     z3 = ml.test_time(lambda: tl.lstsq(X, y), 10)[0]
#     print(torch.allclose(z1, z2, atol=1e-6))
#     print(torch.allclose(z1, z3, atol=1e-6))
#     z1 = ml.test_time(lambda: lstsq(X, y, driver="svd"), 10)
#     z2 = ml.test_time(lambda: tl.lstsq(X, y, driver="gelss"), 10)[0]
#     print(torch.allclose(z1, z2, atol=1e-6))


if __name__ == "__main__":
    x = torch.randn(1000, 1000)
    x = x @ x.T
    # c >(快) lf > ci > lu > qr > svd > eigh > eig
    ml.test_time(lambda: tl.qr(x), 10)
    ml.test_time(lambda: tl.svd(x), 10)
    ml.test_time(lambda: tl.cholesky(x), 10)
    ml.test_time(lambda: tl.lu(x), 10)
    ml.test_time(lambda: tl.lu_factor(x), 10)
    ml.test_time(lambda: tl.eig(x), 10)
    ml.test_time(lambda: tl.eigh(x), 10)
    ml.test_time(lambda: tl.eigvals(x), 10)
    ml.test_time(lambda: tl.eigvalsh(x), 10)

    # st > cs >(快) s > l
    print()
    b = torch.randn(1000, 100)
    ml.test_time(lambda: torch.cholesky_solve(b, x), 10)
    ml.test_time(lambda: tl.solve(x, b), 10)
    ml.test_time(lambda: tl.lstsq(x, b), 10)
    ml.test_time(lambda: tl.solve_triangular(x, b, upper=False), 10)
    ml.test_time(lambda: tl.pinv(x), 10)
    ml.test_time(lambda: tl.pinv(x, hermitian=True), 10)
    ml.test_time(lambda: tl.inv(x), 10)
    ml.test_time(lambda: torch.cholesky_inverse(x), 10)


def cholesky_solve(b: Tensor, L: Tensor) -> Tensor:
    """upper=False. LL^TX=b: Ly=b; L^TX=y"""
    y = tl.solve_triangular(L, b, upper=False)
    return tl.solve_triangular(L.T, y, upper=True)

# if __name__ == "__main__":
#     X = torch.randn(10000, 2000)
#     X = X.T @ X
#     y = torch.randn(2000, 500)
#     L = tl.cholesky(X)
#     z1 = ml.test_time(lambda: torch.cholesky_solve(y, L), 10)
#     z2 = ml.test_time(lambda: cholesky_solve(y, L), 10)
#     print(torch.allclose(z1, z2, atol=1e-2))


def lu_solve(b: Tensor, LU_data: Tensor) -> Tensor:
    """LUX=b: Ly=b; UX=y; 含LU_pivots: 我也不知道怎么实现"""
    y = tl.solve_triangular(LU_data, b, upper=False, unitriangular=True)
    return tl.solve_triangular(LU_data, y, upper=True)


# if __name__ == "__main__":
#     X = torch.randn(2000, 2000).cuda()
#     y = torch.randn(2000, 1000).cuda()
#     LU_data, LU_pivots = tl.lu_factor(X, pivot=False)
#     z1 = ml.test_time(lambda: torch.lu_solve(y, LU_data, LU_pivots), 10)
#     z2 = ml.test_time(lambda: lu_solve(y, LU_data), 10)
#     print(torch.allclose(z1, z2, atol=1e-6))
