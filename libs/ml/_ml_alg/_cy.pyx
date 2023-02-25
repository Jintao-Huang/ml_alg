# distutils: language=c++
from ...alg_fast._types cimport * 

cdef void _assign_rank(floating[::1] rank, integral[::1] idx, int lo, int hi):
    cdef int i
    cdef floating x = (lo+hi+1) / 2.
    for i in range(lo, hi):
        rank[idx[i]]=x

def calc_rank_loop(floating[::1] x, floating[::1] rank, integral[::1] idx) -> None:
    """O(N)
    x: sorted_x. [N]
    rank: [N]. rank[idx]=arange(1..N+1)
    """
    cdef int i, N = x.shape[0]
    cdef int lo = 0, hi = 0
    for i in range(N):
        hi += 1
        if i == N-1 or x[i] != x[i +1]:
            if hi - lo >= 2:
                _assign_rank(rank, idx, lo, hi)
            lo = hi