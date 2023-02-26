# distutils: language=c++
from cython cimport (
	void,
    int, long, float, double, 
    integral, floating, numeric,
)
# from cython.parallel cimport prange
from cython cimport boundscheck, wraparound, cdivision
from cython.operator cimport dereference as deref

ctypedef fused int_float:
	int
	long
	float
	double
# 
from libc.string cimport memset, memcpy
from libc.stdlib cimport rand, srand, malloc, calloc, free
from libc.math cimport exp, log, pow as cpow, fabs
from libc.time cimport time
# 
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.deque cimport deque
from libcpp.list cimport list as clist
from libcpp.algorithm cimport sort
