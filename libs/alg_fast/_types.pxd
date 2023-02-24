from cython cimport (
	void,
    int, long, float, double, 
    integral, floating, numeric,
)
from cython.parallel cimport prange
from cython cimport boundscheck, wraparound, cdivision

ctypedef fused int_float:
	int
	long
	float
	double

from libc.stdlib cimport rand, srand, malloc, free
from libc.time cimport time
