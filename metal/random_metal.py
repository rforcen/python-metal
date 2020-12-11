'''
test for random metal app.
'''
import random
from math import sqrt
from timeit import default_timer as lap

import numpy as np
from numba import njit, prange, float32

from metal import Metal


@njit(cache=True, parallel=True)
def rand_numba(n):
    sqr = lambda x: x * x

    s: float32 = 0
    sq: float32 = 0
    n2 = n * n

    rnd_vect = np.empty((n2), dtype=float32)
    for i in prange(n2):
        x = random.random()
        rnd_vect[i] = x
        s += x
        sq += x * x
    return rnd_vect, s / n2, sqrt((sq - sqr(s) / n2) / (n2 - 1))  # rnds, mean, std


@njit(cache=True, parallel=True)
def stats_numba(data):
    sqr = lambda x: x * x

    s: float32 = 0
    sq: float32 = 0

    n: int = len(data)
    for i in prange(n):
        s += data[i]
        sq += sqr(data[i])

    return s / n, sqrt((sq - sqr(s) / n) / (n - 1))  # mean, std


print('metal random generator')

t0 = lap()
m = Metal('random.metal', 'randomf')
t0 = lap() - t0

sz = 16
n = 1024 * sz

print(f'metal compiled in {t0:.2} run random on size {n * n:,}', end='')

# create i/o buffers to match kernel func. params
brand = m.empty_float(n * n)  # output
bseeds = m.float_buf(np.random.randint(0, 1000000, (3,), dtype='u4'))  # 3 seeds input

m.set_buffers(buffers=(brand, bseeds), threads=(n, n))

t = lap()

m.run()

t = lap() - t

rnd = m.get_buffer(brand, dtype='f4')
print(f' done in {t:.3}"')
print(f'metal: mean, var={stats_numba(rnd)}')

rand_numba(2)

tn = lap()

rnd, mean, std = rand_numba(n)

tn = lap() - tn

print(f'numba: mean, var={mean, std}')

print(f'numba in {tn:.3}", ratio: {tn / t:.3}')
