'''
test for voronoi metal app.
'''
import random
from timeit import default_timer as lap

import numpy as np
from PIL import Image
from numba import njit, prange, uint32

from metal import Metal

t0 = lap()
m = Metal('voronoi.metal', 'Voronoi')
t0 = lap() - t0

sz = 4
w, h = 640 * sz, 480 * sz
n_points = int(w / 2)

print(
    f'compiled in {t0:.2} run voronoi on size {w, h}={w * h} pix, {n_points} points, that\'s {w * h * n_points} iters',
    end='')


@njit(cache=True, parallel=True)
def randpc():  # random point(x,y), color, n_points
    v = np.empty((n_points, 4), dtype=uint32)
    for i in prange(n_points):
        v[i, 0] = random.randint(0, w)  # x(0..w) ,y(0..h)
        v[i, 1] = random.randint(0, h)
        v[i, 2] = random.randint(0, 0x00ffffff)  # color (0..0xffffff)
        v[i, 3] = n_points # use this wasted alignment space for n_points...
    return v


# create i/o buffers to match kernel func. params
# input: points(x,y,color, n_points)
bpointcolor = m.buffer(randpc())
bpix = m.empty_int(w * h)  # output

m.set_buffers(buffers=(bpix, bpointcolor), threads=(w, h))

t = lap()

m.run()

pix = m.get_buffer(bpix, dtype=np.int32)
print(f' done in {lap() - t:.3}", processing image...')

Image.frombytes(mode='RGBA', size=(w, h), data=pix).show()  # .save('name.png', format='png')
