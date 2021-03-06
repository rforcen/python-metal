'''
test for fractal metal app. with several funcs directly replace in source code
'''
from timeit import default_timer as lap

import numpy as np
from PIL import Image

from metal import Metal

t0 = lap()
m = Metal('fractal.metal', 'fractal')
t0 = lap() - t0

sz = 4
w, h = 640 * sz, 480 * sz

print(f'compiled in {t0:.2} run fractal on size {w, h}={w * h} pix', end='')

# create i/o buffers to match kernel func. params
bsize = m.int_buf([w, h]) # input: size(w,h), center(x,y), range(x,y)
bcenter = m.float_buf([0.5, 0])
brange = m.float_buf([-2, 2])

bpix = m.empty_int(w * h) # output

m.set_buffers(buffers=(bpix, bsize, bcenter, brange), threads=(w, h))

t = lap()

m.run()

pix = m.get_buffer(bpix, dtype=np.int32)
print(f' done in {lap() - t:.3}", processing image...')

Image.frombytes(mode='RGBA', size=(w, h), data=pix).show()  # .save('name.png', format='png')
