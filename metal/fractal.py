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

bgeo = m.buffer(np.array([w, h], dtype=np.int32))
bpix = m.empty(w * h * 4)

m.set_buffers(buffers=(bpix, bgeo), threads=(w, h))

t = lap()

m.run()

pix = m.get_buffer(bpix, dtype=np.int32)
print(f' done in {lap() - t:.3}", processing image...')

Image.frombytes(mode='RGBA', size=(w, h), data=pix).show()  # .save('name.png', format='png')
