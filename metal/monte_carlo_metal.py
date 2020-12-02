import numpy

from metal import Metal


def test_Metal():
    monte_carlo_metal_source = '''
    #include <metal_stdlib>
    using namespace metal;

    kernel void monte_carlo(
        const device float2 *in_points [[buffer(0)]], // input points(x,y)
        device atomic_uint &counter    [[buffer(1)]], // generated counter
        uint pos [[thread_position_in_grid]]) 
    {
      if (length(in_points[pos]) < 1.0)
         atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
    }
    '''

    print('generating pi using monte carlo metal code')

    size = 1024 * 4
    s2 = size ** 2

    m = Metal(monte_carlo_metal_source)  # , 'monte_carlo')
    # size*size x,y pairs input and one int output set to 0
    m.set_buffers(buffers=(m.buffer(Metal.rand(s2 * 2, numpy.float32)), cnt_buf := m.buffer(0)))
    m.run()

    mpi = 16. * m.get_buffer(cnt_buf, dtype=numpy.int32)[0] / s2

    print(f'pi={mpi} diff={abs(mpi - numpy.pi)}')


test_Metal()
