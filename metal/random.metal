// xcrun -sdk macosx metal -c random.metal
// from https://github.com/YoussefV/Loki/tree/master

#include <metal_stdlib>
using namespace metal;

class Loki {
private:
    float seed;
    unsigned useed;

    unsigned TausStep(const unsigned z, const int s1, const int s2, const int s3, const unsigned M)
    {
        unsigned b=(((z << s1) ^ z) >> s2);
        return (((z & M) << s3) ^ b);
    }

public:
    Loki(const unsigned seed1, const unsigned seed2, const unsigned seed3) {
        unsigned _seed = seed1 * 1099087573UL;
        unsigned seedb = seed2 * 1099087573UL;
        unsigned seedc = seed3 * 1099087573UL;

        // Round 1: Randomise seed
        unsigned z1 = TausStep(_seed,13,19,12,429496729UL);
        unsigned z2 = TausStep(_seed,2,25,4,4294967288UL);
        unsigned z3 = TausStep(_seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*_seed + 1013904223UL);

        // Round 2: Randomise seed again using second seed
        unsigned r1 = (z1^z2^z3^z4^seedb);

        z1 = TausStep(r1,13,19,12,429496729UL);
        z2 = TausStep(r1,2,25,4,4294967288UL);
        z3 = TausStep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);

        // Round 3: Randomise seed again using third seed
        r1 = (z1^z2^z3^z4^seedc);

        z1 = TausStep(r1,13,19,12,429496729UL);
        z2 = TausStep(r1,2,25,4,4294967288UL);
        z3 = TausStep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);

        seed = (z1^z2^z3^z4) * 2.3283064365387e-10;
    }

    float randf() {
        unsigned hashed_seed = seed * 1099087573UL;

        unsigned z1 = TausStep(hashed_seed,13,19,12,429496729UL);
        unsigned z2 = TausStep(hashed_seed,2,25,4,4294967288UL);
        unsigned z3 = TausStep(hashed_seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*hashed_seed + 1013904223UL);

        float old_seed = seed;

        seed = (z1^z2^z3^z4) * 2.3283064365387e-10;

        return old_seed;
    }

    unsigned randu() {
        unsigned hashed_seed = useed * 1099087573UL;

        unsigned z1 = TausStep(hashed_seed,13,19,12,429496729UL);
        unsigned z2 = TausStep(hashed_seed,2,25,4,4294967288UL);
        unsigned z3 = TausStep(hashed_seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*hashed_seed + 1013904223UL);

        unsigned old_seed = useed;

        useed = (z1^z2^z3^z4) / 2328306436;

        return old_seed;
    }

};


kernel void randomf(device float*rnds[[buffer(0)]],
                    device unsigned*seeds[[buffer(1)]],
                    uint2 position [[thread_position_in_grid]],
                    uint2 tpg[[threads_per_grid]])
{
    uint x=position.x, y=position.y, width=tpg.x;

    rnds[x + y * width] = Loki(x+seeds[0], y+seeds[1], width+seeds[2]).randf();
}

kernel void randomu(device unsigned*rnds[[buffer(0)]],
                    device unsigned*seeds[[buffer(1)]],
                    uint2 position [[thread_position_in_grid]],
                    uint2 tpg[[threads_per_grid]])
{
    uint x=position.x, y=position.y, width=tpg.x;

    rnds[x + y * width] = Loki(x+seeds[0], y+seeds[1], width+seeds[2]).randu();
}
