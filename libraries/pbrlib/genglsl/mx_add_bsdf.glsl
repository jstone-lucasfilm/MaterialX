#include "lib/mx_closure_type.glsl"

void mx_add_bsdf(ClosureData closureData, BSDF in1, BSDF in2, out BSDF result)
{
    // In GLSL, we interpret closure addition as mix(in1, in2, 0.5) * 2, which
    // gives us the following math for the response and throughput.
    result.response = in1.response + in2.response;
    result.throughput = mix(in1.throughput, in2.throughput, 0.5);
}
