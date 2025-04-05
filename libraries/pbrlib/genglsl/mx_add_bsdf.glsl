#include "lib/mx_closure_type.glsl"

void mx_add_bsdf(ClosureData closureData, BSDF in1, BSDF in2, out BSDF result)
{
    // Hardware shading languages require both response and throughput values
    // so we interpret closure addition as mix(in1, in2, 0.5) * 2, giving us
    // the following logic:
    result.response = in1.response + in2.response;
    result.throughput = mix(in1.throughput, in2.throughput, 0.5);
}
