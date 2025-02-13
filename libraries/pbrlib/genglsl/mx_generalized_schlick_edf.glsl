#include "lib/mx_closure_type.glsl"
#include "lib/mx_microfacet.glsl"

void mx_generalized_schlick_edf(ClosureData closureData, vec3 color0, vec3 color90, float exponent, EDF base, out EDF result)
{
    if (closureData.closureType == CLOSURE_TYPE_EMISSION)
    {
        float NdotV = clamp(dot(closureData.N, closureData.V), M_FLOAT_EPS, 1.0);
        vec3 F = mx_fresnel_schlick(NdotV, color0, color90, exponent);
        result = base * F;
    }
}
