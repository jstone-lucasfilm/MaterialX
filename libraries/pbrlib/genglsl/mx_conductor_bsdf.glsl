#include "lib/mx_closure_type.glsl"
#include "lib/mx_microfacet_specular.glsl"

void mx_conductor_bsdf(ClosureData closureData, float weight, vec3 ior_n, vec3 ior_k, vec2 roughness, bool retroreflective, float thinfilm_thickness, float thinfilm_ior, vec3 N, vec3 X, int distribution, inout BSDF bsdf)
{
    bsdf.throughput = vec3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    N = mx_forward_facing_normal(N, closureData.N, closureData.V);

    // Build the tangent frame and express the view direction in tangent space.
    TangentFrame frame = mx_tangent_frame(N, X);
    vec3 Vt = mx_world_to_tangent(frame, closureData.V);
    if (retroreflective)
        Vt.xy = -Vt.xy;
    float NdotV = clamp(Vt.z, M_FLOAT_EPS, 1.0);

    FresnelData fd = mx_init_fresnel_conductor(ior_n, ior_k, thinfilm_thickness, thinfilm_ior);

    vec2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        vec3 Lt = mx_world_to_tangent(frame, closureData.L);
        vec3 Ht = normalize(Vt + Lt);

        float NdotL = clamp(Lt.z, M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(Vt, Ht), M_FLOAT_EPS, 1.0);

        vec3 F = mx_compute_fresnel(VdotH, fd);
        float D = mx_ggx_NDF(Ht, safeAlpha);
        float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

        vec3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);

        // Note: NdotL is cancelled out
        bsdf.response = D * F * G * comp * closureData.occlusion * weight / (4.0 * NdotV);
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        vec3 F = mx_compute_fresnel(NdotV, fd);
        vec3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        vec3 Li = mx_environment_radiance(frame, Vt, safeAlpha, distribution, fd);
        bsdf.response = Li * comp * weight;
    }
}
