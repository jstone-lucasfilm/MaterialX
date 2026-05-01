#include "lib/mx_closure_type.glsl"
#include "lib/mx_microfacet_specular.glsl"

void mx_dielectric_bsdf(ClosureData closureData, float weight, vec3 tint, float ior, vec2 roughness, bool retroreflective, float thinfilm_thickness, float thinfilm_ior, vec3 N, vec3 X, int distribution, int scatter_mode, inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
    {
        return;
    }
    if (closureData.closureType != CLOSURE_TYPE_TRANSMISSION && scatter_mode == 1)
    {
        return;
    }

    N = mx_forward_facing_normal(N, closureData.N, closureData.V);

    // Build the tangent frame and express the view direction in tangent space.
    TangentFrame frame = mx_tangent_frame(N, X);
    vec3 Vt = mx_world_to_tangent(frame, closureData.V);
    // Retroreflective mode is only supported for reflection and indirect.
    if (retroreflective && (closureData.closureType != CLOSURE_TYPE_TRANSMISSION))
        Vt.xy = -Vt.xy;
    float NdotV = clamp(Vt.z, M_FLOAT_EPS, 1.0);

    FresnelData fd = mx_init_fresnel_dielectric(ior, thinfilm_thickness, thinfilm_ior);
    float F0 = mx_ior_to_f0(ior);

    vec2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);
    vec3 safeTint = max(tint, 0.0);

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
        vec3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
        bsdf.throughput = 1.0 - dirAlbedo * weight;

        bsdf.response = D * F * G * comp * safeTint * closureData.occlusion * weight / (4.0 * NdotV);
    }
    else if (closureData.closureType == CLOSURE_TYPE_TRANSMISSION)
    {
        vec3 F = mx_compute_fresnel(NdotV, fd);

        vec3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        vec3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
        bsdf.throughput = 1.0 - dirAlbedo * weight;

        if (scatter_mode != 0)
        {
            bsdf.response = mx_surface_transmission(frame, Vt, safeAlpha, distribution, fd, safeTint) * weight;
        }
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        vec3 F = mx_compute_fresnel(NdotV, fd);

        vec3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        vec3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
        bsdf.throughput = 1.0 - dirAlbedo * weight;

        vec3 Li = mx_environment_radiance(frame, Vt, safeAlpha, distribution, fd);
        bsdf.response = Li * safeTint * comp * weight;
    }
}
