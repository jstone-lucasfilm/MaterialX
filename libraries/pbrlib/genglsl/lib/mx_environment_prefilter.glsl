#include "mx_microfacet_specular.glsl"

// Return the mip level associated with the given alpha in a prefiltered environment.
float mx_latlong_alpha_to_lod(float alpha)
{
    float lodBias = (alpha < 0.25) ? sqrt(alpha) : 0.5 * alpha + 0.375;
    return lodBias * float($envRadianceMips - 1);
}

vec3 mx_environment_radiance(TangentFrame frame, vec3 Vt, vec2 alpha, int distribution, FresnelData fd)
{
    const vec3 Nt = vec3(0.0, 0.0, 1.0);
    vec3 Lt = fd.refraction ? mx_refraction_solid_sphere(-Vt, Nt, fd.ior.x) : -reflect(Vt, Nt);
    vec3 Lw = mx_tangent_to_world(frame, Lt);

    float NdotV = clamp(Vt.z, M_FLOAT_EPS, 1.0);

    float avgAlpha = mx_average_alpha(alpha);
    vec3 FG = mx_ggx_dir_albedo(NdotV, avgAlpha, fd);
    FG = fd.refraction ? vec3(1.0) - FG : FG;

    vec3 Li = mx_latlong_map_lookup(Lw, $envMatrix, mx_latlong_alpha_to_lod(avgAlpha), $envRadiance);
    return Li * FG * $envLightIntensity;
}

vec3 mx_environment_irradiance(vec3 N)
{
    vec3 Li = mx_latlong_map_lookup(N, $envMatrix, 0.0, $envIrradiance);
    return Li * $envLightIntensity;
}
