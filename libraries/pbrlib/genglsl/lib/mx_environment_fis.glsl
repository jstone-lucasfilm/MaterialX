#include "mx_microfacet_specular.glsl"

vec3 mx_environment_radiance(TangentFrame frame, vec3 Vt, vec2 alpha, int distribution, FresnelData fd)
{
    // Compute derived properties.
    float NdotV = clamp(Vt.z, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(alpha);
    float G1V = mx_ggx_smith_G1(NdotV, avgAlpha);

    // Integrate outgoing radiance using filtered importance sampling.
    // http://cgg.mff.cuni.cz/~jaroslav/papers/2008-egsr-fis/2008-egsr-fis-final-embedded.pdf
    vec3 radiance = vec3(0.0);
    int envRadianceSamples = $envRadianceSamples;
    for (int i = 0; i < envRadianceSamples; i++)
    {
        vec2 Xi = mx_spherical_fibonacci(i, envRadianceSamples);

        // Compute the half vector and incoming light direction.
        vec3 Ht = mx_ggx_importance_sample_VNDF(Xi, Vt, alpha);
        vec3 Lt = fd.refraction ? mx_refraction_solid_sphere(-Vt, Ht, fd.ior.x) : -reflect(Vt, Ht);

        // Compute dot products for this sample.
        float NdotL = clamp(Lt.z, M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(Vt, Ht), M_FLOAT_EPS, 1.0);

        // Sample the environment light from the given direction.
        vec3 Lw = mx_tangent_to_world(frame, Lt);
        float pdf = mx_ggx_VNDF_reflection_PDF(Ht, alpha, G1V, NdotV);
        float lod = mx_latlong_compute_lod(Lw, pdf, float($envRadianceMips - 1), envRadianceSamples);
        vec3 sampleColor = mx_latlong_map_lookup(Lw, $envMatrix, lod, $envRadiance);

        // Compute the Fresnel term.
        vec3 F = mx_compute_fresnel(VdotH, fd);

        // Compute the geometric term.
        float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

        // Compute the combined FG term, which simplifies to inverted Fresnel for refraction.
        vec3 FG = fd.refraction ? vec3(1.0) - F : F * G;

        // Add the radiance contribution of this sample.
        // From https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        //   incidentLight = sampleColor * NdotL
        //   microfacetSpecular = D * F * G / (4 * NdotL * NdotV)
        //   pdf = D * G1V / (4 * NdotV);
        //   radiance = incidentLight * microfacetSpecular / pdf
        radiance += sampleColor * FG;
    }

    // Apply the global component of the geometric term and normalize.
    radiance /= G1V * float(envRadianceSamples);

    // Return the final radiance.
    return ($envRadianceSamples == 0 ? vec3(0.0) : radiance) * $envLightIntensity;
}

vec3 mx_environment_irradiance(vec3 N)
{
    vec3 Li = mx_latlong_map_lookup(N, $envMatrix, 0.0, $envIrradiance);
    return Li * $envLightIntensity;
}
