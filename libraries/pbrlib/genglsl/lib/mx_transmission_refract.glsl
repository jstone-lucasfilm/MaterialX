#include "mx_microfacet_specular.glsl"

vec3 mx_surface_transmission(TangentFrame frame, vec3 Vt, vec2 alpha, int distribution, FresnelData fd, vec3 tint)
{
    // Approximate the appearance of surface transmission as glossy
    // environment map refraction, ignoring any scene geometry that might
    // be visible through the surface.
    fd.refraction = true;
    if ($refractionTwoSided)
    {
        tint = mx_square(tint);
    }
    return mx_environment_radiance(frame, Vt, alpha, distribution, fd) * tint;
}
