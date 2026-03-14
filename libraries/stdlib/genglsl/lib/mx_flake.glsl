#include "mx_noise.glsl"

// "Fast Random Rotation Matrices" by James Arvo, Graphics Gems3 P.117
vec3 mx_rotate_flake(vec3 p, vec3 i)
{
    float theta = M_PI * 2.0 * i.x;
    float phi = M_PI * 2.0 * i.y;
    float z = i.z * 2.0;

    float r = sqrt(z);
    float vx = sin(phi) * r;
    float vy = cos(phi) * r;
    float vz = sqrt(2.0 - z);

    float s_theta = sin(theta);
    float c_theta = cos(theta);
    float sx = vx * s_theta - vy * s_theta;
    float sy = vx * c_theta + vy * c_theta;

    mat3 m = mat3(
        vx * sx - s_theta, vx * sy - s_theta, vx * vz,
        vy * sx + c_theta, vy * sy - c_theta, vy * vz,
        vz * sx          , vz * sy          , 1.0 - z
    );

    return m * p;
}

// compute a flake probability for a given flake coverage density x
float mx_flake_density_to_probability(float x)
{
    // constants for numerical fitted curve to observed flake noise density behavior
    const vec4 abcd = vec4(-26.19771808f, 26.39663835f, 85.53857017f, -102.35069432f);
    const vec2 ef = vec2(-101.42634862f, 118.45082288f);
    float xx = x * x;

    return (abcd.x * xx + abcd.y * x) / (abcd.z * xx * x + abcd.w * xx  + ef.x * x + ef.y);
}

void mx_flake(
    float size,
    float roughness,
    float coverage,
    vec3 position,
    vec3 normal,
    vec3 tangent,
    vec3 bitangent,
    out int id,
    out float rand,
    out float presence,
    out vec3 flakenormal
)
{
    float probability = mx_flake_density_to_probability(clamp(coverage, 0.0, 1.0));
    const float flake_diameter = 1.5 / sqrt(3.0);

    vec3 P = position / vec3(size);
    vec3 base_P = floor(P);
    ivec3 base_P_int = ivec3(base_P);

    // flake priority in [0..1], 0: no flake, flakes with higher priority shadow flakes "below" them
    float flake_priority = 0.0;
    ivec3 flake_cell = ivec3(0);

    // Examine the 3x3x3 lattice neighborhood around the sample cell. Flakes are seeded at cell
    // centers but can overlap adjacent cells by up to flake_diameter, so neighbors may contribute
    // at the sample position. For each neighbor we deterministically generate a hash, reject it
    // by the density probability, compute a per-flake priority, and test the rotated, centered
    // flake position for overlap. The highest-priority overlapping flake is selected.
    for (int i = -1; i < 2; ++i)
    {
        for (int j = -1; j < 2; ++j)
        {
            for (int k = -1; k < 2; ++k)
            {
                ivec3 cell = base_P_int + ivec3(i, j, k);

                if (mx_bits_to_01(mx_hash_int(cell.x, cell.y, cell.z)) > probability)
                    continue;

                float priority = mx_bits_to_01(mx_hash_int(cell.x, cell.y, cell.z, 1));
                if (priority < flake_priority)
                    continue;

                vec3 flake_P = base_P + vec3(i, j, k) + vec3(0.5);
                vec3 PP = P - flake_P;
                if (dot(PP, PP) >= flake_diameter * flake_diameter * 4.0)
                    continue;

                vec3 rot;
                rot.x = mx_bits_to_01(mx_hash_int(cell.x, cell.y, cell.z, 2));
                rot.y = mx_bits_to_01(mx_hash_int(cell.x, cell.y, cell.z, 3));
                rot.z = mx_bits_to_01(mx_hash_int(cell.x, cell.y, cell.z, 4));
                PP = mx_rotate_flake(PP, rot);

                if (abs(PP.x) <= flake_diameter &&
                    abs(PP.y) <= flake_diameter &&
                    abs(PP.z) <= flake_diameter)
                {
                    flake_priority = priority;
                    flake_cell = cell;
                }
            }
        }
    }

    if (flake_priority <= 0.0)
    {
        // no flake
        id = 0;
        rand = 0.0;
        presence = 0.0;
        flakenormal = normal;
        return;
    }

    // create a flake normal by importance sampling a microfacet distribution with given roughness
    float xi0 = mx_bits_to_01(mx_hash_int(flake_cell.x, flake_cell.y, flake_cell.z, 5));
    float xi1 = mx_bits_to_01(mx_hash_int(flake_cell.x, flake_cell.y, flake_cell.z, 6));

    uint id_hash = mx_hash_int(flake_cell.x, flake_cell.y, flake_cell.z, 7);
    id = int(id_hash);  // not ideal but MaterialX does not support unsigned integer type
    rand = mx_bits_to_01(id_hash);
    presence = flake_priority;

    float phi = M_PI * 2.0 * xi0;
    float tan_theta = roughness * roughness * sqrt(xi1) / sqrt(1.0 - xi1);  // GGX
    float sin_theta = tan_theta / sqrt(1.0 + tan_theta * tan_theta);
    float cos_theta = sqrt(1.0 - sin_theta * sin_theta);

    flakenormal = tangent * cos(phi) * sin_theta +
                  bitangent * sin(phi) * sin_theta +
                  normal * cos_theta;
    flakenormal = normalize(flakenormal);
}
