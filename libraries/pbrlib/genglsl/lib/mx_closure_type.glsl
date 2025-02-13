// These are defined based on the HwShaderGenerator::ClosureContextType enum
// if that changes - these need to be updated accordingly.

#define CLOSURE_TYPE_DEFAULT 0
#define CLOSURE_TYPE_REFLECTION 1
#define CLOSURE_TYPE_TRANSMISSION 2
#define CLOSURE_TYPE_INDIRECT 3
#define CLOSURE_TYPE_EMISSION 4

struct ClosureData
{
    int closureType;
    vec3 L;
    vec3 V;
    vec3 N;
    vec3 P;
    float occlusion;
};

// Enforce that the given normal is forward-facing in the specified closure context.
vec3 mx_forward_facing_normal(vec3 N, ClosureData closureData)
{
    if (!$forwardFacingNormal)
    {
        return N;
    }

    return (dot(closureData.N, closureData.V) < 0.0) ? -N : N;
}
