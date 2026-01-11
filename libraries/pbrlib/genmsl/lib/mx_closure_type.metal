// MSL-specific closure type definitions.
// These are defined based on the HwShaderGenerator::ClosureContextType enum
// if that changes - these need to be updated accordingly.

#ifndef MX_CLOSURE_TYPE_DEFINED
#define MX_CLOSURE_TYPE_DEFINED

#define CLOSURE_TYPE_DEFAULT 0
#define CLOSURE_TYPE_REFLECTION 1
#define CLOSURE_TYPE_TRANSMISSION 2
#define CLOSURE_TYPE_INDIRECT 3
#define CLOSURE_TYPE_EMISSION 4

struct ClosureData {
    int closureType;
    float3 L;
    float3 V;
    float3 N;
    float3 P;
    float occlusion;
};

ClosureData makeClosureData(int closureType, float3 L, float3 V, float3 N, float3 P, float occlusion)
{
    return {closureType, L, V, N, P, occlusion};
}

#endif
