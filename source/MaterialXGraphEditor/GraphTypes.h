//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GRAPHTYPES_H
#define MATERIALX_GRAPHTYPES_H

#include <MaterialXCore/Node.h>

#include <imgui.h>

inline const std::string NODEDEF_BACKDROP = "ND_backdrop";
inline const std::string NODEDEF_NODEGRAPH = "ND_nodegraph";

inline const std::string GROUP_INPUT_NODES = "Input Nodes";
inline const std::string GROUP_OUTPUT_NODES = "Output Nodes";
inline const std::string GROUP_BACKDROPS = "Backdrops";
inline const std::string GROUP_NODEGRAPH = "Node Graph";

// A link connects two pins and includes a unique id and the ids of the two pins it connects
// Based on the Link struct from ImGui Node Editor blueprints-examples.cpp
struct Link
{
    Link() :
        _startAttr(-1),
        _endAttr(-1)
    {
        static int nextId = 1;
        _id = nextId++;
    }

    int _startAttr, _endAttr;
    int _id;
};

inline std::unordered_map<std::string, ImColor> createPinColorMap()
{
    std::unordered_map<std::string, ImColor> pinColor;

    // Scalar types
    pinColor.emplace("integer", ImColor(255, 255, 28, 255));
    pinColor.emplace("boolean", ImColor(255, 0, 255, 255));
    pinColor.emplace("float", ImColor(50, 100, 255, 255));

    // Vector and color types
    pinColor.emplace("vector2", ImColor(100, 255, 100, 255));
    pinColor.emplace("vector3", ImColor(0, 255, 0, 255));
    pinColor.emplace("vector4", ImColor(100, 0, 100, 255));
    pinColor.emplace("color3", ImColor(178, 34, 34, 255));
    pinColor.emplace("color4", ImColor(50, 10, 255, 255));

    // Matrix types
    pinColor.emplace("matrix33", ImColor(0, 100, 100, 255));
    pinColor.emplace("matrix44", ImColor(50, 255, 100, 255));

    // String types
    pinColor.emplace("string", ImColor(100, 100, 50, 255));
    pinColor.emplace("filename", ImColor(255, 184, 28, 255));

    // Distribution types
    pinColor.emplace("BSDF", ImColor(10, 181, 150, 255));
    pinColor.emplace("EDF", ImColor(255, 50, 100, 255));
    pinColor.emplace("VDF", ImColor(0, 100, 151, 255));

    // Shader and material types
    pinColor.emplace(MaterialX::SURFACE_SHADER_TYPE_STRING, ImColor(150, 255, 255, 255));
    pinColor.emplace(MaterialX::DISPLACEMENT_SHADER_TYPE_STRING, ImColor(155, 50, 100, 255));
    pinColor.emplace(MaterialX::LIGHT_SHADER_TYPE_STRING, ImColor(100, 150, 100, 255));
    pinColor.emplace(MaterialX::VOLUME_SHADER_TYPE_STRING, ImColor(155, 250, 100, 255));
    pinColor.emplace(MaterialX::MATERIAL_TYPE_STRING, ImColor(255, 255, 255, 255));

    // Special types
    pinColor.emplace("none", ImColor(140, 70, 70, 255));
    pinColor.emplace("geomname", ImColor(121, 60, 180, 255));
    pinColor.emplace(MaterialX::MULTI_OUTPUT_TYPE_STRING, ImColor(70, 70, 70, 255));

    // Array types
    pinColor.emplace("integerarray", ImColor(200, 10, 100, 255));
    pinColor.emplace("floatarray", ImColor(25, 250, 100));
    pinColor.emplace("color3array", ImColor(25, 200, 110));
    pinColor.emplace("color4array", ImColor(50, 240, 110));
    pinColor.emplace("vector2array", ImColor(50, 200, 75));
    pinColor.emplace("vector3array", ImColor(20, 200, 100));
    pinColor.emplace("vector4array", ImColor(100, 200, 100));
    pinColor.emplace("geomnamearray", ImColor(150, 200, 100));
    pinColor.emplace("stringarray", ImColor(120, 180, 100));

    return pinColor;
}

inline const ImColor DEFAULT_PIN_COLOR = ImColor(200, 200, 200, 255);

#endif
