//
// Copyright (c) 2023 Apple Inc.
// Licensed under the Apache License v2.0
//

#ifndef GENGLSL_H
#define GENGLSL_H

#include <MaterialXTest/MaterialXGenShader/GenShaderUtil.h>

#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include "CompileMsl.h"

#include <cassert>

namespace mx = MaterialX;

class MslShaderGeneratorTester : public GenShaderUtil::ShaderGeneratorTester
{
  public:
    using ParentClass = GenShaderUtil::ShaderGeneratorTester;

    MslShaderGeneratorTester(mx::ShaderGeneratorPtr shaderGenerator, const mx::FilePathVec& testRootPaths, 
                             const mx::FileSearchPath& searchPath, const mx::FilePath& logFilePath, bool writeShadersToDisk) :
        GenShaderUtil::ShaderGeneratorTester(shaderGenerator, testRootPaths, searchPath, logFilePath, writeShadersToDisk)
    {}

    void setTestStages() override
    {
        _testStages.push_back(mx::Stage::VERTEX);
        _testStages.push_back(mx::Stage::PIXEL);
    }

    // Ignore trying to create shader code for displacementshaders
    void addSkipNodeDefs() override
    {
        _skipNodeDefs.insert("ND_displacement_float");
        _skipNodeDefs.insert("ND_displacement_vector3");
        ParentClass::addSkipNodeDefs();
    }

    void addSkipFiles() override
    {
        // To skip specific files for this render target, add them as below:
        // _skipFiles.insert("example.mtlx");
    }

    void setupDependentLibraries() override
    {
        ParentClass::setupDependentLibraries();

        mx::FilePath lightDir = mx::getDefaultDataSearchPath().find("resources/Materials/TestSuite/lights");
        loadLibrary(lightDir / mx::FilePath("light_compound_test.mtlx"), _dependLib);
        loadLibrary(lightDir / mx::FilePath("light_rig_test_1.mtlx"), _dependLib);
    }

    void compileSource(const std::vector<mx::FilePath>& sourceCodePaths) override
    {
        int i = 0;
        for(const mx::FilePath& sourceCodePath : sourceCodePaths)
        {
            assert(i == 0 || i == 1);
            CompileMslShader(sourceCodePath.asString().c_str(), i == 0 ?  "VertexMain" : "FragmentMain");
            ++i;
        }
    }
    
  protected:
    void getImplementationWhiteList(mx::StringSet& whiteList) override
    {
        whiteList =
        {
            "displacementshader", "volumeshader", "surfacematerial", "volumematerial",
            "IM_constant_", "IM_dot_", "IM_angle", "IM_geompropvalue_boolean", "IM_geompropvalue_string", "IM_geompropvalue_filename",
            "IM_light_", "IM_point_light_", "IM_spot_light_", "IM_directional_light_",
            "ND_surfacematerial", "ND_volumematerial"
        };
    }
};

#endif // GENGLSL_H
