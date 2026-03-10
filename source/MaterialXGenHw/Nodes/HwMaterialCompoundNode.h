//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_HWMATERIALCOMPOUNDNODE_H
#define MATERIALX_HWMATERIALCOMPOUNDNODE_H

#include <MaterialXGenHw/Nodes/HwSurfaceNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Generic implementation for compound material nodes in hardware languages.
///
/// This handles any nodegraph that outputs material type and contains
/// an internal surface node, such as LamaSurface or artist-authored
/// material compounds. The input name mappings are discovered automatically
/// by analyzing the nodegraph's internal surface node connections.
class MX_GENHW_API HwMaterialCompoundNode : public HwSurfaceNode
{
  public:
    static ShaderNodeImplPtr create();

    void initialize(const InterfaceElement& element, GenContext& context) override;
    void addClassification(ShaderNode& node) const override;

  protected:
    const string& getBsdfInputName() const override;
    const string& getEdfInputName() const override;
    const string& getOpacityInputName() const override;

  private:
    string _bsdfInputName;
    string _edfInputName;
    string _opacityInputName;
};

MATERIALX_NAMESPACE_END

#endif
