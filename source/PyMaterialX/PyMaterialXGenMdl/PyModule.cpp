//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyMdlShaderGenerator(py::module& mod);

PYBIND11_MODULE(PyMaterialXGenMdl, mod)
{
    mod.doc() = "Shader generation using the Material Definition Language.";

    // PyMaterialXGenMdl depends on types defined in PyMaterialXGenShader
    PYMATERIALX_IMPORT_MODULE(PyMaterialXGenShader);

    bindPyMdlShaderGenerator(mod);
};
