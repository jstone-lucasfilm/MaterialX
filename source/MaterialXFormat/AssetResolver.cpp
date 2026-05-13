//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXFormat/AssetResolver.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

// Build the search path used by both resolve() and resolveForWrite():
// sibling-of-source first (if provided), then the registered search path.
// This mirrors the legacy GenContext behavior of implicitly trusting the
// parent directory of the referencing document.
FileSearchPath buildFileSearch(const FilePath& sourceContext, const FileSearchPath& searchPath)
{
    FileSearchPath fileSearch;
    if (!sourceContext.isEmpty())
    {
        fileSearch.append(sourceContext);
    }
    fileSearch.append(searchPath);
    return fileSearch;
}

} // anonymous namespace

FilePath AssetResolver::resolve(const FilePath& reference, const FilePath& sourceContext) const
{
    if (reference.isEmpty())
    {
        return FilePath();
    }
    return buildFileSearch(sourceContext, _searchPath).find(reference).getNormalized();
}

FilePath AssetResolver::resolveForWrite(const FilePath& reference, const FilePath& sourceContext) const
{
    if (reference.isEmpty())
    {
        return FilePath();
    }
    return buildFileSearch(sourceContext, _searchPath).find(reference);
}

MATERIALX_NAMESPACE_END
