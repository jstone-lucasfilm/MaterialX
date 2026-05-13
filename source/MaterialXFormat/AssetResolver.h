//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_ASSETRESOLVER_H
#define MATERIALX_ASSETRESOLVER_H

/// @file
/// Centralized resolution of file references made by MaterialX documents.

#include <MaterialXFormat/Export.h>
#include <MaterialXFormat/File.h>

MATERIALX_NAMESPACE_BEGIN

class AssetResolver;

/// A shared pointer to an AssetResolver.
using AssetResolverPtr = std::shared_ptr<AssetResolver>;

/// A shared pointer to a const AssetResolver.
using ConstAssetResolverPtr = std::shared_ptr<const AssetResolver>;

/// @class AssetResolver
/// Resolves a file reference made by a MaterialX document into a concrete
/// FilePath that callers may then open.
///
/// AssetResolver is the single chokepoint for asset resolution in MaterialX.
/// XML loading, XInclude expansion, shader-source lookup, and image lookup
/// all route through resolve().  This centralization is the structural
/// prerequisite for security policy enforcement, which will be added on top
/// in a follow-up change.
///
/// The default implementation matches the legacy FileSearchPath::find
/// behavior: relative references are searched against a sequence of roots;
/// the parent directory of the referencing document is implicitly prepended
/// to the search; absolute references are returned as-is if they exist.
///
/// Subclasses may override resolve() to integrate with studio asset systems,
/// custom URI schemes, sandboxed virtual file systems, or any other source
/// of named content.
class MX_FORMAT_API AssetResolver
{
  public:
    AssetResolver() = default;
    virtual ~AssetResolver() = default;

    /// Append a path to the search path.  Entries are consulted in the
    /// order added.
    void appendSearchPath(const FilePath& path)
    {
        _searchPath.append(path);
    }

    /// Append all entries from the given FileSearchPath to the search path.
    void appendSearchPath(const FileSearchPath& path)
    {
        _searchPath.append(path);
    }

    /// Replace the search path with the given set of entries.
    void setSearchPath(const FileSearchPath& path)
    {
        _searchPath = path;
    }

    /// Return the active search path.
    const FileSearchPath& getSearchPath() const
    {
        return _searchPath;
    }

    /// Resolve an asset reference to a concrete FilePath.
    ///
    /// @param reference The raw reference value from the source document
    ///        (e.g. an `<implementation file="...">` attribute or an
    ///        `xi:include href="..."` attribute).
    /// @param sourceContext The parent directory of the document making the
    ///        reference.  When non-empty it is implicitly trusted as a
    ///        search root for this call only, enabling sibling-file
    ///        resolution.  Pass an empty path when there is no enclosing
    ///        document (e.g. resolving a top-level command-line argument).
    /// @return On a successful match within the search roots, the resolved
    ///         and normalized path.  On no match, the reference itself
    ///         (matching FileSearchPath::find semantics, so callers can
    ///         probe existence and surface the original on failure).
    virtual FilePath resolve(
        const FilePath& reference,
        const FilePath& sourceContext = FilePath()) const;

    /// Resolve a reference whose target need not yet exist on disk.  Walks
    /// the search roots in the same order as resolve(), but returns the
    /// combined path without requiring the file to be present.  Intended
    /// for write operations where the destination is being created.
    virtual FilePath resolveForWrite(
        const FilePath& reference,
        const FilePath& sourceContext = FilePath()) const;

  private:
    FileSearchPath _searchPath;
};

MATERIALX_NAMESPACE_END

#endif
