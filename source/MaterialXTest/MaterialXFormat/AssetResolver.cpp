//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXFormat/AssetResolver.h>
#include <MaterialXFormat/Util.h>

namespace mx = MaterialX;

TEST_CASE("AssetResolver: empty reference yields empty path", "[assetresolver]")
{
    mx::AssetResolver resolver;
    mx::FilePath result = resolver.resolve(mx::FilePath());
    REQUIRE(result.isEmpty());
}

TEST_CASE("AssetResolver: relative reference resolves via registered root", "[assetresolver]")
{
    mx::FileSearchPath dataSearch = mx::getDefaultDataSearchPath();
    mx::FilePath stdlibRoot = dataSearch.find("libraries/stdlib");
    REQUIRE(stdlibRoot.exists());

    mx::AssetResolver resolver;
    resolver.appendSearchPath(stdlibRoot);

    mx::FilePath result = resolver.resolve("stdlib_defs.mtlx");
    REQUIRE(result.exists());
}

TEST_CASE("AssetResolver: relative reference missing from roots returns input", "[assetresolver]")
{
    mx::FileSearchPath dataSearch = mx::getDefaultDataSearchPath();
    mx::AssetResolver resolver;
    resolver.appendSearchPath(dataSearch);

    mx::FilePath result = resolver.resolve("no_such_file_xyz.mtlx");
    REQUIRE_FALSE(result.exists());
    REQUIRE(result.asString() == "no_such_file_xyz.mtlx");
}

TEST_CASE("AssetResolver: absolute reference that exists is returned unchanged", "[assetresolver]")
{
    mx::FileSearchPath dataSearch = mx::getDefaultDataSearchPath();
    mx::FilePath absPath = dataSearch.find("libraries/stdlib/stdlib_defs.mtlx");
    REQUIRE(absPath.exists());
    REQUIRE(absPath.isAbsolute());

    // No search roots registered — exercising the absolute-path path.
    mx::AssetResolver resolver;
    mx::FilePath result = resolver.resolve(absPath.asString());
    REQUIRE(result.exists());
}

TEST_CASE("AssetResolver: absolute reference that does not exist is returned unchanged", "[assetresolver]")
{
    mx::AssetResolver resolver;
#if defined(_WIN32)
    mx::FilePath result = resolver.resolve(mx::FilePath("C:\\definitely\\not\\here\\xyz.mtlx"));
#else
    mx::FilePath result = resolver.resolve(mx::FilePath("/definitely/not/here/xyz.mtlx"));
#endif
    REQUIRE_FALSE(result.exists());
}

TEST_CASE("AssetResolver: sibling-of-source resolution", "[assetresolver]")
{
    mx::FileSearchPath dataSearch = mx::getDefaultDataSearchPath();
    mx::FilePath stdlibRoot = dataSearch.find("libraries/stdlib");
    REQUIRE(stdlibRoot.exists());

    mx::AssetResolver resolver;
    // Intentionally do not add any search roots; rely on sibling context.
    mx::FilePath result = resolver.resolve("stdlib_defs.mtlx", stdlibRoot);
    REQUIRE(result.exists());
}

TEST_CASE("AssetResolver: sibling-of-source falls through to roots", "[assetresolver]")
{
    mx::FileSearchPath dataSearch = mx::getDefaultDataSearchPath();
    mx::FilePath stdlibRoot = dataSearch.find("libraries/stdlib");
    mx::FilePath unrelatedDir = mx::FilePath::getCurrentPath();
    REQUIRE(stdlibRoot.exists());

    mx::AssetResolver resolver;
    resolver.appendSearchPath(stdlibRoot);

    // Sibling context does not contain the file, but a registered root does.
    mx::FilePath result = resolver.resolve("stdlib_defs.mtlx", unrelatedDir);
    REQUIRE(result.exists());
}

TEST_CASE("AssetResolver: getSearchPath reflects appended entries", "[assetresolver]")
{
    mx::FilePath a("a/b/c");
    mx::FilePath d("d/e/f");

    mx::AssetResolver resolver;
    resolver.appendSearchPath(a);
    resolver.appendSearchPath(d);

    REQUIRE(resolver.getSearchPath().size() == 2);
}

TEST_CASE("AssetResolver: subclass can override resolve()", "[assetresolver]")
{
    class FixedResolver : public mx::AssetResolver
    {
      public:
        FixedResolver(mx::FilePath canned) :
            _canned(std::move(canned))
        {
        }
        mx::FilePath resolve(const mx::FilePath& /*reference*/,
                             const mx::FilePath& /*sourceContext*/) const override
        {
            return _canned;
        }

      private:
        mx::FilePath _canned;
    };

    mx::FilePath dataPath = mx::getDefaultDataSearchPath().find("libraries/stdlib/stdlib_defs.mtlx");
    REQUIRE(dataPath.exists());

    FixedResolver resolver(dataPath);
    mx::FilePath result = resolver.resolve(mx::FilePath("ignored.txt"), mx::FilePath());
    REQUIRE(result == dataPath);
}

TEST_CASE("AssetResolver: resolveForWrite honors roots without requiring existence", "[assetresolver]")
{
    mx::FileSearchPath dataSearch = mx::getDefaultDataSearchPath();
    mx::FilePath stdlibRoot = dataSearch.find("libraries/stdlib");
    REQUIRE(stdlibRoot.exists());

    mx::AssetResolver resolver;
    resolver.appendSearchPath(stdlibRoot);

    // Non-existent destination -- resolveForWrite returns a usable path
    // (the input) rather than failing.
    mx::FilePath writePath = resolver.resolveForWrite("about_to_be_created.glsl");
    REQUIRE(!writePath.isEmpty());

    // Existing destination -- resolveForWrite returns the located path.
    mx::FilePath existingWrite = resolver.resolveForWrite("stdlib_defs.mtlx");
    REQUIRE(existingWrite.exists());
}

TEST_CASE("AssetResolver: setSearchPath replaces existing entries", "[assetresolver]")
{
    mx::AssetResolver resolver;
    resolver.appendSearchPath(mx::FilePath("a/b"));
    resolver.appendSearchPath(mx::FilePath("c/d"));
    REQUIRE(resolver.getSearchPath().size() == 2);

    mx::FileSearchPath replacement;
    replacement.append(mx::FilePath("e/f"));
    resolver.setSearchPath(replacement);
    REQUIRE(resolver.getSearchPath().size() == 1);
}

TEST_CASE("AssetResolver: composition wrapper delegates on miss", "[assetresolver]")
{
    class WrapperResolver : public mx::AssetResolver
    {
      public:
        explicit WrapperResolver(mx::ConstAssetResolverPtr inner) :
            _inner(std::move(inner))
        {
        }
        mx::FilePath resolve(const mx::FilePath& reference,
                             const mx::FilePath& sourceContext) const override
        {
            mx::FilePath own = mx::AssetResolver::resolve(reference, sourceContext);
            if (own.exists())
            {
                return own;
            }
            return _inner->resolve(reference, sourceContext);
        }

      private:
        mx::ConstAssetResolverPtr _inner;
    };

    mx::FileSearchPath dataSearch = mx::getDefaultDataSearchPath();
    mx::FilePath stdlibRoot = dataSearch.find("libraries/stdlib");
    REQUIRE(stdlibRoot.exists());

    auto inner = std::make_shared<mx::AssetResolver>();
    inner->appendSearchPath(stdlibRoot);

    WrapperResolver wrapper(inner);
    // Wrapper itself has no roots; falls through to inner.
    mx::FilePath result = wrapper.resolve(mx::FilePath("stdlib_defs.mtlx"), mx::FilePath());
    REQUIRE(result.exists());
}
