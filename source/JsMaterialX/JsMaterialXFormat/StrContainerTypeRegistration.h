//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXFormat/File.h>
#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

template<typename T> struct IsStrContainer : std::false_type {};
template<> struct IsStrContainer<mx::FileSearchPath> : std::true_type {};
template<> struct IsStrContainer<mx::FilePath> : std::true_type {};

using StrContainerIntermediate = std::string;

namespace emscripten 
{

namespace internal 
{

// Register a string-like container type with embind by mapping it to the
// intermediate std::string type.  Explicit specializations are used for the
// const and reference forms so that they take precedence over the generic
// TypeID<const T>, TypeID<T&> specializations introduced in Emscripten 4.0.9.
#define MX_REGISTER_STR_CONTAINER_TYPEID(TYPE)              \
template<> struct TypeID<TYPE> {                            \
  static constexpr TYPEID get() {                           \
    return TypeID<StrContainerIntermediate>::get();         \
  }                                                         \
};                                                          \
template<> struct TypeID<const TYPE> : TypeID<TYPE> {};     \
template<> struct TypeID<TYPE&> : TypeID<TYPE> {};          \
template<> struct TypeID<const TYPE&> : TypeID<TYPE> {};

MX_REGISTER_STR_CONTAINER_TYPEID(mx::FilePath)
MX_REGISTER_STR_CONTAINER_TYPEID(mx::FileSearchPath)

#undef MX_REGISTER_STR_CONTAINER_TYPEID

template<typename T>
struct BindingType<T, typename std::enable_if<IsStrContainer<T>::value, void>::type> {
  typedef typename BindingType<StrContainerIntermediate>::WireType WireType;

  constexpr static WireType toWireType(const T& v) {
    return BindingType<StrContainerIntermediate>::toWireType(v.asString());
  }
  constexpr static T fromWireType(WireType v) {
    return T(BindingType<StrContainerIntermediate>::fromWireType(v));
  }
};

} // namespace internal

} // namespace emscripten
