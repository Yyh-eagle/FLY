// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from learning_interface:msg/MyState.idl
// generated code does not contain a copyright notice

#ifndef LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__TRAITS_HPP_
#define LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__TRAITS_HPP_

#include "learning_interface/msg/detail/my_state__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<learning_interface::msg::MyState>()
{
  return "learning_interface::msg::MyState";
}

template<>
inline const char * name<learning_interface::msg::MyState>()
{
  return "learning_interface/msg/MyState";
}

template<>
struct has_fixed_size<learning_interface::msg::MyState>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<learning_interface::msg::MyState>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<learning_interface::msg::MyState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__TRAITS_HPP_
