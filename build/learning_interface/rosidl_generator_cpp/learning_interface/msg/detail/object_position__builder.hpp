// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from learning_interface:msg/ObjectPosition.idl
// generated code does not contain a copyright notice

#ifndef LEARNING_INTERFACE__MSG__DETAIL__OBJECT_POSITION__BUILDER_HPP_
#define LEARNING_INTERFACE__MSG__DETAIL__OBJECT_POSITION__BUILDER_HPP_

#include "learning_interface/msg/detail/object_position__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace learning_interface
{

namespace msg
{

namespace builder
{

class Init_ObjectPosition_kind
{
public:
  explicit Init_ObjectPosition_kind(::learning_interface::msg::ObjectPosition & msg)
  : msg_(msg)
  {}
  ::learning_interface::msg::ObjectPosition kind(::learning_interface::msg::ObjectPosition::_kind_type arg)
  {
    msg_.kind = std::move(arg);
    return std::move(msg_);
  }

private:
  ::learning_interface::msg::ObjectPosition msg_;
};

class Init_ObjectPosition_f
{
public:
  explicit Init_ObjectPosition_f(::learning_interface::msg::ObjectPosition & msg)
  : msg_(msg)
  {}
  Init_ObjectPosition_kind f(::learning_interface::msg::ObjectPosition::_f_type arg)
  {
    msg_.f = std::move(arg);
    return Init_ObjectPosition_kind(msg_);
  }

private:
  ::learning_interface::msg::ObjectPosition msg_;
};

class Init_ObjectPosition_z
{
public:
  explicit Init_ObjectPosition_z(::learning_interface::msg::ObjectPosition & msg)
  : msg_(msg)
  {}
  Init_ObjectPosition_f z(::learning_interface::msg::ObjectPosition::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_ObjectPosition_f(msg_);
  }

private:
  ::learning_interface::msg::ObjectPosition msg_;
};

class Init_ObjectPosition_y
{
public:
  explicit Init_ObjectPosition_y(::learning_interface::msg::ObjectPosition & msg)
  : msg_(msg)
  {}
  Init_ObjectPosition_z y(::learning_interface::msg::ObjectPosition::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_ObjectPosition_z(msg_);
  }

private:
  ::learning_interface::msg::ObjectPosition msg_;
};

class Init_ObjectPosition_x
{
public:
  Init_ObjectPosition_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ObjectPosition_y x(::learning_interface::msg::ObjectPosition::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_ObjectPosition_y(msg_);
  }

private:
  ::learning_interface::msg::ObjectPosition msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::learning_interface::msg::ObjectPosition>()
{
  return learning_interface::msg::builder::Init_ObjectPosition_x();
}

}  // namespace learning_interface

#endif  // LEARNING_INTERFACE__MSG__DETAIL__OBJECT_POSITION__BUILDER_HPP_
