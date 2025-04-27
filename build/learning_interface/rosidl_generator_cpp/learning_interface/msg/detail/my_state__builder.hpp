// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from learning_interface:msg/MyState.idl
// generated code does not contain a copyright notice

#ifndef LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__BUILDER_HPP_
#define LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__BUILDER_HPP_

#include "learning_interface/msg/detail/my_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace learning_interface
{

namespace msg
{

namespace builder
{

class Init_MyState_b
{
public:
  explicit Init_MyState_b(::learning_interface::msg::MyState & msg)
  : msg_(msg)
  {}
  ::learning_interface::msg::MyState b(::learning_interface::msg::MyState::_b_type arg)
  {
    msg_.b = std::move(arg);
    return std::move(msg_);
  }

private:
  ::learning_interface::msg::MyState msg_;
};

class Init_MyState_a
{
public:
  Init_MyState_a()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MyState_b a(::learning_interface::msg::MyState::_a_type arg)
  {
    msg_.a = std::move(arg);
    return Init_MyState_b(msg_);
  }

private:
  ::learning_interface::msg::MyState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::learning_interface::msg::MyState>()
{
  return learning_interface::msg::builder::Init_MyState_a();
}

}  // namespace learning_interface

#endif  // LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__BUILDER_HPP_
