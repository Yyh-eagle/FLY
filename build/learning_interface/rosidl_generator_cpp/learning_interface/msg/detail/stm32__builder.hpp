// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from learning_interface:msg/STM32.idl
// generated code does not contain a copyright notice

#ifndef LEARNING_INTERFACE__MSG__DETAIL__STM32__BUILDER_HPP_
#define LEARNING_INTERFACE__MSG__DETAIL__STM32__BUILDER_HPP_

#include "learning_interface/msg/detail/stm32__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace learning_interface
{

namespace msg
{

namespace builder
{

class Init_STM32_yaw
{
public:
  explicit Init_STM32_yaw(::learning_interface::msg::STM32 & msg)
  : msg_(msg)
  {}
  ::learning_interface::msg::STM32 yaw(::learning_interface::msg::STM32::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return std::move(msg_);
  }

private:
  ::learning_interface::msg::STM32 msg_;
};

class Init_STM32_state
{
public:
  explicit Init_STM32_state(::learning_interface::msg::STM32 & msg)
  : msg_(msg)
  {}
  Init_STM32_yaw state(::learning_interface::msg::STM32::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_STM32_yaw(msg_);
  }

private:
  ::learning_interface::msg::STM32 msg_;
};

class Init_STM32_id
{
public:
  explicit Init_STM32_id(::learning_interface::msg::STM32 & msg)
  : msg_(msg)
  {}
  Init_STM32_state id(::learning_interface::msg::STM32::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_STM32_state(msg_);
  }

private:
  ::learning_interface::msg::STM32 msg_;
};

class Init_STM32_ifarrive
{
public:
  Init_STM32_ifarrive()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_STM32_id ifarrive(::learning_interface::msg::STM32::_ifarrive_type arg)
  {
    msg_.ifarrive = std::move(arg);
    return Init_STM32_id(msg_);
  }

private:
  ::learning_interface::msg::STM32 msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::learning_interface::msg::STM32>()
{
  return learning_interface::msg::builder::Init_STM32_ifarrive();
}

}  // namespace learning_interface

#endif  // LEARNING_INTERFACE__MSG__DETAIL__STM32__BUILDER_HPP_
