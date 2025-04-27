// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from learning_interface:msg/STM32.idl
// generated code does not contain a copyright notice

#ifndef LEARNING_INTERFACE__MSG__DETAIL__STM32__STRUCT_HPP_
#define LEARNING_INTERFACE__MSG__DETAIL__STM32__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__learning_interface__msg__STM32 __attribute__((deprecated))
#else
# define DEPRECATED__learning_interface__msg__STM32 __declspec(deprecated)
#endif

namespace learning_interface
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct STM32_
{
  using Type = STM32_<ContainerAllocator>;

  explicit STM32_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ifarrive = 0l;
      this->id = 0l;
      this->state = 0l;
      this->yaw = 0.0f;
    }
  }

  explicit STM32_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ifarrive = 0l;
      this->id = 0l;
      this->state = 0l;
      this->yaw = 0.0f;
    }
  }

  // field types and members
  using _ifarrive_type =
    int32_t;
  _ifarrive_type ifarrive;
  using _id_type =
    int32_t;
  _id_type id;
  using _state_type =
    int32_t;
  _state_type state;
  using _yaw_type =
    float;
  _yaw_type yaw;

  // setters for named parameter idiom
  Type & set__ifarrive(
    const int32_t & _arg)
  {
    this->ifarrive = _arg;
    return *this;
  }
  Type & set__id(
    const int32_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__state(
    const int32_t & _arg)
  {
    this->state = _arg;
    return *this;
  }
  Type & set__yaw(
    const float & _arg)
  {
    this->yaw = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    learning_interface::msg::STM32_<ContainerAllocator> *;
  using ConstRawPtr =
    const learning_interface::msg::STM32_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<learning_interface::msg::STM32_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<learning_interface::msg::STM32_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      learning_interface::msg::STM32_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<learning_interface::msg::STM32_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      learning_interface::msg::STM32_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<learning_interface::msg::STM32_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<learning_interface::msg::STM32_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<learning_interface::msg::STM32_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__learning_interface__msg__STM32
    std::shared_ptr<learning_interface::msg::STM32_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__learning_interface__msg__STM32
    std::shared_ptr<learning_interface::msg::STM32_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const STM32_ & other) const
  {
    if (this->ifarrive != other.ifarrive) {
      return false;
    }
    if (this->id != other.id) {
      return false;
    }
    if (this->state != other.state) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    return true;
  }
  bool operator!=(const STM32_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct STM32_

// alias to use template instance with default allocator
using STM32 =
  learning_interface::msg::STM32_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace learning_interface

#endif  // LEARNING_INTERFACE__MSG__DETAIL__STM32__STRUCT_HPP_
