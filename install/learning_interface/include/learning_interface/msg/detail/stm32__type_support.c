// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from learning_interface:msg/STM32.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "learning_interface/msg/detail/stm32__rosidl_typesupport_introspection_c.h"
#include "learning_interface/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "learning_interface/msg/detail/stm32__functions.h"
#include "learning_interface/msg/detail/stm32__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void STM32__rosidl_typesupport_introspection_c__STM32_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  learning_interface__msg__STM32__init(message_memory);
}

void STM32__rosidl_typesupport_introspection_c__STM32_fini_function(void * message_memory)
{
  learning_interface__msg__STM32__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember STM32__rosidl_typesupport_introspection_c__STM32_message_member_array[4] = {
  {
    "ifarrive",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(learning_interface__msg__STM32, ifarrive),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(learning_interface__msg__STM32, id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(learning_interface__msg__STM32, state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "yaw",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(learning_interface__msg__STM32, yaw),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers STM32__rosidl_typesupport_introspection_c__STM32_message_members = {
  "learning_interface__msg",  // message namespace
  "STM32",  // message name
  4,  // number of fields
  sizeof(learning_interface__msg__STM32),
  STM32__rosidl_typesupport_introspection_c__STM32_message_member_array,  // message members
  STM32__rosidl_typesupport_introspection_c__STM32_init_function,  // function to initialize message memory (memory has to be allocated)
  STM32__rosidl_typesupport_introspection_c__STM32_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t STM32__rosidl_typesupport_introspection_c__STM32_message_type_support_handle = {
  0,
  &STM32__rosidl_typesupport_introspection_c__STM32_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_learning_interface
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, learning_interface, msg, STM32)() {
  if (!STM32__rosidl_typesupport_introspection_c__STM32_message_type_support_handle.typesupport_identifier) {
    STM32__rosidl_typesupport_introspection_c__STM32_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &STM32__rosidl_typesupport_introspection_c__STM32_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
