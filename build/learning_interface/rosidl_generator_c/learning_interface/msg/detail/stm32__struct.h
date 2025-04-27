// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from learning_interface:msg/STM32.idl
// generated code does not contain a copyright notice

#ifndef LEARNING_INTERFACE__MSG__DETAIL__STM32__STRUCT_H_
#define LEARNING_INTERFACE__MSG__DETAIL__STM32__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/STM32 in the package learning_interface.
typedef struct learning_interface__msg__STM32
{
  int32_t ifarrive;
  int32_t id;
  int32_t state;
  float yaw;
} learning_interface__msg__STM32;

// Struct for a sequence of learning_interface__msg__STM32.
typedef struct learning_interface__msg__STM32__Sequence
{
  learning_interface__msg__STM32 * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} learning_interface__msg__STM32__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // LEARNING_INTERFACE__MSG__DETAIL__STM32__STRUCT_H_
