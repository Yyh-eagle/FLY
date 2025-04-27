// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from learning_interface:msg/MyState.idl
// generated code does not contain a copyright notice

#ifndef LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__STRUCT_H_
#define LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/MyState in the package learning_interface.
typedef struct learning_interface__msg__MyState
{
  int32_t a;
  int32_t b;
} learning_interface__msg__MyState;

// Struct for a sequence of learning_interface__msg__MyState.
typedef struct learning_interface__msg__MyState__Sequence
{
  learning_interface__msg__MyState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} learning_interface__msg__MyState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // LEARNING_INTERFACE__MSG__DETAIL__MY_STATE__STRUCT_H_
