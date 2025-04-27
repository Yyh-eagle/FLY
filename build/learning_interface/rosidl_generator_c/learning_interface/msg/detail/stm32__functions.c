// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from learning_interface:msg/STM32.idl
// generated code does not contain a copyright notice
#include "learning_interface/msg/detail/stm32__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
learning_interface__msg__STM32__init(learning_interface__msg__STM32 * msg)
{
  if (!msg) {
    return false;
  }
  // ifarrive
  // id
  // state
  // yaw
  return true;
}

void
learning_interface__msg__STM32__fini(learning_interface__msg__STM32 * msg)
{
  if (!msg) {
    return;
  }
  // ifarrive
  // id
  // state
  // yaw
}

bool
learning_interface__msg__STM32__are_equal(const learning_interface__msg__STM32 * lhs, const learning_interface__msg__STM32 * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // ifarrive
  if (lhs->ifarrive != rhs->ifarrive) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // state
  if (lhs->state != rhs->state) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  return true;
}

bool
learning_interface__msg__STM32__copy(
  const learning_interface__msg__STM32 * input,
  learning_interface__msg__STM32 * output)
{
  if (!input || !output) {
    return false;
  }
  // ifarrive
  output->ifarrive = input->ifarrive;
  // id
  output->id = input->id;
  // state
  output->state = input->state;
  // yaw
  output->yaw = input->yaw;
  return true;
}

learning_interface__msg__STM32 *
learning_interface__msg__STM32__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  learning_interface__msg__STM32 * msg = (learning_interface__msg__STM32 *)allocator.allocate(sizeof(learning_interface__msg__STM32), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(learning_interface__msg__STM32));
  bool success = learning_interface__msg__STM32__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
learning_interface__msg__STM32__destroy(learning_interface__msg__STM32 * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    learning_interface__msg__STM32__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
learning_interface__msg__STM32__Sequence__init(learning_interface__msg__STM32__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  learning_interface__msg__STM32 * data = NULL;

  if (size) {
    data = (learning_interface__msg__STM32 *)allocator.zero_allocate(size, sizeof(learning_interface__msg__STM32), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = learning_interface__msg__STM32__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        learning_interface__msg__STM32__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
learning_interface__msg__STM32__Sequence__fini(learning_interface__msg__STM32__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      learning_interface__msg__STM32__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

learning_interface__msg__STM32__Sequence *
learning_interface__msg__STM32__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  learning_interface__msg__STM32__Sequence * array = (learning_interface__msg__STM32__Sequence *)allocator.allocate(sizeof(learning_interface__msg__STM32__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = learning_interface__msg__STM32__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
learning_interface__msg__STM32__Sequence__destroy(learning_interface__msg__STM32__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    learning_interface__msg__STM32__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
learning_interface__msg__STM32__Sequence__are_equal(const learning_interface__msg__STM32__Sequence * lhs, const learning_interface__msg__STM32__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!learning_interface__msg__STM32__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
learning_interface__msg__STM32__Sequence__copy(
  const learning_interface__msg__STM32__Sequence * input,
  learning_interface__msg__STM32__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(learning_interface__msg__STM32);
    learning_interface__msg__STM32 * data =
      (learning_interface__msg__STM32 *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!learning_interface__msg__STM32__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          learning_interface__msg__STM32__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!learning_interface__msg__STM32__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
