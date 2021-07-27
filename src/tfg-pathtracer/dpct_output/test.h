#pragma once

#pragma push_macro("__DEVICE__")
#define __DEVICE__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define __DEVICE__                                                             \
static __inline__ __attribute__((always_inline)) __attribute__((device))

__DEVICE__ double log1p(double);

__cudart_builtin__ constexpr double log1p(double);