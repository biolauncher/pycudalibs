#include <element_kernel.h>

/* elementwise scalar */
ELEMENT_KERNEL(sum, __fadd_rn)
ELEMENT_KERNEL(mul, __fmul_rn)

ELEMENT_KERNEL_IMPL(sum, float)
ELEMENT_KERNEL_IMPL(mul, float)

ARRAY_ELEMENTWISE(sum, float)
ARRAY_ELEMENTWISE(mul, float)

/* elementwise vec/array */
ELEMENT_V_KERNEL(sum, __fadd_rn)
ELEMENT_V_KERNEL(mul, __fmul_rn)

ELEMENT_V_KERNEL_IMPL(sum, float)
ELEMENT_V_KERNEL_IMPL(mul, float)

ARRAY_ELEMENTWISE_VEC(sum, float)
ARRAY_ELEMENTWISE_VEC(mul, float)

/* TODO unary elementwise */
 
