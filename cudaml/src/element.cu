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

/* column broadcast kernel */
ELEMENT_A_KERNEL(sum, __fadd_rn)
ELEMENT_A_KERNEL(mul, __fmul_rn)

ELEMENT_A_KERNEL_IMPL(sum, float)
ELEMENT_A_KERNEL_IMPL(mul, float)

ARRAY_ELEMENTWISE_ARY(sum, float)
ARRAY_ELEMENTWISE_ARY(mul, float)

/* TODO unary elementwise */
 
