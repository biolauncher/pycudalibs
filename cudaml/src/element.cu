#include <element_kernel.h>

/* elementwise binary scalar */
ELEMENT_KERNEL(sum, __fadd_rn)
ELEMENT_KERNEL(mul, __fmul_rn)
ELEMENT_KERNEL(pow, powf)

ELEMENT_KERNEL_IMPL(sum, float)
ELEMENT_KERNEL_IMPL(mul, float)
ELEMENT_KERNEL_IMPL(pow, float)

ARRAY_ELEMENTWISE(sum, float)
ARRAY_ELEMENTWISE(mul, float)
ARRAY_ELEMENTWISE(pow, float)

/* elementwise binary vec/array */
ELEMENT_V_KERNEL(sum, __fadd_rn)
ELEMENT_V_KERNEL(mul, __fmul_rn)
ELEMENT_V_KERNEL(pow, powf)

ELEMENT_V_KERNEL_IMPL(sum, float)
ELEMENT_V_KERNEL_IMPL(mul, float)
ELEMENT_V_KERNEL_IMPL(pow, float)

ARRAY_ELEMENTWISE_VEC(sum, float)
ARRAY_ELEMENTWISE_VEC(mul, float)
ARRAY_ELEMENTWISE_VEC(pow, float)

/* column binary broadcast kernel */
ELEMENT_A_KERNEL(sum, __fadd_rn)
ELEMENT_A_KERNEL(mul, __fmul_rn)
ELEMENT_A_KERNEL(pow, powf)

ELEMENT_A_KERNEL_IMPL(sum, float)
ELEMENT_A_KERNEL_IMPL(mul, float)
ELEMENT_A_KERNEL_IMPL(pow, float)

ARRAY_ELEMENTWISE_ARY(sum, float)
ARRAY_ELEMENTWISE_ARY(mul, float)
ARRAY_ELEMENTWISE_ARY(pow, float)

/* unary elementwise */
ELEMENT_U_KERNEL(sqrt, __fsqrt_rn)
ELEMENT_U_KERNEL(log, __logf)
ELEMENT_U_KERNEL(log2, __log2f)
ELEMENT_U_KERNEL(log10, __log10f)

ELEMENT_U_KERNEL(sin, __sinf)
ELEMENT_U_KERNEL(cos, __cosf)
ELEMENT_U_KERNEL(tan, __tanf)

ELEMENT_U_KERNEL(sinh, sinhf)
ELEMENT_U_KERNEL(cosh, coshf)
ELEMENT_U_KERNEL(tanh, tanhf)

ELEMENT_U_KERNEL(exp, expf)
ELEMENT_U_KERNEL(exp10, exp10f)

ELEMENT_U_KERNEL(sinpi, sinpif)
ELEMENT_U_KERNEL(cospi, cospif)

ELEMENT_U_KERNEL(asin, asinf)
ELEMENT_U_KERNEL(acos, acosf)
ELEMENT_U_KERNEL(atan, atanf)
ELEMENT_U_KERNEL(asinh, asinhf)
ELEMENT_U_KERNEL(acosh, acoshf)
ELEMENT_U_KERNEL(atanh, atanhf)

ELEMENT_U_KERNEL(erf, erff)
ELEMENT_U_KERNEL(erfc, erfcf)
ELEMENT_U_KERNEL(erfinv, erfinvf)
ELEMENT_U_KERNEL(erfcinv, erfcinvf)
ELEMENT_U_KERNEL(lgamma, lgammaf)
ELEMENT_U_KERNEL(tgamma, tgammaf)

ELEMENT_U_KERNEL(trunc, truncf)
ELEMENT_U_KERNEL(round, roundf)
ELEMENT_U_KERNEL(rint, rintf)
ELEMENT_U_KERNEL(floor, floorf)
ELEMENT_U_KERNEL(ceil, ceilf)

// and many more to come?

ELEMENT_U_KERNEL_IMPL(sqrt, float)
ELEMENT_U_KERNEL_IMPL(log, float)
ELEMENT_U_KERNEL_IMPL(log2, float)
ELEMENT_U_KERNEL_IMPL(log10, float)

ELEMENT_U_KERNEL_IMPL(sin, float)
ELEMENT_U_KERNEL_IMPL(cos, float)
ELEMENT_U_KERNEL_IMPL(tan, float)

ELEMENT_U_KERNEL_IMPL(sinh, float)
ELEMENT_U_KERNEL_IMPL(cosh, float)
ELEMENT_U_KERNEL_IMPL(tanh, float)

ELEMENT_U_KERNEL_IMPL(exp, float)
ELEMENT_U_KERNEL_IMPL(exp10, float)

ELEMENT_U_KERNEL_IMPL(sinpi, float)
ELEMENT_U_KERNEL_IMPL(cospi, float)

ELEMENT_U_KERNEL_IMPL(asin, float)
ELEMENT_U_KERNEL_IMPL(acos, float)
ELEMENT_U_KERNEL_IMPL(atan, float)
ELEMENT_U_KERNEL_IMPL(asinh, float)
ELEMENT_U_KERNEL_IMPL(acosh, float)
ELEMENT_U_KERNEL_IMPL(atanh, float)

ELEMENT_U_KERNEL_IMPL(erf, float)
ELEMENT_U_KERNEL_IMPL(erfc, float)
ELEMENT_U_KERNEL_IMPL(erfinv, float)
ELEMENT_U_KERNEL_IMPL(erfcinv, float)
ELEMENT_U_KERNEL_IMPL(lgamma, float)
ELEMENT_U_KERNEL_IMPL(tgamma, float)

ELEMENT_U_KERNEL_IMPL(trunc, float)
ELEMENT_U_KERNEL_IMPL(round, float)
ELEMENT_U_KERNEL_IMPL(rint, float)
ELEMENT_U_KERNEL_IMPL(floor, float)
ELEMENT_U_KERNEL_IMPL(ceil, float)

//

ARRAY_ELEMENTWISE_UNARY(sqrt, float)
ARRAY_ELEMENTWISE_UNARY(log, float)
ARRAY_ELEMENTWISE_UNARY(log2, float)
ARRAY_ELEMENTWISE_UNARY(log10, float)

ARRAY_ELEMENTWISE_UNARY(sin, float)
ARRAY_ELEMENTWISE_UNARY(cos, float)
ARRAY_ELEMENTWISE_UNARY(tan, float)

ARRAY_ELEMENTWISE_UNARY(sinh, float)
ARRAY_ELEMENTWISE_UNARY(cosh, float)
ARRAY_ELEMENTWISE_UNARY(tanh, float)

ARRAY_ELEMENTWISE_UNARY(exp, float)
ARRAY_ELEMENTWISE_UNARY(exp10, float)

ARRAY_ELEMENTWISE_UNARY(sinpi, float)
ARRAY_ELEMENTWISE_UNARY(cospi, float)

ARRAY_ELEMENTWISE_UNARY(asin, float)
ARRAY_ELEMENTWISE_UNARY(acos, float)
ARRAY_ELEMENTWISE_UNARY(atan, float)
ARRAY_ELEMENTWISE_UNARY(asinh, float)
ARRAY_ELEMENTWISE_UNARY(acosh, float)
ARRAY_ELEMENTWISE_UNARY(atanh, float)

ARRAY_ELEMENTWISE_UNARY(erf, float)
ARRAY_ELEMENTWISE_UNARY(erfc, float)
ARRAY_ELEMENTWISE_UNARY(erfinv, float)
ARRAY_ELEMENTWISE_UNARY(erfcinv, float)
ARRAY_ELEMENTWISE_UNARY(lgamma, float)
ARRAY_ELEMENTWISE_UNARY(tgamma, float)

ARRAY_ELEMENTWISE_UNARY(trunc, float)
ARRAY_ELEMENTWISE_UNARY(round, float)
ARRAY_ELEMENTWISE_UNARY(rint, float)
ARRAY_ELEMENTWISE_UNARY(floor, float)
ARRAY_ELEMENTWISE_UNARY(ceil, float)

