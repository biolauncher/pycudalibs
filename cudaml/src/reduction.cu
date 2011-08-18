#include <reduction_kernel.h>

// N.B. API prototypes need to be declared also in cudaml.h 

/* generate reduction kernel for operator */
REDUCTION_KERNEL(sum, __fadd_rn)
REDUCTION_KERNEL(max, max)
REDUCTION_KERNEL(min, min)
REDUCTION_KERNEL(product, __fmul_rn)

/* generate type generic lancher template */
REDUCTION_KERNEL_LAUNCHER(sum)
REDUCTION_KERNEL_LAUNCHER(max)
REDUCTION_KERNEL_LAUNCHER(min)
REDUCTION_KERNEL_LAUNCHER(product)

/* generate type specific kernel launchers */
ARRAY_REDUCTION_KERNEL_LAUNCHER(sum, float)
ARRAY_REDUCTION_KERNEL_LAUNCHER(max, float)
ARRAY_REDUCTION_KERNEL_LAUNCHER(min, float)
ARRAY_REDUCTION_KERNEL_LAUNCHER(product, float)

/* generate type specific api fn */
ARRAY_REDUCE(sum, float)
ARRAY_REDUCE(max, float)
ARRAY_REDUCE(min, float)
ARRAY_REDUCE(product, float)
