/**
 * Bindings to CULA LAPACK routines
 *  defines module: _cula 
 */

#define NO_IMPORT_ARRAY */
#include <pycula.h>
#include <pycuarray.h>


/*
 gesvd - generalized singular value decomposition
 A = U.S.VT
*/

static PyObject* gesvd(PyObject* self, PyObject* args) {
  cuda_Array *A, *S, *U, *VT;
  char jobu, jobvt;

  if (PyArg_ParseTuple(args, "ccO!O!O!O!", 
                       &jobu, &jobvt, 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &S,
                       cuda_ArrayType, &U,
                       cuda_ArrayType, &VT)) {

    // what if A is transposed? then is lda affected? need to check this
    // and also I may need to tweak the memory pitch in the transpose code

    int m = A->a_dims[0];
    int n = A->a_dims[1];

    // leading dimensions 
    int lda = A->a_dims[0];

    // TODO make sure output arrays are of required sizes

    int ldu = U->a_dims[0];
    int ldvt = VT->a_dims[0];

    /*
      void culaSgesvd

      char jobu -  'A' implies all m columns of U are returned in U, 
                   'S' the first min(m,n) columns of U are returned in U,
                   'O' the first min(m.n) columns of U are overwritten in A,
                   'N' no columns of U (left singular vectors) are computed.
      char jobvt - 'A' all N rows of VT are returned in VT,
                   'S' the first min(m,n) rows of VT are returned in VT,
                   'O' the first min(m,n) rows of VT are overwritten in A,
                   'N' no rows of VT (right singular vectors) are computed.
      int m - number of rows in A
      int n - number of columns in A

      A - m*n matrix to factorize into U S VT
      int lda - leading dimension of A (>= max(1,m))
      S - singular values
      U - left singular vectors
      int ldu - leading dimension of U 
      VT - right singular vectors
      int ldvt - leading dimension of VT
    */

    //culaDeviceSgesvd(jobu, jobvt, m, n, 
    //                 A->d_mem->d_ptr, lda, S->d_mem->d_ptr,
    //                 U->d_mem->d_ptr, ldu,
    //                 VT->d_mem->d_ptr, ldvt);

    if (culablas_error("sgesvd")) 
      return NULL;
    else 
      // build a tuple of arrays
      return Py_BuildValue("OOO", S, U, VT);
  
  } else {
    return NULL;
  }
}

