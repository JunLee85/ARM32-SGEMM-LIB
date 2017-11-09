/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include "macro.h"
#include "cblas.h"

#include "common.h"
#ifdef FUNCTION_PROFILE
#include "functable.h"
#endif

#ifdef XDOUBLE
#define ERROR_NAME "QGEMV "
#elif defined(DOUBLE)
#define ERROR_NAME "DGEMV "
#else
#define ERROR_NAME "SGEMV "
#endif

#if 1
#ifdef SMP
static int (*gemv_thread[])(BLASLONG, BLASLONG, FLOAT, FLOAT *, BLASLONG,  FLOAT * , BLASLONG, FLOAT *, BLASLONG, FLOAT *, int) = {
#ifdef XDOUBLE
  qgemv_thread_n, qgemv_thread_t,
#elif defined DOUBLE
  dgemv_thread_n, dgemv_thread_t,
#else
  sgemv_thread_n, sgemv_thread_t,
#endif
};
#endif
#endif


void cblas_sgemv(enum CBLAS_ORDER order,
	   enum CBLAS_TRANSPOSE TransA,
	   blasint m, blasint n,
	   FLOAT alpha,
	   FLOAT  *a, blasint lda,
	   FLOAT  *x, blasint incx,
	   FLOAT beta,
	   FLOAT  *y, blasint incy){
  FLOAT *buffer;
  blasint lenx, leny;
  int trans, buffer_size;
  blasint info, t;
#ifdef SMP
  int nthreads;
#endif

  PRINT_DEBUG_CNAME;

  trans = -1;
  info  =  0;

  if (order == CblasColMajor) {
    if (TransA == CblasNoTrans)     trans = 0;
    if (TransA == CblasTrans)       trans = 1;
    if (TransA == CblasConjNoTrans) trans = 0;
    if (TransA == CblasConjTrans)   trans = 1;

    info = -1;

    if (incy == 0)	  info = 11;
    if (incx == 0)	  info = 8;
    if (lda < MAX(1, m))  info = 6;
    if (n < 0)		  info = 3;
    if (m < 0)		  info = 2;
    if (trans < 0)        info = 1;

  }

  if (order == CblasRowMajor) {
    if (TransA == CblasNoTrans)     trans = 1;
    if (TransA == CblasTrans)       trans = 0;
    if (TransA == CblasConjNoTrans) trans = 1;
    if (TransA == CblasConjTrans)   trans = 0;

    info = -1;

    t = n;
    n = m;
    m = t;

    if (incy == 0)	  info = 11;
    if (incx == 0)	  info = 8;
    if (lda < MAX(1, m))  info = 6;
    if (n < 0)		  info = 3;
    if (m < 0)		  info = 2;
    if (trans < 0)        info = 1;

  }

  if (info >= 0) {
    return;
  }

  if ((m==0) || (n==0)) return;

  lenx = n;
  leny = m;
  if (trans) lenx = m;
  if (trans) leny = n;

  if (beta != ONE) sscal_k(leny, 0, 0, beta, y, abs(incy), NULL, 0, NULL, 0);

  if (alpha == ZERO) return;

  IDEBUG_START;

  FUNCTION_PROFILE_START();

  if (incx < 0) x -= (lenx - 1) * incx;
  if (incy < 0) y -= (leny - 1) * incy;

  buffer_size = m + n + 128 / sizeof(FLOAT);
#ifdef WINDOWS_ABI
  buffer_size += 160 / sizeof(FLOAT) ;
#endif
  // for alignment
  buffer_size = (buffer_size + 3) & ~3;
  STACK_ALLOC(buffer_size, FLOAT, buffer);

#ifdef SMP

  if ( 1L * m * n < 2304L * GEMM_MULTITHREAD_THRESHOLD )
    nthreads = 1;
  else
    nthreads = num_cpu_avail(2);

  if (nthreads == 1) {
#endif
	  if (trans == 1)
		  sgemv_t(m, n, 0, alpha, a, lda, x, incx, y, incy, buffer); //gemv_t_vfp.S
	  else
		  sgemv_n(m, n, 0, alpha, a, lda, x, incx, y, incy, buffer);//gemv_n_vfp3.S
#ifdef SMP
  } else {
   (gemv_thread[(int)trans])(m, n, alpha, a, lda, x, incx, y, incy, buffer, nthreads);//gemv_thread.c
  }
#endif

  STACK_FREE(buffer);
  FUNCTION_PROFILE_END(1, m * n + m + n,  2 * m * n);

  IDEBUG_END;

  return;

}
