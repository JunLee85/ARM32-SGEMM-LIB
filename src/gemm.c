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
#include <stdlib.h>
#include "macro.h"
#include "cblas.h"

#include "common.h"
#ifdef FUNCTION_PROFILE
#include "functable.h"
#endif

#ifndef COMPLEX
#ifdef XDOUBLE
#define ERROR_NAME "QGEMM "
#elif defined(DOUBLE)
#define ERROR_NAME "DGEMM "
#else
#define ERROR_NAME "SGEMM "
#endif
#else
#ifndef GEMM3M
#ifdef XDOUBLE
s
#define ERROR_NAME "XGEMM "
#elif defined(DOUBLE)
d
#define ERROR_NAME "ZGEMM "
#else
#define ERROR_NAME "CGEMM "
#endif
#else
#ifdef XDOUBLE
3
#define ERROR_NAME "XGEMM3M "
#elif defined(DOUBLE)
44
#define ERROR_NAME "ZGEMM3M "
#else
77
#define ERROR_NAME "CGEMM3M "
#endif
#endif
#endif

#ifndef GEMM_MULTITHREAD_THRESHOLD
#define GEMM_MULTITHREAD_THRESHOLD 4
#endif

void cblas_sgemm( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
									blasint m, blasint n, blasint k,
									FLOAT alpha,
									FLOAT *a, blasint lda,
									FLOAT *b, blasint ldb,
									FLOAT beta,
									FLOAT *c, blasint ldc,
									int fractions)
{

  blas_arg_t args;
  int transa, transb;
  blasint nrowa, nrowb, info;

  XFLOAT *buffer;
  XFLOAT *sa, *sb;

  int nthreads_max;
  int nthreads_avail;
  double MNK;

  int mode  =  BLAS_SINGLE  | BLAS_REAL;

  PRINT_DEBUG_CNAME;

  args.alpha = (void *)&alpha;
  args.beta  = (void *)&beta;

  transa = -1;
  transb = -1;
  info   =  0;

  if (order == CblasColMajor) {
    args.m = m;
    args.n = n;
    args.k = k;

    args.a = (void *)a;
    args.b = (void *)b;
    args.c = (void *)c;

    args.lda = lda;
    args.ldb = ldb;
    args.ldc = ldc;

    if (TransA == CblasNoTrans)     transa = 0;
    if (TransA == CblasTrans)       transa = 1;
    if (TransA == CblasConjNoTrans) transa = 0;
    if (TransA == CblasConjTrans)   transa = 1;
    if (TransB == CblasNoTrans)     transb = 0;
    if (TransB == CblasTrans)       transb = 1;
    if (TransB == CblasConjNoTrans) transb = 0;
    if (TransB == CblasConjTrans)   transb = 1;
    nrowa = args.m;
    if (transa & 1) nrowa = args.k;
    nrowb = args.k;
    if (transb & 1) nrowb = args.n;

    info = -1;

    if (args.ldc < args.m) info = 13;
    if (args.ldb < nrowb)  info = 10;
    if (args.lda < nrowa)  info =  8;
    if (args.k < 0)        info =  5;
    if (args.n < 0)        info =  4;
    if (args.m < 0)        info =  3;
    if (transb < 0)        info =  2;
    if (transa < 0)        info =  1;
  }

  if (order == CblasRowMajor) {
	  
    args.m = n;
    args.n = m;
    args.k = k;

    args.a = (void *)b;
    args.b = (void *)a;
    args.c = (void *)c;

    args.lda = ldb;
    args.ldb = lda;
    args.ldc = ldc;

    if (TransB == CblasNoTrans)     transa = 0;
    if (TransB == CblasTrans)       transa = 1;
    if (TransB == CblasConjNoTrans) transa = 0;
    if (TransB == CblasConjTrans)   transa = 1;
    if (TransA == CblasNoTrans)     transb = 0;
    if (TransA == CblasTrans)       transb = 1;
    if (TransA == CblasConjNoTrans) transb = 0;
    if (TransA == CblasConjTrans)   transb = 1;

    nrowa = args.m;
    if (transa & 1) nrowa = args.k;
    nrowb = args.k;
    if (transb & 1) nrowb = args.n;

    info = -1;

    if (args.ldc < args.m) info = 13;
    if (args.ldb < nrowb)  info = 10;
    if (args.lda < nrowa)  info =  8;
    if (args.k < 0)        info =  5;
    if (args.n < 0)        info =  4;
    if (args.m < 0)        info =  3;
    if (transb < 0)        info =  2;
    if (transa < 0)        info =  1;

  }

  if (info >= 0) return;
  if ((args.m == 0) || (args.n == 0)) return;

  IDEBUG_START;

  FUNCTION_PROFILE_START();  

  buffer = (XFLOAT *)blas_memory_alloc(0);

  sa = (XFLOAT *)((BLASLONG)buffer +GEMM_OFFSET_A);
  sb = (XFLOAT *)(((BLASLONG)sa + ((GEMM_P * GEMM_Q * COMPSIZE * SIZE + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);

  mode |= (transa << BLAS_TRANSA_SHIFT);
  mode |= (transb << BLAS_TRANSB_SHIFT);

  nthreads_max = num_cpu_avail(3);
  nthreads_avail = nthreads_max;

  MNK = (double) args.m * (double) args.n * (double) args.k;
  if (MNK <= (65536.0  * (double) GEMM_MULTITHREAD_THRESHOLD)) nthreads_max = 1;

  args.common = NULL;

  if ( nthreads_max > nthreads_avail )
  	args.nthreads = nthreads_avail;
  else
  	args.nthreads = nthreads_max;

  if (args.nthreads == 1)
    sgemm_nn(&args, NULL, NULL, sa, sb, 0, fractions);
  else 
    sgemm_thread_nn(&args, NULL, NULL, sa, sb, 0, fractions);

	blas_memory_free(buffer);

  FUNCTION_PROFILE_END(COMPSIZE * COMPSIZE, args.m * args.k + args.k * args.n + args.m * args.n, 2 * args.m * args.n * args.k);

  IDEBUG_END;

  return;
}
