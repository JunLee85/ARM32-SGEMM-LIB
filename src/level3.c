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

/* This file is a template for level 3 operation */
#include "macro.h"
#include "common.h"

#include "common_level3.h"

#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC) \
	GEMM_BETA((M_TO) - (M_FROM), (N_TO - N_FROM), 0, \
		  BETA[0], NULL, 0, NULL, 0, \
		  (FLOAT *)(C) + ((M_FROM) + (N_FROM) * (LDC)) * COMPSIZE, LDC)

#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
				sgemm_kernel(M, N, K, ALPHA[0], SA, SB, (FLOAT *)(C) + ((X) + (Y) * LDC) * COMPSIZE, LDC)

#define KERNEL_OPERATION_FIX12(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
				sgemm_kernel_fix12(M, N, K, ALPHA[0], SA, SB, (FLOAT *)(C) + ((X) + (Y) * LDC) * COMPSIZE, LDC)
#define KERNEL_OPERATION_FIX13(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
				sgemm_kernel_fix13(M, N, K, ALPHA[0], SA, SB, (FLOAT *)(C) + ((X) + (Y) * LDC) * COMPSIZE, LDC)
#define KERNEL_OPERATION_FIX14(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
				sgemm_kernel_fix14(M, N, K, ALPHA[0], SA, SB, (FLOAT *)(C) + ((X) + (Y) * LDC) * COMPSIZE, LDC)
#define KERNEL_OPERATION_FIX15(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
				sgemm_kernel_fix15(M, N, K, ALPHA[0], SA, SB, (FLOAT *)(C) + ((X) + (Y) * LDC) * COMPSIZE, LDC)

#ifndef A
#define A	args -> a
#endif
#ifndef LDA
#define LDA	args -> lda
#endif
#ifndef B
#define B	args -> b
#endif
#ifndef LDB
#define LDB	args -> ldb
#endif
#ifndef C
#define C	args -> c
#endif
#ifndef LDC
#define LDC	args -> ldc
#endif
#ifndef M
#define M	args -> m
#endif
#ifndef N
#define N	args -> n
#endif
#ifndef K
#define K	args -> k
#endif

#ifdef TIMING
#define START_RPCC()		rpcc_counter = rpcc()
#define STOP_RPCC(COUNTER)	COUNTER  += rpcc() - rpcc_counter
#else
#define START_RPCC()
#define STOP_RPCC(COUNTER)
#endif

int sgemm_nn(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
		  XFLOAT *sa, XFLOAT *sb, BLASLONG dummy, int fractions){
  BLASLONG k, lda, ldb, ldc;
  FLOAT *alpha, *beta;
  FLOAT *a, *b, *c;
  BLASLONG m_from, m_to, n_from, n_to;

  BLASLONG ls, is, js;
  BLASLONG min_l, min_i, min_j;
#if !defined(FUSED_GEMM) || defined(TIMING)
  BLASLONG jjs, min_jj;
#endif

  BLASLONG l1stride, gemm_p, l2size;

#ifdef TIMING
  unsigned long long rpcc_counter;
  unsigned long long innercost  = 0;
  unsigned long long outercost  = 0;
  unsigned long long kernelcost = 0;
  double total;
#endif
  k = K;

  a = (FLOAT *)A;
  b = (FLOAT *)B;
  c = (FLOAT *)C;

  lda = LDA;
  ldb = LDB;
  ldc = LDC;

  alpha = (FLOAT *)args -> alpha;
  beta  = (FLOAT *)args -> beta;

  m_from = 0;
  m_to   = M;

  if (range_m) {
    m_from = *(((BLASLONG *)range_m) + 0);
    m_to   = *(((BLASLONG *)range_m) + 1);
  }

  n_from = 0;
  n_to   = N;

  if (range_n) {
    n_from = *(((BLASLONG *)range_n) + 0);
    n_to   = *(((BLASLONG *)range_n) + 1);
  }

  if (beta) {
    if (beta[0] != ONE) {
	  BETA_OPERATION(m_from, m_to, n_from, n_to, beta, c, ldc);
	}
  }

	if ((k == 0) || (alpha == NULL))
		return 0;
	if (alpha[0] == ZERO)
		return 0;

  l2size = GEMM_P * GEMM_Q;

#ifdef TIMING
  innercost = 0;
  outercost = 0;
  kernelcost = 0;
#endif


  for(js = n_from; js < n_to; js += GEMM_R){
    min_j = n_to - js;
		if (min_j > GEMM_R)
			min_j = GEMM_R;

    for(ls = 0; ls < k; ls += min_l){

      min_l = k - ls;

      if (min_l >= GEMM_Q * 2) {
	gemm_p = GEMM_P;
	min_l  = GEMM_Q;
      } else {
	if (min_l > GEMM_Q) {
					min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M)
							* GEMM_UNROLL_M;
	}
				gemm_p = ((l2size / min_l + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M)
						* GEMM_UNROLL_M;
				while (gemm_p * min_l > l2size)
					gemm_p -= GEMM_UNROLL_M;
      }

      /* First, we have to move data A to L2 cache */
      min_i = m_to - m_from;
      l1stride = 1;

      if (min_i >= GEMM_P * 2) {
	min_i = GEMM_P;
      } else {
	if (min_i > GEMM_P) {
					min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M)
							* GEMM_UNROLL_M;
	} else {
	  l1stride = 0;
	}
      }

      START_RPCC();

		if (12 == fractions)
			sgemm_otcopy_fix12(min_l, min_i,
						(FLOAT *) (a) + ((m_from) + (ls) * (lda)) * COMPSIZE, lda,
						sa);
		else if (13 == fractions)
			sgemm_otcopy_fix13(min_l, min_i,
						(FLOAT *) (a) + ((m_from) + (ls) * (lda)) * COMPSIZE, lda,
						sa);
		else if (14 == fractions)
			sgemm_otcopy_fix14(min_l, min_i,
						(FLOAT *) (a) + ((m_from) + (ls) * (lda)) * COMPSIZE, lda,
						sa);
		else if (15 == fractions)
			sgemm_otcopy_fix15(min_l, min_i,
						(FLOAT *) (a) + ((m_from) + (ls) * (lda)) * COMPSIZE, lda,
						sa);
		else if (0 == fractions)
			sgemm_otcopy(min_l, min_i,
						(FLOAT *) (a) + ((m_from) + (ls) * (lda)) * COMPSIZE, lda,
						sa);
		else
			printf("fractions not support\n");

      STOP_RPCC(innercost);

      for(jjs = js; jjs < js + min_j; jjs += min_jj){
		min_jj = min_j + js - jjs;

				if (min_jj >= 3 * GEMM_UNROLL_N)
					min_jj = 3 * GEMM_UNROLL_N;
				else if (min_jj >= 2 * GEMM_UNROLL_N)
					min_jj = 2 * GEMM_UNROLL_N;
				else if (min_jj > GEMM_UNROLL_N)
					min_jj = GEMM_UNROLL_N;

		START_RPCC();

		if (12 == fractions)
				sgemm_oncopy_fix12(min_l, min_jj,
				(FLOAT *) (b) + ((ls) + (jjs) * (ldb)) * COMPSIZE, ldb,
				(XFLOAT *) ((short int*) sb
						+ min_l * (jjs - js) * COMPSIZE * l1stride));
		else if (13 == fractions)
				sgemm_oncopy_fix13(min_l, min_jj,
				(FLOAT *) (b) + ((ls) + (jjs) * (ldb)) * COMPSIZE, ldb,
				(XFLOAT *) ((short int*) sb
						+ min_l * (jjs - js) * COMPSIZE * l1stride));
		else if (14 == fractions)
				sgemm_oncopy_fix14(min_l, min_jj,
				(FLOAT *) (b) + ((ls) + (jjs) * (ldb)) * COMPSIZE, ldb,
				(XFLOAT *) ((short int*) sb
						+ min_l * (jjs - js) * COMPSIZE * l1stride));
		else if (15 == fractions)
				sgemm_oncopy_fix15(min_l, min_jj,
				(FLOAT *) (b) + ((ls) + (jjs) * (ldb)) * COMPSIZE, ldb,
				(XFLOAT *) ((short int*) sb
						+ min_l * (jjs - js) * COMPSIZE * l1stride));
		else if (0 == fractions)
			sgemm_oncopy(min_l, min_jj, (FLOAT *)(b) + ((ls) + (jjs) * (ldb)) * COMPSIZE, ldb, sb + min_l * (jjs - js) * COMPSIZE * l1stride);
		else
			printf("fractions not support\n");
		STOP_RPCC(outercost);

		START_RPCC();
		if (12 == fractions)
			KERNEL_OPERATION_FIX12(min_i, min_jj, min_l, alpha, sa,
									(XFLOAT *)((short int*)sb + min_l * (jjs - js) * COMPSIZE * l1stride),
									c, ldc, m_from, jjs);
		else if (13 ==fractions)
			KERNEL_OPERATION_FIX13(min_i, min_jj, min_l, alpha, sa,
									(XFLOAT *)((short int*)sb + min_l * (jjs - js) * COMPSIZE * l1stride),
									c, ldc, m_from, jjs);
		else if (14 ==fractions)
			KERNEL_OPERATION_FIX14(min_i, min_jj, min_l, alpha, sa,
									(XFLOAT *)((short int*)sb + min_l * (jjs - js) * COMPSIZE * l1stride),
									c, ldc, m_from, jjs);
		else if (15 ==fractions)
			KERNEL_OPERATION_FIX15(min_i, min_jj, min_l, alpha, sa,
									(XFLOAT *)((short int*)sb + min_l * (jjs - js) * COMPSIZE * l1stride),
									c, ldc, m_from, jjs);
		else if (0 ==fractions)
			KERNEL_OPERATION(min_i, min_jj, min_l, alpha,
					sa, sb + min_l * (jjs - js)  * COMPSIZE * l1stride, c, ldc, m_from, jjs);
		else
			printf("fractions not support\n");
		STOP_RPCC(kernelcost);
      }

      for(is = m_from + min_i; is < m_to; is += min_i){
	min_i = m_to - is;

	if (min_i >= GEMM_P * 2) {
	  min_i = GEMM_P;
				} else if (min_i > GEMM_P) {
					min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M)
							* GEMM_UNROLL_M;
	  }

	START_RPCC();

	if (12 == fractions)
			sgemm_otcopy_fix12(min_l, min_i,
					(FLOAT *) (a) + ((is) + (ls) * (lda)) * COMPSIZE, lda,
			sa);
	else if (13 == fractions)
			sgemm_otcopy_fix13(min_l, min_i,
					(FLOAT *) (a) + ((is) + (ls) * (lda)) * COMPSIZE, lda,
			sa);
	else if (14 == fractions)
			sgemm_otcopy_fix14(min_l, min_i,
					(FLOAT *) (a) + ((is) + (ls) * (lda)) * COMPSIZE, lda,
			sa);
	else if (15 == fractions)
			sgemm_otcopy_fix15(min_l, min_i,
					(FLOAT *) (a) + ((is) + (ls) * (lda)) * COMPSIZE, lda,
			sa);
	else if (0 ==fractions)
			sgemm_otcopy(min_l, min_i,
					(FLOAT *) (a) + ((is) + (ls) * (lda)) * COMPSIZE, lda,
					sa);
	else
		printf("fractions not support\n");

	STOP_RPCC(innercost);

	START_RPCC();
	if (12 == fractions)
		KERNEL_OPERATION_FIX12(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is,
			js);
	else if (13 ==fractions)
		KERNEL_OPERATION_FIX13(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is,
			js);
	else if (14 ==fractions)
		KERNEL_OPERATION_FIX14(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is,
			js);
	else if (15 ==fractions)
		KERNEL_OPERATION_FIX15(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is,
			js);
	else if (0 ==fractions)
		KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is,
				js);
	else
		printf("fractions not support\n");
	STOP_RPCC(kernelcost);
      } /* end of is */
    } /* end of js */
  } /* end of ls */


#ifdef TIMING
  total = (double)outercost + (double)innercost + (double)kernelcost;

  printf( "Copy A : %5.2f Copy  B: %5.2f  Kernel : %5.2f  kernel Effi. : %5.2f Total Effi. : %5.2f\n",
	   innercost / total * 100., outercost / total * 100.,
	  kernelcost / total * 100.,
	  (double)(m_to - m_from) * (double)(n_to - n_from) * (double)k / (double)kernelcost * 100. * (double)COMPSIZE / 2.,
	  (double)(m_to - m_from) * (double)(n_to - n_from) * (double)k / total * 100. * (double)COMPSIZE / 2.);

#endif

  return 0;
}
