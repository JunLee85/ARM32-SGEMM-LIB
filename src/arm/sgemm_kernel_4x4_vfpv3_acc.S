/***************************************************************************
Copyright (c) 2013, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

/**************************************************************************************
* 2013/11/23 Saar
* 	 BLASTEST 		: OK
* 	 CTEST			: OK
* 	 TEST			: OK
*
*
* 2013/11/02 Saar
*	UNROLL_N		4
*	UNROLL_M		4
*	DGEMM_P			128
*	DGEMM_Q			240
*	DGEMM_R			12288
*	A_PRE			128
*	B_PRE			128
*	C_PRE			32
*
* Performance on Odroid U2:
*
* 3072x3072		1 Core:		2.62 GFLOPS	ATLAS: 2.69	GFLOPS
* 3072x3072		2 Cores:	5.23 GFLOPS	ATLAS: 5.27	GFLOPS
* 3072x3072		3 Cores:	7.78 GFLOPS	ATLAS: 7.87	GFLOPS
* 3072x3072		4 Cores:       10.10 GFLOPS	ATLAS: 9.98	GFLOPS
**************************************************************************************/

#define ASSEMBLER
#include "common.h"

#define STACKSIZE 256

#define	OLD_M	r0
#define	OLD_N	r1
#define	OLD_K	r2

#ifdef ARM_SOFTFP_ABI
#define OLD_ALPHA r3
//#define OLD_A	
#else //hard
#define	OLD_A	r3
#define OLD_ALPHA s0
#endif

/******************************************************
* [fp, #-128] - [fp, #-64] is reserved
* for store and restore of floating point
* registers
*******************************************************/

#define LDC	[fp, #-252 ]
#define M	[fp, #-256 ]
#define N	[fp, #-260 ]
#define K	[fp, #-264 ]

#ifndef ARM_SOFTFP_ABI
#define A	[fp, #-268 ]
#endif

#define BKCKUP_R9 [fp, #-244]

#define FP_ZERO [fp, #-240]
#define FP_ZERO_0 [fp, #-240]
#define FP_ZERO_1 [fp, #-236]

#define ALPHA   [fp, #-280]

#ifdef ARM_SOFTFP_ABI
#define A	[fp, #4 ]
#define B	[fp, #8 ]
#define C	[fp, #12 ]
#define OLD_LDC	[fp, #16 ]
#else //hard
#define B	[fp, #4 ]
#define C	[fp, #8 ]
#define OLD_LDC	[fp, #12 ]
#endif

#define I	r0
#define J	r1
#define L	r2

#define	AO	r5
#define	BO	r6

#define	CO1	r8
#define	CO2	r9

#define K1	r7
#define BC	r12

#define A_PRE	128
#define B_PRE	128
#define C_PRE	32

/**************************************************************************************
* Macro definitions
**************************************************************************************/

.macro INIT4x4

	flds			s16, FP_ZERO
	vmov.f32		s17, s16
	vmov.f32		s18, s16
	vmov.f32		s19, s16
	vmov.f32		s20, s16
	vmov.f32		s21, s16
	vmov.f32		s22, s16
	vmov.f32		s23, s16
	vmov.f32		s24, s16
	vmov.f32		s25, s16
	vmov.f32		s26, s16
	vmov.f32		s27, s16
	vmov.f32		s28, s16
	vmov.f32		s29, s16
	vmov.f32		s30, s16
	vmov.f32		s31, s16

.endm

.macro LOAD_4x4MATS
	pld	[ AO , #8 ]
	# load a 4x4 matrix from src1
	vld1.32         { q8-q9 }, [AO]!
	pld	[ BO , #8 ]
	vld1.32         {q0-q1}, [BO]!

	vld1.32         {q10-q11}, [AO]!
	vld1.32         {q2-q3}, [BO]!
.endm

.macro STORE_4x4MATS
	# store two 4x4 matrices to dst
	vst1.32         { q12-q13 }, [C0]!
	vst1.32         { q14-q15 }, [C0]!
.endm

.macro KERNEL4x4_NEON_I
	pld	[ AO , #A_PRE ]
	vld1.32 {q0}, [AO]!
	
	pld	[ BO , #B_PRE ]
	vld1.32 {q2}, [BO]!

	vmla.f32   q4,  q0,  d4[0]

	vld1.32 {q1}, [AO]!
	vmla.f32   q5,  q0,  d4[1]

	vld1.32 {q3}, [BO]!
	vmla.f32   q6,  q0,  d5[0]
	vmla.f32   q7,  q0,  d5[1]
.endm

.macro KERNEL4x4_I
	pld	[ AO , #A_PRE ]
	vld1.32 {q0}, [AO]!
	
	pld	[ BO , #B_PRE ]
	vld1.32 {q2}, [BO]!

	vmul.f32   q4,  q0,  d4[0]

	vld1.32 {q1}, [AO]!
	vmul.f32   q5,  q0,  d4[1]

	vld1.32 {q3}, [BO]!
	vmul.f32   q6,  q0,  d5[0]
	vmul.f32   q7,  q0,  d5[1]
.endm

.macro KERNEL4x4_M2
	pld	[ AO , #A_PRE ]
	vmla.f32   q4,  q1,  d6[0]

	fldmias AO!, { s0 - s3 }
	pld	[ BO , #B_PRE ]

	vmla.f32   q5,  q1,  d6[1]
	fldmias BO!, { s8 - s11 }
	vmla.f32   q6,  q1,  d7[0]
	vmla.f32   q7,  q1,  d7[1]
.endm

.macro KERNEL4x4_M1
	fldmias AO!, { s4 - s7 }
	vmla.f32   q4,  q0,  d4[0]
	vmla.f32   q5,  q0,  d4[1]
	fldmias BO!, { s12 - s15 }
	vmla.f32   q6,  q0,  d5[0]
	vmla.f32   q7,  q0,  d5[1]
.endm

.macro KERNEL4x4_E
	vmla.f32   q4,  q1,  d6[0]
	vmla.f32   q5,  q1,  d6[1]

	vmla.f32   q6,  q1,  d7[0]
	vmla.f32   q7,  q1,  d7[1]
.endm

.macro KERNEL4x4_SUB
	fldmias AO!, { s0 - s3 }	
	fldmias BO!, { s8 - s11 }

	vmla.f32   q4,  q0,  d4[0]
	vmla.f32   q5,  q0,  d4[1]
	vmla.f32   q6,  q0,  d5[0]
	vmla.f32   q7,  q0,  d5[1]
.endm

.macro SAVE4x4
	ldr	 r3  , LDC
	add	 CO2 , CO1, r3
	flds s0, ALPHA
	add	 r4  , CO2, r3

	@load data form C into Q registers
	vld1.32    { q2 }, [CO1]
	@ q2 += alpha * Q4
	vmla.f32   q2,  q4, d0[0]

	vld1.32    { q3 }, [CO2]
	vmla.f32   q3,  q5, d0[0]

	add		   CO2 , CO1, r3
	vst1.32    { q2 }, [CO1]
	vst1.32    { q3 }, [CO2]

	add		   CO2, r4 , r3
	vld1.32    { q2 }, [r4]
	vmla.f32   q2,  q6, d0[0]

	vld1.32    { q3 }, [CO2]
	vmla.f32   q3,  q7, d0[0]

	vst1.32    { q2 }, [r4]
	vst1.32    { q3 }, [CO2]

	add	CO1, CO1, #16
.endm

.macro SAVE4x4_OLD
    // C = alpha * C
	ldr	r3  , LDC
	add	CO2 , CO1, r3
	flds		s0, ALPHA
	add	r4  , CO2, r3

	fldmias CO1, { s8 - s11 }

	fmacs	s8 , s0 , s16
	flds	s12, [CO2]
	fmacs	s9 , s0 , s17
	flds	s13, [CO2, #4 ]
	fmacs	s10, s0 , s18
	flds	s14, [CO2, #8 ]
	fmacs	s11, s0 , s19
	flds	s15, [CO2, #12 ]

	fmacs	s12, s0 , s20
	fsts	s8 , [CO1]
	fmacs	s13, s0 , s21
	fsts	s9 , [CO1, #4 ]
	fmacs	s14, s0 , s22
	fsts	s10, [CO1, #8 ]
	fmacs	s15, s0 , s23
	fsts	s11, [CO1, #12 ]

	pld	[ CO1 , #C_PRE ]

	fldmias r4, { s8 - s11 }

	fmacs	s8 , s0 , s24
	fsts	s12, [CO2]
	fmacs	s9 , s0 , s25
	fsts	s13, [CO2, #4 ]
	fmacs	s10, s0 , s26
	fsts	s14, [CO2, #8 ]
	fmacs	s11, s0 , s27
	fsts	s15, [CO2, #12 ]

	pld	[ CO2 , #C_PRE ]

	add	CO2, r4 , r3


	fldmias CO2, { s12 - s15 }

	fsts	s8 , [r4 ]
	fmacs	s12, s0 , s28
	fsts	s9 , [r4 , #4 ]
	fmacs	s13, s0 , s29
	fsts	s10, [r4 , #8 ]
	fmacs	s14, s0 , s30
	fsts	s11, [r4 , #12 ]
	fmacs	s15, s0 , s31

	pld	[ r4 , #C_PRE ]
	fstmias CO2, { s12 - s15 }
	pld	[ CO2 , #C_PRE ]

	add	CO1, CO1, #16

.endm

/******************************************************************************/

.macro INIT2x4

	flds			s16, FP_ZERO
	vmov.f32		s17, s16
	vmov.f32		s20, s16
	vmov.f32		s21, s16
	vmov.f32		s24, s16
	vmov.f32		s25, s16
	vmov.f32		s28, s16
	vmov.f32		s29, s16

.endm

.macro KERNEL2x4_SUB
	fldmias BO!, { s8 - s11 }
	fldmias AO!, { s0 - s1 }

	fmacs	s16  , s0,  s8
	fmacs	s17  , s1,  s8

	fmacs	s20  , s0,  s9
	fmacs	s21  , s1,  s9

	fmacs	s24  , s0,  s10
	fmacs	s25  , s1,  s10

	fmacs	s28  , s0,  s11
	fmacs	s29  , s1,  s11
.endm

.macro SAVE2x4

	ldr	r3  , LDC
	add	CO2 , CO1, r3
	add	r4  , CO2, r3

	flds		s0, ALPHA

	flds	s8 , [CO1]
	flds	s9 , [CO1, #4 ]

	fmacs	s8 , s0 , s16
	fmacs	s9 , s0 , s17

	fsts	s8 , [CO1]
	fsts	s9 , [CO1, #4 ]

	flds	s12, [CO2]
	flds	s13, [CO2, #4 ]

	fmacs	s12, s0 , s20
	fmacs	s13, s0 , s21

	fsts	s12, [CO2]
	fsts	s13, [CO2, #4 ]

	flds	s8 , [r4 ]
	flds	s9 , [r4 , #4 ]

	fmacs	s8 , s0 , s24
	fmacs	s9 , s0 , s25

	fsts	s8 , [r4 ]
	fsts	s9 , [r4 , #4 ]

	add	CO2, r4 , r3

	flds	s12, [CO2]
	flds	s13, [CO2, #4 ]

	fmacs	s12, s0 , s28
	fmacs	s13, s0 , s29

	fsts	s12, [CO2]
	fsts	s13, [CO2, #4 ]

	add	CO1, CO1, #8

.endm


/******************************************************************************/

.macro INIT1x4

	flds			s16, FP_ZERO
	vmov.f32		s20, s16
	vmov.f32		s24, s16
	vmov.f32		s28, s16
.endm

.macro KERNEL1x4_SUB
	fldmias BO!, { s8 - s11 }
	flds	s0 , [ AO ]

	fmacs	s16  , s0,  s8
	fmacs	s20  , s0,  s9
	fmacs	s24  , s0,  s10
	fmacs	s28  , s0,  s11

	add	AO , AO, #4
.endm

.macro SAVE1x4

	ldr	r3  , LDC
	add	CO2 , CO1, r3
	add	r4  , CO2, r3

	flds		s0, ALPHA

	flds	s8 , [CO1]
	fmacs	s8 , s0 , s16
	fsts	s8 , [CO1]

	flds	s12, [CO2]
	fmacs	s12, s0 , s20
	fsts	s12, [CO2]

	flds	s8 , [r4 ]
	fmacs	s8 , s0 , s24
	fsts	s8 , [r4 ]

	add	CO2, r4 , r3

	flds	s12, [CO2]
	fmacs	s12, s0 , s28
	fsts	s12, [CO2]

	add	CO1, CO1, #4

.endm

/******************************************************************************/
/******************************************************************************/

.macro INIT4x2

	flds			s16, FP_ZERO
	vmov.f32		s17, s16
	vmov.f32		s18, s16
	vmov.f32		s19, s16
	vmov.f32		s20, s16
	vmov.f32		s21, s16
	vmov.f32		s22, s16
	vmov.f32		s23, s16

.endm

.macro KERNEL4x2_SUB
	fldmias BO!, { s8 - s9 }
	fldmias AO!, { s0 - s3 }

	vmla.f32   q4,  q0, d4[0]
	vmla.f32   q5,  q0, d4[1]
.endm

.macro SAVE4x2
	ldr	r3  , LDC
	add	CO2 , CO1, r3

	flds		s0, ALPHA
	
	fldmias CO1, { s8 - s11 }
	vmla.f32   q2,  q4, d0[0]

	fstmias CO1!, { s8 - s11 }

	fldmias CO2, { s12 - s15 }
	vmla.f32   q3,  q5, d0[0]

	fstmias CO2, { s12 - s15 }
.endm


/******************************************************************************/

.macro INIT2x2

	flds			s16, FP_ZERO
	vmov.f32		s17, s16
	vmov.f32		s20, s16
	vmov.f32		s21, s16

.endm



.macro KERNEL2x2_SUB
	fldmias BO!, { s8 - s9 }
	fldmias AO!, { s0 - s1 }

	fmacs	s16  , s0,  s8
	fmacs	s17  , s1,  s8

	fmacs	s20  , s0,  s9
	fmacs	s21  , s1,  s9
.endm

.macro SAVE2x2
	ldr	r3  , LDC
	add	CO2 , CO1, r3

	flds		s0, ALPHA

	flds	s8 , [CO1]
	flds	s9 , [CO1, #4 ]

	fmacs	s8 , s0 , s16
	fmacs	s9 , s0 , s17

	fsts	s8 , [CO1]
	fsts	s9 , [CO1, #4 ]

	flds	s12, [CO2]
	flds	s13, [CO2, #4 ]

	fmacs	s12, s0 , s20
	fmacs	s13, s0 , s21

	fsts	s12, [CO2]
	fsts	s13, [CO2, #4 ]

	add	CO1, CO1, #8

.endm

/******************************************************************************/

.macro INIT1x2

	flds			s16, FP_ZERO
	vmov.f32		s20, s16

.endm



.macro KERNEL1x2_SUB

	flds	s8 , [ BO ]
	flds	s9 , [ BO, #4 ]

	flds	s0 , [ AO ]
	fmacs	s16  , s0,  s8
	fmacs	s20  , s0,  s9

	add	AO , AO, #4
	add	BO , BO, #8

.endm

.macro SAVE1x2

	ldr	r3  , LDC
	add	CO2 , CO1, r3

	flds		s0, ALPHA

	flds	s8 , [CO1]
	fmacs	s8 , s0 , s16
	fsts	s8 , [CO1]

	flds	s12, [CO2]
	fmacs	s12, s0 , s20
	fsts	s12, [CO2]

	add	CO1, CO1, #4

.endm

/******************************************************************************/
/******************************************************************************/

.macro INIT4x1

	flds			s16, FP_ZERO
	vmov.f32		s17, s16
	vmov.f32		s18, s16
	vmov.f32		s19, s16

.endm

.macro KERNEL4x1_SUB
	flds	s8 , [ BO ]

	fldmias AO!, { s0 - s3 }

	vmla.f32   q4,  q0, d4[0]

	add	BO , BO, #4
.endm

.macro SAVE4x1
	flds		s0, ALPHA

	fldmias CO1, { s8 - s11 }
	vmla.f32   q2,  q4, d0[0]

	fstmias CO1!, { s8 - s11 }
.endm

/******************************************************************************/

.macro INIT2x1

	flds			s16, FP_ZERO
	vmov.f32		s17, s16

.endm



.macro KERNEL2x1_SUB

	flds	s8 , [ BO ]

	flds	s0 , [ AO ]
	flds	s1 , [ AO, #4 ]

	fmacs	s16  , s0,  s8
	fmacs	s17  , s1,  s8

	add	AO , AO, #8
	add	BO , BO, #4

.endm

.macro SAVE2x1


	flds		s0, ALPHA

	flds	s8 , [CO1]
	flds	s9 , [CO1, #4 ]

	fmacs	s8 , s0 , s16
	fmacs	s9 , s0 , s17

	fsts	s8 , [CO1]
	fsts	s9 , [CO1, #4 ]

	add	CO1, CO1, #8

.endm

/******************************************************************************/

.macro INIT1x1

	flds			s16, FP_ZERO

.endm



.macro KERNEL1x1_SUB

	flds	s8 , [ BO ]

	flds	s0 , [ AO ]

	fmacs	s16  , s0,  s8

	add	AO , AO, #4
	add	BO , BO, #4

.endm

.macro SAVE1x1


	flds		s0, ALPHA

	flds	s8 , [CO1]
	fmacs	s8 , s0 , s16
	fsts	s8 , [CO1]

	add	CO1, CO1, #4

.endm


.macro DO_M4X4MATS_NEON_I
	pld	[ AO , #A_PRE ]
	vmul.f32   q12,  q8,  d0[0]
	vmla.f32   q12,  q9,  d2[0]
	vmla.f32   q12,  q10, d4[0]
	vmla.f32   q12,  q11, d6[0]

	pld	[ BO , #B_PRE ]
	vmul.f32   q13,  q8,  d0[1]
	vmla.f32   q13,  q9,  d2[1]
	vmla.f32   q13,  q10, d4[1]
	vmla.f32   q13,  q11, d6[1]

	vmul.f32   q14,  q8,  d1[0]
	vmla.f32   q14,  q9,  d3[0]
	vmla.f32   q14,  q10, d5[0]
	vmla.f32   q14,  q11, d7[0]

	vmul.f32   q15,  q8,  d1[1]
	vmla.f32   q15,  q9,  d3[1]
	vmla.f32   q15,  q10, d5[1]
	vmla.f32   q15,  q11, d7[1]
.endm

.macro DO_M4X4MATS_NEON_LOOP
#if 1
	//pld		   [ AO , #128 ]
	//pld		   [ BO , #128 ]
	vld1.32    { q8-q9 }, [AO]!
	vld1.32    {q0-q1}, [BO]!

	vmla.f32   q12,  q8,  d0[0]
	vmla.f32   q13,  q8,  d0[1]
	vld1.32    {q2-q3}, [BO]!
	vmla.f32   q14,  q8,  d1[0]
	vmla.f32   q15,  q8,  d1[1]

	vmla.f32   q12,  q9,  d2[0]
	vmla.f32   q13,  q9,  d2[1]
	vld1.32    {q10-q11}, [AO]!
	vmla.f32   q14,  q9,  d3[0]
	vmla.f32   q15,  q9,  d3[1]

	vmla.f32   q12,  q10, d4[0]
	vmla.f32   q13,  q10, d4[1]
	vmla.f32   q14,  q10, d5[0]
	vmla.f32   q15,  q10, d5[1]

	vmla.f32   q12,  q11, d6[0]
	vmla.f32   q13,  q11, d6[1]
	vmla.f32   q14,  q11, d7[0]
	vmla.f32   q15,  q11, d7[1]
#else
	//pld	[ AO , #8 ]
	vmla.f32   q12,  q8,  d0[0]
	vmla.f32   q12,  q9,  d2[0]
	vmla.f32   q12,  q10, d4[0]
	vmla.f32   q12,  q11, d6[0]

	//pld	[ BO , #8 ]
	vmla.f32   q13,  q8,  d0[1]
	vmla.f32   q13,  q9,  d2[1]
	vmla.f32   q13,  q10, d4[1]
	vmla.f32   q13,  q11, d6[1]

	vmla.f32   q14,  q8,  d1[0]
	vmla.f32   q14,  q9,  d3[0]
	vmla.f32   q14,  q10, d5[0]
	vmla.f32   q14,  q11, d7[0]

	vmla.f32   q15,  q8,  d1[1]
	vmla.f32   q15,  q9,  d3[1]
	vmla.f32   q15,  q10, d5[1]
	vmla.f32   q15,  q11, d7[1]
#endif
.endm

.macro DO_M4X4MATS_NEON_QR_TO_SR
	vmov q4, q12
	vmov q5, q13
	vmov q6, q14
	vmov q7, q15
.endm


/**************************************************************************************
* End of macro definitions
**************************************************************************************/

	PROLOGUE

	.align 5

	push	{r4 - r9, fp}
	add	fp, sp, #24
	sub	sp, sp, #STACKSIZE				// reserve stack

	@backup MNK to stack  [OLD_M r0] [OLD_N r1] [OLD_K r2]
	str	OLD_M, M
	str	OLD_N, N
	str	OLD_K, K
	
#ifdef ARM_SOFTFP_ABI
	str	OLD_ALPHA, ALPHA
#else //hard
	@backup A ALPHA to stack [OLD_A r3] [OLD_ALPHA s0]
	str	OLD_A, A
	vstr OLD_ALPHA, ALPHA
#endif
	sub	r3, fp, #128
	vstm	r3, { s8 - s31} 				// store floating point registers

    movs    r4, #0
    str     r4, FP_ZERO
    str     r4, FP_ZERO_1

	@LDC = 4bytes * OLD_LDC
	ldr	r3, OLD_LDC
	lsl	r3, r3, #2					// ldc = ldc * 4
	str	r3, LDC						// LDC = 4bytes * OLD_LDC

	@load K BC from stack
	ldr	K1, K					    //K1 r7
	ldr	BC, B						//BC R12

	@ if ((J = N / 4) <= 0) goto L2_BEGIN
	ldr	J, N						//R1
	asrs J, J, #2				    //J = J / 4
	ble	sgemm_kernel_L2_BEGIN

sgemm_kernel_L4_BEGIN:
	@ C += LDC*4
	ldr	CO1, C						// CO1 = C
	ldr	r4 , LDC
	lsl	r4 , r4 , #2				// LDC * 4
	add	r3 , r4, CO1
	str	r3 , C						// store

	ldr	AO, A						// AO = A
	@ preftech 160
    pld [AO , #A_PRE-64]
    pld [AO , #A_PRE-32]

sgemm_kernel_L4_M4_BEGIN:
	@ if ((I = M / 4) <= 0) goto L4_M2
	ldr	I, M						// M rows
	asrs I, I, #2					// I = M / 4
	ble	sgemm_kernel_L4_M2_BEGIN

sgemm_kernel_L4_M4_20:
	@ L = K/2
	@ if (L < 2) goto L4_M4_32
	mov	BO, BC
	asrs L, K1, #1					// L = K / 2
	cmp	L, #2
	blt	sgemm_kernel_L4_M4_32

#define ENABLE_NEON
@ L > 4
@=============4*4========================
#ifdef ENABLE_NEON
	@loop start here for 4*4
sgemm_kernel_L4_M4_NEON:
	LOAD_4x4MATS
	DO_M4X4MATS_NEON_I  //Q12 = q8,  d0[0], then Q12 += q8,  d0[0]

	subs L, L, #2
	cmp	L, #2
	bge sgemm_kernel_L4_M4_NEON_LOOP

	@ dump Q TO R first
	DO_M4X4MATS_NEON_QR_TO_SR

	@ if (0 == L)
	cmp	L, #0
	ble sgemm_kernel_L4_M4_44

	@ 2   4*1  x 1*4
	KERNEL4x4_NEON_I
	KERNEL4x4_E

	subs L, L, #1
	b sgemm_kernel_L4_M4_44

sgemm_kernel_L4_M4_NEON_LOOP:
	DO_M4X4MATS_NEON_LOOP  //Q12 += q8,  d0[0] not clear Q12 first

	subs L, L, #2
	cmp	L, #2
	bge sgemm_kernel_L4_M4_NEON_LOOP

	DO_M4X4MATS_NEON_QR_TO_SR

	cmp	L, #0
	ble sgemm_kernel_L4_M4_44

	@ last
	KERNEL4x4_NEON_I
	KERNEL4x4_E

	subs L, L, #1
	b sgemm_kernel_L4_M4_44
#else
	KERNEL4x4_I     @  4*1 X 1*4
	KERNEL4x4_M2    @  4*1 X 1*4

	@ L -= 2
	@ if (L <= 0) goto L4_M4_22a
	subs L, L, #2		//L -= 2
	ble	sgemm_kernel_L4_M4_22a
	.align 5

sgemm_kernel_L4_M4_22:

	KERNEL4x4_M1
	KERNEL4x4_M2
	@ L -= 1
	@ if (L > 0) do loop; else goto L4_M4_22a
	subs L, L, #1
	bgt	sgemm_kernel_L4_M4_22

sgemm_kernel_L4_M4_22a:

	KERNEL4x4_M1
	@ 4*4 end all data caculate store into s16~s31
	KERNEL4x4_E

	b sgemm_kernel_L4_M4_44
#endif

sgemm_kernel_L4_M4_32:

	tst	L, #1
	ble	sgemm_kernel_L4_M4_40

	KERNEL4x4_I

	KERNEL4x4_E

	b	 sgemm_kernel_L4_M4_44


sgemm_kernel_L4_M4_40:

	INIT4x4


sgemm_kernel_L4_M4_44:

	ands	L , K1, #1					// L = L % 2
	@ if(L <= 0) goto L4_M4
	ble	sgemm_kernel_L4_M4_100

sgemm_kernel_L4_M4_46:

	KERNEL4x4_SUB

	subs	L, L, #1
	bne	sgemm_kernel_L4_M4_46

sgemm_kernel_L4_M4_100:
	SAVE4x4_OLD

sgemm_kernel_L4_M4_END:

	subs	I, I, #1
	bne	sgemm_kernel_L4_M4_20


sgemm_kernel_L4_M2_BEGIN:

	ldr	I, M
	tst	I , #3
	ble	sgemm_kernel_L4_END

	tst	I, #2					// I = I / 2
	ble	sgemm_kernel_L4_M1_BEGIN

sgemm_kernel_L4_M2_20:

	INIT2x4

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L4_M2_40

sgemm_kernel_L4_M2_22:

	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB

	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L4_M2_22


sgemm_kernel_L4_M2_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L4_M2_100

sgemm_kernel_L4_M2_42:

	KERNEL2x4_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L4_M2_42

sgemm_kernel_L4_M2_100:

	SAVE2x4

sgemm_kernel_L4_M2_END:


sgemm_kernel_L4_M1_BEGIN:

	tst	I, #1					// I = I % 2
	ble	sgemm_kernel_L4_END

sgemm_kernel_L4_M1_20:

	INIT1x4

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L4_M1_40

sgemm_kernel_L4_M1_22:
	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB

	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L4_M1_22


sgemm_kernel_L4_M1_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L4_M1_100

sgemm_kernel_L4_M1_42:

	KERNEL1x4_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L4_M1_42

sgemm_kernel_L4_M1_100:

	SAVE1x4


sgemm_kernel_L4_END:

	mov	r3, BC
	mov	r4, K1
	lsl	r4, r4, #4					// k * 4 * 4
	add	r3, r3, r4					// B = B + K * 4 * 4
	mov	BC, r3

	subs	J , #1						// j--
	bgt	sgemm_kernel_L4_BEGIN



/*********************************************************************************************/

sgemm_kernel_L2_BEGIN:

	ldr	J , N
	tst	J , #3
	ble	sgemm_kernel_L999

	tst	J , #2
	ble	sgemm_kernel_L1_BEGIN

	ldr	CO1, C						// CO1 = C
	ldr	r4 , LDC
	lsl	r4 , r4 , #1					// LDC * 2
	add	r3 , r4, CO1
	str	r3 , C						// store C

	ldr	AO, A						// AO = A
        //pld     [AO , #A_PRE-96]
        //pld     [AO , #A_PRE-64]
        //pld     [AO , #A_PRE-32]



sgemm_kernel_L2_M4_BEGIN:

	ldr	I, M
	asrs	I, I, #2					// I = I / 4
	ble	sgemm_kernel_L2_M2_BEGIN

sgemm_kernel_L2_M4_20:

	INIT4x2

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L2_M4_40
	.align 5

sgemm_kernel_L2_M4_22:
	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB

	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L2_M4_22


sgemm_kernel_L2_M4_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L2_M4_100

sgemm_kernel_L2_M4_42:

	KERNEL4x2_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L2_M4_42

sgemm_kernel_L2_M4_100:

	SAVE4x2

sgemm_kernel_L2_M4_END:

	subs	I, I, #1
	bgt	sgemm_kernel_L2_M4_20


sgemm_kernel_L2_M2_BEGIN:

	ldr	I, M
	tst	I , #3
	ble	sgemm_kernel_L2_END

	tst	I, #2					// I = I / 2
	ble	sgemm_kernel_L2_M1_BEGIN

sgemm_kernel_L2_M2_20:

	INIT2x2

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L2_M2_40

sgemm_kernel_L2_M2_22:

	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB

	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L2_M2_22


sgemm_kernel_L2_M2_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L2_M2_100

sgemm_kernel_L2_M2_42:

	KERNEL2x2_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L2_M2_42

sgemm_kernel_L2_M2_100:

	SAVE2x2

sgemm_kernel_L2_M2_END:


sgemm_kernel_L2_M1_BEGIN:

	tst	I, #1					// I = I % 2
	ble	sgemm_kernel_L2_END

sgemm_kernel_L2_M1_20:

	INIT1x2

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L2_M1_40

sgemm_kernel_L2_M1_22:
	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB

	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L2_M1_22


sgemm_kernel_L2_M1_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L2_M1_100

sgemm_kernel_L2_M1_42:

	KERNEL1x2_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L2_M1_42

sgemm_kernel_L2_M1_100:

	SAVE1x2


sgemm_kernel_L2_END:

	mov	r3, BC
	mov	r4, K1
	lsl	r4, r4, #3					// k * 2 * 4
	add	r3, r3, r4					// B = B + K * 2 * 4
	mov	BC, r3

/*********************************************************************************************/

sgemm_kernel_L1_BEGIN:

	ldr	J , N
	tst	J , #1
	ble	sgemm_kernel_L999


	ldr	CO1, C						// CO1 = C
	ldr	r4 , LDC
	add	r3 , r4, CO1
	str	r3 , C						// store C

	ldr	AO, A						// AO = A
    //pld     [AO , #A_PRE-96]
    //pld     [AO , #A_PRE-64]
    //pld     [AO , #A_PRE-32]

sgemm_kernel_L1_M4_BEGIN:

	ldr	I, M
	asrs	I, I, #2					// I = I / 4
	ble	sgemm_kernel_L1_M2_BEGIN

sgemm_kernel_L1_M4_20:

	INIT4x1

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L1_M4_40
	.align 5

sgemm_kernel_L1_M4_22:
	KERNEL4x1_SUB
	KERNEL4x1_SUB
	KERNEL4x1_SUB
	KERNEL4x1_SUB

	KERNEL4x1_SUB
	KERNEL4x1_SUB
	KERNEL4x1_SUB
	KERNEL4x1_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L1_M4_22


sgemm_kernel_L1_M4_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L1_M4_100

sgemm_kernel_L1_M4_42:

	KERNEL4x1_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L1_M4_42

sgemm_kernel_L1_M4_100:

	SAVE4x1

sgemm_kernel_L1_M4_END:

	subs	I, I, #1
	bgt	sgemm_kernel_L1_M4_20


sgemm_kernel_L1_M2_BEGIN:

	ldr	I, M
	tst	I , #3
	ble	sgemm_kernel_L1_END

	tst	I, #2					// I = I / 2
	ble	sgemm_kernel_L1_M1_BEGIN

sgemm_kernel_L1_M2_20:

	INIT2x1

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L1_M2_40

sgemm_kernel_L1_M2_22:

	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB

	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L1_M2_22


sgemm_kernel_L1_M2_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L1_M2_100

sgemm_kernel_L1_M2_42:

	KERNEL2x1_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L1_M2_42

sgemm_kernel_L1_M2_100:

	SAVE2x1

sgemm_kernel_L1_M2_END:


sgemm_kernel_L1_M1_BEGIN:

	tst	I, #1					// I = I % 2
	ble	sgemm_kernel_L1_END

sgemm_kernel_L1_M1_20:

	INIT1x1

	mov	BO, BC
	asrs	L , K1, #3					// L = L / 8
	ble	sgemm_kernel_L1_M1_40

sgemm_kernel_L1_M1_22:
	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB

	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L1_M1_22


sgemm_kernel_L1_M1_40:

	ands	L , K1, #7					// L = L % 8
	ble	sgemm_kernel_L1_M1_100

sgemm_kernel_L1_M1_42:

	KERNEL1x1_SUB

	subs	L, L, #1
	bgt	sgemm_kernel_L1_M1_42

sgemm_kernel_L1_M1_100:

	SAVE1x1


sgemm_kernel_L1_END:


sgemm_kernel_L999:

	sub	r3, fp, #128
	vldm	r3, { s8 - s31}					// restore floating point registers

	movs	r0, #0						// set return value
	sub	sp, fp, #24
	pop	{r4 - r9, fp}
	bx	lr

	EPILOGUE

