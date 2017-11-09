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
* 2013/11/05 Saar
* 	 BLASTEST 		: OK
* 	 CTEST			: OK
* 	 TEST			: OK
*
**************************************************************************************/

#define ASSEMBLER
#include "common.h"

#ifndef FRACTION
#define FRACTION 14
#endif

#define FRACTIONB 2*FRACTION

#define STACKSIZE 256

#define	OLD_M	r0
#define	OLD_N	r1
#define	OLD_A	r2
#define	OLD_LDA	r3


/******************************************************
* [fp, #-128] - [fp, #-64] is reserved
* for store and restore of floating point
* registers
*******************************************************/

#define LDA	[fp, #-260 ]

#define B	[fp, #4 ]

#define M	r0
#define N	r1
#define A	r2

#define	BO	r5

#define	AO1	r6
#define	AO2	r7
#define	AO3	r8
#define	AO4	r9

#define I	r3
#define	J	r12

#define A_PRE	192

/*=====================4====================*/
.macro FLOAT2FIX_4_0
	vcvt.S32.F32 q0,q0,#FRACTION
.endm

.macro FLOAT2FIX_4_4
	vcvt.S32.F32 q1,q1,#FRACTION
.endm

.macro FLOAT2FIX_4_8
	vcvt.S32.F32 q2,q2,#FRACTION
.endm

.macro FLOAT2FIX_4_12
	vcvt.S32.F32 q3,q3,#FRACTION
.endm

.macro FLOAT2FIX_4_NARROW
	vmovn.i32 d8,  q0
	vmovn.i32 d9,  q1
	vmovn.i32 d10, q2
	vmovn.i32 d11, q3
.endm


/*=====================2====================*/
.macro FLOAT2FIX_2_0
	vcvt.S32.F32 d0,d0,#FRACTION
.endm

.macro FLOAT2FIX_2_2
	vcvt.S32.F32 d1,d1,#FRACTION
.endm

.macro FLOAT2FIX_2_4
	vcvt.S32.F32 d2,d2,#FRACTION
.endm

.macro FLOAT2FIX_2_6
	vcvt.S32.F32 d3,d3,#FRACTION
.endm

.macro FLOAT2FIX_2_NARROW
	vmovn.i32 d8,  q0
	vmovn.i32 d9,  q1
.endm

.macro FLOAT2FIX_1_NARROW
	vmovn.i32 d8,  q0
.endm

/**************************************************************************************
* Macro definitions
**************************************************************************************/

.macro COPY4x4
	vld1.32 {q0}, [AO1]!
	vld1.32 {q1}, [AO2]!
	vld1.32 {q2}, [AO3]!
	vtrn.32 d0,d2
	vld1.32 {q3}, [AO4]!

	vtrn.32 d1,d3
	vtrn.32 d4,d6
	vswp d1,d4
	vtrn.32 d5,d7
	vswp d3,d6

	vcvt.S32.F32 q0,q0,#FRACTION
	vmovn.i32 d8,  q0
	vcvt.S32.F32 q1,q1,#FRACTION
	vmovn.i32 d9,  q1

	vcvt.S32.F32 q2,q2,#FRACTION

	vmovn.i32 d10,  q2
	vcvt.S32.F32 q3,q3,#FRACTION
	vmovn.i32 d11,  q3

	fstmias	BO!, { s16 - s23 }
.endm

.macro COPY1x4
	flds s0 , [ AO1, #0  ]
	flds s1 , [ AO2, #0  ]
	add	AO1, AO1, #4
	flds s2 , [ AO3, #0  ]
	add	AO2, AO2, #4
	flds s3 , [ AO4, #0  ]

	FLOAT2FIX_4_0
	FLOAT2FIX_1_NARROW
	
	add	AO3, AO3, #4

	fstmias	BO!, { s16 - s17 }
	add	AO4, AO4, #4
.endm

.macro COPY4x2
	vld1.32 {q0}, [AO1]!
	vld1.32 {q1}, [AO2]!
	vzip.32 q0,q1

	FLOAT2FIX_4_0
	FLOAT2FIX_4_4	
	FLOAT2FIX_2_NARROW

	fstmias	BO!, { s16 - s19 }
.endm

.macro COPY1x2
	flds s0 , [ AO1, #0  ]
	flds s1 , [ AO2, #0  ]
	add	AO1, AO1, #4
	
	FLOAT2FIX_2_0
	FLOAT2FIX_1_NARROW

	add	AO2, AO2, #4
	fstmias	BO!, { s16 }
.endm

.macro COPY4x1
	vld1.32 {q0}, [AO1]!

	FLOAT2FIX_4_0
	FLOAT2FIX_1_NARROW

	fstmias	BO!, { s16 - s17 }
.endm

.macro COPY1x1
	flds s0 , [ AO1, #0  ]

	vcvt.S16.F32 s0,s0,#FRACTION
	
	push {r0}
	vmov r0, s0
	strh r0, [BO]
	pop {r0}

	add	BO, BO, #2
	add	AO1, AO1, #4
.endm


/**************************************************************************************
* End of macro definitions
**************************************************************************************/

	PROLOGUE

	.align 5

	push	{r4 - r9, fp}
	add	fp, sp, #24
	sub	sp, sp, #STACKSIZE				// reserve stack


	lsl	r3, r3, #2					// lda = lda * 4
	str	r3, LDA

	sub	r4, fp, #128
	vstm	r4, { s8 - s31} 				// store floating point registers

	ldr	BO, B

sgemm_ncopy_L4_BEGIN:

	asrs	J, N, #2					// J = N / 4
	ble	sgemm_ncopy_L2_BEGIN

sgemm_ncopy_L4_M4_BEGIN:

	mov	AO1, A						// AO1 = A
	ldr	r4 , LDA
	add	AO2, AO1, r4
	add	AO3, AO2, r4
	add	AO4, AO3, r4
	add	A  , AO4, r4					// A = A + 4 * LDA

	asrs	I, M, #2					// I = M / 4
	ble	sgemm_ncopy_L4_M4_40

sgemm_ncopy_L4_M4_20:

	pld	[ AO1, #A_PRE ]
	pld	[ AO2, #A_PRE ]
	pld	[ AO3, #A_PRE ]
	pld	[ AO4, #A_PRE ]
	COPY4x4

	subs	I , I , #1
	ble	sgemm_ncopy_L4_M4_40

	COPY4x4

	subs	I , I , #1
	bne	sgemm_ncopy_L4_M4_20


sgemm_ncopy_L4_M4_40:

	ands	I, M , #3
	ble	sgemm_ncopy_L4_M4_END

sgemm_ncopy_L4_M4_60:

	COPY1x4

	subs	I , I , #1
	bne	sgemm_ncopy_L4_M4_60


sgemm_ncopy_L4_M4_END:

	subs	J , J, #1						// j--
	bne	sgemm_ncopy_L4_M4_BEGIN



/*********************************************************************************************/

sgemm_ncopy_L2_BEGIN:

	tst	N, #3
	ble	sgemm_ncopy_L999

	tst	N, #2
	ble	sgemm_ncopy_L1_BEGIN

sgemm_ncopy_L2_M4_BEGIN:

	mov	AO1, A						// AO1 = A
	ldr	r4 , LDA
	add	AO2, AO1, r4
	add	A  , AO2, r4 					// A = A + 2 * LDA

	asrs	I, M, #2					// I = M / 4
	ble	sgemm_ncopy_L2_M4_40

sgemm_ncopy_L2_M4_20:

	COPY4x2

	subs	I , I , #1
	bne	sgemm_ncopy_L2_M4_20


sgemm_ncopy_L2_M4_40:

	ands	I, M , #3
	ble	sgemm_ncopy_L2_M4_END

sgemm_ncopy_L2_M4_60:

	COPY1x2

	subs	I , I , #1
	bne	sgemm_ncopy_L2_M4_60


sgemm_ncopy_L2_M4_END:


/*********************************************************************************************/

sgemm_ncopy_L1_BEGIN:

	tst	N, #1
	ble	sgemm_ncopy_L999


sgemm_ncopy_L1_M4_BEGIN:

	mov	AO1, A						// AO1 = A
	ldr	r4 , LDA
	add	A  , AO1, r4 					// A = A + 1 * LDA

	asrs	I, M, #2					// I = M / 4
	ble	sgemm_ncopy_L1_M4_40

sgemm_ncopy_L1_M4_20:

	COPY4x1

	subs	I , I , #1
	bne	sgemm_ncopy_L1_M4_20


sgemm_ncopy_L1_M4_40:

	ands	I, M , #3
	ble	sgemm_ncopy_L1_M4_END

sgemm_ncopy_L1_M4_60:

	COPY1x1

	subs	I , I , #1
	bne	sgemm_ncopy_L1_M4_60


sgemm_ncopy_L1_M4_END:


sgemm_ncopy_L999:

	sub	r3, fp, #128
	vldm	r3, { s8 - s31}					// restore floating point registers

	movs	r0, #0						// set return value
	sub	sp, fp, #24
	pop	{r4 - r9, fp}
	bx	lr

	EPILOGUE

