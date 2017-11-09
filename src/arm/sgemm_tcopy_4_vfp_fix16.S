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
* 2013/11/06 Saar
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

#define B	[fp, #4 ]
#define A	[fp, #-248 ]

#define M	r0
#define N	r1
#define M4	r2

#define	LDA	r5

#define	AO1	r6
#define	BO1	r7
#define	BO2	r8
#define	BO3	r9

#define I	r4
#define	J	r12

#define A_PRE	256

/**************************************************************************************
* Macro definitions
**************************************************************************************/
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


/*==========================================*/
.macro COPY4x4_1

	pld	[ AO1, #A_PRE  ]
	vld1.32         { q0 }, [AO1]
	FLOAT2FIX_4_0
	vmovn.i32 d8,  q0

	add	r3, AO1, LDA
	pld	[ r3, #A_PRE  ]
	vld1.32         { q1 }, [r3]
	FLOAT2FIX_4_4
	vmovn.i32 d9,  q1

	add	r3, r3, LDA
	pld	[ r3, #A_PRE  ]
	vld1.32         { q2 }, [r3]
	FLOAT2FIX_4_8
	vmovn.i32 d10, q2

	add	r3, r3, LDA
	pld	[ r3, #A_PRE  ]
	vld1.32         { q3 }, [r3]
	FLOAT2FIX_4_12
	vmovn.i32 d11, q3

	add	AO1, AO1, #16
  fstmias	BO1, { s16 - s23 }
  add	BO1, BO1, M4
.endm

.macro COPY4x4_2

	vld1.32         { q0 }, [AO1]
	FLOAT2FIX_4_0
	vmovn.i32 d8,  q0
	
	add	r3, AO1, LDA

	vld1.32         { q1 }, [r3]
	FLOAT2FIX_4_4
	vmovn.i32 d9,  q1
	
	add	r3, r3, LDA

	vld1.32         { q2 }, [r3]
	FLOAT2FIX_4_8
	vmovn.i32 d10,  q2

	add	r3, r3, LDA

	vld1.32         { q3 }, [r3]
	FLOAT2FIX_4_12
	vmovn.i32 d11,  q3

	add	AO1, AO1, #16
	fstmias	BO1, { s16 - s23 }

  add	BO1, BO1, M4
.endm


.macro COPY2x4
	fldmias	AO1, { s0 - s1 }
	FLOAT2FIX_2_0

	add	r3, AO1, LDA
	fldmias	r3, { s2 - s3 }
	FLOAT2FIX_2_2	
	
	add	r3, r3, LDA
	fldmias	r3, { s4 - s5 }
	FLOAT2FIX_2_4

	add	r3, r3, LDA
	fldmias	r3, { s6 - s7 }
	FLOAT2FIX_2_6	
	
	FLOAT2FIX_2_NARROW

	add	AO1, AO1, #8
	fstmias	BO2!, { s16 - s19 }
.endm

.macro COPY1x4
	fldmias	AO1, { s0 }

	add	r3, AO1, LDA
	fldmias	r3, { s1 }

	add	r3, r3, LDA
	fldmias	r3, { s2 }

	add	r3, r3, LDA
	fldmias	r3, { s3 }
	
	FLOAT2FIX_4_0
	FLOAT2FIX_1_NARROW

	add	AO1, AO1, #4
	fstmias	BO3!, { s16 - s17 }
.endm

/*************************************************************************************************************************/

.macro COPY4x2
	fldmias	AO1, { s0 - s3 }
	FLOAT2FIX_4_0
	
	add	r3, AO1, LDA
	fldmias	r3, { s4 - s7 }
	FLOAT2FIX_4_4
		
	FLOAT2FIX_2_NARROW
	
	add	AO1, AO1, #16  
	fstmias	BO1, { s16 - s19 }
	add	BO1, BO1, M4
.endm

.macro COPY2x2

	fldmias	AO1, { s0 - s1 }

	add	r3, AO1, LDA
	fldmias	r3, { s2 - s3 }
	
	FLOAT2FIX_4_0
	FLOAT2FIX_1_NARROW

	add	AO1, AO1, #8
	fstmias	BO2!, { s16 - s17 }
.endm

.macro COPY1x2

	fldmias	AO1, { s0 }

	add	r3, AO1, LDA
	fldmias	r3, { s1 }
	
	FLOAT2FIX_2_0
	FLOAT2FIX_1_NARROW

	add	AO1, AO1, #4
	fstmias	BO3!, { s16 }
.endm

/*************************************************************************************************************************/

.macro COPY4x1

	fldmias	AO1, { s0 - s3 }
	
	FLOAT2FIX_4_0
	FLOAT2FIX_1_NARROW
	
	add	AO1, AO1, #16
	fstmias	BO1, { s16 - s17 }
	add	BO1, BO1, M4
.endm

.macro COPY2x1

	fldmias	AO1, { s0 - s1 }

	FLOAT2FIX_2_0
	FLOAT2FIX_1_NARROW

	add	AO1, AO1, #8
	fstmias	BO2!, { s16 }
.endm

.macro COPY1x1
	fldmias	AO1, { s0 }
	
	vcvt.S16.F32 s0,s0,#FRACTION
	
	push {r0}
	vmov r0, s0
	strh r0, [BO3]
	pop {r0}

	add	AO1, AO1, #4
	add	BO3, BO3, #2
.endm

/**************************************************************************************
* End of macro definitions
**************************************************************************************/

	PROLOGUE

	.align 5

	push	{r4 - r9, fp}
	add	fp, sp, #24
	sub	sp, sp, #STACKSIZE				// reserve stack

	str	OLD_A, A					// store A

	lsl	LDA, OLD_LDA, #2				// lda = lda * SIZE

	sub	r4, fp, #128
	vstm	r4, { s8 - s31} 				// store floating point registers

	lsl	r4 , M, #1					// M * SIZE  fix change from 2 to 1

	ldr	r3, B

	and	BO2 , N , #-4
	and	BO3 , N , #-2

	mul	BO2, BO2, r4
	mul	BO3, BO3, r4

	add	BO2 , BO2, r3
	add	BO3 , BO3, r3

	lsl	M4, M, #3					// M4 = M * 4 * SIZE fix change from 4 to 3

	
sgemm_tcopy_L4_BEGIN:

	asrs	J, M, #2					// J = N / 4
	ble	sgemm_tcopy_L2_BEGIN

sgemm_tcopy_L4_M4_BEGIN:

	ldr	AO1, A						// AO1 = A
	lsl	r3, LDA, #2					// r3 = 4 * LDA
	add	r3, r3 , AO1					// A = A + 4 * LDA
	str	r3, A						// store A

	ldr	BO1, B
	add	r3, BO1, #32					// B = B + 16 * SIZE fix change from 64 to 32
	str	r3, B

	asrs	I, N, #2					// I = M / 4
	ble	sgemm_tcopy_L4_M4_40

sgemm_tcopy_L4_M4_20:

	COPY4x4_1

	subs	I , I , #1
	ble	sgemm_tcopy_L4_M4_40

	COPY4x4_2

	subs	I , I , #1
	bne	sgemm_tcopy_L4_M4_20


sgemm_tcopy_L4_M4_40:

	tst	N , #2
	ble	sgemm_tcopy_L4_M4_60

	COPY2x4


sgemm_tcopy_L4_M4_60:

	tst	N, #1
	ble	sgemm_tcopy_L4_M4_END

	COPY1x4


sgemm_tcopy_L4_M4_END:

	subs	J , J, #1						// j--
	bne	sgemm_tcopy_L4_M4_BEGIN



/*********************************************************************************************/

sgemm_tcopy_L2_BEGIN:

	tst	M, #3
	ble	sgemm_tcopy_L999

	tst	M, #2
	ble	sgemm_tcopy_L1_BEGIN

sgemm_tcopy_L2_M4_BEGIN:

	ldr	AO1, A						// AO1 = A
	lsl	r3, LDA, #1					// r3 = 2 * LDA
	add	r3, r3 , AO1					// A = A + 2 * LDA
	str	r3, A						// store A

	ldr	BO1, B
	add	r3, BO1, #16					// B = B + 8 * SIZE fix change from 32 to 16
	str	r3, B

	asrs	I, N, #2					// I = M / 4
	ble	sgemm_tcopy_L2_M4_40

sgemm_tcopy_L2_M4_20:

	COPY4x2

	subs	I , I , #1
	bne	sgemm_tcopy_L2_M4_20


sgemm_tcopy_L2_M4_40:

	tst	N , #2
	ble	sgemm_tcopy_L2_M4_60

	COPY2x2

sgemm_tcopy_L2_M4_60:

	tst	N , #1
	ble	sgemm_tcopy_L2_M4_END

	COPY1x2


sgemm_tcopy_L2_M4_END:


/*********************************************************************************************/

sgemm_tcopy_L1_BEGIN:

	tst	M, #1
	ble	sgemm_tcopy_L999


sgemm_tcopy_L1_M4_BEGIN:

	ldr	AO1, A						// AO1 = A
	add	r3, LDA , AO1					// A = A + 1 * LDA
	str	r3, A						// store A

	ldr	BO1, B
	add	r3, BO1, #8					// B = B + 4 * SIZE   fix change from 16 to 8
	str	r3, B

	asrs	I, N, #2					// I = M / 4
	ble	sgemm_tcopy_L1_M4_40

sgemm_tcopy_L1_M4_20:

	COPY4x1

	subs	I , I , #1
	bne	sgemm_tcopy_L1_M4_20


sgemm_tcopy_L1_M4_40:

	tst	N , #2
	ble	sgemm_tcopy_L1_M4_60

	COPY2x1

sgemm_tcopy_L1_M4_60:

	tst	N , #1
	ble	sgemm_tcopy_L1_M4_END

	COPY1x1

sgemm_tcopy_L1_M4_END:

sgemm_tcopy_L999:

	sub	r3, fp, #128
	vldm	r3, { s8 - s31}					// restore floating point registers

	mov	r0, #0						// set return value
	sub	sp, fp, #24
	pop	{r4 - r9, fp}
	bx	lr

	EPILOGUE

