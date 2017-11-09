/****************copyright lee (tydslj@qq.com)*****************/
#define ASSEMBLER
#include "common.h"

#ifndef FRACTION
#define FRACTION 14
#endif

#define FRACTIONB 2*FRACTION

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

#define A_PRE	256
#define B_PRE	256
#define C_PRE	64

/**************************************************************************************
* Macro definitions
**************************************************************************************/

.macro INIT4x4
	veor q4, q4
	veor q5, q5
	veor q6, q6
	veor q7, q7
.endm
	
.macro KERNEL4x4_NEON_I
	pld	[ AO , #A_PRE ]
	vld1.32  { d0 }, [AO]!
	
	pld	[ BO , #B_PRE ]
	vld1.32  { d4 }, [BO]!
	
	vmlal.s16 q4, d0, d4[0]
	vld1.32  { d1 }, [AO]!

	vmlal.s16 q5, d0, d4[1]
	vrev64.32 d4, d4
	vmlal.s16 q6, d0, d4[0]
	vld1.32  { d5 }, [BO]!
	
	vmlal.s16 q7, d0, d4[1]
.endm

.macro KERNEL4x4_I
	pld	[ AO , #A_PRE ]
	vld1.32  { d0 }, [AO]!
	
	pld	[ BO , #B_PRE ]
	vld1.32  { d4 }, [BO]!

	vmull.s16 q4, d0, d4[0]
	vld1.32  { d1 }, [AO]!
	
	vmull.s16 q5, d0, d4[1]
	vrev64.32 d4, d4
	vmull.s16 q6, d0, d4[0]
	vld1.32  { d5 }, [BO]!
	vmull.s16 q7, d0, d4[1]
.endm

.macro KERNEL4x4_M2
	pld	[ AO , #A_PRE ]
	vmlal.s16 q4, d1, d5[0]
	vld1.32  { d0 }, [AO]!
	
	vmlal.s16 q5, d1, d5[1]
	vrev64.32 d5, d5
	pld	[ BO , #B_PRE ]
	vmlal.s16 q6, d1, d5[0]
	vld1.32  { d4 }, [BO]!	
	vmlal.s16 q7, d1, d5[1]	
.endm

.macro KERNEL4x4_M1	
	vmlal.s16 q4, d0, d4[0]
	vld1.32  { d1 }, [AO]!

	vmlal.s16 q5, d0, d4[1]
	vrev64.32 d4, d4
	vmlal.s16 q6, d0, d4[0]
	vld1.32  { d5 }, [BO]!
	
	vmlal.s16 q7, d0, d4[1]
.endm

.macro KERNEL4x4_E
	vmlal.s16 q4, d1, d5[0]
	vmlal.s16 q5, d1, d5[1]
	vrev64.32 d5, d5
	vmlal.s16 q6, d1, d5[0]
	vmlal.s16 q7, d1, d5[1]
.endm

.macro KERNEL4x4_SUB	
	vld1.32  { d0 }, [AO]!
	vld1.32  { d4 }, [BO]!
	
	vmlal.s16 q4, d0, d4[0]
	vmlal.s16 q5, d0, d4[1]
	vrev64.32 d4, d4
	vmlal.s16 q6, d0, d4[0]
	vmlal.s16 q7, d0, d4[1]
.endm

.macro SAVE4x4_OLD
	ldr	r3  , LDC
	add	CO2 , CO1, r3
	flds		s0, ALPHA
	vcvt.F32.S32 d8,d8,#FRACTIONB
	add	r4  , CO2, r3
	vcvt.F32.S32 d9,d9,#FRACTIONB
	@ vcvt.F32.S32 q4,q4,#FRACTIONB
	fldmias CO1, { s8 - s11 }

	vld1.32 {q3}, [CO2]
	vcvt.F32.S32 d10,d10,#FRACTIONB
	vmla.f32   q2,  q4,  d0[0]	
	vcvt.F32.S32 d11,d11,#FRACTIONB
	@ vcvt.F32.S32 q5,q5,#FRACTIONB

	vmla.f32   q3,  q5,  d0[0]
	fsts	s8 , [CO1]
	vcvt.F32.S32 d12,d12,#FRACTIONB
	fsts	s9 , [CO1, #4 ]
	fsts	s10, [CO1, #8 ]
	vcvt.F32.S32 d13,d13,#FRACTIONB
	@ vcvt.F32.S32 q6,q6,#FRACTIONB
	fsts	s11, [CO1, #12 ]

	pld	[ CO1 , #C_PRE ]
	
	fldmias r4, { s8 - s11 }

	fmacs	s8 , s0 , s24
	fsts	s12, [CO2]
	vcvt.F32.S32 d14,d14,#FRACTIONB
	fmacs	s9 , s0 , s25
	fsts	s13, [CO2, #4 ]
	fmacs	s10, s0 , s26
	fsts	s14, [CO2, #8 ]
	vcvt.F32.S32 d15,d15,#FRACTIONB
	@ vcvt.F32.S32 q7,q7,#FRACTIONB
	fmacs	s11, s0 , s27
	fsts	s15, [CO2, #12 ]

	pld	[ CO2 , #C_PRE ]

	add	CO2, r4 , r3

	fldmias CO2, { s12 - s15 }

	vmla.f32   q3,  q7,  d0[0]
	fstmias r4, { s8 - s11 }

	pld	[ r4 , #C_PRE ]
	fstmias CO2, { s12 - s15 }
	pld	[ CO2 , #C_PRE ]

	add	CO1, CO1, #16
.endm

/******************************************************************************/

.macro INIT2x4
	veor d8, d8
	veor d10, d10
	veor d12, d12
	veor d14, d14
.endm

.macro KERNEL2x4_SUB
	vld1.32 {d0[0]}, [AO]
	vld1.32 {d4}, [BO]
	
	@ result in s4 s5 s6 s7
	vmull.s16 q1, d4, d0[0]
	vmull.s16 q3, d4, d0[1]
	vtrn.32 d2, d6
	vtrn.32 d3, d7
	vadd.i32 d8,  d8,  d2
	vadd.i32 d10, d10, d6
	add AO, AO, #4
	vadd.i32 d12, d12, d3
	vadd.i32 d14, d14, d7	

	add BO, BO, #8	
.endm

.macro SAVE2x4
	ldr	r3  , LDC
	add	CO2 , CO1, r3
	add	r4  , CO2, r3

	flds		s0, ALPHA
	vcvt.F32.S32 d8,d8,#FRACTIONB
	flds	s8 , [CO1]

	flds	s9 , [CO1, #4 ]

	fmacs	s8 , s0 , s16
	fmacs	s9 , s0 , s17

	fsts	s8 , [CO1]
	fsts	s9 , [CO1, #4 ]
	vcvt.F32.S32 d10,d10,#FRACTIONB
	flds	s12, [CO2]
	flds	s13, [CO2, #4 ]

	fmacs	s12, s0 , s20
	fmacs	s13, s0 , s21

	fsts	s12, [CO2]
	fsts	s13, [CO2, #4 ]
	vcvt.F32.S32 d12,d12,#FRACTIONB
	flds	s8 , [r4 ]
	flds	s9 , [r4 , #4 ]

	fmacs	s8 , s0 , s24
	fmacs	s9 , s0 , s25

	fsts	s8 , [r4 ]
	fsts	s9 , [r4 , #4 ]
	vcvt.F32.S32 d14,d14,#FRACTIONB
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
	ldrsh r4, [AO]
	vmov s0, r4	

	vld1.32 {d4}, [BO]
	
	vmull.s16 q1, d4, d0[0]
	
	vmov s12, s16
	vmov s13, s20
	vmov s14, s24
	vmov s15, s28
	
	vadd.i32 q3, q1, q3
	
	vmov s16, s12
	vmov s20, s13
	add	AO , AO, #2
	vmov s24, s14
	vmov s28, s15	
	
	add BO, BO, #8	
.endm

.macro SAVE1x4
	ldr	r3  , LDC
	add	CO2 , CO1, r3
	add	r4  , CO2, r3

	flds		s0, ALPHA
	vcvt.F32.S32 s16,s16,#FRACTIONB
	flds	s8 , [CO1]
	fmacs	s8 , s0 , s16
	fsts	s8 , [CO1]
	
	vcvt.F32.S32 s20,s20,#FRACTIONB
	flds	s12, [CO2]
	fmacs	s12, s0 , s20
	fsts	s12, [CO2]

	vcvt.F32.S32 s24,s24,#FRACTIONB
	flds	s8 , [r4 ]
	fmacs	s8 , s0 , s24
	fsts	s8 , [r4 ]

	add	CO2, r4 , r3
	
	vcvt.F32.S32 s28,s28,#FRACTIONB
	flds	s12, [CO2]
	fmacs	s12, s0 , s28
	fsts	s12, [CO2]

	add	CO1, CO1, #4
.endm

/******************************************************************************/

.macro INIT4x2
	veor q4, q4
	veor q5, q5
.endm

.macro KERNEL4x2_SUB
	vld1.32 {d4[0]}, [BO]
	vld1.32 {d0}, [AO]
	
	vmlal.s16 q4, d0, d4[0]
	add	AO , AO, #8
	vmlal.s16 q5, d0, d4[1]	
	
	add BO, BO, #4
.endm

.macro SAVE4x2
	ldr	r3  , LDC
	add	CO2 , CO1, r3

	flds		s0, ALPHA

	flds	s8 , [CO1]
	vcvt.F32.S32 d8,d8,#FRACTIONB
	flds	s9 , [CO1, #4 ]
	flds	s10, [CO1, #8 ]
	vcvt.F32.S32 d9,d9,#FRACTIONB
	@ vcvt.F32.S32 q4,q4,#FRACTIONB
	flds	s11, [CO1, #12 ]

	vmla.f32 q2, q4, d0[0]

	fsts	s8 , [CO1]
	vcvt.F32.S32 d10,d10,#FRACTIONB
	fsts	s9 , [CO1, #4 ]
	fsts	s10, [CO1, #8 ]
	vcvt.F32.S32 d11,d11,#FRACTIONB
	@ vcvt.F32.S32 q5,q5,#FRACTIONB
	fsts	s11, [CO1, #12 ]

	vld1.32  { q3 }, [CO2]
	
	vmla.f32 q3, q5, d0[0]

	fstmias CO2, { s12 - s15 }

	add	CO1, CO1, #16
.endm

/******************************************************************************/
.macro INIT2x2
	veor d8, d8
	veor d10, d10
.endm

.macro KERNEL2x2_SUB
	push {r0-r3}
	
	ldrsh r0, [AO]	
	ldrsh r2, [BO]
	
	mul r1, r0, r2  //s16
	vmov r3, s16
	add r3,r3,r1
	vmov s16, r3
	
	ldrsh r1, [AO, #2]	
	mul r2, r1, r2	//s17
	vmov r3, s17
	add r3,r3,r2
	vmov s17, r3
	
	ldrsh r3, [BO, #2]
	mul r0, r0, r3  //s20
	vmov r2, s20
	add r2,r2,r0
	vmov s20, r2
	
	mul r1, r1, r3  //s21
	vmov r2, s21
	add r2,r2,r1
	vmov s21, r2
	
	pop {r0-r3}
	
	add	AO , AO, #4
	add	BO , BO, #4
.endm

.macro SAVE2x2
	ldr	r3  , LDC
	add	CO2 , CO1, r3

	flds		s0, ALPHA
	
	flds	s8 , [CO1]
	vcvt.F32.S32 d8,d8,#FRACTIONB
	flds	s9 , [CO1, #4 ]

	fmacs	s8 , s0 , s16
	fmacs	s9 , s0 , s17

	fsts	s8 , [CO1]
	vcvt.F32.S32 d10,d10,#FRACTIONB
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
	push {r0-r3}
	
	ldrsh r0, [AO]
	ldrsh r1, [BO]

	mul r2, r0, r1
	vmov r3, s16
	add r3,r3,r2
	vmov s16, r3
	
	ldrsh r1, [BO, #2]
	mul r2, r0, r1
	vmov r3, s20
	add r3,r3,r2
	vmov s20, r3
	
	pop {r0-r3}	
	
	add	AO , AO, #2
	add	BO , BO, #4
.endm

.macro SAVE1x2
	ldr	r3  , LDC
	add	CO2 , CO1, r3

	flds		s0, ALPHA
	vcvt.F32.S32 s16,s16,#FRACTIONB
	flds	s8 , [CO1]
	fmacs	s8 , s0 , s16
	fsts	s8 , [CO1]
	
	vcvt.F32.S32 s20,s20,#FRACTIONB
	flds	s12, [CO2]
	fmacs	s12, s0 , s20
	fsts	s12, [CO2]

	add	CO1, CO1, #4
.endm

/******************************************************************************/

.macro INIT4x1
	veor q4, q4
.endm

.macro KERNEL4x1_SUB
	vld1.32 {d0}, [AO]
	
	ldrsh r4, [BO]
	vmov s8, r4
	
	vmlal.s16 q4, d0, d4[0]
	
	add BO, BO, #2
	add AO, AO, #8
.endm

.macro SAVE4x1
	flds		s0, ALPHA	
	fldmias CO1, { s8 - s11 }	

	vcvt.F32.S32 q4,q4,#FRACTIONB
	
	vmla.f32   q2,  q4,  d0[0]
	fstmias CO1, { s8 - s11 }

	add	CO1, CO1, #16
.endm

/******************************************************************************/

.macro INIT2x1
	veor d8, d8
.endm

.macro KERNEL2x1_SUB
	push {r0-r3}
	
	ldrsh r0, [BO]
	ldrsh r1, [AO]

	mul r2, r0, r1
	vmov r3, s16
	add r3,r3,r2
	vmov s16, r3
	
	ldrsh r1, [AO, #2]
	mul r2, r0, r1
	vmov r3, s17
	add r3,r3,r2
	vmov s17, r3
	
	pop {r0-r3}	
	
	add	AO , AO, #4
	add	BO , BO, #2	
.endm

.macro SAVE2x1
	flds		s0, ALPHA

	flds	s8 , [CO1]
	vcvt.F32.S32 d8,d8,#FRACTIONB
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
	push {r0-r1}
	
	ldrsh r0, [BO]
	ldrsh r1, [AO]

	mul r1, r1, r0	
	vmov r0, s16
	add r0,r0,r1
	vmov s16, r0	
	
	pop {r0-r1}
	
	add	AO , AO, #2
	add	BO , BO, #2
.endm

.macro SAVE1x1
	flds		s0, ALPHA

	flds	s8 , [CO1]
	vcvt.F32.S32 s16,s16,#FRACTIONB
	fmacs	s8 , s0 , s16
	fsts	s8 , [CO1]

	add	CO1, CO1, #4
.endm

.macro DO_M4X4MATS_NEON_I
	pld	[ AO , #A_PRE ]
	vld1.32         { q8 }, [AO]!
	
	pld	[ BO , #B_PRE ]
	vld1.32         { q0 }, [BO]!
	
	vmull.s16 q4, d16, d0[0]	
	vmull.s16 q5, d16, d0[1]
	vrev64.32 d0, d0
	vld1.32         { q9 }, [AO]!
	vmull.s16 q6, d16, d0[0]
	vmull.s16 q7, d16, d0[1]
	
	vmlal.s16 q4, d17, d1[0]	
	vmlal.s16 q5, d17, d1[1]	
	vrev64.32 d1, d1
	vld1.32         { q1 }, [BO]!
	vmlal.s16 q6, d17, d1[0]
	vmlal.s16 q7, d17, d1[1]

	vmlal.s16 q4, d18, d2[0]	
	vmlal.s16 q5, d18, d2[1]	
	vrev64.32 d2, d2
	vmlal.s16 q6, d18, d2[0]
	vmlal.s16 q7, d18, d2[1]
	
	vmlal.s16 q4, d19, d3[0]	
	vmlal.s16 q5, d19, d3[1]	
	vrev64.32 d3, d3
	vmlal.s16 q6, d19, d3[0]
	vmlal.s16 q7, d19, d3[1]
.endm

.macro DO_M4X4MATS_NEON_LOOP
	vld1.32    { q8 }, [AO]!
	vld1.32    { q0 }, [BO]!
	
	vmlal.s16 q4, d16, d0[0]	
	vmlal.s16 q5, d16, d0[1]
	vrev64.32 d0, d0
	vld1.32    { q9 }, [AO]!
	vmlal.s16 q6, d16, d0[0]
	vmlal.s16 q7, d16, d0[1]	
	pld	[ AO , #A_PRE ]
	vld1.32    { q1 }, [BO]!
	vmlal.s16 q4, d17, d1[0]
	vmlal.s16 q5, d17, d1[1]
	vrev64.32 d1, d1
	vmlal.s16 q6, d17, d1[0]
	vmlal.s16 q7, d17, d1[1]
	
	vmlal.s16 q4, d18, d2[0]
	vmlal.s16 q5, d18, d2[1]
	vrev64.32 d2, d2
	vmlal.s16 q6, d18, d2[0]
	vmlal.s16 q7, d18, d2[1]
	pld	[ BO , #B_PRE ]
	vmlal.s16 q4, d19, d3[0]	
	vmlal.s16 q5, d19, d3[1]
	vrev64.32 d3, d3
	vmlal.s16 q6, d19, d3[0]
	vmlal.s16 q7, d19, d3[1]		
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

/****************[4*4]***************/
sgemm_kernel_L4_M4_NEON:
	DO_M4X4MATS_NEON_I  //Q12 = q8,  d0[0], then Q12 += q8,  d0[0]

	subs L, L, #2
	cmp	L, #2
	bge sgemm_kernel_L4_M4_NEON_LOOP

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

	cmp	L, #0
	ble sgemm_kernel_L4_M4_44

	@ last
	KERNEL4x4_NEON_I
	KERNEL4x4_E

	subs L, L, #1
	b sgemm_kernel_L4_M4_44

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
	lsl	r4, r4, #3					// k * 4 * 4  fix change from 4 to 3
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
	lsl	r4, r4, #2					// k * 2 * 4 fix change from 3 to 2
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

