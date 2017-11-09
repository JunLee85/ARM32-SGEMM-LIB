#define SRC 	r0
#define DST 	r1
#define WIDTH 	r2
#define HEIGHT 	r3

#define SRC_COPY_WIDTH r4
#define ROW_LEFT_DWORDS r4

#define SRC_COPY_HEIGHT r5
#define LEFT_ROWS r5

#define DST_PITCH r6

#define SRC_ADDR r0
#define DST_ADDR r1
#define DST_ADDR_2 r7
#define DST_ADDR_3 r8

#define HOR_OFFSET r9

#define OLD_SRC 				[fp,#-4]
#define OLD_DST 				[fp,#-8]
#define OLD_SRC_COPY_WIDTH      [fp,#-12]
	
	.equ      VERSION_MAJOR,    1
	.equ      VERSION_MINOR,    0
	.equ      VERSION_REVISION, 0

	.equ      PHASE,            1
	.equ      COPYRIGHT_YEAR,   2017

COPYRIGHT_HOLDER:
    .asciz              		"tianylijun@163.com"
    .equ      IM2COL_OK,        0
    .equ      IM2COL_ERR,      -1

#include "im2col_k3x3_copy.code"

/*********************************************************************************************
* This function implements the im2col function for kernel size 3*3, pad 0, stride 1
* int im2col_k3x3_acc(float *pSrc, float *pDst, unsigned int width, unsigned int height)
* @param[in]  *pSrc            points to the output pointers
* @param[in]  *pDst            points to input pointers
* @param[in]  width            width of input buffer
* @param[in]  height           height of input buffer
* [fp, #-128] - [fp, #-256] is reserved for store and restore of floating point registers
**********************************************************************************************/
	.align 5
	.section .text
	.global im2col_k3x3_acc
	.type im2col_k3x3_acc, %function

im2col_k3x3_acc:
	push {r4-r9, fp}                    // r0-r3 is param registers
	mov	fp, sp
	sub	sp, sp, #320				    // reserve stack
	
	# store 4 params into stack			[fp-0] - [fp-64] is rsved
	str SRC, OLD_SRC					// [fp-4]
	str DST, OLD_DST					// [fp-8]
	
	# backup float registers into [fp-192] - [fp-320]
	sub	r4, fp, #192
	vstm r4, {s0-s31} 			        // store floating point registers
	
	# do params check
	DO_PARAM_CHECK
	
	# caculate SRC COPY PITCH & backup
	subs SRC_COPY_WIDTH, WIDTH, #2
	str SRC_COPY_WIDTH, OLD_SRC_COPY_WIDTH			   // [fp-12]
	
	subs SRC_COPY_HEIGHT, HEIGHT, #2
	
	# caculate DST PITCH
	mul DST_PITCH, SRC_COPY_WIDTH, SRC_COPY_HEIGHT     // DST_PITCH r6
	lsl DST_PITCH, DST_PITCH, #2
	
	# clear 0
	eor HOR_OFFSET, HOR_OFFSET

VERTICAL_LOOP_START:

/***************************************************************************/
# copy first row into dst
FIRST_ROW_LOOP_COPY_16_BYTES:
	# ROW_LEFT_DWORDS is nick name of SRC_COPY_WIDTH
	cmp ROW_LEFT_DWORDS, #4    
	bge COPY_16_BYTES

FIRST_ROW_LOOP_COPY_8_BYTES:
	cmp ROW_LEFT_DWORDS, #2
	bge COPY_8_BYTES

FIRST_ROW_LOOP_COPY_4_BYTES:
	cmp ROW_LEFT_DWORDS, #0
	bgt DO_COPY_1_DWORDS
    
	# ROW COPY FINISHED LEFT_ROWS IS NICK NAME OF SRC_COPY_HEIGHT
	subs LEFT_ROWS, LEFT_ROWS, #1
	cmp LEFT_ROWS, #1
	bge SECOND_ROW_COPY_TWICE_START
	
	#copy last but one row
	LBO_ROWS_COPY_FIRST
	#copy last row
	LAST_ROWS_COPY_FIRST
	
DO_COPY_1_DWORDS:
	COPY_1_DWORDS_1
	b FIRST_ROW_LOOP_COPY_4_BYTES
	
COPY_8_BYTES:
	COPY_2_DWORDS_1
	b FIRST_ROW_LOOP_COPY_8_BYTES
	
COPY_16_BYTES:
	COPY_4_DWORDS_1
	b FIRST_ROW_LOOP_COPY_16_BYTES

	
/***************************************************************************/
# SECOND ROW START
SECOND_ROW_COPY_TWICE_START:	
	# revocer ROW_LEFT_DWORDS
	ldr ROW_LEFT_DWORDS, OLD_SRC_COPY_WIDTH
	
	# start src second row copy
	add SRC_ADDR, SRC_ADDR, #8

	# updata DST_ADDR_2, DST_ADDR_3
	ldr DST_ADDR_2, OLD_DST
	push {r9}
	asrs r9, HOR_OFFSET, #2
	mul r9, DST_PITCH, r9
	add DST_ADDR_2, DST_ADDR_2, r9

	mov r9, #3
	mul r9, DST_PITCH, r9
	add DST_ADDR_2, DST_ADDR_2, r9
	add DST_ADDR_3, DST_ADDR_2, r9
	pop {r9}

# copy second row
SECOND_ROW_LOOP_COPY_16_BYTES:
	cmp ROW_LEFT_DWORDS, #4
	bge COPY_16_BYTES_TWICE
	
SECOND_ROW_LOOP_COPY_8_BYTES:
	cmp ROW_LEFT_DWORDS, #2
	bge COPY_8_BYTES_TWICE

SECOND_ROW_LOOP_COPY_4_BYTES:
	cmp ROW_LEFT_DWORDS, #0
	bgt DO_COPY_1_DWORDS_TWICE
	
	subs LEFT_ROWS, LEFT_ROWS, #1
	cmp LEFT_ROWS, #1
	bge THIRED_ROW_START
	
	#copy last but one row
	LBO_ROWS_COPY_SECOND
	#copy last row
	LAST_ROWS_COPY_SECOND

DO_COPY_1_DWORDS_TWICE:
	COPY_1_DWORDS_1_2
	b SECOND_ROW_LOOP_COPY_4_BYTES
	
COPY_8_BYTES_TWICE:
	COPY_2_DWORDS_1_2
	b SECOND_ROW_LOOP_COPY_8_BYTES
	
COPY_16_BYTES_TWICE:
	COPY_4_DWORDS_1_2
	b SECOND_ROW_LOOP_COPY_16_BYTES

/***************************************************************************/
# THIRD ROW START
THIRED_ROW_START:

# LOOP THIRD ROW START
LOOP_THIRED_ROW_START:
	# revocer ROW_LEFT_DWORDS
	ldr ROW_LEFT_DWORDS, OLD_SRC_COPY_WIDTH

	# start src third row copy
	add SRC_ADDR, SRC_ADDR, #8

THIRD_ROW_LOOP_COPY_64_BYTES:
	cmp ROW_LEFT_DWORDS, #16
	bge COPY_64_BYTES_TREBLE

THIRD_ROW_LOOP_COPY_32_BYTES:
	cmp ROW_LEFT_DWORDS, #8
	bge COPY_32_BYTES_TREBLE

# copy third row
THIRD_ROW_LOOP_COPY_16_BYTES:
	cmp ROW_LEFT_DWORDS, #4
	bge COPY_16_BYTES_TREBLE
	
THIRD_ROW_LOOP_COPY_8_BYTES:
	cmp ROW_LEFT_DWORDS, #2
	bge COPY_8_BYTES_TREBLE

THIRD_ROW_LOOP_COPY_4_BYTES:
	cmp ROW_LEFT_DWORDS, #0
	bgt DO_COPY_1_DWORDS_THREE_TIMES
	
	subs LEFT_ROWS, LEFT_ROWS, #1
	cmp LEFT_ROWS, #0
	bgt LOOP_THIRED_ROW_START
	
	#copy last but one row
	LBO_ROWS_COPY_THIRD
	#copy last row
	LAST_ROWS_COPY_THIRD	
	
DO_COPY_1_DWORDS_THREE_TIMES:
	COPY_1_DWORDS_1_2_3
	b THIRD_ROW_LOOP_COPY_4_BYTES
	
COPY_8_BYTES_TREBLE:
	COPY_2_DWORDS_1_2_3
	b THIRD_ROW_LOOP_COPY_8_BYTES
	
COPY_16_BYTES_TREBLE:
	COPY_4_DWORDS_1_2_3
	b THIRD_ROW_LOOP_COPY_16_BYTES

COPY_32_BYTES_TREBLE:
	COPY_8_DWORDS_1_2_3
	b THIRD_ROW_LOOP_COPY_32_BYTES
	
COPY_64_BYTES_TREBLE:
	COPY_16_DWORDS_1_2_3
	b THIRD_ROW_LOOP_COPY_64_BYTES

# ONE VERTICAL FINISHED	
DO_VERITCAL_ITERATION_END:
	add HOR_OFFSET, HOR_OFFSET, #4

	cmp HOR_OFFSET, #12
	bge IM2COL_END

	# check for hor iteration
	ldr SRC_ADDR, OLD_SRC
	add SRC_ADDR, SRC_ADDR, HOR_OFFSET

	# recove ROW_LEFT_DWORDS
	ldr ROW_LEFT_DWORDS, OLD_SRC_COPY_WIDTH

	# recove LEFT_ROWS (SRC_COPY_HEIGHT)
	subs SRC_COPY_HEIGHT, HEIGHT, #2

	# updata DST_ADDR
	ldr DST_ADDR, OLD_DST

	push {HEIGHT}
	asrs r3, HOR_OFFSET, #2
	mul r3, DST_PITCH, r3
	add DST_ADDR, DST_ADDR, r3
	pop {HEIGHT}

	b VERTICAL_LOOP_START

IM2COL_END:
	sub	r4, fp, #192
	vldm r4, {s0-s31}		    	// restore floating point registers
	mov	r0, #0						// set return value

	mov	sp, fp
	pop	{r4-r9, fp}
	bx	lr
