SRCFILE := $(wildcard ./*.c)
SRCFILE += $(wildcard ./interface/*.c)
SRCFILE += $(wildcard ./arm/*.c)
SRCFILE += $(wildcard ./arm/*.S)

OBJFILE := $(SRCFILE)

CFLAG 	:= $(DEBUG_FLAG) $(FIXD) -I. -I../serial -I../inc -marm -mfloat-abi=hard -march=armv7-a -mfpu=neon -Wall -fPIC -fmessage-length=0 \
					 -DMAX_STACK_ALLOC=2048 -DF_INTERFACE_GFORT -DNO_LAPACK -DNO_LAPACKE -DSMP_SERVER -DNO_WARMUP -DTHREADED_LEVEL3 -DTT -DNO_AFFINITY -DMAX_CPU_NUMBER=4 -UDOUBLE -UCOMPLEX

ASMFLAG := -c $(DEBUG_FLAG) -I. -I../serial -I../inc  -marm -mfloat-abi=hard -march=armv7-a -mfpu=neon -Wall -fPIC -fmessage-length=0 \
					 -DMAX_STACK_ALLOC=2048 -DARCH_ARM -DF_INTERFACE_GFORT -DNO_LAPACK -DNO_LAPACKE \
					 -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=4 -DNO_AFFINITY -UDOUBLE -UCOMPLEX -mfpu=neon

ifeq ($(PLATFORM), android)
CFLAG 	+= -DOS_ANDROID=1
ASMFLAG += -DOS_ANDROID=1
else
CFLAG 	+= -DOS_LINUX
ASMFLAG += -DOS_LINUX
endif

AllObjFile = $(SRCFILE:.c=.o)

%.o: %.c
	$(CC) $(CFLAG) -o $@ -c $<
	
all:$(AllObjFile)

	$(CC) $(ASMFLAG) -o cblas_sgemv.o ./gemv/gemv.c
	$(CC) $(ASMFLAG) -UTRANSA -UCONJ -UXCONJ -DASMNAME=sgemv_thread_n -DASMFNAME=sgemv_thread_n_ -DNAME=sgemv_thread_n_ -DCNAME=sgemv_thread_n -DCHAR_NAME=\"sgemv_thread_n_\" -DCHAR_CNAME=\"sgemv_thread_n\" ./gemv/gemv_thread.c -o sgemv_thread_n.o
	$(CC) $(ASMFLAG) -DTRANSA -UCONJ -UXCONJ -DASMNAME=sgemv_thread_t -DASMFNAME=sgemv_thread_t_ -DNAME=sgemv_thread_t_ -DCNAME=sgemv_thread_t -DCHAR_NAME=\"sgemv_thread_t_\" -DCHAR_CNAME=\"sgemv_thread_t\" ./gemv/gemv_thread.c -o sgemv_thread_t.o

	$(CC) $(ASMFLAG) -DFRACTION=12 -DASMNAME=sgemm_otcopy_fix12 -DASMFNAME=sgemm_otcopy_fix12_ -DNAME=sgemm_otcopy_fix12_ -DCNAME=sgemm_otcopy_fix12 -DCHAR_NAME=\"sgemm_otcopy_fix12_\" -DCHAR_CNAME=\"sgemm_otcopy_fix12\" ./arm/sgemm_tcopy_4_vfp_fix16.S -o sgemm_otcopy_fix12.o 	
	$(CC) $(ASMFLAG) -DFRACTION=12 -DASMNAME=sgemm_kernel_fix12 -DASMFNAME=sgemm_kernel_fix12_ -DNAME=sgemm_kernel_fix12_ -DCNAME=sgemm_kernel_fix12 -DCHAR_NAME=\"sgemm_kernel_fix12_\" -DCHAR_CNAME=\"sgemm_kernel_fix12\" ./arm/sgemm_kernel_4x4_vfpv3_fix16.S -I./src -o sgemm_kernel_fix12.o	
	$(CC) $(ASMFLAG) -DFRACTION=12 -DASMNAME=sgemm_oncopy_fix12 -DASMFNAME=sgemm_oncopy_fix12_ -DNAME=sgemm_oncopy_fix12_ -DCNAME=sgemm_oncopy_fix12 -DCHAR_NAME=\"sgemm_oncopy_fix12_\" -DCHAR_CNAME=\"sgemm_oncopy_fix12\" ./arm/sgemm_ncopy_4_vfp_fix16.S -o sgemm_oncopy_fix12.o 
	
	$(CC) $(ASMFLAG) -DFRACTION=13 -DASMNAME=sgemm_otcopy_fix13 -DASMFNAME=sgemm_otcopy_fix13_ -DNAME=sgemm_otcopy_fix13_ -DCNAME=sgemm_otcopy_fix13 -DCHAR_NAME=\"sgemm_otcopy_fix13_\" -DCHAR_CNAME=\"sgemm_otcopy_fix13\" ./arm/sgemm_tcopy_4_vfp_fix16.S -o sgemm_otcopy_fix13.o 	
	$(CC) $(ASMFLAG) -DFRACTION=13 -DASMNAME=sgemm_kernel_fix13 -DASMFNAME=sgemm_kernel_fix13_ -DNAME=sgemm_kernel_fix13_ -DCNAME=sgemm_kernel_fix13 -DCHAR_NAME=\"sgemm_kernel_fix13_\" -DCHAR_CNAME=\"sgemm_kernel_fix13\" ./arm/sgemm_kernel_4x4_vfpv3_fix16.S -I./src -o sgemm_kernel_fix13.o	
	$(CC) $(ASMFLAG) -DFRACTION=13 -DASMNAME=sgemm_oncopy_fix13 -DASMFNAME=sgemm_oncopy_fix13_ -DNAME=sgemm_oncopy_fix13_ -DCNAME=sgemm_oncopy_fix13 -DCHAR_NAME=\"sgemm_oncopy_fix13_\" -DCHAR_CNAME=\"sgemm_oncopy_fix13\" ./arm/sgemm_ncopy_4_vfp_fix16.S -o sgemm_oncopy_fix13.o 

	$(CC) $(ASMFLAG) -DFRACTION=14 -DASMNAME=sgemm_otcopy_fix14 -DASMFNAME=sgemm_otcopy_fix14_ -DNAME=sgemm_otcopy_fix14_ -DCNAME=sgemm_otcopy_fix14 -DCHAR_NAME=\"sgemm_otcopy_fix14_\" -DCHAR_CNAME=\"sgemm_otcopy_fix14\" ./arm/sgemm_tcopy_4_vfp_fix16.S -o sgemm_otcopy_fix14.o 	
	$(CC) $(ASMFLAG) -DFRACTION=14 -DASMNAME=sgemm_kernel_fix14 -DASMFNAME=sgemm_kernel_fix14_ -DNAME=sgemm_kernel_fix14_ -DCNAME=sgemm_kernel_fix14 -DCHAR_NAME=\"sgemm_kernel_fix14_\" -DCHAR_CNAME=\"sgemm_kernel_fix14\" ./arm/sgemm_kernel_4x4_vfpv3_fix16.S -I./src -o sgemm_kernel_fix14.o	
	$(CC) $(ASMFLAG) -DFRACTION=14 -DASMNAME=sgemm_oncopy_fix14 -DASMFNAME=sgemm_oncopy_fix14_ -DNAME=sgemm_oncopy_fix14_ -DCNAME=sgemm_oncopy_fix14 -DCHAR_NAME=\"sgemm_oncopy_fix14_\" -DCHAR_CNAME=\"sgemm_oncopy_fix14\" ./arm/sgemm_ncopy_4_vfp_fix16.S -o sgemm_oncopy_fix14.o 

	$(CC) $(ASMFLAG) -DFRACTION=15 -DASMNAME=sgemm_otcopy_fix15 -DASMFNAME=sgemm_otcopy_fix15_ -DNAME=sgemm_otcopy_fix15_ -DCNAME=sgemm_otcopy_fix15 -DCHAR_NAME=\"sgemm_otcopy_fix15_\" -DCHAR_CNAME=\"sgemm_otcopy_fix15\" ./arm/sgemm_tcopy_4_vfp_fix16.S -o sgemm_otcopy_fix15.o 	
	$(CC) $(ASMFLAG) -DFRACTION=15 -DASMNAME=sgemm_kernel_fix15 -DASMFNAME=sgemm_kernel_fix15_ -DNAME=sgemm_kernel_fix15_ -DCNAME=sgemm_kernel_fix15 -DCHAR_NAME=\"sgemm_kernel_fix15_\" -DCHAR_CNAME=\"sgemm_kernel_fix15\" ./arm/sgemm_kernel_4x4_vfpv3_fix16.S -I./src -o sgemm_kernel_fix15.o	
	$(CC) $(ASMFLAG) -DFRACTION=15 -DASMNAME=sgemm_oncopy_fix15 -DASMFNAME=sgemm_oncopy_fix15_ -DNAME=sgemm_oncopy_fix15_ -DCNAME=sgemm_oncopy_fix15 -DCHAR_NAME=\"sgemm_oncopy_fix15_\" -DCHAR_CNAME=\"sgemm_oncopy_fix15\" ./arm/sgemm_ncopy_4_vfp_fix16.S -o sgemm_oncopy_fix15.o 
	
	$(CC) $(ASMFLAG) -DASMNAME=sgemm_otcopy -DASMFNAME=sgemm_otcopy_ -DNAME=sgemm_otcopy_ -DCNAME=sgemm_otcopy -DCHAR_NAME=\"sgemm_otcopy_\" -DCHAR_CNAME=\"sgemm_otcopy\" ./arm/sgemm_tcopy_4_vfp_acc.S -o sgemm_otcopy.o 	
	$(CC) $(ASMFLAG) -DASMNAME=sgemm_kernel -DASMFNAME=sgemm_kernel_ -DNAME=sgemm_kernel_ -DCNAME=sgemm_kernel -DCHAR_NAME=\"sgemm_kernel_\" -DCHAR_CNAME=\"sgemm_kernel\" ./arm/sgemm_kernel_4x4_vfpv3_acc.S -I./src -o sgemm_kernel.o	
	$(CC) $(ASMFLAG) -DASMNAME=sgemm_oncopy -DASMFNAME=sgemm_oncopy_ -DNAME=sgemm_oncopy_ -DCNAME=sgemm_oncopy -DCHAR_NAME=\"sgemm_oncopy_\" -DCHAR_CNAME=\"sgemm_oncopy\" ./arm/sgemm_ncopy_4_vfp_acc.S -o sgemm_oncopy.o 

	$(CC) $(ASMFLAG) -DTRANS -DASMNAME=sgemv_t -DASMFNAME=sgemv_t_ -DNAME=sgemv_t_ -DCNAME=sgemv_t -DCHAR_NAME=\"sgemv_t_\" -DCHAR_CNAME=\"sgemv_t\" ./arm/gemv_t_vfp.S -o sgemv_t.o 
	$(CC) $(ASMFLAG) -UTRANS -DASMNAME=sgemv_n -DASMFNAME=sgemv_n_ -DNAME=sgemv_n_ -DCNAME=sgemv_n -DCHAR_NAME=\"sgemv_n_\" -DCHAR_CNAME=\"sgemv_n\" ./arm/gemv_n_vfpv3.S -o sgemv_n.o  
	$(CC) $(ASMFLAG) -DASMNAME=sasum_k -DASMFNAME=sasum_k_ -DNAME=sasum_k_ -DCNAME=sasum_k -DCHAR_NAME=\"sasum_k_\" -DCHAR_CNAME=\"sasum_k\" ./arm/asum_vfp.S -o sasum_k.o
	$(CC) $(ASMFLAG) -DASMNAME=saxpy_k -DASMFNAME=saxpy_k_ -DNAME=saxpy_k_ -DCNAME=saxpy_k -DCHAR_NAME=\"saxpy_k_\" -DCHAR_CNAME=\"saxpy_k\" ./arm/axpy_vfp.S -o saxpy_k.o
	
	$(AR) rcs ../lib/$(LIBFNAME) *.o ./arm/*.o ./interface/*.o

	@rm -f *.o
	@rm -f ./arm/*.o
	@rm -f ./interface/*.o
	
clean:
	@rm -f *.o
	@rm -f ./arm/*.o
	@rm -f ./interface/*.o

