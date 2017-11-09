BLAS_ROOT := $(shell pwd)
BLAS_INC  := $(BLAS_ROOT)/inc

export BLAS_ROOT BLAS_INC

.PHONY: all clean android linux 

all:
	make -f Makefile_linux
	make -f Makefile_android

linux:
	make -f Makefile_linux

android:
	make -f Makefile_android
	
clean:
	make -f Makefile_linux clean
	make -f Makefile_android clean
	@rm -f *.a
	
