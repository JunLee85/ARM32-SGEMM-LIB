# ARM32-SGEMM-LIB
a fast sgemm lib with neon & fix16 optimization on arm 32 based on openblas

How to build on android
1. make android

How to build test on android
1. cd test/jni
2. ndk-build

How to build on linux
1. make linux

How to build test on linux
1. make linux

How to use fix16

void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
								 blasint m, blasint n, blasint k,
								 float alpha,
								 float *a, blasint lda,
								 float *b, blasint ldb,
								 float beta,
								 float *c, blasint ldc,
								 int fractions) ;

fractions: the last param of this func is the fractions you want to use.
1. 0 mean use float32 not use fix16
2. other than 0 mean use fix16, current only fraction 12, 13, 14, 15 support. Meanwhile you can change the src/Makefile to add support other fraction
(user no need change the input data into fix16, the lib will change the float32 data into fix16 to do the inner compute, and finnal output float32 result)
