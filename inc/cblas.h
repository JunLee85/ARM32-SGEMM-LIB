#ifndef CBLAS_H
#define CBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OPENBLAS_CONST
#define OPENBLAS_CONST 
#endif

void openblas_set_num_threads(int num_threads);
void goto_set_num_threads(int num_threads);
int openblas_get_num_threads(void);
int openblas_get_num_procs(void);
char* openblas_get_config(void);
char* openblas_get_corename(void);
int openblas_get_parallel(void);

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef int blasint;

void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
								 blasint m, blasint n, blasint k,
								 float alpha,
								 float *a, blasint lda,
								 float *b, blasint ldb,
								 float beta,
								 float *c, blasint ldc,
								 int fractions) ;

void cblas_sgemv(enum CBLAS_ORDER order,
								 enum CBLAS_TRANSPOSE TransA,
								 blasint m, blasint n,
								 float alpha,
								 float  *a, blasint lda,
								 float  *x, blasint incx,
								 float beta,
								 float  *y, blasint incy);

void cblas_scopy(blasint n, 
								 float *x, 
								 blasint incx, 
								 float *y, blasint incy);

float cblas_sasum(blasint n, float *x, blasint incx);
void cblas_cscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float *alpha, float *X, OPENBLAS_CONST blasint incX);
void cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
#ifdef __cplusplus
}
#endif

#endif
