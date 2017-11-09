#ifndef OPENBLAS_CONFIG_H
#define OPENBLAS_CONFIG_H

#define OPENBLAS___64BIT__ 1
#define OPENBLAS_BUNDERSCORE _
#define OPENBLAS_NEEDBUNDERSCORE 1

#define OPENBLAS_NUM_SHAREDCACHE 1
#define OPENBLAS_NUM_CORES 1

#ifdef OPENBLAS_NEEDBUNDERSCORE
#define BLASFUNC(FUNC) FUNC##_
#else
#define BLASFUNC(FUNC) FUNC
#endif

#define xdouble double

typedef long BLASLONG;
typedef unsigned long BLASULONG;

#define FLOATRET	float

#define CBLAS_INDEX size_t

#include <stdio.h>

#endif /* OPENBLAS_CONFIG_H */
