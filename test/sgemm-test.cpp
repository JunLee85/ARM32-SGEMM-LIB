#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "cblas.h"

#define TYPE "float"
#ifdef FIX_ENABLE
#undef TYPE
#define TYPE "fix"
#endif

#define FLOAT2FIX(fixt, fracbits, x) fixt(((x)*(float)((fixt(1)<<(fracbits)))))
#define FIX2FLOAT(fracbits,x) ((float)(x)/((1)<<fracbits))

typedef short int fix_t;

static float *A;
static float *B;
static float *C;
static float *C_REF;
static float *C_ASM;

static fix_t *fixA;
static fix_t *fixB;
static int   *fixC;
static float *fixFloatC;


static void matricMul(int M, int N, int K, float*a, float*b, float *c) {
	int m, n, k;

	float maxA = 0.0f;
	float maxB = 0.0f;
	float maxC = 0.0f;
	float maxSum = 0.0f;
	float maxMulValue = 0.0f;

	float mul = 0.0f;

	for (m = 0; m < M; m++)
		for (n = 0; n < N; n++) {
			float sum = 0.0f;

			if (fabs(*(c + m * N + n)) > maxC)
				maxC = fabs(*(c + m * N + n));

			for (k = 0; k < K; k++) {
				if (fabs(*(a + m * K + k)) > maxA)
					maxA = fabs(*(a + m * K + k));
				if (fabs(*(b + k * N + n)) > maxB)
					maxB = fabs(*(b + k * N + n));

				mul = *(a + m * K + k) * *(b + k * N + n);

				if (fabs(mul) > maxMulValue)
					maxMulValue = fabs(mul);

				sum += mul;
			}

			if (fabs(sum) > maxSum)
				maxSum = fabs(sum);

			*(c + m * N + n) = sum;
		}

	printf("maxMul %f, maxA: %f, maxB:%f, maxC:%f, maxSum:%f\n",
					maxMulValue, maxA, maxB, maxC, maxSum);
}

static void matricMulFix(int M, int N, int K, fix_t*a, fix_t*b, int*c) {
	int m, n, k;
	for (m = 0; m < M; m++)
		for (n = 0; n < N; n++) {
			int sum = 0;
			for (k = 0; k < K; k++)
				sum += (int) (*(a + m * K + k)) * (int) (*(b + k * N + n));
			*(c + m * N + n) = sum;
		}
}

int main(int argc, char* argv[]) {
	int i, j, loop;
	int m = 512;
	int n = 30;
	int k = 4608;
	int flag = 0;
	int loopcnt = 1;
	int fractions = 0;
	struct timeval tv_s, tv_e;
	float total_time = .0f;
	long ttime = 0;

	printf("e.g. : ./sgemm-test-linux m n k print-flag loop-cnt fractions\n");

	if (argc > 1)
		m = atoi(argv[1]);
	if (argc > 2)
		n = atoi(argv[2]);
	if (argc > 3)
		k = atoi(argv[3]);
	if (argc > 4)
		flag = atoi(argv[4]);
	if (argc > 5)
		loopcnt = atoi(argv[5]);
	if (argc > 6)
		fractions = atoi(argv[6]);

	printf("MNK [%d %d %d], print flag: %d, loop cnt: %d fractions:%d\n", m, n, k, flag,	loopcnt, fractions);
	
	if (((12 > fractions) || (15 < fractions)) && (0 != fractions))
	{
		printf("ERROR: Only fractions 12 - 15 supported, cur is  %d\n", fractions);
		return -1;
	}

	A = (float *)malloc(sizeof(float)*(m * k + k * n + 3 * m * n) +
						sizeof(fix_t) * (m * k + k * n ) +
						sizeof(int) * m * n +
						sizeof(float) * m * n );
	memset(A, 0, sizeof(float)*(m * k + k * n + 3 * m * n) +
				 sizeof(fix_t) * (m * k + k * n ) +
				 sizeof(int) * m * n +
				 sizeof(float) * m * n);
	B = A + m * k;
	C = B + k * n;
	C_REF = C + m * n;
	C_ASM = C_REF + m * n;

	if (0 != fractions)
	{
		fixA = (fix_t *)(C_ASM + m * n);
		fixB = fixA + m * k;
		fixC = (int *)(fixB + k * n);
		fixFloatC = (float *)(fixC + m * n);
	}

	for (i = 0; i < m * k; i++) A[i] = (rand()%800)/10000.0f;
	printf("==================floatA init ok================\n");
	if (0 != flag)
	{
		printf("==================floatA================\n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < k; j++) {
				printf("%.3f ", A[i * k + j]);
			}
			printf("\n");
		}
		if (0 != fractions)
			printf("==================fixA================\n");
	}

	if (0 != fractions)
	{
		for (i = 0; i < m; i++) {
			for (j = 0; j < k; j++) {
				fix_t fix_a = FLOAT2FIX((fix_t), fractions, A[i * k + j]);
				fixA[i * k + j] = fix_a;
				if (0 != flag)
					printf("%04x ", fix_a);
			}
			if (0 != flag)
				printf("\n");
		}
		if (0 != flag)
			printf("\n");

		printf("==================fixA ok================\n");
	}

	for (i = 0; i < k * n; i++) B[i] = (rand()%800)/10000.0f;
	printf("==================floatB init ok================\n");
	if (0 != flag)
	{
		printf("==================floatB================\n");
		for (i = 0; i < k; i++) {
			for (j = 0; j < n; j++) {
				printf("%.3f ", B[i * n + j]);
			}
			printf("\n");
		}
		if (0 != fractions)
			printf("==================fixB================\n");
	}

	if (0 != fractions)
	{
		for (i = 0; i < k; i++) {
			for (j = 0; j < n; j++) {
				fix_t fix_a = FLOAT2FIX((fix_t), fractions, B[i * n + j]);
				fixB[i * n + j] = fix_a;
				if (0 != flag)
					printf("%04x ", fix_a);
			}
			if (0 != flag)
				printf("\n");
		}
		if (0 != flag)
			printf("\n");
		printf("==================fixB ok================\n");
	}

	matricMul(m, n, k, A, B, C_REF);
	if (0 != flag)
	{
		printf("\n==================c-ref floatA*floatB================\n");
		for (i = 0; (0 != flag) && i < m; i++) {
			for (j = 0; j < n; j++)
				printf(" %f", *(C_REF + i * n + j));
			printf("\n");
		}
	}
	printf("==================floatA*floatB ok================\n");

	if (0 != fractions)
	{
		printf("==================c-ref fixA*fixB================\n");
		matricMulFix(m, n, k, fixA, fixB, fixC);
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++)
			{
				fixFloatC[i*n+j] = FIX2FLOAT(2*fractions, *(fixC + i * n + j));
				if (0 != flag)
					printf(" %f", fixFloatC[i*n+j]);
			}
			if (0 != flag)
				printf("\n");
		}
		printf("==================c-ref fixA*fixB ok================\n");

		printf("==================start blas fixA*fixB================\n");
	}
	else
		printf("==================start blas A*B================\n");

	for (loop = 1; loop <= loopcnt; loop++) {
		gettimeofday(&tv_s, NULL);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k,
								B, n, 0, C_ASM, n, fractions);
		gettimeofday(&tv_e, NULL);
		total_time += (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) / 1000.0f;
		printf("[%d/%d end]\n", loop, loopcnt);
	}	

	if (0 != fractions)
		printf("==================end blas fixA*fixB================\n");
	else
		printf("==================end blas A*B================\n");
	
	int sameFlag = 1;
	
	if (0 != fractions)
	{
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++)
			{
				if (fabs(*(fixFloatC + i * n + j) - *(C_ASM + i * n + j))/fabs(*(fixFloatC + i * n + j)) > 0.1f)
				{
					sameFlag = 0;
					printf("asm diff with fix ref [%f != %f] diff [%f > %f] [%x != %x] at (%d, %d)\n",
									*(fixFloatC + i * n + j), *(C_ASM + i * n + j),
									fabs(*(fixFloatC + i * n + j) - *(C_ASM + i * n + j)),
									FIX2FLOAT(fractions,1),
									*(unsigned int *)(fixFloatC + i * n + j), *(unsigned int *)(C_ASM + i * n + j),
									i, j);
					break;
				}
			}
			if (0 == sameFlag)
				break;
		}
	}
	else
	{
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++)
			{
				if (fabs(*(C_REF + i * n + j) - *(C_ASM + i * n + j)) > FIX2FLOAT(fractions, 1)/*FLT_MIN*/)
				{
					sameFlag = 0;
					printf("asm diff with ref [%f != %f] at (%d, %d)\n",
									*(C_REF + i * n + j), *(C_ASM + i * n + j),
									i, j);
					break;
				}
			}
			if (0 == sameFlag)
				break;
		}
	}

	printf("==================compare %s================\n",((1==sameFlag)?"same":"diff"));
	
	if (1 == sameFlag)
	{
		if (0 != fractions)
		{
			float maxDiff = .0f;
			float maxDiffRefC =.0f, maxDiffAsm = .0f;
			float maxDiffRatio = .0f;
			float maxDiffRefCRatio =.0f, maxDiffAsmRatio = .0f;
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++)
				{
					if (fabs(*(C_REF + i * n + j) - *(C_ASM + i * n + j)) > maxDiff)
					{
						maxDiffRefC = *(C_REF + i * n + j);
						maxDiffAsm = *(C_ASM + i * n + j);
						maxDiff = fabs(*(C_REF + i * n + j) - *(C_ASM + i * n + j));
					}
					if ((fabs(*(C_REF + i * n + j) - *(C_ASM + i * n + j)) / fabs(*(C_REF + i * n + j))) > maxDiffRatio)
					{
						maxDiffRefCRatio = *(C_REF + i * n + j);
						maxDiffAsmRatio = *(C_ASM + i * n + j);
						maxDiffRatio = fabs(*(C_REF + i * n + j) - *(C_ASM + i * n + j)) / fabs(*(C_REF + i * n + j));
					}
				}
			}
			printf("maxDiff abs   %f [%f, %f]\n        ratio %f [%f, %f]\n", maxDiff, maxDiffRefC, maxDiffAsm, maxDiffRatio, maxDiffRefCRatio, maxDiffAsmRatio);
		}
	}

	for (i = 0; (0 != flag) && i < m; i++) {
		for (j = 0; j < n; j++)
			printf(" %f", *(C_ASM + i * n + j));
		printf("\n");
	}

	printf("\n[%s] mean time: %.3f ms (total_time : %.3f, loopcnt:%d)\n",
					TYPE, total_time / loopcnt, total_time, loopcnt);
	free(A);
	return 0;
}
