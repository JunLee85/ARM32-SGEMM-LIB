#ifdef OS_ANDROID
//#define OS_ANDROID	1
#else
#define OS_LINUX	1	
#endif

#define ARCH_ARM	1
#define C_GCC	1
#define __32BIT__	1
#define PTHREAD_CREATE_FUNC	pthread_create
#define BUNDERSCORE	_
#define NEEDBUNDERSCORE	1
#define ARMV7
#define L1_DATA_SIZE 65536
#define L1_DATA_LINESIZE 32
#define L2_SIZE 512488
#define L2_LINESIZE 32
#define DTB_DEFAULT_ENTRIES 64
#define DTB_SIZE 4096
#define L2_ASSOCIATIVE 4
#define HAVE_VFPV3
#define HAVE_VFP
#define CORE_ARMV7
#define CHAR_CORENAME "ARMV7"
#define GEMM_MULTITHREAD_THRESHOLD	4
