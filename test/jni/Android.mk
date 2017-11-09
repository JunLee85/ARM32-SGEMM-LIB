LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := sgemm-lib
LOCAL_SRC_FILES := ../../lib/libsgemm_android.a
include $(PREBUILT_STATIC_LIBRARY) 

include $(CLEAR_VARS)
LOCAL_MODULE    := sgemm-test-android
LOCAL_SRC_FILES := ../sgemm-test.cpp

LOCAL_C_INCLUDES := ../../inc 
LOCAL_STATIC_LIBRARIES := sgemm-lib
LOCAL_CPPFLAGS := -DANDROID -fpic -std=c++11 -frtti -fomit-frame-pointer -fdata-sections -mthumb -fno-exceptions -O3 -march=armv7-a -mfloat-abi=hard --fast-math -mfpu=neon-vfpv4
LOCAL_LDFLAGS := -Wl,--no-warn-mismatch -lm_hard -mhard-float -D_NDK_MATH_NO_SOFTFP=1
include $(BUILD_EXECUTABLE)
