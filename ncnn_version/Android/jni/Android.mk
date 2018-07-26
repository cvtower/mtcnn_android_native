LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
OPENCV_LIB_TYPE:=STATIC  
include D:\sw\opencv-3.4.0-android-sdk\OpenCV-android-sdk/sdk/native/jni/OpenCV.mk 
LOCAL_MODULE     := mtcnn_baseline
LOCAL_SRC_FILES  := ./../../mtcnn/mtcnn.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)/include

LOCAL_LDLIBS     += -llog -ldl -lm -lncnn -fopenmp -L$(LOCAL_PATH)/lib
#LOCAL_LDLIBS += -fuse-ld=gold

LOCAL_CFLAGS += -O3 -fexceptions -fPIE -D__ARM_NEON__   -mfpu=neon -mfloat-abi=softfp -mvectorize-with-neon-quad -fPIE
LOCAL_LDFLAGS += -Wl,--fix-cortex-a8 -fPIE -pie -Wno-unused-parameter
LOCAL_CXXFLAGS :=  -std=c++11 -D__ARM_NEON__   -mfpu=neon -mfloat-abi=softfp -mvectorize-with-neon-quad -fPIE -O3 -DNDEBUG

LOCAL_CFLAGS +:= -O3 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS := -O3 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_LDFLAGS += -Wl,--gc-sections

LOCAL_CPPFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp

LOCAL_CXXFLAGS +:= -fopenmp
LOCAL_CFLAGS += -fPIE -fopenmp
LOCAL_LDFLAGS += -fPIE -pie
#LOCAL_ALLOW_UNDEFINED_SYMBOLS := true

include $(BUILD_EXECUTABLE)