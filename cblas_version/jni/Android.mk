LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
OPENCV_LIB_TYPE:=STATIC  
include D:\sw\opencv-3.4.0-android-sdk\OpenCV-android-sdk/sdk/native/jni/OpenCV.mk 
LOCAL_MODULE     := mtcnn_baseline
LOCAL_SRC_FILES  := ./../src/mtcnn.cpp  \
					./../src/network.cpp  \
					./../src/pBox.cpp  \
					./../src/pikaqiu.cpp
LOCAL_C_INCLUDES += ./../src

LOCAL_C_INCLUDES += $(LOCAL_PATH)/inc

LOCAL_LDLIBS     += -llog -ldl -lm -lblas -L$(LOCAL_PATH)/lib
#LOCAL_LDLIBS += -fuse-ld=gold

LOCAL_CXXFLAGS :=  -std=c++11 -D__ARM_NEON__   -mfpu=neon -mfloat-abi=softfp -mvectorize-with-neon-quad -fPIE -O3 -DNDEBUG


LOCAL_CFLAGS += -fPIE
LOCAL_LDFLAGS += -fPIE -pie
#LOCAL_ALLOW_UNDEFINED_SYMBOLS := true

include $(BUILD_EXECUTABLE)