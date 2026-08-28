# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#include ../../../../../make/nvdefs.mk
include /drive/drive-linux/make/nvdefs.mk

TARGETS = nvsipl_camera_event_gen_socket_jpg_sender

OPENCV_ROOT := /app/opencv_test/opencv4_test/opencv_source/opencv
OPENCV_BUILD := $(OPENCV_ROOT)/build
OPENCV_INC_DIRS := -I$(OPENCV_ROOT)/include \
		    -I$(OPENCV_BUILD) \
		    -I$(OPENCV_ROOT)/modules/calib3d/include \
		    -I$(OPENCV_ROOT)/modules/core/include \
		    -I$(OPENCV_ROOT)/modules/dnn/include \
		    -I$(OPENCV_ROOT)/modules/features2d/include \
		    -I$(OPENCV_ROOT)/modules/flann/include \
		    -I$(OPENCV_ROOT)/modules/gapl/include \
		    -I$(OPENCV_ROOT)/modules/highgui/include \
		    -I$(OPENCV_ROOT)/modules/imgcodecs/include \
		    -I$(OPENCV_ROOT)/modules/imgproc/include \
		    -I$(OPENCV_ROOT)/modules/java/include \
		    -I$(OPENCV_ROOT)/modules/js/include \
		    -I$(OPENCV_ROOT)/modules/ml/include \
		    -I$(OPENCV_ROOT)/modules/objc/include \
		    -I$(OPENCV_ROOT)/modules/objdetect/include \
		    -I$(OPENCV_ROOT)/modules/photo/include \
		    -I$(OPENCV_ROOT)/modules/python/include \
		    -I$(OPENCV_ROOT)/modules/stitching/include \
		    -I$(OPENCV_ROOT)/modules/ts/include \
		    -I$(OPENCV_ROOT)/modules/video/include \
		    -I$(OPENCV_ROOT)/modules/videoio/include \
		    -I$(OPENCV_ROOT)/modules/world/include \
		    
OPENCV_LIB_DIRS := -L$(OPENCV_BUILD)/lib \
		    -L$(OPENCV_BUILD)/3rdparty/lib
		    
OPENCV_LIBS := -lopencv_core \
		-lopencv_imgproc \
		-lopencv_imgcodecs \
		-lopencv_videoio \
		-lopencv_highgui


CXXFLAGS := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -std=c++17 -fexceptions -frtti -fPIC -DEVSIM_ENABLE_CV_UDP
CPPFLAGS := -I/drive/drive-linux/include/nvmedia_6x \
            -I/drive/drive-linux/plugin \
            $(NV_PLATFORM_CPPFLAGS) \
            $(NV_PLATFORM_SDK_INC) \
            -DWIN_INTERFACE_CUSTOM \
            -D_POSIX_C_SOURCE=200112L

LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS   := CUtils.o
OBJS   += main.o
OBJS   += EventGenerator.o
ifeq ($(NV_PLATFORM_SAFETY), 0)
  OBJS   += CComposite.o
  OBJS   += CCompositeNvSci.o
endif

LDLIBS  += -L../../lib
LDLIBS  += -lnvsipl_sampleplugin
LDLIBS  += -lnvsipl_query
LDLIBS  += -lnvsipl
LDLIBS  += -L../utils/lib/
LDLIBS  += -lnvscibuf
LDLIBS  += -lnvscisync
LDLIBS  += -lnvscistream
LDLIBS  += -lcam_fsync

ifeq ($(NV_PLATFORM_OS), Linux)
  LDLIBS += -lpthread
  LDLIBS += -ldl
  LDLIBS += -lrt
  LDLIBS += -lstdc++
  LDLIBS += $(OPENCV_LIBS)
else
  LDLIBS += -lsocket
  LDLIBS += -lc++
  LDLIBS += $(OPENCV_LIBS)
endif

ifeq ($(NV_PLATFORM_OS), QNX)
  CPPFLAGS += -DNVMEDIA_QNX
  LDLIBS += -lnvtime
  LDLIBS += -lnvos_s3_safety
  LDLIBS += -lnvdvms_client
endif

ifeq ($(NV_PLATFORM_SAFETY), 0)
  OBJS += CNvWfd.o
  LDLIBS += -ltegrawfd
  LDLIBS += -lnvmedia2d
endif

.PHONY: default
default: $(TARGETS)

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

nvmTest_utils_lib:
	$(MAKE) -C ../utils/lib

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
