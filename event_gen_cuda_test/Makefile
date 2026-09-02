# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

################################################################################
# Combined build: NVSIPL camera app + CUDA event generator, one binary.
#
# Previously two separate worlds:
#   - the NVSIPL camera app built via nvdefs.mk (this file's original form -
#     see the git history / the plain "Makefile" this replaces). nvdefs.mk
#     sets up the DRIVE-OS cross toolchain (CXX/LD/AR), NvSci/NVSIPL include
#     and lib paths, and safety-build knobs (NV_PLATFORM_SAFETY etc).
#   - EventGeneratorCUDA.cu compiled/tested standalone with nvcc directly,
#     using a cudaNvSci-sample-style Makefile (TARGET_ARCH detection,
#     GENCODE_FLAGS, findnvsci.mk, etc).
#
# This Makefile keeps nvdefs.mk driving everything it already drove, and
# adds ONE extra rule that invokes nvcc as a CROSS COMPILER (not as the
# final linker) to turn EventGeneratorCUDA.cu into a plain .o file, built
# with -ccbin pointed at the SAME cross host compiler nvdefs.mk already
# configured. That keeps the resulting object ABI-compatible with every
# other .o in this build, so the whole thing links in one pass with the
# same $(LD) call the NVSIPL side always used - nvcc never touches final
# linking here, so there's no nvcc-vs-nvdefs.mk linker-flag conflict to
# reconcile.
#
# The old cudaNvSci-style Makefile (Makefile.txt / findnvsci.mk) still has
# a use: keep it around for compiling/running EventGeneratorCUDA in
# isolation (e.g. against test_event_generator_cuda.cpp) when you're
# iterating on the kernel itself and don't want to relink the whole camera
# app for every change. It is NOT used by this Makefile.
################################################################################

include ../../../../../make/nvdefs.mk

TARGETS = nvsipl_camera

# --- CUDA event-generator toggle ---------------------------------------
# make EVSIM_ENABLE_CUDA=1            -> build & link EventGeneratorCUDA.o,
#                                         define -DEVSIM_ENABLE_CUDA so
#                                         CNvSIPLConsumer.hpp's #ifdef picks
#                                         it up and EnableEventGeneration()
#                                         can actually construct it.
# make  (default, EVSIM_ENABLE_CUDA=0) -> exactly the original nvsipl_camera
#                                         build: CPU event generator only,
#                                         no CUDA toolkit required at all.
EVSIM_ENABLE_CUDA ?= 0

# --- CUDA toolkit location / target selection ---------------------------
CUDA_PATH ?= /usr/local/cuda

# GPU arch(s) to build device code for - Xavier=sm_72, Orin=sm_87. Override
# on the command line (CUDA_SMS="72 87") if the binary needs to run on more
# than one target.
CUDA_SMS ?= 87

# CUDA toolkit's cross target subdir for this NV_PLATFORM_OS/TARGET_ARCH.
# nvdefs.mk already resolved the DRIVE-OS cross triple - this just needs to
# match it for CUDA's own layout (targets/<arch>-<os>/{include,lib}).
# Adjust if your NV_PLATFORM_ARCH variable name/values differ - these are
# the two combinations this project currently targets.
ifeq ($(NV_PLATFORM_OS), QNX)
  CUDA_TARGET_DIR ?= targets/aarch64-qnx
else
  CUDA_TARGET_DIR ?= targets/aarch64-linux
endif

NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

CUDA_GENCODE_FLAGS := $(foreach sm,$(CUDA_SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
HIGHEST_CUDA_SM     := $(lastword $(sort $(CUDA_SMS)))
CUDA_GENCODE_FLAGS  += -gencode arch=compute_$(HIGHEST_CUDA_SM),code=compute_$(HIGHEST_CUDA_SM)

CXXFLAGS := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -std=c++14 -fexceptions -frtti -fPIC
CPPFLAGS := -I../../../../../include/nvmedia_6x -I../plugin $(NV_PLATFORM_CPPFLAGS) $(NV_PLATFORM_SDK_INC) -DWIN_INTERFACE_CUSTOM -D_POSIX_C_SOURCE=200112L

ifeq ($(EVSIM_ENABLE_CUDA), 1)
  CPPFLAGS += -DEVSIM_ENABLE_CUDA -I$(CUDA_PATH)/include
endif

LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

# EventGenerator.o (CPU backend) is part of every build - it's the fallback
# whether or not EVSIM_ENABLE_CUDA is set, and EnableEventGeneration()
# always needs it linked in. Uses make's built-in .cpp.o rule (via
# CXX/CXXFLAGS/CPPFLAGS above), same as CUtils.o/main.o - no explicit
# recipe needed. If EventGenerator.cpp is already compiled as part of
# another object in your real OBJS list (or lives in a static lib you link
# via LDLIBS), drop this line rather than double-compiling it.
OBJS   := CUtils.o
OBJS   += main.o
OBJS   += EventGenerator.o
ifeq ($(EVSIM_ENABLE_CUDA), 1)
  OBJS += EventGeneratorCUDA.o
endif
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

ifeq ($(EVSIM_ENABLE_CUDA), 1)
  LDLIBS += -L$(CUDA_PATH)/$(CUDA_TARGET_DIR)/lib -lcudart
endif

ifeq ($(NV_PLATFORM_OS), Linux)
  LDLIBS += -lpthread
  LDLIBS += -ldl
  LDLIBS += -lrt
  LDLIBS += -lstdc++
else
  LDLIBS += -lsocket
  LDLIBS += -lc++
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

# --- CUDA compile rule ---------------------------------------------------
# nvcc generates the device code and hands host-side code to $(CXX) (via
# -ccbin), so this .o is produced by the exact same host compiler as every
# other .o in OBJS. CPPFLAGS/CXXFLAGS are forwarded through -Xcompiler so
# nvcc doesn't try to interpret DRIVE-OS/NVSIPL flags itself - only nvcc's
# own flags (gencode, --std) are passed directly.
ifeq ($(EVSIM_ENABLE_CUDA), 1)
EventGeneratorCUDA.o: EventGeneratorCUDA.cu
	$(NVCC) $(CUDA_GENCODE_FLAGS) --std=c++14 \
	    $(addprefix -Xcompiler ,$(CXXFLAGS)) \
	    $(addprefix -Xcompiler ,$(CPPFLAGS)) \
	    -I$(CUDA_PATH)/include -o $@ -c $<
endif

nvmTest_utils_lib:
	$(MAKE) -C ../utils/lib

clean clobber:
	rm -rf $(OBJS) $(TARGETS) EventGeneratorCUDA.o
