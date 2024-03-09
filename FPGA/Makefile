# This is a GNU Makefile.

# It can be used to compile an OpenCL program with

# You must configure INTELFPGAOCLSDKROOT to point the root directory of the Intel(R) FPGA SDK for OpenCL(TM)
# software installation.


HOST_DIR ?= host/src
TARGET = CG.exe
AOCL_UTILS = ./common

# Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
#ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
#$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
#endif
#ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
#$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
#endif

SRCS := $(wildcard host/src/conjugate_gradients.cpp $(AOCL_UTILS)/src/AOCLUtils/*.cpp)
INC_DIRS := $(AOCL_UTILS)/inc host/inc
INC_DIRS += device

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

# Make it all!
all : 
	g++ -fPIC $(foreach D,$(INC_DIRS),-I$D) $(SRCS) -o $(TARGET) -DLINUX $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -lrt -DHOST -DBURST_LENGTH=32

# Standard make targets
clean :
	@rm -f $(OBJS) $(TARGET)

