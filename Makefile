#------------------------------------------------------------------------------#
# This makefile was generated by 'cbp2make' tool rev.147                       #
#------------------------------------------------------------------------------#


WORKDIR = `pwd`

CC = gcc
CXX = g++
AR = ar
LD = g++
WINDRES = windres

INC = -I/home/stylix/local/include
CFLAGS = -Wall -fexceptions
RESINC = 
LIBDIR = -L/home/stylix/local/lib
LIB = ../lib/ins/libins.a ../lib/sifthesaff/libsifthesaff.a ../lib/alphautils/libalphautils.a -lopencv_stitching -lopencv_videostab -lopencv_gpu -lopencv_legacy -lopencv_ts -lopencv_nonfree -lopencv_contrib -lopencv_calib3d -lopencv_objdetect -lopencv_features2d -lopencv_video -lopencv_photo -lopencv_highgui -lopencv_flann -lopencv_imgproc -lopencv_ml -lopencv_core -lx264 -lfaac -lgomp -lhdf5 -lhdf5_hl_cpp -lhdf5_cpp -lhdf5_hl -lmpi_cxx -lmpi
LDFLAGS = `pkg-config --libs opencv` -lgomp -lrt -lpthread -ldl

INC_DEBUG = $(INC)
CFLAGS_DEBUG = $(CFLAGS) -g -fopenmp `pkg-config --cflags opencv`
RESINC_DEBUG = $(RESINC)
RCFLAGS_DEBUG = $(RCFLAGS)
LIBDIR_DEBUG = $(LIBDIR)
LIB_DEBUG = $(LIB)
LDFLAGS_DEBUG = $(LDFLAGS)
OBJDIR_DEBUG = obj/Debug
DEP_DEBUG = 
OUT_DEBUG = bin/Debug/ins_offline

INC_RELEASE = $(INC)
CFLAGS_RELEASE = $(CFLAGS) -march=core2 -O3 -fopenmp `pkg-config --cflags opencv`
RESINC_RELEASE = $(RESINC)
RCFLAGS_RELEASE = $(RCFLAGS)
LIBDIR_RELEASE = $(LIBDIR)
LIB_RELEASE = $(LIB)
LDFLAGS_RELEASE = $(LDFLAGS) -s
OBJDIR_RELEASE = obj/Release
DEP_RELEASE = 
OUT_RELEASE = bin/Release/ins_offline

OBJ_DEBUG = $(OBJDIR_DEBUG)/ins_offline.o

OBJ_RELEASE = $(OBJDIR_RELEASE)/ins_offline.o

all: debug release

clean: clean_debug clean_release

before_debug: 
	test -d bin/Debug || mkdir -p bin/Debug
	test -d $(OBJDIR_DEBUG) || mkdir -p $(OBJDIR_DEBUG)

after_debug: 

debug: before_debug out_debug after_debug

out_debug: before_debug $(OBJ_DEBUG) $(DEP_DEBUG)
	$(LD) $(LIBDIR_DEBUG) -o $(OUT_DEBUG) $(OBJ_DEBUG)  $(LDFLAGS_DEBUG) $(LIB_DEBUG)

$(OBJDIR_DEBUG)/ins_offline.o: ins_offline.cpp
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c ins_offline.cpp -o $(OBJDIR_DEBUG)/ins_offline.o

clean_debug: 
	rm -f $(OBJ_DEBUG) $(OUT_DEBUG)
	rm -rf bin/Debug
	rm -rf $(OBJDIR_DEBUG)

before_release: 
	test -d bin/Release || mkdir -p bin/Release
	test -d $(OBJDIR_RELEASE) || mkdir -p $(OBJDIR_RELEASE)

after_release: 

release: before_release out_release after_release

out_release: before_release $(OBJ_RELEASE) $(DEP_RELEASE)
	$(LD) $(LIBDIR_RELEASE) -o $(OUT_RELEASE) $(OBJ_RELEASE)  $(LDFLAGS_RELEASE) $(LIB_RELEASE)

$(OBJDIR_RELEASE)/ins_offline.o: ins_offline.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c ins_offline.cpp -o $(OBJDIR_RELEASE)/ins_offline.o

clean_release: 
	rm -f $(OBJ_RELEASE) $(OUT_RELEASE)
	rm -rf bin/Release
	rm -rf $(OBJDIR_RELEASE)

.PHONY: before_debug after_debug clean_debug before_release after_release clean_release

