include ../../ThirdLib.mk

PROGRAM=btm
cc=g++

INCLUDES=-I.
LIB_PATH= 

SRC_DIRS=.

SRC=$(wildcard $(SRC_DIRS)/*.cpp)
OBJS=$(patsubst %.cpp,%.o,$(SRC))
LIBS= -lpthread

CXXFLAGS=-Wall -fopenmp -O3

all:$(OBJS)
	$(cc) -o $(PROGRAM) $(CXXFLAGS) $(OBJS) $(LIBS)
$(OBJS):%.o:%.cpp
	$(cc) -c $(CXXFLAGS) -I$(INCLUDES) $< -o $@

clean:
	rm -rf ${PROGRAM} *.o ../common/*.o
