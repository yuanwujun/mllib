include ../../ThirdLib.mk

PROGRAM=var_lda
cc=g++

PUB_DIR=../../lib

INCLUDES= ../../common/include -I../common/include -I../../eigen -I$(GOOGLEINC) -I$(EIGEN)
LIB_PATH= $(PUB_DIR)/common -L$(GOOGLELIB) -L$(PUB_DIR)/eigen

SRC_DIRS=.

SRC=$(wildcard $(SRC_DIRS)/*.cpp ../common/*.cpp)
OBJS=$(patsubst %.cpp,%.o,$(SRC))
LIBS= -lpthread -lgflags -lglog -leigen -lcommon

CXXFLAGS=-Wall -g

all:$(OBJS)
	$(cc) -o $(PROGRAM) $(CXXFLAGS) $(OBJS) -L$(LIB_PATH) $(LIBS)
$(OBJS):%.o:%.cpp
	$(cc) -c $(CXXFLAGS) -I$(INCLUDES) $< -o $@

clean:
	rm -rf ${PROGRAM} *.o ../common/*.o
