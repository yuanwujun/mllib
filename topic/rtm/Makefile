include ../../ThirdLib.mk

PROGRAM=rtm
cc=g++

PUB_DIR=../../lib

INCLUDES= ../../common/include -I../../liblinear -I../../eigen -I$(GOOGLEINC) -I$(EIGEN)
LIB_PATH= $(PUB_DIR)/common -L$(PUB_DIR)/LinearModel -L$(GOOGLELIB) -L$(PUB_DIR)/eigen

SRC_DIRS=.

SRC=$(wildcard $(SRC_DIRS)/*.cpp)
OBJS=$(patsubst %.cpp,%.o,$(SRC))
LIBS= -lpthread -lgflags -lglog -leigen -lcommon -llinear

CXXFLAGS=-Wall

all:$(OBJS)
	$(cc) -o $(PROGRAM) $(CXXFLAGS) $(OBJS) -L$(LIB_PATH) $(LIBS)
$(OBJS):%.o:%.cpp
	$(cc) -c $(CXXFLAGS) -I$(INCLUDES) $< -o $@

clean:
	rm -rf ${PROGRAM} *.o
