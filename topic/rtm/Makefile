PROGRAM=rtm
cc=g++

PUB_DIR=../../lib

INCLUDES= ../../common/include -I../../eigen -I$(HOME)/lib/include -I$(HOME)/working/project/coderepos/thirdparty/Eigen
LIB_PATH= $(PUB_DIR)/common -L$(HOME)/lib/lib -L$(PUB_DIR)/eigen

SRC_DIRS=.

SRC=$(wildcard $(SRC_DIRS)/*.cpp)
OBJS=$(patsubst %.cpp,%.o,$(SRC))
LIBS= -lpthread -lgflags -lglog -leigen -lcommon

CXXFLAGS=-Wall

all:$(OBJS)
	$(cc) -o $(PROGRAM) $(CXXFLAGS) $(OBJS) -L$(LIB_PATH) $(LIBS)
$(OBJS):%.o:%.cpp
	$(cc) -c $(CXXFLAGS) -I$(INCLUDES) $< -o $@

clean:
	rm -rf ${PROGRAM} *.o
