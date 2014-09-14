##target lib###
LIBS=libcommon.a
DEST=../lib/common

##complier##
CXX=g++
CXXFLAGS=-D_DEBUG -g -static
LDFLAGS=

##include dir and source dir##
INCLUDE_DIRS=-I./include -I$(HOME)/lib/include
SRC_DIRS=.

##objects##
OBJS=$(patsubst %.cpp, %.o, $(wildcard $(SRC_DIRS)/*.cpp))
SRC=$(wildcard $(SRC_DIRS)/%.cpp)

vpath %.h $(INCLUDE_DIRS)
vpath %.cpp $(SRC_DIRS)

$(LIBS):$(OBJS)
	-@if [ ! -d ${DEST} ]; then mkdir -p ${DEST}; fi
	ar r $(DEST)/$@ $^

$(OBJS):%.o:%.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< $(LDFLAGS) -o $@ 
	
.PHONY : clean

clean:
	rm -rf $(OBJS)

