CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -static

LIBS=liblinear.a
DEST=../lib/LinearModel

$(LIBS):tron.o linear.o blas/blas.a
	-@if [ ! -d ${DEST} ]; then mkdir -p ${DEST}; fi
	ar r $(DEST)/$@ $^

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ tron.o linear.o
