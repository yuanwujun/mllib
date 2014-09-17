CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -static

LIBS=liblinear.a
DEST=../lib/LinearModel

$(LIBS):tron.o linear.o dnrm2.o daxpy.o ddot.o dscal.o
	-@if [ ! -d ${DEST} ]; then mkdir -p ${DEST}; fi
	ar r $(DEST)/$@ $^

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

dnrm2.o: dnrm2.c blas.h blasp.h
	$(CC) $(CFLAGS) -c -o dnrm2.o dnrm2.c

daxpy.o: daxpy.c blas.h blasp.h
	$(CC) $(CFLAGS) -c -o daxpy.o daxpy.c

ddot.o: ddot.c blas.h blasp.h
	$(CC) $(CFLAGS) -c -o ddot.o ddot.c

dscal.o: dscal.c blas.h blasp.h
	$(CC) $(CFLAGS) -c -o dscal.o dscal.c

clean:
	rm -f *~ *.o linear.o
