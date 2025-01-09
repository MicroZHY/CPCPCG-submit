CC=sw9gcc
CXX=sw9g++

CFLAGS= -ffast-math -mslave -msimd -mieee -Ofast -funroll-loops   -mfma  -lm_slave -fprefetch-loop-arrays  -mpws
CXXFLAGS= -mhost -mieee -mftz -fpermissive -lm  -Ofast
INCLUDE=-I.

EXE=pcg_solve

all: $(EXE)

$(EXE): main.o pcg.o slave.o
	$(CXX) -mhybrid -o $(EXE) $^ -L. -lpcg_solve

main.o:	main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ 

pcg.o: pcg.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@	

slave.o: slave.c
	$(CC) $(CFLAGS)  -c $< -o $@

clean:
	$(RM) $(EXE) *.o;
