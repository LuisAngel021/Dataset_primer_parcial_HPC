G++ = g++
FLAGS = -ansi -pedantic -Wall -D_POSIX_C_SOURCE=199309L
PROGS = main linearregression extractiondata CMakeLists

all: $(PROGS)

main: 
	$(G++) $(FLAGS) $@.cpp -cpp
	$(G++) $(FLAGS) $@.o -o $@

linearregression: 
	$(G++) $(FLAGS) $@.cpp -cpp
	$(G++) $(FLAGS) $@.o -o $@

extractiondata:
	$(GCC) $(FLAGS) $@.cpp -cpp
	$(GCC) $(FLAGS) $@.o -o $@

CMakeLists:
	$(GCC) $(FLAGS) $@.cpp -cpp
	$(GCC) $(FLAGS) $@.o -o $@

clean:
	$(RM) $(PROGS) *.o
