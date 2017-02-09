CXXFLAGS = -std=c++11 -Iinclude -fopenmp -fopenacc
LDFLAGS = -lm 

test_sources = $(wildcard src/test_*.cc)
test_objects = $(patsubst src/%.cc, src/%.o, $(test_sources))

test: test_main
	./test_main

test_main: $(test_objects)
	echo $(test_objects)
	$(CXX) $(CXXFLAGS) $(test_objects) -o $@

clean:
	rm -f test_main src/*.o
