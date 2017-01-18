
# Configuration & compiling is handled by CMake; this Makefile
# simply provides some shortcuts for common actions.

all: build
	cd build && cmake .. ; make

test:
	python -m unittest discover

ccmake:
	cd build && ccmake ..

build:
	mkdir build

clean:
	rm -rf build
