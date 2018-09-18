_example.so : example.o example_wrap.o
	gcc -shared example.o example_wrap.o -o _example.so -lpython2.7

example.o : example.c
	gcc -c -fPIC -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 example.c

example_wrap.o : example_wrap.c
	gcc -c -fPIC -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 example_wrap.c

example_wrap.c example.py : example.i example.h
	swig -python example.i

clean:
	rm -f *.o *.so example_wrap.* example.py*

test:
	python test.py

all: _example.so test

.PHONY: clean test all

.DEFAULT_GOAL := all
