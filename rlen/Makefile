all:
	echo "building rlen_cython"
	python setup.py build_ext --inplace

clean:
	- python setup.py clean
	- rm *.so rlen.c
	- rm -rf build/
