test: cpu-new-forward.o new-forward.o test.o
	nvcc -std=c++11 -o test cpu-new-forward.o test.o new-forward.o ../libwb/lib/libwb.so

new-forward.o: custom/new-forward.cu
	nvcc -std=c++11 -I ..../libwb/ -rdc=true -c custom/new-forward.cu

test.o: test.cu
	nvcc -std=c++11 -I ..../libwb/ -I custom/ -rdc=true -c test.cu

cpu-new-forward.o: custom/cpu-new-forward.cc
	gcc -std=c++11 -c custom/cpu-new-forward.cc

.PHONY: clean
clean:
	rm *.o test
