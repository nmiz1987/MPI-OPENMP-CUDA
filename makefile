build:
	mpicxx -fopenmp -c main.c
	nvcc -I./inc -c cudaFunctions.cu
	mpicxx -fopenmp -o final main.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt
	
clean:
	rm -f *.o ./final


run:
	mpiexec -np 2 ./final

	
runOn2:
	mpiexec -np 2 -machinefile mf --map-by node  ./final
