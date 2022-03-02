#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include <iostream>
#include <stdlib.h>
#include <string.h>

// define variables in constant memory for fast access
__constant__ char cons_level1[sizeof(char) * 27 * 27];
__constant__ char cons_level2[sizeof(char) * 27 * 27];
__constant__ char cons_seq1[sizeof(char) * BUF_SIZE_SEQ1];
__constant__ int cons_seq1_size;
__constant__ int cons_weights[sizeof(int) * 4];

void checkStatus(cudaError_t cudaStatus, char *dev_arr, std::string err)
{
    if (cudaStatus != cudaSuccess)
    {
        free(dev_arr);
        std::cout << err << std::endl;
        exit(1);
    }
}

void checkStatusInt(cudaError_t cudaStatus, int *dev_arr, std::string err)
{
    if (cudaStatus != cudaSuccess)
    {
        free(dev_arr);
        std::cout << err << std::endl;
        exit(1);
    }
}

void send_mat_levels_cuda(char mat_level1[27 * 27], char mat_level2[27 * 27], int size)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpyToSymbol(cons_level1, mat_level1, sizeof(char) * 27 * 27);
    checkStatus(cudaStatus, cons_level1, "Cuda cudaMemcpyToSymbol cons_level1 Faild");
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpyToSymbol(cons_level2, mat_level2, sizeof(char) * 27 * 27);
    checkStatus(cudaStatus, cons_level2, "Cuda cudaMemcpyToSymbol cons_level2 Faild");
    cudaStatus = cudaDeviceSynchronize();
}

void send_Seq1_To_Cuda(char *seq1, int seq1_size)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpyToSymbol(cons_seq1, seq1, seq1_size);
    checkStatus(cudaStatus, cons_seq1, "Cuda cudaMemcpyToSymbol cons_seq1 Faild");
    int *size = &seq1_size; // cuda wants size as pionter
    cudaStatus = cudaMemcpyToSymbol(cons_seq1_size, size, sizeof(int));
    checkStatus(cudaStatus, cons_seq1, "Cuda cudaMemcpyToSymbol cons_seq1 Faild");
}

void send_weights_cuda(int weights[4])
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpyToSymbol(cons_weights[0], weights, sizeof(int) * 4);
    checkStatus(cudaStatus, cons_seq1, "Cuda cudaMemcpyToSymbol dev_w1 Faild");
}

__global__ void calc_result(char *seq, int seq_size, int *dev_count_signs, int *dev_result_score, int *dev_result_offset, int *dev_result_mutant)
{
    // share memory for each block, later it will marge with the global memory
    extern __shared__ int shared_count_signs[]; // [$, %, #, ' ']

    //offset number of threads from previous block, plus current location in block
    //       row        num_cols     current_column
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= seq_size)
        return;

    if (seq_size == cons_seq1_size)
    {
        // reset sings arr before calculation
        dev_count_signs[0] = 0;
        dev_count_signs[1] = 0;
        dev_count_signs[2] = 0;
        dev_count_signs[3] = 0;
        shared_count_signs[0] = 0;
        shared_count_signs[1] = 0;
        shared_count_signs[2] = 0;
        shared_count_signs[3] = 0;

        __syncthreads(); // wait for threads to zero out shared memory

        if (seq[id] == cons_seq1[id])
            atomicAdd(&(shared_count_signs[0]), 1);
        else if (cons_level1[(seq[id] - 'A' + 1) + 27 * (cons_seq1[id] - 'A' + 1)] == 1)
            atomicAdd(&(shared_count_signs[1]), 1);
        else if (cons_level2[(seq[id] - 'A' + 1) + 27 * (cons_seq1[id] - 'A' + 1)] == 1)
            atomicAdd(&(shared_count_signs[2]), 1);
        else
            atomicAdd(&(shared_count_signs[3]), 1);

        __syncthreads(); // wait for threads to finish calc there emelemnts

        atomicAdd(&(dev_count_signs[threadIdx.x]), shared_count_signs[threadIdx.x]); // sum all the blocks

        __syncthreads(); // make sure every thread finish

        dev_result_score[0] = (cons_weights[0] * dev_count_signs[0] - cons_weights[1] * dev_count_signs[1] - cons_weights[2] * dev_count_signs[2] - cons_weights[3] * dev_count_signs[3]);
        dev_result_offset[0] = 0; // in this case we don't have n (offset)
        dev_result_mutant[0] = 0; // in this case we don't have k (mutant)
    }
    else // diffrent sizes
    {
        int offset = 0;
        int mutant = 0;
        int loc_max_offset = 0;
        int loc_max_mutant = 0;
        int max = INT_MIN; // smallest int posibale
        __syncthreads();   // wait for threads to zero out shared memory

        for (offset = 0; offset < cons_seq1_size - seq_size; offset++)
        {
            for (mutant = 0; mutant < seq_size; mutant++) // mutant
            {
                // reset sings arr before calculation
                dev_count_signs[0] = 0;
                dev_count_signs[1] = 0;
                dev_count_signs[2] = 0;
                dev_count_signs[3] = 0;
                shared_count_signs[0] = 0;
                shared_count_signs[1] = 0;
                shared_count_signs[2] = 0;
                shared_count_signs[3] = 0;

                __syncthreads(); // wait for threads to zero out shared memory

                if ((id < mutant) || mutant == 0) //mutant==0 for whan the offset is at the and to avoid overflow form seq1
                {
                    if (seq[id] == cons_seq1[id + offset])
                        atomicAdd(&(shared_count_signs[0]), 1);
                    else if (cons_level1[(seq[id] - 'A' + 1) + 27 * (cons_seq1[id + offset] - 'A' + 1)] == 1)
                        atomicAdd(&(shared_count_signs[1]), 1);
                    else if (cons_level2[(seq[id] - 'A' + 1) + 27 * (cons_seq1[id + offset] - 'A' + 1)] == 1)
                        atomicAdd(&(shared_count_signs[2]), 1);
                    else
                        atomicAdd(&(shared_count_signs[3]), 1);
                }
                else
                {
                    if (seq[id] == cons_seq1[id + offset + 1])
                        atomicAdd(&(shared_count_signs[0]), 1);
                    else if (cons_level1[(seq[id] - 'A' + 1) + 27 * (cons_seq1[id + offset + 1] - 'A' + 1)] == 1)
                        atomicAdd(&(shared_count_signs[1]), 1);
                    else if (cons_level2[(seq[id] - 'A' + 1) + 27 * (cons_seq1[id + offset + 1] - 'A' + 1)] == 1)
                        atomicAdd(&(shared_count_signs[2]), 1);
                    else
                        atomicAdd(&(shared_count_signs[3]), 1);
                }

                __syncthreads(); // wait for threads to finish calc there emelemnts

                atomicAdd(&(dev_count_signs[threadIdx.x]), shared_count_signs[threadIdx.x]); // sum all the blocks

                __syncthreads(); // make sure every thread finish

                if (max < cons_weights[0] * dev_count_signs[0] - cons_weights[1] * dev_count_signs[1] - cons_weights[2] * dev_count_signs[2] - cons_weights[3] * dev_count_signs[3])
                {
                    max = cons_weights[0] * dev_count_signs[0] - cons_weights[1] * dev_count_signs[1] - cons_weights[2] * dev_count_signs[2] - cons_weights[3] * dev_count_signs[3];
                    loc_max_offset = offset; // store current biggest n (offset)
                    loc_max_mutant = mutant; // store current biggest k (mutant)
                }
            }
        }

        dev_result_score[0] = max;
        dev_result_offset[0] = loc_max_offset; // store biggest n (offset)
        dev_result_mutant[0] = loc_max_mutant; // store biggest k (mutant)

    }
    __syncthreads(); // make sure every thread finish
}

void send_divided_Seq2_To_Cuda(char *seq2_divided, int seq2_size, int num_rows_each_proc, int *local_score, int *local_offset, int *local_mutant)
{
    int i;
    char *dev_seq2 = 0;         // will containt the seq2 in cuda
    int *dev_count_signs = 0;   // will containt the count of signs is array in cuda
    int *dev_result_score = 0;  // will containt the result score is array in cuda
    int *dev_result_offset = 0; // will containt the result offset is array in cuda
    int *dev_result_mutant = 0; // will containt the result k (mutant) is array in cuda

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void **)&dev_seq2, seq2_size);
    checkStatus(cudaStatus, dev_seq2, "Cuda Malloc dev_seq2 Faild");

    cudaStatus = cudaMalloc((void **)&dev_result_score, sizeof(int) * num_rows_each_proc);
    checkStatusInt(cudaStatus, dev_result_score, "Cuda Malloc dev_result_score Faild");

    cudaStatus = cudaMalloc((void **)&dev_result_offset, sizeof(int) * num_rows_each_proc);
    checkStatusInt(cudaStatus, dev_result_offset, "Cuda Malloc dev_result_offset Faild");

    cudaStatus = cudaMalloc((void **)&dev_result_mutant, sizeof(int) * num_rows_each_proc);
    checkStatusInt(cudaStatus, dev_result_mutant, "Cuda Malloc dev_result_mutant Faild");

    cudaStatus = cudaMemcpy(dev_seq2, seq2_divided, seq2_size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus, dev_seq2, "Cuda cudaMemcpyHostToDevice seq2 Faild");

    for (i = 0; i < num_rows_each_proc; i++)
    {
        cudaStatus = cudaMalloc((void **)&dev_count_signs, sizeof(int) * strlen(seq2_divided + i * BUF_SIZE_SEQ2));
        checkStatusInt(cudaStatus, dev_count_signs, "Cuda Malloc dev_count_signs Faild");
        cudaDeviceProp prop;
        cudaStatus = cudaGetDeviceProperties(&prop, 0);
        int numThreads = prop.maxThreadsPerBlock < seq2_size ? prop.maxThreadsPerBlock : seq2_size;
        int numOfBlocks = seq2_size / numThreads;
        int extraBlock = (seq2_size % numThreads) != 0;

        calc_result<<<numOfBlocks + extraBlock, numThreads, sizeof(int) * strlen(seq2_divided + i * BUF_SIZE_SEQ2)>>>
            (dev_seq2 + i * BUF_SIZE_SEQ2, strlen(seq2_divided + i * BUF_SIZE_SEQ2),
            dev_count_signs, dev_result_score + i, dev_result_offset + i, dev_result_mutant + i);

        cudaStatus = cudaDeviceSynchronize();
        checkStatus(cudaStatus, dev_seq2, "Cuda cudaDeviceSynchronize Faild!!");
    }

    cudaStatus = cudaMemcpy(local_score, dev_result_score, sizeof(int) * num_rows_each_proc, cudaMemcpyDeviceToHost);
    checkStatusInt(cudaStatus, local_score, "Cuda cudaMemcpyDeviceToHost result Faild");
    cudaStatus = cudaMemcpy(local_offset, dev_result_offset, sizeof(int) * num_rows_each_proc, cudaMemcpyDeviceToHost);
    checkStatusInt(cudaStatus, local_offset, "Cuda cudaMemcpyDeviceToHost offset Faild");
    cudaStatus = cudaMemcpy(local_mutant, dev_result_mutant, sizeof(int) * num_rows_each_proc, cudaMemcpyDeviceToHost);
    checkStatusInt(cudaStatus, local_mutant, "Cuda cudaMemcpyDeviceToHost mutant Faild");
   
    

    // free allocate memory
    cudaStatus = cudaFree(dev_seq2);
    checkStatus(cudaStatus, dev_seq2, "Cuda dev_seq2 free Faild");
    cudaStatus = cudaFree(dev_count_signs);
    checkStatusInt(cudaStatus, dev_count_signs, "Cuda dev_count_signs free Faild");
    cudaStatus = cudaFree(dev_result_score);
    checkStatusInt(cudaStatus, dev_result_score, "Cuda dev_result_score free Faild");
    cudaStatus = cudaFree(dev_result_offset);
    checkStatusInt(cudaStatus, dev_result_offset, "Cuda dev_result_offset free Faild");
    cudaStatus = cudaFree(dev_result_mutant);
    checkStatusInt(cudaStatus, dev_result_mutant, "Cuda dev_result_mutant free Faild");
}
