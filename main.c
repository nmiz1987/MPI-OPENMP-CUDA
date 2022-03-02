#include "myProto.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <ctype.h>

enum ranks
{
	ROOT
};

void build_mat(int size, char group[11][11], char mat[27 * 27])
{
	int i, j, k;
	char c1, c2;

	// init matrix with 0
	for (i = 0; i < 27; i++)
	{
		for (j = 0; j < 27; j++)
		{
			mat[i + size * j] = 0;
		}
	}

	// build every pair possibale
	for (i = 0; i < size; i++) // for every word
	{
		for (j = 0; j < strlen(group[i]); j++) // get the first char
		{
			c1 = group[i][j];
			for (k = 0; k < strlen(group[i]); k++) // get the first char
			{
				c2 = group[i][k];
				// 1 = there is a pair, (c1 - 'A'+1) convert char to int for the index starting from 1
				// note: Do not use index 0 in order to be able to make cross in matxrix!
				mat[(c1 - 'A' + 1) + 27 * (c2 - 'A' + 1)] = 1; // mat[c1 - 'A'+1][c2 - 'A'+1] = 1;  //e.g ['A']['B']
				mat[(c2 - 'A' + 1) + 27 * (c1 - 'A' + 1)] = 1; // mat[c2 - 'A'+1][c1 - 'A'+1] = 1;  //e.g ['B']['A']
			}
		}
	}
}

int main(int argc, char **argv)
{
	int my_rank, num_procs;
	char *seq1, *seq2_all, *seq2_divided, *buffer;
	int *local_score, *local_offset, *local_mutant;
	int *local_remain_score, *local_remain_offset, *local_remain_mutant;
	int *arr_score, *arr_offset, *arr_mutant;
	char matrix_level_1[27 * 27], matrix_level_2[27 * 27];
	int i, j, remain_size, my_size, num_rows_each_proc;
	int weights[4];
	int num_of_seq2;
	char c1, c2;

	char group1[11][11] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF", "", ""};
	char group2[11][11] = {"SAG", "SGND", "NEQHRK", "HFY", "ATV", "STPA", "NDEQHK", "FVLIM", "CSA", "STNK", "SNDEQK"};

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	seq1 = (char *)malloc(sizeof(char) * BUF_SIZE_SEQ1); // everyone needs seq1

	if (my_rank == ROOT)
	{

		// build to data structure to easy search
		build_mat(11, group1, matrix_level_1);
		build_mat(11, group2, matrix_level_2);

		// read weights from user
		fscanf(stdin, "%d %d %d %d", weights, weights + 1, weights + 2, weights + 3);

		// get the string for seq1
		fscanf(stdin, "%s", seq1);

// convert string to uppercase
#pragma omp parallel for
		for (i = 0; i < strlen(seq1); i++)
		{
			if (seq1[i] >= 'a' && seq1[i] <= 'z')
				seq1[i] = toupper(seq1[i]);
		}

		// get the number of seq2
		fscanf(stdin, "%d", &num_of_seq2);

		// allocate the seq2 array
		seq2_all = (char *)malloc(sizeof(char) * num_of_seq2 * BUF_SIZE_SEQ2);

// start readint all seq2
#pragma omp parallel for
		for (i = 0; i < num_of_seq2; i++)
		{
			buffer = (char *)malloc(BUF_SIZE_SEQ2 * sizeof(char)); // allocate new buffer to store data
			fscanf(stdin, "%s", buffer);						   // read line to buffer
			// convert string to uppercase
			for (j = 0; j < strlen(buffer); j++)
			{
				if (buffer[j] >= 'a' && buffer[j] <= 'z')
					buffer[j] = toupper(buffer[j]);
			}
			strcpy(seq2_all + BUF_SIZE_SEQ2 * i, buffer); // read line from stdin
		}

		if (num_procs < num_of_seq2) // if there are more seq2's then workers
		{
			num_rows_each_proc = num_of_seq2 / num_procs;				  // the size for each proc
			remain_size = num_of_seq2 - (num_procs * num_rows_each_proc); // the remains size to take care
		}
		else // if there are more workers then seq2's
		{
			num_rows_each_proc = 1; // each worker calc only 1 of seq2
			remain_size = 0;
		}

		my_size = num_rows_each_proc * BUF_SIZE_SEQ2; // total size for each worker

		arr_score = (int *)malloc(sizeof(int) * num_of_seq2);  // will store the result score
		arr_offset = (int *)malloc(sizeof(int) * num_of_seq2); // will store the result n (offset)
		arr_mutant = (int *)malloc(sizeof(int) * num_of_seq2); // will store the result k (mutant)

		// send to 2 levels to cude because the calculation will be there
		send_mat_levels_cuda(matrix_level_1, matrix_level_2, 27 * 27);

		// send weights to cude because the calculation will be there
		send_weights_cuda(weights);

		// ROOT sent seq1 to cuda, cude will need it for comparesion
		send_Seq1_To_Cuda(seq1, strlen(seq1));

		seq2_divided = (char *)malloc(sizeof(char) * my_size);		   // allocate seq2 in root accordind to the size it needs
		local_score = (int *)calloc(num_rows_each_proc, sizeof(int));  // will get the local score for each seq2
		local_offset = (int *)calloc(num_rows_each_proc, sizeof(int)); // will get the local resule n (offset) for each seq2
		local_mutant = (int *)calloc(num_rows_each_proc, sizeof(int)); // will get the local resule k (mutant) for each seq2

		if (remain_size > 0)
		{
			local_remain_score = (int *)calloc(num_rows_each_proc, sizeof(int));  // will get the local remain score for each seq2
			local_remain_offset = (int *)calloc(num_rows_each_proc, sizeof(int)); // will get the local remain resule n (offset) for each seq2
			local_remain_mutant = (int *)calloc(num_rows_each_proc, sizeof(int)); // will get the local remain resule k (mutant) for each seq2
		}
	}

	MPI_Bcast(seq1, BUF_SIZE_SEQ1, MPI_CHAR, ROOT, MPI_COMM_WORLD);		// make sure every worker knows seq1
	MPI_Bcast(weights, sizeof(int) * 4, MPI_INT, ROOT, MPI_COMM_WORLD); // make sure every worker knows weights
	MPI_Bcast(&my_size, 1, MPI_INT, ROOT, MPI_COMM_WORLD);				// make sure every worker knows the size to allocate array for seq2_divided
	MPI_Bcast(&num_rows_each_proc, 1, MPI_INT, ROOT, MPI_COMM_WORLD);	// make sure everyone knows the size to allocate array for seq2_divided

	if (my_rank != ROOT)
	{
		// build to data structure to easy search
		build_mat(11, group1, matrix_level_1);
		build_mat(11, group2, matrix_level_2);

		if (my_rank <= num_rows_each_proc) // if there are more workers then seq2 we don't want everyone to send data to cuda
		{
			send_mat_levels_cuda(matrix_level_1, matrix_level_2, 27 * 27); // send to 2 levels to cude because the calculation will be there
			send_weights_cuda(weights);									   // send weights to cude because the calculation will be there
			send_Seq1_To_Cuda(seq1, strlen(seq1));						   // worker sent seq1 to cuda, the cude will need it for comparesion
		}

		// allocate seq2 in each worker accordind to the size it received
		seq2_divided = (char *)malloc(sizeof(char) * my_size);		   // allocate seq2 in root accordind to the size it needs
		local_score = (int *)calloc(num_rows_each_proc, sizeof(int));  // will get the local resule score for each seq2
		local_offset = (int *)calloc(num_rows_each_proc, sizeof(int)); // will get the local resule n (offset) for each seq2
		local_mutant = (int *)calloc(num_rows_each_proc, sizeof(int)); // will get the local resule k (mutant) for each seq2
	}

	MPI_Scatter(seq2_all, my_size, MPI_CHAR, seq2_divided, my_size, MPI_CHAR, ROOT, MPI_COMM_WORLD); //spread seq2 to workers and self

	// By now cude has weights, seq1 and the 2 levels matrixs, root & workers has seq2_devided - its part of the seq2

	if (my_rank == ROOT)
	{
		// sent seq2 to cuda
		send_divided_Seq2_To_Cuda(seq2_divided, my_size, num_rows_each_proc, local_score, local_offset, local_mutant);

		// if the are remain seq2, root needs to take care of them
		if (remain_size > 0)
			send_divided_Seq2_To_Cuda(seq2_all + (num_of_seq2 - remain_size) * BUF_SIZE_SEQ2, remain_size * BUF_SIZE_SEQ2, remain_size, local_remain_score, local_remain_offset, local_remain_mutant);
	}

	if (my_rank != ROOT)
	{
		// sent seq2 to cuda
		send_divided_Seq2_To_Cuda(seq2_divided, my_size, num_rows_each_proc, local_score, local_offset, local_mutant);
	}

	// Gather all the results
	MPI_Gather(local_score, num_rows_each_proc, MPI_INT, arr_score, num_rows_each_proc, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Gather(local_offset, num_rows_each_proc, MPI_INT, arr_offset, num_rows_each_proc, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Gather(local_mutant, num_rows_each_proc, MPI_INT, arr_mutant, num_rows_each_proc, MPI_INT, ROOT, MPI_COMM_WORLD);

	if (my_rank == ROOT)
	{
		// Print results
		for (i = 0; i < num_of_seq2 - remain_size; i++)
		{
			printf("#%d: score: %d, n: %d, k: %d\n", i, arr_score[i], arr_offset[i], arr_mutant[i]);
		}
		for (i = 0; i < remain_size; i++)
		{
			if (strlen((seq2_all + num_procs * num_rows_each_proc * BUF_SIZE_SEQ2) + BUF_SIZE_SEQ2 * i) > 0)
				printf("#%d: score: %d, n: %d, k: %d\n", i + num_procs * num_rows_each_proc, local_remain_score[i], local_remain_offset[i], local_remain_mutant[i]);
		}
	}

	if (my_rank = ROOT)
	{
		free(seq2_divided);
		free(arr_score);
		free(arr_offset);
		free(arr_mutant);
		free(local_score);
		free(local_offset);
		free(local_mutant);
		if (remain_size > 0)
		{
			free(local_remain_score);
			free(local_remain_offset);
			free(local_remain_mutant);
		}
		free(seq2_all);
		free(seq1);
		free(buffer);
	}
	if (my_rank != ROOT)
	{
		free(seq2_divided);
		free(seq1);
		free(local_score);
		free(local_offset);
		free(local_mutant);
		free(buffer);
	}

	MPI_Finalize();
	return 0;
}
