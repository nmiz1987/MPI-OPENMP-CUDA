#pragma once

#define BUF_SIZE_SEQ1 3000
#define BUF_SIZE_SEQ2 2000


void send_mat_levels_cuda(char mat_level1[27*27], char mat_level2[27*27], int size);
void send_Seq1_To_Cuda(char* seq1, int seq1_size);
void send_divided_Seq2_To_Cuda(char* seq2_divided, int seq2_size, int num_rows_each_proc, int* local_score, int* local_offset, int* local_k);
void send_weights_cuda(int weights[4]);