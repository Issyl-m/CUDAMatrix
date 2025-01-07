/*
Copyright (c) 2025 Andrés Morán (andres.moran.l@uc.cl)
Licensed under the terms of the MIT License (see ./LICENSE).
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

// Constants

const int DEFAULT_N_THREADS_PER_DIM = 32; // max 1024 per block

// Structures

struct GaussianEliminationCtx { 
  int prime_number;
  int mod_p_pivot_seek_from_row;
  int mod_p_curr_col;
  int mod_p_row_to_push;
  int mod_p_pivot_val;
};

// Kernels and devices

__device__ int positive_modulo(int i, int n) { 
  /*
    Input: i arbitrary, n: modulus, n > 0
    Output: positive i % n representative
  */
  return (i % n + n) % n;
}

__device__ int mod_2_inverse(int a) {
  if (a & 0x00000001) return 1;
  return -1; 
}

__device__ int mod_3_inverse(int a) {
  int b = a % 3;
  if (b == 0) return -1;
  return b;
}

__device__ int mod_p_inverse(int p, int a) {
  /*
    Extended Euclidean division
    Mod p multiplicative inverse
    Output: x_1 = a^{-1}
  */
  int u = a;
  int v = p;
  
  if (p == 2)
    return mod_2_inverse(a);

  if (p == 3)
    return mod_3_inverse(a);

  if (u % v == 0) {
    return -1;
  }

  u = positive_modulo(a, p);
  
  int x_1 = 1;
  int x_2 = 0;
  
  while (u != 1) {
    int q = v/u;
    int r = v - q*u;
    int x = x_2 - q*x_1;

    v = u;
    u = r;
    
    x_2 = x_1;
    x_1 = x;
  }

  return positive_modulo(x_1, p);
}

__global__ void mod_p_gaussian_clean_column(GaussianEliminationCtx *ctx, int *A, int n_rows, int n_cols, int curr_col) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= n_rows || x <= (*ctx).mod_p_pivot_seek_from_row-1 || (*ctx).mod_p_row_to_push == -1 ) {
    return;
  }

  A[x*(n_cols+1) + curr_col] = 0;
}

__global__ void mod_p_gaussian_elimination(GaussianEliminationCtx *ctx, int *A, int n_rows, int n_cols, int curr_col, int curr_row) { 
  /*
    Integer matrix routine.
  */
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (curr_row <= (*ctx).mod_p_pivot_seek_from_row-1 || n_cols <= curr_col + x || (*ctx).mod_p_row_to_push == -1 || x == 0) { 
    // if (*ctx).mod_p_row_to_push == -1 we have a zero column
    return;
  }
 
  // curr_col = curr_col + 1. Ensure to remove the (curr_row*n_cols + curr_col)-garbage
  A[curr_row*n_cols + curr_col + x] -=\
    (A[curr_row*n_cols + curr_col] * A[((*ctx).mod_p_pivot_seek_from_row-1)*n_cols + curr_col + x]);
  A[curr_row*n_cols + curr_col + x] = positive_modulo(A[curr_row*n_cols + curr_col + x], (*ctx).prime_number);
}

__global__ void mod_p_exchange_rows(GaussianEliminationCtx *ctx, int *A, int n_rows, int n_cols, int curr_col) { 
  /*
     Integer matrix routine.
  */
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  int tmp_input;
  int src_row;
  int dst_row;

  src_row = (*ctx).mod_p_row_to_push;
  if (src_row == -1 || x + curr_col >= n_cols) {
      return;
  }

  dst_row = (*ctx).mod_p_pivot_seek_from_row - 1; // skip last found

  if (src_row == dst_row) {
      return;
  }

  tmp_input = A[n_cols*dst_row + curr_col + x];

  // mod_p_inverse for reduction purposes
  A[n_cols*dst_row + curr_col + x] = A[n_cols*src_row + curr_col + x] * mod_p_inverse((*ctx).prime_number, mod_p_pivot_val);
  A[n_cols*src_row + curr_col + x] = tmp_input;
}

__global__ void mod_p_seek_row_to_push(GaussianEliminationCtx *ctx, int *A, int n_rows, int n_cols, int curr_col) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  int pivot_candidate_val;

  if (x < (*ctx).mod_p_pivot_seek_from_row || x >= n_rows) {
      return;
  }

  if (atomicCAS(&((*ctx).mod_p_curr_col), curr_col - 1, curr_col) == curr_col - 1) { // TODO: check performance        
    (*ctx).mod_p_row_to_push = -1;                                                                                                                                    
  }

  pivot_candidate_val = A[x * n_cols + curr_col];

  if (pivot_candidate_val % (*ctx).prime_number != 0) {
    if (atomicCAS(&((*ctx).mod_p_row_to_push), -1, x) == -1) {
      (*ctx).mod_p_pivot_seek_from_row += 1;
      (*ctx).mod_p_pivot_val = pivot_candidate_val;
    }
  }
}
  
// Host utils 

__host__ void print_matrix(vector<int> &matrix, int n_rows, int n_cols) { 
  /*
    Integer matrix routine
  */
 
  printf("[+] %u x %u matrix\n", n_rows, n_cols);
  
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      printf("%d\t\t", matrix[i*n_cols+j]);
    }
    printf("\n");
  }
  printf("\n");
}

// Main 

int main(int argc, char *argv[]) {
  // Initialize data: sample matrix 

  size_t M_rows = 5;
  size_t M_cols = 6;

  vector<int> h_M(M_rows * M_cols);
  
  for (int i = 0; i < M_rows * M_cols; i++) {
    if (true) {
      h_M[i] = i % M_cols + M_cols * (i / M_cols);
    } else {
      h_M[i] = 0;
    }
  }

  for (int i = 0; i < 5; i++){
      h_M[6*i] = 0;
  }

  h_M[1] = 0;

  print_matrix(h_M, M_rows, M_cols);

  // Device TODO: split into separate procedures

  int h_M_size = M_rows*M_cols*sizeof(int);  
  int *d_M;
  GaussianEliminationCtx *d_ctx;

  cudaMalloc(&d_M, h_M_size);
  cudaMalloc(&d_ctx, sizeof(GaussianEliminationCtx));

  GaussianEliminationCtx h_ctx;
  h_ctx.prime_number = 5;
  h_ctx.mod_p_pivot_seek_from_row = 0;
  h_ctx.mod_p_curr_col = 0;
  h_ctx.mod_p_row_to_push = -1;

  cudaMemcpy(d_M, h_M.data(), h_M_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ctx, &h_ctx, sizeof(GaussianEliminationCtx), cudaMemcpyHostToDevice);

  // Run kernels 

  for (int j = 0; j < M_cols; j++) {
    int block_size;
    int num_blocks;

    block_size = M_rows / DEFAULT_N_THREADS_PER_DIM + 1; // we could have threads outside the matrix                                   
    num_blocks = M_rows / block_size + 1;
    mod_p_seek_row_to_push <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_cols, j);
    cudaDeviceSynchronize();                         

    block_size = (M_cols - j) / DEFAULT_N_THREADS_PER_DIM + 1;
    num_blocks = M_cols / block_size + 1;               
    mod_p_exchange_rows <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_cols, j);
    cudaDeviceSynchronize();

    for (int i = 0; i < M_rows; i++) {
      block_size = (M_cols - j + 1) / DEFAULT_N_THREADS_PER_DIM + 1;
      num_blocks = M_cols / block_size + 1;
      mod_p_gaussian_elimination <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_cols, j, i);
    }
    cudaDeviceSynchronize();
    
    block_size = M_rows / DEFAULT_N_THREADS_PER_DIM + 1;
    num_blocks = M_rows / block_size + 1;
    mod_p_gaussian_clean_column <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_rows, j);
    cudaDeviceSynchronize();
  }

  // TODO: backward substitution

  // Parse data 

  cudaMemcpy(h_M.data(), d_M, h_M_size, cudaMemcpyDeviceToHost);
  
  cudaFree(d_M);
  cudaFree(d_ctx);

  print_matrix(h_M, M_rows, M_cols);

  // free(h_M);

  return 0;
}

