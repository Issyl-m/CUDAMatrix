/*
Copyright (c) 2025 Andrés Morán (andres.moran.l@uc.cl)
Licensed under the terms of the MIT License (see ./LICENSE).
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm> // tests

using std::cout;
using std::vector;
using std::generate;

// Constants

const int DEFAULT_N_THREADS_PER_DIM = 32; // max 1024 per block
const int DEFAULT_SHARED_MEM = 32*32 * 2; // 32768 bytes


// Kernels and devices 

__device__ int positive_modulo(int i, int n) { 
  /*
    Input: i arbitrary, n: modulus, n > 0
    Output: positive i % n representative
  */
  if (n == 2) {
    return i & 0x00000001;
  }
  return (i % n + n) % n;
}

__global__ void mod_p_matrix_multiplication(int prime_number, int *__restrict__ M, int M_rows, int M_cols, int *__restrict__ N, int N_rows, int N_cols, int *O) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int output = 0;

  __shared__ int __align__(8) s_M[DEFAULT_SHARED_MEM];
  __shared__ int __align__(8) s_N[DEFAULT_SHARED_MEM];

  if (x >= M_rows || y >= N_cols) {
    return;
  }

  for (int block_num = 0; block_num <= M_cols / DEFAULT_N_THREADS_PER_DIM; block_num++) {
    int i = threadIdx.x * DEFAULT_N_THREADS_PER_DIM + threadIdx.y;
    int j = block_num * DEFAULT_N_THREADS_PER_DIM + threadIdx.y;
    int k = block_num * DEFAULT_N_THREADS_PER_DIM + threadIdx.x;

    if (j >= M_cols) {
      s_M[i] = 0;
    } else {
      s_M[i] = M[x * M_cols + j];
    } 

    if (k >= M_cols) {
      s_N[i] = 0;
    } else {
      s_N[i] = N[k * N_cols + y];
    } 

    __syncthreads();
    
    for (int k = 0; k < DEFAULT_N_THREADS_PER_DIM; k++) {
      output = positive_modulo(prime_number, output + s_M[threadIdx.x * DEFAULT_N_THREADS_PER_DIM + k] * s_N[k * DEFAULT_N_THREADS_PER_DIM + threadIdx.y]);
    }

    __syncthreads();
  }

  O[x * N_cols + y] = output;
}

// Host utils 

__host__ void print_matrix(int prime_number, vector<int> &matrix, int n_rows, int n_cols) { 
  /*
    Integer matrix routine
  */
 
  printf("[+] %u x %u matrix\n", n_rows, n_cols);
  
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      printf("%d\t\t", ((matrix[i*n_cols+j] % prime_number) + prime_number) % prime_number);
    }
    printf("\n");
  }
  printf("\n");
}

// Main 

int main(int argc, char *argv[]) {
  // Initialize data: sample matrix 

  // size_t M_rows = 5; // 5x6
  // size_t M_cols = 6;

  size_t M_rows = 4; // 5x6
  size_t M_cols = 8;

  int prime_number = 5;

  vector<int> h_M(M_rows * M_cols, 1);

  generate(h_M.begin(), h_M.end(), [] {
    static int i = 0;
    int r = 0;

    int row = i / 8;
    int col = i % 8;

    if (row == col) {
      r = 2;
    } else {
      r = 0;
    }
    i++;
    return r;
  });

  size_t N_rows = 8; // 5x6
  size_t N_cols = 4;

  vector<int> h_N(N_rows * N_cols, 1);

  generate(h_N.begin(), h_N.end(), [] {
    static int i = 0;
    int r = 0;

    int row = i / 4;
    int col = i % 4;

    if (row == col) {
      r = 3;
    } else {
      r = 0;
    }
    i++;
    return r;
  });
  
  // h_M = {
  //    0, 0, 2, 3, 4, 3,
  //    0, 2, 3, 4, 0, 3,
  //    0, 3, 4, 0, 1, 3,
  //    0, 4, 0, 1, 2, 3,
  //    0, 0, 1, 2, 3, 3,
  // };

  vector<int> h_O(M_rows * N_cols);

  print_matrix(prime_number, h_M, M_rows, M_cols);
  print_matrix(prime_number, h_N, N_rows, N_cols);
  print_matrix(prime_number, h_O, M_rows, N_cols);

  // Device TODO: split into separate procedures

  int h_M_size = M_rows*M_cols*sizeof(int);  
  int *d_M;

  int h_N_size = N_rows*N_cols*sizeof(int);  
  int *d_N;

  int h_O_size = M_rows*N_cols*sizeof(int);  
  int *d_O;
  
  cudaMalloc(&d_M, h_M_size);
  cudaMalloc(&d_N, h_N_size);
  cudaMalloc(&d_O, h_O_size);

  cudaMemcpy(d_M, h_M.data(), h_M_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N.data(), h_N_size, cudaMemcpyHostToDevice);

  // Run kernels 

  dim3 num_threads_2d(DEFAULT_N_THREADS_PER_DIM, DEFAULT_N_THREADS_PER_DIM);
  dim3 num_blocks_2d((M_rows - 1) / DEFAULT_N_THREADS_PER_DIM + 1, (N_cols - 1) / DEFAULT_N_THREADS_PER_DIM + 1);

  mod_p_matrix_multiplication <<< num_blocks_2d, num_threads_2d >>> (prime_number, d_M, M_rows, M_cols, d_N, N_rows, N_cols, d_O);
  cudaDeviceSynchronize();                         

  // Parse data 

  cudaMemcpy(h_O.data(), d_O, h_O_size, cudaMemcpyDeviceToHost);

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_O);

  print_matrix(prime_number, h_O, M_rows, N_cols);
  
  // free(h_M);
  // free(h_N);
  // free(h_O);

  return 0;
}
