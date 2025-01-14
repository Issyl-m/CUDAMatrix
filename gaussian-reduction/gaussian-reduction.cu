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
const int DEFAULT_SHARED_MEM = 32*32 * 4 * 2; // 32768 bytes

// Structures

struct __align__(16) GaussianEliminationCtx {
  int prime_number;
  int mod_p_pivot_seek_from_row;
  int mod_p_curr_col;
  int mod_p_row_to_push;
  int mod_p_pivot_val; // TODO: possibly unused
};

// Kernels and devices 

/*
  Mod p linear algebra routines for small primes (<MAX_INT)
  TODO:
  - Separate routines for p >> 1, small primes and p = 2
  - Sparse matrix Gaussian elimination (TODO), multiplication (SELLP)
*/

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

__device__ int mod_2_inverse(int a) {
  return ((a & 0x00000001) << 1) - 1; // Output: -1 if error
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

__global__ void mod_p_gaussian_backward_reduction(GaussianEliminationCtx *__restrict__ ctx, int *__restrict__ A, int n_rows, int n_cols, int curr_col, int *__restrict__ pivot_locations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
 
  int pivot_row = pivot_locations[curr_col];
  
  if (curr_col + x + 1 >= n_cols || y >= pivot_row || pivot_row == -1) { //
    return;
  }
  
  int curr_entry = A[y * n_cols + curr_col + x + 1];
  curr_entry -= A[y * n_cols + curr_col] * A[pivot_row * n_cols + curr_col + x + 1] * 1;
  A[y * n_cols + curr_col + x + 1] = positive_modulo(curr_entry, (*ctx).prime_number); // TODO: performance, int size
}

__global__ void mod_p_gaussian_backward_clean_column(GaussianEliminationCtx *__restrict__ ctx, int *__restrict__ A, int n_rows, int n_cols, int curr_col, int *__restrict__ pivot_locations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  int pivot_row = pivot_locations[curr_col];

  if (x >= pivot_row || pivot_row == -1 ) {
    return;
  }

  A[x*n_cols + curr_col] = 0;
}

__global__ void mod_p_gaussian_clean_column(GaussianEliminationCtx *__restrict__ ctx, int *__restrict__ A, int n_rows, int n_cols, int curr_col) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= n_rows || x <= (*ctx).mod_p_pivot_seek_from_row-1 || (*ctx).mod_p_row_to_push == -1 ) {
    return;
  }

  A[x*n_cols + curr_col] = 0;
}

__global__ void mod_p_gaussian_elimination(GaussianEliminationCtx *__restrict__ ctx, int *__restrict__ A, int n_rows, int n_cols, int curr_col) { 
  /*
    Integer matrix routine.
  */
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_row = blockIdx.y * blockDim.y + threadIdx.y;
  
  int mod_p_pivot_seek_from_row = (*ctx).mod_p_pivot_seek_from_row;

  // (*ctx).mod_p_row_to_push == -1: zero column
  if (curr_row <= mod_p_pivot_seek_from_row-1 || \
      curr_row >= n_rows || \
      curr_col + x + 1 >= n_cols || \
      (*ctx).mod_p_row_to_push == -1) {
    return;
  }

  int curr_entry = A[curr_row*n_cols + curr_col + x + 1];
  curr_entry -= (A[curr_row*n_cols + curr_col] * A[(mod_p_pivot_seek_from_row-1)*n_cols + curr_col + x + 1]) * 1;
  A[curr_row*n_cols + curr_col + x + 1] = positive_modulo(curr_entry, (*ctx).prime_number); // TODO: performance, int size
}

__global__ void mod_p_exchange_rows(GaussianEliminationCtx *__restrict__ ctx, int *__restrict__ A, int n_rows, int n_cols, int curr_col) { 
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

  // mod_p_inverse for reduction purposes
  A[n_cols*src_row + curr_col + x] *= mod_p_inverse((*ctx).prime_number, (*ctx).mod_p_pivot_val);

  if (src_row == dst_row) {
      return;
  }

  tmp_input = A[n_cols*dst_row + curr_col + x];

  A[n_cols*dst_row + curr_col + x] = A[n_cols*src_row + curr_col + x]; 
  A[n_cols*src_row + curr_col + x] = tmp_input;
}

__global__ void mod_p_seek_row_to_push(GaussianEliminationCtx *__restrict__ ctx, int *__restrict__ A, int n_rows, int n_cols, int curr_col, int *__restrict__ pivot_locations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  int pivot_candidate_val;

  if (x < (*ctx).mod_p_pivot_seek_from_row || x >= n_rows) {
      return;
  }

  if (atomicCAS(&((*ctx).mod_p_curr_col), curr_col - 1, curr_col) == curr_col - 1) { // TODO: check performance        
    (*ctx).mod_p_row_to_push = -1;                                                                                                                                    
    pivot_locations[curr_col] = -1;
  }

  pivot_candidate_val = A[x * n_cols + curr_col];

  if (pivot_candidate_val % (*ctx).prime_number != 0) { // TODO: slow for variable prime_number
    if (atomicCAS(&((*ctx).mod_p_row_to_push), -1, x) == -1) {
      pivot_locations[curr_col] = (*ctx).mod_p_pivot_seek_from_row;
      (*ctx).mod_p_pivot_seek_from_row += 1;
      (*ctx).mod_p_pivot_val = pivot_candidate_val;
    }
  }
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

  size_t M_rows = 40000; // 5x6
  size_t M_cols = 80000;

  int prime_number = 5;

  vector<int> h_M(M_rows * M_cols, 1);

  generate(h_M.begin(), h_M.end(), [] {
    static int i = 0;
    int r = 0;

    int row = i / 40000;
    int col = i % 80000;

    if (i % 7 < 5 || i % 29 > 20) 
      r = (2 + i) % 5;
    if (col <= 10 || row > 40000 || row < 1)
      r = 0;
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

  print_matrix(prime_number, h_M, M_rows, M_cols);

  // Device TODO: split into separate procedures

  int h_M_size = M_rows*M_cols*sizeof(int);  
  int *d_M, *d_pivot_locations;
  GaussianEliminationCtx *d_ctx;

  cudaMalloc(&d_M, h_M_size);
  cudaMalloc(&d_pivot_locations, sizeof(int)*M_cols);
  cudaMalloc(&d_ctx, sizeof(GaussianEliminationCtx));

  GaussianEliminationCtx h_ctx;
  h_ctx.prime_number = prime_number;
  h_ctx.mod_p_pivot_seek_from_row = 0;
  h_ctx.mod_p_curr_col = 0;
  h_ctx.mod_p_row_to_push = -1;

  cudaMemcpy(d_M, h_M.data(), h_M_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ctx, &h_ctx, sizeof(GaussianEliminationCtx), cudaMemcpyHostToDevice);

  // Run kernels 
  
  dim3 num_threads_2d(DEFAULT_N_THREADS_PER_DIM, DEFAULT_N_THREADS_PER_DIM);
  int num_blocks;
  
  for (int j = 0; j < M_cols; j++) {
    dim3 num_blocks_2d((M_cols - j - 1) / DEFAULT_N_THREADS_PER_DIM + 1, M_rows / DEFAULT_N_THREADS_PER_DIM + 1);

    num_blocks = M_rows / DEFAULT_N_THREADS_PER_DIM + 1;
    mod_p_seek_row_to_push <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_cols, j, d_pivot_locations);
    cudaDeviceSynchronize();                         

    num_blocks = M_cols / DEFAULT_N_THREADS_PER_DIM + 1;               
    mod_p_exchange_rows <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_cols, j);
    cudaDeviceSynchronize();
        
    mod_p_gaussian_elimination <<< num_blocks_2d, num_threads_2d >>> (d_ctx, d_M, M_rows, M_cols, j); 
    cudaDeviceSynchronize();
    
    num_blocks = M_rows / DEFAULT_N_THREADS_PER_DIM + 1;
    mod_p_gaussian_clean_column <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_cols, j);
    cudaDeviceSynchronize();
  }

  for (int i = 0; i < M_cols; i++) {
    dim3 num_blocks_2d((M_cols - i - 1) / DEFAULT_N_THREADS_PER_DIM + 1, M_rows / DEFAULT_N_THREADS_PER_DIM + 1);

    mod_p_gaussian_backward_reduction <<< num_blocks_2d, num_threads_2d >>> (d_ctx, d_M, M_rows, M_cols, i, d_pivot_locations);
    cudaDeviceSynchronize();

    num_blocks = M_rows / DEFAULT_N_THREADS_PER_DIM + 1;
    mod_p_gaussian_backward_clean_column <<< num_blocks, DEFAULT_N_THREADS_PER_DIM >>> (d_ctx, d_M, M_rows, M_cols, i, d_pivot_locations); // to avoid more flow control
    cudaDeviceSynchronize();
  }
  
  // Parse data 

  cudaMemcpy(h_M.data(), d_M, h_M_size, cudaMemcpyDeviceToHost);
  
  cudaFree(d_M);
  cudaFree(d_pivot_locations);
  cudaFree(d_ctx);

  print_matrix(prime_number, h_M, M_rows, M_cols);

  // free(h_M);

  return 0;
}

