#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<string.h>

int N = (1 << 9);

void Initialization(float* M);

void gemm_baseline(float* A, float* B, float* C);

int main(void){
    clock_t start, end;
    //malloc A,B,C
    
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    srand((unsigned)time(NULL));
    Initialization(A);
    Initialization(B);

    memset(C, 0.0, N * N * sizeof(float));

    start = clock();
    gemm_baseline(A, B, C);
    end = clock();
    printf("CPU time = %lf s\n", (double)(end - start) / CLOCKS_PER_SEC);
    free(A);
    free(B);
    free(C);
    return 0;
}

void gemm_baseline(float* A, float* B, float* C)
{
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void Initialization(float* M)
{
    for(int i = 0; i < N * N; i++)
    {
        M[i] = (rand() % 10) / 1.0;
    }
}