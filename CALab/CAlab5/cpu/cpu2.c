#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<immintrin.h>
#include<math.h>
#include<string.h>

int N = (1 << 10);

int gemm_verify(float *A, float *B, float *C); // you can use inline function
void gemm_avx(float *A, float *B, float *C); // you can use inline function
void Initialization(float *M);
void Reverse_Matrix(float *M);

int main()
{
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
    gemm_avx(A, B, C);
    // use gemm_baseline verify gemm_avx

    end = clock();
    printf("CPU time = %lf s\n", (double)(end - start) / CLOCKS_PER_SEC);
    if(gemm_verify(A, B, C) == 0)
    {
        printf("Result is Correct!\n");
    }
    else
    {
        printf("Result is Incorrect\n");
    }
    free(A);
    free(B);
    free(C);
    return 0;
}


int gemm_verify(float *A, float *B, float *C) {
    float* T = (float*)malloc(N * N * sizeof(float));
    memset(T, 0, sizeof(float) * N * N);
    Reverse_Matrix(B);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                T[i * N + j] += A[i * N + k] * B[j * N + k];
            }
        }
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(fabs(C[i * N + j] - T[i * N + j]) > pow(10, -1))
                return 1;
        }
    }
    return 0;
}

void gemm_avx(float *A, float *B, float *C) {
    __m256 Va, Vc;
    __m256 avx_mul;
    __m256 avx_sum = _mm256_setzero_ps();
    float *temp = (float *)malloc(8 * sizeof(float));

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k += 8){
                Va = _mm256_loadu_ps(&A[i * N + k]);
                for(int l = 0; l < 8; l++)
                {
                    temp[l] = B[(k + l) * N + j];
                }
                Vc = _mm256_loadu_ps(temp);
                avx_mul = _mm256_mul_ps(Va, Vc);
                avx_sum = _mm256_add_ps(avx_sum, avx_mul);
            }
            for(int k = 0; k < 8; k++){
                C[i * N + j] += avx_sum[k];
            }
            avx_sum = _mm256_setzero_ps();
        }
    }
}

void Initialization(float *M)
{
    for(int i = 0; i < N * N; i++){
        M[i] = (rand() % 10) / 1.0;
    }
}

void Reverse_Matrix(float *M)
{
    float temp;
    for(int i = 1; i < N; i++)
    {
        for(int j = 0; j < i; j++)
        {
            temp = M[i * N + j];
            M[i * N + j] = M[j * N + i];
            M[j * N + i] = temp;
        }
    }
}
