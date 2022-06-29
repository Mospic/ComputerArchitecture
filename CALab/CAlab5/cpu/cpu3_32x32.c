#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<string.h>
#include<immintrin.h>
#include<math.h>

#define BLOCKSIZE 32
#define MVL 8

int N = (1 << 10);
void printboard(float *A);
void printboardN(float *A, int n);

int gemm_verify(float *A, float *B, float *C);
void gemm_avx_block(float *A, float *B, float *C);
void Initialization(float *M);
void Reverse_Matrix(float *M);

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
    Reverse_Matrix(B);

    
    start = clock();

    gemm_avx_block(A, B, C);
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
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                T[i * N + j] += A[i * N + k] * B[j * N + k];
            }
        }
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(fabs(C[i * N + j] - T[i * N + j]) > pow(10, 0))
                return 1;
        }
    }
    return 0;
}


void Initialization(float* M)
{
    for(int i = 0; i < N * N; i++)
    {
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

void gemm_avx_block(float *A, float *B, float *C)
{
    for (int i = 0; i < N; i += BLOCKSIZE)
    {
        for (int j = 0; j < N; j += BLOCKSIZE)
        {
            __m256 Vec[BLOCKSIZE][BLOCKSIZE/MVL], a[BLOCKSIZE][BLOCKSIZE/MVL], b[BLOCKSIZE][BLOCKSIZE/MVL], temp;
            int ans;
            for(int m = 0; m < BLOCKSIZE; m++)
            {
                for(int l = 0 ; l < BLOCKSIZE/MVL; l++)
                    Vec[m][l] = _mm256_setzero_ps();
            }
            for(int k = 0; k < N; k += BLOCKSIZE)
            { 
                for(int m = 0; m < BLOCKSIZE; m++)
                {
                    for(int l = 0 ; l < BLOCKSIZE/MVL; l++)
                    {
                        a[m][l] = _mm256_loadu_ps(&A[(i + m) * N + k + l * MVL]);
                        b[m][l] = _mm256_loadu_ps(&B[(j + m) * N + k + l * MVL]);
                    }
                }
                for(int l = 0; l < BLOCKSIZE; l++)
                {
                    for(int m = 0; m < BLOCKSIZE; m++)
                    {
                        ans = 0;
                        for(int n = 0; n < BLOCKSIZE/MVL; n++)
                        {
                            temp = _mm256_mul_ps(a[l][n], b[m][n]);
                            for(int n = 0; n < BLOCKSIZE; n++)
                                ans += (int)temp[n];
                        }
                        Vec[l][m / MVL][m % MVL] += ans;
                    }
                } 
            }
            for(int k = 0; k < BLOCKSIZE; k++)
            {
                for(int l = 0; l < BLOCKSIZE / MVL;l++)
                {
                    _mm256_storeu_ps(C + (i + k) * N + j + l * MVL, Vec[k][l]);
                }
            }
        }
    }
}


void printboard(float *A)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%f ",A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printboardN(float *A, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%f\t",A[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
}