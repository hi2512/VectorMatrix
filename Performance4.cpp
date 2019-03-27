#include <iostream>
#include <papi.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <cpuid.h>
//#include <mkl.h>


using namespace std;

void clearCache() {
	int size = 1024 * 1024 * 40;
	char * ar = (char *) malloc(size);
	int i;
	for(i = 0; i < size; i++) {
		ar[i] = rand();
	}
	free(ar);
}

void printMatrix(float ** mat, int rows, int columns) {
	int i, j;
	for (i = 0; i < rows; i++) {
		printf("[ ");
		for (j = 0; j < columns; j++) {
			printf("%f ", mat[i][j]);
		}
		printf("]\n");
	}
	printf("\n");
}

//hw page says use this
float ** allocate2DArrayOfFloat(int x, int y) {
	int TypeSize = sizeof(float);
	float ** ppi = (float **) calloc(x, sizeof(float*));
	float * pool = (float *) calloc(x, y * TypeSize);
	unsigned char * curPtr = (unsigned char *) pool;
	int i;
	if (!ppi || !pool) { /* Quit if either allocation failed */
		if (ppi)
			free(ppi);
		if (pool)
			free(pool);
		return NULL;
	}
	/* Create row vector */
	//enters the locations into ppi
	for (i = 0; i < x; i++) {
		*(ppi + i) = (float *) curPtr;
		curPtr += y * TypeSize;
	}
	return ppi;
}

double ** getMat(int rows, int columns) {
	int i;
	double ** mat1 = (double **) calloc(rows, sizeof(double *));
	for (i = 0; i < rows; i++) {
		mat1[i] = (double *) calloc(columns, sizeof(double));
	}
	return mat1;
}

float ** mm(float ** mat1, int rows1, int columns1, float ** mat2, int rows2, int columns2,
		float ** resMat) {
	if (columns1 != rows2) {
		printf("CANNOT DO MATRIX MULTIPLICATION\n");
		return NULL;
	}
	int evt[2] = {PAPI_L1_TCM, PAPI_FP_OPS};
	long long vals[2];

	PAPI_start_counters(evt, 2);

	int i, j, k;
	for (i = 0; i < rows1; i++) {
		for (k = 0; k < columns1/*or rows2*/; k++) {
			for (j = 0; j < columns2; j++) {
				resMat[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	PAPI_stop_counters(vals, 2);
	printf("Standard L1 cache misses: %.16lld FLOPS: %.16lld\n", vals[0], vals[1]);
	return resMat;
}

float ** blockmm(float ** mat1, int rows1, int columns1, float ** mat2, int rows2, int columns2,
		float ** resMat, int NU, int MU) {
	if (columns1 != rows2) {
		printf("CANNOT DO MATRIX MULTIPLICATION\n");
		return NULL;
	}
	int NB = columns1 /*or rows 2*/;

	if (NB % NU) {
		cout << "cleanup required" << endl;
		return NULL;
	}
	//l1 cache is 32kb, 32 * 1024, 90~ ^ 2
	for (int i = 0; i < NB; i += NU) {
		for (int k = 0; k < NB; k++) {
			for (int j = 0; j < NB; j += MU) {

				for (int n = 0; n < NU; n++) {
					for (int m = 0; m < MU; m++) {
						resMat[i + n][j + m] += mat1[i + n][k] * mat2[k][j + m];
					}
				}
				//resMat[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	return resMat;
}

float ** bmm(float ** mat1, int rows1, int columns1, float ** mat2, int rows2, int columns2,
		float ** resMat, int NU, int MU) {
	if (columns1 != rows2) {
		printf("CANNOT DO MATRIX MULTIPLICATION\n");
		return NULL;
	}
	int NB = columns1 /*or rows 2*/;

	if (NB % NU) {
		cout << "cleanup required" << endl;
		return NULL;
	}
	int evt[2] = {PAPI_L1_TCM, PAPI_FP_OPS};
	long long vals[2];

	PAPI_start_counters(evt, 2);

	//l1 cache is 32kb, 32 * 1024, 90~ ^ 2
	for (int i = 0; i < NB; i += MU) {
		for (int k = 0; k < NB; k++) {
			for (int j = 0; j < NB; j += NU) {

				for (int n = 0; n < NU; n++) {
					for (int m = 0; m < MU; m++) {
						resMat[i + n][j + m] += mat1[i + n][k] * mat2[k][j + m];
					}
				}
				//resMat[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	PAPI_stop_counters(vals, 2);
	printf("Standard block L1 cache misses: %.16lld FLOPS: %.16lld\n", vals[0], vals[1]);
	return resMat;
}

float ** vmm(float ** mat1, int rows1, int columns1, float ** mat2, int rows2, int columns2,
		float ** resMat) {
	if (columns1 != rows2) {
		printf("CANNOT DO MATRIX MULTIPLICATION\n");
		return NULL;
	}
	if (rows1 % 4) {
		cout << "not divisible by 4" << endl;
		return NULL;
	}
	float temp[4];
	float a[4];
	float b[4];
	int NB = rows1 / 4;
	int evt[2] = {PAPI_L1_TCM, PAPI_FP_OPS};
	long long vals[2];

	PAPI_start_counters(evt, 2);
	__m128 rA, rB, rC;
	for (int i = 0; i < rows1; i++) {
		for (int k = 0; k < columns1; k += 4) {
			rC = _mm_set1_ps(0.f);
			//rC = _mm_load_ps(resMat[i] + 4 * k);
			for (int j = 0; j < columns2; j++) {
				rA = _mm_load_ps(mat1[i] + k);
				_mm_store_ps(a, rA);
				rB = _mm_set_ps(mat2[k + 3][j], mat2[k + 2][j], mat2[k + 1][j], mat2[k][j]);
				_mm_store_ps(b, rB);
				//rC = _mm_add_ps(rC, _mm_mul_ps(rA, rB));
				_mm_store_ps(temp, _mm_mul_ps(rA, rB));
				/*
				cout << "a vals: " << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << endl;
				cout << "b vals: " << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << endl;
				cout << "temp vals: " << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
				*/
				resMat[i][j] += temp[0] + temp[1] + temp[2] + temp[3];
			}
			//_mm_store_ps(resMat[i], rC);
			//resMat[i][j] += mat1[i][k] * mat2[k][j];
			//printMatrix(resMat, rows1, rows1);
		}
	}
	PAPI_stop_counters(vals, 2);
	printf("Vectorized L1 cache misses: %.16lld FLOPS: %.16lld\n", vals[0], vals[1] * 4);

	return resMat;
}

float ** vbmm(float ** mat1, int rows1, int columns1, float ** mat2, int rows2, int columns2,
		float ** resMat) {
	if (columns1 != rows2) {
		printf("CANNOT DO MATRIX MULTIPLICATION\n");
		return NULL;
	}
	if (rows1 % 4) {
		cout << "not divisible by 4" << endl;
		return NULL;
	}
	int NB = rows1 / 4;
	int evt[2] = {PAPI_L1_TCM, PAPI_FP_OPS};
	long long vals[2];

	PAPI_start_counters(evt, 2);
	__m128 rA, rB, rC;
	for (int i = 0; i < rows1; i++) {
		for (int k = 0; k < NB; k++) {
			rC = _mm_set1_ps(0.f);
			//rC = _mm_load_ps(resMat[i] + 4 * k);
			for (int j = 0; j < columns2; j++) {
				rA = _mm_load_ps(mat2[j] + 4 * k);
				rB = _mm_set1_ps(*(mat1[i] + j));
				rC = _mm_add_ps(rC, _mm_mul_ps(rA, rB));
			}
			_mm_store_ps(resMat[i] + 4 * k, rC);
			//printMatrix(resMat, rows1, rows1);
		}
	}
	PAPI_stop_counters(vals, 2);
	printf("Vectorized block L1 cache misses: %.16lld FLOPS: %.16lld\n", vals[0], vals[1] * 4);

	return resMat;
}

float ** imm(float ** mat1, int rows1, int columns1, float ** mat2, int rows2, int columns2,
		float ** resMat) {
	if (columns1 != rows2) {
		printf("CANNOT DO MATRIX MULTIPLICATION\n");
		return NULL;
	}

	int evt[2] = {PAPI_L1_TCM, PAPI_FP_OPS};
	long long vals[2];

	/*
	PAPI_start_counters(evt, 1);
	for (int i = 0; i < rows1; i++) {
		for (int k = 0; k < columns1; k++) {
			for (int j = 0; j < columns2; j++) {
			}
			//printMatrix(resMat, rows1, rows1);
		}
	}
	PAPI_stop_counters(vals, 1);
	*/
	//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows1, columns1, rows2, mat1[0], rows1, mat2[0], rows2, resMat[0], columns1);

	//printf("Intel L1 cache misses: %.16lld\n", vals[0]);

	return resMat;
}

int main() {
	PAPI_library_init(PAPI_VER_CURRENT);

	const int matSize = 8;
	float ** mat1 = allocate2DArrayOfFloat(matSize, matSize);
	float ** mat2 = allocate2DArrayOfFloat(matSize, matSize);
	float ** resMat = allocate2DArrayOfFloat(matSize, matSize);
	float ** resMat2 = allocate2DArrayOfFloat(matSize, matSize);
	float ** resMat3 = allocate2DArrayOfFloat(matSize, matSize);
	float ** resMat4 = allocate2DArrayOfFloat(matSize, matSize);
	float ** resMat5 = allocate2DArrayOfFloat(matSize, matSize);

	int add = 0;
	for (int i = 0; i < matSize; i++) {
		for (int j = 0; j < matSize; j++) {
			mat1[i][j] = add++;
			//mat1[i][j] = 2;
		}
	}
	add = 1;
	for (int i = 0; i < matSize; i++) {
		for (int j = 0; j < matSize; j++) {
			mat2[i][j] = add++;
			//mat2[i][j] = 2;
		}
	}
	printMatrix(mat1, matSize, matSize);
	printMatrix(mat2, matSize, matSize);

	const int NU = 1;
	const int MU = 5;

	clearCache();
	mm(mat1, matSize, matSize, mat2, matSize, matSize, resMat);
	clearCache();
	vmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat3);
	clearCache();
	vbmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat4);
	/*
	clearCache();
	mm(mat1, matSize, matSize, mat2, matSize, matSize, resMat);
	//blockmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat, NU, MU);
	clearCache();
	bmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat2, NU, MU);
	clearCache();
	vmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat3);
	clearCache();
	vbmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat4);
	clearCache();
	imm(mat1, matSize, matSize, mat2, matSize, matSize, resMat5);
	*/
	/*
	cout << "standard: " << endl;
	printMatrix(resMat, matSize, matSize);
	cout << "vector: " << endl;
	printMatrix(resMat2, matSize, matSize);
	cout << "block: " << endl;
	printMatrix(resMat3, matSize, matSize);
	cout << "intel: " << endl;
	printMatrix(resMat4, matSize, matSize);
	*/
	cout << "standard: " << endl;
	printMatrix(resMat, matSize, matSize);
	cout << "vector: " << endl;
	printMatrix(resMat3, matSize, matSize);
	cout << "vector block: " << endl;
	printMatrix(resMat4, matSize, matSize);

	return 0;
}
