#include <iostream>
#include <papi.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

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
	int evt[1] = {PAPI_L1_TCM};
	long long vals[1];

	PAPI_start_counters(evt, 1);

	int i, j, k;
	for (i = 0; i < rows1; i++) {
		for (k = 0; k < columns1/*or rows2*/; k++) {
			for (j = 0; j < columns2; j++) {
				resMat[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	PAPI_stop_counters(vals, 1);
	printf("L1 cache misses: %.16lld\n", vals[0]);
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

	int NB = rows1 / 4;
	int evt[1] = {PAPI_L1_TCM};
	long long vals[1];

	PAPI_start_counters(evt, 1);
	__m128 rA, rB, rC;
	for (int i = 0; i < rows1; i++) {
		for (int k = 0; k < NB; k++) {
			rC = _mm_set1_ps(0.f);
			for (int j = 0; j < columns2; j++) {
				rA = _mm_load_ps(mat2[j] + 4 * k);
				rB = _mm_set1_ps(*(mat1[i] + j));
				rC = _mm_add_ps(rC, _mm_mul_ps(rA, rB));
			}
			_mm_store_ps(resMat[i] + 4 * k, rC);
			//printMatrix(resMat, rows1, rows1);
		}
	}
	PAPI_stop_counters(vals, 1);
	printf("L1 cache misses: %.16lld\n", vals[0]);

	return resMat;
}

int main() {
	PAPI_library_init(PAPI_VER_CURRENT);

	const int matSize = 20;
	float ** mat1 = allocate2DArrayOfFloat(matSize, matSize);
	float ** mat2 = allocate2DArrayOfFloat(matSize, matSize);
	float ** resMat = allocate2DArrayOfFloat(matSize, matSize);
	float ** resMat2 = allocate2DArrayOfFloat(matSize, matSize);

	int add = 0;
	for (int i = 0; i < matSize; i++) {
		for (int j = 0; j < matSize; j++) {
			mat1[i][j] = add++;
		}
	}
	add = 1;
	for (int i = 0; i < matSize; i++) {
		for (int j = 0; j < matSize; j++) {
			mat2[i][j] = add++;
		}
	}
	printMatrix(mat1, matSize, matSize);
	printMatrix(mat2, matSize, matSize);

	const int NU = 80;
	const int MU = 80;

	mm(mat1, matSize, matSize, mat2, matSize, matSize, resMat);
	//blockmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat, NU, MU);
	vmm(mat1, matSize, matSize, mat2, matSize, matSize, resMat2);
	cout << "standard: " << endl;
	printMatrix(resMat, matSize, matSize);
	cout << "vector: " << endl;
	printMatrix(resMat2, matSize, matSize);
	return 0;
}
