// #include "slave.h"
// #include "simd.h"
#include <math.h>
#include "pcg_def.h"
#include <string.h>
#include <crts.h>

typedef struct
{
	int rows;
	int *row_off;
	int *cols;
	double *data;
	int data_size;
} CsrMatrix;

typedef struct
{
	double *pre_mat_val;
	double *preD;
} Precondition;

typedef struct
{
	double *upper;
	double *lower;
	double *diag;
	int *uPtr;
	int *lPtr;
	int faces;
	int cells;
} LduMatrix;

__thread_local double *val; // non-zero elements per core
__thread_local int *cols;	// column of non-zero elements per core
__thread_local int *row_off;
__thread_local double *result;
__thread_local int *diag_pos_slave;
__thread_local double *diag_inverse;
__thread_local double *diag;

__thread_local double *r;
__thread_local double *x;
__thread_local double *p;

extern int *row_off_host;
extern int *cols_host;

typedef struct
{
	double *r;
	double *z;
	double *p;
	double *Ax;
	double sumprod;
	double sumprod_old;
	double residual;
	double alpha;
	double beta;

	double *x;
	double *source;
} PCG;

typedef struct
{
	const CsrMatrix *csr_matrix;
	double *val;
	int *diag_pos;
	const LduMatrix *ldu_matrix;
	const PCG *pcg;
} Para_A_ldm_malloc;

void slave_A_ldm_malloc(Para_A_ldm_malloc *para)
{
	Para_A_ldm_malloc slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_A_ldm_malloc));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	const LduMatrix *ldu_matrix = slavePara.ldu_matrix;
	const PCG *pcg = slavePara.pcg;

	// 计算从核接收数组数据长度和接收位置
	int len = csr_matrix->rows / 64;
	int rest = csr_matrix->rows % 64;
	int start_row;
	if (CRTS_tid < rest)
	{
		len++;
		start_row = CRTS_tid * len;
	}
	else
	{
		start_row = CRTS_tid * len + rest;
	}

	row_off = (int *)ldm_malloc(sizeof(int) * (len + 1));

	CRTS_dma_get(row_off, row_off_host + start_row, (len + 1) * sizeof(int));
	// CRTS_dma_get(row_off, slavePara.csr_matrix->row_off + start_row, (len + 1) * sizeof(int));

	int start_core = row_off[0];
	int nnz_per_core = row_off[len] - start_core;
	val = (double *)ldm_malloc(sizeof(double) * nnz_per_core);
	CRTS_dma_get(val, csr_matrix->data + start_core, nnz_per_core * sizeof(double));
	cols = (int *)ldm_malloc(sizeof(int) * nnz_per_core);
	result = (double *)ldm_malloc(sizeof(double) * len);
	CRTS_dma_get(cols, cols_host + start_core, nnz_per_core * sizeof(int));
	// CRTS_dma_get(cols, (slavePara.csr_matrix)->cols + start_core, nnz_per_core * sizeof(int));

	diag_pos_slave = (int *)ldm_malloc(sizeof(int) * len);
	CRTS_dma_get(diag_pos_slave, slavePara.diag_pos + start_row, len * sizeof(int));
	diag = (double *)ldm_malloc(sizeof(double) * len);
	diag_inverse = (double *)ldm_malloc(sizeof(double) * len);
	CRTS_dma_get(diag, ldu_matrix->diag + start_row, len * sizeof(double));

	// kernal fusion
	r = (double *)ldm_malloc(sizeof(double) * len);
	x = (double *)ldm_malloc(sizeof(double) * len);
	p = (double *)ldm_malloc(sizeof(double) * len);
	CRTS_dma_get(x, pcg->x + start_row, len * sizeof(double));

	// if (CRTS_tid == 0)
	// {
	// 	for (int i = 0; i < len; i++)
	// 	{
	// 		printf("diag_pos_slave[%d] = % d\n", i, diag_pos_slave[i]);
	// 	}
	// }

	for (int i = 0; i < len; i++)
	{
		diag_inverse[i] = 1.0 / diag[i];
	}
}

typedef struct
{
	const CsrMatrix *csr_matrix;
	PCG *pcg;
	double *result_x;
} Para_A_ldm_free;

void slave_A_ldm_free(Para_A_ldm_free *para)
{
	Para_A_ldm_free slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_A_ldm_free));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	double *result_x = slavePara.result_x;

	// 计算从核接收数组数据长度和接收位置
	int len = csr_matrix->rows / 64;
	int rest = csr_matrix->rows % 64;
	int start_row;
	if (CRTS_tid < rest)
	{
		len++;
		start_row = CRTS_tid * len;
	}
	else
	{
		start_row = CRTS_tid * len + rest;
	}

	CRTS_dma_put(result_x + start_row, x, len * sizeof(double));
	// athread_ssync_array();

	int nnz_per_core = csr_matrix->row_off[start_row + len] - csr_matrix->row_off[start_row];
	ldm_free(val, nnz_per_core * sizeof(double));
	ldm_free(cols, nnz_per_core * sizeof(int));
	ldm_free(row_off, len * sizeof(int));
	ldm_free(result, len * sizeof(double));
	ldm_free(diag_pos_slave, len * sizeof(int));
	ldm_free(diag_inverse, len * sizeof(double));
	ldm_free(diag, len * sizeof(double));

	ldm_free(r, len * sizeof(double));
	ldm_free(x, len * sizeof(double));
	ldm_free(p, len * sizeof(double));
}

typedef struct
{
	double *p;
	double *z;
	double beta;
	int cells;
} Para_axpy;

typedef struct
{
	double *b;
	double *Ax;
	double *r;
	int cells;
} Para_bsubAx;

typedef struct
{
	double *r;
	double *z;
	double *result;
	int cells;
} Para_gsumProd;

typedef struct
{
	double *x;
	double *p;
	double *r;
	double *Ax;
	double alpha;
	int cells;
} Para_xprA;

typedef struct
{
	double *r;
	int cells;
	double *result;
} Para_gsumMag;

typedef struct
{
	const CsrMatrix *csr_matrix;
	double *vec;
	double *val;
	double *result;
} Para_prespmv;

#define dataBufferSize 2000
__thread_local crts_rply_t DMARply = 0;
__thread_local unsigned int DMARplyCount = 0;

void slave_prespmv(Para_prespmv *para)
{
	Para_prespmv slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_prespmv));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	double *vec = slavePara.vec;
	// 计算从核接收数组数据长度和接收位置
	int len = csr_matrix->rows / 64;
	int rest = csr_matrix->rows % 64;
	int start_row;
	if (CRTS_tid < rest)
	{
		len++;
		start_row = CRTS_tid * len;
	}
	else
	{
		start_row = CRTS_tid * len + rest;
	}

	// int *row_off = (int *)ldm_malloc(sizeof(int) * (len + 1));

	// 接收数组数据
	// CRTS_dma_get(row_off, csr_matrix->row_off + start_row, (len + 1) * sizeof(int));

	int nnz_per_core = row_off[len] - row_off[0];
	// int *cols = (int *)ldm_malloc(sizeof(int) * nnz_per_core);

	// double *result = (double *)ldm_malloc(sizeof(double) * len);
	// double *val = (double *)ldm_malloc(sizeof(double) * nnz_per_core);
	//  double *vec_test = (double *)ldm_malloc(sizeof(double) * len); // smallest
	//  double *vec_static = (double *)ldm_malloc(sizeof(double) * len);

	// CRTS_dma_get(cols, csr_matrix->cols + row_off[0], nnz_per_core * sizeof(int));
	// CRTS_dma_get(result, slavePara.result + start_row, len * sizeof(double));
	// CRTS_dma_get(val, slavePara.val + row_off[0], nnz_per_core * sizeof(double));
	// CRTS_dma_get(vec_static, slavePara.vec + start_row, len * sizeof(double));

	// 计算
	int start_core = row_off[0];

	for (int i = 0; i < len; i++)
	{
		val[diag_pos_slave[i] - start_core] = 0.0;
	}

	for (int i = 0; i < len; i++)
	{
		int start = row_off[i];
		int num = row_off[i + 1] - row_off[i];
		double temp = 0;
		for (int j = 0; j < num; j++)
		{
			int cols_ = cols[start + j - start_core];
			// if ((cols_ >= start_row) && (cols_ <= start_row + len - 1))
			// 	temp += vec_static[cols_ - start_row] * val[start + j - start_core];
			// else
			temp += vec[cols_] * val[start + j - start_core];
			// vec_test[i] = (i % 128) * 0.01;
			// temp += vec_test[i] * val[start + j - start_core];
		}
		result[i] = temp;
	}

	// int start_core = row_off[start_row];

	// for (int i = 0; i < len; i++)
	// {
	// 	int start = row_off[start_row + i];
	// 	int num = row_off[start_row + i + 1] - row_off[start_row + i];
	// 	double temp = 0;
	// 	for (int j = 0; j < num; j++)
	// 	{
	// 		temp += vec[cols[start + j]] * val[start + j - start_core];
	// 	}
	// 	result[i] = temp;
	// }

	// 传回计算结果
	CRTS_dma_put(slavePara.result + start_row, result, len * sizeof(double));
	// ldm_free(result, len * sizeof(double));
	//  ldm_free(vec_static, len * sizeof(double));
	// ldm_free(val, nnz_per_core * sizeof(double));
	//  ldm_free(row_off, (len + 1) * sizeof(int));
	//  ldm_free(cols, nnz_per_core * sizeof(int));
	//  ldm_free(vec_test, len * sizeof(double));
}

void slave_axpy(Para_axpy *para)
{
	// DMARplyCount = 0;
	// CRTS_ssync_array();
	Para_axpy slavePara;
	// 接收结构体数据
	// CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	// DMARplyCount++;
	// CRTS_dma_wait_value(&DMARply, DMARplyCount);

	CRTS_dma_get(&slavePara, para, sizeof(Para_axpy));

	double beta = slavePara.beta;
	int cells = slavePara.cells;
	// if (_PEN == 0)
	// 	printf("cell num: %d\n", cells);
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}

	// double *pp = (double *)ldm_malloc(sizeof(double) * len);
	// double *zz = (double *)ldm_malloc(sizeof(double) * len);
	// CRTS_ssync_array();

	// 接收数组数据

	// CRTS_dma_iget(p, slavePara.p + addr, len * sizeof(double), &DMARply);
	// CRTS_dma_iget(zz, slavePara.z + addr, len * sizeof(double), &DMARply);
	// DMARplyCount += 2;
	// CRTS_dma_wait_value(&DMARply, DMARplyCount);
	// CRTS_ssync_array();

	// CRTS_dma_get(p, slavePara.p + addr, len * sizeof(double));

	// CRTS_dma_get(pp, slavePara.p + addr, len * sizeof(double));
	// CRTS_dma_get(zz, slavePara.z + addr, len * sizeof(double));

	// if (_PEN == 63)
	// 	printf("22222\n");
	// 计算

	for (int i = 0; i < len; i++)
	{
		// pp[i] = zz[i] + beta * pp[i];
		// pp[i] = result[i] + beta * pp[i];
		p[i] = result[i] + beta * p[i];
	}
	// 传回计算结果
	// CRTS_dma_iput(slavePara.p + addr, pp, len * sizeof(double), &DMARply);
	// DMARplyCount++;
	// CRTS_dma_wait_value(&DMARply, DMARplyCount);
	// CRTS_ssync_array();
	CRTS_dma_put(slavePara.p + addr, p, len * sizeof(double));

	// if (_PEN == 63)
	// 	printf("33333\n");
	// ldm_free(pp, len * sizeof(double));
	// ldm_free(zz, len * sizeof(double));
}

void slave_bsubAx(Para_bsubAx *para)
{

	Para_bsubAx slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_bsubAx));
	int cells = slavePara.cells;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}
	// double *b = (double *)ldm_malloc(sizeof(double) * len);
	// double *Ax = (double *)ldm_malloc(sizeof(double) * len);
	// double *r = (double *)ldm_malloc(sizeof(double) * len);

	// 接收数组数据
	CRTS_dma_get(p, slavePara.b + addr, len * sizeof(double));
	// CRTS_dma_get(b, slavePara.b + addr, len * sizeof(double));
	// CRTS_dma_get(Ax, slavePara.Ax + addr, len * sizeof(double));
	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));

	// 计算
	for (int i = 0; i < len; i++)
	{
		// r[i] = b[i] - Ax[i];
		// r[i] = b[i] - result[i];
		r[i] = p[i] - result[i];
	}
	// 传回计算结果
	// CRTS_dma_put(slavePara.r + addr, r, len * sizeof(double));
	// ldm_free(b, len * sizeof(double));
	// ldm_free(Ax, len * sizeof(double));
	// ldm_free(r, len * sizeof(double));
}

void slave_gsumProd(Para_gsumProd *para)
{
	Para_gsumProd slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_gsumProd));
	int cells = slavePara.cells;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}
	// double *z = (double *)ldm_malloc(sizeof(double) * len);
	// double *r = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据
	// CRTS_dma_get(z, slavePara.z + addr, len * sizeof(double));
	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));
	// 计算
	double ret = .0;
	for (int i = 0; i < len; i++)
	{
		// ret += z[i] * r[i];
		ret += result[i] * r[i];
	}

	for (int i = 0; i < len; i++)
	{
		// ret += z[i] * r[i];
		// p[i] = z[i];
		p[i] = result[i];
	}
	// 传回计算结果
	CRTS_dma_put(slavePara.result + CRTS_tid, &ret, 1 * sizeof(double));
	// ldm_free(z, len * sizeof(double));
	// ldm_free(r, len * sizeof(double));
}

typedef struct
{
	double *r;
	double *z;
	double *result;
	int cells;
} Para_gsumProd_2;

void slave_gsumProd_2(Para_gsumProd_2 *para)
{
	Para_gsumProd_2 slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_gsumProd_2));
	int cells = slavePara.cells;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}
	// double *z = (double *)ldm_malloc(sizeof(double) * len);
	// double *r = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据
	// CRTS_dma_get(z, slavePara.z + addr, len * sizeof(double));
	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));
	// 计算
	double ret = .0;
	for (int i = 0; i < len; i++)
	{
		// ret += z[i] * r[i];
		ret += result[i] * r[i];
	}
	// 传回计算结果
	CRTS_dma_put(slavePara.result + CRTS_tid, &ret, 1 * sizeof(double));
	// ldm_free(z, len * sizeof(double));
	// ldm_free(r, len * sizeof(double));
}

typedef struct
{
	double *r;
	double *z;
	double *result;
	int cells;
} Para_gsumProd_3;

void slave_gsumProd_3(Para_gsumProd_3 *para)
{
	Para_gsumProd_3 slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_gsumProd_3));
	int cells = slavePara.cells;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}
	// double *z = (double *)ldm_malloc(sizeof(double) * len);
	// double *r = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据
	// CRTS_dma_get(z, slavePara.z + addr, len * sizeof(double));
	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));
	// 计算
	double ret = .0;
	for (int i = 0; i < len; i++)
	{
		// ret += z[i] * r[i];
		// ret += z[i] * result[i];
		ret += p[i] * result[i];
		// ret += result[i] * r[i];
	}
	// 传回计算结果
	CRTS_dma_put(slavePara.result + CRTS_tid, &ret, 1 * sizeof(double));
	// ldm_free(z, len * sizeof(double));
	// ldm_free(r, len * sizeof(double));
}

void slave_xprA(Para_xprA *para)
{
	Para_xprA slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_xprA));
	int cells = slavePara.cells;
	double alpha = slavePara.alpha;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}
	// double *x = (double *)ldm_malloc(sizeof(double) * len);
	// double *p = (double *)ldm_malloc(sizeof(double) * len);
	// double *r = (double *)ldm_malloc(sizeof(double) * len);
	// double *Ax = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据
	// CRTS_dma_get(x, slavePara.x + addr, len * sizeof(double));
	// CRTS_dma_get(p, slavePara.p + addr, len * sizeof(double));
	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));
	// CRTS_dma_get(Ax, slavePara.Ax + addr, len * sizeof(double));
	// 计算

	for (int i = 0; i < len; i++)
	{
		x[i] = x[i] + alpha * p[i];
		// r[i] = r[i] - alpha * Ax[i];
		r[i] = r[i] - alpha * result[i];
	}
	// 传回计算结果
	// CRTS_dma_put(slavePara.x + addr, x, len * sizeof(double));
	// CRTS_dma_put(slavePara.r + addr, r, len * sizeof(double));
	// ldm_free(x, len * sizeof(double));
	// ldm_free(p, len * sizeof(double));
	// ldm_free(r, len * sizeof(double));
	// ldm_free(Ax, len * sizeof(double));
}

void slave_gsumMag(Para_gsumMag *para)
{
	Para_gsumMag slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_gsumMag));
	int cells = slavePara.cells;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}

	// double *r = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据

	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));
	// 计算
	double ret = .0;
	for (int i = 0; i < len; i++)
	{
		ret += fabs(r[i]);
	}
	// 传回计算结果
	CRTS_dma_put(slavePara.result + CRTS_tid, &ret, 1 * sizeof(double));

	// ldm_free(r, len * sizeof(double));
}

typedef struct
{
	const CsrMatrix *csr_matrix;
	double *vec;
	double *result;
} Para_spmv;

void slave_spmv(Para_spmv *para)
{
	Para_spmv slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_spmv));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	double *vec = slavePara.vec;
	// 计算从核接收数组数据长度和接收位置
	int len = csr_matrix->rows / 64;
	int rest = csr_matrix->rows % 64;
	int start_row;
	if (CRTS_tid < rest)
	{
		len++;
		start_row = CRTS_tid * len;
	}
	else
	{
		start_row = CRTS_tid * len + rest;
	}

	// int *row_off = (int *)ldm_malloc(sizeof(int) * (len + 1));

	// 接收数组数据
	// CRTS_dma_get(row_off, csr_matrix->row_off + start_row, (len + 1) * sizeof(int));

	int nnz_per_core = row_off[len] - row_off[0];
	// int *cols = (int *)ldm_malloc(sizeof(int) * nnz_per_core);

	// double *result = (double *)ldm_malloc(sizeof(double) * len);
	// double *val = (double *)ldm_malloc(sizeof(double) * nnz_per_core);
	// double *vec_static = (double *)ldm_malloc(sizeof(double) * len);
	// double *vec_test = (double *)ldm_malloc(sizeof(double) * len); // smallest

	// CRTS_dma_get(cols, csr_matrix->cols + row_off[0], nnz_per_core * sizeof(int));
	// CRTS_dma_get(result, slavePara.result + start_row, len * sizeof(double));
	// CRTS_dma_get(val, (slavePara.csr_matrix)->data + row_off[0], nnz_per_core * sizeof(double));
	// CRTS_dma_get(vec_static, slavePara.vec + start_row, len * sizeof(double));
	// 计算
	int start_core = row_off[0];

	// if (CRTS_tid == 0)
	// {
	// 	for (int i = 0; i < len; i++)
	// 	{
	// 		printf("diag_pos_slave[%d] = % d\n", i, diag_pos_slave[i]);
	// 	}
	// }
	for (int i = 0; i < len; i++)
	{
		val[diag_pos_slave[i] - start_core] = diag[i];
	}

	for (int i = 0; i < len; i++)
	{
		int start = row_off[i];
		int num = row_off[i + 1] - row_off[i];
		double temp = 0;
		for (int j = 0; j < num; j++)
		{
			int cols_ = cols[start + j - start_core];
			// if ((cols_ >= start_row) && (cols_ <= start_row + len - 1))
			// 	temp += vec_static[cols[start + j - start_core] - start_row] * val[start + j - start_core];
			// else
			temp += vec[cols_] * val[start + j - start_core];
			// vec_test[i] = (i % 128) * 0.01;
			// temp += vec_test[i] * val[start + j - start_core];
		}
		result[i] = temp;
	}

	// int start_core = row_off[start_row];

	// for (int i = 0; i < len; i++)
	// {
	// 	int start = row_off[start_row + i];
	// 	int num = row_off[start_row + i + 1] - row_off[start_row + i];
	// 	double temp = 0;
	// 	for (int j = 0; j < num; j++)
	// 	{
	// 		temp += vec[cols[start + j]] * val[start + j - start_core];
	// 	}
	// 	result[i] = temp;
	// }

	// 传回计算结果
	CRTS_dma_put(slavePara.result + start_row, result, len * sizeof(double));
	// ldm_free(result, len * sizeof(double));
	//  ldm_free(vec_static, len * sizeof(double));
	// ldm_free(val, nnz_per_core * sizeof(double));
	// ldm_free(row_off, (len + 1) * sizeof(int));
	// ldm_free(cols, nnz_per_core * sizeof(int));
	// ldm_free(vec_test, len * sizeof(double));
}

 

typedef struct
{
	const CsrMatrix *csr_matrix;
	const Precondition *pre;
	double *rAPtr;
	double *wAPtr;
} Para_pcg_precondition_csr;

// error edition
void slave_pcg_precondition_csr(Para_pcg_precondition_csr *para)
{
	Para_pcg_precondition_csr slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_pcg_precondition_csr));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	const Precondition *pre = slavePara.pre;
	double *wAPtr = slavePara.wAPtr;
	// 计算从核接收数组数据长度和接收位置
	int len = csr_matrix->rows / 64;
	int rest = csr_matrix->rows % 64;
	int start_row;
	if (CRTS_tid < rest)
	{
		len++;
		start_row = CRTS_tid * len;
	}
	else
	{
		start_row = CRTS_tid * len + rest;
	}

	int *row_off = (int *)ldm_malloc(sizeof(int) * (len + 1));
	// 接收数组数据
	CRTS_dma_get(row_off, csr_matrix->row_off + start_row, (len + 1) * sizeof(int));

	int nnz_per_core = row_off[len] - row_off[0];

	double *gAPtr = (double *)ldm_malloc(len * sizeof(double));
	// double *wAPtr = (double *)ldm_malloc(len * sizeof(double));
	double *preD = (double *)ldm_malloc(len * sizeof(double));
	double *val = (double *)ldm_malloc(nnz_per_core * sizeof(double));
	double *rAPtr = (double *)ldm_malloc(len * sizeof(double));

	CRTS_dma_get(val, slavePara.pre->pre_mat_val + start_row, nnz_per_core * sizeof(double));
	CRTS_dma_get(preD, slavePara.pre->preD + start_row, len * sizeof(double));
	CRTS_dma_get(rAPtr, slavePara.rAPtr + start_row, len * sizeof(double));

	for (int cell = 0; cell < len; cell++)
	{
		wAPtr[cell + start_row] = preD[cell] * rAPtr[cell];
	}

	memset(gAPtr, 0, len * sizeof(double));
	athread_ssync_array();
	int start_core = row_off[0];

	// for (int deg = 1; deg < 2; deg++)
	// {
	// spmv;

	for (int i = 0; i < len; i++)
	{
		int start = row_off[i];
		int num = row_off[i + 1] - row_off[i];
		double temp = 0;
		for (int j = 0; j < num; j++)
		{
			if ((cols[start + j - start_core] >= start_row) && (cols[start + j - start_core] <= start_row + len - 1))
				temp += wAPtr[cols[start + j - start_core] - start_row] * val[start + j - start_core];
			else
				temp += wAPtr[cols[start + j - start_core]] * val[start + j - start_core];
		}
		gAPtr[i] = temp;
	}

	// athread_ssync_array();
	// v_sub_dot_product(csr_matrix.rows, rAPtr, gAPtr, pre.preD, wAPtr);
	for (int cell = 0; cell < len; cell++)
	{
		wAPtr[cell + start_row] = (rAPtr[cell] - gAPtr[cell]) * preD[cell];
	}

	// void v_sub_dot_product(const int nCells, const double *sub, const double *subed, const double *vec, double *result)
	// {
	// 	for (int cell = 0; cell < nCells; cell++)
	// 	{
	// 		result[cell] = (sub[cell] - subed[cell]) * vec[cell];
	// 	}
	// }

	memset(gAPtr, 0, len * sizeof(double));
	athread_ssync_array();
	// }

	// CRTS_dma_put(slavePara.wAPtr + start_row, wAPtr, len * sizeof(double));
	ldm_free(gAPtr, len * sizeof(double));
	ldm_free(preD, len * sizeof(double));
	ldm_free(rAPtr, len * sizeof(double));
	ldm_free(val, nnz_per_core * sizeof(double));
	ldm_free(row_off, (len + 1) * sizeof(int));
	// ldm_free(wAPtr, len * sizeof(double));
}

typedef struct
{
	double *r;
	double *z;
	double *result;
	int cells;
} Para_dot_product;

void slave_dot_product(Para_dot_product *para)
{
	Para_dot_product slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_dot_product));
	int cells = slavePara.cells;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}
	// double *result = (double *)ldm_malloc(sizeof(double) * len);
	// double *z = (double *)ldm_malloc(sizeof(double) * len);
	// double *r = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据
	// CRTS_dma_get(z, slavePara.z + addr, len * sizeof(double));
	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));
	// 计算
	for (int i = 0; i < len; i++)
	{
		// result[i] = z[i] * diag_inverse[i];
		result[i] = r[i] * diag_inverse[i];
		// result[i] = z[i] / diag[i];

		// result[i] = z[i] * r[i];
	}
	// 传回计算结果
	CRTS_dma_put(slavePara.result + addr, result, len * sizeof(double));
	// ldm_free(z, len * sizeof(double));
	// ldm_free(r, len * sizeof(double));
	// ldm_free(result, len * sizeof(double));
}

typedef struct
{
	double *r;
	double *z;
	double *vec;
	double *result;
	int cells;
} Para_sub_dot_product;

void slave_sub_dot_product(Para_sub_dot_product *para)
{
	Para_sub_dot_product slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_sub_dot_product));
	int cells = slavePara.cells;
	// 计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if (CRTS_tid < rest)
	{
		len++;
		addr = CRTS_tid * len;
	}
	else
	{
		addr = CRTS_tid * len + rest;
	}
	// double *wAptr = (double *)ldm_malloc(sizeof(double) * len);
	// double *z = (double *)ldm_malloc(sizeof(double) * len);
	// double *r = (double *)ldm_malloc(sizeof(double) * len);
	// double *vec = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据
	// CRTS_dma_get(z, slavePara.z + addr, len * sizeof(double));
	// CRTS_dma_get(r, slavePara.r + addr, len * sizeof(double));
	// CRTS_dma_get(vec, slavePara.vec + addr, len * sizeof(double));
	// 计算

	for (int i = 0; i < len; i++)
	{
		// result[i] = (r[i] - z[i]) * vec[i];
		// wAptr[i] = (r[i] - z[i]) * diag_inverse[i];
		// wAptr[i] = (r[i] - result[i]) * diag_inverse[i];
		result[i] = (r[i] - result[i]) * diag_inverse[i];
		// result[i] = (r[i] - z[i]) / diag[i];
	}
	// 传回计算结果
	CRTS_dma_put(slavePara.result + addr, result, len * sizeof(double));
	// ldm_free(z, len * sizeof(double));
	// ldm_free(r, len * sizeof(double));
	// ldm_free(vec, len * sizeof(double));
	// ldm_free(wAptr, len * sizeof(double));
}

typedef struct
{
	const CsrMatrix *csr_matrix;
	const Precondition *pre;
} Para_init_precondition_csr;

void slave_init_precondition_csr(Para_init_precondition_csr *para)
{
	Para_init_precondition_csr slavePara;
	// 接收结构体数据

	CRTS_dma_get(&slavePara, para, sizeof(Para_init_precondition_csr));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	const Precondition *pre = slavePara.pre;
	// 计算从核接收数组数据长度和接收位置
	int len = csr_matrix->rows / 64;
	int rest = csr_matrix->rows % 64;
	int start_row;
	if (CRTS_tid < rest)
	{
		len++;
		start_row = CRTS_tid * len;
	}
	else
	{
		start_row = CRTS_tid * len + rest;
	}

	int *row_off = (int *)ldm_malloc(sizeof(int) * (len + 1));

	// 接收数组数据
	CRTS_dma_get(row_off, csr_matrix->row_off + start_row, (len + 1) * sizeof(int));

	int nnz_per_core = row_off[len] - row_off[0];

	int *cols = (int *)ldm_malloc(sizeof(int) * nnz_per_core);

	CRTS_dma_get(cols, csr_matrix->cols + row_off[0], nnz_per_core * sizeof(int));

	// double *gAPtr = (double *)ldm_malloc(len * sizeof(double));
	// double *wAPtr = (double *)ldm_malloc(len * sizeof(double));
	double *pre_mat_val = (double *)ldm_malloc(nnz_per_core * sizeof(double));
	double *preD = (double *)ldm_malloc(len * sizeof(double));
	// double *data = (double *)ldm_malloc(nnz_per_core * sizeof(double));
	// double *val = (double *)ldm_malloc(nnz_per_core * sizeof(double));
	// double *rAPtr = (double *)ldm_malloc(len * sizeof(double));
	// CRTS_dma_get(pre_mat_val, slavePara.pre->pre_mat_val + row_off[0], nnz_per_core * sizeof(double));

	// CRTS_dma_get(preD, slavePara.pre->preD + start_row, len * sizeof(double));

	int start_core = row_off[0];
	for (int i = 0; i < len; i++)
	{
		for (int j = row_off[i]; j < row_off[i + 1]; j++)
		{
			if (cols[j - start_core] == (i + start_row))
			{
				pre_mat_val[j - start_core] = 0.;
				preD[i] = 1.0 / csr_matrix->data[j];
			}
			else
			{
				pre_mat_val[j - start_core] = csr_matrix->data[j];
			}
		}
	}

	CRTS_dma_put(slavePara.pre->pre_mat_val + row_off[0], pre_mat_val, nnz_per_core * sizeof(double));
	CRTS_dma_put(slavePara.pre->preD + start_row, preD, len * sizeof(double));
	ldm_free(preD, len * sizeof(double));
	// ldm_free(rAPtr, len * sizeof(double));
	ldm_free(pre_mat_val, nnz_per_core * sizeof(double));
	ldm_free(cols, nnz_per_core * sizeof(int));
	ldm_free(row_off, (len + 1) * sizeof(int));
}

typedef struct
{
	const CsrMatrix *csr_matrix;
	const LduMatrix *ldu_matrix;
} Para_ldu_to_csr;

typedef struct
{
	const CsrMatrix *csr_matrix;
	const LduMatrix *ldu_matrix;
} Para_parallel_convert;

extern int *diag_pos;
extern int *upper_pos;
extern int *lower_pos;

void slave_parallel_convert(Para_parallel_convert *para)
{
	Para_parallel_convert slavePara;
	// 接收结构体数据

	CRTS_dma_get(&slavePara, para, sizeof(Para_parallel_convert));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	const LduMatrix *ldu_matrix = slavePara.ldu_matrix;
	int cells = ldu_matrix->cells;
	int faces = ldu_matrix->faces;
	// 计算从核接收数组数据长度和接收位置
	int len_cells = cells / 64;
	int rest_cells = cells % 64;
	int addr_cells;
	if (CRTS_tid < rest_cells)
	{
		len_cells++;
		addr_cells = CRTS_tid * len_cells;
	}
	else
	{
		addr_cells = CRTS_tid * len_cells + rest_cells;
	}
	// 计算从核接收数组数据长度和接收位置
	int len_faces = faces / 64;
	int rest_faces = faces % 64;
	int addr_faces;
	if (CRTS_tid < rest_faces)
	{
		len_faces++;
		addr_faces = CRTS_tid * len_faces;
	}
	else
	{
		addr_faces = CRTS_tid * len_faces + rest_faces;
	}

	// lower
	for (int i = 0; i < len_faces; i++)
	{
		csr_matrix->data[lower_pos[i + addr_faces]] = ldu_matrix->lower[i + addr_faces];
	}

	// diag
	for (int i = 0; i < len_cells; i++)
	{
		csr_matrix->data[diag_pos[i + addr_cells]] = ldu_matrix->diag[i + addr_cells];
	}

	// upper
	for (int i = 0; i < len_faces; i++)
	{

		csr_matrix->data[upper_pos[i + addr_faces]] = ldu_matrix->upper[i + addr_faces];
	}
}

typedef struct
{
	const CsrMatrix *csr_matrix;
	double *vec;
	double *result;
	double *source;
	double *singlesum;
} Para_spmv_fusion;

void slave_spmv_fusion(Para_spmv_fusion *para)
{
	Para_spmv_fusion slavePara;
	// 接收结构体数据
	CRTS_dma_get(&slavePara, para, sizeof(Para_spmv_fusion));
	const CsrMatrix *csr_matrix = slavePara.csr_matrix;
	double *vec = slavePara.vec;

	// 计算从核接收数组数据长度和接收位置
	int len = csr_matrix->rows / 64;
	int rest = csr_matrix->rows % 64;
	int start_row;
	if (CRTS_tid < rest)
	{
		len++;
		start_row = CRTS_tid * len;
	}
	else
	{
		start_row = CRTS_tid * len + rest;
	}

	double *source = (double *)ldm_malloc(sizeof(double) * len);
	// 接收数组数据
	CRTS_dma_get(source, slavePara.source + start_row, len * sizeof(double));

	// 接收数组数据

	int nnz_per_core = row_off[len] - row_off[0];

	int start_core = row_off[0];

	// for (int i = 0; i < len; i++)
	// {
	// 	val[diag_pos_slave[i] - start_core] = diag[i];
	// }

	for (int i = 0; i < len; i++)
	{
		int start = row_off[i];
		int num = row_off[i + 1] - row_off[i];
		double temp = 0;
		for (int j = 0; j < num; j++)
		{
			int cols_ = cols[start + j - start_core];

			temp += vec[cols_] * val[start + j - start_core];
		}
		// result[i] = temp;
		result[i] = source[i] - temp; // r =b-Ax
	}

	double ret = .0;
	for (int i = 0; i < len; i++)
	{
		ret += fabs(result[i]);
	}
	// 传回计算结果
	CRTS_dma_put(slavePara.singlesum + CRTS_tid, &ret, 1 * sizeof(double));

	ldm_free(source, len * sizeof(double));
}

 