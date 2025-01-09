#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <crts.h>
#include <fstream>
#include <iostream>

#include "pcg.h"
using namespace std;

#define NUMOFCORE 64
// 示例
// typedef struct
// {
//     int rows;
//     int *row_off;
//     int *cols;
//     double *data;
//     int data_size;
// } CsrMatrix;

int num_pcg_solve = 0;

typedef struct
{
    const CsrMatrix *csr_matrix;
    double *val;
    int *diag_pos;
    const LduMatrix *ldu_matrix;
    const PCG *pcg;
} Para_A_ldm_malloc;

extern "C" void slave_A_ldm_malloc(Para_A_ldm_malloc *para);

typedef struct
{
    const CsrMatrix *csr_matrix;
    PCG *pcg;
    double *result_x;
} Para_A_ldm_free;

extern "C" void slave_A_ldm_free(Para_A_ldm_free *para);

typedef struct
{
    double *p;
    double *z;
    double beta;
    int cells;
} Para_axpy;
extern "C" void slave_axpy(Para_axpy *para);

typedef struct
{
    double *b;
    double *Ax;
    double *r;
    int cells;
} Para_bsubAx;

extern "C" void slave_bsubAx(Para_bsubAx *para);

typedef struct
{
    double *r;
    double *z;
    double *result;
    int cells;
} Para_gsumProd;

typedef struct
{
    double *r;
    double *z;
    double *result;
    int cells;
} Para_gsumProd_3;

typedef struct
{
    double *r;
    double *z;
    double *result;
    int cells;
} Para_gsumProd_2;

extern "C" void slave_gsumProd(Para_gsumProd *para);
extern "C" void slave_gsumProd_3(Para_gsumProd_3 *para);
extern "C" void slave_gsumProd_2(Para_gsumProd_2 *para);

typedef struct
{
    double *x;
    double *p;
    double *r;
    double *Ax;
    double alpha;
    int cells;
} Para_xprA;

extern "C" void slave_xprA(Para_xprA *para);

typedef struct
{
    double *r;
    int cells;
    double *result;
} Para_gsumMag;

extern "C" void slave_gsumMag(Para_gsumMag *para);
typedef struct
{
    const CsrMatrix *csr_matrix;
    double *vec;
    double *val;
    double *result;
} Para_prespmv;

extern "C" void slave_prespmv(Para_prespmv *para);

typedef struct
{
    const CsrMatrix *csr_matrix;
    double *vec;
    double *result;
} Para_spmv;

extern "C" void slave_spmv(Para_spmv *para);

typedef struct
{
    const CsrMatrix *csr_matrix;
    double *vec;
    double *result;
    double *source;
    double *singlesum;
} Para_spmv_fusion;

extern "C" void slave_spmv_fusion(Para_spmv_fusion *para);

typedef struct
{
    const CsrMatrix *csr_matrix;
    const Precondition *pre;
    double *rAPtr;
    double *wAPtr;
} Para_pcg_precondition_csr;

extern "C" void slave_pcg_precondition_csr(Para_pcg_precondition_csr *para);

typedef struct
{
    double *r;
    double *z;
    double *result;
    int cells;
} Para_dot_product;

extern "C" void slave_dot_product(Para_dot_product *para);

typedef struct
{
    double *r;
    double *z;
    double *vec;
    double *result;
    int cells;
} Para_sub_dot_product;

extern "C" void slave_sub_dot_product(Para_sub_dot_product *para);

typedef struct
{
    const CsrMatrix *csr_matrix;
    const Precondition *pre;
} Para_init_precondition_csr;

extern "C" void slave_init_precondition_csr(Para_init_precondition_csr *para);

typedef struct
{
    const CsrMatrix *csr_matrix;
    const LduMatrix *ldu_matrix;
} Para_parallel_convert;

extern "C" void slave_parallel_convert(Para_parallel_convert *para);

typedef struct
{
    const CsrMatrix *csr_matrix;
    const LduMatrix *ldu_matrix;
} Para_ldu_to_csr;

extern "C" void slave_ldu_to_csr(Para_ldu_to_csr *para);

// prepare for  col,row_off
void processing(const LduMatrix &ldu_matrix, CsrMatrix &csr_matrix);

int *diag_pos;
int *upper_pos;
int *lower_pos;

int *row_off_host;
int *cols_host;

// return  迭代步数和残差
PCGReturn pcg_solve(const LduMatrix &ldu_matrix, double *source, double *x, int maxIter, double tolerance, double normfactor)
{

    static int isInit = 0;
    if (isInit == 0)
    {
        // 从核初始化
        CRTS_init();
        isInit = 1;
    }
    int iter = 0;
    int cells = ldu_matrix.cells;
    int faces = ldu_matrix.faces;

    PCG pcg;
    pcg.r = (double *)malloc(cells * sizeof(double));
    pcg.z = (double *)malloc(cells * sizeof(double));
    pcg.p = (double *)malloc(cells * sizeof(double));
    pcg.Ax = (double *)malloc(cells * sizeof(double));
    pcg.x = x;
    pcg.source = source;

    Precondition pre;
    CsrMatrix csr_matrix;
    process(ldu_matrix, csr_matrix);

    // 并行矩阵格式转换ldutocsr
    Para_parallel_convert para_parallel_convert;
    para_parallel_convert.csr_matrix = &csr_matrix;
    para_parallel_convert.ldu_matrix = &ldu_matrix;

    athread_spawn(slave_parallel_convert, &para_parallel_convert);
    // 等待从核线程组终止
    athread_join();

    // allocate memory for val per core
    Para_A_ldm_malloc para_A_ldm_malloc;
    para_A_ldm_malloc.csr_matrix = &csr_matrix;
    // para_A_ldm_malloc.val = pre.pre_mat_val;
    para_A_ldm_malloc.diag_pos = diag_pos;
    para_A_ldm_malloc.ldu_matrix = &ldu_matrix;
    para_A_ldm_malloc.pcg = &pcg;

    // 启动从核
    athread_spawn(slave_A_ldm_malloc, &para_A_ldm_malloc);
    // 等待从核线程组终止
    athread_join();

    num_pcg_solve++;

    // AX = A * X
    Para_spmv para_spmv;
    para_spmv.csr_matrix = &csr_matrix;
    para_spmv.vec = x;
    para_spmv.result = pcg.Ax;
    // 启动从核
    athread_spawn(slave_spmv, &para_spmv);
    // 等待从核线程组终止
    athread_join();

    // r = b - A * x
    Para_bsubAx para_bsubAx;
    para_bsubAx.b = source;
    para_bsubAx.Ax = pcg.Ax;
    para_bsubAx.r = pcg.r;
    para_bsubAx.cells = cells;
    //  启动从核
    athread_spawn(slave_bsubAx, &para_bsubAx); // 完成kf
    // 等待从核线程组终止
    athread_join();

    double *localsumProd = (double *)malloc(NUMOFCORE * sizeof(double));
    double *localsummag = (double *)malloc(NUMOFCORE * sizeof(double));

    Para_gsumMag para_gsumMag;
    para_gsumMag.r = pcg.r;
    para_gsumMag.cells = cells;
    para_gsumMag.result = localsummag;

    athread_spawn(slave_gsumMag, &para_gsumMag); // 完成kf
    // 等待从核线程组终止
    athread_join();

    pcg.residual = 0.;
    for (int i = 0; i < NUMOFCORE; i++)
    {
        pcg.residual += localsummag[i];
    }

    double init_residual = pcg.residual;

    if (fabs(pcg.residual / normfactor) > tolerance)
    {
        do
        {
            if (iter == 0)
            {
                // z = M(-1) * r
                pcg_precondition_csr(ldu_matrix, csr_matrix, pre, pcg.r, pcg.z);

                Para_gsumProd para_gsumProd;
                para_gsumProd.z = pcg.z;
                para_gsumProd.r = pcg.r;
                para_gsumProd.result = localsumProd;
                para_gsumProd.cells = cells;

                athread_spawn(slave_gsumProd, &para_gsumProd); // 完成kf
                // 等待从核线程组终止
                athread_join();
                pcg.sumprod = 0;
                for (int i = 0; i < NUMOFCORE; i++)
                {
                    pcg.sumprod += localsumProd[i];
                }
                // iter ==0 ; p = z
                memcpy(pcg.p, pcg.z, cells * sizeof(double));
            }
            else
            {
                pcg.sumprod_old = pcg.sumprod;
                // z = M(-1) * r
                pcg_precondition_csr(ldu_matrix, csr_matrix, pre, pcg.r, pcg.z);

                Para_gsumProd_2 para_gsumProd_2; // 调用3次
                para_gsumProd_2.z = pcg.z;
                para_gsumProd_2.r = pcg.r;
                para_gsumProd_2.result = localsumProd;
                para_gsumProd_2.cells = cells;
                athread_spawn(slave_gsumProd_2, &para_gsumProd_2); // 完成kf
                // 等待从核线程组终止
                athread_join();
                pcg.sumprod = 0;
                for (int i = 0; i < NUMOFCORE; i++)
                {
                    pcg.sumprod += localsumProd[i];
                }

                // beta = tol_1 / tol_0
                // p = z + beta * p
                pcg.beta = pcg.sumprod / pcg.sumprod_old;

                Para_axpy para_axpy;
                para_axpy.p = pcg.p;
                para_axpy.z = pcg.z;
                para_axpy.beta = pcg.beta;
                para_axpy.cells = cells;
                // 启动从核
                // 仅出现一次
                athread_spawn(slave_axpy, &para_axpy); // 仅调用一次
                // 等待从核线程组终止  //完成kf
                athread_join();
            }

            // Ax = A * p
            Para_spmv para_spmv;
            para_spmv.csr_matrix = &csr_matrix;
            para_spmv.vec = pcg.p;
            para_spmv.result = pcg.Ax;
            // 启动从核
            athread_spawn(slave_spmv, &para_spmv);
            // 等待从核线程组终止
            athread_join();

            // alpha = tol_0 / tol_1 = (swap(r) * z) / ( swap(p) * A * p)
            // 求和 Ax*p
            Para_gsumProd_3 para_gsumProd_3; // 调用3次
            para_gsumProd_3.z = pcg.p;
            para_gsumProd_3.r = pcg.Ax;
            para_gsumProd_3.result = localsumProd;
            para_gsumProd_3.cells = cells;

            athread_spawn(slave_gsumProd_3, &para_gsumProd_3); // 完成kf
            // 等待从核线程组终止
            athread_join();
            double sum = 0.;
            for (int i = 0; i < NUMOFCORE; i++)
            {
                sum += localsumProd[i];
            }

            pcg.alpha = pcg.sumprod / sum;

            // x = x + alpha * p
            // r = r - alpha * Ax
            Para_xprA para_xprA;
            para_xprA.x = pcg.x;
            para_xprA.p = pcg.p;
            para_xprA.r = pcg.r;
            para_xprA.Ax = pcg.Ax;
            para_xprA.alpha = pcg.alpha;
            para_xprA.cells = cells;
            // 启动从核
            athread_spawn(slave_xprA, &para_xprA); // 仅调用一次
            // 等待从核线程组终止
            athread_join();

            // tol_1 = swap(z) *r
            para_gsumMag.r = pcg.r;
            para_gsumMag.cells = cells;
            para_gsumMag.result = localsummag;
            athread_spawn(slave_gsumMag, &para_gsumMag); // 完成kf
            // 等待从核线程组终止
            athread_join();
            pcg.residual = 0;
            for (int i = 0; i < NUMOFCORE; i++)
            {
                pcg.residual += localsummag[i];
            }

        } while (++iter < maxIter && (pcg.residual / normfactor) >= tolerance);
    }
    // 释放LDM
    Para_A_ldm_free para_A_ldm_free;
    para_A_ldm_free.csr_matrix = &csr_matrix;
    para_A_ldm_free.result_x = pcg.x;

    // 启动从核
    athread_spawn(slave_A_ldm_free, &para_A_ldm_free);
    // 等待从核线程组终止
    athread_join();

    INFO("PCG: init residual = %e, final residual = %e, iterations: %d\n", init_residual, pcg.residual, iter);

    free(localsumProd);
    free(localsummag);
    free_pcg(pcg);
    PCGReturn pcg_return;
    pcg_return.residual = pcg.residual;
    pcg_return.iter = iter;
    return pcg_return;
}

void parallel_convert(const LduMatrix &ldu_matrix, CsrMatrix &csr_matrix)
{

    // lower
    for (int i = 0; i < ldu_matrix.faces; i++)
    {
        csr_matrix.data[lower_pos[i]] = ldu_matrix.lower[i];
    }

    // diag
    for (int i = 0; i < ldu_matrix.cells; i++)
    {
        csr_matrix.data[diag_pos[i]] = ldu_matrix.diag[i];
    }

    // upper
    for (int i = 0; i < ldu_matrix.faces; i++)
    {

        csr_matrix.data[upper_pos[i]] = ldu_matrix.upper[i];
    }
}

void process(const LduMatrix &ldu_matrix, CsrMatrix &csr_matrix)
{

    csr_matrix.rows = ldu_matrix.cells;
    csr_matrix.data_size = 2 * ldu_matrix.faces + ldu_matrix.cells;

    csr_matrix.data = (double *)malloc(csr_matrix.data_size * sizeof(double));

    csr_matrix.row_off = (int *)malloc((csr_matrix.rows + 1) * sizeof(int));
    csr_matrix.cols = (int *)malloc(csr_matrix.data_size * sizeof(int));

    if (num_pcg_solve == 0 || num_pcg_solve == 200 || num_pcg_solve == 400)
    {
        int row, col, offset;
        int *tmp = (int *)malloc((csr_matrix.rows + 1) * sizeof(int));

        csr_matrix.row_off[0] = 0;

        for (int i = 1; i < csr_matrix.rows + 1; i++)
            csr_matrix.row_off[i] = 1;

        for (int i = 0; i < ldu_matrix.faces; i++)
        {
            row = ldu_matrix.uPtr[i];
            col = ldu_matrix.lPtr[i];
            csr_matrix.row_off[row + 1]++;
            csr_matrix.row_off[col + 1]++;
        }

        for (int i = 0; i < ldu_matrix.cells; i++)
        {
            csr_matrix.row_off[i + 1] += csr_matrix.row_off[i];
        }

        memcpy(&tmp[0], &csr_matrix.row_off[0], (ldu_matrix.cells + 1) * sizeof(int));

        diag_pos = (int *)malloc(ldu_matrix.cells * (sizeof(int)));
        upper_pos = (int *)malloc(ldu_matrix.faces * (sizeof(int)));
        lower_pos = (int *)malloc(ldu_matrix.faces * (sizeof(int)));
        // lower
        for (int i = 0; i < ldu_matrix.faces; i++)
        {
            row = ldu_matrix.uPtr[i];
            col = ldu_matrix.lPtr[i];
            offset = tmp[row]++;
            csr_matrix.cols[offset] = col;
            lower_pos[i] = offset;
        }

        // diag
        for (int i = 0; i < ldu_matrix.cells; i++)
        {
            offset = tmp[i]++;
            csr_matrix.cols[offset] = i;
            diag_pos[i] = offset;
        }

        // upper
        for (int i = 0; i < ldu_matrix.faces; i++)
        {
            row = ldu_matrix.lPtr[i];
            col = ldu_matrix.uPtr[i];
            offset = tmp[row]++;
            csr_matrix.cols[offset] = col;
            upper_pos[i] = offset;
        }

        row_off_host = (int *)malloc((ldu_matrix.cells + 1) * (sizeof(int)));
        cols_host = (int *)malloc(csr_matrix.data_size * (sizeof(int)));

        memcpy(row_off_host, csr_matrix.row_off, (ldu_matrix.cells + 1) * sizeof(int));
        memcpy(cols_host, csr_matrix.cols, (csr_matrix.data_size) * sizeof(int));
    }
    csr_matrix.row_off = row_off_host;
    csr_matrix.cols = cols_host;
}

void csr_spmv(const CsrMatrix &csr_matrix, double *vec, double *result)
{
    for (int i = 0; i < csr_matrix.rows; i++)
    {
        int start = csr_matrix.row_off[i];
        int num = csr_matrix.row_off[i + 1] - csr_matrix.row_off[i];
        double temp = 0;
        for (int j = 0; j < num; j++)
        {
            temp += vec[csr_matrix.cols[start + j]] * csr_matrix.data[start + j];
        }
        result[i] = temp;
    }
}

void csr_precondition_spmv(const CsrMatrix &csr_matrix, double *vec, double *val, double *result)
{
    for (int i = 0; i < csr_matrix.rows; i++)
    {
        int start = csr_matrix.row_off[i];
        int num = csr_matrix.row_off[i + 1] - csr_matrix.row_off[i];
        double temp = 0;
        for (int j = 0; j < num; j++)
        {
            temp += vec[csr_matrix.cols[start + j]] * val[start + j];
        }
        result[i] = temp;
    }
}

void v_dot_product(const int nCells, const double *vec1, const double *vec2, double *result)
{
    for (int cell = 0; cell < nCells; cell++)
    {
        result[cell] = vec1[cell] * vec2[cell];
    }
}

void v_sub_dot_product(const int nCells, const double *sub, const double *subed, const double *vec, double *result)
{
    for (int cell = 0; cell < nCells; cell++)
    {
        result[cell] = (sub[cell] - subed[cell]) * vec[cell];
    }
}

// jacobi的预条件
//  struct Precondition{
//      double *pre_mat_val;
//      double *preD;
//  };
void pcg_init_precondition_csr(const CsrMatrix &csr_matrix, Precondition &pre)
{
    for (int i = 0; i < csr_matrix.rows; i++)
    {
        for (int j = csr_matrix.row_off[i]; j < csr_matrix.row_off[i + 1]; j++)
        {
            if (csr_matrix.cols[j] == i)
            {
                pre.pre_mat_val[j] = 0.;
                pre.preD[i] = 1.0 / csr_matrix.data[j];
            }
            else
            {
                pre.pre_mat_val[j] = csr_matrix.data[j];
            }
        }
    }
}

void pcg_precondition_csr(const LduMatrix &ldu_matrix, const CsrMatrix &csr_matrix, const Precondition &pre, double *rAPtr, double *wAPtr)
{
    double *gAPtr = (double *)malloc(csr_matrix.rows * sizeof(double));
    // element-wise multiply ; result is wAptr
    // v_dot_product(csr_matrix.rows, pre.preD, rAPtr, wAPtr);

    Para_dot_product para_dot_product;
    // para_dot_product.r = pre.preD;
    para_dot_product.r = pre.preD;
    para_dot_product.z = rAPtr;
    para_dot_product.result = wAPtr;
    para_dot_product.cells = csr_matrix.rows;

    athread_spawn(slave_dot_product, &para_dot_product); // 完成kf
    // 等待从核线程组终止
    athread_join();

    // memset(gAPtr, 0, csr_matrix.rows * sizeof(double));
    // for (int deg = 1; deg < 2; deg++)
    // {
    // csr_precondition_spmv(csr_matrix, wAPtr, pre.pre_mat_val, gAPtr);
    Para_prespmv para_prespmv;
    para_prespmv.csr_matrix = &csr_matrix;
    para_prespmv.vec = wAPtr;
    para_prespmv.val = pre.pre_mat_val;
    para_prespmv.result = gAPtr;
    // 启动从核
    athread_spawn(slave_prespmv, &para_prespmv);
    // 等待从核线程组终止
    athread_join();
    // v_sub_dot_product(csr_matrix.rows, rAPtr, gAPtr, pre.preD, wAPtr);
    Para_sub_dot_product para_sub_dot_product;
    para_sub_dot_product.r = rAPtr;
    para_sub_dot_product.z = gAPtr;
    para_sub_dot_product.vec = pre.preD;
    para_sub_dot_product.result = wAPtr;
    para_sub_dot_product.cells = csr_matrix.rows;

    athread_spawn(slave_sub_dot_product, &para_sub_dot_product);
    // 等待从核线程组终止
    athread_join();
    // memset(gAPtr, 0, csr_matrix.rows * sizeof(double));
    // }
    free(gAPtr);
}

double pcg_gsumMag(double *r, int size)
{
    double ret = .0;
    for (int i = 0; i < size; i++)
    {
        ret += fabs(r[i]);
    }
    return ret;
}

double pcg_gsumProd(double *z, double *r, int size)
{
    double ret = .0;
    for (int i = 0; i < size; i++)
    {
        ret += z[i] * r[i];
    }
    return ret;
}

void free_pcg(PCG &pcg)
{
    free(pcg.r);
    free(pcg.z);
    free(pcg.p);
    free(pcg.Ax);
}

void free_csr_matrix(CsrMatrix &csr_matrix)
{
    free(csr_matrix.cols);
    // free(csr_matrix.data);
    free(csr_matrix.row_off);
}

void free_precondition(Precondition &pre)
{
    free(pre.preD);
    free(pre.pre_mat_val);
}
