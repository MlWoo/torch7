#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.c"
#else

#define BLAS_ALL_LOG     0
#define GEMM_TOP3_LOG    0

#if BLAS_ALL_LOG | GEMM_TOP3_LOG
#include <sys/time.h>
#endif

#ifdef BLAS_F2C
# define ffloat double
#else
# define ffloat float
#endif

TH_EXTERNC void dswap_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void sswap_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void dscal_(int *n, double *a, double *x, int *incx);
TH_EXTERNC void sscal_(int *n, float *a, float *x, int *incx);
TH_EXTERNC void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void scopy_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void daxpy_(int *n, double *a, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void saxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);
TH_EXTERNC double ddot_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
TH_EXTERNC void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
TH_EXTERNC void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda);
TH_EXTERNC void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
TH_EXTERNC void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
TH_EXTERNC void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);



void THBlas_(swap)(long n, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dswap_(&i_n, x, &i_incx, y, &i_incy);
#else
    sswap_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++)
    {
      real z = x[i*incx];
      x[i*incx] = y[i*incy];
      y[i*incy] = z;
    }
  }
}

void THBlas_(scal)(long n, real a, real *x, long incx)
{
  if(n == 1)
    incx = 1;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;

#if defined(TH_REAL_IS_DOUBLE)
    dscal_(&i_n, &a, x, &i_incx);
#else
    sscal_(&i_n, &a, x, &i_incx);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++) {
      if (a == 0) {
        x[i*incx] = 0;
      } else {
        x[i*incx] *= a;
      }
    }
  }
}

void THBlas_(copy)(long n, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dcopy_(&i_n, x, &i_incx, y, &i_incy);
#else
    scopy_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++)
      y[i*incy] = x[i*incx];
  }
}

void THBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    daxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#else

#if BLAS_ALL_LOG
    struct timeval start,end;gettimeofday(&start,NULL);
    saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    gettimeofday(&end,NULL);
    double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
    printf("saxpy time = %.2f ms, n=%d, a=%.4f, incx=%d, incy=%d \n", duration, i_n,a,i_incx,i_incy);
#else
    saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#endif
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++)
      y[i*incy] += a*x[i*incx];
  }
}

real THBlas_(dot)(long n, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    return (real) ddot_(&i_n, x, &i_incx, y, &i_incy);
#else
    return (real) sdot_(&i_n, x, &i_incx, y, &i_incy);
#endif
  }
#endif
  {
    long i;
    real sum = 0;
    for(i = 0; i < n; i++)
    sum += x[i*incx]*y[i*incy];
    return sum;
  }
}

void THBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#else

#if BLAS_ALL_LOG 
    struct timeval start,end;gettimeofday(&start,NULL);
    sgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    gettimeofday(&end,NULL);
    double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
    printf("sgemv time = %.2f ms, trans=%d, m=%d, n=%d, lda=%d, alpha=%.4f, ldb=%d, beta=%.4f, ldc=%d\n", duration,trans,i_m,i_n,i_lda,alpha,i_incx,beta,i_incy);
#else
    sgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#endif
#endif
    return;
  }
#endif
  {
    long i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        real sum = 0;
        real *row_ = a+lda*i;
        for(j = 0; j < m; j++)
          sum += x[j*incx]*row_[j];
	if (beta == 0)
	  y[i*incy] = alpha*sum;
	else
	  y[i*incy] = beta*y[i*incy] + alpha*sum;
      }
    }
    else
    {
      if(beta != 1)
        THBlas_(scal)(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        real *column_ = a+lda*j;
        real z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}

void THBlas_(ger)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dger_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#else
    sger_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#endif
    return;
  }
#endif
  {
    long i, j;
    for(j = 0; j < n; j++)
    {
      real *column_ = a+j*lda;
      real z = alpha*y[j*incy];
      for(i = 0; i < m; i++)
        column_[i] += z*x[i*incx] ;
    }
  }
}

void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

#if defined(TH_REAL_IS_DOUBLE)
    dgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#else
#if BLAS_ALL_LOG
    struct timeval start,end;gettimeofday(&start,NULL);
    sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
    gettimeofday(&end,NULL);
    double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
    printf("sgemm time = %.2f ms, transa=%d, transb=%d, m=%d, n=%d, k=%d, lda=%d, alpha=%.4f, ldb=%d, beta=%.4f, ldc=%d\n", duration,transa,transb,i_m,i_n,i_k,i_lda,alpha,i_ldb,beta,i_ldc);
#else

#if GEMM_TOP3_LOG
    int id = 0;
    if(i_m == 35820 && i_n == 64 && i_k== 500){id = 1;}
    if(i_m == 500 && i_n == 64 && i_k== 35820){id = 2;}
    if(i_m == 500 && i_n == 35820 && i_k== 64){id = 3;}
    if(id != 0){
        struct timeval start, end;
        gettimeofday(&start,NULL);
        printf("size match\n");
        int i=0;
        for(i=0; i < 1; i++)
        {
          printf(" i=%d",i);
          sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
        }
        gettimeofday(&end,NULL);
        double duration = (float)(end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec) ;
        printf("\ngemmid = %d, avg time = %.2f us \n",id, duration);
    }
    else{
    sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
    }
#else
    if(transa=='n' && transb=='n' && i_m==500 && i_n==64 && i_k==2000)
    {
        printf("opt gemm4 code");
        sgemm4_opt(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
    }

    else
    {

       sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);

    }
#endif //GEMM_TOP3_LOG

#endif //BLAS_ALL_LOG

#endif
    return;
  }
#endif
  {
    long i, j, l;
    if(!transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
  }
}

#endif
