#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THGemmOpt.c"
#else


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


void THBlas_(result_check)(int len, real * buf1, real * buf2)
{
  printf("result check start \n");
}


void THBlas_(gemm_size1)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc)
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
    sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
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
