#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVectorDefault.c"
#else

void THVector_(copy_DEFAULT)(real *x, const real *y, const ptrdiff_t n) {
  ptrdiff_t i = 0;
#if defined(TH_REAL_IS_BYTE)
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )
#endif
#endif
  for(i = 0; i <n-4; i+=4)
  {
    x[i] = y[i];
    x[i+1] = y[i+1];
    x[i+2] = y[i+2];
    x[i+3] = y[i+3];
  }
#if defined(TH_REAL_IS_BYTE)
  i = n - (n%4);
#endif
  for(; i < n; i++)
    x[i] = y[i];
}

void THVector_(copy2_DEFAULT)(real *x, const real *y, const ptrdiff_t n) {
  ptrdiff_t i = 0;

#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) ) private (i)
#endif
  #pragma ivdep
  for(i=0; i <n; i++)
  {
    x[i] = y[i];
  }
}

void THVector_(fill_DEFAULT)(real *x, const real c, const ptrdiff_t n) {
  ptrdiff_t i = 0;

#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) ) private (i)
#endif
  #pragma ivdep
  for(i=0; i <n; i++)
  {
    x[i] = c;
  }
}

void THVector_(cadd_DEFAULT)(real *z, const real *x, const real *y, const real c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    z[i] = x[i] + c * y[i];
    z[i+1] = x[i+1] + c * y[i+1];
    z[i+2] = x[i+2] + c * y[i+2];
    z[i+3] = x[i+3] + c * y[i+3];
  }

  for(; i<n; i++)
    z[i] = x[i] + c * y[i];
}

void THVector_(adds_DEFAULT)(real *y, const real *x, const real c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    y[i] = x[i] + c;
    y[i+1] = x[i+1] + c;
    y[i+2] = x[i+2] + c;
    y[i+3] = x[i+3] + c;
  }

  for(; i<n; i++)
    y[i] = x[i] + c;
}

void THVector_(cmul_DEFAULT)(real *z, const real *x, const real *y, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    z[i] = x[i] * y[i];
    z[i+1] = x[i+1] * y[i+1];
    z[i+2] = x[i+2] * y[i+2];
    z[i+3] = x[i+3] * y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] * y[i];
}

void THVector_(muls_DEFAULT)(real *y, const real *x, const real c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    y[i] = x[i] * c;
    y[i+1] = x[i+1] * c;
    y[i+2] = x[i+2] * c;
    y[i+3] = x[i+3] * c;
  }

  for(; i < n; i++)
    y[i] = x[i] * c;
}

void THVector_(cdiv_DEFAULT)(real *z, const real *x, const real *y, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    z[i] = x[i] / y[i];
    z[i+1] = x[i+1] / y[i+1];
    z[i+2] = x[i+2] / y[i+2];
    z[i+3] = x[i+3] / y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] / y[i];
}

void THVector_(divs_DEFAULT)(real *y, const real *x, const real c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    y[i] = x[i] / c;
    y[i+1] = x[i+1] / c;
    y[i+2] = x[i+2] / c;
    y[i+3] = x[i+3] / c;
  }

  for(; i < n; i++)
    y[i] = x[i] / c;
}

#endif
