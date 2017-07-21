#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif
#include <x86intrin.h>
void THTensor_(copy2)(THTensor *tensor, THTensor *src)
{
  if (THTensor_(isContiguous)(tensor) && THTensor_(isContiguous)(src) && THTensor_(nElement)(tensor) == THTensor_(nElement)(src)) {
    real *sp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    ptrdiff_t sz = THTensor_(nElement)(tensor);
#ifndef TH_REAL_IS_HALF
    THVector_(copy)(rp, sp, sz); 
#else
#ifdef _OPENMP
    ptrdiff_t i;
    
    #pragma omp parallel for if (sz > TH_OMP_OVERHEAD_THRESHOLD_COPY) private (i)
    for(i=0; i<sz; i++){
      rp[i] = sp[i];
    }  
#else
    memcpy(rp, sp, sz * sizeof(real));
#endif
#endif
  } else {
#ifdef _OPENMP 
    TH_TENSOR_APPLY2_ADVANCED_INDEX(real, tensor, real, src, *tensor_data = *src_data;)
#else
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
#endif
  }
}

void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  ptrdiff_t tensorSize = THTensor_(nElement)(tensor);                     
  ptrdiff_t srcSize = THTensor_(nElement)(src);                     
  int tensorContig = THTensor_(isContiguous)(tensor)? 1:0;                 
  int srcContig = THTensor_(isContiguous)(src)? 1:0;                 
  if (tensorSize == srcSize){
    if ( tensorContig && srcContig) {
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(tensor);
#ifndef TH_REAL_IS_HALF
      THVector_(copy)(rp, sp, srcSize); 
#else  
      
#ifdef _OPENMP
      if (srcSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) {
        ptrdiff_t i;   
        #pragma omp parallel for private (i)
        for(i=0; i<srcSize; i++){
          rp[i] = sp[i];
        }
      } else {
        memcpy(rp, sp, srcSize * sizeof(real));  
      }  
#else
      memcpy(rp, sp, srcSize * sizeof(real));
#endif

#endif
    } else {
#ifdef _OPENMP 
      TH_TENSOR_APPLY2_ADVANCED_INDEX2(srcSize, tensorContig, srcContig, real, tensor, real, src, *tensor_data = *src_data;)
#else
      TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
#endif
    }
  } else {
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
  }
}


#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_TO_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = TH_float2half((float)*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)TH_half2float(*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_TO_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = *src_data;) \
}

#ifndef TH_REAL_IS_HALF
IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)
IMPLEMENT_THTensor_COPY_FROM_HALF(Half, THHalf)
#else
/* only allow pass-through for Half */
IMPLEMENT_THTensor_COPY_TO_FROM_HALF(Half, THHalf)
IMPLEMENT_THTensor_COPY_TO_HALF(Byte, unsigned char)
IMPLEMENT_THTensor_COPY_TO_HALF(Char, char)
IMPLEMENT_THTensor_COPY_TO_HALF(Short, short)
IMPLEMENT_THTensor_COPY_TO_HALF(Int, int)
IMPLEMENT_THTensor_COPY_TO_HALF(Long, long)
IMPLEMENT_THTensor_COPY_TO_HALF(Float, float)
IMPLEMENT_THTensor_COPY_TO_HALF(Double, double)

#endif /* REAL_IS_HALF */

#endif
