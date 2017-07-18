#ifndef RANDOM_H
#define RANDOM_H
typedef unsigned long RNG;

inline RNG RNGInit(long seed)
{
    RNG rng = seed ? (unsigned long)seed : (unsigned long)(long)-1;
    return rng;
}

inline unsigned int RandInt( RNG* rng )
{
    unsigned long temp = *rng;
    temp = (unsigned long)(unsigned)temp*1554115554 + (temp >> 32);
    *rng = temp;
    return (unsigned long)temp;
}

inline ptrdiff_t SearchingIndex(ptrdiff_t index, long *stride, long dim, long* size)
{
  ptrdiff_t i = 0;
  ptrdiff_t rem;
  ptrdiff_t offset = 0;
  for(i = dim-1; i >= 0; --i) {
    rem = index%size[i];
    offset += rem*stride[i];
    index /= size[i];
  }
  return offset;
}


#endif
