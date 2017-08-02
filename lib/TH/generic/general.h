#ifndef GENERAL_H
#define GENERAL_H
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
