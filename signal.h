#ifndef SIGNAL_H
#define SIGNAL_H
__device__ __noinline__ ftype SrcTFSF_Sx(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Sy(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Sz(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Tx(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Ty(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Tz(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Vx(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Vy(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ ftype SrcTFSF_Vz(const int s, const int v, const int a,  const ftype tt);
__device__ __noinline__ bool inSF(const int _s, const int _a, const int _v) ;

__device__ __forceinline__ ftype SrcSurf_Sx(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ __forceinline__ ftype SrcSurf_Sy(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ __forceinline__ ftype SrcSurf_Sz(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ __forceinline__ ftype SrcSurf_Tx(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ __forceinline__ ftype SrcSurf_Ty(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ __forceinline__ ftype SrcSurf_Tz(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ __forceinline__ ftype SrcSurf_Vx(const int s, const int v, const int a,  const ftype tt) {return 0;}
__device__ __forceinline__ ftype SrcSurf_Vy(const int s, const int v, const int a,  const ftype tt) {return 0;}
__device__ __forceinline__ ftype SrcSurf_Vz(const int s, const int v, const int a,  const ftype tt) {return 0;}


extern __constant__ TFSFsrc src;
extern  TFSFsrc shotpoint;

#endif
