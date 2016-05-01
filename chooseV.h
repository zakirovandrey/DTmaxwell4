//#define MINBLOCKS (FTYPESIZE*Nv*20>0xc000)?1:2
#define MINBLOCKS 1
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz,MINBLOCKS) torreD0 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0); 
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz,MINBLOCKS) torreD1 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0); 
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreS0 (int ix, int y0, int izBeg, int izEnd,  int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreS1 (int ix, int y0, int izBeg, int izEnd,  int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreIs0 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreIs1 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreId0 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreId1 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreXs0 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreXs1 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreXd0 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreXd1 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
/*__global__ void __launch_bounds__(Nz) torreDD0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreDD1(int ix, int y0, int Nt, int t0);
*/
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreTFSF0     (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreTFSF1     (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreITFSF0    (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) torreITFSF1    (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);

template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreD0    (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreD1    (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreS0    (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreS1    (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreIs0   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreIs1   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreId0   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreId1   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreXs0   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreXs1   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreXd0   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreXd1   (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreTFSF0 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreTFSF1 (int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreITFSF0(int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
template<int zform> __global__ void __launch_bounds__((Nz>NzMax)?NzMax:Nz) PMLStorreITFSF1(int ix, int y0, int izBeg, int izEnd, int Nt, int t0);
