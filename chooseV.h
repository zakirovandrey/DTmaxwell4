//#define MINBLOCKS (FTYPESIZE*Nv*20>0xc000)?1:2
#define MINBLOCKS 1
__global__ void __launch_bounds__(Nz,MINBLOCKS) torreD0 (int ix, int y0, int Nt, int t0); 
__global__ void __launch_bounds__(Nz,MINBLOCKS) torreD1 (int ix, int y0, int Nt, int t0); 
__global__ void __launch_bounds__(Nz) torreS0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreS1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreIs0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreIs1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreId0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreId1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreXs0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreXs1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreXd0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreXd1 (int ix, int y0, int Nt, int t0);
/*__global__ void __launch_bounds__(Nz) torreDD0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreDD1(int ix, int y0, int Nt, int t0);
*/
__global__ void __launch_bounds__(Nz) torreTFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreTFSF1(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreITFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreITFSF1(int ix, int y0, int Nt, int t0);

__global__ void __launch_bounds__(Nz) PMLStorreD0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreD1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreS0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreS1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreIs0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreIs1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreId0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreId1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreXs0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreXs1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreXd0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreXd1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreTFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreTFSF1(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreITFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreITFSF1(int ix, int y0, int Nt, int t0);

