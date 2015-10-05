#ifndef ARR3D_PARS_HPP
#define ARR3D_PARS_HPP

//------------------------------------
struct Arr3D_pars {
  float* Arr3Dbuf;
  size_t BufSize;
  float fMin, fMax;
  char* fName;
  int Nx, Ny, Nz;//размер массива
  bool inGPUmem, inCPUmem;
  Arr3D_pars(): inGPUmem(false), inCPUmem(false), Arr3Dbuf(0), BufSize(0) {}
  //int read_from_file(char* fName);
  void clear() { delete Arr3Dbuf; Arr3Dbuf = 0; BufSize = 0; inGPUmem = inCPUmem = false; fMin=0.0; fMax=1.0; }
  inline unsigned int get_ind(int ix, int iy, int iz) { return ix+(iy+iz*Ny)*Nx; }
  inline float* get_ptr(int ix, int iy, int iz) { return Arr3Dbuf+get_ind(ix,iy,iz); }
  void reset_min_max() {
    if(Arr3Dbuf==NULL || Nx*Ny*Nz==0) return;
    fMin = fMax = Arr3Dbuf[0];
    for(size_t i=0; i<Nx*Ny*Nz; i++) {
      float v=Arr3Dbuf[i];
      if(v<fMin) fMin = v;
      if(v>fMax) fMax = v;
    }
    //printf("reset_min_max: fMin=%g, fMax=%g\n", fMin, fMax);
  }
  void reset(int _Nx, int _Ny, int _Nz) {
    Nx = _Nx; Ny = _Ny; Nz = _Nz;
    fName = 0;
    fMin = -1.0; fMax = 1.0;
  }
};

#endif//ARR3D_PARS_HPP
