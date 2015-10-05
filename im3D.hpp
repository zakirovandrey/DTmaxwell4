#ifndef IM3D_HPP
#define IM3D_HPP
#ifndef CUDA_BASIC_PARS
#define CUDA_BASIC_PARS
const int NW=32;//, SW=32, NT=SW*NW;//число тредов cuda в warp-е и block-е
#endif//CUDA_BASIC_PARS
#include "Arr3Dpars.hpp"

//const int Npx=1;//размер ячейки в пикселях
//NT не больше ограничения на число потоков в блоке (1024), также NT*Nregs/Thread для всех cuda-методов должно быть не больше числа регистров в SM (32K/64K) (-Xptxas="-v" как раз и выводит Nregs/Thread, пока что не более 30) 
//unsigned const int M0 = 0x55555555;
//unsigned const int M1 = 0xAAAAAAAA;

struct im3D_pars4save {
  float tstep, density;
  bool draw_mesh_flag, draw_box_flag;
  int ix0,iy0,iz0;
  float viewRotation[3];
  float viewTranslation[3];
  float BoxFactor[3];
  float MeshBox[3];
  float Dzoom[3];
  float bkgr_col[3], mesh_col[3], box_col[3];
  float step[3];
  float Dmesh;
  char drop_dir[64];
};

struct im3D_pars: public im3D_pars4save {
  int Nx, Ny, Nz;//размер массива
  int bNx, bNy;//размер картинки
  int pal_sh, xy_sh, xz_sh, zy_sh, zx_sh, yz_sh, xyz_sh;
  float x_zoom, y_zoom, z_zoom;
  float fMin, fMax;
  char* fName;
  bool checkNcopy(Arr3D_pars& arr) {
    if(Nx != arr.Nx || Ny != arr.Ny || Nz != arr.Nz) return false;
    fName = arr.fName;
    fMin = arr.fMin;
    fMax = arr.fMax;
    return true;
  }
  void reset() {
    for(int i=0; i<3; i++) {
      viewRotation[i] = 0.0;
      viewTranslation[i] = 0.0;
      BoxFactor[i] = 1.0;
      MeshBox[i] = 100.0;
      Dzoom[i] = 1.0;
      step[i] = 1.0;
      bkgr_col[i] = 0.1;
      mesh_col[i] = 0.8;
      box_col[i] = 1.0;
    }
    mesh_col[2] = 0.2;
    viewTranslation[2] =-4.0;
    Dmesh = 5.0;
    tstep = 2.0; density = 0.5;
    draw_mesh_flag=true; draw_box_flag=true;
    drop_dir[0] = '.'; drop_dir[1] = 0;
  }
  void reset(Arr3D_pars& arr) {
    fName = arr.fName;
    Nx = arr.Nx; Ny = arr.Ny; Nz = arr.Nz;
    fMin = arr.fMin; fMax = arr.fMax;
    for(int i=0; i<3; i++) if(Dzoom[i] <= 0.0) Dzoom[i] = 1.0;
    x_zoom=1.0/Dzoom[0]; y_zoom=1.0/Dzoom[1]; z_zoom=1.0/Dzoom[2];
    int NxZ=Nx/x_zoom, NyZ=Ny/y_zoom, NzZ=Nz/z_zoom;
    xy_sh = xz_sh = zy_sh = zx_sh = yz_sh = xyz_sh = 0;
    if(NzZ<=NxZ && NzZ<=NyZ) zx_sh = yz_sh = -1;
    else if(NyZ<=NxZ && NyZ<NzZ) zx_sh = zy_sh = -1;
    else if(NxZ<NyZ && NxZ<NzZ) xz_sh = yz_sh = -1;
    int w=NxZ+((yz_sh<0)?NzZ:NyZ), x_gap=w%NW?(NW-w%NW):0; bNx = w+x_gap;
    int h=NyZ+((zx_sh<0)?NzZ:NxZ), y_gap=h%NW?(NW-h%NW):0; bNy = h+y_gap;
    pal_sh = bNx*bNy;
    xy_sh = bNx*(bNy-NyZ);
    if(zx_sh>=0) zx_sh = bNx-NzZ; else xyz_sh = bNx-NzZ;
    if(yz_sh>=0) yz_sh = bNx-NyZ; else xyz_sh = bNx-NyZ;
    if(xz_sh>=0) xz_sh = 0; else xyz_sh = 0;
    if(zy_sh>=0) zy_sh = xy_sh+bNx-NzZ; else xyz_sh = xy_sh+bNx-NzZ;
    bNy+=20;
    ix0 = Nx/2; iy0 = Ny/2; iz0 = Nz/2;
  }
  void reset0(int x, int y) {
    switch((int(x*x_zoom)<Nx?0:1)|(y>=xy_sh/bNx?0:2)) {
      case 0: ix0 = x*x_zoom; iy0 = (y-xy_sh/bNx)*y_zoom; break;
      case 1: if(zy_sh>=0 && x>=zy_sh%bNx) { iy0 = (y-xy_sh/bNx)*y_zoom; iz0 = (x-zy_sh%bNx)*z_zoom; } break;
      case 2: if(xz_sh>=0 && int(y*z_zoom)<Nz) { ix0 = x*x_zoom; iz0 = y*z_zoom; } break;
      case 3: if(yz_sh>=0 && x>=yz_sh%bNx && int(y*z_zoom)<Nz) { iz0 = y*z_zoom; iy0 = (x-yz_sh%bNx)*y_zoom; }
         else if(zx_sh>=0 && x>=zx_sh%bNx && int(y*x_zoom)<Nx) { ix0 = y*x_zoom; iz0 = (x-zx_sh%bNx)*z_zoom; }
      break;
    }
  }
  void print_help();
  bool key_func(unsigned char key, int x, int y);
  void recalc_func();
  void mouse_func(int button, int state, int x, int y);
  void motion_func(int x, int y);
  char* reset_title();
  void clear4exit();
  void init3D(Arr3D_pars& arr);
  void initCuda(Arr3D_pars& arr);
  void recalc_im3D();
  void recalc3D_im3D();
  void save_bmp4backgrownd();
};
#endif//IM3D_HPP
