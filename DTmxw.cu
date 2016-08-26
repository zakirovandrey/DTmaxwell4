#include "cuda_math.h"
#include "cuda_math_double.h"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef MPI_ON
#include <mpi.h>
#endif
#include "omp.h"
#include "params.h"
#include "init.h" 
#include "signal.hpp"
#include "diamond.cu"

__global__ void calc_limits(float* buf, float* fLims) {
  float2 fLim;
  float* pf=buf+blockIdx.x*Nz+threadIdx.x;
  fLim.x = fLim.y = *pf;

  for(int i=0; i<Nx; i++,pf+=Ny*Nz) {
    float v=*pf;
    if(v<fLim.x) fLim.x = v;
    if(v>fLim.y) fLim.y = v;
  }
  __shared__ float2 fLim_sh[Nz];
  fLim_sh[threadIdx.x] = fLim;
  if(threadIdx.x>warpSize) return;
  for(int i=threadIdx.x; i<Nz; i+=warpSize) {
    float2 v=fLim_sh[i];
    if(v.x<fLim.x) fLim.x = v.x;
    if(v.y>fLim.y) fLim.y = v.y;
  }
  fLim_sh[threadIdx.x] = fLim;
  if(threadIdx.x>0) return;
  for(int i=0; i<warpSize; i++) {
    float2 v=fLim_sh[i];
    if(v.x<fLim.x) fLim.x = v.x;
    if(v.y>fLim.y) fLim.y = v.y;
  }
  fLims[2*blockIdx.x  ] = fLim.x;
  fLims[2*blockIdx.x+1] = fLim.y;
}

#include "im2D.h"
#include "im3D.hpp"
int type_diag_flag=0;

im3D_pars im3DHost;

char* FuncStr[] = {"Hx","Hy","Hz", "Ex", "Ey", "Ez", "Eampl", "Mat"};
#ifdef USE_TEX_2D
texture<char, cudaTextureType2D> index_tex;
#else
texture<char, cudaTextureType3D> index_tex;
#endif
cudaArray* index_texArray=0;

__device__ float pow2(float v) { return v*v; }
__device__ double pow2(double v) { return v*v; }
#define MXW_DRAW_ANY(val) *pbuf = val;
__global__ void mxw_draw(float* buf) {
  const ftype d3=1./3; ftype val=0;
  int zshift=blockIdx.y/(Na/NyBloch); if(threadIdx.x+zshift>=Nz) zshift=Nz-1-threadIdx.x;
  int iz=threadIdx.x+zshift;
  DiamondRag* p=&pars.get_plaster(blockIdx.x,blockIdx.y);
  ModelRag* ind=&pars.get_index(blockIdx.x,blockIdx.y);
  const int Npls=2*NDT*NDT;
  if(pars.DrawOnlyRag){
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    float* pbuf=&buf[threadIdx.x+NT*(iy+Ny*ix)];
    switch(pars.nFunc) {
      case 1: {  MXW_DRAW_ANY(p->Si[3].trifld.one[iz] ); } break; //Hy
      case 2: {  MXW_DRAW_ANY(p->Si[4].trifld.two[iz].x); } break; //Hz
      case 0: {  MXW_DRAW_ANY(p->Si[4].trifld.two[iz].y); } break; //Hx
      case 4: {  MXW_DRAW_ANY(p->Vi[4].trifld.one[iz]  ); } break; //Ey
      case 3: {  MXW_DRAW_ANY(p->Vi[3].trifld.two[iz].x); } break; //Ex
      case 5: {  MXW_DRAW_ANY(p->Vi[3].trifld.two[iz].y); } break; //Ez
    }
  } else {
  for(int idom=0; idom<Npls; idom++) {
    int Ragdir=0;
    if(pars.nFunc==3 || pars.nFunc==4 || pars.nFunc==5) Ragdir=1;
    int shx=-NDT+idom/NDT+idom%NDT;
    int shy=+NDT-idom/NDT+idom%NDT;
    if(Ragdir) shy=0+idom/NDT-idom%NDT;
    int idev=0, nextY=NStripe(0);
    #if 1
      while(blockIdx.y>=nextY) nextY+=NStripe(++idev);
      shy-=idev*2*NDT;
      if(blockIdx.y==nextY-NStripe(idev) && idom< Npls/2 && Ragdir==1 && idev!=0     ) continue;
      if(blockIdx.y==nextY-NStripe(idev) && idom>=Npls/2 && Ragdir==0 && idev!=0     ) continue;
      if(blockIdx.y==nextY-1             && idom< Npls/2 && Ragdir==0 && idev!=NDev-1) continue;
      if(blockIdx.y==nextY-1             && idom>=Npls/2 && Ragdir==1 && idev!=NDev-1) continue;
    #endif
    int ix = blockIdx.x*2*NDT+shx+4;
    int iy = blockIdx.y*2*NDT+shy+2;
    float* pbuf=&buf[threadIdx.x+NT*((iy/2)%(Ny-1)+Ny*(ix/2))];
    switch(pars.nFunc) {
      case 1: if(idom%2==0) { pbuf+=NT*Ny; MXW_DRAW_ANY(p->Si[idom/2].trifld.one[iz]  ); } break; //Hy
      case 2: if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Si[idom/2].trifld.two[iz].x); } break; //Hz
      case 0: if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Si[idom/2].trifld.two[iz].y); } break; //Hx
      case 4: if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(p->Vi[idom/2].trifld.one[iz]  ); } break; //Ey
      case 3: if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Vi[idom/2].trifld.two[iz].x); } break; //Ex
      case 5: if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Vi[idom/2].trifld.two[iz].y); } break; //Ez
      case 6: if(idom%2==0) { pbuf+=0    ; ftype Ey = p->Vi[idom/2].trifld.one[iz]  ; atomicAdd(pbuf, Ey*Ey); }
        else  if(idom%2==1) {              ftype Ex = p->Vi[idom/2].trifld.two[iz].x; atomicAdd(pbuf, Ex*Ex);
                                           ftype Ez = p->Vi[idom/2].trifld.two[iz].y; atomicAdd(pbuf, Ez*Ez); } break; //
      #ifdef USE_TEX_2D
      case 7: if(idom%2==0) { pbuf+=0    ; float index = tex2D(index_tex, iy, ix); MXW_DRAW_ANY(index); } break;
      #else
      case 7: if(idom%2==0) { pbuf+=0    ; float index = tex3D(index_tex, threadIdx.x*2, iy, ix); MXW_DRAW_ANY(index); } break;
      #endif
    }
  }
  }
}

struct any_idle_func_struct {
    virtual void step() {}
};
struct idle_func_calc: public any_idle_func_struct {
  float t;
  void step();
};
void idle_func_calc::step() {
  calcStep();
  CHECK_ERROR( cudaMemset(parsHost.arr4im.Arr3Dbuf,0,((long long int)Nx)*Ny*Nz*sizeof(float)) );
  mxw_draw<<<dim3((USE_UVM==2)?Np:Ns,Na),NT>>>(parsHost.arr4im.Arr3Dbuf);
  im3DHost.initCuda(parsHost.arr4im);
  recalc_at_once=true;
}

unsigned char* read_config_file(int& n){
  n = 0; int c; 
  FILE* cfgfile;
  cfgfile = fopen("acts.cfg","r");
  if (cfgfile==NULL) return NULL;
  else {
    c = fgetc(cfgfile); if(c == EOF) {printf("config file is empty"); return NULL; } 
    n = 0;
    while(c != EOF) {
      c = fgetc(cfgfile);
      n++;
    }
    fclose(cfgfile);
  }
  unsigned char* actlist = NULL;
  cfgfile = fopen("acts.cfg","r");
  if (cfgfile==NULL) return NULL;
  else {
    actlist = new unsigned char[n];
    for(int i=0; i<n; i++) { 
      actlist[i] = (unsigned char)fgetc(cfgfile);
      if     (actlist[i]=='\n') actlist[i] = 13;
      else if(actlist[i]=='2' ) actlist[i] = 50;
      else if(actlist[i]=='3' ) actlist[i] = 51;
    }
    fclose(cfgfile);
  }
  return actlist; 
}
int iact = 0;
int nact = 0;
unsigned char* sequence_act = NULL; 
static void key_func(unsigned char key, int x, int y) {
  if(type_diag_flag>=2) printf("keyN=%d, coors=(%d,%d)\n", key, x, y);
  if(key == 'h') {
    printf("\
======= Управление mxw3D:\n\
  <¦>  \tИзменение функции для визуализации: WEH¦Sx¦Ez¦Ey¦Ex¦Hx¦Hy¦Hz¦Sy¦Sz¦eps\n\
«Enter»\tПересчёт одного большого шага\n\
   b   \tвключает пересчёт в динамике (см. «Управление динамикой»)\n\
"); im3DHost.print_help();
    return;
  }
  switch(key) {
  //case '>': if(parsHost.nFunc<parsHost.MaxFunc) parsHost.nFunc++; break;
  //case '<': if(parsHost.nFunc>0) parsHost.nFunc--; break;
  case '>': parsHost.nFunc = (parsHost.nFunc+1)%parsHost.MaxFunc; break;
  case '<': parsHost.nFunc = (parsHost.nFunc+parsHost.MaxFunc-1)%parsHost.MaxFunc; break;
  case 13: calcStep(); break;
  case 'c':
    {
    printf("reading config file\n");
    sequence_act = read_config_file(nact);
    glutPostRedisplay();
    return; 
    }
  default: if(!im3DHost.key_func(key, x, y)) {
  if(type_diag_flag>=0) printf("По клавише %d в позиции (%d,%d) нет никакого действия\n", key, x, y);
  } return;
  }
  copy2dev( parsHost, pars );
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  CHECK_ERROR( cudaMemset(parsHost.arr4im.Arr3Dbuf,0,((long long int)Nx)*Ny*Nz*sizeof(float)) );
  mxw_draw<<<dim3((USE_UVM==2)?Np:Ns,Na),NT>>>(parsHost.arr4im.Arr3Dbuf);
  im3DHost.initCuda(parsHost.arr4im);
  recalc_at_once=true;
}
static void draw_func() { 
  if (iact<nact) { 
    key_func(sequence_act[iact],0,0);
    iact++;
    glutPostRedisplay();
  }
  if(nact>0 && iact==nact) delete[] sequence_act;
  im3DHost.fName = FuncStr[parsHost.nFunc]; im2D.draw(im3DHost.reset_title()); 
}

//void (*idle_func_ptr)(float* );
static void idle_func() { im3DHost.recalc_func(); }
static void mouse_func(int button, int state, int x, int y) { im3DHost.mouse_func(button, state, x, y); }
static void motion_func(int x, int y) { im3DHost.motion_func(x, y); }

double PMLgamma_func(int i, int N, ftype dstep){ //return 0; 
  if(i>=N-3) return 0;
  if(i<0) i=0;
  N-=3;
  double attenuation_factor = 4;
  double sigma_max= shotpoint.V_max*log(10000)*( (attenuation_factor+1)/(2*(N*dstep*0.5)) );
  double x_max = pow(sigma_max, 1./attenuation_factor);
  double x = x_max-i*(x_max/N);
  return pow(x, attenuation_factor);
}
double PMLgamma_funcY(int i, int N, ftype dstep){ //return 0;
  if(i>=N-3) return 0;
  if(i<0) i=0;
  N-=3;
  double attenuation_factor = 4;
  double sigma_max= shotpoint.V_max*log(10000)*( (attenuation_factor+1)/(2*(N*dstep*0.5)) );
  double x_max = pow(sigma_max, 1./attenuation_factor);
  double x = x_max-i*(x_max/N);
  return pow(x, attenuation_factor);
}
double PMLgamma_funcZ(int i, int N, ftype dstep){ //return 0; 
  if(i>=N-3) return 0;
  if(i<0) i=0;
  N-=3;
  double attenuation_factor = 4;
  double sigma_max= shotpoint.V_max*log(10000)*( (attenuation_factor+1)/(2*(N*dstep*0.5)) );
  double x_max = pow(sigma_max, 1./attenuation_factor);
  double x = x_max-i*(x_max/N);
  return pow(x, attenuation_factor);
}
void setPMLcoeffs(ftype* k1x, ftype* k2x, ftype* k1y, ftype* k2y, ftype* k1z, ftype* k2z) {
  for(int i=0; i<KNpmlx; i++){
    k2x[i] = 1.0/(1.0+0.5*dt*PMLgamma_func(KNpmlx/2-abs(i-KNpmlx/2)-3, KNpmlx/2-3, dx));
    k1x[i] = 2.0*k2x[i]-1;
  }
  for(int i=0; i<KNpmly; i++){
    //k2y[i] = 1.0/(1.0+0.5*dt*PMLgamma_funcY(KNpmly-i, KNpmly, dy));
    k2y[i] = 1.0/(1.0+0.5*dt*PMLgamma_func((i<KNpmly/2)?i:(KNpmly-i-3), KNpmly/2, dy));
    k1y[i] = 2.0*k2y[i]-1;
  }
  for(int i=0; i<KNpmlz; i++){
    k2z[i] = 1.0/(1.0+0.5*dt*PMLgamma_funcZ((i<KNpmlz/2)?i:(KNpmlz-i-1), KNpmlz/2, dz));
    if(i<KNpmlz/2) k2z[i] = 1.0;
    k1z[i] = 2.0*k2z[i]-1;
  }
}
void setPeer2Peer(int node,int subnode, int* isp2p){
  for(int i=0; i<NDev; i++) for(int j=i+1; j<NDev; j++) {
      int canp2p=0; CHECK_ERROR(cudaDeviceCanAccessPeer(&canp2p,i,j));
      if(canp2p) { 
        CHECK_ERROR(cudaSetDevice(i)); CHECK_ERROR(cudaDeviceEnablePeerAccess(j,0));
        CHECK_ERROR(cudaSetDevice(j)); CHECK_ERROR(cudaDeviceEnablePeerAccess(i,0));
              printf("node.subnode %d.%d: %d<-->%d can Peer2Peer\n"   , node, subnode, i,j);
      } else  printf("node.subnode %d.%d: %d<-->%d cannot Peer2Peer\n", node, subnode, i,j);
      if(j==i+1) isp2p[i]=canp2p;
  }
  CHECK_ERROR(cudaSetDevice(0)); 
}
void GeoParamsHost::set(){
  
  #ifndef USE_WINDOW
  if(Np!=Ns) { printf("Error: if not defined USE_WINDOW Np must be equal Ns\n"); exit(-1); }
  #endif//USE_WINDOW

  node=0; subnode=0; int Nprocs=1;
  #ifdef MPI_ON
  MPI_Comm_rank (MPI_COMM_WORLD, &node);   subnode=node%NasyncNodes; node/= NasyncNodes;
  MPI_Comm_size (MPI_COMM_WORLD, &Nprocs); 
  if(node==0) printf("Total MPI tasks: %d\n", Nprocs);
  #endif
  if(Nprocs%NasyncNodes!=0) { printf("Error: mpi procs (%d) must be dividable by NasyncNodes(%d)\n",Nprocs,NasyncNodes); exit(-1); }
  Nprocs/= NasyncNodes;
  mapNodeSize = new int[Nprocs];
  int accSizes=0;
  mapNodeSize[0] = Np/Nprocs+Ns/2; for(int i=1; i<Nprocs; i++) mapNodeSize[i] = Np/Nprocs+Ns;
  int sums=0; for(int i=0; i<Nprocs-1; i++) sums+= mapNodeSize[i]-Ns; mapNodeSize[Nprocs-1]=Np-sums;
  for(int i=0; i<Nprocs; i++) {
    if(node==i) printf("X-size=%d rags on node %d\n", mapNodeSize[i], i);
    #ifdef MPI_ON
    if(mapNodeSize[i]<2*Ns && Nprocs>1) { printf("Data on node %d is too small\n", i); exit(-1); }
    #endif
    accSizes+= mapNodeSize[i];
  }
  if(accSizes-Ns*(Nprocs-1)!=Np) { printf("Error: sum (mapNodes) must be = Np+Ns*(Nprocs-1)\n"); exit(-1); }
  #ifdef MPI_ON
  if(mapNodeSize[0]       <=Npmlx/2+Ns+Ns && Nprocs>1) { printf("Error: mapNodeSize[0]<=Npmlx/2+Ns+Ns\n"); exit(-1); }
  if(mapNodeSize[Nprocs-1]<=Npmlx/2+Ns+Ns && Nprocs>1) { printf("Error: mapNodeSize[Nodes-1]<=Npmlx/2+Ns+Ns\n"); exit(-1); }
  #endif
  if(Np%Ns!=0) { printf("Error: Np must be dividable by Ns\n"); exit(-1); }
  if(NB%NA!=0) { printf("Error: NB must be dividable by NA\n"); exit(-1); }
  if(NB<NA   ) { printf("Error: NB < NA\n"); exit(-1); }
  omp_set_num_threads(8);

  //dir= new string("/Run/zakirov/tmp/"); //ix=Nx+Nbase/2; Yshtype=0;
  dir= new std::string(im3DHost.drop_dir);
  drop.dir=dir;
  struct stat st = {0};

  if (stat(dir->c_str()     , &st) == -1)  mkdir(dir->c_str()     , 0700);
  if (stat(swap_dir->c_str(), &st) == -1)  mkdir(swap_dir->c_str(), 0700);
  
  for(int i=0;i<NDev-1;i++) isp2p[i]=0;
  setPeer2Peer(node,subnode,isp2p);
  for(int i=0;i<NDev-1;i++) isp2p[i]=1;
  
  if(node==0) print_info();
  Tsteps+= node*Ntime;
  if(node==0) printf("Full %d Big steps\n", Tsteps/Ntime);
  if(node==0) printf("Grid size: %dx%d Rags /%dx%dx%d Yee_cells/, TorreH=%d\n", Np, Na, Np*NDT,Na*NDT,Nv, Ntime);
  if(node==0) printf("Window size: %d, copy-shift step %d \n", Ns, Window::NTorres );
  if(gridNx%NDT!=0) { printf("Error: gridNx must be dividable by %d\n", NDT); exit(-1); }
  if(gridNy%NDT!=0) { printf("Error: gridNy must be dividable by %d\n", NDT); exit(-1); }
  if(dt*sqrt(1/(dx*dx)+1/(dy*dy)+1/(dz*dz))>6./7.) { printf("Error: Courant condition is not satisfied\n"); exit(-1); }
  if(dispReg::sL>dispReg::sR) { printf("error: dispReg::sL>dispReg::sR\n"); exit(-1); }
  if(dispReg::vL>dispReg::vR) { printf("error: dispReg::vL>dispReg::vR\n"); exit(-1); }
//  if(sizeof(DiamondRag)!=sizeof(RagArray)) { printf("Error: sizeof(DiamondRag)=%d != sizeof(RagArray)\n", sizeof(DiamondRag),sizeof(RagArray)); exit(-1); }
  int NaStripe=0; for(int i=0;i<NDev;i++) NaStripe+=NStripe[i]; if(NaStripe!=Na) { printf("Error: sum(NStripes[i])!=NA\n"); exit(-1); }
  iStep = 0; isTFSF=true;
  Zcnt=0.5*Nz*dz;
  IndNx=2*(Np*NDT+2); IndNy=2*(Na*NDT+1); IndNz=2*Nz;
  #ifdef COFFS_DEFAULT
  IndNx=1; IndNy=1; IndNz=1;
  #endif
  nFunc = 0; MaxFunc = sizeof(FuncStr)/sizeof(char*);
  size_t size_xz     = Ns   *sizeof(DiamondRag   );
  size_t size_xzModel= Ns   *sizeof(ModelRag     );
  size_t sz          = Na*size_xz;
  size_t szModel     = Na*size_xzModel;
  size_t szBuf       = Ntime*sizeof(halfRag      );
  size_t szPMLa      = Ns*Npmly*sizeof(DiamondRagPML);
  size_t size_xzPMLs = Npmlx/2*sizeof(DiamondRagPML);
  #ifdef NOPMLS
  size_xzPMLs = 0;
  #endif
  size_t szPMLs      = Na*size_xzPMLs;
  size_t size_xzDisp =(dispReg::sRdev-dispReg::sL)*sizeof(DiamondRagDisp);
  size_t szDisp =Na*size_xzDisp;
  if(node==0) {
  printf("GPU Cell's Array size     : %7.2fM = %7.2fM(Main)+%7.2fM(Buffers)+%7.2fM(Model)+%7.2fM(Disp)+%7.2fM(PMLs)+%7.2fM(PMLa)\n", 
           (sz+2*NDev*szBuf+szModel+szDisp+szPMLs+szPMLa)/(1024.*1024.),
           sz     /(1024.*1024.), 
      NDev*szBuf*2/(1024.*1024.), 
           szModel/(1024.*1024.), 
           szDisp /(1024.*1024.), 
           szPMLs /(1024.*1024.), 
           szPMLa /(1024.*1024.)  );
  for(int istrp=0; istrp<NDev-1; istrp++) printf( "                   Stripe%d: %7.2fM = %7.2fM      +%7.2fM         +%7.2fM       +%7.2fM      +%7.2fM\n", istrp, 
           (size_xz*NStripe[istrp]+2*szBuf+size_xzModel*NStripe[istrp]+size_xzDisp*NStripe[istrp]+size_xzPMLs*NStripe[istrp])/(1024.*1024.),
           size_xz    *NStripe[istrp ]/(1024.*1024.), 
           szBuf*2                    /(1024.*1024.), 
           size_xzModel*NStripe[istrp]/(1024.*1024.),
           size_xzDisp*NStripe[istrp ]/(1024.*1024.),
           size_xzPMLs*NStripe[istrp ]/(1024.*1024.)  );
                                          printf( "                   Stripe%d: %7.2fM = %7.2fM      +%7.2fM         +%7.2fM       +%7.2fM      +%7.2fM      +%7.2fM\n", NDev-1, 
           (size_xz*NStripe[NDev-1]+2*szBuf+size_xzModel*NStripe[NDev-1]+size_xzDisp*NStripe[NDev-1]+size_xzPMLs*NStripe[NDev-1]+szPMLa)/(1024.*1024.),
           size_xz    *NStripe[NDev-1]/(1024.*1024.), 
           szBuf*2                    /(1024.*1024.), 
           size_xzModel*NStripe[NDev-1]/(1024.*1024.),
           size_xzDisp*NStripe[NDev-1]/(1024.*1024.),
           size_xzPMLs*NStripe[NDev-1]/(1024.*1024.), 
           szPMLa                      /(1024.*1024.)  );
  }
  size_t freemem[NDev], totalmem[NDev];
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR( cudaSetDevice(idev) );
    CHECK_ERROR( cudaMalloc( (void**)&(ragsInd  [idev]), size_xzModel*NStripe[idev]) );
    CHECK_ERROR( cudaMalloc( (void**)&(rags     [idev]), size_xz     *NStripe[idev]) );
    CHECK_ERROR( cudaMalloc( (void**)&(p2pBufM  [idev]), szBuf     ) );
    CHECK_ERROR( cudaMalloc( (void**)&(p2pBufP  [idev]), szBuf     ) );
    CHECK_ERROR( cudaMalloc( (void**)&(ragsDisp [idev]), size_xzDisp *NStripe[idev]) );
//    CHECK_ERROR( cudaMalloc( (void**)&(ragsPMLs[idev]), size_xzPMLs*NStripe[idev]    ) );
    #ifndef USE_WINDOW
    CHECK_ERROR( cudaMalloc( (void**)&(ragsPMLsL[idev]), size_xzPMLs*NStripe[idev]    ) );
    CHECK_ERROR( cudaMalloc( (void**)&(ragsPMLsR[idev]), size_xzPMLs*NStripe[idev]    ) );
    #endif
    CHECK_ERROR( cudaMalloc( (void**)& ragsPMLa[idev]  , szPMLa ) );
    CHECK_ERROR( cudaMemset(rags    [idev], 0, size_xz    *NStripe[idev])  );
    CHECK_ERROR( cudaMemset(p2pBufM [idev], 0, szBuf)  );
    CHECK_ERROR( cudaMemset(p2pBufP [idev], 0, szBuf)  );
    CHECK_ERROR( cudaMemset(ragsInd [idev], 0, size_xzModel*NStripe[idev]) );
    CHECK_ERROR( cudaMemset(ragsDisp[idev], 0, size_xzDisp*NStripe[idev])  );
    #ifndef USE_WINDOW
    cudaMemset(ragsPMLsL[idev], 0,  size_xzPMLs*NStripe[idev]);
    cudaMemset(ragsPMLsR[idev], 0,  size_xzPMLs*NStripe[idev]);
    #endif
    if(idev==NDev-1)
    CHECK_ERROR( cudaMemset(ragsPMLa[idev], 0,                     szPMLa) );
    CHECK_ERROR( cudaMemGetInfo(&freemem[idev], &totalmem[idev]));
    printf("Node/subnode %3d/%d : device %d: GPU memory free %.2fM of %.2fM\n", node, subnode, idev, freemem[idev]/(1024.*1024.), totalmem[idev]/(1024.*1024.) );
  }
  CHECK_ERROR( cudaSetDevice(0) );

  const int Nn = mapNodeSize[node];
  #if 1//USE_WINDOW
  printf("Allocating RAM memory on node %d: %g Gb\n", node, (Nn*Na*sizeof(DiamondRag)+Nn*Na*sizeof(ModelRag)+Nn*Na*sizeof(DiamondRagDisp)+Nn*Npmly*sizeof(DiamondRagPML)+Npmlx*Na*sizeof(DiamondRagPML))/(1024.*1024.*1024.));
  #if USE_UVM==2
  #ifdef SWAP_DATA
  char swapdata[256]; sprintf(swapdata, "%s/swapdata.%d.%d", swap_dir->c_str(), node,subnode);
  int swp_data; swp_data = open(swapdata,O_RDWR|O_TRUNC|O_CREAT, 0666);
  if(swp_data==-1) { char s[128]; sprintf(s,"Error opening file %s at %d.%d",swapdata,node,subnode); perror(s); exit(-1); }
  lseek(swp_data, Nn*Na*sizeof(DiamondRag), SEEK_SET);
  write(swp_data, "", 1); lseek(swp_data, 0, SEEK_SET);
  data = (DiamondRag*)mmap(0, Nn*Na*sizeof(DiamondRag), PROT_READ|PROT_WRITE, MAP_SHARED, swp_data,0);
  if(data == MAP_FAILED) { char s[128]; sprintf(s,"Error mmap data at %d.%d",node,subnode); perror(s); exit(-1); }
  close(swp_data);
  #else
  CHECK_ERROR( cudaMallocHost(&data     , Nn*Na     *sizeof(DiamondRag    )) );
  #endif//SWAP_DATA
  memset(data     , 0, Nn*Na     *sizeof(DiamondRag    ));
  CHECK_ERROR( cudaMallocHost(&dataInd  , Nn*Na     *sizeof(ModelRag      )) ); memset(dataInd  , 0, Nn*Na     *sizeof(ModelRag      ));
  CHECK_ERROR( cudaMallocHost(&dataDisp , Nn*Na     *sizeof(DiamondRagDisp)) ); memset(dataDisp , 0, Nn*Na     *sizeof(DiamondRagDisp));
  CHECK_ERROR( cudaMallocHost(&dataPMLa , Nn*Npmly  *sizeof(DiamondRagPML )) ); memset(dataPMLa , 0, Nn*Npmly  *sizeof(DiamondRagPML ));
  CHECK_ERROR( cudaMallocHost(&dataPMLsL, Npmlx/2*Na*sizeof(DiamondRagPML )) ); memset(dataPMLsL, 0, Npmlx/2*Na*sizeof(DiamondRagPML ));
  CHECK_ERROR( cudaMallocHost(&dataPMLsR, Npmlx/2*Na*sizeof(DiamondRagPML )) ); memset(dataPMLsR, 0, Npmlx/2*Na*sizeof(DiamondRagPML ));
  if (node==1) printf("data allocated, pointer to %p\n", data);
  for(int i=0; i<node; i++) data    -= mapNodeSize[i]*Na   ; data    +=node*Ns*Na;
  for(int i=0; i<node; i++) dataInd -= mapNodeSize[i]*Na   ; dataInd +=node*Ns*Na;
  for(int i=0; i<node; i++) dataDisp-= mapNodeSize[i]*Na   ; dataDisp+=node*Ns*Na;
  for(int i=0; i<node; i++) dataPMLa-= mapNodeSize[i]*Npmly; dataPMLa+=node*Ns*Npmly;
  if (node==1) printf("now data points to %p\n", data);
  #else
  data     = new DiamondRag   [Nn*Na   ]; memset(data    , 0, Nn*Na   *sizeof(DiamondRag   ));
  dataPMLa = new DiamondRagPML[Nn*Npmly]; memset(dataPMLa, 0, Nn*Npmly*sizeof(DiamondRagPML));
  dataPMLs = new DiamondRagPML[Npmlx*Na]; memset(dataPMLs, 0, Npmlx*Na*sizeof(DiamondRagPML));
  #endif
  #endif
  fflush(stdout);
  //size_t size_rdma = sizeof(DiamondRag)*(NDT*NDT/2+1);
  size_t size_rdma = szBuf;
  CHECK_ERROR( cudaMallocHost( (void**)&rdma_send_buf, size_rdma ) );
  CHECK_ERROR( cudaMallocHost( (void**)&rdma_recv_buf, size_rdma ) );
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR( cudaMallocHost( (void**)&(p2pBufM_host_snd[idev]), szBuf     ) );
    CHECK_ERROR( cudaMallocHost( (void**)&(p2pBufP_host_snd[idev]), szBuf     ) );
    CHECK_ERROR( cudaMallocHost( (void**)&(p2pBufM_host_rcv[idev]), szBuf     ) );
    CHECK_ERROR( cudaMallocHost( (void**)&(p2pBufP_host_rcv[idev]), szBuf     ) );
    CHECK_ERROR( cudaMemset(p2pBufM_host_snd[idev], 0, szBuf)  );
    CHECK_ERROR( cudaMemset(p2pBufP_host_snd[idev], 0, szBuf)  );
    CHECK_ERROR( cudaMemset(p2pBufM_host_rcv[idev], 0, szBuf)  );
    CHECK_ERROR( cudaMemset(p2pBufP_host_rcv[idev], 0, szBuf)  );
  }
  for(int i=0; i<NDev-1; i++) {
    size_t size_p2p = sizeof(DiamondRag)*(NDT*NDT/2+1);
    p2p_buf[i]=0;
    if(isp2p[i]) CHECK_ERROR( cudaMallocHost( (void**)&p2p_buf[i], size_p2p ) );
  }

  drop.init();
  texs.init();
  cuTimer t0; t0.init();
  int xL=0; for(int inode=0; inode<node; inode++) xL+= mapNodeSize[inode]; xL-= Ns*node;
  int xR = xL+mapNodeSize[node];
/*  omp_set_num_threads(4);
  for(int x=0;x<Np;x++) {
    printf("Initializing h-parameter %.2f%%      \r",100*double(x+1)/Np); fflush(stdout);
    if(x>=xL && x<xR) { 
      #pragma omp parallel for
      for(int y=0;y<Na;y++) dataInd[x*Na+y].set(x,y);
    }
  }*/
//  printf("t0=%g\n",t0.gettime());
  
  sensors = new std::vector<Sensor>();

  if(Nz<NyBloch) { printf("Error: Nz must be >= NyBloch\n"); exit(-1);}
  //if(NyBloch!=1 && Ntime!=1) { printf("Error: Ntime must be =1 for NyBloch!=1\n"); exit(-1);}
}

void GeoParamsHost::sensors_set_rag(){
  int xL=0; for(int inode=0; inode<node; inode++) xL+= mapNodeSize[inode]; xL-= Ns*node;
  int xR = xL+mapNodeSize[node];
  for(int x=0;x<Np;x++) {
    if(x>=xL && x<xR) { 
      for(int y=0;y<Na;y++) {
         for(vector<Sensor>::iterator sit=sensors->begin(); sit!=sensors->end(); ++sit) {
           sit->set_rag(x,y);
         }
      }
    }
  }
}
void init_index() {
  //-------Set PML coeffs----------------------------//
  hostKpmlx1 = new ftype[KNpmlx]; hostKpmlx2 = new ftype[KNpmlx];
  hostKpmly1 = new ftype[KNpmly]; hostKpmly2 = new ftype[KNpmly];
  hostKpmlz1 = new ftype[KNpmlz]; hostKpmlz2 = new ftype[KNpmlz];
  setPMLcoeffs(hostKpmlx1, hostKpmlx2, hostKpmly1, hostKpmly2, hostKpmlz1, hostKpmlz2);
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR( cudaSetDevice(idev) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlx1, hostKpmlx1, sizeof(ftype)*KNpmlx) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlx2, hostKpmlx2, sizeof(ftype)*KNpmlx) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmly1, hostKpmly1, sizeof(ftype)*KNpmly) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmly2, hostKpmly2, sizeof(ftype)*KNpmly) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlz1, hostKpmlz1, sizeof(ftype)*KNpmlz) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlz2, hostKpmlz2, sizeof(ftype)*KNpmlz) );
  }
  CHECK_ERROR( cudaSetDevice(0) );
  //-----------------------------------------------------------------------------------//

/*  parsHost.sensors->push_back(Sensor("Ez",(X0+Rreson)/dx+1,Y0/dy,Z0/dz));
  parsHost.sensors->push_back(Sensor("Ez",(X0-Rreson)/dx-1,Y0/dy,Z0/dz));
  parsHost.sensors->push_back(Sensor("Ez",X0/dx,(Y0+Rreson)/dy+1,Z0/dz));
  parsHost.sensors->push_back(Sensor("Ez",X0/dx,(Y0-Rreson)/dy-1,Z0/dz));*/
  ftype Xdip=(shotpoint.BoxMs+shotpoint.BoxPs)*0.5;
  ftype Ydip=(shotpoint.BoxMa+shotpoint.BoxPa)*0.5;
  ftype Zdip=(shotpoint.BoxMv+shotpoint.BoxPv)*0.5;
  int X0=int(round(Xdip/dx));
  int Y0=int(round(Ydip/dy));
  int Z0=int(round(0.150/dz));
  for(double xsen=0.0; xsen<Np*NDT*dx/2; xsen+=1.0) {
    int xcrd = X0+int(round(xsen/dx));
    parsHost.sensors->push_back(Sensor("Ex",xcrd,Y0,Z0));
    parsHost.sensors->push_back(Sensor("Ey",xcrd,Y0,Z0));
    parsHost.sensors->push_back(Sensor("Ez",xcrd,Y0,Z0));
    parsHost.sensors->push_back(Sensor("Hx",xcrd,Y0,Z0));
    parsHost.sensors->push_back(Sensor("Hy",xcrd,Y0,Z0));
    parsHost.sensors->push_back(Sensor("Hz",xcrd,Y0,Z0));
  }
  for(double ysen=1.0; ysen<Na*NDT*dy/2; ysen+=1.0) {
    int ycrd = Y0+int(round(ysen/dy));
    parsHost.sensors->push_back(Sensor("Ex",X0,ycrd,Z0));
    parsHost.sensors->push_back(Sensor("Ey",X0,ycrd,Z0));
    parsHost.sensors->push_back(Sensor("Ez",X0,ycrd,Z0));
    parsHost.sensors->push_back(Sensor("Hx",X0,ycrd,Z0));
    parsHost.sensors->push_back(Sensor("Hy",X0,ycrd,Z0));
    parsHost.sensors->push_back(Sensor("Hz",X0,ycrd,Z0));
  }

  parsHost.sensors_set_rag();

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();
  #ifdef USE_TEX_2D
  CHECK_ERROR(cudaMallocArray(&index_texArray, &channelDesc, parsHost.IndNy,parsHost.IndNx));
  #else
  CHECK_ERROR(cudaMalloc3DArray(&index_texArray, &channelDesc, make_cudaExtent(parsHost.IndNz,parsHost.IndNy,parsHost.IndNx)));
  #endif
}
void set_texture(char* index_arr, const int ix=0){
  CHECK_ERROR(cudaUnbindTexture(index_tex));
  cudaMemcpy3DParms myparms = {0};
  myparms.srcPos = make_cudaPos(0,0,0);
  myparms.dstPos = make_cudaPos(0,0,0);
  myparms.dstArray = index_texArray;
  myparms.kind = cudaMemcpyHostToDevice;
  index_tex.normalized = false;
  index_tex.filterMode = cudaFilterModePoint;//cudaFilterModeLinear; //filter_pal?cudaFilterModePoint:cudaFilterModeLinear;
  #ifdef USE_TEX_2D
  myparms.srcPtr = make_cudaPitchedPtr(&index_arr[ix*parsHost.IndNy], sizeof(char), 1, parsHost.IndNy);
  myparms.extent = make_cudaExtent(sizeof(char),parsHost.IndNy,parsHost.IndNx);
  CHECK_ERROR(cudaMemcpy2DToArray(index_texArray,0,0,&index_arr[ix*parsHost.IndNy],parsHost.IndNy*sizeof(char),parsHost.IndNy*sizeof(char),parsHost.IndNx,cudaMemcpyHostToDevice));
//  index_tex.readMode =  cudaReadModeElementType;
  index_tex.addressMode[0] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  index_tex.addressMode[1] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  #else
  myparms.srcPtr = make_cudaPitchedPtr(&index_arr[ix*parsHost.IndNy*parsHost.IndNz], parsHost.IndNz*sizeof(char), parsHost.IndNz, parsHost.IndNy);
//  myparms.srcPtr = make_cudaPitchedPtr(index_arr, Ny*sizeof(char), Nx, Ny);
  myparms.extent = make_cudaExtent(parsHost.IndNz*sizeof(char),parsHost.IndNy,parsHost.IndNx);
  CHECK_ERROR(cudaMemcpy3D(&myparms));
//  index_tex.readMode =  cudaReadModeElementType;
  index_tex.addressMode[0] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  index_tex.addressMode[1] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  index_tex.addressMode[2] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  #endif
  CHECK_ERROR(cudaBindTextureToArray(index_tex, index_texArray));
}

__device__ __constant__ CoffStruct coffs[Nmats];
__device__ __constant__ DispStruct DispCoffs[Nmats];
CoffStruct* __restrict__ coffsHost;
DispStruct* __restrict__ DispCoffsHost;

inline bool inCell(double y){
   double period = 0.82;
   double cell = 0.72;
   double shift = 0.5*cell;
   if (fmod(y+shift+1000*period, period)<=cell) return true;
   return false;
 };

void init_material(char* &index_arr) {
  const float n1=1, n2=sqrt(5.0), n3=sqrt(2*2);
  coffsHost     = new CoffStruct[Nmats];
  DispCoffsHost = new DispStruct[Nmats];
  coffsHost[IndAir  ].set_eps(1.0  , 0); DispCoffsHost[IndAir  ].setNondisp(); 
  coffsHost[IndGold ].set_eps(n2*n2, 1);
  float3 gyr = make_float3(0.,0.,0.1);
  coffsHost[IndBIG  ].set_eps(n2*n2, 0, gyr.x,gyr.y,gyr.z); DispCoffsHost[IndBIG  ].setNondisp();
  coffsHost[IndGGG  ].set_eps(n3*n3, 0); DispCoffsHost[IndGGG  ].setNondisp();
  coffsHost[IndOut  ].set_eps(100*100, 0); DispCoffsHost[IndOut  ].setNondisp();
  coffsHost[IndVac  ].set_eps(1*1, 0); DispCoffsHost[IndVac  ].setNondisp();
  coffsHost[IndAg   ].set_eps(n1*n1, 1);
  coffsHost[IndOrg  ].set_eps(1.75*1.75, 0); DispCoffsHost[IndOrg  ].setNondisp();
  coffsHost[IndITO  ].set_eps(1.8*1.8  , 0); DispCoffsHost[IndITO  ].setNondisp();
  coffsHost[IndGlass].set_eps(1.5*1.5  , 0); DispCoffsHost[IndGlass].setNondisp();
  coffsHost[IndSh1  ].set_eps(1*1, 0); DispCoffsHost[IndSh1  ].setNondisp();
  coffsHost[IndSh2  ].set_eps(2*2, 0); DispCoffsHost[IndSh2  ].setNondisp();
  coffsHost[IndSh3  ].set_eps(3*3, 0); DispCoffsHost[IndSh3  ].setNondisp();
  coffsHost[IndSh4  ].set_eps(4*4, 0); DispCoffsHost[IndSh4  ].setNondisp();
  coffsHost[IndSh5  ].set_eps(5*5, 0); DispCoffsHost[IndSh5  ].setNondisp();
  
  LorenzDisp ld1;
  LorenzDisp ld2;
  ld1.setDrude(8.5, 4.9*M_PI, 0.21, 1e-3);
  ld2.set(1.0, 0.5*2*M_PI, 30*2*M_PI, 340*M_PI);
  vector<LorenzDisp> LDvec; LDvec.push_back(ld1); LDvec.push_back(ld2);
  DispCoffsHost[IndGold ].set(1, LDvec, LDvec.size(), 0);

  LorenzDisp AgP;
  AgP.setDrude(1.0, 136.66/3.0/*M_PI*/, 2.69e13/2/3e14, 0);
  vector<LorenzDisp> Agvec; Agvec.push_back(AgP);
  DispCoffsHost[IndAg ].set(1, Agvec, Agvec.size(), 0);

  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR( cudaSetDevice(idev) );
    CHECK_ERROR( cudaMemcpyToSymbol(coffs, coffsHost, sizeof(CoffStruct)*Nmats) );
    CHECK_ERROR( cudaMemcpyToSymbol(DispCoffs, DispCoffsHost, sizeof(DispStruct)*Nmats) );
  }
  CHECK_ERROR( cudaSetDevice(0) );
  //------------------------------------------------//
//  const int Nx=16, Ny=16, Nz=16;
  index_arr=new char[parsHost.IndNx*parsHost.IndNy*parsHost.IndNz];
  char* p=index_arr;

  for(int ix=0; ix<parsHost.IndNx; ix++) for(int iy=0,iz=0; iy<parsHost.IndNy; iy++) 
  #ifndef USE_TEX_2D
  for(iz=0; iz<parsHost.IndNz; iz++) 
  #endif
  {
    float x=ix*0.5*dx, y=iy*0.5*dy, z=iz*0.5*dz;
    float Xc=0.5*Np*NDT*dx, Yc=0.5*Na*NasyncNodes*NDT*dy, Zc=0.5*Nv*dz;
    if( (x-Xc)*(x-Xc) + (y-Yc)*(y-Yc) + (z-Zc)*(z-Zc)<= Rparticle*Rparticle ) p[0]=IndGGG;//1./(n2*n2);
//    if( (z-Zc)*(z-Zc)<= Rparticle*Rparticle ) p[0]=IndGold;//1./(n2*n2);
    else p[0]=IndAir;

    p[0]=IndAir;
/*    bool isGold=0;
    int nperiod=0;
    for (nperiod=0;nperiod<8; nperiod++) { isGold = isGold || 
    (x-Xc)*(x-Xc)<=0.04*0.04/4 && (y-Yc+nperiod*0.730)*(y-Yc+nperiod*0.730)<=0.630*0.630/4 ||
    (x-Xc)*(x-Xc)<=0.04*0.04/4 && (y-Yc-nperiod*0.730)*(y-Yc-nperiod*0.730)<=0.630*0.630/4;
    }
    
    //float Ylength = 2*Na*NDT*dy;//(Na-Npmly-2)*NDT*dy;
    float Ylength = (Na-Npmly-2)*NDT*dy;
    float Zlength = Nv*dz;
    float Z0 = Nv*dz*0.5;
    float Y0 = Na*NDT*dy*0.5;

    if (
      (fabs(y-Y0   ) <= 0.5*Ylength     ) &&
      (fabs(z-Z0   ) <= 0.5*Zlength     ) &&
      (fabs(x-xGold) <= 0.5*thinGold) &&
      ( inCell(y-Yc) )
    ) p[0] = IndGold;
// Get BIG
    else if(
      (fabs(y-Y0   ) <= 0.5*Ylength     ) &&
      (fabs(z-Z0   ) <= 0.5*Zlength     ) &&
      (fabs(x-xBIG)  <= 0.5*thinBIG)
    ) p[0] = IndBIG;
// Get Air
    else if (
      (fabs(y-Y0   ) <= 0.5*Ylength     ) &&
      (fabs(z-Z0   ) <= 0.5*Zlength     ) &&
      (x>xBIG+0.5*thinBIG)
    ) p[0] = IndGGG;
    else
     p[0] = IndAir;
*/

//    if ((x-Xc)*(x-Xc)<=0.04*0.04/4 && (y-Yc>nperiod*0.730)) isGold=1;
//    if( (x-Xc)*(x-Xc)+(y-Yc)*(y-Yc)+(z-Zc)*(z-Zc)<Rparticle*Rparticle  ) p[0]=IndMat2;//1./(n2*n2);
//    else if(x>Xc+Rparticle) p[0]=IndMat1;
//    else p[0]=IndVac;//1.;
    p++;
  }
  parsHost.inc_ang = asin(dt*(NyBloch-1)/(Na*NDT*dy));//1*1/180.0*M_PI;
  if(Ny<NDT*Na) parsHost.DrawOnlyRag=1;
  else parsHost.DrawOnlyRag=0;
  //printf("tfsfSm=%g, tfsfSp=%g\n",tfsfSm/dx, tfsfSp/dx);
  printf("incident angle =%g degrees\n",parsHost.inc_ang/M_PI*180);
}
int print_help() {
  printf("using: ./DFmxw [--help] [--zoom \"1. 1. 1.\"] [--step \"1. 1. 1.\"] [--box \"1. 1. 1.\"] [--mesh \"200. 200. 200.\"] [--Dmesh 5.] [--drop_dir \".\"] [--bkgr_col \"0.1 0.1 0.1\"] [--mesh_col \"0.8 0.8 0.2\"] [--box_col \"1. 1. 1.\"] [--sensor \"1 1 1\"]\n");
  printf("  --zoom\tмасштабный фактор, действует на 2D режим и размер окна, [1. 1. 1.];\n");
  printf("  --box \tкоррекция пропорций размера бокса в 3D режиме, [1. 1. 1.];\n");
  printf("  --step \tшаги между точками, действует только на тики, [1. 1. 1.];\n");
  printf("  --mesh\tрасстояние между линиями сетки в боксе по координатам в ячейках (до коррекции), [200. 200. 200.];\n");
  printf("  --Dmesh\tширина линии сетки в пикселях (со сглаживанием выглядит несколько уже), [5.];\n");
  printf("  --drop_dir\tимя директории, в которую будут сохраняться различные файлы, [.];\n");
  printf("  --bkgr_col\tцвет фона, [0.1 0.1 0.1];\n");
  printf("  --mesh_col\tцвет линий сетки, [0.8 0.8 0.2];\n");
  printf("  --box_col\tцвет линий бокса, [1.0 1.0 1.0];\n");
  printf("  --sensor\tкоординаты сенсора, можно задавать несколько сенсоров;\n");
  return 0;
}
void read_float3(float* v, char* str) {
  for(int i=0; i<3; i++) { v[i] = strtof(str, &str); str++; }
}
float read_float(char* str) {
  return atof(str);
}
void add_sensor(int ix, int iy, int iz);

bool help_only=false, test_only=false;
int Tsteps=Ntime*10;
int _main(int argc, char** argv) {
  #ifdef MPI_ON
//  MPI_Init(&argc,&argv);
  int ismpith;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ismpith);
  switch(ismpith) {
    case MPI_THREAD_SINGLE:     printf("MPI multithreading implementation MPI_TREAD_SINGLE\n"); break;
    case MPI_THREAD_FUNNELED:   printf("MPI multithreading implementation MPI_TREAD_FUNNELED\n"); break;
    case MPI_THREAD_SERIALIZED: printf("MPI multithreading implementation MPI_THREAD_SERIALIZED\n"); break;
    case MPI_THREAD_MULTIPLE:   printf("MPI multithreading implementation MPI_THREAD_MULTIPLE\n"); break;
    default: printf("Unknown MPI multithreading implementation\n"); break;
  }
  //if (ismpith != MPI_THREAD_MULTIPLE) { printf("Error: MPI implementation does not support multithreading\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
  MPI_Type_contiguous( sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, &MPI_DMDRAGTYPE );
  MPI_Type_contiguous( sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, &MPI_RAGPMLTYPE );
  MPI_Type_contiguous( sizeof(halfRag      )/sizeof(ftype), MPI_FTYPE, &MPI_HLFRAGTYPE );
  MPI_Type_commit(&MPI_DMDRAGTYPE);
  MPI_Type_commit(&MPI_RAGPMLTYPE);
  MPI_Type_commit(&MPI_HLFRAGTYPE);
  #endif
  argv ++; argc --;
  im3DHost.reset(); parsHost.swap_dir=new std::string("./");
  while(argc>0 && strncmp(*argv,"--",2)==0) {
    if(strncmp(*argv,"--help",6)==0) return print_help();
    else if(strcmp(*argv,"--test")==0) { test_only = true; argv ++; argc --; continue; }
    if(strcmp(*argv,"--box")==0) read_float3(im3DHost.BoxFactor, argv[1]);
    else if(strcmp(*argv,"--test")==0) test_only = true;
    else if(strcmp(*argv,"--mesh")==0) read_float3(im3DHost.MeshBox, argv[1]);
    else if(strcmp(*argv,"--Dmesh")==0) im3DHost.Dmesh=read_float(argv[1]);
    else if(strcmp(*argv,"--zoom")==0) read_float3(im3DHost.Dzoom, argv[1]);
    else if(strcmp(*argv,"--step")==0) read_float3(im3DHost.step, argv[1]);
    else if(strcmp(*argv,"--bkgr_col")==0) read_float3(im3DHost.bkgr_col, argv[1]);
    else if(strcmp(*argv,"--mesh_col")==0) read_float3(im3DHost.mesh_col, argv[1]);
    else if(strcmp(*argv,"--box_col")==0) read_float3(im3DHost.box_col, argv[1]);
    else if(strcmp(*argv,"--drop_dir")==0) strcpy(im3DHost.drop_dir,argv[1]);
    else if(strcmp(*argv,"--swap_dir")==0) parsHost.swap_dir=new std::string(argv[1]);
    else if(strcmp(*argv,"--sensor")==0) { float v[3]; read_float3(v, argv[1]); add_sensor(v[0], v[1], v[2]); }
    else { printf("Illegal parameters' syntax notation\n"); return print_help(); }
    //else if(strcmp(*argv,"--")==0) read_float3(im3DHost., argv[1]);
    printf("par: %s; vals: %s\n", argv[0], argv[1]);
    argv +=2; argc -=2;
  };
  im2D.get_device(2,0);
  if(test_only) printf("No GL\n");
  else printf("With GL\n");
try {
  if(type_diag_flag>=1) printf("Настройка опций визуализации по умолчанию\n");
  //imHost.reset();
  cudaTimer tm; tm.start();
  //if(GridNy>50) Tsteps=Ntime*10;
  parsHost.set();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  copy2dev( parsHost, pars );
  copy2dev( shotpoint, src );
  shotpoint.check();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  init();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  init_index();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  init_material(parsHost.index_arr);
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  set_texture(parsHost.index_arr);
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  copy2dev( parsHost, pars );
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );

  if(test_only) {
    for(int i=0; i<Tsteps/Ntime; i++) {
//    while(true) {
      tm.start();
      calcStep();
//      double tCpu=tm.stop();
//      printf("run time: %.2f msec, %.2f Gcells/sec\n", tCpu, 1.e-6*Ntime*Nx*Ny*Nz/tCpu);
//return 0;
    }
    return 0;
  }

  tm.start();
  parsHost.reset_im();
  im3DHost.reset(parsHost.arr4im);
  copy2dev( parsHost, pars );
  CHECK_ERROR( cudaMemset(parsHost.arr4im.Arr3Dbuf,0,((long long int)Nx)*Ny*Nz*sizeof(float)) );
  mxw_draw<<<dim3((USE_UVM==2)?Np:Ns,Na),NT>>>(parsHost.arr4im.Arr3Dbuf);
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  im2D.get_device(2,0);
  im2D.init_image(argc,argv, im3DHost.bNx, im3DHost.bNy, "im3D");
  im3DHost.init3D(parsHost.arr4im); im3DHost.iz0=Nx-1; im3DHost.key_func('b',0,0);

  if(type_diag_flag>=1) printf("Настройка GLUT и запуск интерфейса\n");
  glutIdleFunc(idle_func);
  glutKeyboardFunc(key_func);
  glutMouseFunc(mouse_func);
  glutMotionFunc(motion_func);
  glutDisplayFunc(draw_func);
  if(type_diag_flag>=0) printf("Init cuda device: %.1f msec\n", tm.stop());
  glutMainLoop();
} catch(...) {
  printf("Возникла какая-то ошибка.\n");
}
  parsHost.clear();
  return -1;
}
int main(int argc, char** argv) {
  return _main(argc,argv);
}

float get_val_from_arr3D(int ix, int iy, int iz) {
  Arr3D_pars& arr=parsHost.arr4im;
  if(arr.inCPUmem) return arr.Arr3Dbuf[arr.get_ind(ix,iy,iz)];
  float res=0.0;
  if(arr.inGPUmem) exit_if_ERR(cudaMemcpy(&res, arr.get_ptr(ix,iy,iz), sizeof(float), cudaMemcpyDeviceToHost));
  return res;
}
Arr3D_pars& set_lim_from_arr3D() {
  Arr3D_pars& arr=parsHost.arr4im;
  if(arr.inCPUmem) arr.reset_min_max();
  if(arr.inGPUmem) {
    float* fLims=0,* fLimsD=0;
    exit_if_ERR(cudaMalloc((void**) &fLimsD, 2*Ny*sizeof(float)));
    calc_limits<<<Ny,Nz>>>(arr.Arr3Dbuf, fLimsD);
    fLims=new float[2*Ny];
    exit_if_ERR(cudaMemcpy(fLims, fLimsD, 2*Ny*sizeof(float), cudaMemcpyDeviceToHost));
    exit_if_ERR(cudaFree(fLimsD));
    arr.fMin = fLims[0]; arr.fMax = fLims[1];
    for(int i=0; i<Ny; i++) {
      if(fLims[2*i  ]<arr.fMin) arr.fMin = fLims[2*i  ];
      if(fLims[2*i+1]>arr.fMax) arr.fMax = fLims[2*i+1];
    }
    delete fLims;
  }
  return arr;
}
