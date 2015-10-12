#ifndef _PARAMS_H
#define _PARAMS_H
#include <stdio.h>
#include "cuda_math.h"
#include <assert.h>
#include <vector>
#include <string>
#include <limits.h>
#include <cfloat>
#define STATIC_ASSERT( x ) typedef char __STATIC_ASSERT__[( x )?1:-1]

#define USE_UVM 2

#ifndef USE_UVM
#define USE_UVM 0
#endif

#ifdef USE_DOUBLE
typedef double ftype;
#define MPI_FTYPE MPI_DOUBLE
#define FTYPESIZE 8
typedef double2 ftype2;
typedef double3 ftype3;
typedef double4 ftype4;
template<typename T1,typename T2> __host__ __device__ ftype2 make_ftype2(const T1& f1, const T2& f2) { return make_double2(f1,f2); }
#else//use float
typedef float ftype;
#define MPI_FTYPE MPI_FLOAT
#define FTYPESIZE 4
typedef float2 ftype2;
typedef float3 ftype3;
typedef float4 ftype4;
template<typename T1,typename T2> __host__ __device__ ftype2 make_ftype2(const T1& f1, const T2& f2) { return make_float2(f1,f2); }
#endif

#include "py_consts.h"

#if defined MPI_ON && defined TEST_RATE
#error MPI_ON and TEST_RATE are defined simultaneously
#endif

#ifndef NA
#define NA (gridNz/3)
#endif
#ifndef NV
#define NV (gridNy)
#endif
#ifndef NTIME
#define NTIME (NS-3)
#endif
#ifndef DYSH
#define DYSH
#endif

const int Ns=NS;
const int Na=NA;
const int Nv=NV;

const int Ntime=NTIME;

#if NDev==1
#define Nstrp0 (NA)
#define STRIPES {Nstrp0,}
#elif NDev==2
#define Nstrp0 (NA/2 DYSH)
#define STRIPES {Nstrp0,NA-Nstrp0}
#else
//#define Nstrp0 (NA/3 )
#define STRIPES {NA/3,NA/3,NA-NA/3-NA/3}
#endif

//static_assert(NX>=WX, "Error: NX<=Window size");  // c++11
#if defined USE_AIVLIB_MODEL && defined COFFS_DEFAULT
#error AIVLIB_MODEL and COFFS_DEFAULT are defined simultaneously
#endif

#ifdef MPI_ON
#include <mpi.h>
#endif

#define NStripe(i) (devNStripe[i])
const int NStripe[NDev] = STRIPES;
extern __constant__ int devNStripe[NDev];
const int Nx=((USE_UVM==2)?Np:Ns)*NDT+2;
const int Ny=Na*NDT+1;
//const int Ny=Nstrp0*NDT+1;
const int Nz=Nv;
const int NT=Nz;
const int Nover=0;
extern int* mapNodeSize;

const int KNpmlz=2*Npmlz;//128;
const int KNpmly=2*Npmly*NDT;//128;
const int KNpmlx=2*Npmlx*NDT;//128;

#define SquareGrid 0
const ftype dr=5*5./512.;
//const ftype dx=dr, dy=dr, dz=dr, dt=1*5./512.;
const ftype dx=ds, dy=da, dz=dv;
const ftype dtdx=dt/dx, dtdy=dt/dy, dtdz=dt/dz;
//STATIC_ASSERT((dx==dy) && (SquareGrid==1));
//STATIC_ASSERT((dx==dz) && (SquareGrid==1));

#define DEBUG_PRINT(debug) ;//printf debug;
#define DEBUG_MPI(debug)   ;//printf debug;
#define CHECK_ERROR(err) CheckError( err, __FILE__,__LINE__)
static void CheckError( cudaError_t err, const char *file, int line) {
  if(err!=cudaSuccess){
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

class cuTimer {
  cudaEvent_t tstart,tend;
  cudaStream_t st;
  float diftime;
  public:
  cuTimer(const cudaStream_t& stream=0): diftime(0) {
    CHECK_ERROR( cudaEventCreate(&tstart) ); 
    CHECK_ERROR( cudaEventCreate(&tend  ) );
    CHECK_ERROR( cudaEventRecord(tstart,stream) ); st=stream;
  }
  ~cuTimer(){
    CHECK_ERROR( cudaEventDestroy(tstart) );
    CHECK_ERROR( cudaEventDestroy(tend) );
  }
  float gettime(){
    CHECK_ERROR( cudaEventRecord(tend,st) );
    CHECK_ERROR( cudaEventSynchronize(tend) );
    CHECK_ERROR( cudaEventElapsedTime(&diftime, tstart,tend) ); 
    return diftime;
  }
};

#include "im2D.h"
#include "im3D.hpp"
extern image2D im2D;
extern float calcTime, calcPerf; extern int TimeStep;
extern bool recalc_at_once;
extern char* FuncStr[];

struct DmdArraySe{
  ftype  fld[26][Nz];
  ftype2 fldPML[9][Npmlz];
  ftype2 fldPML1[4][Npmlz]; ftype fldPML2[4][Npmlz];
//  static const int2 SelfShift = {0,0};
//  static const int3 DataShift = DATAShifts;;
};
struct DmdArraySo{
  ftype  fld[28][Nz];
  ftype2 fldPML[9][Npmlz];
  ftype2 fldPML1[5][Npmlz]; ftype fldPML2[5][Npmlz];
//  static const int2 SelfShift = {-NDT,NDT};
//  static const int3 DataShift = DATAShifts;;
};
struct DmdArrayVe{
  ftype  fld[13][Nz]; ftype2 fldPML[13][Npmlz];
//  static const int2 SelfShift = {0,NDT};
//  static const int3 DataShift = DATAShifts;;
};
struct DmdArrayVo{
  ftype  fld[14][Nz]; ftype2 fldPML[14][Npmlz];
//  static const int2 SelfShift = {-NDT,0};
//  static const int3 DataShift = DATAShifts;;
};
struct RagArrayQuat{
  DmdArraySe dmdSe; 
  DmdArraySo dmdSo;
  DmdArrayVo dmdVo;
  DmdArrayVe dmdVe;
};
struct RagArray{
  ftype  fld[81][Nz];
  ftype2 fldPML[45][Npmlz];
  ftype2 fldPML1[9][Npmlz]; ftype fldPML2[9][Npmlz];
};
struct TwoDomS {
    struct {
      ftype  one[Nz];
      ftype2 two[Nz];
    } trifld;
  ftype fldPML[3][Npmlz];
};
struct TwoDomV {
    //ftype fld[3][Nz];
    struct {
      ftype  one[Nz];
      ftype2 two[Nz];
    } trifld;
  ftype fldPML[3][Npmlz];
};
struct DiamondRag{
  TwoDomS Si[NDT*NDT];
  TwoDomV Vi[NDT*NDT];
  inline static void copyM(const int idev, int ixrag, cudaStream_t& stream);
  inline static void copyP(const int idev, int ixrag, cudaStream_t& stream);
  inline static void SendMPIm(const int node, int ixrag);
  inline static void SendMPIp(const int node, int ixrag);
  template<const int Did> inline static void copyDiamond(DiamondRag* dstRag, DiamondRag* srcRag, cudaStream_t& stream) {
    if(Did==0) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Si[0        ].trifld, &srcRag->Si[0        ].trifld, sizeof(TwoDomS)*(NDT*NDT/2+1), cudaMemcpyDeviceToDevice, stream) );
    if(Did==1) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Si[NDT*NDT/2].trifld, &srcRag->Si[NDT*NDT/2].trifld, sizeof(TwoDomS)*(NDT*NDT/2+1), cudaMemcpyDeviceToDevice, stream) );
    if(Did==2) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Vi[0        ].trifld, &srcRag->Vi[0        ].trifld, sizeof(TwoDomV)*(NDT*NDT/2+1), cudaMemcpyDeviceToDevice, stream) );
    if(Did==3) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Vi[NDT*NDT/2].trifld, &srcRag->Vi[NDT*NDT/2].trifld, sizeof(TwoDomV)*(NDT*NDT/2+1), cudaMemcpyDeviceToDevice, stream) );
/*    if(Did==0) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Si[0        ].fld[0], &srcRag->Si[0        ].fld[0], sizeof(TwoDomS)*(NDT*NDT/2  )+sizeof(ftype)*4*Nz, cudaMemcpyDeviceToDevice, stream) );
    if(Did==1) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Si[NDT*NDT/2].fld[4], &srcRag->Si[NDT*NDT/2].fld[4], sizeof(TwoDomS)*(NDT*NDT/2+1)-sizeof(ftype)*4*Nz, cudaMemcpyDeviceToDevice, stream) );
    if(Did==2) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Vi[0        ].fld[0], &srcRag->Vi[0        ].fld[0], sizeof(TwoDomV)*(NDT*NDT/2  )+sizeof(ftype)*1*Nz, cudaMemcpyDeviceToDevice, stream) );
    if(Did==3) CHECK_ERROR( cudaMemcpyAsync(&dstRag->Vi[NDT*NDT/2].fld[1], &srcRag->Vi[NDT*NDT/2].fld[1], sizeof(TwoDomV)*(NDT*NDT/2+1)-sizeof(ftype)*1*Nz, cudaMemcpyDeviceToDevice, stream) );
*/  }
  template<const int Did> inline static void MPIsendDiamond(DiamondRag* dstRag, DiamondRag* srcRag, const int dstnode, const int srcnode, DiamondRag* send_buf, DiamondRag* recv_buf) {
    #ifdef MPI_ON
      #ifdef GPUDIRECT_RDMA
      #define MPI_GPUDIRECT(command,buf,size,mpitype,rank,tag) MPI_##command(buf,size,mpitype,rank,tag,MPI_COMM_WORLD,&req_##command);
      #else
      #define MPI_GPUDIRECT_Isend(buf,size,mpitype,rank,tag) { \
        cudaMemcpy(send_buf,buf,size*sizeof(ftype),cudaMemcpyDeviceToHost);\
        MPI_Isend(send_buf,size,mpitype,rank,tag,MPI_COMM_WORLD,&req_Isend);}
      #define MPI_GPUDIRECT_Irecv(buf,size,mpitype,rank,tag) { \
        MPI_Irecv(recv_buf,size,mpitype,rank,tag,MPI_COMM_WORLD,&req_Irecv);\
        MPI_Wait(&req_Irecv,&stat);\
        cudaMemcpy(buf,recv_buf,size*sizeof(ftype),cudaMemcpyHostToDevice);}
      #define MPI_GPUDIRECT(command,buf,size,mpitype,rank,tag) { MPI_GPUDIRECT_##command(buf,size,mpitype,rank,tag); }
      #endif
    //printf("MPIsendrecv src=%d dst=%d\n",srcnode,dstnode);
    MPI_Status stat; MPI_Request req_Isend,req_Irecv; int doSnd=0,doRcv=0;
    if(dstnode<srcnode && srcnode%NasyncNodes>0 || dstnode>srcnode && srcnode%NasyncNodes<NasyncNodes-1) {
      if(Did==0) MPI_GPUDIRECT(Isend, &srcRag->Si[0        ].trifld, sizeof(TwoDomS)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, dstnode, 0 );
      if(Did==1) MPI_GPUDIRECT(Isend, &srcRag->Si[NDT*NDT/2].trifld, sizeof(TwoDomS)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, dstnode, 0 );
      if(Did==2) MPI_GPUDIRECT(Isend, &srcRag->Vi[0        ].trifld, sizeof(TwoDomV)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, dstnode, 0 );
      if(Did==3) MPI_GPUDIRECT(Isend, &srcRag->Vi[NDT*NDT/2].trifld, sizeof(TwoDomV)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, dstnode, 0 );
      doSnd=1;
    }
    const int fromnode = srcnode+srcnode-dstnode;
    if(fromnode<srcnode && srcnode%NasyncNodes>0 || fromnode>srcnode && srcnode%NasyncNodes<NasyncNodes-1) {
      if(Did==0) MPI_GPUDIRECT(Irecv, &dstRag->Si[0        ].trifld, sizeof(TwoDomS)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, fromnode, 0 );
      if(Did==1) MPI_GPUDIRECT(Irecv, &dstRag->Si[NDT*NDT/2].trifld, sizeof(TwoDomS)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, fromnode, 0 );
      if(Did==2) MPI_GPUDIRECT(Irecv, &dstRag->Vi[0        ].trifld, sizeof(TwoDomV)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, fromnode, 0 );
      if(Did==3) MPI_GPUDIRECT(Irecv, &dstRag->Vi[NDT*NDT/2].trifld, sizeof(TwoDomV)*(NDT*NDT/2+1)/sizeof(ftype), MPI_FTYPE, fromnode, 0 );
      doRcv=1;
    }
    if(doSnd) MPI_Wait(&req_Isend,&stat);
    //if(doRcv) MPI_Wait(&req_Irecv,&stat);
    #endif
  }
};
struct TwoDomSpml { ftype3 fld[6][Nz]; };
struct TwoDomVpml { ftype3 fld[3][Nz]; };
struct DiamondRagPML{
  TwoDomSpml Si[NDT*NDT];
  TwoDomVpml Vi[NDT*NDT];
};

//const int tfsfH=1000*Ny;
//const ftype xsrc=Nx/2*dx, ysrc=40*dy, zsrc=Nz/2*dz;
const ftype Rparticle=0.500;
//const ftype tfsfSm=(Ns/2*NDT-3)*dx-Rparticle, tfsfSp=(Ns/2*NDT+3)*dx+Rparticle; 
//const ftype tfsfAm=(Npmly/2*NDT+3)*dy, tfsfAp=(Na-Npmly/2-3)*NDT*dy;
//const ftype tfsfVm=(Npmlz/2+3)*dz, tfsfVp=(Nz-Npmlz/2-3)*dz; 

const ftype Xlength = Ns*NDT*dx;
const ftype Ylength = Na*NDT*dy;
const ftype thinBIG = 2.200;//0.5*Xlength;
const ftype thinGold = 0.04;
const ftype xBIG  = Xlength - 0.5*thinBIG-0.8;
const ftype xGold = xBIG - 0.5*thinBIG - 0.5*thinGold;

//const ftype tfsfSm=Xlength/3.0, tfsfSp=Xlength/3.0*2; 
//const ftype tfsfAm=Ylength/3.0, tfsfAp=Ylength/3.0*2; 
//const ftype tfsfVm=(dz==FLT_MAX)?0:((Npmlz/2+2)*dz), tfsfVp=(dz==FLT_MAX)?FLT_MAX:(Nv*dz-(Npmlz/2+2)*dz);
/*const ftype tfsfSm=Ns/2*NDT*dx-Rparticle-20*dx, tfsfSp=Ns/2*NDT*dx+Rparticle+20*dx; 
const ftype tfsfAm=-10, tfsfAp=Na*NDT*dy+10; 
const ftype tfsfVm=-10, tfsfVp=Nv*dz+10; */

//const int Xdisp0 = Npmlx/2+10;
//const int Xdisp1 = Ns-Npmlx/2-10;
//const int Xdisp0 = 0;  //tfsfSm/dx/NDT-2;
//const int Xdisp1 = Ns;//tfsfSp/dx/NDT+2;//(xGold+0.5*thinGold)/dx/NDT+2;
namespace dispReg{
  const int sL=0;
  const int sR=0;//Ns;
  const int vL=0;
  const int vR=0;//Nv;
};
const int dispNz=dispReg::vR-dispReg::vL;

#define MD 2
struct TwoDomJ {
    ftype fld[3][dispNz][MD];
  //ftype fldPML[3][Npmlz][MD];
};
struct TwoDomE {
    ftype fld[3][dispNz];
  //ftype fldPML[3][Npmlz];
};
struct DiamondRagDisp{
  TwoDomE Em[NDT*NDT];
//  TwoDomE Ep[NDT*NDT];
  TwoDomJ Jm[NDT*NDT];
  TwoDomJ Jp[NDT*NDT];
};

struct Sensor;
#include "drop.cu"
#include "texmodel.cuh"
struct GeoParams{
  unsigned int nFunc, bgMat;
  int iStep; int wleft; int GPUx0;

  float Rz;
  float Ycnt,Zcnt;
  
  char* index_arr;
  int IndNx,IndNy,IndNz;
  
  float inc_ang;
  int DrawOnlyRag;

  ModelRag* ragsInd[NDev];
  DiamondRag* rags[NDev];
  DiamondRagPML* ragsPMLa; 
  DiamondRagPML* ragsPMLsL[NDev]; 
  DiamondRagPML* ragsPMLsR[NDev]; 
  DiamondRagDisp* ragsDisp[NDev];
  ModelRag* dataInd;
  DiamondRag* data; 
  DiamondRagDisp* dataDisp;
  DiamondRagPML* dataPMLa, *dataPMLs, *dataPMLsL, *dataPMLsR;
  DiamondRag* rdma_send_buf;
  DiamondRag* rdma_recv_buf;
  ModelTexs texs;
  SeismoDrops drop;
  
  int node,subnode;

  std::vector<Sensor>* sensors;
// Think about members!!!  WTF??

  __device__ DiamondRag& get_plaster(const int ix, const int iy) { 
    #ifdef USE_WINDOW
    return data[ix*Na+iy];
    #else
  // only for two devices;
    int idev = (iy<NStripe(0))?0:1; int iym= (idev==0)?0:NStripe(0);
    return rags[idev][ix*NStripe(idev)+iy-iym];
    #endif
  }
  __device__ DiamondRagDisp& get_disp(const int ix, const int iy) { 
    return dataDisp[ix*Na+iy];
  }
  __device__ ModelRag& get_index(const int ix, const int iy) { 
    return dataInd[ix*Na+iy];
  }

  bool vb;

};
const int MaxBlk=15;
extern struct GeoParamsHost: public GeoParams {
  Arr3D_pars arr4im;
  unsigned int MaxFunc;
  static const int Nstreams=10+Na/MaxBlk;
  bool isTFSF;
  std::string* dir;
  void set();
  void reset_im() {
    arr4im.reset(Nz,Ny,Nx);
    arr4im.BufSize = sizeof(float)*Nz*Ny*Nx;
    CHECK_ERROR( cudaMalloc((void**) (&arr4im.Arr3Dbuf), arr4im.BufSize) );
    CHECK_ERROR( cudaMemset(arr4im.Arr3Dbuf, 0, arr4im.BufSize) );
    arr4im.inGPUmem = true;
    //memcpy(&im3DHost, &arr4im, sizeof(im3DHost));
  }
  void clear() {
    cudaFree (rags);
    cudaFree (arr4im.Arr3Dbuf);
    arr4im.clear();
//    CHECK_ERROR(cudaFreeArray(eps_texArray));
  }
  void checkSizes() { }
} parsHost;
extern __constant__ GeoParams pars;

inline void DiamondRag::copyM(const int idev, int ixrag, cudaStream_t& stream){ //diamonds 0 and 3
  copyDiamond<0>( &parsHost.rags[idev-1][(ixrag%Ns+1)*NStripe[idev-1]-1], &parsHost.rags[idev  ][ ixrag%Ns   *NStripe[idev  ]  ], stream );
  copyDiamond<3>( &parsHost.rags[idev-1][(ixrag%Ns+1)*NStripe[idev-1]-1], &parsHost.rags[idev  ][ ixrag%Ns   *NStripe[idev  ]  ], stream );
}
inline void DiamondRag::copyP(const int idev, int ixrag, cudaStream_t& stream){ //diamonds 1 and 2
  copyDiamond<1>( &parsHost.rags[idev+1][ ixrag%Ns   *NStripe[idev+1]  ], &parsHost.rags[idev  ][(ixrag%Ns+1)*NStripe[idev  ]-1], stream );
  ixrag++;
  copyDiamond<2>( &parsHost.rags[idev+1][ ixrag%Ns   *NStripe[idev+1]  ], &parsHost.rags[idev  ][(ixrag%Ns+1)*NStripe[idev  ]-1], stream );
}
inline void DiamondRag::SendMPIm(const int mpirank, int ixrag){ //diamonds 0 and 3
  MPIsendDiamond<0>( &parsHost.rags[NDev-1][(ixrag%Ns+1)*NStripe[NDev-1]-1], &parsHost.rags[0][ ixrag%Ns   *NStripe[0]  ], mpirank-1, mpirank, parsHost.rdma_send_buf, parsHost.rdma_recv_buf);
  MPIsendDiamond<3>( &parsHost.rags[NDev-1][(ixrag%Ns+1)*NStripe[NDev-1]-1], &parsHost.rags[0][ ixrag%Ns   *NStripe[0]  ], mpirank-1, mpirank, parsHost.rdma_send_buf, parsHost.rdma_recv_buf);
}
inline void DiamondRag::SendMPIp(const int mpirank, int ixrag){ //diamonds 1 and 2
  MPIsendDiamond<1>( &parsHost.rags[0][ ixrag%Ns   *NStripe[0]  ], &parsHost.rags[NDev-1][(ixrag%Ns+1)*NStripe[NDev-1]-1], mpirank+1, mpirank, parsHost.rdma_send_buf, parsHost.rdma_recv_buf );
  ixrag++;
  MPIsendDiamond<2>( &parsHost.rags[0][ ixrag%Ns   *NStripe[0]  ], &parsHost.rags[NDev-1][(ixrag%Ns+1)*NStripe[NDev-1]-1], mpirank+1, mpirank, parsHost.rdma_send_buf, parsHost.rdma_recv_buf );
}
/*  CHECK_ERROR( cudaMemcpyAsync(&parsHost.rags[idev-1][(ixrag%Ns+1)*NStripe[idev-1]-1].Si[0].fld[0], 
                               &parsHost.rags[idev  ][ ixrag%Ns   *NStripe[idev  ]  ].Si[0].fld[0],
                               sizeof(TwoDomS)*(NDT*NDT/2  )+sizeof(ftype)*4*Nz, cudaMemcpyDeviceToDevice, stream) );
  CHECK_ERROR( cudaMemcpyAsync(&parsHost.rags[idev-1][(ixrag%Ns+1)*NStripe[idev-1]-1].Vi[NDT*NDT/2].fld[1], 
                               &parsHost.rags[idev  ][ ixrag%Ns   *NStripe[idev  ]  ].Vi[NDT*NDT/2].fld[1],
                               sizeof(TwoDomV)*(NDT*NDT/2+1)-sizeof(ftype)*1*Nz, cudaMemcpyDeviceToDevice, stream) );
  CHECK_ERROR( cudaMemcpyAsync(&parsHost.rags[idev+1][ ixrag%Ns   *NStripe[idev+1]  ].Si[NDT*NDT/2].fld[4], 
                               &parsHost.rags[idev  ][(ixrag%Ns+1)*NStripe[idev  ]-1].Si[NDT*NDT/2].fld[4],
                               sizeof(TwoDomS)*(NDT*NDT/2+1)-sizeof(ftype)*4*Nz, cudaMemcpyDeviceToDevice, stDo[idev]) );
                               ixrag++;
  CHECK_ERROR( cudaMemcpyAsync(&parsHost.rags[idev+1][ ixrag%Ns   *NStripe[idev+1]  ].Vi[0].fld[0], 
                               &parsHost.rags[idev  ][(ixrag%Ns+1)*NStripe[idev  ]-1].Vi[0].fld[0],
                               sizeof(TwoDomV)*(NDT*NDT/2  )+sizeof(ftype)*1*Nz, cudaMemcpyDeviceToDevice, stDo[idev]) );
*/

inline void print_info(){
  printf("Devices: %d\n", NDev);
  printf("NasyncNodes: %d\n", NasyncNodes);
  #ifdef GPUDIRECT_RDMA
  printf("GPUDirect RDMA +\n");
  #else
  printf("GPUDirect RDMA -\n");
  #endif
  #ifdef USE_AIVLIB_MODEL
  printf("USE_AIVLIB +\n");
  #else
  printf("USE_AIVLIB -\n");
  #endif
  #ifdef MPI_ON
  printf("MPI_ON +\n");
  #else
  printf("MPI_OFF\n");
  #endif
  #ifdef MPI_TEST
  printf("MPI_TEST +\n");
  #else
  printf("MPI_TEST -\n");
  #endif
  #ifdef TEST_RATE
  printf("TEST_RATE: %d\n", TEST_RATE);
  if(NDev!=1 || NasyncNodes!=1) { printf("Error: Test_Rate works only for non-mpi NDev=1 and NasyncNodes=1\n"); exit(-1);}
  #ifdef MPI_ON
  printf("Error: Test_Rate works only for non-mpi\n"); exit(-1);
  #endif
  #else
  printf("TEST_RATE -\n");
  #endif
  #ifdef USE_WINDOW
  printf("USE_WINDOW +\n");
  #else
  printf("USE_WINDOW -\n");
  #endif
  #ifdef COFFS_DEFAULT
  printf("COFFS_DEFAULT +\n");
  #else
  printf("COFFS_DEFAULT -\n");
  #endif
  #ifdef USE_TEX_2D
  printf("USE_TEX_2D +\n");
  #else
  printf("USE_TEX_2D -\n");
  #endif
}

#include "sensor.h"

#include "dispersion.cpp"
#include "defs.h"

int _main(int argc, char** argv);
extern int Tsteps;

#pragma diag_suppress declared_but_not_referenced
#pragma diag_suppress set_but_not_used
#endif //_PARAMS_H
