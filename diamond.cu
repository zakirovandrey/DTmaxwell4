#include <stdio.h>
#include <errno.h>
#include <omp.h>
#include <semaphore.h>
#ifdef MPI_ON
#include <mpi.h>
#endif
#include "chooseV.h"
#include "signal.h"

int* mapNodeSize;
//=============================================
ftype* __restrict__ hostKpmlx1; ftype* __restrict__ hostKpmlx2;
ftype* __restrict__ hostKpmly1; ftype* __restrict__ hostKpmly2;
ftype* __restrict__ hostKpmlz1; ftype* __restrict__ hostKpmlz2;
GeoParamsHost parsHost;
__constant__ GeoParams pars;
__constant__ int devNStripe[NDev] = STRIPES;
__constant__ ftype Kpmlx1[(KNpmlx==0)?1:KNpmlx];
__constant__ ftype Kpmlx2[(KNpmlx==0)?1:KNpmlx];
__constant__ ftype Kpmly1[(KNpmly==0)?1:KNpmly];
__constant__ ftype Kpmly2[(KNpmly==0)?1:KNpmly];
__constant__ ftype Kpmlz1[(KNpmlz==0)?1:KNpmlz];
__constant__ ftype Kpmlz2[(KNpmlz==0)?1:KNpmlz];
//__shared__ ftype2 shared_fld[2][7][Nz];
//__shared__ ftype2 shared_fld[(FTYPESIZE*Nv*28>0xc000)?7:14][Nv];
__shared__ ftype2 shared_fld[SHARED_SIZE][Nv];

#include "window.hpp"
struct AsyncMPIexch{
  int even,ix,t0,Nt,mpirank; bool do_run;
  sem_t sem_mpi, sem_calc;
  void exch(const int _even, const int _ix, const int _t0, const int _Nt, const int _mpirank) {
    even=_even; ix=_ix; t0=_t0; Nt=_Nt, mpirank=_mpirank;
    if(sem_post(&sem_mpi)<0) printf("exch sem_post error %d\n",errno);
  }
  void exch_sync(){ if(sem_wait(&sem_calc)<0) printf("exch_sync sem error %d\n",errno); }
  void run() {
    if(sem_wait(&sem_mpi)<0) printf("run sem_wait error %d\n",errno);
    if(do_run==0) return;
    if(even==0) for(int ixrag=ix; ixrag<ix+Nt-t0; ixrag++) DiamondRag::SendMPIp(mpirank, ixrag);
    if(even==1) for(int ixrag=ix; ixrag<ix+Nt-t0; ixrag++) DiamondRag::SendMPIm(mpirank, ixrag);
    if(sem_post(&sem_calc)<0) printf("run sem_post error %d\n",errno);;
  }
} ampi_exch;
#define IFPMLS(func,a,b,c,d,args) {/*printf(#func" idev=%d ix=%d iym=%d Nblocks=%d\n", idev,ix, iym, a);*/if(isPMLs) PMLS##func<<<a,b,c,d>>>args; else func<<<a,b,c,d>>>args; }
//#define IFPMLS(func,a,b,c,d,args) { if(!isPMLs) func<<<a,b,c,d>>>args; }
//#define IFPMLS(func,a,b,c,d,args) func<<<a,b,c,d>>>args;
template<int even> inline void Window::Dtorre(int ix, int Nt, int t0, double disbal[NDev], bool isPMLs, bool isTFSF) {
  if(Nt<=t0 || Nt<=0) return;
  DEBUG_PRINT(("Dtorre%d isPMLs=%d isTFSF=%d ix=%d, t0=%d Nt=%d wleft=%d\n", even, isPMLs, isTFSF, ix,t0,Nt, parsHost.wleft));
  const int Nth=Nv; 
  double tt1 = omp_get_wtime();
  CHECK_ERROR( cudaSetDevice(0) );
  cudaStream_t stPMLbot; CHECK_ERROR( cudaStreamCreate(&stPMLbot) );
  cudaStream_t stPMLtop; CHECK_ERROR( cudaStreamCreate(&stPMLtop) );
  cudaStream_t stDm[NDev],stDo[NDev]; for(int i=0;i<NDev;i++) { if(i!=0) CHECK_ERROR( cudaSetDevice(i) ); CHECK_ERROR( cudaStreamCreate(&stDm[i]) ); CHECK_ERROR( cudaStreamCreate(&stDo[i]) ); }
  cudaStream_t stI   ; CHECK_ERROR( cudaStreamCreate(&stI   ) );
  cudaStream_t stX   ; CHECK_ERROR( cudaStreamCreate(&stX   ) );
  cudaStream_t stP   ; if(even==0) { cudaSetDevice(NDev-1); CHECK_ERROR( cudaStreamCreate(&stP   ) ); } else
                       if(even==1) { cudaSetDevice(0     ); CHECK_ERROR( cudaStreamCreate(&stP   ) ); }
  CHECK_ERROR( cudaSetDevice(0) );

  int iym=0, iyp=0; 
  int Nblk=0;   iyp++;
  int Iy=iym, Xy, D1oy[NDev], D0oy[NDev], Dmy[NDev], DmBlk[NDev], Syb,Syt, SybBlk,SytBlk;
  int is_oneL[NDev], is_oneU[NDev], is_many[NDev], is_I[NDev], is_X[NDev], is_Sb[NDev], is_St[NDev], is_P[NDev];
  for(int i=0; i<NDev; i++) { is_oneL[i]=0; is_oneU[i]=0; is_many[i]=0; is_I[i]=0; is_X[i]=0; is_Sb[i]=0; is_St[i]=0; is_P[i]=0; }
  is_I[0]=1;
  iym=iyp; Nblk=0; while(iyp<Npmly/2) { iyp++; Nblk++; } if(Nblk>0) is_Sb[0]=1; Syb=iym; SybBlk=Nblk; 
  for(int idev=0,nextY=0; idev<NDev; idev++) {
    nextY+=NStripe[idev]; if(idev==NDev-1) nextY-=Npmly/2;
    if(idev!=0) {
    // Dtorre1 only
      if(iyp<nextY && even==1) is_oneL[idev]=1;
      D1oy[idev]=iyp; if(iyp<nextY) iyp++;
    }
    iym=iyp; Nblk=0;  while(iyp<nextY-(idev==NDev-1?0:1)) { iyp++; Nblk++; }
    // Main Region
    if(Nblk>0) is_many[idev]=1;
    Dmy[idev]=iym, DmBlk[idev]=Nblk;
    if(idev!=NDev-1) {
    // Dtorre0 only
      if(iyp<nextY && even==0) is_oneU[idev]=1;
      D0oy[idev]=iyp; if(iyp<nextY) iyp++;
    }
  }
  iym=iyp; Nblk=0;  while(iyp<Na-1) { iyp++; Nblk++; }
  if(Nblk>0) is_St[NDev-1]=1;
  is_X[NDev-1]=1;
  Syt=iym; SytBlk=Nblk; Xy=iyp;
  if(subnode!=0) {
    is_I [0]=0; if(even==1) is_P[0]=1;
    is_Sb[0]=0; DmBlk[0]+=SybBlk; Dmy[0]=Syb; 
  }
  if(subnode!=NasyncNodes-1) {
    is_X [NDev-1]=0; if(even==0) is_P[NDev-1]=1; 
    is_St[NDev-1]=0; DmBlk[NDev-1]+=SytBlk;
  }

  for(int idev=0; idev<NDev; idev++) {
    if(idev!=0) CHECK_ERROR( cudaSetDevice(idev) );
    if(is_oneL[idev] && even==1 &&  isTFSF ) IFPMLS(torreTFSF1 ,1          ,Nth,0,stDo[idev],(ix,D1oy[idev],Nt,t0))
    if(is_oneL[idev] && even==1 && !isTFSF ) IFPMLS(torreD1    ,1          ,Nth,0,stDo[idev],(ix,D1oy[idev],Nt,t0))
    if(is_oneU[idev] && even==0 &&  isTFSF ) IFPMLS(torreTFSF0 ,1          ,Nth,0,stDo[idev],(ix,D0oy[idev],Nt,t0))
    if(is_oneU[idev] && even==0 && !isTFSF ) IFPMLS(torreD0    ,1          ,Nth,0,stDo[idev],(ix,D0oy[idev],Nt,t0))
    if(is_I[idev]    && even==0 && Npmly==0) IFPMLS(torreId0   ,1          ,Nth,0,stI       ,(ix,Iy        ,Nt,t0))
    if(is_I[idev]    && even==0 && Npmly!=0) IFPMLS(torreIs0   ,1          ,Nth,0,stI       ,(ix,Iy        ,Nt,t0))
    if(is_I[idev]    && even==1 && Npmly==0) IFPMLS(torreId1   ,1          ,Nth,0,stI       ,(ix,Iy        ,Nt,t0))
    if(is_I[idev]    && even==1 && Npmly!=0) IFPMLS(torreIs1   ,1          ,Nth,0,stI       ,(ix,Iy        ,Nt,t0))
    if(is_X[idev]    && even==0 && Npmly==0) IFPMLS(torreXd0   ,1          ,Nth,0,stX       ,(ix,Xy        ,Nt,t0))
    if(is_X[idev]    && even==0 && Npmly!=0) IFPMLS(torreXs0   ,1          ,Nth,0,stX       ,(ix,Xy        ,Nt,t0))
    if(is_X[idev]    && even==1 && Npmly==0) IFPMLS(torreXd1   ,1          ,Nth,0,stX       ,(ix,Xy        ,Nt,t0))
    if(is_X[idev]    && even==1 && Npmly!=0) IFPMLS(torreXs1   ,1          ,Nth,0,stX       ,(ix,Xy        ,Nt,t0))
    if(is_P[idev]    && even==0            ) IFPMLS(torreD0    ,1          ,Nth,0,stP       ,(ix,Xy        ,Nt,t0))
    if(is_P[idev]    && even==1            ) IFPMLS(torreD1    ,1          ,Nth,0,stP       ,(ix,Iy        ,Nt,t0))
    if(is_Sb[idev]   && even==0            ) IFPMLS(torreS0    ,SybBlk     ,Nth,0,stPMLbot  ,(ix,Syb       ,Nt,t0))
    if(is_Sb[idev]   && even==1            ) IFPMLS(torreS1    ,SybBlk     ,Nth,0,stPMLbot  ,(ix,Syb       ,Nt,t0))
    if(is_St[idev]   && even==0            ) IFPMLS(torreS0    ,SytBlk     ,Nth,0,stPMLtop  ,(ix,Syt       ,Nt,t0))
    if(is_St[idev]   && even==1            ) IFPMLS(torreS1    ,SytBlk     ,Nth,0,stPMLtop  ,(ix,Syt       ,Nt,t0))
    if(is_many[idev] && even==0 && isTFSF  ) IFPMLS(torreTFSF0 ,DmBlk[idev],Nth,0,stDm[idev],(ix,Dmy[idev] ,Nt,t0))
    if(is_many[idev] && even==1 && isTFSF  ) IFPMLS(torreTFSF1 ,DmBlk[idev],Nth,0,stDm[idev],(ix,Dmy[idev] ,Nt,t0))
    if(is_many[idev] && even==0 && !isTFSF ) IFPMLS(torreD0    ,DmBlk[idev],Nth,0,stDm[idev],(ix,Dmy[idev] ,Nt,t0))
    if(is_many[idev] && even==1 && !isTFSF ) IFPMLS(torreD1    ,DmBlk[idev],Nth,0,stDm[idev],(ix,Dmy[idev] ,Nt,t0))
    if(is_oneL[idev] && even==1            ) for(int ixrag=ix; ixrag<ix+Nt-t0; ixrag++) DiamondRag::copyM(idev, ixrag, stDo[idev]);
    if(is_oneU[idev] && even==0            ) for(int ixrag=ix; ixrag<ix+Nt-t0; ixrag++) DiamondRag::copyP(idev, ixrag, stDo[idev]);
  }

  /*
  if(even==0                       ) IFPMLS(torreI0   ,1   ,Nth,0,stPMLm   ,(ix,iym,Nt,t0))
  if(even==1                       ) IFPMLS(torreI1   ,1   ,Nth,0,stPMLm   ,(ix,iym,Nt,t0))
  for(int idev=0,nextY=0; idev<NDev; idev++) {
    if(idev!=0) CHECK_ERROR( cudaSetDevice(idev) );
    nextY+=NStripe[idev]; if(idev==NDev-1) nextY-=Npmly;
    if(idev!=0) { iym=iyp;
    if(iyp<nextY && even==1 && isTFSF ) IFPMLS(torreTFSF1,1,Nth,0,stDo[idev],(ix,iym,Nt,t0))
    if(iyp<nextY && even==1 && !isTFSF) IFPMLS(torreD1   ,1,Nth,0,stDo[idev],(ix,iym,Nt,t0))
    if(iyp<nextY && even==1 ) for(int ixrag=ix; ixrag<ix+Nt-t0; ixrag++) DiamondRag::copyM(idev, ixrag, stDo[idev]);
    if(iyp<nextY) iyp++;
    }
    iym=iyp; Nblk=0;  while(iyp<nextY-(idev==NDev-1?0:1)) { iyp++; Nblk++; }
    if(Nblk>0 && even==0 && isTFSF ) IFPMLS(torreTFSF0,Nblk,Nth,0,stDm[idev],(ix,iym,Nt,t0))
    if(Nblk>0 && even==1 && isTFSF ) IFPMLS(torreTFSF1,Nblk,Nth,0,stDm[idev],(ix,iym,Nt,t0))
    if(Nblk>0 && even==0 && !isTFSF) IFPMLS(torreD0   ,Nblk,Nth,0,stDm[idev],(ix,iym,Nt,t0))
    if(Nblk>0 && even==1 && !isTFSF) IFPMLS(torreD1   ,Nblk,Nth,0,stDm[idev],(ix,iym,Nt,t0))
    if(idev!=NDev-1) { iym=iyp;
    if(iyp<nextY && even==0 && isTFSF ) IFPMLS(torreTFSF0,1,Nth,0,stDo[idev],(ix,iym,Nt,t0))
    if(iyp<nextY && even==0 && !isTFSF) IFPMLS(torreD0   ,1,Nth,0,stDo[idev],(ix,iym,Nt,t0))
    if(iyp<nextY && even==0 ) for(int ixrag=ix; ixrag<ix+Nt-t0; ixrag++) DiamondRag::copyP(idev, ixrag, stDo[idev]);
    if(iyp<nextY) iyp++;
    }
  }
  iym=iyp; Nblk=0;  while(iyp<Na-1        ) { iyp++; Nblk++; } 
  if(Nblk>0 && even==0             ) IFPMLS(torreS0   ,Nblk,Nth,0,stPMLp   ,(ix,iym,Nt,t0))
  if(Nblk>0 && even==1             ) IFPMLS(torreS1   ,Nblk,Nth,0,stPMLp   ,(ix,iym,Nt,t0))
  if(even==0                       ) IFPMLS(torreX0   ,1   ,Nth,0,stX      ,(ix,iyp,Nt,t0))
  if(even==1                       ) IFPMLS(torreX1   ,1   ,Nth,0,stX      ,(ix,iyp,Nt,t0))*/
  
  CHECK_ERROR( cudaSetDevice(0) );

  if(!doneMemcopy) {
    if(even==0) MemcopyDtH(ix4copy);
    if(even==1) MemcopyHtD(ix4copy);
    CHECK_ERROR( cudaStreamSynchronize(streamCopy) ); if(even==1) doneMemcopy=true;
  }
  CHECK_ERROR( cudaStreamSynchronize(stP   ) );
  if(NasyncNodes>1) ampi_exch.exch(even, ix, t0, Nt, node*NasyncNodes+subnode);
  CHECK_ERROR( cudaStreamSynchronize(stPMLbot) ); 
  CHECK_ERROR( cudaStreamSynchronize(stPMLtop) );
  CHECK_ERROR( cudaStreamSynchronize(stI   ) );
  CHECK_ERROR( cudaStreamSynchronize(stX   ) );
  for(int i=0;i<NDev;i++) CHECK_ERROR( cudaStreamSynchronize(stDo[i]) );
  for(int i=0;i<NDev;i++) { double tt=omp_get_wtime(); CHECK_ERROR( cudaStreamSynchronize(stDm[i]) ); disbal[i]+=omp_get_wtime()-tt; }
  CHECK_ERROR( cudaStreamDestroy(stPMLbot) );
  CHECK_ERROR( cudaStreamDestroy(stPMLtop) );
  CHECK_ERROR( cudaStreamDestroy(stI   ) ); 
  CHECK_ERROR( cudaStreamDestroy(stX   ) ); 
  CHECK_ERROR( cudaStreamDestroy(stP   ) ); 
  for(int i=0;i<NDev;i++) CHECK_ERROR( cudaStreamDestroy(stDo[i]) );
  for(int i=0;i<NDev;i++) CHECK_ERROR( cudaStreamDestroy(stDm[i]) );
  if(NasyncNodes>1) ampi_exch.exch_sync();
}
inline void Window::Dtorres(int ix, int Nt, int t0, double disbal[NDev], bool isPMLs, bool isTFSF) {
  #ifdef BLOCH_BND_Y
    DtorreBloch<0>(ix,iy,Nt,t0,isPMLs,isTFSF); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    DtorreBloch<1>(ix,iy,Nt,t0,isPMLs,isTFSF); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  #else
  Dtorre<0>(ix,Nt,t0,disbal,isPMLs,isTFSF); //cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  Dtorre<1>(ix,Nt,t0,disbal,isPMLs,isTFSF); //cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  #endif
    /////DtorreBloch<0>(ix,((ny-(parsHost.iStep%NyBloch)+NyBloch)%NyBloch)*Na/NyBloch,Nt,t0,isPMLs,isTFSF); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    /////DtorreBloch<1>(ix,((ny-(parsHost.iStep%NyBloch)+NyBloch)%NyBloch)*Na/NyBloch,Nt,t0,isPMLs,isTFSF); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
}
template<int even> inline void DtorreBloch(int ix, int tiy, int Nt, int t0, bool isPMLs=false, bool isTFSF=false) {
  if(Nt<=t0 || Nt<=0) return;
  //printf("Dtorre%d isPMLs=%d isTFSF=%d ix=%d, t0=%d Nt=%d tiy=%d\n", even, isPMLs, isTFSF, ix,t0,Nt, tiy);
  const int Nth=Nv; 
  cudaStream_t stPMLm; CHECK_ERROR( cudaStreamCreate(&stPMLm) );
  cudaStream_t stPMLp; CHECK_ERROR( cudaStreamCreate(&stPMLp) );
  cudaStream_t stD   ; CHECK_ERROR( cudaStreamCreate(&stD) );

  int iym=0, iyp=0;
  int Nblk=Na/NyBloch;
  if(Nblk>0 && even==0 && isTFSF ) IFPMLS(torreTFSF0,Nblk,Nth,0,stD   ,(ix,tiy,Nt,t0))
  if(Nblk>0 && even==1 && isTFSF ) IFPMLS(torreTFSF1,Nblk,Nth,0,stD   ,(ix,tiy,Nt,t0))
  if(Nblk>0 && even==0 && !isTFSF) IFPMLS(torreD0   ,Nblk,Nth,0,stD   ,(ix,tiy,Nt,t0))
  if(Nblk>0 && even==1 && !isTFSF) IFPMLS(torreD1   ,Nblk,Nth,0,stD   ,(ix,tiy,Nt,t0))

  CHECK_ERROR( cudaStreamSynchronize(stPMLm) );
  CHECK_ERROR( cudaStreamSynchronize(stPMLp) );
  CHECK_ERROR( cudaStreamSynchronize(stD) );
  CHECK_ERROR( cudaStreamDestroy(stPMLm) );
  CHECK_ERROR( cudaStreamDestroy(stPMLp) );
  CHECK_ERROR( cudaStreamDestroy(stD)    );
}

#ifdef MPI_ON
MPI_Request reqSp, reqSm, reqRp, reqRm, reqSp_pml, reqSm_pml, reqRp_pml, reqRm_pml;
MPI_Status status;
int flag;
mpi_message Window::mes[8];
//#define BLOCK_SEND
#define MPI_TEST
//#define MPI_NUDGE
//#define USE_MPI_THREADING

#ifdef BLOCK_SEND
#define SendMPI(p,sz,tp,rnk,tag,world,req) MPI_Send(p,sz,tp,rnk,tag,world);
#define RecvMPI(p,sz,tp,rnk,tag,world,req) MPI_Recv(p,sz,tp,rnk,tag,world,&status);
#define doWait 0
#else
#ifndef USE_MPI_THREADING
#define WaitMPI(nreq,req,st) MPI_Wait(req,st)
#define SendMPI(p,sz,tp,rnk,tag,world,req,nreq) MPI_Isend(p,sz,tp,rnk,tag,world,req);
#define RecvMPI(p,sz,tp,rnk,tag,world,req,nreq) MPI_Irecv(p,sz,tp,rnk,tag,world,req);
#else
#define WaitMPI(nreq,req,st) { mpi_message* mes = &window.mes[nreq]; \
       int s=pthread_join(mes->mpith,0); if(s!=0) printf("node %d: Error joining thread %ld retcode=%d\n",window.node,mes->mpith,s); }
static void* send_func(void* args){
  mpi_message *mes = (mpi_message*)args;
  MPI_Send(mes->buf,mes->count,mes->datatype,mes->dest,mes->tag,mes->comm);
  return 0;
}
static void* recv_func(void* args){
  mpi_message *mes = (mpi_message*)args;
  MPI_Status stat;
  MPI_Recv(mes->buf,mes->count,mes->datatype,mes->dest,mes->tag,mes->comm,&stat);
  return 0;
}
#define SendMPI(p,sz,tp,rnk,tag,world,req,nreq) {mpi_message* mes = &window.mes[nreq]; mes->set(p,sz,tp,rnk,tag,world); \
      if(pthread_create(&mes->mpith,0,send_func,(void*)mes)!=0) {printf("Error: cannot create thread for MPI_send %d node=%d\n",nreq,window.node); MPI_Abort(MPI_COMM_WORLD, 1);};}
#define RecvMPI(p,sz,tp,rnk,tag,world,req,nreq) {mpi_message* mes = &window.mes[nreq]; mes->set(p,sz,tp,rnk,tag,world); \
      if(pthread_create(&mes->mpith,0,recv_func,(void*)mes)!=0) {printf("Error: cannot create thread for MPI_recv %d node=%d\n",nreq,window.node); MPI_Abort(MPI_COMM_WORLD, 1);};}
#endif//USE_MPI_THREADING
#define doWait 1
#endif
#endif// MPI_ON
int calcStep(){
//  CHECK_ERROR( cudaDeviceSetSharedMemConfig ( cudaSharedMemBankSizeEightByte ) );
  if(parsHost.iStep==0) printf("Starting...\n");
  cuTimer t0;
  int torreNum=0;
  CHECK_ERROR(cudaDeviceSynchronize());
  #ifdef TEST_RATE
  for(int ix=Ns-Ntime; ix>0; ix--) {
//    printf("ix=%d\n",ix);
    const int block_spacing = TEST_RATE;
    torreD0<<<(Na-2)/block_spacing,Nv>>>(ix, 1, Ntime, 0); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    torreD1<<<(Na-2)/block_spacing,Nv>>>(ix, 1, Ntime, 0); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    torreNum++;
  }
  #else
  Window window; window.prepare();
  int node_shift=0; for(int inode=0; inode<window.node; inode++) node_shift+= mapNodeSize[inode]; node_shift-= Ns*window.node;
  int nsize=mapNodeSize[window.node]; int nL=node_shift; int nR=nL+nsize;
  #ifdef MPI_ON
  if(parsHost.iStep==0) {
    int wleftP=nR-Ns;
    int wleftM=nL;
    if(window.node!=window.Nprocs-1) {
      DEBUG_MPI(("Recv P (node %d) wleft=%d\n", window.node, wleftP));
      //MPI_Isend(&window.data    [wleftP*Na   ], Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, window.node+1, 2+0, MPI_COMM_WORLD, &reqSp);
      //MPI_Isend(&window.dataPMLa[wleftP*Npmly], Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, window.node+1, 2+1, MPI_COMM_WORLD, &reqSp_pml);
      #ifndef BLOCK_SEND
      int doSR=1;
      #ifdef MPI_TEST
      doSR=0;
      #endif
      RecvMPI(&window.data    [wleftP*Na   ], doSR*Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+0, MPI_COMM_WORLD, &reqRp    , 2);
      RecvMPI(&window.dataPMLa[wleftP*Npmly], doSR*Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+1, MPI_COMM_WORLD, &reqRp_pml, 6);
      #endif
    }
    if(window.node!=0              ) {
      //DEBUG_MPI(("Send&Recv M (node %d) wleft=%d\n", window.node, wleftM));
      //MPI_Isend(&window.data    [wleftM*Na   ], Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, window.node-1, 2-2, MPI_COMM_WORLD, &reqSm);
      //MPI_Isend(&window.dataPMLa[wleftM*Npmly], Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, window.node-1, 2-1, MPI_COMM_WORLD, &reqSm_pml);
      //MPI_Irecv(&window.data    [wleftM*Na   ], Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, window.node-1, 2+0, MPI_COMM_WORLD, &reqRm);
      //MPI_Irecv(&window.dataPMLa[wleftM*Npmly], Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, window.node-1, 2+1, MPI_COMM_WORLD, &reqRm_pml);
    }
  }
  #endif
  while(window.w0+Ns>=0) {
//    window.Memcopy();
    #ifdef MPI_ON
    #ifdef BLOCK_SEND
    if( !(parsHost.wleft>=nR && window.node!=window.Nprocs-1 || parsHost.wleft<nL-Ns && window.node!=0) ) {
      if(parsHost.wleft==nR-1     && window.node!=window.Nprocs-1) { 
        DEBUG_MPI(("bl Recv P(%d) (node %d) wleft=%d tag=%d\n", nR-Ns, window.node, parsHost.wleft, parsHost.iStep+0));
        RecvMPI(&window.data    [(nR-Ns)*Na   ], Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, window.node+1, 2+(parsHost.iStep+0)*2+0, MPI_COMM_WORLD, &reqRp    , 2);
        RecvMPI(&window.dataPMLa[(nR-Ns)*Npmly], Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, window.node+1, 2+(parsHost.iStep+0)*2+1, MPI_COMM_WORLD, &reqRp_pml, 6);
        DEBUG_MPI(("Ok Recv P(%d) (node %d) wleft=%d tag=%d\n", nR-Ns, window.node, parsHost.wleft, parsHost.iStep+0));
      }
      //if(parsHost.wleft==nR-Ns-Ns && window.node!=window.Nprocs-1) {
      if(parsHost.wleft==nL+Ns && window.node!=window.Nprocs-1) {
        DEBUG_MPI(("bl Send P(%d) (node %d) wleft=%d tag=%d\n", nR-Ns, window.node, parsHost.wleft, parsHost.iStep+1));
        SendMPI(&window.data    [(nR-Ns)*Na   ], Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, window.node+1, 2+(parsHost.iStep+1)*2+0, MPI_COMM_WORLD, &reqSp     ,0);
        SendMPI(&window.dataPMLa[(nR-Ns)*Npmly], Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, window.node+1, 2+(parsHost.iStep+1)*2+1, MPI_COMM_WORLD, &reqSp_pml, 4);
        DEBUG_MPI(("Ok Send P(%d) (node %d) wleft=%d tag=%d\n", nR-Ns, window.node, parsHost.wleft, parsHost.iStep+1));
      }
      if(parsHost.wleft==nL+Ns  && window.node!=0               && parsHost.iStep!=0) { 
        DEBUG_MPI(("bl Recv M(%d) (node %d) wleft=%d tag=%d\n", nL, window.node, parsHost.wleft, parsHost.iStep+0));
        RecvMPI(&window.data    [ nL    *Na   ], Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, window.node-1, 2+(parsHost.iStep+0)*2+0, MPI_COMM_WORLD, &reqRm    , 3);
        RecvMPI(&window.dataPMLa[ nL    *Npmly], Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, window.node-1, 2+(parsHost.iStep+0)*2+1, MPI_COMM_WORLD, &reqRm_pml, 7);
        DEBUG_MPI(("Ok Recv M(%d) (node %d) wleft=%d tag=%d\n", nL, window.node, parsHost.wleft, parsHost.iStep-1));
      }
      window.calcDtorres(nL,nR, parsHost.wleft<nL && window.node!=0, parsHost.wleft>=nR-Ns && window.node!=window.Nprocs-1);
      if(parsHost.wleft==nL-Ns  && window.node!=0              ) {
        DEBUG_MPI(("bl Send M(%d) (node %d) wleft=%d tag=%d\n", nL, window.node, parsHost.wleft, parsHost.iStep+0));
        SendMPI(&window.data    [ nL    *Na   ], Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, window.node-1, 2+(parsHost.iStep+0)*2+0, MPI_COMM_WORLD, &reqSm    , 1);
        SendMPI(&window.dataPMLa[ nL    *Npmly], Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, window.node-1, 2+(parsHost.iStep+0)*2+1, MPI_COMM_WORLD, &reqSm_pml, 5);
        DEBUG_MPI(("Ok Send M(%d) (node %d) wleft=%d tag=%d\n", nL, window.node, parsHost.wleft, parsHost.iStep+0));
      }
    }
    #else//BLOCK_SEND not def
    if( true /*!(parsHost.wleft>=nR && window.node!=window.Nprocs-1 || parsHost.wleft<nL-Ns && window.node!=0)*/ ) {
      bool doSend[2] = {1,1}; bool doRecv[2] = {1,1};
      #ifdef MPI_TEST
      if(parsHost.iStep  -window.node<0) { doSend[0]=0; doSend[1]=0; }
      if(parsHost.iStep+1-window.node<0) { doRecv[0]=0; doRecv[1]=0; }
      #endif
      if(doWait && parsHost.wleft==nR+(Ns-Ntime-1)   ) {
        DEBUG_MPI(("waiting P (node %d) wleft=%d\n", window.node, parsHost.wleft)); 
        if(window.node!=window.Nprocs-1                     ) { WaitMPI(2,&reqRp, &status);WaitMPI(6,&reqRp_pml, &status); }
        if(window.node!=0               && parsHost.iStep!=0) { WaitMPI(1,&reqSm, &status);WaitMPI(5,&reqSm_pml, &status); }
      }
      if(parsHost.wleft==nR-Ns-Ns-1 && window.node!=window.Nprocs-1) {
        DEBUG_MPI(("Send&Recv P(%d) (node %d) wleft=%d\n", parsHost.wleft+Ns, window.node, parsHost.wleft));
        SendMPI(&window.data    [(nR-Ns)*Na   ], doSend[1]*Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+0, MPI_COMM_WORLD, &reqSp    ,0);
        SendMPI(&window.dataPMLa[(nR-Ns)*Npmly], doSend[1]*Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+1, MPI_COMM_WORLD, &reqSp_pml,4);
        RecvMPI(&window.data    [(nR-Ns)*Na   ], doRecv[1]*Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+0, MPI_COMM_WORLD, &reqRp    ,2);
        RecvMPI(&window.dataPMLa[(nR-Ns)*Npmly], doRecv[1]*Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+1, MPI_COMM_WORLD, &reqRp_pml,6);
      }
      if(doWait && parsHost.wleft==nL+Ns+(Ns-Ntime-1)   && parsHost.iStep!=0) { 
        DEBUG_MPI(("waiting M (node %d) wleft=%d\n", window.node, parsHost.wleft)); 
        if(window.node!=0              ) { WaitMPI(3,&reqRm, &status);WaitMPI(7,&reqRm_pml, &status); }
        if(window.node!=window.Nprocs-1) { WaitMPI(0,&reqSp, &status);WaitMPI(4,&reqSp_pml, &status); }
      }
      #ifdef MPI_NUDGE
      if(doWait && (parsHost.wleft+Ns)%1==0) {
        if(parsHost.iStep!=0 && window.node!=window.Nprocs-1) { DEBUG_MPI(("testing sendP (node %d) wleft=%d\n", window.node, parsHost.wleft)); MPI_Test(&reqSp, &flag, &status);MPI_Test(&reqSp_pml, &flag, &status); }
        if(                     window.node!=window.Nprocs-1) { DEBUG_MPI(("testing recvP (node %d) wleft=%d\n", window.node, parsHost.wleft)); MPI_Test(&reqRp, &flag, &status);MPI_Test(&reqRp_pml, &flag, &status); }
        if(parsHost.iStep!=0 && window.node!=0              ) { DEBUG_MPI(("testing sendM (node %d) wleft=%d\n", window.node, parsHost.wleft)); MPI_Test(&reqSm, &flag, &status);MPI_Test(&reqSm_pml, &flag, &status); }
        if(parsHost.iStep!=0 && window.node!=0              ) { DEBUG_MPI(("testing recvM (node %d) wleft=%d\n", window.node, parsHost.wleft)); MPI_Test(&reqRm, &flag, &status);MPI_Test(&reqRm_pml, &flag, &status); }
      }
      #endif
      #ifdef MPI_TEST
      if(parsHost.iStep-window.node>0)
      #endif
      
      ampi_exch.do_run=1;
      if(NasyncNodes>1) { if(sem_init(&ampi_exch.sem_calc, 0,0)==-1) printf("Error semaphore init errno=%d\n", errno);
                          if(sem_init(&ampi_exch.sem_mpi , 0,0)==-1) printf("Error semaphore init errno=%d\n", errno); }
      #pragma omp parallel num_threads(2)
      {
      if(omp_get_thread_num()==1) {
        window.calcDtorres(nL,nR, parsHost.wleft<nL && window.node!=0, parsHost.wleft>=nR-Ns && window.node!=window.Nprocs-1);
        ampi_exch.do_run=0; if(NasyncNodes>1) if(sem_post(&ampi_exch.sem_mpi)<0) printf("sem_post_mpi end error %d\n",errno); 
      }
        #pragma omp master
        if(NasyncNodes>1) { while(ampi_exch.do_run) ampi_exch.run(); if(sem_post(&ampi_exch.sem_calc)<0) printf("sem_post_calc end error %d\n",errno); }
      }
      if(NasyncNodes>1) { if(sem_destroy(&ampi_exch.sem_mpi )<0) printf("sem_destroy error %d\n",errno);
                          if(sem_destroy(&ampi_exch.sem_calc)<0) printf("sem_destroy error %d\n",errno); }

      if(parsHost.wleft==nL-Ns-1  && window.node!=0              ) {
        DEBUG_MPI(("Send&Recv M(%d) (node %d) wleft=%d\n", parsHost.wleft+Ns+1, window.node, parsHost.wleft));
        SendMPI(&window.data    [ nL    *Na   ], doSend[0]*Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep  )*2+0, MPI_COMM_WORLD, &reqSm    ,1);
        SendMPI(&window.dataPMLa[ nL    *Npmly], doSend[0]*Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep  )*2+1, MPI_COMM_WORLD, &reqSm_pml,5);
        RecvMPI(&window.data    [ nL    *Na   ], doRecv[0]*Ns*Na   *sizeof(DiamondRag   )/sizeof(ftype), MPI_FTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+0, MPI_COMM_WORLD, &reqRm,    3);
        RecvMPI(&window.dataPMLa[ nL    *Npmly], doRecv[0]*Ns*Npmly*sizeof(DiamondRagPML)/sizeof(ftype), MPI_FTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+1, MPI_COMM_WORLD, &reqRm_pml,7);
      }
    }
    #endif//BLOCK_SEND
    #else//MPI_ON not def
    window.calcDtorres();
    #endif//MPI_ON
    window.synchronize();
  }
//    printf("ix=%d\n",ix);
/*    int zones[] = {0, Npmlx/2, tfsfSm/dx/NDT-2, tfsfSp/dx/NDT+2, Ns-Npmlx/2, Ns}; int izon=0;
    Dtorres(max(ix,zones[izon]), min(Ntime,zones[izon+1]-ix), max(zones[izon]-ix,0), true );
    
    izon++;
    Dtorres(max(ix,zones[izon]), min(Ntime,zones[izon+1]-ix), max(zones[izon]-ix,0), false);
    izon++;
    Dtorres(max(ix,zones[izon]), min(Ntime,zones[izon+1]-ix), max(zones[izon]-ix,0), false, ((parsHost.iStep+1)*Ntime*dt<shotpoint.tStop)?true:false);
    izon++;
    Dtorres(max(ix,zones[izon]), min(Ntime,zones[izon+1]-ix), max(zones[izon]-ix,0), false);

    izon++;
    Dtorres(max(ix,zones[izon]), min(Ntime,zones[izon+1]-ix), max(zones[izon]-ix,0), true );*/
  #endif

  double calcTime=t0.gettime();
  double yee_cells = 0;
  double overhead=0;
  #ifndef TEST_RATE
  yee_cells = NDT*NDT*Ntime*(unsigned long long)(Nv*((Na+1-NDev)*NasyncNodes+1-NasyncNodes))*Np;
  overhead = window.RAMcopytime/window.GPUcalctime;
  printf("Step %d /node %d/ subnode %d/: Time %9.09f ms |overhead %3.03f%% | ", parsHost.iStep, window.node, window.subnode, calcTime, 100*overhead);
//  for(int idev=0;idev<NDev;idev++) printf("%3.03f%% ", 100*window.disbal[idev]/window.GPUcalctime);
  printf("|rate %9.09f GYee_cells/sec |total grid %ld cells | isTFSF=%d \n", 1.e-9*yee_cells/(calcTime*1.e-3), (unsigned long long)yee_cells/Ntime, (parsHost.iStep+1)*Ntime*dt<shotpoint.tStop );
  #else
  yee_cells = NDT*NDT*Ntime*(unsigned long long)(Nv*((Na-2)/TEST_RATE))*torreNum;
  printf("Step %d: Time %9.09f ms |overhead %3.03f%% |rate %9.09f %d %d %d %d (GYee cells/sec,Np,Na,Nv,Ntime) |isTFSF=%d \n", parsHost.iStep, calcTime, 100*overhead, 1.e-9*yee_cells/(calcTime*1.e-3), Np,Na,Nv,Ntime, (parsHost.iStep+1)*Ntime*dt<shotpoint.tStop );
  #endif
  #ifdef MPI_ON
  double AllCalcTime;
  MPI_Reduce(&calcTime, &AllCalcTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if(window.node==0 && 0) printf("===(%3d)===AllCalcTime %9.09f sec |rate %9.09f GYee_cells/sec\n", parsHost.iStep, AllCalcTime*1e-3, 1.e-9*yee_cells/(AllCalcTime*1.e-3) );
  #endif
  parsHost.iStep++;
  copy2dev(parsHost, pars);
  return 0; 
}
