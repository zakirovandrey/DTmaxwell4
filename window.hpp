#include <omp.h>
#include <string>
using namespace std;

void set_texture(char* index_arr, const int ix);
//void set_texture(const int ix);

struct __align__(32) mpi_message {
#ifdef MPI_ON
  pthread_t mpith;
  void* buf;
  int count;
  MPI_Datatype datatype;
  int dest;
  int tag;
  MPI_Comm comm;
  void set(void* p,int sz,MPI_Datatype tp,int rnk,int _tag,MPI_Comm world){
    buf=p; count=sz; datatype=tp; dest=rnk; tag=_tag; comm=world;
  }
#endif
};

struct Window {
  static const int NTorres=1;
  int x0, w0, ix4copy;
  DiamondRagDisp* dataDisp;
  ModelRag* dataInd;
  DiamondRag* data;
  DiamondRagPML* dataPMLa;
  DiamondRagPML* dataPMLsL, *dataPMLsR;
  double GPUcalctime, RAMcopytime, Textime;
  double disbal[NDev];
  cudaStream_t streamCopy[NDev];
  double timerPMLtop, timerI, timerDm[NDev];
  double timerPMLbot, timerX, timerDo[NDev];
  double timerP, timerCopy, timerExec;
  cudaEvent_t copyEventStart[NDev], copyEventEnd[NDev];

  int tDnum;
  bool doneMemcopy;
  int node, subnode, Nprocs;
  static mpi_message mes[8];
  Window(): GPUcalctime(0),RAMcopytime(0),Textime(0),tDnum(0),doneMemcopy(0) { 
    for(int idev=0;idev<NDev;idev++) { 
      CHECK_ERROR( cudaSetDevice(idev) ); 
      CHECK_ERROR( cudaStreamCreate(&streamCopy[idev]) ); 
      disbal[idev]=0;
    }
    CHECK_ERROR(cudaSetDevice(0)); 
  }
  ~Window() { 
    for(int idev=0;idev<NDev;idev++) {
      CHECK_ERROR( cudaStreamDestroy(streamCopy[idev]) ); 
      CHECK_ERROR( cudaEventDestroy(copyEventStart[idev]) ); 
      CHECK_ERROR( cudaEventDestroy(copyEventEnd[idev]) ); 
    }
  }
  void prepare(){
    x0 = Ns-Ntime-NTorres; w0=Np+x0; parsHost.wleft=Np; parsHost.GPUx0=w0-x0; copy2dev(parsHost, pars);
    if(Ns-Ntime<2*NTorres) { printf("Error: Ns-Ntime<2*NTorres | %d-%d<2*%d \n", Ns,Ntime,NTorres); exit(-1); }
    if(Np%Ns!=0) { printf("Error: Np=%d must be dividable by Ns=%d \n", Np,Ns); exit(-1); }
    if(Ns%NTorres!=0) { printf("Error: Ns=%d must be dividable by NTorres=%d \n", Ns,NTorres); exit(-1); }
    node=0; subnode=0; Nprocs=1;
    #ifdef MPI_ON
    MPI_Comm_rank (MPI_COMM_WORLD, &node);   subnode = node%NasyncNodes; node/= NasyncNodes;
    MPI_Comm_size (MPI_COMM_WORLD, &Nprocs); Nprocs/= NasyncNodes;
    #ifdef DROP_DATA
    parsHost.drop.open(parsHost.iStep);
    #endif//DROP_DATA
    #endif
    dataInd  = parsHost.dataInd;
    data     = parsHost.data;
    dataDisp  = parsHost.dataDisp;
    dataPMLa = parsHost.dataPMLa;
    dataPMLsL = parsHost.dataPMLsL;
    dataPMLsR = parsHost.dataPMLsR;

/*    cuTimer t0; 
    for(int idev=0, iy=0; idev<NDev; iy+=NStripe(idev), idev++) {
      for(int ix=0; ix<Ns   ; ix++) CHECK_ERROR( cudaMemcpyAsync(&parsHost.rags    [idev][ix], &data    [(Np-Ns+ix)*Na+iy], sizeof(DiamondRag   )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy) );
      for(int ix=0; ix<Npmlx; ix++) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLs[idev][ix], &dataPMLs[ ix       *Na+iy], sizeof(DiamondRagPML)*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy) );
    }
      for(int ix=0; ix<Ns   ; ix++) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLa      [ix], &dataPMLa[(Np-Ns+ix)*Npmly], sizeof(DiamondRagPML)*1*Npmly        , cudaMemcpyHostToDevice, streamCopy) );
    CHECK_ERROR(cudaStreamSynchronize(streamCopy));
    RAMcopytime+=t0.gettime();*/
    cuTimer t1; t1.init();
    //------------set_texture(parsHost.index_arr, Np-Ns);
    CHECK_ERROR(cudaDeviceSynchronize());
    Textime+=t1.gettime();

    timerPMLtop=0; timerI=0; for(int i=0;i<NDev;i++) timerDm[i]=0;
    timerPMLbot=0; timerX=0; for(int i=0;i<NDev;i++) timerDo[i]=0;
    timerP=0; timerCopy=0; timerExec=0;
    for(int idev=0;idev<NDev;idev++) {
      CHECK_ERROR( cudaSetDevice(idev) ); 
      CHECK_ERROR( cudaEventCreate(&copyEventStart[idev]) );
      CHECK_ERROR( cudaEventCreate(&copyEventEnd[idev]  ) );
    }
      CHECK_ERROR( cudaSetDevice(0) ); 
  }
  void finalize(){ 
    #ifdef DROP_DATA
    parsHost.drop.close(); 
    #endif//DROP_DATA
  }
  template<int even> inline void Dtorre(int ix, int Nt, int t0, double disbal[NDev], bool isPMLs=false, bool isTFSF=false);
  inline void Dtorres(int ix, int Nt, int t0, double disbal[NDev], bool isPMLs=false, bool isTFSF=false);
  void calcDtorres(const int nL=0, const int nR=Np, const bool isOnlyMemcopyDtH=false, const bool isOnlyMemcopyHtD=false) { 
    double t0=omp_get_wtime();
    DEBUG_MPI(("CalcDtorres OnlyMemcopyDtH=%d  OnlyMemcopyHtD=%d (node %d) wleft=%d\n", isOnlyMemcopyDtH, isOnlyMemcopyHtD, node, parsHost.wleft));
    //pthread_t tid = pthread_self();
    //cpu_set_t cpuset;
    //CPU_ZERO(&cpuset); for(int j=0; j<24; j++) CPU_SET(j, &cpuset);
    //int s0 = pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
    //int s1 = pthread_getaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
    //printf("    CPU "); for (int j=0; j<CPU_SETSIZE; j++) if(CPU_ISSET(j, &cpuset)) printf(" %d", j); printf("\n");
    
    for(int itorre=0; itorre<NTorres; itorre++) {
      doneMemcopy=false;
      int iw=w0-itorre;
      int ix = (iw-w0+x0+Ntime+1+Ns)%Ns;
      ix4copy=ix;
      bool isOnlyMemcopy = isOnlyMemcopyDtH || isOnlyMemcopyHtD; 
      if(!isOnlyMemcopy) {
      if((parsHost.iStep+1)*Ntime*dt<shotpoint.tStop) {
        int zones[] = {0, Npmlx/2, shotpoint.BoxMs/dx/NDT-2, shotpoint.BoxPs/dx/NDT+2, Np-Npmlx/2, Np}; int izon=0;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, true );
    
        izon++;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, false);
        izon++;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, false, ((parsHost.iStep+1)*Ntime*dt<shotpoint.tStop)?true:false);
        izon++;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, false);

        izon++;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, true );
      } else {
        int zones[] = {0, Npmlx/2, Np-Npmlx/2, Np}; int izon=0;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, true );
        izon++;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, false);
        izon++;
        Dtorres((max(iw,zones[izon])-w0+x0+Ns)%Ns, min(Ntime,zones[izon+1]-iw), max(zones[izon]-iw,0), disbal, true );
      } }
      if(iw+Ntime+1-Ns>=nR && node!=Nprocs-1 || iw+Ntime+1<nL && node!=0) doneMemcopy=true;
      if(!doneMemcopy) {
        for(int idev=0; idev<NDev; idev++) {
          CHECK_ERROR( cudaSetDevice(idev) ); 
          CHECK_ERROR( cudaEventRecord(copyEventStart[idev], streamCopy[idev]) );
        } CHECK_ERROR( cudaSetDevice(0) ); 
        if(!isOnlyMemcopy || isOnlyMemcopyDtH) MemcopyDtH(ix);
        if(!isOnlyMemcopy || isOnlyMemcopyHtD) MemcopyHtD(ix); 
        for(int idev=0; idev<NDev; idev++) {
          CHECK_ERROR( cudaSetDevice(idev) ); 
          CHECK_ERROR( cudaEventRecord(copyEventEnd[idev], streamCopy[idev]) );
          CHECK_ERROR( cudaStreamSynchronize(streamCopy[idev]) );
        } CHECK_ERROR( cudaSetDevice(0) );
        float elapsed=0;
        for(int idev=0; idev<NDev; idev++) {
          float elapsed_idev;
          CHECK_ERROR( cudaEventElapsedTime(&elapsed_idev, copyEventStart[idev], copyEventEnd[idev]) );
          elapsed=max(elapsed,elapsed_idev);
        }
        timerCopy+= elapsed; timerExec+=elapsed;
        doneMemcopy=true;
      }
    }
    GPUcalctime+=omp_get_wtime()-t0;
  }
  inline void RAMexch(const int ixdev, const int ixhost) {
    DEBUG_PRINT(("exchange copy Xhost->Xdevice->Xhost = %d->%d->%d  \\ %d %d \\ mallocR FreeR mallocL FreeL / %d %d %d %d\n", ixhost-Ns, ixdev, ixhost, ixhost-Ns<Np && ixhost-Ns>=0, ixhost   <Np && ixhost   >=0, ixhost-Ns==Np-1, ixhost==Np-Npmlx/2-1, ixhost-Ns==Npmlx/2, ixhost==-1));
    for(int idev=0; idev<NDev; idev++) {
      if(ixhost-Ns==Np-1 || ixhost==Np-Npmlx/2-1 || ixhost-Ns==Npmlx/2 || ixhost==-1) CHECK_ERROR( cudaSetDevice(idev) );
      if(ixhost-Ns==Np        -1) CHECK_ERROR( cudaMalloc((void**)&(parsHost.ragsPMLsR[idev]), Npmlx/2*NStripe[idev]*sizeof(DiamondRagPML)) );
      if(ixhost   ==Np-Npmlx/2-1) CHECK_ERROR( cudaFree(parsHost.ragsPMLsR[idev]) );
      if(ixhost-Ns==Npmlx/2     ) CHECK_ERROR( cudaMalloc((void**)&(parsHost.ragsPMLsL[idev]), Npmlx/2*NStripe[idev]*sizeof(DiamondRagPML)) );
      if(ixhost   ==          -1) CHECK_ERROR( cudaFree(parsHost.ragsPMLsL[idev]) );
    } 
    if(ixhost-Ns==Np-1 || ixhost==Np-Npmlx/2-1 || ixhost-Ns==Npmlx/2 || ixhost==-1) CHECK_ERROR( cudaSetDevice(0) );
    for(int idev=0, iy=0; idev<NDev; iy+=NStripe[idev], idev++) {
      if(ixhost   <Np      && ixhost   >=0         ) CHECK_ERROR( cudaMemcpyAsync(&data     [ixhost*Na+iy]                      , &parsHost.rags     [idev][ixdev*NStripe[idev]], sizeof(DiamondRag    )*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
      if(ixhost-Ns<Np      && ixhost-Ns>=0         ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.rags     [idev][ixdev*NStripe[idev]], &data     [(ixhost-Ns)*Na+iy]                 , sizeof(DiamondRag    )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost<dispReg::sR && ixhost>=dispReg::sL ) CHECK_ERROR( cudaMemcpyAsync(&dataDisp [ixhost*Na+iy]                      , &parsHost.ragsDisp [idev][ixdev*NStripe[idev]], sizeof(DiamondRagDisp)*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
      if(ixhost-Ns<dispReg::sR && ixhost-Ns>=dispReg::sL) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsDisp [idev][ixdev*NStripe[idev]], &dataDisp [(ixhost-Ns)*Na+iy]            , sizeof(DiamondRagDisp)*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost   <Npmlx/2 && ixhost   >=0         ) CHECK_ERROR( cudaMemcpyAsync(&dataPMLsL[ixhost*Na+iy]                      , &parsHost.ragsPMLsL[idev][ixdev*NStripe[idev]], sizeof(DiamondRagPML )*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
      if(ixhost-Ns<Npmlx/2 && ixhost-Ns>=0         ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLsL[idev][ixdev*NStripe[idev]], &dataPMLsL[(ixhost-Ns)*Na+iy]                 , sizeof(DiamondRagPML )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost   <Np      && ixhost   >=Np-Npmlx/2) CHECK_ERROR( cudaMemcpyAsync(&dataPMLsR[(ixhost-Np+Npmlx/2)*Na+iy]                      , &parsHost.ragsPMLsR[idev][(ixdev-Ns+Npmlx/2)*NStripe[idev]], sizeof(DiamondRagPML)*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
      if(ixhost-Ns<Np      && ixhost-Ns>=Np-Npmlx/2) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLsR[idev][(ixdev-Ns+Npmlx/2)*NStripe[idev]], &dataPMLsR[(ixhost-Ns-Np+Npmlx/2)*Na+iy]                   , sizeof(DiamondRagPML)*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost-Ns<Np      && ixhost-Ns>=0         ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsInd  [idev][ixdev*NStripe[idev]], &dataInd  [(ixhost-Ns)*Na+iy]                 , sizeof(ModelRag     )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
    }
      if(ixhost   <Np && ixhost   >=0) CHECK_ERROR( cudaMemcpyAsync(&dataPMLa[ixhost*Npmly]                  , &parsHost.ragsPMLa  [ixdev*Npmly]        , sizeof(DiamondRagPML)*1*Npmly        , cudaMemcpyDeviceToHost, streamCopy[NDev-1]) );
      if(ixhost-Ns<Np && ixhost-Ns>=0) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLa  [ixdev*Npmly]        , &dataPMLa[(ixhost-Ns)*Npmly]             , sizeof(DiamondRagPML)*1*Npmly        , cudaMemcpyHostToDevice, streamCopy[NDev-1]) );
  }
  inline void RAMexchDtH(const int ixdev, const int ixhost) {
    DEBUG_PRINT(("exchange copy Xdevice->Xhost = %d->%d  // %d // (node %d) mallocR FreeR mallocL FreeL / %d %d %d %d\n", ixdev, ixhost, ixhost   <Np && ixhost   >=0, node, ixhost-Ns==Np-1, ixhost==Np-Npmlx/2-1, ixhost-Ns==Npmlx/2, ixhost==-1));
    //DEBUG_MPI  (("exchange copy Xdevice->Xhost = %d->%d  // %d // (node %d) mallocR FreeR mallocL FreeL / %d %d %d %d\n", ixdev, ixhost, ixhost   <Np && ixhost   >=0, node, ixhost-Ns==Np-1, ixhost==Np-Npmlx/2-1, ixhost-Ns==Npmlx/2, ixhost==-1));
    for(int idev=0; idev<NDev; idev++) {
      if(ixhost-Ns==Np-1 || ixhost==Np-Npmlx/2-1 || ixhost-Ns==Npmlx/2 || ixhost==-1) CHECK_ERROR( cudaSetDevice(idev) );
      if(ixhost-Ns==Np        -1) CHECK_ERROR( cudaMalloc((void**)&(parsHost.ragsPMLsR[idev]), Npmlx/2*NStripe[idev]*sizeof(DiamondRagPML)) );
      if(ixhost   ==Np-Npmlx/2-1) CHECK_ERROR( cudaFree(parsHost.ragsPMLsR[idev]) );
      if(ixhost-Ns==Npmlx/2     ) CHECK_ERROR( cudaMalloc((void**)&(parsHost.ragsPMLsL[idev]), Npmlx/2*NStripe[idev]*sizeof(DiamondRagPML)) );
      if(ixhost   ==          -1) CHECK_ERROR( cudaFree(parsHost.ragsPMLsL[idev]) );
    } 
    if(ixhost-Ns==Np-1 || ixhost==Np-Npmlx/2-1 || ixhost-Ns==Npmlx/2 || ixhost==-1) CHECK_ERROR( cudaSetDevice(0) );
    for(int idev=0, iy=0; idev<NDev; iy+=NStripe[idev], idev++) {
      if(ixhost   <Np      && ixhost   >=0         ) CHECK_ERROR( cudaMemcpyAsync(&data     [ ixhost            *Na+iy], &parsHost.rags     [idev][ ixdev            *NStripe[idev]], sizeof(DiamondRag    )*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
      if(ixhost<dispReg::sR && ixhost>=dispReg::sL ) CHECK_ERROR( cudaMemcpyAsync(&dataDisp [ ixhost            *Na+iy], &parsHost.ragsDisp [idev][ ixdev            *NStripe[idev]], sizeof(DiamondRagDisp)*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
      if(ixhost   <Npmlx/2 && ixhost   >=0         ) CHECK_ERROR( cudaMemcpyAsync(&dataPMLsL[ ixhost            *Na+iy], &parsHost.ragsPMLsL[idev][ ixdev            *NStripe[idev]], sizeof(DiamondRagPML )*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
      if(ixhost   <Np      && ixhost   >=Np-Npmlx/2) CHECK_ERROR( cudaMemcpyAsync(&dataPMLsR[(ixhost-Np+Npmlx/2)*Na+iy], &parsHost.ragsPMLsR[idev][(ixdev-Ns+Npmlx/2)*NStripe[idev]], sizeof(DiamondRagPML )*1*NStripe[idev], cudaMemcpyDeviceToHost, streamCopy[idev]) );
    }
      if(ixhost   <Np      && ixhost   >=0         ) CHECK_ERROR( cudaMemcpyAsync(&dataPMLa [ ixhost            *Npmly], &parsHost.ragsPMLa       [ ixdev*Npmly                    ], sizeof(DiamondRagPML)*1*Npmly        , cudaMemcpyDeviceToHost, streamCopy[NDev-1]) );
  }
  inline void RAMexchHtD(const int ixdev, const int ixhost) {
    DEBUG_PRINT(("exchange copy Xhost->Xdevice = %d->%d  // %d // (node %d)\n", ixhost-Ns, ixdev, ixhost-Ns<Np && ixhost-Ns>=0, node));
    //DEBUG_MPI  (("exchange copy Xhost->Xdevice = %d->%d  // %d // (node %d)\n", ixhost-Ns, ixdev, ixhost-Ns<Np && ixhost-Ns>=0, node));
    for(int idev=0, iy=0; idev<NDev; iy+=NStripe[idev], idev++) {
      if(ixhost-Ns<Np      && ixhost-Ns>=0         ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.rags     [idev][ ixdev            *NStripe[idev]], &data     [(ixhost-Ns)           *Na+iy], sizeof(DiamondRag    )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost-Ns<dispReg::sR && ixhost-Ns>=dispReg::sL ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsDisp [idev][ ixdev      *NStripe[idev]], &dataDisp [(ixhost-Ns)           *Na+iy], sizeof(DiamondRagDisp)*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost-Ns<Npmlx/2 && ixhost-Ns>=0         ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLsL[idev][ ixdev            *NStripe[idev]], &dataPMLsL[(ixhost-Ns)           *Na+iy], sizeof(DiamondRagPML )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost-Ns<Np      && ixhost-Ns>=Np-Npmlx/2) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLsR[idev][(ixdev-Ns+Npmlx/2)*NStripe[idev]], &dataPMLsR[(ixhost-Ns-Np+Npmlx/2)*Na+iy], sizeof(DiamondRagPML )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
      if(ixhost-Ns<Np      && ixhost-Ns>=0         ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsInd  [idev][ ixdev            *NStripe[idev]], &dataInd  [(ixhost-Ns)           *Na+iy], sizeof(ModelRag      )*1*NStripe[idev], cudaMemcpyHostToDevice, streamCopy[idev]) );
    }
      if(ixhost-Ns<Np      && ixhost-Ns>=0         ) CHECK_ERROR( cudaMemcpyAsync(&parsHost.ragsPMLa       [ ixdev            *Npmly        ], &dataPMLa [(ixhost-Ns)           *Npmly], sizeof(DiamondRagPML)*1*Npmly        , cudaMemcpyHostToDevice, streamCopy[NDev-1]) );
  }
  void Memcopy() {
    for(int itorre=NTorres-1; itorre>=0; itorre--) { int ix=(x0+1+Ntime+itorre)%Ns; RAMexch(ix, w0+(ix-x0+Ns)%Ns); }
    const int x1dev =(x0+Ntime+1)%Ns, x2dev =(x0+Ntime+NTorres)%Ns;
    const int x1host= w0+Ntime+1-Ns , x2host= w0+Ntime+NTorres-Ns;
    parsHost.texs.copyTexs(x1dev, x2dev, x1host, x2host,  streamCopy);
  }
  void MemcopyDtH(const int ix) {
    bool doCopy=1; if(parsHost.iStep>0) doCopy=0;
    #ifdef USE_WINDOW
    RAMexchDtH(ix, w0+(ix-x0+Ns)%Ns);
    doCopy=1;
    #endif
  }
  void MemcopyHtD(const int ix) {
    bool doCopy=1; if(parsHost.iStep>0) doCopy=0;
    #ifdef USE_WINDOW
    RAMexchHtD(ix, w0+(ix-x0+Ns)%Ns);
    doCopy=1;
    #endif
    if(doCopy) {
    if(parsHost.wleft==Np)
    parsHost.texs.copyTexs       (ix+1, 1+w0+(ix-x0+Ns)%Ns-Ns, streamCopy);
    parsHost.texs.copyTexs       (ix  ,   w0+(ix-x0+Ns)%Ns-Ns, streamCopy);
    //parsHost.drop.copy_drop_cells(ix  ,   w0+(ix-x0+Ns)%Ns-Ns, streamCopy);
    }
  }
  void synchronize() {
    double t0=omp_get_wtime();
    for(int idev=0; idev<NDev; idev++) CHECK_ERROR( cudaStreamSynchronize(streamCopy[idev]) );
    CHECK_ERROR( cudaDeviceSynchronize() );
    DEBUG_PRINT(("window synchronization\n"));
    x0 = (x0-NTorres+Ns)%Ns; w0-=NTorres; parsHost.wleft-=NTorres; parsHost.GPUx0=w0-x0; copy2dev(parsHost, pars);
    DEBUG_PRINT(("window x0=%d w0=%d wleft=%d GPUx0=%d\n", x0,w0, parsHost.wleft, parsHost.GPUx0));
    RAMcopytime+=omp_get_wtime()-t0;
  }
/*  void drop(string f){
    FILE* pFile;
    char fname[256];
    sprintf(fname, "%s/Inv2%05d.arr", parsHost.dir->c_str(), parsHost.iStep);
    pFile = fopen(fname,"w");
    int zero=0, twelve = 12, dim = 3, sizeofT = sizeof(float);  
    fwrite(&twelve , sizeof(int  ), 1, pFile);  //size of comment
    fwrite(&zero   , sizeof(int  ), 1, pFile);    // comment
    fwrite(&zero   , sizeof(int  ), 1, pFile);    //comment
    fwrite(&zero   , sizeof(int  ), 1, pFile);    //comment
    fwrite(&dim    , sizeof(int  ), 1, pFile);     //dim = 
    fwrite(&sizeofT, sizeof(float), 1, pFile); //data size
    fwrite(&Nz     , sizeof(int  ), 1, pFile);
    fwrite(&Ny     , sizeof(int  ), 1, pFile);
    fwrite(&LX     , sizeof(int  ), 1, pFile);
    printf("saving %s\n",fname);
    for(int x=0; x<LX; x++) for(int y=0; y<Ny; y++) for(int z=0; z<Nz; z++) {
//      if(data[x*Ny+y].Ez[z]!=0.) printf("%d %d %d %g\n",x,y,z,data[x*Ny+y].Ez[z] );
      float Sx=data[x*(Ny+Nover)+y+((y<tfsfH)?0:Nover)].Sx[z]; float Sy=data[x*Ny+y+((y<tfsfH)?0:Nover)].Sy[z]; float Sz=data[x*Ny+y+((y<tfsfH)?0:Nover)].Sz[z];
      float Tx=data[x*(Ny+Nover)+y+((y<tfsfH)?0:Nover)].Tx[z]; float Ty=data[x*Ny+y+((y<tfsfH)?0:Nover)].Ty[z]; float Tz=data[x*Ny+y+((y<tfsfH)?0:Nover)].Tz[z];
      float val = 1./3.*(Sx*Sy+Sx*Sy+Sy*Sz-Tx*Tx-Ty*Ty-Tz*Tz); val = (val<0)?(-sqrt(-val)):sqrt(val);
      fwrite(&val, sizeof(float), 1, pFile);
    }
    fclose(pFile);
  }*/
};

