#include "params.h"
#include "texmodel.cuh"
//using namespace aiv;
#ifdef MPI_ON
#include <mpi.h>
#endif

__constant__ float texStretchH;
__constant__ float2 texStretch[MAX_TEXS];
__constant__ float2 texShift[MAX_TEXS];
__constant__ float2 texStretchShow;
__constant__ float2 texShiftShow;
#ifdef USE_TEX_REFS
texture<coffS_t, cudaTextureType3D, cudaReadModeElementType> layerRefS;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefV;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefT;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefTa;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefTi;
#endif
void ModelTexs::init(){
  int node=0, Nprocs=1;
  #ifdef MPI_ON
  MPI_Comm_rank (MPI_COMM_WORLD, &node);
  MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
  #endif
  //---------------------------------------------------//--------------------------------------
  ShowTexBinded=0;
  Ntexs=1; // get from aivModel
  if(Ntexs>MAX_TEXS) { printf("Error: Maximum number of texs is reached (%d>%d)\n", Ntexs, MAX_TEXS); exit(-1); }
  HostLayerS = new coffS_t*[Ntexs]; HostLayerV = new float*[Ntexs]; HostLayerT = new float*[Ntexs]; HostLayerTi = new float*[Ntexs]; HostLayerTa = new float*[Ntexs]; 
  for(int idev=0;idev<NDev;idev++) { DevLayerS[idev] = new cudaArray*[Ntexs]; DevLayerV[idev] = new cudaArray*[Ntexs]; DevLayerT[idev] = new cudaArray*[Ntexs]; DevLayerTi[idev] = new cudaArray*[Ntexs]; DevLayerTa[idev] = new cudaArray*[Ntexs]; }
  for(int idev=0;idev<NDev;idev++) { layerS_host[idev] = new cudaTextureObject_t[Ntexs]; layerV_host[idev] = new cudaTextureObject_t[Ntexs]; layerT_host[idev] = new cudaTextureObject_t[Ntexs]; layerTi_host[idev] = new cudaTextureObject_t[Ntexs];  layerTa_host[idev] = new cudaTextureObject_t[Ntexs]; }
  for(int idev=0;idev<NDev;idev++) { CHECK_ERROR( cudaSetDevice(idev) );
    CHECK_ERROR( cudaMalloc((void**)&layerS [idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerV [idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerT [idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerTi[idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerTa[idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
  }
  CHECK_ERROR( cudaSetDevice(0) );
  int Nh=1; unsigned long long texsize_onhost=0, texsize_ondevs=0;
  texN    = new int3 [Ntexs];     //get from aivModel
  tex0    = new int  [Ntexs];     //get from aivModel
  texStep = new float[Ntexs];     //get from aivModel
  float2 texStretchHost[MAX_TEXS];
  float2 texShiftHost[MAX_TEXS];
  for(int ind=0; ind<Ntexs; ind++) {
    #ifdef USE_AIVLIB_MODEL
    //get texN from aivModel
    get_texture_size(texN[ind].x, texN[ind].y, texN[ind].z);
    #else
    // My own texN
    texN[ind].x  = Np/2+1;
    texN[ind].y  = Nz/32+1;
    texN[ind].z  = Nh  ;
    tex0[ind]  = 0     ;//in_Yee_cells
    texStep[ind]  = 3.0;//in_Yee_cells
    #endif
    tex0[ind]  = 0     ;//in_Yee_cells
    texStep[ind]  = Np*3.0/(texN[ind].x-1);//in_Yee_cells

    int texNwindow = int(ceil(Ns*NDT/texStep[ind])+2);
    texStretchHost[ind].x = 1.0/(2*texStep[ind]*texNwindow);
    texStretchHost[ind].y = 1.0/(2*Nz)*(texN[ind].y-1)/texN[ind].y;
    texShiftHost[ind].x = 1.0/(2*texNwindow);
    texShiftHost[ind].y = 1.0/(2*texN[ind].y);
    texsize_onhost+= texN[ind].x*texN[ind].y*texN[ind].z;
    texsize_ondevs+= texNwindow*texN[ind].y*texN[ind].z;
    if(node==0) printf("Texture%d Size %dx%dx%d (Nx x Ny x Nh)\n", ind, texN[ind].x, texN[ind].y, texN[ind].z);
    if(node==0) printf("Texture%d Stepx %g\n", ind, texStep[ind]);
    if(texStep[ind]<NDT) { printf("Texture profile step is smaller than 3*Yee_cells; Is it right?\n"); exit(-1); }
  }
  float2 texStretchShowHost = make_float2(1.0/(2*NDT*Np)*(texN[0].x-1)/texN[0].x, 0.);
  float2 texShiftShowHost   = make_float2(1./(2*texN[0].x), 0.);
  for(int i=0; i<NDev; i++) {
    CHECK_ERROR( cudaSetDevice(i) );
    h_scale = 2*((1<<16)/(2*texN[0].z)); const float texStretchH_host = 1.0/(texN[0].z*h_scale);
    CHECK_ERROR( cudaMemcpyToSymbol(texStretchH   ,&texStretchH_host, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texStretch    , texStretchHost, sizeof(float2)*Ntexs, 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texShift      , texShiftHost  , sizeof(float2)*Ntexs, 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texStretchShow, &texStretchShowHost, sizeof(float2)*Ntexs, 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texShiftShow  , &texShiftShowHost  , sizeof(float2)*Ntexs, 0, cudaMemcpyHostToDevice) );
  }
  CHECK_ERROR( cudaSetDevice(0) );
  if(node==0) printf("Textures data on host   : %.3fMB\n", texsize_onhost*(sizeof(coffS_t)+2*sizeof(float))/(1024.*1024.));
  if(node==0) printf("Textures data on devices: %.3fMB\n", texsize_ondevs*(sizeof(coffS_t)+2*sizeof(float))/(1024.*1024.));
  cudaChannelFormatDesc channelDesc;
  for(int ind=0; ind<Ntexs; ind++) {
    const int texNx = texN[ind].x, texNy = texN[ind].y, texNh = texN[ind].z;
    int texNwindow = int(ceil(Ns*NDT/texStep[ind])+2);
    HostLayerS[ind] = new coffS_t[texNx*texNy*texNh]; //get pointer from aivModel
    HostLayerV[ind] = new float  [texNx*texNy*texNh]; //get pointer from aivModel
    #ifndef ANISO_TR
    HostLayerT[ind]  = new float  [texNx*texNy*texNh]; //get pointer from aivModel
    #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
    HostLayerTi[ind] = new float  [texNx*texNy*texNh]; //get pointer from aivModel
    HostLayerTa[ind] = new float  [texNx*texNy*texNh]; //get pointer from aivModel
    HostLayerT[ind] = HostLayerTa[ind];
    #endif
    for(int idev=0;idev<NDev;idev++) { CHECK_ERROR( cudaSetDevice(idev) ); 
    channelDesc = cudaCreateChannelDesc<coffS_t>(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerS[idev][ind], &channelDesc, make_cudaExtent(texNy,texNh,texNwindow)) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerV[idev][ind], &channelDesc, make_cudaExtent(texNy,texNh,texNwindow)) );
    #ifndef ANISO_TR
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerT [idev][ind], &channelDesc, make_cudaExtent(texNy,texNh,texNwindow)) );
    #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerTi[idev][ind], &channelDesc, make_cudaExtent(texNy,texNh,texNwindow)) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerTa[idev][ind], &channelDesc, make_cudaExtent(texNy,texNh,texNwindow)) );
    #endif
    }
    CHECK_ERROR( cudaSetDevice(0) );
    ftype* rhoArr; rhoArr=new ftype[texNh+1];
    for(int ix=0; ix<texNx; ix++) for(int iy=0; iy<texNy; iy++) {
      for(int ih=0; ih<texNh; ih++) { //or get from aivModel
        // remember about yshift for idev>0
        float Vp=defCoff::Vp, Vs=defCoff::Vs, rho=defCoff::rho, drho=defCoff::drho;
        ftype C11=Vp*Vp        , C13=Vp*Vp-2*Vs*Vs, C12=Vp*Vp-2*Vs*Vs;
        ftype C31=Vp*Vp-2*Vs*Vs, C33=Vp*Vp        , C32=Vp*Vp-2*Vs*Vs;
        ftype C21=Vp*Vp-2*Vs*Vs, C23=Vp*Vp-2*Vs*Vs, C22=Vp*Vp;
        ftype C44=Vs*Vs, C66=Vs*Vs, C55=Vs*Vs;
        #ifdef USE_AIVLIB_MODEL
        GeoPhysPar p = get_texture_cell(ix,iy,ih-((ih==texNh)?1:0)); Vp=p.Vp; Vs=p.Vs; rho=p.sigma; drho=1.0/rho;

        ftype Vp_q = Vp, Vs_q1 = Vs, Vs_q2 = Vs;
        //------Anisotropy flag-------//
//        if(rho<0) { rho = -rho; Vp_q = p.Vp_q; Vs_q1 = p.Vs_q1; Vs_q2 = p.Vs_q2; }
        ftype eps = 0, delta = 0, gamma = 0;
        eps   = Vp_q /Vp-1;
        delta = Vs_q1/Vs-1;
        gamma = Vs_q2/Vs-1

        ftype xx = Vp*Vp;
        ftype yy = (-Vs*Vs+sqrt((Vp*Vp-Vs*Vs)*(Vp*Vp*(1+2*delta)-Vs*Vs)));
        ftype zz = (2*eps+1)*Vp*Vp - (2*gamma+1)*2*Vs*Vs;
        ftype ww = (2*eps+1)*Vp*Vp;
        ftype ii = Vs*Vs;
        ftype aa = (2*gamma+1)*Vs*Vs;
        //C11,C12,C13;
        //C21,C22,C23;
        //C31,C32,C33;
        #else
        //if(ix<texNx/4)   Vp*= (1.0-0.5)/(texNx/4)*ix+0.5;
        //if(ix>3*texNx/4) Vp*= (0.5-1.0)/(texNx/4)*ix+0.5+4*(1.0-0.5);
        #endif
        rhoArr[ih] = rho;
        HostLayerV[ind][ix*texNy*texNh+ih*texNy+iy] = drho;
        #ifndef ANISO_TR
        HostLayerS[ind][ix*texNy*texNh+ih*texNy+iy] = make_float2( Vp*Vp, Vp*Vp-2*Vs*Vs )*rho;
        HostLayerT[ind][ix*texNy*texNh+ih*texNy+iy] = Vs*Vs*rho;
        #elif ANISO_TR==1
        C11 = xx; C12 = yy; C23 = zz; C22 = ww; C44 = aa; C55 = ii;
        HostLayerS[ind][ix*texNy*texNh+ih*texNy+iy] = make_float4( C11, C12, C23, C22 )*rho;
        HostLayerTa[ind][ix*texNy*texNh+ih*texNy+iy] = C44*rho;
        HostLayerTi[ind][ix*texNy*texNh+ih*texNy+iy] = C55*rho;
        #elif ANISO_TR==2
        C22 = xx; C12 = yy; C13 = zz; C11 = ww; C55 = aa; C44 = ii;
        HostLayerS[ind][ix*texNy*texNh+ih*texNy+iy] = make_float4( C22, C12, C13, C11 )*rho;
        HostLayerTa[ind][ix*texNy*texNh+ih*texNy+iy] = C55*rho;
        HostLayerTi[ind][ix*texNy*texNh+ih*texNy+iy] = C44*rho;
        #elif ANISO_TR==3
        C33 = xx; C13 = yy; C12 = zz; C11 = ww; C66 = aa; C44 = ii;
        HostLayerS[ind][ix*texNy*texNh+ih*texNy+iy] = make_float4( C33, C13, C12, C11 )*rho;
        HostLayerTa[ind][ix*texNy*texNh+ih*texNy+iy] = C66*rho;
        HostLayerTi[ind][ix*texNy*texNh+ih*texNy+iy] = C44*rho;
        #else
        #error ANISO_TYPE ANISO_TR not implemented yet
        #endif
      }
      #ifdef USE_AIVLIB_MODEL
      if(iy==0) { printf("Testing get_h ix=%d/%d \r", ix, texNx-1); fflush(stdout); }
      int aivTexStepX=Np*NDT*2/(texNx-1); //in half-YeeCells
      int aivTexStepY=2*Nz/(texNy-1); //in half-YeeCells
      for(int xx=(ix==texNx-1?1:0); xx<((ix==0)?1:aivTexStepX); xx++) for(int yy=(iy==texNy-1?1:0); yy<((iy==0)?1:aivTexStepY); yy++) {
        for(int iz=0; iz<Na*NDT*2; iz++) {
          unsigned short h = get_h(ix*aivTexStepX-xx, iy*aivTexStepY-yy, -iz*0.5*da);
          int id = h/(2*h_scale), idd=h%(2*h_scale); 
          //int id = floor((h)/double(1<<16)*112);
          float rho1 = rhoArr[2*id];
          float rho2 = rhoArr[2*id+1];
          if(id<0 || 2*id>=texNh || idd>h_scale || rho1<=0 || rho2<=0)
             printf("Error: ix=%d-%d iy=%d-%d iz=%g id=%d h%%h_scale=%d rho1=%g rho2=%g\n", ix*aivTexStepX, xx, iy*aivTexStepY, yy, -iz*0.5*da, id, idd, rho1,rho2);
        }
      }
      #endif
    }
    delete rhoArr;
  }
  printf("\n");
  for(int idev=0;idev<NDev;idev++) { CHECK_ERROR( cudaSetDevice(idev) ); 
  CHECK_ERROR( cudaMemcpy(layerS [idev], layerS_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerV [idev], layerV_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerT [idev], layerT_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerTi[idev], layerTi_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerTa[idev], layerTa_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  }
  CHECK_ERROR( cudaSetDevice(0) );

  if(node==0) printf("creating texture objects...\n");
  cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  for(int idev=0;idev<NDev;idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    for(int ind=0; ind<Ntexs; ind++) {
      /*const int texNx = ceil(Ns*NDT/texStep[ind])+2, texNy = texN[ind].y, texNh = texN[ind].z;
      cudaMemcpy3DParms copyparms={0}; copyparms.srcPos=make_cudaPos(0,0,0); copyparms.dstPos=make_cudaPos(0,0,0);
      copyparms.kind=cudaMemcpyHostToDevice;
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerS[ind][0], texNh*sizeof(float2), texNh, texNy);
      copyparms.dstArray = DevLayerS[idev][ind];
      copyparms.extent = make_cudaExtent(texNh,texNy,texNx);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerV[ind][0], texNh*sizeof(float ), texNh, texNy);
      copyparms.dstArray = DevLayerV[idev][ind];
      copyparms.extent = make_cudaExtent(texNh,texNy,texNx);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerT[ind][0], texNh*sizeof(float ), texNh, texNy);
      copyparms.dstArray = DevLayerT[idev][ind];
      copyparms.extent = make_cudaExtent(texNh,texNy,texNx);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );*/

      cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
      texDesc.normalizedCoords = 1;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.addressMode[0] = cudaAddressModeClamp; // in future try to test ModeBorder
      texDesc.addressMode[1] = cudaAddressModeClamp; // in future try to test ModeBorder
      texDesc.addressMode[2] = cudaAddressModeWrap;
      resDesc.res.array.array = DevLayerS[idev][ind]; //CHECK_ERROR( cudaCreateTextureObject(&layerS_host[idev][ind], &resDesc, &texDesc, NULL) );
      resDesc.res.array.array = DevLayerV[idev][ind]; //CHECK_ERROR( cudaCreateTextureObject(&layerV_host[idev][ind], &resDesc, &texDesc, NULL) );
      resDesc.res.array.array = DevLayerT[idev][ind]; //CHECK_ERROR( cudaCreateTextureObject(&layerT_host[idev][ind], &resDesc, &texDesc, NULL) );
      if(ind==0){
      #if TEX_MODEL_TYPE!=1
      resDesc.res.array.array = DevLayerS[idev][ind]; //CHECK_ERROR( cudaCreateTextureObject(&TexlayerS[idev], &resDesc, &texDesc, NULL) );
      resDesc.res.array.array = DevLayerV[idev][ind]; //CHECK_ERROR( cudaCreateTextureObject(&TexlayerV[idev], &resDesc, &texDesc, NULL) );
      resDesc.res.array.array = DevLayerT[idev][ind]; //CHECK_ERROR( cudaCreateTextureObject(&TexlayerT[idev], &resDesc, &texDesc, NULL) );
      #endif
      }
    }
    #ifdef USE_TEX_REFS
    layerRefS.addressMode[0] = cudaAddressModeClamp; layerRefV.addressMode[0] = cudaAddressModeClamp; layerRefT.addressMode[0] = cudaAddressModeClamp; layerRefTi.addressMode[0] = cudaAddressModeClamp; layerRefTa.addressMode[0] = cudaAddressModeClamp;
    layerRefS.addressMode[1] = cudaAddressModeClamp; layerRefV.addressMode[1] = cudaAddressModeClamp; layerRefT.addressMode[1] = cudaAddressModeClamp; layerRefTi.addressMode[1] = cudaAddressModeClamp; layerRefTa.addressMode[1] = cudaAddressModeClamp;
    layerRefS.addressMode[2] = cudaAddressModeWrap;  layerRefV.addressMode[2] = cudaAddressModeWrap;  layerRefT.addressMode[2] = cudaAddressModeWrap;  layerRefTi.addressMode[2] = cudaAddressModeWrap;  layerRefTa.addressMode[2] = cudaAddressModeWrap;
    layerRefS.filterMode = cudaFilterModeLinear; layerRefV.filterMode = cudaFilterModeLinear; layerRefT.filterMode = cudaFilterModeLinear;layerRefTi.filterMode = cudaFilterModeLinear;layerRefTa.filterMode = cudaFilterModeLinear;
    layerRefS.normalized = true; layerRefV.normalized = true; layerRefT.normalized = true; layerRefTi.normalized = true; layerRefTa.normalized = true;
    channelDesc = cudaCreateChannelDesc<coffS_t>(); CHECK_ERROR( cudaBindTextureToArray(layerRefS , DevLayerS [idev][0], channelDesc) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefV , DevLayerV [idev][0], channelDesc) );
    #ifndef ANISO_TR
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefT , DevLayerT [idev][0], channelDesc) );
    #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefTi, DevLayerTi[idev][0], channelDesc) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefTa, DevLayerTa[idev][0], channelDesc) );
    #endif//ANISO_TR
    #endif//USE_TEX_REFS

    CHECK_ERROR( cudaMemcpy(layerS [idev], layerS_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerV [idev], layerV_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerT [idev], layerT_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerTi[idev], layerTi_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerTa[idev], layerTa_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  }
  CHECK_ERROR(cudaSetDevice(0));

}
void ModelTexs::copyTexs(const int x1dev, const int x2dev, const int x1host, const int x2host, cudaStream_t& streamCopy){
}
void ModelTexs::copyTexs(const int xdev, const int xhost, cudaStream_t& streamCopy){
  if(xhost==Np) for(int ind=0; ind<Ntexs; ind++) copyTexs(xhost+ceil(texStep[ind]/NDT), xhost+ceil(texStep[ind]/NDT), streamCopy);
  for(int idev=0;idev<NDev;idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    for(int ind=0; ind<Ntexs; ind++) {
      int texNwindow = int(ceil(Ns*NDT/texStep[ind])+2);
      //if(xhost*NDT<=tex0[ind] || xhost*NDT>tex0[ind]+texN[ind].x*texStep[ind]) continue;
      if(xhost*NDT<=tex0[ind]) continue;
      if(floor(xhost*NDT/texStep[ind])==floor((xhost-1)*NDT/texStep[ind])) continue;
      int storeX  = int(floor(xhost*NDT/texStep[ind])-1+texNwindow)%texNwindow;
      int loadX = int(floor((xhost*NDT-tex0[ind])/texStep[ind])-1);
      double numXf = NDT/texStep[ind];
      int numX = (numXf<=1.0)?1:floor(numXf);

      DEBUG_PRINT(("copy Textures to dev%d, ind=%d hostx=%d -> %d=devx (num=%d) // texNwindow=%d\n", idev, ind, loadX, storeX, numX, texNwindow));

      const int texNz = texN[ind].y, texNy = texN[ind].z;
      cudaMemcpy3DParms copyparms={0}; copyparms.srcPos=make_cudaPos(0,0,loadX); copyparms.dstPos=make_cudaPos(0,0,storeX);
      copyparms.kind=cudaMemcpyHostToDevice;
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerS[ind][0], texNz*sizeof(coffS_t), texNz, texNy);
      copyparms.dstArray = DevLayerS[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerV[ind][0], texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerV[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy) );
      #ifndef ANISO_TR
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerT[ind][0], texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerT[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy) );
      #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerTi[ind][0],texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerTi[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerTa[ind][0],texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerTa[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy) );
      #else
      #error UNKNOWN ANISO_TYPE
      #endif
    }
  }
  CHECK_ERROR(cudaSetDevice(0));
}

void ModelRag::set(int x, int y) {
    #if TEX_MODEL_TYPE==1
    for(int i=0;i<4        ;i++) for(int iz=0;iz<Nz;iz++) I[i][iz]=0;
    #endif
    for(int i=0;i<32;i++) for(int iz=0;iz<Nz;iz++) { h[i][iz].x=0; h[i][iz].y=0; }
  // set values from aivModel
  // remember about yshift for idev>0
    int idev=0; int ym=0;
    while(y>=ym && idev<NDev) { ym+=NStripe[idev]; idev++; }
    y-= idev-1;
    const int d_index[64][3] = { {-3, +3, 1}, {-2, +3, 0}, {-2, +4, 1}, {-1, +4, 0}, {-1, +5, 1}, {+0, +5, 0}, 
                                 {-2, +2, 1}, {-1, +2, 0}, {-1, +3, 1}, {+0, +3, 0}, {+0, +4, 1}, {+1, +4, 0}, 
                                 {-1, +1, 1}, {+0, +1, 0}, {+0, +2, 1}, {+1, +2, 0}, {+1, +3, 1}, {+2, +3, 0}, 
                                 {+0, +0, 1}, {+1, +0, 0}, {+1, +1, 1}, {+2, +1, 0}, {+2, +2, 1}, {+3, +2, 0},
                                 {+1, -1, 1}, {+2, -1, 0}, {+2, +0, 1}, {+3, +0, 0}, {+3, +1, 1}, {+4, +1, 0}, 
                                 {+2, -2, 1}, {+3, -2, 0}, {+3, -1, 1}, {+4, -1, 0}, {+4, +0, 1}, {+5, +0, 0},

                                 {-3, +0, 1}, {-2, -1, 1}, {-1, -1, 0}, {-1, -2, 1}, 
                                 {-2, +1, 1}, {-1, +1, 0}, {-1, +0, 1}, {+0, -1, 1}, {+1, -1, 0}, 
                                 {-1, +2, 1}, {+0, +1, 1}, {+1, +1, 0}, {+1, +0, 1}, 
                                 {+0, +3, 1}, {+1, +3, 0}, {+1, +2, 1}, {+2, +1, 1}, {+3, +1, 0}, 
                                 {+1, +4, 1}, {+2, +3, 1}, {+3, +3, 0}, {+3, +2, 1}, 
                                 {+2, +5, 1}, {+3, +5, 0}, {+3, +4, 1}, {+4, +3, 1}, {+5, +3, 0},
                                 {0,0,0} };
    #ifdef USE_AIVLIB_MODEL
    const double corrCoff1 = 1.0/double(H_MAX_SIZE)*(parsHost.texs.texN[0].z-1);
    const double corrCoff2 = 1.0/parsHost.texs.texN[0].z*H_MAX_SIZE;
    for(int i=0;i<32;i++) for(int iz=0;iz<Nz;iz++) {
      int3 x4h;
      x4h = make_int3(x*2*NDT+d_index[2*i  ][0], iz*2+d_index[2*i  ][2], y*2*NDT+d_index[2*i  ][1]); x4h = check_bounds(x4h);
      h[i][iz].x = get_h(x4h.x, x4h.y, -x4h.z*0.5*dy) + parsHost.texs.h_scale/2;
      //h[i][iz].x = ((x4h.x*x4h.y-x4h.z*0.5*dy)*corrCoff1+0.5)*corrCoff2;
      x4h = make_int3(x*2*NDT+d_index[2*i+1][0], iz*2+d_index[2*i+1][2], y*2*NDT+d_index[2*i+1][1]); x4h = check_bounds(x4h);
      h[i][iz].y = get_h(x4h.x, x4h.y, -x4h.z*0.5*dy) + parsHost.texs.h_scale/2;
      //h[i][iz].y = ((x4h.x*x4h.y-x4h.z*0.5*dy)*corrCoff1+0.5)*corrCoff2;
    }
    #endif
}


