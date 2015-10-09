#ifndef DEFS_H
#define DEFS_H
const ftype p9d8 = 9./8.;
const ftype m1d24=-1./24.;
const ftype p27= 27.;
const ftype dtdxd24 = dt/dx/24.;
const ftype dtdyd24 = dt/dy/24.;
const ftype dtdzd24 = dt/dz/24.;
#if SquareGrid==1
const ftype dtdrd24 = dt/dr/24.;
#else
const ftype dtdrd24 = 1.0;
#endif
const ftype drdt24 = dr/dt*24.;

extern ftype* __restrict__ hostKpmlx1; extern ftype* __restrict__ hostKpmlx2;
extern ftype* __restrict__ hostKpmly1; extern ftype* __restrict__ hostKpmly2;
extern ftype* __restrict__ hostKpmlz1; extern ftype* __restrict__ hostKpmlz2;
extern __constant__ ftype Kpmlx1[(KNpmlx==0)?1:KNpmlx];
extern __constant__ ftype Kpmlx2[(KNpmlx==0)?1:KNpmlx];
extern __constant__ ftype Kpmly1[(KNpmly==0)?1:KNpmly];
extern __constant__ ftype Kpmly2[(KNpmly==0)?1:KNpmly];
extern __constant__ ftype Kpmlz1[(KNpmlz==0)?1:KNpmlz];
extern __constant__ ftype Kpmlz2[(KNpmlz==0)?1:KNpmlz];

template<class Ph, class Pd> static void copy2dev(Ph &hostP, Pd &devP) {
  for(int i=0; i<NDev; i++) {
    CHECK_ERROR( cudaSetDevice(i) );
    CHECK_ERROR( cudaMemcpyToSymbol(devP, &hostP, sizeof(Pd)) );
  }
  CHECK_ERROR( cudaSetDevice(0) );
}
//__device__ __forceinline__ static bool isOutS(const int x) { return (x<0+NDT || x>=Ns*2*NDT-2*NDT); }
__device__ __forceinline__ static bool isOutS(const int x) { return (x<0+NDT || x>=Ns*2*NDT-NDT); }
//__device__ __forceinline__ static bool isOutS(const int x) { return false; }
__device__ __forceinline__ static bool inPMLsync(const int x) { return (x<Npmlx/2*2*NDT-NDT && x>=0 || x<Ns*2*NDT && x>=Ns*2*NDT-Npmlx/2*2*NDT+NDT); }
//__device__ __forceinline__ static bool inPMLsync(const int x) { return true; }

#ifdef USE_TEX_REFS
#define TEX_MODEL_TYPE 0
#endif

#define GLOBAL(x) ( x+2*NDT*glob_ix )
__device__ inline int get_iz(const int nth) {
  int iz=nth;
  if(nth<Npmlz && nth>=Npmlz/2) iz+=Nv-Npmlz;
  else if(nth>=Npmlz) iz-=Npmlz/2;
  return iz;
}
__host__ __device__ inline int get_pml_iy(const int iy) {
  const int diy=(iy+Na)%Na;
  return (diy<Npmly/2)?diy:(diy-Na+Npmly);
  //const int pmliy=iy-Na+Npmly;
  //return pmliy;
}
__device__ inline int get_pml_ix(const int ix) {
  if (ix<Npmlx/2*2*NDT) return (ix+KNpmlx)%KNpmlx;
  else return (ix-Ns*2*NDT+Npmlx*2*NDT)%KNpmlx;
}

__device__ inline int get_idev(const int iy, int& ym) {
  int idev=0; ym=0;
  while(iy>=ym && idev<NDev) { ym+=NStripe(idev); idev++; }
  ym-=NStripe(idev-1);
  return idev-1;
}
__device__ inline int Vrefl(const int iz, const int incell=0) {
  return (iz+Nv)%Nv;
  /*if(iz>=Nv) return Nv+Nv-iz-1-incell;*/
  //return iz;
}

struct __align__(16) ftype8 { ftype4 u, v; };
//extern __shared__ ftypr2 shared_fld[2][7][Nv];
//extern __shared__ ftype2 shared_fld[(FTYPESIZE*Nv*28>0xc000)?7:14][Nv];
extern __shared__ ftype2 shared_fld[SHARED_SIZE][Nv];

struct __align__(16) CoffStruct{
  float deps,depsXX,depsYY,depsZZ;
  float coffAnisoXY,coffAnisoXZ,coffAnisoYX,coffAnisoYZ,coffAnisoZX,coffAnisoZY;
  float gyrationXX,gyrationXY,gyrationXZ;
  float gyrationYX,gyrationYY,gyrationYZ;
  float gyrationZX,gyrationZY,gyrationZZ;
  bool isDisp;

  void set_eps(double eps, bool disp=0, double gx=0, double gy=0, double gz=0, double w=shotpoint.w) { 
    deps = 1./eps; isDisp=disp;
    double epsX = eps, epsY=eps, epsZ=eps;
    depsXX=deps; depsYY=deps; depsZZ=deps;
    gx*= w*dt; gy*= w*dt; gz*= w*dt;
    double D = 4*epsX*epsY*epsZ+epsX*gx*gx+epsY*gy*gy+epsZ*gz*gz;
    depsXX      =1.0/D*(gx*gx+4*epsY*epsZ); coffAnisoXY =1.0/D*(gx*gy+2*epsZ*gz)  ; coffAnisoXZ =1.0/D*(gx*gy-2*epsY*gy);
    coffAnisoYX =1.0/D*(gx*gy-2*epsZ*gz)  ; depsYY      =1.0/D*(gy*gy+4*epsX*epsZ); coffAnisoYZ =1.0/D*(gy*gz+2*epsX*gx);
    coffAnisoZX =1.0/D*(gx*gz+2*epsY*gy)  ; coffAnisoZY =1.0/D*(gy*gz-2*epsX*gx)  ; depsZZ      =1.0/D*(gz*gz+4*epsX*epsY);

    gyrationXX =1.0/D*( 4*epsX*epsY*epsZ+epsX*gx*gx-epsY*gy*gy-epsZ*gz*gz); gyrationXY=1.0/D*( 4*epsY*epsZ*gz+2*epsY*gx*gy)                      ; gyrationXZ=1.0/D*(-4*epsY*epsZ*gy+2*epsZ*gx*gz);
    gyrationYX =1.0/D*(-4*epsX*epsZ*gz+2*epsX*gx*gy)                      ; gyrationYY=1.0/D*( 4*epsX*epsY*epsZ+epsY*gy*gy-epsZ*gz*gz-epsX*gx*gx); gyrationYZ=1.0/D*( 4*epsX*epsZ*gx+2*epsZ*gy*gz);
    gyrationZX =1.0/D*( 4*epsX*epsY*gy+2*epsX*gx*gz)                      ; gyrationZY=1.0/D*(-4*epsX*epsY*gx+2*epsY*gy*gz)                      ; gyrationZZ=1.0/D*( 4*epsX*epsY*epsZ+epsZ*gz*gz-epsX*gx*gx-epsY*gy*gy);
    
    /*depsXX      = deps; coffAnisoXY =0   ; coffAnisoXZ =0;
    coffAnisoYX = 0   ; depsYY      =deps; coffAnisoYZ =0;
    coffAnisoZX = 0   ; coffAnisoZY =0   ; depsZZ      =deps;

    gyrationXX =1; gyrationXY=0; gyrationXZ=0;
    gyrationYX =0; gyrationYY=1; gyrationYZ=0;
    gyrationZX =0; gyrationZY=0; gyrationZZ=1;*/
  }
};
struct DispStruct;

const int Nmats=5;
extern __device__ __constant__ CoffStruct coffs[Nmats];
extern __device__ __constant__ DispStruct DispCoffs[Nmats];
extern cudaArray* index_texArray;
//#define TEXCOFFE(x,y,z) coffs[tex3D(index_tex, z,(y+Na*NDT)%(Na*NDT),x)].deps
//#define TEXCOFFE(x,y,z) dtdrd24
#define TEXCOFFS(nind,xt,yt,z,I,h)  ;
#define TEXCOFFTx(nind,xt,yt,z,I,h) ArrcoffT[nind] = 1;
#define TEXCOFFTy(nind,xt,yt,z,I,h) ArrcoffT[nind] = 1;
#define TEXCOFFTz(nind,xt,yt,z,I,h) ArrcoffT[nind] = 1;
#ifdef COFFS_DEFAULT
#define TEXCOFFVx(nind,xt,yt,z,I,h) index = 0; ArrcoffV[nind] = 1;
#define TEXCOFFVy(nind,xt,yt,z,I,h) index = 0; ArrcoffV[nind] = 1;
#define TEXCOFFVz(nind,xt,yt,z,I,h) index = 0; ArrcoffV[nind] = 1;
#define ISDISP(xt,yt,z) 0
#define TEXCOFFDISP(xt,yt,z) DispCoffs[0]
#else//COFFS not DEFAULT
#ifdef USE_TEX_2D
extern texture<char, cudaTextureType2D> index_tex;
#define TEXCOFFVx(nind,xt,yt,z,I,h) index = tex2D(index_tex, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt)); ArrcoffV[nind] = coffs[index].depsXX; AnisoE[nind] = coffs[index];
#define TEXCOFFVy(nind,xt,yt,z,I,h) index = tex2D(index_tex, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt)); ArrcoffV[nind] = coffs[index].depsYY; AnisoE[nind] = coffs[index];
#define TEXCOFFVz(nind,xt,yt,z,I,h) index = tex2D(index_tex, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt)); ArrcoffV[nind] = coffs[index].depsZZ; AnisoE[nind] = coffs[index];
#define ISDISP(xt,yt,z) coffs[tex2D(index_tex, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt))].isDisp
#define TEXCOFFDISP(xt,yt,z) DispCoffs[tex2D(index_tex, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt))]
#else//not USE_TEX_2D
extern texture<char, cudaTextureType3D> index_tex;
#define TEXCOFFVx(nind,xt,yt,z,I,h) index = tex3D(index_tex, z, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt)); ArrcoffV[nind] = 1;//coffs[index].depsXX; AnisoE[nind] = coffs[index];
#define TEXCOFFVy(nind,xt,yt,z,I,h) index = tex3D(index_tex, z, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt)); ArrcoffV[nind] = 1;//coffs[index].depsYY; AnisoE[nind] = coffs[index];
#define TEXCOFFVz(nind,xt,yt,z,I,h) index = tex3D(index_tex, z, (yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt)); ArrcoffV[nind] = 1;//coffs[index].depsZZ; AnisoE[nind] = coffs[index];
#define ISDISP(xt,yt,z) coffs[tex3D(index_tex, z,(yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt))].isDisp
#define TEXCOFFDISP(xt,yt,z) DispCoffs[tex3D(index_tex, z,(yt+iy*2*NDT+Na*2*NDT)%(Na*2*NDT), GLOBAL(xt))]
#endif//USE_TEX_2D
#endif//COFFS_DEFAULT

#ifdef BLOCH_BND_Y
#define tshift_coeff 0
#else
#define tshift_coeff Ntime
#endif
#ifdef TEST_RATE
#define BLOCK_SPACING TEST_RATE
#else
#define BLOCK_SPACING 1
#endif

#define REG_DEC(EVENTYPE) \
  const int iy=(y0+BLOCK_SPACING*blockIdx.x)%Na;\
  const int tshift=tshift_coeff*pars.iStep;\
  const bool inPMLv = (threadIdx.x<Npmlz);\
  const bool inDisp = (threadIdx.x>=dispReg::vL && threadIdx.x<dispReg::vR);\
  const int iz=get_iz(threadIdx.x); const int pml_iz=threadIdx.x, Kpml_iz=2*threadIdx.x;\
  /*const int izP0=iz, izP1 = (iz+1)%Nv, izP2 = (iz+2)%Nv, izM1 = (iz-1+Nv)%Nv, izM2 = (iz-2+Nv)%Nv;*/\
  const int izP0=iz, izP0m=iz, izP1m = Vrefl(iz+1,0), izP2m = Vrefl(iz+2,0), izM1m = Vrefl(iz-1,0), izM2m = Vrefl(iz-2,0);\
  const int          izP0c=iz, izP1c = Vrefl(iz+1,1), izP2c = Vrefl(iz+2,1), izM1c = Vrefl(iz-1,1), izM2c = Vrefl(iz-2,1);\
  const int izdisp=iz-dispReg::vL;\
  const int Kpml_iy=get_pml_iy(iy)*NDT*2; int Kpml_ix=0;\
  int it=t0; ftype difx[100],dify[100],difz[100]; ftype zerov=0.;\
  register ftype coffV = dtdrd24, coffT=dtdrd24, coffS=0;\
  register ftype ArrcoffV[100], ArrcoffT[100], ArrcoffS[100]; register CoffStruct AnisoE[100];\
  char index;\
  ftype2 regPml; ftype regPml2; \
  register ftype2 reg_fldV[250], reg_fldS[250]; register ftype reg_R;\
  const int iy_p0=iy, iy_p1=iy+1, iy_p2=iy+2, iy_p3=iy+3;\
  const int iy_m1=iy-1, iy_m2=iy-2;\
  const int dStepT=1, dStepX=1, dStepRag=Na, dStepRagPML=Npmly; \
      ftype src0x,src1x,src2x,src3x;\
      ftype src0y,src1y,src2y,src3y;\
      ftype src0z,src1z,src2z,src3z;\
      bool upd_inSF, neigh_inSF;\
  ftype rot;\
\
  int glob_ix = (ix+pars.GPUx0+NS-pars.wleft)%NS+pars.wleft;\
  int ymC=0,ymM=0,ymP=0;\
  const int idevC=get_idev(iy  ,ymC); \
  const int idevM=get_idev(iy-1,ymM); \
  const int idevP=get_idev(iy+1,ymP); \
  int y_tmp=0; const int curDev=get_idev(y0, y_tmp); \
  const int dStepRagC=NStripe(idevC);\
  const int dStepRagM=NStripe(idevM);\
  const int dStepRagP=NStripe(idevP);\
  DiamondRag      * __restrict__ RAG0       = &pars.rags[curDev][iy  -ymC];\
  DiamondRag      * __restrict__ RAGcc      = RAG0+ ix           *dStepRagC;\
  DiamondRag      * __restrict__ RAGmc      = RAG0+((ix-1+Ns)%Ns)*dStepRagC;\

#define PTR_DEC \
  const int ixm=(ix-1+Ns)%Ns, ixp=(ix+1)%Ns;\
                                 RAGcc      = RAG0+ix *dStepRagC;\
  DiamondRag      * __restrict__ RAGcm      = RAGcc-1;\
  DiamondRag      * __restrict__ RAGcp      = RAGcc+1;\
                                 RAGmc      = RAG0+ixm*dStepRagC;\
  DiamondRag      * __restrict__ RAGmm      = RAGmc-1;\
  DiamondRag      * __restrict__ RAGmp      = RAGmc+1;\
  DiamondRag      * __restrict__ RAGpc      = RAG0+ixp*dStepRagC;\
  DiamondRag      * __restrict__ RAGpm      = RAGpc-1;\
  DiamondRag      * __restrict__ RAGpp      = RAGpc+1;\
  DiamondRagPML   * __restrict__ ApmlRAGcc  = &pars.ragsPMLa[ix *Npmly+get_pml_iy(iy)];\
  DiamondRagPML   * __restrict__ ApmlRAGmc  = &pars.ragsPMLa[ixm*Npmly+get_pml_iy(iy)];\
  DiamondRagPML   * __restrict__ ApmlRAGpc  = &pars.ragsPMLa[ixp*Npmly+get_pml_iy(iy)];\
  DiamondRagPML   * __restrict__ SpmlRAGcc;/*  = &pars.ragsPMLs[idevC][((ix  <Npmlx/2)? ix   :(ix  -Ns+Npmlx))*dStepRagC   +iy-ymC];*/\
  DiamondRagPML   * __restrict__ SpmlRAGmc;/*  = &pars.ragsPMLs[idevC][((ix-1<Npmlx/2)?(ix-1):(ix-1-Ns+Npmlx))*dStepRagC   +iy-ymC];*/\
  DiamondRagPML   * __restrict__ SpmlRAGpc;/*  = &pars.ragsPMLs[idevC][((ix+1<Npmlx/2)?(ix+1):(ix+1-Ns+Npmlx))*dStepRagC   +iy-ymC];*/\
  if(ix  <Npmlx/2) SpmlRAGcc  = &pars.ragsPMLsL[idevC][  ix              *dStepRagC   +iy-ymC];\
  else             SpmlRAGcc  = &pars.ragsPMLsR[idevC][ (ix  -Ns+Npmlx/2)*dStepRagC   +iy-ymC];\
  if(ix-1<Npmlx/2) SpmlRAGmc  = &pars.ragsPMLsL[idevC][ (ix-1           )*dStepRagC   +iy-ymC];\
  else             SpmlRAGmc  = &pars.ragsPMLsR[idevC][ (ix-1-Ns+Npmlx/2)*dStepRagC   +iy-ymC];\
  if(ix+1<Npmlx/2) SpmlRAGpc  = &pars.ragsPMLsL[idevC][ (ix+1           )*dStepRagC   +iy-ymC];\
  else             SpmlRAGpc  = &pars.ragsPMLsR[idevC][((ix+1-Ns+Npmlx/2)%(Npmlx/2))*dStepRagC   +iy-ymC];\
  DiamondRagDisp  * __restrict__ rdispcc   = &pars.ragsDisp[idevC][ (ix-dispReg::sL                   )*dStepRagC   +iy-ymC];\
  DiamondRagDisp  * __restrict__ rdisppc   = &pars.ragsDisp[idevC][((ix+1-dispReg::sL)%(dispReg::sR-dispReg::sL))*dStepRagC   +iy-ymC];\

#define I01 1
#define I02 2
#define I03 3
#define I04 4
#define I05 5
#define I06 6
#define I07 7
#define I08 8
#define I09 9
#define I10 10
#define I11 11
#define I12 12
#define I13 13
#define I14 14
#define I15 15
#define I16 16
#define I17 17
#define I18 18
#define I19 19
#define I20 20
#define I21 21
#define I22 22
#define I23 23
#define I24 24
#define I25 25
#define I26 26
#define I27 27
#define I28 28
#define I29 29
#define I30 30
#define I31 31                                                                                                                                                                                                                              
#define I32 32                                                                                                                                                                                                                              
#define I33 33                                                                                                                                                                                                                              
#define I34 34                                                                                                                                                                                                                              
#define I35 35                                                                                                                                                                                                                              
#define I36 36                                                                                                                                                                                                                              
#define I37 37                                                                                                                                                                                                                              
#define I38 38                                                                                                                                                                                                                              
#define I39 39                                                                                                                                                                                                                              
#define I40 40                                                                                                                                                                                                                              
#define I41 41                                                                                                                                                                                                                              
#define I42 42                                                                                                                                                                                                                              
#define I43 43                                                                                                                                                                                                                              
#define I44 44                                                                                                                                                                                                                              
#define I45 45                                                                                                                                                                                                                              
#define I46 46                                                                                                                                                                                                                              
#define I47 47                                                                                                                                                                                                                              
#define I48 48                                                                                                                                                                                                                              
#define I49 49                                                                                                                                                                                                                              
#define I50 50                                                                                                                                                                                                                              
#define I51 51                                                                                                                                                                                                                              
#define I52 52                                                                                                                                                                                                                              
#define I53 53                                                                                                                                                                                                                              
#define I54 54                                                                                                                                                                                                                              
#define I55 55                                                                                                                                                                                                                              
#define I56 56                                                                                                                                                                                                                              
#define I57 57                                                                                                                                                                                                                              
#define I58 58                                                                                                                                                                                                                              
#define I59 59                                                                                                                                                                                                                              
#define I60 60                                                                                                                                                                                                                              
#define I61 61                                                                                                                                                                                                                              
#define I62 62                                                                                                                                                                                                                              
#define I63 63                                                                                                                                                                                                                              
#define I64 64                                                                                                                                                                                                                              
#define I65 65                                                                                                                                                                                                                              
#define I66 66                                                                                                                                                                                                                              
#define I67 67                                                                                                                                                                                                                              
#define I68 68                                                                                                                                                                                                                              
#define I69 69
#define I70 70
#define I71 71
#define I72 72
#define I73 73
#define I74 74
#define I75 75
#define I76 76
#define I77 77
#define I78 78
#define I79 79
#define I80 80
#define I81 81
#define I82 82
#define I83 83
#define I84 84
#define I85 85
#define I86 86
#define I87 87
#define I88 88
#define I89 89
#define I90 90
#define I91 91
#define I92 92
#define I93 93
#define I94 94
#define I95 95
#define I96 96
#define I97 97
#define I98 98
#define I99 99

#endif//DEFS_H
