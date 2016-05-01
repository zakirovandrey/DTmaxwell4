const int NDT=3;
#ifdef USE_DOUBLE
typedef double ftype;
#define MPI_FTYPE MPI_DOUBLE
#else
typedef float ftype;
#define MPI_FTYPE MPI_FLOAT
#endif

#define gridNx 600 // Axis Sync
#define gridNz 640 // Axis Vectorization
#define gridNy 300 // Axis Async 

//#define ANISO_TR 2

#define NDev 1
#define GPUDIRECT_RDMA

#define DROP_DATA
//#define USE_AIVLIB_MODEL
//#define MPI_ON
//#define MPI_TEST
//#define TEST_RATE 1
#define NOPMLS
#define USE_WINDOW
#define COFFS_DEFAULT
#define COFFS_IN_DEVICE
//#define TIMERS_ON
//#define SWAP_DATA
//#define CLICK_BOOM
#define SHARED_SIZE 5

#ifndef NS
#define NS 25
const int Np=gridNx/3;
const int GridNx=gridNx;
#else
const int Np=NP;//NS*1;
const int GridNx=3*Np;
#endif//NS
#ifndef NA
const int GridNy=gridNy;
#else 
const int GridNy=3*NA;
#endif//NA
#ifndef NB
#define NB NA
#endif
#ifndef NV
const int GridNz=gridNz;
#else
const int GridNz=NV;
#endif
const int NyBloch=1;//200;
//#define BLOCH_BND_Y

#define USE_TEX_2D
#undef USE_TEX_2D

const int Nzw=128;

const int Npmlx=2*1;//2*24;
const int Npmly=2*0;//24;
const int Npmlz=0*2*16;//128;

const ftype ds=0.005, da=0.005, dv=0.005, dt=0.0024;

extern struct TFSFsrc{
  ftype tStop;
  ftype srcXs, srcXa, srcXv, sphR;
  ftype BoxMs, BoxPs; 
  ftype BoxMa, BoxPa; 
  ftype BoxMv, BoxPv;
  ftype V_max;
  int start;
  //----------------------------------------------//
  ftype wavelength, w,k, Omega, kEnv, theta, phi;
  //-------------------------------------------//
  ftype cSrc;

  void set(const double, const double, const double);
  void check();
} shotpoint;

