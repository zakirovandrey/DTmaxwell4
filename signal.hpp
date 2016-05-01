#ifndef SIGNAL_HPP
#define SIGNAL_HPP
#ifdef MPI_ON
#include <mpi.h>
#endif

#include "lambda_func.hpp"
#define S L7
#include "signal.h"

__constant__ TFSFsrc src;
TFSFsrc shotpoint;

struct PlaneTFSF{
  ftype x,y,z, phase, kx,ky,kz, Ephi,Etheta,Er, Hphi,Htheta,Hr;
  static const ftype A=1.0;
  __device__ PlaneTFSF(const ftype _t, const ftype _x, const ftype _y, const ftype _z) {
    const ftype dxs=dx*0.5,dxa=dy*0.5,dxv=dz*0.5;
    
    x = dxs*(_x); y=dxa*(_y+pars.subnode*Na*2*NDT); z = dxv*_z; ftype t=_t*dt;
    ftype phiz=src.phi;//(_z-1.0)/2*1.0/180.0*M_PI;
    //phiz=pars.inc_ang;//1*1/180.0*M_PI;
    //if(fabs(_z-Nz)<0.1*dz) phiz=0*M_PI/6.; else phiz=M_PI/2.;
    
    ftype kx=src.k*sinf(src.theta)*cosf(phiz), ky=src.k*sinf(src.theta)*sinf(phiz), kz=src.k*cosf(src.theta);
    //ftype delay = fabs(2*Na*NDT*dy*sin(phiz));
    if(dz!=FLT_MAX) phase = src.w*t-kx*(x-src.BoxMs)-ky*(y-src.BoxMa)-kz*(z-src.BoxMv);
    else            phase = src.w*t-kx*(x-src.BoxMs)-ky*(y-src.BoxMa);
    ftype Yc=Na*NasyncNodes*NDT*dy*0.5;
    ftype Zc=Nz*dz*0.5;
    ftype waist = 10000.30;
    ftype gaussEnv = expf(-(y-Yc)*(y-Yc)/(waist*waist));//-(z-Zc)*(z-Zc)/(0.3*Zc*0.3*Zc));
    //if (fabs(gaussEnv)<1.e-4 || z>(src.BoxMv+src.BoxPv)*0.5*10) gaussEnv=0;
    Ephi   =  A*((phase*src.kEnv<0 || phase*src.kEnv>2*M_PI)?0:1)*0.5*(1-cosf(phase*src.kEnv))*sinf(phase)*gaussEnv;
    Htheta = -A*((phase*src.kEnv<0 || phase*src.kEnv>2*M_PI)?0:1)*0.5*(1-cosf(phase*src.kEnv))*sinf(phase)*gaussEnv;
    if(0 && phase*src.kEnv>M_PI){ 
      Ephi   =  A*sinf(phase)*gaussEnv;
      Htheta = -A*sinf(phase)*gaussEnv;
    }
    Etheta=0; Hphi=0; Er=0; Hr=0;
    //if(phase>0) printf("kz=%g phase=%g %g %g %g %g\n", kz, phase, _x,_y,_z,_t);
  }
  __device__ inline ftype getEx() { return -Ephi*sinf(src.phi)+Etheta*cosf(src.theta)*cosf(src.phi)+Er*sinf(src.theta)*cosf(src.phi); }
  __device__ inline ftype getEy() { return  Ephi*cosf(src.phi)+Etheta*cosf(src.theta)*sinf(src.phi)+Er*sinf(src.theta)*sinf(src.phi); }
  __device__ inline ftype getEz() { return -Etheta*sinf(src.theta)+Er*cosf(src.theta); }
  __device__ inline ftype getHx() { return -Hphi*sinf(src.phi)+Htheta*cosf(src.theta)*cosf(src.phi)+Hr*sinf(src.theta)*cosf(src.phi); }
  __device__ inline ftype getHy() { return  Hphi*cosf(src.phi)+Htheta*cosf(src.theta)*sinf(src.phi)+Hr*sinf(src.theta)*sinf(src.phi); }
  __device__ inline ftype getHz() { return -Htheta*sinf(src.theta)+Hr*cosf(src.theta); }
};
struct DipoleTFSF{
  float x,y,z, phase, kx,ky,kz, Ephi,Etheta,Er, Hphi,Htheta,Hr;
  float3 E, B;
  static const float A=1.0;
  __device__ DipoleTFSF(const float _t, const float _x, const float _y, const float _z) {
    const float dxs=dx*0.5,dxa=dy*0.5,dxv=dz*0.5;
    
    x = dxs*(_x); y=dxa*_y; z = dxv*_z; float t=_t*dt;
    
    ftype Xdip=(src.BoxMs+src.BoxPs)*0.5;
    ftype Ydip=(src.BoxMa+src.BoxPa)*0.5;
    ftype Zdip=(src.BoxMv+src.BoxPv)*0.5;
    float3 n = make_float3(x-Xdip,y-Ydip,z-Zdip);
    ftype R = length(n);
    const ftype c=src.cSrc;
    phase = src.w*t-src.w/c*R;
    float3 d  = make_float3(    sin(phase)*0.5*(1-cos(phase*src.kEnv)),0,0);
    float3 ds = make_float3(src.w*  cos(phase)*0.5-0.5*src.w*cos(phase)*cos(phase*src.kEnv)+0.5*sin(phase)*src.w*src.kEnv*sin(phase*src.kEnv),0,0);
    float3 dss= make_float3(-src.w*src.w*sin(phase)*0.5+0.5*src.w*src.w*sin(phase)*cos(phase*src.kEnv)+0.5*src.w*cos(phase)*src.w*src.kEnv*sin(phase*src.kEnv)+0.5*src.w*cos(phase)*src.w*src.kEnv*sin(phase*src.kEnv)+0.5*sin(phase)*src.w*src.kEnv*src.w*src.kEnv*cos(phase*src.kEnv) ,0,0);
    
    d*=   (phase*src.kEnv<0 || phase*src.kEnv>2*M_PI)?0:1;
    ds*=  (phase*src.kEnv<0 || phase*src.kEnv>2*M_PI)?0:1;
    dss*= (phase*src.kEnv<0 || phase*src.kEnv>2*M_PI)?0:1;

    //E = (3*n*dot(n,d)-d)/(R*R*R) + (3*n*dot(n,ds)-ds)/(c*R*R) + (n*dot(n,dss)-dss)/(c*c*R);
    //B = cross(n, E+d/(R*R*R));
    n = normalize(n);
    E = c*(cross(cross(dss,n),n)/(c*c*R) + (3*dot(ds,n)*n-ds)/(c*R*R) + (3*dot(d,n)*n-d)/(R*R*R));
    B = cross(dss,n)/(c*c*R) + cross(ds,n)/(c*R*R);
  }
  __device__ inline ftype getEx() { return E.x; }
  __device__ inline ftype getEy() { return E.y; }
  __device__ inline ftype getEz() { return E.z; }
  __device__ inline ftype getHx() { return B.x; }
  __device__ inline ftype getHy() { return B.y; }
  __device__ inline ftype getHz() { return B.z; }
};

void TFSFsrc::set(const double _Vp, const double _Vs, const double _Rho) {
    double c=_Vp;
    V_max=c;
    cSrc = c;
    wavelength=0.1;//2*0.8940637561158599;
    k=2*M_PI/wavelength;
    w=c*k;
    Omega = 0.5*w;
    Omega = 1.0*w;
    kEnv=Omega/w;
    theta=0.2*M_PI/2.; phi=20/180.0*M_PI;//M_PI/30.;
    theta=0; phi=M_PI/2;
}
void TFSFsrc::check(){
    int node=0, Nprocs=1;
    #ifdef MPI_ON
    MPI_Comm_rank (MPI_COMM_WORLD, &node);
    MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
    #endif
    if (node!=0) return;
    printf("TF/SF source: Box(s,v,a) |%g-%g|x|%g-%g|x|%g-%g|\n", BoxMs,BoxPs, BoxMv,BoxPv, BoxMa,BoxPa);
    printf("TF/SF source: Shotpoint(s,v,a) = %g,%g,%g\n", srcXs, srcXv, srcXv);
    printf("TF/SF source: wavelength = %g, kEnv=%g, tStop=%g\n", wavelength,kEnv,tStop);
    printf("TF/SF stops after %d steps\n", int(tStop/dt));
    if(BoxMs<0        ) printf("BoxMs<0: incorrect results are possible\n");
    if(BoxMa<0        ) printf("BoxMa<0: incorrect results are possible\n");
    if(BoxMv<0        ) printf("BoxMv<0: incorrect results are possible\n");
    if(BoxPs>Np*NDT*ds            ) printf("BoxPs>gridX: incorrect results are possible\n");
    if(BoxPa>Na*NasyncNodes*NDT*da) printf("BoxPa>gridY: incorrect results are possible\n");
    if(BoxPv>Nv*dv                ) printf("BoxPv>gridZ: incorrect results are possible\n");
}

__device__ __noinline__ ftype SrcTFSF_Vx(const int s, const int v, const int a,  const ftype tt){
  /*Plane*/DipoleTFSF src(tt, s,a,v); return src.getEx();
}
__device__ __noinline__ ftype SrcTFSF_Vy(const int s, const int v, const int a,  const ftype tt){
  /*Plane*/DipoleTFSF src(tt, s,a,v); return src.getEy();
}
__device__ __noinline__ ftype SrcTFSF_Vz(const int s, const int v, const int a,  const ftype tt){
  /*Plane*/DipoleTFSF src(tt, s,a,v); return src.getEz();
}
__device__ __noinline__ ftype SrcTFSF_Tx(const int s, const int v, const int a,  const ftype tt){
  /*Plane*/DipoleTFSF src(tt, s,a,v); return src.getHx();
}
__device__ __noinline__ ftype SrcTFSF_Ty(const int s, const int v, const int a,  const ftype tt){
  /*Plane*/DipoleTFSF src(tt, s,a,v); return src.getHy();
}
__device__ __noinline__ ftype SrcTFSF_Tz(const int s, const int v, const int a,  const ftype tt){
  /*Plane*/DipoleTFSF src(tt, s,a,v); return src.getHz();
}
__device__ __noinline__ bool inSF(const int _s, const int _a, const int _v) { 
  ftype s = _s*0.5*dx, a=(_a+pars.subnode*Na*2*NDT)*0.5*dy, v=_v*0.5*dz;
  if(dz==FLT_MAX) v=FLT_MAX/2; 
  //return !(s>tfsfSm && s<tfsfSp && a>tfsfAm && a<tfsfAp && v>tfsfVm && v<tfsfVp); 
  //bool into = (s>src.BoxMs && s<src.BoxPs && a>src.BoxMa && a<src.BoxPa && v>src.BoxMv && v<src.BoxPv); 
  bool into = (s-src.srcXs)*(s-src.srcXs)+(a-src.srcXa)*(a-src.srcXa)+(v-src.srcXv)*(v-src.srcXv)<=src.sphR*src.sphR; 
  return into;
}
__device__ ftype  SrcTFSF_Sx(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ ftype  SrcTFSF_Sy(const int s, const int v, const int a,  const ftype tt) {return 0;};
__device__ ftype  SrcTFSF_Sz(const int s, const int v, const int a,  const ftype tt) {return 0;};


#undef S
#endif
