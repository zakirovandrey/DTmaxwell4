#include "params.h"
#include <vector>
using namespace std;
const int Md=MD;
struct LorenzDisp {
  double a0,a1,b0,b1,b2;
  LorenzDisp() : a0(0.),a1(0.),b0(0.),b1(0.),b2(0.) {};
  void set(double Deps=1., double wp=0., double gp=0., double gp_=0.) {
    a0 = Deps*wp*wp;
    a1 = Deps*gp_;
    b0 = wp*wp;
    b1 = 2*gp;
    b2 = 1;
  }
  void setDrude(double Deps=1., double wp=0., double gp=0., double gp_=0.) {
    a0 = Deps*wp*wp;
    a1 = Deps*gp_;
    b0 = 0.0;
    b1 = 2*gp;
    b2 = 1;
  }
};

struct DispStruct {
  ftype kEJ[3][Md][2]; 
  ftype kE[3][3];
  ftype kJ[3][Md][5];
  int isSet;
  double epsinf, sigma;
//  vector<LorenzDisp> Ld;

//  DispStruct(): isSet(0) {}
//  void setNondisp(MatStruct& mat) { set(mat.eps, vector<LorenzDisp>(), 0, 0.); isSet=0; }
  void setNondisp() { }
/*  void set(DispStruct& disp1, DispStruct& disp2, float f1, float f2){
    vector<LorenzDisp> Ld_t = disp1.Ld;
    Ld_t.insert( Ld_t.end(), disp2.Ld.begin(), disp2.Ld.end() );
    for(int i=0; i<Ld_t.size(); i++) { float f; if(i<disp1.Ld.size()) f=f1; else f=f2; Ld_t[i].a0*=f; Ld_t[i].a1*=f; }
    set(disp1.epsinf*f1+disp2.epsinf*f2, Ld_t, Ld_t.size(), disp1.sigma*f1+disp2.sigma*f2);
  }*/
  void set(double einf, vector<LorenzDisp> _Ld, int mD, double sm) {
    isSet = 1;

    epsinf=einf; sigma=sm; 

    vector<LorenzDisp> Ld = _Ld;

    double alpha[Md];
    double ksi[Md]; 
    double zetp[Md];
    double zetm[Md];
    double zet[Md];
    double zetm_sum=0., zetp_sum=0., zet_sum=0.;
    for(int i=0;i<mD;i++) {
      alpha[i] = (4-2*Ld[i].b0*dt*dt)/(Ld[i].b1*dt+2);
      ksi[i]   = (Ld[i].b1*dt-2)/(Ld[i].b1*dt+2);
      zetp[i]  = (+Ld[i].a0*dt*dt+2*Ld[i].a1*dt)/(Ld[i].b1*dt+2);
      zetm[i]  = (-Ld[i].a0*dt*dt+2*Ld[i].a1*dt)/(Ld[i].b1*dt+2);
      zet[i]   = -(4*Ld[i].a1*dt)/(Ld[i].b1*dt+2);
      
      zetm_sum+=zetm[i]; zetp_sum+=zetp[i]; zet_sum+=zet[i];
    }
    for(int i=mD;i<Md;i++) {
      alpha[i] = 2;
      ksi[i]   = -1;
      zetp[i]  = 0.0;
      zetm[i]  = 0.0;
      zet[i]   = 0;
      zetm_sum+=zetm[i]; zetp_sum+=zetp[i]; zet_sum+=zet[i];
    }
 
    double C1 = ( -(zetm_sum)                     ) / ( 2*epsinf + sigma*dt + (zetp_sum) );
    double C2 = ( 2*epsinf - sigma*dt - (zet_sum) ) / ( 2*epsinf + sigma*dt + (zetp_sum) );
    double C3 = ( 2*dt                            ) / ( 2*epsinf + sigma*dt + (zetp_sum) );
    for(int xc=0; xc<3; xc++) {
      for(int i=0; i<Md; i++) {
        kJ[xc][i][0] = alpha[i]; kJ[xc][i][1] = ksi[i]; kJ[xc][i][2] = zetp[i]/dt; kJ[xc][i][3] = zetm[i]/dt; kJ[xc][i][4] = zet[i]/dt; 
        kEJ[xc][i][0] = C3*0.5*(1+alpha[i]); kEJ[xc][i][1] = C3*0.5*ksi[i];
      }
      kE[xc][0] = C1; kE[xc][1] = C2; kE[xc][2] = C3/dt;
    }
  }
};

__device__ inline void countDisp (ftype& Ep, ftype& Em, ftype* Jp, ftype* Jm, ftype& rotdt, DispStruct& CfArr, int xc, int md=Md) {
  ftype Etmp = Ep;
  ftype sum=0.;
  for(int i=0; i<md; i++) sum+= CfArr.kEJ[xc][i][0]*Jp[i] + CfArr.kEJ[xc][i][1]*Jm[i];  // в статье слагаемые также!!!
  Ep = CfArr.kE[xc][0]*Em + CfArr.kE[xc][1]*Ep + CfArr.kE[xc][2]*rotdt - sum;  //тут rotor*dt = (Dnew-Dold) !!!

  ftype Jtmp[MD];
  for(int i=0; i<md; i++) {
    Jtmp[i] = Jp[i];
    Jp[i] = CfArr.kJ[xc][i][0]*Jp[i] + CfArr.kJ[xc][i][1]*Jm[i] + CfArr.kJ[xc][i][2]*Ep + CfArr.kJ[xc][i][3]*Em + CfArr.kJ[xc][i][4]*Etmp;
  }

  Em = Etmp;
  for(int i=0; i<md; i++) Jm[i] = Jtmp[i];
}

