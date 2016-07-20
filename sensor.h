#ifndef SENSOR_H
#define SENSOR_H
#include <stdio.h>
#include <vector>
#include <string>
struct Sensor{
  int x,y,z;
  int2 cRag;
  int2 mmPnt, mpPnt, pmPnt, ppPnt, cmPnt, cpPnt, mcPnt, pcPnt, ccPnt;
  std::string dir;
  std::string fld;
  FILE* pFile;
  Sensor(std::string _fld, int _x=0, int _y=0, int _z=0): x(_x), y(_y), z(_z), fld(_fld), dir(*parsHost.dir) {
    cRag=make_int2(0,0);
    mmPnt=make_int2(0,0); mpPnt=make_int2(0,0); pmPnt=make_int2(0,0); ppPnt=make_int2(0,0); cmPnt=make_int2(0,0); cpPnt=make_int2(0,0); mcPnt=make_int2(0,0); pcPnt=make_int2(0,0); ccPnt=make_int2(0,0);
    printf("Set sensor %s at %d %d %d\n", fld.c_str(), x,y,z); 
    char fname[256]; sprintf(fname, "%s/%s-x%05dy%05dz%05d.dat",dir.c_str(),fld.c_str(),x,y,z );
    pFile = fopen(fname, "w");
    if (pFile==NULL) { printf("Error openning file %s\n", fname); exit(-1); }
  }
  int set_rag(int ragx, int ragy){
    int ragfound=0;
    const int Ex_in_rag[9][2] = { {-2,-1}, {-2,+1}, {+0,-1}, {+0,+1}, {+0,+3}, {+2,+1}, {+2,+3}, {+2,+5}, {+4,+3} };
    const int Ey_in_rag[9][2] = { {-3,+0}, {-1,-2}, {-1,+0}, {-1,+2}, {+1,+0}, {+1,+2}, {+1,+4}, {+3,+2}, {+3,+4} };
    const int Ez_in_rag[9][2] = { {-1,-1}, {-1,+1}, {+1,-1}, {+1,+1}, {+1,+3}, {+3,+1}, {+3,+3}, {+3,+5}, {+5,+3} };
    const int Hx_in_rag[9][2] = { {-1,+4}, {-1,+2}, {+1,+4}, {+1,+2}, {+1,+0}, {+3,+2}, {+3,+0}, {+3,-2}, {+5,+0} };
    const int Hy_in_rag[9][2] = { {-2,+3}, {+0,+5}, {+0,+3}, {+0,+1}, {+2,+3}, {+2,+1}, {+2,-1}, {+4,+1}, {+4,-1} };
    const int Hz_in_rag[9][2] = { {-2,+4}, {-2,+2}, {+0,+4}, {+0,+2}, {+0,+0}, {+2,+2}, {+2,+0}, {+2,-2}, {+4,+0} };
    const int Si_in_rag[9][2] = { {-3,+3}, {-1,+5}, {-1,+3}, {-1,+1}, {+1,+3}, {+1,+1}, {+1,-1}, {+3,+1}, {+3,-1} };
    int2 into_shift;
    for(int i=0; i<9; i++) {
      if(fld=="Ex") into_shift=make_int2(Ex_in_rag[i][0],Ex_in_rag[i][1]); else
      if(fld=="Ey") into_shift=make_int2(Ey_in_rag[i][0],Ey_in_rag[i][1]); else
      if(fld=="Ez") into_shift=make_int2(Ez_in_rag[i][0],Ez_in_rag[i][1]); else
      if(fld=="Hx") into_shift=make_int2(Hx_in_rag[i][0],Hx_in_rag[i][1]); else
      if(fld=="Hy") into_shift=make_int2(Hy_in_rag[i][0],Hy_in_rag[i][1]); else
      if(fld=="Hz") into_shift=make_int2(Hz_in_rag[i][0],Hz_in_rag[i][1]);
      int2 x4s = make_int2(ragx*2*NDT+into_shift.x, ragy*2*NDT+into_shift.y);
      if(x*2  ==x4s.x && y*2  ==x4s.y) { ccPnt.x=ragx*Na+ragy; ccPnt.y=i; cRag.x = ragx*Na+ragy; cRag.y = i; ragfound=1; }
      if(x*2-1==x4s.x && y*2  ==x4s.y) { mcPnt.x=ragx*Na+ragy; mcPnt.y=i; ragfound=1; }
      if(x*2+1==x4s.x && y*2  ==x4s.y) { pcPnt.x=ragx*Na+ragy; pcPnt.y=i; ragfound=1; }
      if(x*2  ==x4s.x && y*2-1==x4s.y) { cmPnt.x=ragx*Na+ragy; cmPnt.y=i; ragfound=1; }
      if(x*2  ==x4s.x && y*2+1==x4s.y) { cpPnt.x=ragx*Na+ragy; cpPnt.y=i; ragfound=1; }
      if(x*2-1==x4s.x && y*2-1==x4s.y) { mmPnt.x=ragx*Na+ragy; mmPnt.y=i; ragfound=1; }
      if(x*2+1==x4s.x && y*2-1==x4s.y) { pmPnt.x=ragx*Na+ragy; pmPnt.y=i; ragfound=1; }
      if(x*2-1==x4s.x && y*2+1==x4s.y) { mpPnt.x=ragx*Na+ragy; mpPnt.y=i; ragfound=1; }
      if(x*2+1==x4s.x && y*2+1==x4s.y) { ppPnt.x=ragx*Na+ragy; ppPnt.y=i; ragfound=1; }
    }
    return ragfound;
  }
  void write(int device=0){
    ftype fval;
    //if(device) { CHECK_ERROR( cudaMemcpy(&val, &parsHost.cells[IndexTILED(x,y).x], sizeof(Cell), cudaMemcpyDeviceToHost) ); CHECK_ERROR(cudaDeviceSynchronize()); }
    //DiamondRag* rag = &parsHost.data[nRag];
    DiamondRag* ragMM = &parsHost.data[mmPnt.x]; int idomMM = mmPnt.y;
    DiamondRag* ragPM = &parsHost.data[pmPnt.x]; int idomPM = pmPnt.y;
    DiamondRag* ragMP = &parsHost.data[mpPnt.x]; int idomMP = mpPnt.y;
    DiamondRag* ragPP = &parsHost.data[ppPnt.x]; int idomPP = ppPnt.y;
    DiamondRag* ragCM = &parsHost.data[cmPnt.x]; int idomCM = cmPnt.y;
    DiamondRag* ragCP = &parsHost.data[cpPnt.x]; int idomCP = cpPnt.y;
    DiamondRag* ragMC = &parsHost.data[mcPnt.x]; int idomMC = mcPnt.y;
    DiamondRag* ragPC = &parsHost.data[pcPnt.x]; int idomPC = pcPnt.y;
    DiamondRag* ragCC = &parsHost.data[ccPnt.x]; int idomCC = ccPnt.y;
    const int zM=(z==0)?(z+1):(z-1); const int zP=z; const int zC=z;
    if(fld=="Ex") fval = 0.25*( ragCM->Vi[idomCM].trifld.two[zM].x + ragCM->Vi[idomCM].trifld.two[zP].x + ragCP->Vi[idomCP].trifld.two[zM].x + ragCP->Vi[idomCP].trifld.two[zP].x ); else
    if(fld=="Ey") fval = 0.25*( ragMC->Vi[idomMC].trifld.one[zM]   + ragMC->Vi[idomMC].trifld.one[zP]   + ragPC->Vi[idomPC].trifld.one[zM]   + ragPC->Vi[idomPC].trifld.one[zP]   ); else
    if(fld=="Ez") fval = 0.25*( ragMM->Vi[idomMM].trifld.two[zC].y + ragMP->Vi[idomMP].trifld.two[zC].y + ragPM->Vi[idomPM].trifld.two[zC].y + ragPP->Vi[idomPP].trifld.two[zC].y ); else
    if(fld=="Hx") fval = 0.5 *( ragMC->Si[idomMC].trifld.two[zC].y + ragPC->Si[idomPC].trifld.two[zC].y ); else
    if(fld=="Hy") fval = 0.5 *( ragCM->Si[idomCM].trifld.one[zC]   + ragCP->Si[idomCP].trifld.one[zC]   ); else
    if(fld=="Hz") fval = 0.5 *( ragCC->Si[idomCC].trifld.two[zM].x + ragPC->Si[idomPC].trifld.two[zP].x ); else
    fval=0;

    /*if(fld=="Ex") fval = rag->Vi[idom].trifld.two[z].x; else
    if(fld=="Ey") fval = rag->Vi[idom].trifld.one[z]  ; else
    if(fld=="Ez") fval = rag->Vi[idom].trifld.two[z].y; else
    if(fld=="Hx") fval = rag->Si[idom].trifld.two[z].y; else
    if(fld=="Hy") fval = rag->Si[idom].trifld.one[z]  ; else
    if(fld=="Hz") fval = rag->Si[idom].trifld.two[z].x; else
    fval=0;*/

    fprintf(pFile, "%g\n", fval);
    fflush(pFile);
  }
};

#endif
