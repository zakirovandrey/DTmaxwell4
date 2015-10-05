#ifndef SENSOR_H
#define SENSOR_H
#include <stdio.h>
#include <vector>
#include <string>
struct Sensor{
  int x,y,z;
  std::string dir;
  std::string fld;
  FILE* pFile;
  Sensor(std::string _fld, int _x=0, int _y=0, int _z=0): x(_x), y(_y), z(_z), fld(_fld), dir(*parsHost.dir) {
    printf("Set sensor %s at %d %d %d\n", fld.c_str(), x,y,z); 
    char fname[256]; sprintf(fname, "%s/%s-x%05dy%05dz%05d.dat",dir.c_str(),fld.c_str(),x,y,z );
    pFile = fopen(fname, "w");
    if (pFile==NULL) { printf("Error openning file %s\n", fname); exit(-1); }
  }
  void write(int device=0){/*
    Cell val; ftype fval;
    if(device) { CHECK_ERROR( cudaMemcpy(&val, &parsHost.cells[IndexTILED(x,y).x], sizeof(Cell), cudaMemcpyDeviceToHost) ); CHECK_ERROR(cudaDeviceSynchronize()); }
    else val = parsHost.data[IndexTILED(x,y).x];
    if(fld=="Ex") fval = val.Ex[z]; else
    if(fld=="Ey") fval = val.Ey[z]; else
    if(fld=="Ez") fval = val.Ez[z]; else
    if(fld=="Hx") fval = val.Hx[z]; else
    if(fld=="Hy") fval = val.Hy[z]; else
    if(fld=="Hz") fval = val.Hz[z];
    fprintf(pFile, "%g\n", fval);*/
    fflush(pFile);
  }
};

#endif
