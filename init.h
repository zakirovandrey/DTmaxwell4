#ifndef _INIT_H
#define _INIT_H
#include "params.h"

void init(){
//  set_data<<<dim3(Ns,Na),Nv>>>();
  cudaDeviceSynchronize();
  CHECK_ERROR( cudaGetLastError() );
}

void drop(){
/*  host_cells = new Cell[Nx*Ny];
  CHECK_ERROR( cudaMemcpy( host_cells, parsHost.cells, Nx*Ny*sizeof(Cell), cudaMemcpyDeviceToHost ) );
  FILE* file = fopen("res.arr", "w");
  if(file==NULL) perror("Cannot open file res.arr\n");
  int nil=0, dim=3, szT=4;
  fwrite(&nil, 4, 1, file);
  fwrite(&dim, 4, 1, file);
  fwrite(&szT, 4, 1, file);
  fwrite(&Nx , 4, 1, file);
  fwrite(&Ny , 4, 1, file);
  fwrite(&Nz , 4, 1, file);
//  for(int i=0; i<Nx*Ny; i++) fwrite(host_cells[i].Ex, sizeof(float), Nz, file);
  fclose(file);*/
}

#endif //_INIT_H
