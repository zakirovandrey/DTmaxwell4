#include <sys/stat.h>
#include <unistd.h>
struct SeismoDrops {
  int node,subnode, Nprocs;
  std::string* dir;
  int FldDrop[6]; //Ex,Ey,Ez,Hx,Hy,Hz
  std::string* fnameprefix[6];
  #ifdef MPI_ON
  MPI_File* file[6];
  MPI_Offset head_offset;
  #endif
  void init() {
    node=0; subnode=0; Nprocs=1;
    char* sfld[] = {"Ex","Ey","Ez","Hx","Hy","Hz"};
    for(int ifld=0; ifld<6; ifld++) {
      char pre[256];
      sprintf(pre, "%s/rag%s", dir->c_str(), sfld[ifld]);
      fnameprefix[ifld] = new std::string(pre);
    }
    #ifdef MPI_ON
    MPI_Comm_rank (MPI_COMM_WORLD, &node);
    MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
    subnode = node%NasyncNodes;
    node /= NasyncNodes; Nprocs /= NasyncNodes;
    #endif
    if(node==0 && subnode==0) {
      int check_exist=0; int it=0;
      while(check_exist<6) {
        check_exist=0;
        for(int ifld=0; ifld<6; ifld++) {
          char fname[256]; sprintf(fname, "%s_%05d.arr", fnameprefix[ifld]->c_str(), it);
          struct stat buffer; 
          if(stat(fname, &buffer)==0) remove(fname);
          else check_exist++;
        }
        it++;
      }
    }
    #ifdef MPI_ON
    for(int ifld=0; ifld<6; ifld++) file[ifld] = new MPI_File; 
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    FldDrop[0]=1; FldDrop[1]=0; FldDrop[2]=0;
    FldDrop[3]=0; FldDrop[4]=0; FldDrop[5]=0;
  }
  void open(const int it){
    #ifdef MPI_ON
    for(int ifld=0; ifld<6; ifld++) {
      if(FldDrop[ifld]==0) continue;
      MPI_Status status;
      char fname[256]; sprintf(fname, "%s_%05d.arr", fnameprefix[ifld]->c_str(), it);
      MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE|MPI_MODE_EXCL|MPI_MODE_WRONLY, MPI_INFO_NULL, file[ifld]);
      //MPI_File_set_atomicity(*file[ifld], true);
      int zero=0, twelve = 12, dim = 3, sizeofT = sizeof(ftype);
      head_offset=0;
      int regNx=Np, regNy=Na*NasyncNodes, regNz=Nv;
      if(node==0 && subnode==0) MPI_File_seek(*file[ifld], 0, MPI_SEEK_SET);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &twelve , 1, MPI_INT, &status); head_offset+= sizeof(twelve);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &zero   , 1, MPI_INT, &status); head_offset+= sizeof(zero);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &zero   , 1, MPI_INT, &status); head_offset+= sizeof(zero);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &zero   , 1, MPI_INT, &status); head_offset+= sizeof(zero);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &dim    , 1, MPI_INT, &status); head_offset+= sizeof(dim);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &sizeofT, 1, MPI_INT, &status); head_offset+= sizeof(sizeofT);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &regNz  , 1, MPI_INT, &status); head_offset+= sizeof(regNz);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &regNy  , 1, MPI_INT, &status); head_offset+= sizeof(regNy);
      if(node==0 && subnode==0) MPI_File_write(*file[ifld], &regNx  , 1, MPI_INT, &status); head_offset+= sizeof(regNx);
    }
    #endif
  }
  void close(){
    #ifdef MPI_ON
    for(int ifld=0; ifld<6; ifld++) if(FldDrop[ifld]!=0) MPI_File_close(file[ifld]);
    #endif
  }
  void drop(const int xstart, const int xend, const DiamondRag* data, const int it) {
    #ifdef DROP_DATA
    DEBUG_MPI(("drop data node=%d subnode=%d it=%d\n",node,subnode,it));
    for(int ifld=0; ifld<6; ifld++) {
      if(FldDrop[ifld]==0) continue;
      size_t ptr_shift=0;
      char fname[256]; sprintf(fname, "%s_%05d.arr", fnameprefix[ifld]->c_str(), it);
      #ifndef MPI_ON
      FILE* pFile; pFile = fopen(fname,"w");
      int zero=0, twelve = 12, dim = 3, sizeofT = sizeof(ftype);  
      fwrite(&twelve , sizeof(int  ), 1, pFile);  //size of comment
      fwrite(&zero   , sizeof(int  ), 1, pFile);    // comment
      fwrite(&zero   , sizeof(int  ), 1, pFile);    //comment
      fwrite(&zero   , sizeof(int  ), 1, pFile);    //comment
      fwrite(&dim    , sizeof(int  ), 1, pFile);     //dim = 
      fwrite(&sizeofT, sizeof(int  ), 1, pFile); //data size
      fwrite(&Nv     , sizeof(int  ), 1, pFile);
      fwrite(&Na     , sizeof(int  ), 1, pFile);
      fwrite(&Np     , sizeof(int  ), 1, pFile);
      //printf("saving %s\n",fname);
      for(int x=0; x<Np; x++) for(int y=0; y<Na; y++) for(int z=0; z<Nv; z++) {
        ftype val = 0;
        switch(ifld){
          case 0: val = data[ptr_shift+x*Na+y].Vi[0].trifld.two[z].x; break;
          case 1: val = data[ptr_shift+x*Na+y].Vi[0].trifld.one[z];   break;
          case 2: val = data[ptr_shift+x*Na+y].Vi[0].trifld.two[z].y; break;
          case 3: val = data[ptr_shift+x*Na+y].Si[0].trifld.two[z].y; break;
          case 4: val = data[ptr_shift+x*Na+y].Si[0].trifld.one[z];   break;
          case 5: val = data[ptr_shift+x*Na+y].Si[0].trifld.two[z].x; break;
          default: break;
        }
        fwrite(&val, sizeof(ftype), 1, pFile);
      }
      fclose(pFile);
      #else // MPI_ON
      MPI_Status status;
      int regNx=Np, regNy=Na*NasyncNodes, regNz=Nv;
      for(int x=xstart; x<xend; x++) {
        int node_shift=x; 
        for(int inode=0; inode<node; inode++) node_shift+= mapNodeSize[inode]; node_shift-= Ns*node;
        ptr_shift = node_shift*Na;
        MPI_Offset offset    = head_offset+node_shift*regNy*regNz*sizeof(ftype);
        offset   += subnode*Na*regNz*sizeof(ftype);
        MPI_File_seek(*file[ifld], offset, MPI_SEEK_SET);
        //printf("node=%d subnode=%d x=%d offset=%ld\n", node,subnode, x, offset);
        for(int y=0; y<Na; y++) {
          ftype val[Nv];
          switch(ifld){
            case 0: for(int z=0; z<Nv; z++) val[z] = data[ptr_shift+y].Vi[0].trifld.two[z].x; break;
            case 1: for(int z=0; z<Nv; z++) val[z] = data[ptr_shift+y].Vi[0].trifld.one[z];   break;
            case 2: for(int z=0; z<Nv; z++) val[z] = data[ptr_shift+y].Vi[0].trifld.two[z].y; break;
            case 3: for(int z=0; z<Nv; z++) val[z] = data[ptr_shift+y].Si[0].trifld.two[z].y; break;
            case 4: for(int z=0; z<Nv; z++) val[z] = data[ptr_shift+y].Si[0].trifld.one[z];   break;
            case 5: for(int z=0; z<Nv; z++) val[z] = data[ptr_shift+y].Si[0].trifld.two[z].x; break;
            default: break;
          }
          MPI_File_write(*file[ifld], val, Nv, MPI_FTYPE, &status);
        }
      }
      #endif//MPI_ON
    }
    DEBUG_MPI(("end of drop data node=%d subnode=%d it=%d\n",node,subnode,it));
    #endif//DROP_DATA
  }
};
