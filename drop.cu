struct SeismoDrops {
  int node,Nprocs;
  std::string* dir;
  void init() {
    node=0; Nprocs=1;
    #ifdef MPI_ON
    MPI_Comm_rank (MPI_COMM_WORLD, &node);
    MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
    #endif
  }
};
