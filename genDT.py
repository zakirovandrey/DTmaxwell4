#!/usr/bin/python
import itertools
from multiprocessing import Process, Lock

def genfunc(Yt):
  import sys
  import genAlg as gen
  gen.order=4

  defout = sys.__stdout__

  gen.PMLS=0
  gen.AsyncType='D'
  for spml in 0,1:
    gen.PMLS=spml
    fl = open("ker%s%s.inc.cu"%(Yt,("","_pmls")[spml]), 'w')
    print >>fl, '#include "params.h"'
    if Yt[-4:]=='TFSF' or Yt[0]=="I": print >>fl, '#include "signal.h"'
    if Yt=="D" and spml==0: print >>fl, 'const int Nzb = 768*4/FTYPESIZE;'
    for Dcase in 0,1:
      minB = "Nzb/Nz+Nz/(Nzb+1)" if Yt=="D" and spml==0 else "1"
      print >>fl, '__global__ void __launch_bounds__(Nz,%s) %storre%s%d (int ix, int y0, int Nt, int t0) {'%(minB,("","PMLS")[spml],Yt,Dcase)
      print >>fl, '  REG_DEC(%d)'%Dcase
      if Yt[-4:]=='TFSF': print >>fl, '  #include "%s%d.inc.cu"'%(Yt,Dcase)
      else         : print >>fl, '  #include "%s%d%s.inc.cu"'%(Yt,Dcase,("","_pmls")[spml])
      #if Yt=='TFSF': print >>fl, '  if(inPMLv){\n    #include "D%d_pmlv.inc.cu"   \n  }else{\n    #include "TFSF%d.inc.cu"\n  }\n}'%(Dcase,Dcase)
      #else         : print >>fl, '  if(inPMLv){\n    #include "%s%d%s_pmlv.inc.cu"\n  }else{\n    #include "%s%d%s.inc.cu"\n  }\n}'%(Yt,Dcase,("","_pmls")[spml],Yt,Dcase,("","_pmls")[spml])
      print >>fl, '  POSTEND(%d)\n}'%Dcase
    fl.close()
    if Yt[-4:]=='TFSF' and spml==1: continue
    for zpml in 0,:
      for Dcase in 0,1:
        LargeNV=1
        fname = "%s%d%s%s.inc.cu"%(''.join(Yt), Dcase, ("","_pmls")[spml], ("","_pmlv")[zpml])
        defout.write("generating %s\n"%fname)
        Yt=''.join(Yt); gen.AsyncType=Yt
        sys.stdout = open(fname, 'w')

        gen.make_DTorre(typus=Dcase, vpml=zpml, atype=Yt, spml=spml, Disp=1, LargeNV=LargeNV)
        #sys.stdout.flush()
        #sys.stdout.close()

procs = []
for _Yt in ('D','S','Is','Id','Xs','Xd','TFSF','ITFSF','DISP'): procs.append( Process(target=genfunc, args=(_Yt,)) )
for p in procs: p.start()
for p in procs: p.join()
