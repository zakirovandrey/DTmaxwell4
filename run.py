#!/usr/bin/python
# -*- coding: utf-8 -*-

from ctypes import *
#mpi = CDLL('libmpi.so.0', RTLD_GLOBAL)
#mpi = CDLL('/usr/lib64/mpich2/lib/libmpich.so.1.2', RTLD_GLOBAL)

from math import *
import sys
import os
import DTmxw
GridNx = DTmxw.cvar.GridNx
GridNy = DTmxw.cvar.GridNy
GridNz = DTmxw.cvar.GridNz
dx=DTmxw.cvar.ds
dz=DTmxw.cvar.dv
dy=DTmxw.cvar.da
dt=DTmxw.cvar.dt

center  = [ GridNx/2*dx, GridNy/2*dy, GridNz/2*dz]

SS = DTmxw.cvar.shotpoint
#SS.wavelength=0.6;
SS.srcXs, SS.srcXv, SS.srcXa = center[0],center[1],center[2];
SS.BoxMs, SS.BoxPs = SS.srcXs-10*dx, SS.srcXs+10*dx; 
SS.BoxMa, SS.BoxPa = SS.srcXa-10*dz, SS.srcXa+10*dz; 
SS.BoxMv, SS.BoxPv = SS.srcXv-10*dy, SS.srcXv+10*dy;
boxDiagLength=sqrt((SS.BoxPs-SS.BoxMs)**2+(SS.BoxPa-SS.BoxMa)**2+(SS.BoxMv-SS.BoxPv)**2)
cL=1

SS.set(0,0,0)

SS.tStop = 0#boxDiagLength/2/cL+8/(SS.w/2)+10*dt # 5000*dt; # ((BoxPs-BoxMs)+(BoxPa-BoxMa)+(BoxMv-BoxPv))/c+2*M_PI/Omega;

DTmxw.cvar.Tsteps=100*(10 if GridNy>200 else 100)
DTmxw._main(sys.argv)
