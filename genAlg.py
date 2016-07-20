from math import *
import itertools
import re

NDT=3
SqGrid=0

YbndI=0
YbndX=NDT
def isOutA (y, Yt=None): return not {"I":y>=YbndI, "X":y<=YbndX, "D":True}.get(Yt[0],True)

TwoDominoSz   ={"S":6,"V":3}
TwoDominoSzPML={"S":6,"V":3}

class data():
  Registers={}
  Shareds={}
  def __hash__(self):
    if not self: return 0 # empty
    return hash((self.coord, self.typus, self.proj))
  def __eq__(self, other): return list(self.coord)==list(other.coord) and self.proj==other.proj
  def __init__(self,coord, Sproj=None, only=False, zeroval=None):
    self.coord=coord;
    self.typus = "S" if all(c%2 for c in coord) else None if all(c%2==0 for c in coord) else "T" if sum(coord)%2 else "V"
    self.name=self.getname(Sproj)
    self.memtype="G"

    self.hasdz=False
    if only: return
    c=self.coord
    self.neigh=[[],[],[]]
    self.nearE=[[],[]]
    for n in range(order): 
      shift = (-n-1,n)[n%2]#*0.5
      Ashift=shift
      if isOutA(c[1]+shift, data.Atype): Ashift=2*{'I':YbndI,'X':YbndX}.get(data.Atype[0],0)-2*c[1]-shift
      zeroval=None
      if data.Atype[0]=="X" and c[1]+shift<=YbndX: zeroval="zerov"
      self.neigh[0].append( data((c[0]+shift,c[1],c[2] ), Sproj=0,only=1) )
      self.neigh[1].append( data((c[0],c[1]+Ashift,c[2]), Sproj=1,only=1, zeroval=zeroval) )
      self.neigh[2].append( data((c[0],c[1],c[2]+shift ), Sproj=2,only=1) )
      if self.neigh[0][-1].typus=="S":
        self.nearE[0].append( data((c[0]+shift,c[1],c[2] ), Sproj=1,only=1) )
        self.nearE[1].append( data((c[0]+shift,c[1],c[2] ), Sproj=2,only=1) )
      if self.neigh[1][-1].typus=="S":
        self.nearE[0].append( data((c[0],c[1]+Ashift,c[2]), Sproj=2,only=1, zeroval=zeroval) )
        self.nearE[1].append( data((c[0],c[1]+Ashift,c[2]), Sproj=0,only=1, zeroval=zeroval) )
      if self.neigh[2][-1].typus=="S":
        self.nearE[0].append( data((c[0],c[1],c[2]+shift ), Sproj=0,only=1) )
        self.nearE[1].append( data((c[0],c[1],c[2]+shift ), Sproj=1,only=1) )
    if self.neigh[2][0].name and self.name: self.hasdz=True
    self.aniso_neigh=[[[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]],[[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]],[[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]]
    for xyz in (0,1,2):
      if self.proj==xyz: self.aniso_neigh[xyz]="Diag"; continue
      shift = [0,0,0]
      for iaver in (0,1,2,3):
        shift[self.proj] = (1,-1, 1,-1)[iaver]
        shift[xyz]       = (1, 1,-1,-1)[iaver]
        for difxyz in (0,1,2):
          shift[difxyz]+= 1
          self.aniso_neigh[xyz][iaver][difxyz].append( data((c[0]+shift[0],c[1]+shift[1],c[2]+shift[2]), Sproj=0, only=1) )
          shift[difxyz]-= 2
          self.aniso_neigh[xyz][iaver][difxyz].append( data((c[0]+shift[0],c[1]+shift[1],c[2]+shift[2]), Sproj=0, only=1) )
          shift[difxyz]+= 1
  
  def getname_glob(self, Sproj):
    if not self.typus: return None
    for x,y in itertools.product((-2,-1,0,1), (-1,0,1)): 
      fldNum=0; self.fldindPML=0; self.fldindS=0
      for datN, dat_sh in enumerate(Plaster.DATAshifts):
        abs_c = map(sum, zip((x*2*NDT,y*2*NDT,0), dat_sh))
        if abs_c==[self.coord[0],self.coord[1],self.coord[2]%2]:
          self.plsId=(x,y); self.datId=datN;
          hlist = [c%2 for c in self.coord]
          self.proj=Sproj if self.typus=='S' else hlist.index(1) if self.typus=='T' else hlist.index(0)
          self.fldind=fldNum+(Sproj if self.typus=='S' else 0)
          dmdtype=self.typus.replace("T","S")
          find = self.fldind-(TwoDominoSz['S']*NDT*NDT if dmdtype=='V' else 0)
          self.two_domino = (find/TwoDominoSz[dmdtype], find%TwoDominoSz[dmdtype])
          head = "RAG%s%s->%si[%d]"%('mcp'[x+1],'mcp'[y+1],dmdtype,self.two_domino[0])
          if self.typus=="V": self.globname_pair = "%s.trifld.%s[iz%+d]%s"%(head,("one","two","two")[self.two_domino[1]]  ,self.coord[2]/2,("",".x",".y")[self.two_domino[1]])
          else              : self.globname_pair = "%s.trifld.%s[iz%+d]%s"%(head,("one","one","two")[self.two_domino[1]/2],self.coord[2]/2,("","","","",".x",".y")[self.two_domino[1]])
          return "%s.fld[%d][iz%+d]"%(head,self.two_domino[1],self.coord[2]/2)
#          return "%sPLASTER(%2d,%2d).%si[%d].fld[%d][iz%+d]"%('mcp'[y+1],x,y,dmdtype,self.two_domino[0],self.two_domino[1],self.coord[2]/2)
#          return "PLASTER(%2d,%2d).fld[I%02d][iz%+d]"%(x,y,self.fldind,self.coord[2]/2)
        if all(ac%2 for ac in abs_c):   fldNum+=3; self.fldindS+=1
        elif any(ac%2 for ac in abs_c): fldNum+=1; self.fldindPML+=0 if map(lambda ac: ac%2, abs_c)==[0,0,1] else 1
    print "Data globname not found coords=",self.coord; exit(-1)
    
  def getname(self, Sproj=None):
    c=tuple(self.coord)
    self.globname = self.getname_glob(Sproj)
    if data.PMLS and self.globname:
      self.globname      = "*(isOutS(ix*2*NDT+(%d))?&zerov:(&%s))"%(self.coord[0], self.globname)
      self.globname_pair = "*(isOutS(ix*2*NDT+(%d))?&zerov:(&%s))"%(self.coord[0], self.globname_pair)
    return self.upgrade()
  def upgrade(self):
    if not self.globname: return None
    elif self in data.Registers: self.name=data.Registers[self]
    elif self in data.Shareds  : self.name=data.Shareds[self]
    else: self.name=self.globname_pair
    return self.name

  def save(self):
    if data.PMLS:
      savename = self.globname_pair
      if savename: print "  if(!isOutS(ix*2*NDT+(%g))) "%self.coord[0]
      if savename: print "  %s = %s; // %s"%(savename, self.name, self.coord)
    else:
      #if   (self.typus,self.proj) in [("S",0),("S",2),("V",0)]  : return
      #elif (self.typus,self.proj) in [("S",1),("T",1),("V",2)]  : print "  %s = %s; // %s"%(self.globname_pair[:-2], self.name[:-2], self.coord)
      #elif (self.typus,self.proj) in [("V",1),("T",2),("T",0), ]: print "  %s = %s; // %s"%(self.globname_pair, self.name, self.coord)
      if   (self.typus,self.proj) in [("S",0),("S",1),("S",2),("T",2),("V",0)]: return
      elif (self.typus,self.proj) in [("T",0),("V",2)]                        : print "  %s = %s; // %s"%(self.globname_pair[:-2], self.name[:-2], self.coord)
      elif (self.typus,self.proj) in [("T",1),("V",1)]                        : print "  %s = %s; // %s"%(self.globname_pair, self.name, self.coord)

  def updatePML(self, pmltype, difs, signs):
    n=0; pmlnames=self.pmlnames[:]; splitnums=(0,1,2);
    pml1d = (list(pmltype)==[0,0,1] and self.typus in "SV") or ([pmltype[(1,0,2)[self.proj]],pmltype[2]]==[0,1] and self.typus=="T")
    if pml1d and self.typus in "SV": pmlnames = ["regPml.x", "regPml.y", "regPml.y" ]; print "    regPml.y = %s; regPml.x = %s-regPml.y;"%(re.sub("fld\[.{1,3}\]","fldPML[pmlRagV_%d]"%self.two_domino[1],self.pmlnames[0]), self.name)
    if pml1d and self.typus=="T"   : pmlnames = ["regPml.x", "regPml.y"             ]; print "    regPml.y = %s; regPml.x = %s-regPml.y;"%(re.sub("fld\[.{1,3}\]","fldPML[pmlRagT_%d]"%self.two_domino[1],self.pmlnames[0]), self.name)
#    if self.typus=="S" and list(pmltype)==[0,0,1]: self.pmlnames = map(lambda name: name.replace(".Sx",".Si").replace(".Sy",".Si").replace(".Sz",".Si"), self.pmlnames); pmlnames=self.pmlnames[:]
    if self.typus=="S": return
    for i in 0,1,2:
      if not difs[i]: continue;
      pmlname=pmlnames[n]
      c="xyz"[i]; sign=signs[i]; difk24 = "dif%s[%d]"%(c,uniqdifN)
      #if data.PrepareDifs and self.typus=="V" and i==0: difk24 = "difx[%d]"%uniqdifN
      if self.typus=='V' and i==self.proj: difk24='0';
      Kpmlind = "Kpml_i%s%+d"%(c,self.coord[i])
      #for cyclic PML ### if c=='y': Kpmlind = "(Kpml_iy%+d+KNpmly)%%KNpmly"%self.coord[i]
      if c=='y': Kpmlind = "(Kpml_iy%+d)%%KNpmly"%self.coord[i]
      if c=='x': Kpmlind = "Kpml_ix"
      Kpml1,Kpml2 = map(lambda n: ("Kpml%s%d[%s]"%(c,n,Kpmlind)), (1,2))
      if self.typus=='V' and i==self.proj: Kpml1='1'; Kpml2='0'
      if pmltype[i]: print "    %s = %s*%s %s %s*%s;"%(pmlname, Kpml1, pmlname, sign, Kpml2, difk24 )
      elif i==0 and self.typus in "SV" and pml1d: print "    %s += %s %s "%(pmlname, sign, difk24),
      elif i==1 and self.typus in "SV" and pml1d: print "    %s %s ;"%(sign, difk24); splitnums=(0,2)
      elif          self.typus=="T"    and pml1d: print "    %s %s= %s ;"%(pmlname, sign, difk24)
      else:  print "    %s %s= %s;"%(pmlname, sign, difk24)
      n+=1
    if pml1d and self.typus in "SV": print "    %s = regPml.y;"%(re.sub("fld\[.{1,3}\]","fldPML[pmlRagV_%d]"%self.two_domino[1],self.pmlnames[0]))
    if pml1d and self.typus=="T"   : print "    %s = regPml.y;"%(re.sub("fld\[.{1,3}\]","fldPML[pmlRagT_%d]"%self.two_domino[1],self.pmlnames[0]))
    if pml1d: print "      %s= %s ;"%( self.name, "+".join((pmlnames[0],pmlnames[2]) if splitnums==(0,2) else pmlnames) )
    else    : print "      %s= %s ;"%( self.name, "+".join((pmlnames[0],pmlnames[2]) if splitnums==(0,2) else pmlnames) )
#      if pmltype[i] and difs[i]: print "  pmlname%d = %s; pmlname%d = %s*pmlname%d %s %s*%s; %s=pmlname%d; "%(n, pmlname, n, Kpml1, n, sign, Kpml2, difk24, pmlname, n); n+=1
#      elif              difs[i]: print "  pmlname%d = %s; pmlname%d %s= %s; %s = pmlname%d;"%(n, pmlname, n, sign, difk24, pmlname,n); n+=1
#    print "    %s = coff%s*( pmlname1 + pmlname2 );"%( self.name, self.typus )
  
  def prepare_difs(self, time=0):
    if not self.name or not data.PrepareDifs: return
    print "  //------- prepare difsXY for field %s"%self.typus,"xyz"[self.proj],self.coord
    difx, dify, difz = None, None, None
    acc_coff = {2:["1",], 4:["p27","1"]}[order]
    hind = self.fldind-self.fldindS*2-(self.proj if self.typus=="S" else 0)
    if self.typus!="S" or self.proj==0: print "  coff%s = Arrcoff%s[%d];"%(self.typus,self.typus,hind)
    cofftype_aniso = []
    cofftype       = map(lambda np: "*coffS.%s"%'yx'[self.proj==np],(0,1,2)) if self.typus=="S" else ["*coff%s"%self.typus,]*3
    if self.typus=="S":
      cofftype_aniso.append( map(lambda xyzw: "*coffS.%s"%xyzw, ("xyy","ywz","yzw")[self.proj]) )
      cofftype_aniso.append( map(lambda xyzw: "*coffS.%s"%xyzw, ("wyz","yxy","zyw")[self.proj]) )
      cofftype_aniso.append( map(lambda xyzw: "*coffS.%s"%xyzw, ("wzy","zwy","yyx")[self.proj]) )
    dtdr_coffs = (cofftype[0],cofftype[1],cofftype[2]) if SqGrid else ("*dtdxd24%s"%cofftype[0],"*dtdyd24%s"%cofftype[1],"*dtdzd24%s"%cofftype[2])
    if self.neigh[0][0].name: difx = "-(%13s) ONE+(%13s) ONE+%s*%13s-%s*%13s"%(self.neigh[0][2].name, self.neigh[0][3].name, acc_coff[0],self.neigh[0][0].name, acc_coff[0],self.neigh[0][1].name, )
    if self.neigh[1][0].name: dify = "-(%13s) ONE+(%13s) ONE+%s*%13s-%s*%13s"%(self.neigh[1][2].name, self.neigh[1][3].name, acc_coff[0],self.neigh[1][0].name, acc_coff[0],self.neigh[1][1].name, )
    if self.neigh[1][0].name and data.Atype[0]=="I" and self.typus=="V":
      free_refl = map(lambda n: "+-"[self.coord[1]+(-n-1,n)[n%2]<0], range(order) )
      dify = "-(%s%13s) ONE+(%s%13s) ONE+(%s%s*%13s)-(%s%s*%13s)"%(free_refl[2], self.neigh[1][2].name, free_refl[3], self.neigh[1][3].name, free_refl[0],acc_coff[0],self.neigh[1][0].name, free_refl[1],acc_coff[0],self.neigh[1][1].name, )
    global uniqdifN;
    signx,signy,signz = "+++"
    if self.typus!="S": signx,signy,signz = map(lambda xyz: "+-"[((xyz-self.proj)%3+"VT".index(self.typus))%2] , (0,1,2) )
    #if self.typus=="V":   print "  difx[%2d] = (%s)%s + (%s)%s;"%(uniqdifN,difx,dtdr_coffs[0], dify,dtdr_coffs[1]);
    if self.typus=="S":
      difx = "%13s*0.5625+%13s*0.5625-0.0625*%13s-0.0625*%13s"%(self.neigh[0][0].name, self.neigh[0][1].name, self.neigh[0][2].name, self.neigh[0][3].name, )
      dify = "%13s*0.5625+%13s*0.5625-0.0625*%13s-0.0625*%13s"%(self.neigh[1][0].name, self.neigh[1][1].name, self.neigh[1][2].name, self.neigh[1][3].name, )
      dtdr_coffs = map(lambda np: ("*0","")[self.proj==np],(0,1,2))
      #print "  difx[%2d] = (%s)%s + (%s)%s;"%(uniqdifN,difx,dtdr_coffs[0], dify,dtdr_coffs[1]);
    else:
      if difx: print "  difx[%2d] = (%s)%s;"%(uniqdifN,difx, dtdr_coffs[0]) if self.proj!=0 else "  difx[%2d] = 0;"%uniqdifN
      if dify: print "  dify[%2d] = (%s)%s;"%(uniqdifN,dify, dtdr_coffs[1]) if self.proj!=1 else "  dify[%2d] = 0;"%uniqdifN
    uniqdifN+= 1
  def update(self, time=0):
    if not self.name: return
    if isOutA(self.coord[1], data.Atype): return
    print "  //------- update %s"%self.typus,"xyz"[self.proj],self.coord
    checkZdmd = "if(isCONz%s(%g,%g))"%(self.typus, (self.coord[0]+2*NDT)%(2*NDT), self.coord[2])
    print "  %s{"%checkZdmd
    #print "  h = modelRag%s->h[%2d][iz];"%("MCP"[1+self.plsId[0]],self.fldind-self.fldindS*2-(self.proj if self.typus=="S" else 0) )
    hind = self.fldind-self.fldindS*2-(self.proj if self.typus=="S" else 0)
    #print "  TEXCOFF%s(%d, %+d, %+d, iz*2%+d, I,h[%2d]);"%(self.typus+('','xyz'[fld.proj])[fld.typus=='T'],hind,self.coord[0],self.coord[1],self.coord[2], hind)
    if self.typus!="S" or self.proj==0: print "  coff%s = Arrcoff%s[%d];"%(self.typus,self.typus,hind)
    pml=1; pmlx=0; pmly=0; pmlz=1
    if data.Atype=='S' or data.Atype[0:2]=='Xs' or data.Atype[0:2]=='Is': pmly=1
    difx, dify, difz = None, None, None
    acc_coff = {2:["1",], 4:["p27","1"]}[order]
    if self.neigh[0][0].name: difx = "-(%13s) ONE+(%13s) ONE+%s*%13s-%s*%13s"%(self.neigh[0][2].name, self.neigh[0][3].name, acc_coff[0],self.neigh[0][0].name, acc_coff[0],self.neigh[0][1].name, )
    if self.neigh[1][0].name: dify = "-(%13s) ONE+(%13s) ONE+%s*%13s-%s*%13s"%(self.neigh[1][2].name, self.neigh[1][3].name, acc_coff[0],self.neigh[1][0].name, acc_coff[0],self.neigh[1][1].name, )
    if self.neigh[2][0].name: difz = "-(%13s) ONE+(%13s) ONE+%s*%13s-%s*%13s"%(self.neigh[2][2].name, self.neigh[2][3].name, acc_coff[0],self.neigh[2][0].name, acc_coff[0],self.neigh[2][1].name, )
    if self.neigh[1][0].name and data.Atype[0]=="I" and self.typus=="V":
      free_refl = map(lambda n: "+-"[self.coord[1]+(-n-1,n)[n%2]<0], range(order) )
      dify = "-(%s%13s) ONE+(%s%13s) ONE+(%s%s*%13s)-(%s%s*%13s)"%(free_refl[2], self.neigh[1][2].name, free_refl[3], self.neigh[1][3].name, free_refl[0],acc_coff[0],self.neigh[1][0].name, free_refl[1],acc_coff[0],self.neigh[1][1].name, )
    if self.typus=="S":
      difx = "%13s*0.5625+%13s*0.5625-0.0625*%13s-0.0625*%13s"%(self.neigh[0][0].name, self.neigh[0][1].name, self.neigh[0][2].name, self.neigh[0][3].name, )
      dify = "%13s*0.5625+%13s*0.5625-0.0625*%13s-0.0625*%13s"%(self.neigh[1][0].name, self.neigh[1][1].name, self.neigh[1][2].name, self.neigh[1][3].name, )
      difz = "%13s*0.5625+%13s*0.5625-0.0625*%13s-0.0625*%13s"%(self.neigh[2][0].name, self.neigh[2][1].name, self.neigh[2][2].name, self.neigh[2][3].name, )
      dtdr_coffs = map(lambda np: ("*0","")[self.proj==np],(0,1,2))
    for xyz in 0,1,2:
      def SrcFunc(f, nearf): return "" 
      if data.Atype[-4:]=='TFSF' and self.neigh[xyz][0].name: 
        global numfnear; numfnear=0
        def SrcFunc(f, nearf): 
          global numfnear; srcsign = "src%d%s"%(numfnear,'xyz'[xyz]); numfnear+=1
          return "+%s"%srcsign
        def makeSigns(f, nearf, ni): 
          val0 = "%g+i%s"%(    f.coord[xyz],'xyz'[xyz])
          valn = "%g+i%s"%(nearf.coord[xyz],'xyz'[xyz])
          crd = "SAV"[xyz];  srcsign='checktest'
          if nearf.coord[xyz]!=f.coord[xyz]:
            print "  src%d%s = 0.0f; upd_inSF = inSF(%g+glob_ix*2*NDT,%g+phys_iy*2*NDT,%g+iz*2); neigh_inSF = inSF(%g+glob_ix*2*NDT,%g+phys_iy*2*NDT,%g+iz*2);"%(ni,'xyz'[xyz], f.coord[0],f.coord[1],f.coord[2],nearf.coord[0],nearf.coord[1],nearf.coord[2])
            print "  if     (!upd_inSF &&  neigh_inSF) src%d%s =  SrcTFSF_%s%s(glob_ix*2*NDT+(%g), iz*2+(%g), %g+phys_iy*2*NDT, tshift+it+%g);"%(ni,'xyz'[xyz], nearf.typus, "xyz"[nearf.proj], nearf.coord[0], nearf.coord[2], nearf.coord[1], time)
            print "  else if( upd_inSF && !neigh_inSF) src%d%s = -SrcTFSF_%s%s(glob_ix*2*NDT+(%g), iz*2+(%g), %g+phys_iy*2*NDT, tshift+it+%g);"%(ni,'xyz'[xyz], nearf.typus, "xyz"[nearf.proj], nearf.coord[0], nearf.coord[2], nearf.coord[1], time)
          else: return
        for ni in 2,3,0,1: makeSigns(self,self.neigh[xyz][ni],ni)
        near_tfsf = map(lambda nachbar: "(%s%s)"%(nachbar.name,SrcFunc(self,nachbar)), self.neigh[xyz])
        dif = "-(%s) ONE + (%s) ONE +%s*%s-%s*%s"%(near_tfsf[2], near_tfsf[3], acc_coff[0],near_tfsf[0], acc_coff[0],near_tfsf[1], )
        if xyz==1 and self.neigh[1][0].name and data.Atype[0]=="I" and self.typus=="V":
          free_refl = map(lambda n: "+-"[self.coord[1]+(-n-1,n)[n%2]<0], range(order) )
          dif = "-(%s%s) ONE+(%s%s) ONE+(%s%s*%s)-(%s%s*%s)"%(free_refl[2], near_tfsf[2], free_refl[3], near_tfsf[3], free_refl[0],acc_coff[0],near_tfsf[0], free_refl[1],acc_coff[0],near_tfsf[1], )
        if self.typus=="S":
          dif = "%13s*0.5625+%13s*0.5625-0.0625*%13s-0.0625*%13s"%(near_tfsf[0], near_tfsf[1], near_tfsf[2], near_tfsf[3])
          dtdr_coffs = map(lambda np: ("*0","")[self.proj==np],(0,1,2))
        if xyz==0 and self.neigh[xyz][0].typus!="S": difx=dif
        if xyz==1 and self.neigh[xyz][0].typus!="S": dify=dif
        if xyz==2 and self.neigh[xyz][0].typus!="S": difz=dif
    global uniqdifN;
    cofftype = map(lambda np: "*coffS.%s"%'yx'[self.proj==np],(0,1,2)) if self.typus=="S" else ["*coff%s"%self.typus,]*3
    dtdr_coffs = cofftype if SqGrid else ("*dtdxd24%s"%cofftype[0],"*dtdyd24%s"%cofftype[1],"*dtdzd24%s"%cofftype[2])
    if self.typus=="S": dtdr_coffs = map(lambda np: ("*0","")[self.proj==np],(0,1,2))
    if self.typus=="V" and self.proj==0: difx='0'
    if self.typus=="V" and self.proj==1: dify='0'
    if self.typus=="V" and self.proj==2: difz='0'
    if   (not data.PrepareDifs) and difx and self.typus!="S": print "  difx[%2d] = (%s)%s;"%(uniqdifN,difx, dtdr_coffs[0]); 
    if   (not data.PrepareDifs) and dify and self.typus!="S": print "  dify[%2d] = (%s)%s;"%(uniqdifN,dify, dtdr_coffs[1]); 
    if                              difz and self.typus!="S": print "  difz[%2d] = (%s)%s;"%(uniqdifN,difz, dtdr_coffs[2]);
    pmlfields = "Vx,Vy,Vz,Tx,Ty,Tz,Sx,Sy,Sz,Ex,Ey,Ez,Hx,Hy,Hz"
    signx,signy,signz = "+++"
    if self.typus!="S": signx,signy,signz = map(lambda xyz: "+-"[((xyz-self.proj)%3+"VT".index(self.typus))%2] , (0,1,2) )
    pmlz = data.inPMLv
    #summands = list(itertools.compress( map( lambda (sign,xyz): " %s %s"%(sign, "dif%s[%d]"%(xyz,uniqdifN)), zip((signx,signy,signz),'xyz') ), (difx,dify,difz)))
    #if data.Disp and self.typus=="V" and not 0*data.PMLS: 
    #  if data.PMLS: print "    if(!isOutS(ix*2*NDT+(%g))) "%(self.coord[0])
    #  self.updateDisp("(%s)/coffV;"%(''.join(summands)))
    if data.PMLS: print "  if(!isOutS(ix*2*NDT+(%g))) if(inPMLsync(ix*2*NDT+(%g))) {"%(self.coord[0],self.coord[0])
    if data.PMLS: print "    Kpml_ix=get_pml_ix(ix*2*NDT+(%d));"%floor(self.coord[0])
    self.Gyration(hind)
    for pmlx in ((0,),(1,0))[data.PMLS]:
      summands = list(itertools.compress( map( lambda (sign,xyz): " %s %s"%(sign, "dif%s[%d]"%(xyz,uniqdifN)), zip((signx,signy,signz),'xyz') ), (difx,dify,difz)))
      if data.Disp and self.typus=="V" and pmlx==0: self.updateDisp("(%s)/coffV;"%(''.join(summands)))
      for pmlz in ((0,),(1,0))[bool(difz) and self.typus!="S"]:
        if pmlz==1: print "  if(inPMLv){"
        pml_ever = (difx and pmlx) or (dify and pmly) or (difz and pmlz)
        pname = self.globname.replace("RAG","SpmlRAG" if bool(difx and pmlx) else "ApmlRAG" if bool(dify and pmly) else "RAG").replace("iz",("pml_",'')[bool(difx and pmlx or dify and pmly)]+"iz")
        if data.PMLS: pname = pname[pname.find(':')+3:-2]
        self.pmlnames = list(itertools.compress( map( lambda proj: reduce(lambda n,v: n.replace(v, v+proj), [pname,]+pmlfields.split(",")), "xyz" ), (difx,dify,difz) ))
        if bool(difx and pmlx) or bool(dify and pmly):
          for nn in range(len(self.pmlnames)): self.pmlnames[nn]+= ".%s"%('xyz'[nn])
        if pml_ever: self.updatePML((pmlx,pmly,pmlz), (difx,dify,difz), (signx,signy,signz))
        else:
          summands = list(itertools.compress( map( lambda (sign,xyz): " %s %s"%(sign, "dif%s[%d]"%(xyz,uniqdifN)), zip((signx,signy,signz),'xyz') ), (difx and self.typus!=0, dify and self.typus!=1, difz and self.typus!=2) ))
          if data.PrepareDifs and self.typus in "S": summands = ("difx[%d]"%uniqdifN, "%s difz[%d]"%(signz,uniqdifN) )
          if self.typus=='S': pass #print "      %s = %s ;"%(self.name, ''.join(summands) )
          else              : print "      %s+= %s ;"%(self.name, ''.join(summands) )
        if pmlz==1 and difz and self.typus!='S': print "  } else {"
      if difz and self.typus!='S': print "  }"
      if data.PMLS and pmlx==1: print "  } else {"
      if data.Disp and self.typus=="V" and pmlx==0: print "  }"
    if data.PMLS: print "  }"
    #if data.Disp and self.typus=="V" and not 0*data.PMLS: print "  }"

    if data.Atype[0]=='I' and not isOutA(self.coord[1], data.Atype) and isOutA(self.coord[1]-2, data.Atype):
      print "  %s+= SrcSurf_%s%s(glob_ix*2*NDT+(%g), iz*2+(%g), %g+phys_iy*2*NDT, pars.iStep*Ntime+it+%g);"%(self.name, self.typus,'xyz'[self.proj], self.coord[0], self.coord[2], self.coord[1], time)
      #if self.typus!="S": print "  #ifndef DROP_ONLY_V"
      #if self.typus=="S": fval = "%s*0.5625+%s*0.5625-%s*0.0625-%s*0.0625"%tuple(map(lambda n: n.name, self.neigh[self.proj]))
      #else: fval = self.name
      #print "  dropPP(ix*NDT%+d, %d-chunk%s[0], iz, it, channel%s, %s);"%(
      #    self.coord[0]/2, self.coord[0]/2, self.typus+('xyz'[self.proj] if self.typus!='S' else 'i'), self.typus+'xyz'[self.proj], fval)
      #if self.typus!="S": print "  #endif// DROP_ONLY_V"

    rot=["","",""]
    for i in (0,1,2):
      continue
      if self.typus!="V" or i==self.proj: continue
      for iaver in (0,1,2,3):
        for difxyz in (0,1,2):
          if i==difxyz: continue
          sign = "+-"[((difxyz-i)%3)%2]
          rot[i]+= "%s(%s-%s)*dtd%s"%(sign,self.aniso_neigh[i][iaver][difxyz][0].name, self.aniso_neigh[i][iaver][difxyz][1].name, "xyz"[difxyz])
      rot[i] = "(%s)*0.25"%rot[i]
      print "  %s+= AnisoE[%2d].coffAniso%s%s*%s;"%(self.name, hind, 'XYZ'[self.proj],'XYZ'[i], rot[i])
    print "  }//checkZdmd"
    uniqdifN+= 1

  def Gyration(self,hind):
    return
    if self.typus=="V":
      Ep = "0.5*(%s+%s)"%(self.nearE[0][0].name, self.nearE[0][1].name)
      Em = "0.5*(%s+%s)"%(self.nearE[1][0].name, self.nearE[1][1].name)
      E = ['','','']
      E[self.proj]=self.name; E[(self.proj+1)%3]=Ep; E[(self.proj+2)%3]=Em;
      Ex,Ey,Ez = E
      print   "  %s = AnisoE[%2d].gyration%sX*%s + AnisoE[%2d].gyration%sY*%s + AnisoE[%2d].gyration%sZ*%s;"%(self.name,hind,'XYZ'[self.proj],Ex,hind,'XYZ'[self.proj],Ey,hind,'XYZ'[self.proj],Ez)

  def updateDisp(self, rot):
    Em,Ep,Jm,Jp = map(lambda EJ: self.globname.replace("RAG","rdisp").replace("Vi",EJ).replace("iz","izdisp"), ("Em","Ep","Jm","Jp"))
    Ep=self.name;
    if data.PMLS: Em = Em.split(":")[1][2:-2]
    if data.PMLS: Jm = Jm.split(":")[1][2:-2]
    if data.PMLS: Jp = Jp.split(":")[1][2:-2]
    print "  if(ISDISP(%d,%d,2*iz%+d)){ "%(self.coord[0],self.coord[1],self.coord[2])
    print "    rot = %s;"%rot
    print "    countDisp(%s,\n     %s,\n     %s,\n     %s, rot,TEXCOFFDISP((%d), (%d), 2*iz+%d), %d);"%(Ep,Em,Jp,Jm,self.coord[0],self.coord[1],self.coord[2], self.proj)
    print "  } else {"

class Plaster():
  Dmd_shifts = ( ((-NDT,NDT), "S", 0),
                 ((0   ,0  ), "S", 1),
                 ((-NDT,0  ), "V", 2),
                 ((0   ,NDT), "V", 3)  )
  '''
  DATAindexes = (
                      (0,0,1), 
             (1,-1,1),(1,0,0),(1,1,1),
    (2,-2,1),(2,-1,0),(2,0,1),(2,1,0),(2,2,1),
    (3,-2,0),(3,-1,1),(3,0,0),(3,1,1),(3,2,0),
             (4,-1,0),(4,0,1),(4,1,0),
                      (5,0,0)                  )'''
  DATAindexes = (
    (0, 0,1),(1, 0,0),(1, 1,1),(2, 1,0),(2, 2,1),(3, 2,0),
             (1,-1,1),(2,-1,0),(2, 0,1),(3, 0,0),(3, 1,1),(4,1,0),
                      (2,-2,1),(3,-2,0),(3,-1,1),(4,-1,0),(4,0,1),(5,0,0)  )
  DMDsize = len(DATAindexes)
  def __init__(self, **kwargs): self.__dict__.update(kwargs)
Plaster.DATAshifts = map(lambda indx: (Plaster.Dmd_shifts[0][0][0]+indx[0], Plaster.Dmd_shifts[0][0][1]+indx[1], indx[2]), Plaster.DATAindexes)+\
                     map(lambda indx: (Plaster.Dmd_shifts[1][0][0]+indx[0], Plaster.Dmd_shifts[1][0][1]+indx[1], indx[2]), Plaster.DATAindexes)+\
                     map(lambda indx: (Plaster.Dmd_shifts[2][0][0]+indx[0], Plaster.Dmd_shifts[2][0][1]-indx[1], indx[2]), Plaster.DATAindexes)+\
                     map(lambda indx: (Plaster.Dmd_shifts[3][0][0]+indx[0], Plaster.Dmd_shifts[3][0][1]-indx[1], indx[2]), Plaster.DATAindexes)

def getColorType(x,y,z): return "S" if sum(c%2 for c in [x,y,z])%2==1 else "V"


class Diamond():
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
    found=0
    for x,y in itertools.product((-2,-1,0,1), (-1,0,1)):
      if found: break
      for dmd_sh in Plaster.DATAshifts[::Plaster.DMDsize]:
        if self.left[0]==x*2*NDT+dmd_sh[0] and self.left[1]==y*2*NDT+dmd_sh[1] and self.typus==getColorType(self.left[0],self.left[1],dmd_sh[2]):
          self.plsId = (x,y); self.DmdId = Plaster.DATAshifts.index(dmd_sh)/Plaster.DMDsize; found=1; break
    if found==0: print "Error: Diamond not found\n"; exit(-1)
  def upgrade_names(self):
    for fld in self.fields:
      for dat in [fld,]+fld.neigh[0]+fld.neigh[1]+fld.neigh[2]: dat.upgrade()

  def iterate(self, callback):
    for c in Plaster.DATAshifts[Plaster.DMDsize*self.DmdId:Plaster.DMDsize*(self.DmdId+1)]:
      absolute_c = tuple(map(sum, zip(c[:2], [px*2*NDT for px in self.plsId])) + [c[2],])
      global find
      if (not any(ac%2 for ac in absolute_c)) or all(ac%2 for ac in absolute_c): find+=1; continue
      #if not any(ac%2 for ac in absolute_c): find+=1; continue
      callback(absolute_c)
  def set_regs(self):
    def estimate_reg(crd):
      global find
      for ind in range(1+2*all(ac%2 for ac in crd)): regname="reg_fld%s[%2d].%s"%(self.typus, find/2, 'xy'[find%2]); data.Registers[data(crd,ind)]=regname; find+=1
    self.iterate(estimate_reg)
  def set_datas(self):
    self.fields = []
    def estimate_dat(crd):
      for Si in ([0,],[0,1,2])[all(ac%2 for ac in crd)]: self.fields.append( data(crd, Si) )
    self.iterate(estimate_dat)
#    self.near.append( map(lambda pm: Diamond(left=map(sum, zip(self.left,(pm*NDT,0))), type='SV'[self.type=='S']), (-1,1) ) )
#    self.near.append( map(lambda pm: Diamond(left=map(sum, zip(self.left,(0,pm*NDT))), type='SV'[self.type=='S']), (-1,1) ) )

class DiamondTorre():
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
    start_point = ((0,0),(-NDT,NDT))[self.typus]
    self.dmds = [ Diamond(left=map(sum,zip(start_point,(-NDT,0   ))), typus='V'), #tail
                  Diamond(left=map(sum,zip(start_point,(0   ,-NDT))), typus='V'), #wings
                  Diamond(left=map(sum,zip(start_point,(0   ,+NDT))), typus='V'), #wings
                  Diamond(left=map(sum,zip(start_point,(NDT ,+NDT))), typus='S'), #wings
                  Diamond(left=map(sum,zip(start_point,(NDT ,-NDT))), typus='S'), #wings
                ]
    self.dmds.append( Diamond(left=map(sum,zip(start_point,(0   ,0))), typus='S') ) #body
    self.dmds.append( Diamond(left=map(sum,zip(start_point,(NDT ,0))), typus='V') ) #nose
    global find; find=0
    self.dmds[0].set_regs() #tail
    find_old=find
    self.dmds[1].set_regs() #wing_M
    self.dmds[2].set_regs() #wing_P
    find=find_old
    self.dmds[3].set_regs()
    self.dmds[4].set_regs()
    self.dmds[5].set_regs()
    for d in self.dmds: d.set_datas()
    
  def prepare(self): # loading data in registers before Torre-loop
    print "#define pmlRagV_0 0"
    print "#define pmlRagV_1 1"
    print "#define pmlRagV_2 2"
    print "#define pmlRagT_3 0"
    print "#define pmlRagT_4 none"
    print "#define pmlRagT_5 1"
    for fld in self.dmds[5].fields: print "#define pmlRagSh_%02d I%02d //"%(fld.fldind, fld.fldindS if fld.typus=='S' else fld.fldindPML), fld.typus,fld.proj,fld.coord
    for fld in self.dmds[6].fields: print "#define pmlRagSh_%02d I%02d //"%(fld.fldind, fld.fldindS if fld.typus=='S' else fld.fldindPML), fld.typus,fld.proj,fld.coord
    for fld in self.dmds[0].fields: print "%s = %s; // "%(fld.name, fld.globname_pair) , fld.coord
    for fld in self.dmds[5].fields: print "%s = %s; // "%(fld.name, fld.globname_pair) , fld.coord
    
  def loop(self):
    print "for(;it<Nt;it+=dStepT, ix=(ix+dStepX)%Ns, glob_ix+=dStepX/*, RAGcc+=dStepRagC, rm+=dStepRagM, rp+=dStepRagP, rcPMLa+=dStepRagPML, rcPMLs+=dStepRagC*/) {"
    print "  PTR_DEC"
    #if data.Atype[0]=='I': print "  RPOINT_CHUNK_HEAD"
    shrn=-1; time=0;
    for did,d in enumerate(self.dmds[-2:]):
      for fld in d.fields:
        if fld.typus=="S" and fld.proj!=0: continue
        hind = fld.fldind-fld.fldindS*2-(fld.proj if fld.typus=="S" else 0)
        print "  TEXCOFF%s(%d, %+d, %+d, iz*2%+d, I,h[%2d].%s);"%(fld.typus+('','xyz'[fld.proj])[fld.typus in 'VT'],hind,fld.coord[0],fld.coord[1],fld.coord[2], hind/2, 'xy'[hind%2])
      if data.LargeNV: shrn=-1; print "  __syncthreads();"
      for fld in self.dmds[1+did*2].fields :
        if data.PMLS: print "%s = %s; // "%(fld.name, fld.globname_pair) , fld.coord
        elif (fld.typus,fld.proj) in [("V",0),("T",2)]: print "%s = %s; // "%(fld.name[:-2], fld.globname_pair[:-2]) , fld.coord
        elif (fld.typus,fld.proj) in [("S",0),("S",1),("S",2),("V",2),("T",0)]: continue; 
        else        : print "%s = %s; // "%(fld.name, fld.globname_pair) , fld.coord
      for fld in self.dmds[2+did*2].fields : 
        if data.PMLS: print "%s = %s; // "%(fld.name, fld.globname_pair) , fld.coord
        elif (fld.typus,fld.proj) in [("V",0),("T",2)]: print "%s = %s; // "%(fld.name[:-2], fld.globname_pair[:-2]) , fld.coord
        elif (fld.typus,fld.proj) in [("S",0),("S",1),("S",2),("V",2),("T",0)]: continue; 
        else        : print "%s = %s; // "%(fld.name, fld.globname_pair) , fld.coord
      for fld in d.fields:
        if fld.hasdz and fld.typus!="S" and not (fld.typus=="V" and fld.proj==2):
          if fld.neigh[2][3] not in data.Shareds:# and fld.typus!="S" and not (fld.typus=="V" and fld.proj==2):
            shrn+=1
            if not isOutA(fld.coord[1], data.Atype): print "  shared_fld[%2d][izP0].%s = %s; //"%(shrn/2, 'xy'[shrn%2], fld.neigh[2][(1,0)[fld.coord[2]]].name), fld.coord[:2]
          for zz in range(order):
            near = fld.neigh[2][zz]
            if near not in data.Registers and near not in data.Shareds: data.Shareds[near]="shared_fld[%2d][iz%s].%s"%(shrn/2, "PM"[near.coord[2]<0]+str(abs(near.coord[2]/2))+"mc"[near.coord[2]%2], 'xy'[shrn%2])
      global uniqdifN; uniqdifN=0
      if data.PrepareDifs:
        for fld in d.fields: 
          if fld.neigh[0][3].name and fld.neigh[0][3] not in data.Registers:
            print "  reg_R = %s;"%fld.neigh[0][3].name
            fld.neigh[0][3].name = "reg_R"; data.Registers[fld.neigh[0][3]] = "reg_R"
          fld.prepare_difs(time=time)
          if not fld.hasdz: uniqdifN-=1; fld.update(time=time); fld.save();
          if fld.neigh[0][2].name and fld.neigh[0][2] in data.Registers and (fld.typus!="S" or fld.proj==2):
            old = fld.neigh[0][2]; new = fld.neigh[0][3]
            #old.save();
            print "  %s = %s; // "%(old.name, new.name) , new.coord
            data.Registers[new] = data.Registers.pop(old)
            for Sother in data.Registers:
              if old.typus=="S" and Sother.coord==old.coord or \
                 (Sother.typus,Sother.proj)==("T",0) and list(Sother.coord)==[fld.coord[0]-NDT+1, fld.coord[1], (fld.coord[2]+1)%2] or \
                 (Sother.typus,Sother.proj)==("V",0) and list(Sother.coord)==[fld.coord[0]-NDT-1, fld.coord[1], (fld.coord[2]+1)%2]:
                Snew = data((Sother.coord[0]+2*NDT, Sother.coord[1], Sother.coord[2]),Sother.proj,1)
                print "  %s = %s; //additional--->> "%(data.Registers[Sother], Snew.name) , Snew.coord
                data.Registers[Snew] = data.Registers.pop(Sother)
          #renewing names
          for dd in self.dmds: dd.upgrade_names()
        print "  __syncthreads();"
        uniqdifN=0
        for fld in d.fields:
          if fld.hasdz: fld.update(time=time); fld.save();
          else        : uniqdifN+=1
      else:
        for dd in self.dmds: dd.upgrade_names()
        print "  __syncthreads();"
        for fld in d.fields:
          if fld.neigh[0][3].name and fld.neigh[0][3] not in data.Registers:
            print "  reg_R = %s;"%fld.neigh[0][3].name
            fld.neigh[0][3].name = "reg_R"; data.Registers[fld.neigh[0][3]] = "reg_R"
          fld.update(time=time)
          fld.save();
          if fld.neigh[0][2].name and fld.neigh[0][2] in data.Registers and (fld.typus!="S" or fld.proj==2):
            old = fld.neigh[0][2]; new = fld.neigh[0][3]
            #old.save();
            print "  %s = %s; // "%(old.name, new.name) , new.coord
            data.Registers[new] = data.Registers.pop(old)
            for Sother in data.Registers:
              if old.typus=="S" and Sother.coord==old.coord or \
                 (Sother.typus,Sother.proj)==("T",0) and list(Sother.coord)==[fld.coord[0]-NDT+1, fld.coord[1], (fld.coord[2]+1)%2] or \
                 (Sother.typus,Sother.proj)==("V",0) and list(Sother.coord)==[fld.coord[0]-NDT-1, fld.coord[1], (fld.coord[2]+1)%2]:
                Snew = data((Sother.coord[0]+2*NDT, Sother.coord[1], Sother.coord[2]),Sother.proj,1)
                print "  %s = %s; //additional--->> "%(data.Registers[Sother], Snew.name) , Snew.coord
                data.Registers[Snew] = data.Registers.pop(Sother)
          #renewing names
          for dd in self.dmds: dd.upgrade_names()
      time+=0.5
    #if data.Atype[0]=='I': print "  RPOINT_CHUNK_SHIFT"
    print "}"
  def finalize(self):
    #print "it-=dStepT; ix-=dStepX; r0-=dStepRag;"
    #for fld in self.dmds[-1].fields: fld.save();
    data.Registers.clear()
    data.Shareds.clear()


def make_DTorre(typus, vpml=0, atype="D", spml=0, Disp=0, LargeNV=0): #type=0 or 1
#  for dmd in Plaster.Dmd_shifts:
#    for datN, dat_sh in enumerate(Plaster.DATAshifts):
#       abs_c = map(sum, zip((0*2*NDT,0*2*NDT), dat_sh[:2], dmd[0])) + [dat_sh[2],]
#       print "{%s,%s,%s}, "%(abs_c[0], abs_c[1], abs_c[2]),

  data.inPMLv=vpml; data.Atype=atype; data.PMLS=spml; data.LargeNV=LargeNV; data.Disp=Disp
  data.PrepareDifs = False#data.Atype=="D" and not data.PMLS #False#data.Atype=="D" and not data.PMLS
  DT = DiamondTorre(typus=typus)
  DT.prepare()
  DT.loop()
  DT.finalize()
