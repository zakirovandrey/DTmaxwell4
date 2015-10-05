//im3D считывает и визуализирует трёхмерные поля,
//получение исходного кода: bzr checkout bzr+ssh://photon/Save/BZR-for-all/lev/im3D
//автор: Вадим Левченко VadimLevchenko@mail.ru
// запуск: ./im3D <имя-файла-массива> [<имя-файла-массива> ...]
//целевой размер массива от 100 до 1500 элементов по каждой координате
//предполагается, что файлы массивов записаны в формате массивов aivlib-а или drp

#include "fpal.h"
#include "im2D.h"
#include "im3D.hpp"

#include "cuda_math.h"
image2D im2D;
image_pars imHost; __constant__ image_pars im;
__constant__ im3D_pars im3D;

float runTime=0.0, SmoothFPS=0.0;
bool recalc_at_once=true, recalc_always=false, save_anim_flag=false, draw_edges_flag=false;
int anim_acc=0, render_type=2;
//float3 viewRotation;
//float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);

texture<float, cudaTextureType3D> data3D_tex;
cudaArray* data3D_texArray=0;
char* optfName="im3DI.opt";//Имя файла для сохранения опций визуализации
FILE* gpPipe=0;

//#include <string.h>
#include <fcntl.h>
//#include <unistd.h>
//#include <time.h>

#include <malloc.h>

char WinTitle[1024],* baseTitleStr="2", addTitleStr[5]; int TitleStrInd=0;
int optfid=-1;

float calcTime, calcPerf; int TimeStep;
char* im3D_pars::reset_title() {
  char* pTitle=WinTitle, TitleStr[20];
  strcpy(TitleStr,baseTitleStr);
  strncpy(TitleStr+strlen(baseTitleStr),addTitleStr,4);
  if(fName) { sprintf(pTitle, "%s", fName); pTitle += strlen(pTitle); }
  if(strpbrk(TitleStr,"23")) { sprintf(pTitle, " (%dx%dx%d)", Nx,Ny,Nz); pTitle += strlen(pTitle); }
  if(strpbrk(TitleStr,"xyzXYZ")) { sprintf(pTitle, "/(%dx%dx%d)", ix0,iy0,iz0); pTitle += strlen(pTitle); }
  if(strpbrk(TitleStr,"aA=-+_06789")) { sprintf(pTitle, " %g<f<%g", imHost.fmin,imHost.fmax); pTitle += strlen(pTitle); }
  if(strpbrk(TitleStr,"pP")) { sprintf(pTitle, " pal:(%g)^%g*%g*%g;", imHost.pscale, imHost.gamma_pal, imHost.brightness_coff, imHost.max_rgb); pTitle += strlen(pTitle); }
  if(strpbrk(TitleStr,"bG")) { sprintf(pTitle, " calc: %.2f sec, %.2fG cells/sec; %d steps;", 1e-3*calcTime, calcPerf, TimeStep); pTitle += strlen(pTitle); }
  if(strpbrk(TitleStr,"tT\20")) { sprintf(pTitle, " transp: %s,%d", imHost.transparency_discrete_flag?"discr":"mode",imHost.transparency_mode); pTitle += strlen(pTitle); }
  //if(strpbrk(TitleStr,"gG")) { sprintf(pTitle, " ", ); pTitle += strlen(pTitle); }
  //sprintf(WinTitle, " %.1f fps", , ,recalc_always?SmoothFPS:1000./runTime);
  //printf(WinTitle, " render: %.1f fps", , recalc_always?SmoothFPS:1000./runTime);
  return WinTitle;
}

char* getfName(im3D_pars& im3D) {
  static char fName[]="image._______________";
  if(im3D.fName) strncpy(fName, im3D.fName, sizeof(fName)-1)[sizeof(fName)-1] = 0;
  if(strrchr(fName,'.')) strrchr(fName,'.')[0] = 0;
  return fName;
}

bool save_png(im3D_pars& im3D, image_pars& im) {
  char png_name[1024]; sprintf(png_name, "%s_%d.png", getfName(im3D), im.nFrame);
  im2D.out2png(png_name);
  return false;
}
__global__ void save_gp3D();
const int tileSz=16, tilesN=16;

bool save_gp(im3D_pars& im3D, image_pars& im) {
  char gp_name[1024], png_name[1024], fName[]="image._______________";
  if(im3D.fName) strncpy(fName, im3D.fName, sizeof(fName)-1)[sizeof(fName)-1] = 0;
  if(strrchr(fName,'.')) strrchr(fName,'.')[0] = 0;
  sprintf(png_name, "%s/%s_%d.png", im3D.drop_dir, fName, im.nFrame);
  im2D.out2png(png_name);
  sprintf( gp_name, "%s/%s.%d.gp", im3D.drop_dir, fName, im.nFrame);
  //sprintf( gp_name, "a.gp", fName, im.nFrame);
  FILE* gp=fopen(gp_name, "w"),* old_stdout=stdout;
  fprintf(gp, "unset key\n");
  fprintf(gp, "unset border\n");
  fprintf(gp, "unset xtics\n");
  fprintf(gp, "set x2tics border\n");
  fprintf(gp, "set x2range [%g:%g]\n", im.fmin, im.fmax);
  fprintf(gp, "unset ytics\n");
  //fprintf(gp, "load \"labels.gp\"\n");
  //printf("viewRotation: %g, %g\n", im3D.viewRotation[0], im3D.viewRotation[1]);
  //printf("viewTranslation: %g, %g, %g\n", im3D.viewTranslation[0], im3D.viewTranslation[1], im3D.viewTranslation[2]);
  const int Sgp=(tilesN-1)*tileSz;
  stdout = gp;
  exit_if_ERR(cudaThreadSynchronize());
  save_gp3D <<<dim3((im2D.Nx+Sgp-1)/Sgp,(im2D.Ny+Sgp-1)/Sgp),dim3(tilesN,tilesN)>>>();
  exit_if_ERR(cudaThreadSynchronize());
  stdout = old_stdout;
  fprintf(gp, "plot[0:%g][0:%g] \"%s\" binary filetype=png dx=1 dy=1 with rgbimage\n", float(im3D.bNx), float(im3D.bNy), png_name);
  fprintf(gp, "pause -1\n");
  fclose(gp);
  if(type_diag_flag>=0) printf("Зарамочное оформление сохранено в %s\n", gp_name);
  return false;
}

float get_val_from_arr3D(int ix, int iy, int iz);
Arr3D_pars& set_lim_from_arr3D();
void reset();

void im3D_pars::clear4exit() {
  im2D.clear();
  exit_if_ERR(cudaFreeArray(data3D_texArray));
}
void save_bmp4backgrownd();
struct any_idle_func_struct {
  virtual void step() {}
} xyz_void,* xyz=&xyz_void;
struct idle_func_struct3D: public any_idle_func_struct {
  float* par, val;
  void set(float* _par, float _val) { par = _par; val = _val; }
  void step() { *par += val; }
} xyz3D;
struct idle_func_struct2D: public any_idle_func_struct {
  int* i0, N, di;
  void set(int* _i0, int _N, int _di) { i0=_i0; N=_N; di=_di; }
  void step() { *i0 += di; if(*i0<0) *i0=N-1; else if(*i0>=N) *i0=0; }
} xyz2D;
struct idle_func_calc: public any_idle_func_struct {
  float t;
  void step();
} icalc;
struct idle_func_calcNdrop: public idle_func_calc {
  FILE* sensorsStr;
  int* sensors;
  int Nsensors;
  idle_func_calcNdrop(): Nsensors(0), sensors(0), sensorsStr(0) {}
  ~idle_func_calcNdrop() { delete sensors; }
  void add_sensor(int ix, int iy, int iz) {
    int* pi=sensors;
    for(int i=0; i<Nsensors; i++, pi+=3) if(pi[0] == ix && pi[1] == iy && pi[2] == iz)
      { printf("Сенсор (%d,%d,%d) уже задан. Вы делаете что-то не то!\n", ix, iy, iz); return; }
    printf("Создаю новый сенсор в точке (%d,%d,%d), файл <sensors.dat> будет очищен.\n", ix, iy, iz);
    Nsensors++;
    if(sensors == 0) sensors = (int*)malloc(Nsensors*3*sizeof(int));
    else sensors = (int*)realloc(sensors, Nsensors*3*sizeof(int));
    pi = sensors+3*(Nsensors-1);
    pi[0] = ix; pi[1] = iy; pi[2] = iz;
    sensorsStr = fopen("sensors.dat", "w");
    fclose(sensorsStr);
  }
  char* save_section(im3D_pars& im3D, image_pars& im) {
    printf("f(%d,%d,%d) = %g\n", im3D.ix0, im3D.iy0, im3D.iz0, get_val_from_arr3D(im3D.ix0, im3D.iy0, im3D.iz0));
    static char dat_name[1024]; sprintf(dat_name, "%s/%s_%d.dat", im3D.drop_dir, getfName(im3D), im.nFrame);
    FILE* dat=fopen(dat_name, "w");
    for(int i=0; i<im3D.Nx; i++) fprintf(dat, "%d %g\n", i, get_val_from_arr3D(i, im3D.iy0, im3D.iz0));
    fprintf(dat, "\n\n");
    for(int i=0; i<im3D.Ny; i++) fprintf(dat, "%d %g\n", i, get_val_from_arr3D(im3D.ix0, i, im3D.iz0));
    fprintf(dat, "\n\n");
    for(int i=0; i<im3D.Nz; i++) fprintf(dat, "%d %g\n", i, get_val_from_arr3D(im3D.ix0, im3D.iy0, i));
    fclose(dat);
    return dat_name;
  }
  void step() {
    idle_func_calc::step();
    if(Nsensors==0) return;
    sensorsStr = fopen("sensors.dat", "a");
    fprintf(sensorsStr, "%g", t);
    int* pi=sensors;
    for(int i=0; i<Nsensors; i++, pi+=3) fprintf(sensorsStr, "\t%g", get_val_from_arr3D(pi[0], pi[1], pi[2]));
    fprintf(sensorsStr, "\n");
    fclose(sensorsStr);
  }
} icalcNdrop;
void add_sensor(int ix, int iy, int iz) { icalcNdrop.add_sensor(ix, iy, iz); }

int print_help();

void im3D_pars::print_help() {
  ::print_help();
  printf("\
======= Общее управление программой:\n\
 «ESC» \tВыход из программы\n\
  3¦2  \tпереключает рендеринг 3D¦2D в сечениях (%dD)\n\
<Enter¦BackSpace>\tПереход к следующему¦предыдущему массиву\n\
  w¦W  \tСохранение текущего набора опций визуализации в файл «%s»¦то же, но предыдущий набор не переписывается, можно сохранить произвольное число наборов последовательно\n\
  r¦R  \tЗагрузка ранее сохранённых наборов опций последовательно¦загрузка без перехода к следующему набору\n\
  f¦F  \tПереход к началу¦концу файла сохранённых наборов опций\n\
  v¦V  \tУвеличение¦уменьшение уровня вывода диагностики (%d)\n\
«Ctr-v»\tПечатает диагностику, особено актуально, если заголовок окна не виден\n\
  s¦S  \tСохранение картинки в формате png|вместе с зарамочным оформлением в gnuplot\n\
 #¦@¦$ \tВ режиме 3D переключение режима фона: сетка¦сохранённая картинка¦рёбра бокса\n\
   !   \tсохранить картинку для фона\n\
  m¦M  \tУменьшение¦увеличение шага вдоль луча для соответствующего изменения муара (%g), ВНИМАНИЕ: при мелком шаге может очень медленно прорисовывать\n\
  d¦D  \tУвеличение¦уменьшение плотности цвета при суммировании вдоль луча (%g)\n\
  a¦A  \tУстановка пределов палитры из пределов массива¦ то же из вычисленных ранее значений fMin..fMax\n\
«Ctr-a»\tВычисляет пределы массива, устнавливает fMin..fMax и пределы палитры\n\
   1   \tвключает/выключает рисование сохраняемых 1D сечений в gnuplot\n\
  o¦O  \tДля точки (x0,y0,z0): печатает в терминале значение текущего поля и выводит в файл сечения вдоль лучей, проходящих через неё¦Добавляет сенсор\n\
«Ctr-o»\tВыводит в окно gnuplot содержимое файла sensors.dat или (при отсутствии сенсоров) - сечения вдоль лучей\n\
======= Управление динамикой:\n\
  g¦G  \tОтключение¦включение постоянной перерисовки в цикле GLUT (%d)\n\
xyz¦XYZ\tВ режиме 2D: Увеличение¦уменьшение координат точки сечения параллелепипеда данных плоскостями сечений (%d,%d,%d)\n\
 yz¦XY \tВ режиме 3D: Вращение вокруг осей x,y вперёд¦назад (%g,%g)\n\
  z¦Z  \tВ режиме 3D: Приближение¦удаление объекта (%g)\n\
======= Управление мышью (L¦R¦M --- левая¦правая¦средняя кнопки):\n\
   L   \tВ режиме 2D переустанавливает срезы, исходя из координат выбранной точки\n\
 L¦R¦M \tВ режиме 3D: вращение¦изменение масштаба¦сдвиг рисунка\n\
 В районе палитры (верхние 20 точек):\n\
 L¦R¦M \tустанавливает нижний¦верхний пределы¦центр палитры, исходя из x-координаты выбранной точки\n\
  L¦R  \tВ режиме «Ctl-t» (бинарной прозрачности) делает цвет прозрачным¦видимым\n\
", render_type, optfName, type_diag_flag, tstep, density, recalc_always, ix0, iy0, iz0, viewRotation[0], viewRotation[1], viewTranslation[2]); imHost.print_help();
}
// normal          shift           Ctrl
//«DEL»
//`   45         ~    %^&*()    `1234567890 
//q e   u        Q E   UI       qwer yu   []
//     hjk ;'         HJK :"    asdfghjkl;'\
//    bn ,.          BN <>      zx  bnm,./ 
bool im3D_pars::key_func(unsigned char key, int x, int y) {
  recalc_at_once=true;
  size_t rs=0;
  //printf("%s: %c\n", TitleStr, key);
  addTitleStr[(TitleStrInd++)%4] = key;
  switch(key) {
  case 1: { Arr3D_pars& arr=set_lim_from_arr3D(); fMin = arr.fMin; fMax = arr.fMax; }
  case 'A': imHost.set_lim(fMin, fMax); return true;
  case 'a': { Arr3D_pars& arr=set_lim_from_arr3D(); imHost.set_lim(arr.fMin, arr.fMax); } return true;
  case 'i': ::reset(); return true;
  case 'w': if(optfid>=0) lseek(optfid,-int(sizeof(fpal_pars)+sizeof(im3D_pars4save)), SEEK_CUR);
  case 'W': if(optfid>=0) {
    rs=write(optfid, &imHost, sizeof(fpal_pars));
    rs=write(optfid, this, sizeof(im3D_pars4save));
  } recalc_at_once=false; return true;
  case 'R': if(optfid>=0) lseek(optfid,-int(sizeof(fpal_pars)+sizeof(im3D_pars4save)), SEEK_CUR);
  case 'r': if(optfid>=0) {
    rs=read(optfid, &imHost, sizeof(fpal_pars));
    rs=read(optfid, this, sizeof(im3D_pars4save));
  } return true;
  case 'f': if(optfid>=0) lseek(optfid,0, SEEK_SET); recalc_at_once=false; return true;
  case 'F': if(optfid>=0) lseek(optfid,0, SEEK_END); recalc_at_once=false; return true;
  case 22: recalc_at_once=false;
    printf("%s\nFrame %d (%.2f/%.2f fps), last run Times: %7.2f msec\n", WinTitle, imHost.nFrame, SmoothFPS, 1000./runTime, runTime);
    return true;
  case 'v': recalc_at_once=false; type_diag_flag++; return true;
  case 'V': recalc_at_once=false; type_diag_flag--; return true;
  case 'S': recalc_at_once=save_gp(*this, imHost); return true;
  case 's': recalc_at_once=save_png(*this, imHost); return true;
  case 'm': tstep /= sqrt(sqrt(2)); density /= sqrt(sqrt(2)); return true;
  case 'M': tstep *= sqrt(sqrt(2)); density *= sqrt(sqrt(2)); return true;
  case 'd': density *= sqrt(sqrt(2)); return true;
  case 'D': density /= sqrt(sqrt(2)); return true;
  case '@': imHost.draw_bmp4backgrownd_flag ^= true; return true;
  case '#': draw_mesh_flag ^= true; return true;
  case '$': draw_box_flag ^= true; return true;
  case '!': save_bmp4backgrownd(); return true;
  case 50: render_type=2; return true;
  case 51: render_type=3; return true;
  case 'g': recalc_always=false; return true;
  case 'G': recalc_always=true; return true;
  case 'O': recalc_at_once=false; icalcNdrop.add_sensor(ix0, iy0, iz0); return true;
  case 'o': recalc_at_once=false; icalcNdrop.save_section(*this, imHost); return true;
  case '1': recalc_at_once=false; {
    if(gpPipe) pclose(gpPipe);
    gpPipe = popen("gnuplot", "w");
    if(icalcNdrop.Nsensors == 0)
      fprintf(gpPipe, "set style data l;\nplot '%s' i 0 t '*,%d,%d', '' i 1 t '%d,*,%d', '' i 2 t '%d,%d,*'\n", icalcNdrop.save_section(*this, imHost), iy0, iz0, ix0, iz0, ix0, iy0);
    else {
      int* pi=icalcNdrop.sensors;
      fprintf(gpPipe, "set style data l;\nplot 'sensors.dat' u 1:2 t '%d,%d,%d'", pi[0],pi[1],pi[2]); pi+=3;
      for(int i=1; i<icalcNdrop.Nsensors; i++, pi+=3) fprintf(gpPipe, ", '' u 1:%d t '%d,%d,%d'", i+2, pi[0],pi[1],pi[2]);
      fprintf(gpPipe, "\n");
    }
    fflush(gpPipe);
    } return true;
  case 'b': xyz = &icalcNdrop; return true;
  case 'x': 
    if(render_type==2) { xyz2D.set(&ix0, Nx, 1); xyz = &xyz2D; xyz->step(); }
    else if(render_type==3) { xyz3D.set(&viewRotation[0], 0.5f); xyz = &xyz3D; xyz->step(); }
    return true;
  case 'X':
    if(render_type==2) { xyz2D.set(&ix0, Nx,-1); xyz = &xyz2D; xyz->step(); }
    else if(render_type==3) { xyz3D.set(&viewRotation[0],-0.5f); xyz = &xyz3D; xyz->step(); }
     return true;
  case 'y':
    if(render_type==2) { xyz2D.set(&iy0, Ny, 1); xyz = &xyz2D; xyz->step(); }
    else if(render_type==3) { xyz3D.set(&viewRotation[1], 0.5f); xyz = &xyz3D; xyz->step(); }
     return true;
  case 'Y':
    if(render_type==2) { xyz2D.set(&iy0, Ny,-1); xyz = &xyz2D; xyz->step(); }
    else if(render_type==3) { xyz3D.set(&viewRotation[1],-0.5f); xyz = &xyz3D; xyz->step(); }
     return true;
  case 'z':
    if(render_type==2) { xyz2D.set(&iz0, Nz, 1); xyz = &xyz2D; xyz->step(); }
    else if(render_type==3) { xyz3D.set(&viewTranslation[2], 0.01f); xyz = &xyz3D; xyz->step(); }
     return true;
  case 'Z':
    if(render_type==2) { xyz2D.set(&iz0, Nz,-1); xyz = &xyz2D; xyz->step(); }
    else if(render_type==3) { xyz3D.set(&viewTranslation[2],-0.01f); xyz = &xyz3D; xyz->step(); }
    return true;
  case 27: clear4exit(); exit(0);
  default: if(imHost.key_func(key, x, y)) return true;
  }
  recalc_at_once=false;
  if(rs==0) return false;
  return false;
}

int ox, oy;
int buttonState = 0;

#ifdef CLICK_BOOM
#include "params.h"
extern __constant__ TFSFsrc src;
#endif

void im3D_pars::mouse_func(int button, int state, int x, int y) {
  if(y<20 && state == GLUT_DOWN) {
    if(imHost.transparency_discrete_flag) {
      int ic=floor(0.5+(imHost.pscale)*float(x)/float(bNx));
      switch(button) {
        case 0: imHost.transparency_mode |= (1<<ic); break;
        case 1: imHost.transparency_mode ^= (1<<ic); break;
        case 2: imHost.transparency_mode &= ~(1<<ic); break;
      };
    } else {
    float f=imHost.fmin + x/float(bNx)*(imHost.fmax-imHost.fmin);
    switch(button) {
      case 0: imHost.set_lim(f,imHost.fmax); break; 
      case 2: imHost.set_lim(imHost.fmin,f); break; 
      case 1:
      float df=(f-imHost.fmin)>(imHost.fmax-f)?(f-imHost.fmin):(imHost.fmax-f);
      imHost.set_lim(f-df,f+df); break; 
    };
    if(type_diag_flag>=3) printf("mouse pal: %d,%d, button %d, state %d\n", x,y, button, state);
    recalc_at_once=true;
    }
    return;
  }
  if(render_type==3) {
    if (state == GLUT_DOWN) buttonState  |= 1<<button;
    else if(state == GLUT_UP) buttonState = 0;
    ox = x;
    oy = y;
  } else {
    if (state == GLUT_DOWN) { if(0<=x && x<bNx && 0<=y && y<bNy) reset0(x,bNy-1-y); }
  }
  #ifdef CLICK_BOOM
  shotpoint.srcXs=(-50+x)*dx;
  shotpoint.srcXa=(Ny-y+20)*dy;
  shotpoint.BoxMs=shotpoint.srcXs-5.1*dx; shotpoint.BoxPs=shotpoint.srcXs+5.1*dx;
  shotpoint.BoxMa=shotpoint.srcXa-5.1*dy; shotpoint.BoxPa=shotpoint.srcXa+5.1*dy;
  shotpoint.start=parsHost.iStep*Ntime;
//  if(shotpoint.BoxMs>Npmlx/2*NDT*dx && shotpoint.BoxPs<(Np-Npmlx/2)*NDT*dx && shotpoint.BoxPa<(Na-Npmly)*NDT*dy) {
  ftype lengthes[3] = {shotpoint.BoxPs-shotpoint.BoxMs, shotpoint.BoxPa-shotpoint.BoxMa, shotpoint.BoxMv-shotpoint.BoxPv};
  ftype boxDiagLength = sqrt(lengthes[0]*lengthes[0]+lengthes[1]*lengthes[1]+lengthes[2]*lengthes[2]);
  shotpoint.tStop = boxDiagLength/2.0/min(shotpoint.Vp,0.0001+shotpoint.Vs)+8/(M_PI*shotpoint.F0)+10*dt+shotpoint.start*dt;
    copy2dev( shotpoint, src ); shotpoint.check();
//  }
  #endif
  recalc_at_once=true;
  glutPostRedisplay();
}

void im3D_pars::motion_func(int x, int y) {
  if(type_diag_flag>=3) printf("motion pal: %d,%d -> %d,%d\n",ox,oy, x,y);
  if(y<20) {
    return;
  }
  float dx, dy;
  dx = (float)(x - ox);
  dy = (float)(y - oy);

  if (buttonState == 4) // right = zoom
    viewTranslation[2] += dy / 100.0f;
  else if (buttonState == 2) { // middle = translate
    viewTranslation[0] += dx / 100.0f;
    viewTranslation[1] -= dy / 100.0f;
  }
  else if (buttonState == 1) { // left = rotate
    viewRotation[0] += dy / 5.0f;
    viewRotation[1] += dx / 5.0f;
  }

  ox = x;
  oy = y;
  recalc_at_once=true;
  glutPostRedisplay();
}
//int cfX=0, cfY=0;

__global__ void im3Ddraw_xy() {
  int x=blockIdx.x*blockDim.x+threadIdx.x, ix=x*im3D.x_zoom;
  int y=blockIdx.y*blockDim.y+threadIdx.y, iy=y*im3D.y_zoom;
  if(iy<im3D.Ny && ix<im3D.Nx) im.bmp[im3D.xy_sh+x+y*im3D.bNx] = im.get_color(tex3D(data3D_tex, ix,iy,im3D.iz0));
}
__global__ void im3Ddraw_xz() {
  int x=blockIdx.x*blockDim.x+threadIdx.x, ix=x*im3D.x_zoom;
  int z=blockIdx.y*blockDim.y+threadIdx.y, iz=z*im3D.z_zoom;
  if(iz<im3D.Nz && ix<im3D.Nx) im.bmp[im3D.xz_sh+x+z*im3D.bNx] = im.get_color(tex3D(data3D_tex, ix,im3D.iy0,iz));
}
__global__ void im3Ddraw_zy() {
  int z=blockIdx.x*blockDim.x+threadIdx.x, iz=z*im3D.z_zoom;
  int y=blockIdx.y*blockDim.y+threadIdx.y, iy=y*im3D.y_zoom;
  if(iy<im3D.Ny && iz<im3D.Nz) im.bmp[im3D.zy_sh+z+y*im3D.bNx] = im.get_color(tex3D(data3D_tex, im3D.ix0,iy,iz));
}
__global__ void im3Ddraw_zx() {
  int z=blockIdx.x*blockDim.x+threadIdx.x, iz=z*im3D.z_zoom;
  int x=blockIdx.y*blockDim.y+threadIdx.y, ix=x*im3D.x_zoom;
  if(iz<im3D.Nz && ix<im3D.Nx) im.bmp[im3D.zx_sh+z+x*im3D.bNx] = im.get_color(tex3D(data3D_tex, ix,im3D.iy0,iz));
}
__global__ void im3Ddraw_yz() {
  int y=blockIdx.x*blockDim.x+threadIdx.x, iy=y*im3D.y_zoom;
  int z=blockIdx.y*blockDim.y+threadIdx.y, iz=z*im3D.z_zoom;
  if(iy<im3D.Ny && iz<im3D.Nz) im.bmp[im3D.yz_sh+y+z*im3D.bNx] = im.get_color(tex3D(data3D_tex, im3D.ix0,iy,iz));
}
__global__ void draw_pal() {
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  uchar4 col=im.get_color(im.fmin+(float(x)/im3D.bNx)*(im.fmax-im.fmin));
  uchar4* bmp = im.bmp+im3D.pal_sh;
  for(int y=0; y<20; y++, bmp += im3D.bNx) bmp[x] = col;
}
__global__ void negate() {
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  uchar4 col=make_uchar4(255,255,255,255);
  uchar4* bmp = im.bmp+x;
  for(int y=0; y<im3D.bNy; y++, bmp += im3D.bNx) bmp[0] = col-bmp[0];
}
float invViewMatrix[12];
typedef struct {
  float4 m[3];
} float3x4;

//Код 3D рендеринга позаимствован из примеров cuda5.5: 2_Graphics/volumeRender/volumeRender_kernel.cu
__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
struct Ray {
  float3 o;   // origin
  float3 d;   // direction
};

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar) {
  // compute intersection of ray with all six bbox planes
  float3 invR = make_float3(1.0f) / r.d;
  float3 tbot = invR * (boxmin - r.o);
  float3 ttop = invR * (boxmax - r.o);

  // re-order intersections to find smallest and largest on each axis
  float3 tmin = fminf(ttop, tbot);
  float3 tmax = fmaxf(ttop, tbot);

  // find the largest tmin and the smallest tmax
  float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

  *tnear = largest_tmin;
  *tfar = smallest_tmax;

  return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uchar4 rgbaFloatToInt(float4 rgba, uchar4 bk) {
  float a=rgba.w, da=(1.-a)/255.;
  rgba.x = __saturatef(bk.x*da+a*rgba.x);   // clamp to [0.0, 1.0]
  rgba.y = __saturatef(bk.y*da+a*rgba.y);
  rgba.z = __saturatef(bk.z*da+a*rgba.z);
  rgba.w = __saturatef(rgba.w);
  return make_uchar4((rgba.x*255.f), (rgba.y*255.f), (rgba.z*255.f), (rgba.w*255.f));
}

__device__ uchar4 rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return make_uchar4((rgba.x*255.f), (rgba.y*255.f), (rgba.z*255.f), (rgba.w*255.f));
}

__device__ float smooth(float x) { return __saturatef(1.0f-x*x); } 
__device__ uchar4 backgrownd_mesh(Ray r, float3 boxmin, float3 boxmax) {
  float3 invR = make_float3(1.0f) / r.d;
  float3 tB = invR * (boxmin - r.o);
  float3 tT = invR * (boxmax - r.o);
  float3 mb=(float3&)im3D.MeshBox;

  float tz=r.d.z<0?tB.z:tT.z, xZ=r.o.x+r.d.x*tz, yZ=r.o.y+r.d.y*tz;
  float ty=r.d.y<0?tB.y:tT.y, zY=r.o.z+r.d.z*ty, xY=r.o.x+r.d.x*ty;
  float tx=r.d.x<0?tB.x:tT.x, yX=r.o.y+r.d.y*tx, zX=r.o.z+r.d.z*tx;
  float mval=im3D.Dmesh;
       if(xZ>=boxmin.x && yZ>=boxmin.y && xZ<=boxmax.x && yZ<=boxmax.y) mval=fminf(fabsf(remainderf(xZ, mb.x)), fabsf(remainderf(yZ, mb.y)));
  else if(zY>=boxmin.z && xY>=boxmin.x && zY<=boxmax.z && xY<=boxmax.x) mval=fminf(fabsf(remainderf(zY, mb.z)), fabsf(remainderf(xY, mb.x)));
  else if(yX>=boxmin.y && zX>=boxmin.z && yX<=boxmax.y && zX<=boxmax.z) mval=fminf(fabsf(remainderf(yX, mb.y)), fabsf(remainderf(zX, mb.z)));
  float delta=smooth(2.f*mval/im3D.Dmesh);
  float3 mcol=((float3&)(im3D.bkgr_col))*(1.0f-delta)+((float3&)(im3D.mesh_col))*delta;
  return make_uchar4(__saturatef(mcol.x)*255, __saturatef(mcol.y)*255, __saturatef(mcol.z)*255, 0);
}
//------------------------------
inline __device__ void set_boxMinMax(float3& boxMin, float3& boxMax) {
  boxMax = make_float3(0.5f*im3D.BoxFactor[0]*im3D.Nx, 0.5f*im3D.BoxFactor[1]*im3D.Ny, 0.5f*im3D.BoxFactor[2]*im3D.Nz);
  boxMin=make_float3(-1.0f)*boxMax;
}
inline __device__ void set_eyeRay(Ray& eyeRay, int x, int y) {
  const float dbNxy=2.0f/(im3D.bNx+im3D.bNy);
  const int Nsum=im3D.Nx+im3D.Ny+im3D.Nz;
  eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 0.32f*Nsum)));
  eyeRay.d = normalize(make_float3((x-im3D.bNx/2)*dbNxy, (y-im3D.bNy/2)*dbNxy, -2.0f));
  eyeRay.d = mul(c_invViewMatrix, eyeRay.d);
}

__device__ uchar4 backgrownd_box(Ray r, float3 boxmin, float3 boxmax) {
  float3 invR = make_float3(1.0f) / r.d;
  float3 tB = invR * (boxmin - r.o);
  float3 tT = invR * (boxmax - r.o);

  float tz=r.d.z<0?tB.z:tT.z, xZ=r.o.x+r.d.x*tz, yZ=r.o.y+r.d.y*tz;
  float ty=r.d.y<0?tB.y:tT.y, zY=r.o.z+r.d.z*ty, xY=r.o.x+r.d.x*ty;
  float tx=r.d.x<0?tB.x:tT.x, yX=r.o.y+r.d.y*tx, zX=r.o.z+r.d.z*tx;
  float zval=im3D.Dmesh;
       if(xZ>=boxmin.x && yZ>=boxmin.y && xZ<=boxmax.x && yZ<=boxmax.y) zval=fminf(fminf(xZ-boxmin.x, yZ-boxmin.y), fminf(boxmax.x-xZ, boxmax.y-yZ));
  else if(zY>=boxmin.z && xY>=boxmin.x && zY<=boxmax.z && xY<=boxmax.x) zval=fminf(fminf(xY-boxmin.x, zY-boxmin.z), fminf(boxmax.x-xY, boxmax.z-zY));
  else if(yX>=boxmin.y && zX>=boxmin.z && yX<=boxmax.y && zX<=boxmax.z) zval=fminf(fminf(zX-boxmin.z, yX-boxmin.y), fminf(boxmax.z-zX, boxmax.y-yX));
  float delta=smooth(zval/im3D.Dmesh);
  float3 zcol=((float3&)(im3D.bkgr_col))*(1.0f-delta)+((float3&)(im3D.box_col))*delta;
  return make_uchar4(__saturatef(zcol.x)*255, __saturatef(zcol.y)*255, __saturatef(zcol.z)*255, 255);
}

__device__ uchar4 backgrownd_meshNbox(Ray r, float3 boxmin, float3 boxmax) {
  float3 invR = make_float3(1.0f) / r.d;
  float3 tB = invR * (boxmin - r.o);
  float3 tT = invR * (boxmax - r.o);
  float3 mb=(float3&)im3D.MeshBox;

  float tz=r.d.z<0?tB.z:tT.z, xZ=r.o.x+r.d.x*tz, yZ=r.o.y+r.d.y*tz;
  float ty=r.d.y<0?tB.y:tT.y, zY=r.o.z+r.d.z*ty, xY=r.o.x+r.d.x*ty;
  float tx=r.d.x<0?tB.x:tT.x, yX=r.o.y+r.d.y*tx, zX=r.o.z+r.d.z*tx;
  float mval=im3D.Dmesh, zval=im3D.Dmesh;
  if(xZ>=boxmin.x && yZ>=boxmin.y && xZ<=boxmax.x && yZ<=boxmax.y) {
    mval=fminf(fabsf(remainderf(xZ, mb.x)), fabsf(remainderf(yZ, mb.y)));
    zval=fminf(fminf(xZ-boxmin.x, yZ-boxmin.y), fminf(boxmax.x-xZ, boxmax.y-yZ));
  } else if(zY>=boxmin.z && xY>=boxmin.x && zY<=boxmax.z && xY<=boxmax.x) {
    mval=fminf(fabsf(remainderf(zY, mb.z)), fabsf(remainderf(xY, mb.x)));
    zval=fminf(fminf(xY-boxmin.x, zY-boxmin.z), fminf(boxmax.x-xY, boxmax.z-zY));
  } else if(yX>=boxmin.y && zX>=boxmin.z && yX<=boxmax.y && zX<=boxmax.z) {
    mval=fminf(fabsf(remainderf(yX, mb.y)), fabsf(remainderf(zX, mb.z)));
    zval=fminf(fminf(zX-boxmin.z, yX-boxmin.y), fminf(boxmax.z-zX, boxmax.y-yX));
  }
  float mdel=smooth(2.f*mval/im3D.Dmesh), zdel=smooth(zval/im3D.Dmesh);
  float3 mcol=((float3&)(im3D.bkgr_col))*(1.0f-mdel)+((float3&)(im3D.mesh_col))*mdel, zcol=mcol*(1.0f-zdel)+((float3&)(im3D.box_col))*zdel;
  return make_uchar4(__saturatef(zcol.x)*255, __saturatef(zcol.y)*255, __saturatef(zcol.z)*255, 255);
}

__device__ void mk_pts(int x, int y, uchar4 col) {
  const int ps=2;
  if(x+1<ps || x+ps>=im3D.bNx || y+1<ps || y+ps>=im3D.bNy) return;
  for(int ix=1-ps; ix<ps; ix++) for(int iy=1-ps; iy<ps; iy++)
    im.bmp[(iy+y)*im3D.bNx + x+ix] = col;
}
__device__ void mk_box(int x, int y, uchar4 col) {
  if(x<0 || x+tileSz>=im3D.bNx || y<0 || y+tileSz>=im3D.bNy) return;
  for(int ix=0; ix<tileSz; ix++) im.bmp[y*im3D.bNx + x+ix] = im.bmp[(tileSz+y)*im3D.bNx + x+ix] = col;
  for(int iy=0; iy<tileSz; iy++) im.bmp[(iy+y)*im3D.bNx + x] = im.bmp[(iy+y)*im3D.bNx + x+tileSz] = col;
}
inline bool __device__ is_inside(float2 pt, float2 p0, float2 px, float2 py) {
  float v1=(p0.x - pt.x) * (px.y - p0.y) - (px.x - p0.x) * (p0.y - pt.y);
  float v2=(px.x - pt.x) * (py.y - px.y) - (py.x - px.x) * (px.y - pt.y);
  float v3=(py.x - pt.x) * (p0.y - py.y) - (p0.x - py.x) * (py.y - pt.y);
  return (v1*v2>=0.0 && v1*v3>=0.0 && v2*v3>=0.0);
}
inline float2 __device__ pt_inside(float2 pt, float2 p0, float2 px, float2 py) {
  float2 res;
  res.x = ((pt.x-p0.x)*(py.y-p0.y)-(pt.y-p0.y)*(py.x-p0.x))/((px.x-p0.x)*(py.y-p0.y)-(px.y-p0.y)*(py.x-p0.x));
  res.y = ((pt.x-p0.x)*(px.y-p0.y)-(pt.y-p0.y)*(px.x-p0.x))/((py.x-p0.x)*(px.y-p0.y)-(py.y-p0.y)*(px.x-p0.x));
  return res;
}
__global__ void save_gp3D() {
  __shared__ float2 fm[3][tilesN][tilesN];//координаты точки в области с сеткой
  __shared__ int hit[tilesN][tilesN];//индекс области попадания луча: 1-z 2-y 4-x 0-молоко
  const int Sgp=(tilesN-1)*tileSz;
  int x=blockIdx.x*Sgp+threadIdx.x*tileSz, y=blockIdx.y*Sgp+threadIdx.y*tileSz;
  float3 boxMin, boxMax; set_boxMinMax(boxMin, boxMax);
  Ray r; set_eyeRay(r, x,y);
  float3 invR = make_float3(1.0f) / r.d;
  float3 tB = invR * (boxMin - r.o);
  float3 tT = invR * (boxMax - r.o);
  float tz=r.d.z<0?tB.z:tT.z, xZ=r.o.x+r.d.x*tz, yZ=r.o.y+r.d.y*tz;
  float ty=r.d.y<0?tB.y:tT.y, zY=r.o.z+r.d.z*ty, xY=r.o.x+r.d.x*ty;
  float tx=r.d.x<0?tB.x:tT.x, yX=r.o.y+r.d.y*tx, zX=r.o.z+r.d.z*tx;
  fm[2][threadIdx.x][threadIdx.y] = make_float2(xZ, yZ);
  fm[1][threadIdx.x][threadIdx.y] = make_float2(zY, xY);
  fm[0][threadIdx.x][threadIdx.y] = make_float2(yX, zX);
  if(xZ>=boxMin.x && yZ>=boxMin.y && xZ<=boxMax.x && yZ<=boxMax.y) hit[threadIdx.x][threadIdx.y] = 1; //mk_pts(x,y, red);}
  else if(zY>=boxMin.z && xY>=boxMin.x && zY<=boxMax.z && xY<=boxMax.x) hit[threadIdx.x][threadIdx.y] = 2; //mk_pts(x,y, green);}
  else if(yX>=boxMin.y && zX>=boxMin.z && yX<=boxMax.y && zX<=boxMax.z) hit[threadIdx.x][threadIdx.y] = 4; //mk_pts(x,y, blue);}
  else hit[threadIdx.x][threadIdx.y] = 0;
  __syncthreads();

  int hitA=0, hitM=0;
  if(threadIdx.x<tilesN-1 && threadIdx.y<tilesN-1) {
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) {
      int h=hit[threadIdx.x+i][threadIdx.y+j];
      if(h>0) { hitA++; hitM |= h; }
    }
  }
  int cs=abs(2*hitM-7)/2;
  if(hitA==0 || hitA==4 || cs>=3) return;
  bool is4tick=false, is4bnd=false, is4axis=false;
  is4bnd = hitM==1 || hitM==2 || hitM==4;
  is4axis= hitM==3 || hitM==5 || hitM==6;
  int cp=(cs+1)%3, cm=(cs+2)%3;
  float2 tick_sh={0.0,0.0}, tick2sh={0.0,0.0}; float tick_val;
  const float tick_gap=20.;
  float2 pt, spt={0.,0.}; float bMax[]={boxMax.x,boxMax.y,boxMax.z}, bMin[]={boxMin.x,boxMin.y,boxMin.z};
  int labN=(blockIdx.x*(tilesN-1)+threadIdx.x)+gridDim.x*(tilesN-1)*(blockIdx.y*(tilesN-1)+threadIdx.y);
  if(is4axis) {
    float2 p0=fm[cm][threadIdx.x][threadIdx.y], px=fm[cm][threadIdx.x+1][threadIdx.y], py=fm[cm][threadIdx.x][threadIdx.y+1];
    if(fabs(p0.x-bMax[cs])<fabs(p0.x-bMin[cs])) { pt.x = bMax[cs]; spt.x = tick_gap; }
    else { pt.x = bMin[cs]; spt.x = -tick_gap; }
    pt.y = fabs(p0.y-bMax[cp])<fabs(p0.y-bMin[cp])?bMax[cp]:bMin[cp];
    tick_sh = pt_inside(pt, p0,px,py);
    tick2sh = pt_inside(pt+spt, p0,px,py);
    printf("set label %d \"%c\" at %g,%g front center\n", labN, "xyz?"[cs], x+tick2sh.x*tileSz,y+tick2sh.y*tileSz);
  } else if(is4bnd) {
    float2 fmin,fmax; fmin = fmax = fm[cs][threadIdx.x][threadIdx.y];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) {
      float2 f = fm[cs][threadIdx.x+i][threadIdx.y+j];
      if(f.x<fmin.x) fmin.x = f.x;
      if(f.y<fmin.y) fmin.y = f.y;
      if(f.x>fmax.x) fmax.x = f.x;
      if(f.y>fmax.y) fmax.y = f.y;
    }
    if(fmin.x<bMin[cp] || fmax.x>bMax[cp]) {// cM = cm;
      int mmin=floorf(fmin.y/im3D.MeshBox[cm]), mmax=floorf(fmax.y/im3D.MeshBox[cm]);
      if(mmin != mmax) is4tick = true;
      pt.x = fmin.x<bMin[cp]?bMin[cp]:bMax[cp]; spt.x = fmin.x<bMin[cp]?-tick_gap:tick_gap;
      pt.y = mmax*im3D.MeshBox[cm];
      tick_val = pt.y*im3D.step[cm];
    } else if(fmin.y<bMin[cm] || fmax.y>bMax[cm]) {// cM = cp;
      int mmin=floorf(fmin.x/im3D.MeshBox[cp]), mmax=floorf(fmax.x/im3D.MeshBox[cp]);
      if(mmin != mmax) is4tick = true;
      pt.x = mmax*im3D.MeshBox[cp];
      pt.y = fmin.y<bMin[cm]?bMin[cm]:bMax[cm]; spt.y = fmin.y<bMin[cm]?-tick_gap:tick_gap;
      tick_val = pt.x*im3D.step[cp];
    }
    if(is4tick) {
      float2 p0=fm[cs][threadIdx.x][threadIdx.y], px=fm[cs][threadIdx.x+1][threadIdx.y], py=fm[cs][threadIdx.x][threadIdx.y+1], p1=fm[cs][threadIdx.x+1][threadIdx.y+1];
      if(is_inside(pt, p0,px,py)) {
        tick_sh = pt_inside(pt, p0,px,py);
        tick2sh = pt_inside(pt+spt, p0,px,py);
      } else if(is_inside(pt, p1,py,px)) {
        tick_sh = 1.0-pt_inside(pt, p1,py,px);
        tick2sh = 1.0-pt_inside(pt+spt, p1,py,px);
      } else is4tick = false;
      if(is4tick) printf("set label %d \"%g\" at %g,%g front %s\n", labN, tick_val, x+tick2sh.x*tileSz,y+tick2sh.y*tileSz, (tick2sh.x<tick_sh.x)?"right":"left");
    }
  }
  uchar4 red=make_uchar4(255,0,0,0), green=make_uchar4(0,255,0,0), blue=make_uchar4(0,0,255,0);
  uchar4 ltred=make_uchar4(128,0,0,0), ltgreen=make_uchar4(0,128,0,0), ltblue=make_uchar4(0,0,128,0);
  if(is4axis) {
    mk_box(x,y, red);
    mk_pts(x+tick2sh.x*tileSz,y+tick2sh.y*tileSz, red);
  } else if(is4tick) {
    mk_box(x,y, blue);
    mk_pts(x+tick2sh.x*tileSz,y+tick2sh.y*tileSz, blue);
  } else if(is4bnd) mk_box(x,y, green);
  else mk_box(x,y, ltblue);
}
__global__ void render3D() {
  const float opacityThreshold = 0.95f;
  const float density=im3D.density, brightness=im.max_rgb;
  float3 boxMin, boxMax; set_boxMinMax(boxMin, boxMax);

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  //if ((x >= im3D.bNx) || (y >= im3D.bNy)) return;
  //if(x==0 && y==0) printf("block: %gx%gx%g\n", boxMax.x, boxMax.y, boxMax.z);

  // calculate eye ray in world space
  Ray eyeRay; set_eyeRay(eyeRay, x,y);
  const int Nsum=im3D.Nx+im3D.Ny+im3D.Nz;

  uchar4& vbmp=im.bmp[y*im3D.bNx + x];
  if(im.draw_bmp4backgrownd_flag && im.bmp4backgrownd != 0) vbmp = im.bmp4backgrownd[y*im3D.bNx + x];
  else if(im3D.draw_mesh_flag && im3D.draw_box_flag) vbmp = backgrownd_meshNbox(eyeRay, boxMin, boxMax);
  else if(im3D.draw_mesh_flag) vbmp = backgrownd_mesh(eyeRay, boxMin, boxMax);
  else if(im3D.draw_box_flag) vbmp = backgrownd_box(eyeRay, boxMin, boxMax);
  // find intersection with box
  float tnear, tfar;
  int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

  if (!hit) return;

  if(tnear < 0.0f) tnear = 0.0f;     // clamp to near plane
  if(tnear+im3D.tstep*Nsum<tfar) tfar = tnear+im3D.tstep*Nsum;
  // march along ray from front to back, accumulating color
  float4 sum = make_float4(0.0f);
  //float3 pos = eyeRay.o + eyeRay.d*tnear;
  //float3 step = eyeRay.d*im3D.tstep;
  const float3 SzfdBox=make_float3(im3D.Nx,im3D.Ny,im3D.Nz)/(boxMax-boxMin);
  float3 pos_sc = (eyeRay.o + eyeRay.d*tnear-boxMin)*SzfdBox-0.5f;
  const float3 step_sc = (eyeRay.d*im3D.tstep)*SzfdBox;
  const float pscale=im.pscale*0.01f, fscale=100.0f*im.fscale, fmin=0.5f-im.fmin*fscale;

  for(float t=tnear; t<tfar; t+=im3D.tstep, pos_sc += step_sc) {
    // read from 3D texture
    //float4 col = im.get_color_f4(tex3D(data3D_tex, pos_sc.x, pos_sc.y, pos_sc.z));
    float f = tex3D(data3D_tex, pos_sc.x, pos_sc.y, pos_sc.z);
    float4 col = tex1D(fpal_col_tex, 0.5f + pscale*tex1D(fpal_scale_tex, fmin+f*fscale));
    col.w *= density;

    // "under" operator for back-to-front blending
    //sum = lerp(sum, col, col.w);

    // pre-multiply alpha
    col.x *= col.w;
    col.y *= col.w;
    col.z *= col.w;
    // "over" operator for front-to-back blending
    sum = sum + col*(1.0f - sum.w);

    // exit early if opaque
    if (sum.w > opacityThreshold) break;
   // pos_sc += step_sc;
  }

  sum *= brightness;

  // write output color
  vbmp = rgbaFloatToInt(sum, vbmp);
  //if(threadIdx.x==0 && threadIdx.y==0) vbmp = make_uchar4(255,255,255,255);
}

void im3D_pars::save_bmp4backgrownd() {
try {
  uchar4* devPtr; size_t size;
  exit_if_ERR(cudaGraphicsMapResources(1, &im2D.resource, NULL));
  if(imHost.negate_flag) ::negate <<<bNx/NW,NW>>>();
  exit_if_ERR(cudaGraphicsResourceGetMappedPointer((void**) &devPtr, &size, im2D.resource));
  if(imHost.bmp4backgrownd != 0) exit_if_ERR(cudaFree(imHost.bmp4backgrownd));
  exit_if_ERR(cudaMalloc((void**) &imHost.bmp4backgrownd, size));
  exit_if_ERR(cudaMemcpy(imHost.bmp4backgrownd, devPtr, size, cudaMemcpyDeviceToDevice));
  im2D.unmapAfterDraw();
} catch(...) {
  printf("save_bmp4backgrownd: Возникла какая-то ошибка.\n");
}
}
void im3D_pars::recalc_im3D() {
try {
  imHost.bmp = im2D.map4draw();
  imHost.bind2draw();
  exit_if_ERR(cudaMemcpyToSymbol(im, &imHost, sizeof(imHost)));
  exit_if_ERR(cudaMemcpyToSymbol(im3D, this, sizeof(im3D_pars)));
  exit_if_ERR(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeof(float4)*3));
  //exit_if_ERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  //Pal via Tex
  int NxB=(Nx/x_zoom+NW-1)/NW, NyB=(Ny/y_zoom+NW-1)/NW, NzB=(Nz/z_zoom+NW-1)/NW;
  if(xy_sh>=0) im3Ddraw_xy <<<dim3(NxB,NyB),dim3(NW,NW)>>>();
  if(xz_sh>=0) im3Ddraw_xz <<<dim3(NxB,NzB),dim3(NW,NW)>>>();
  if(zy_sh>=0) im3Ddraw_zy <<<dim3(NzB,NyB),dim3(NW,NW)>>>();
  if(zx_sh>=0) im3Ddraw_zx <<<dim3(NzB,NxB),dim3(NW,NW)>>>();
  if(yz_sh>=0) im3Ddraw_yz <<<dim3(NyB,NzB),dim3(NW,NW)>>>();
  if(imHost.draw_flag) draw_pal <<<bNx/NW,NW>>>();
  if(imHost.negate_flag) ::negate <<<bNx/NW,NW>>>();
  imHost.nFrame++;
  imHost.unbindAfterDraw();
  im2D.unmapAfterDraw();
} catch(...) {
  printf("recalc_im3D: Возникла какая-то ошибка.\n");
}
}
void im3D_pars::recalc3D_im3D() {
try {
  // use OpenGL to build view matrix
  GLfloat modelView[16];
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glRotatef(-viewRotation[0], 1.0, 0.0, 0.0);
  glRotatef(-viewRotation[1], 0.0, 1.0, 0.0);
  glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2]);
  glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
  glPopMatrix();
  for(int i=0; i<12; i++) invViewMatrix[i] = modelView[4*(i&3)+i/4];
  exit_if_ERR(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeof(float4)*3));
  //copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
  imHost.bmp = im2D.map4draw();
  imHost.bind2draw();
  exit_if_ERR(cudaMemcpyToSymbol(im, &imHost, sizeof(imHost)));
  exit_if_ERR(cudaMemcpyToSymbol(im3D, this, sizeof(im3D_pars)));
  //exit_if_ERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  render3D <<<dim3(bNx/NW,bNy/NW),dim3(NW,NW)>>>();
  if(imHost.draw_flag) draw_pal <<<bNx/NW,NW>>>();
  if(imHost.negate_flag) ::negate <<<bNx/NW,NW>>>();
  imHost.nFrame++;
  imHost.unbindAfterDraw();
  im2D.unmapAfterDraw();
} catch(...) {
  printf("recalc3D_im3D: Возникла какая-то ошибка.\n");
}
}

#include <cufft.h>

inline __device__ float my_fabsC(float2& v) { return v.x;}//hypotf(v.x, v.y); }
inline __device__ int my_abs(int v) { return v>=0?v:-v; }
//inline __device__ int my_abs(int v) { return v==0?1:v>=0?v:-v; }

__global__ void cmplx2abs(cufftComplex *dataC, cufftReal *dataR) {
  //float* pC=(float*)(dataC+blockIdx.x*(blockDim.x/2+1));
  //dataR[blockIdx.x*blockDim.x+threadIdx.x] = pC[threadIdx.x];
  dataR[blockIdx.x*blockDim.x+threadIdx.x] = my_fabsC(dataC[blockIdx.x*(blockDim.x/2+1)+my_abs(blockDim.x/2-threadIdx.x)]);
}
inline void exit_if_ERR(cufftResult rs) {
  if(rs == CUFFT_SUCCESS) return;
  printf("Непонятная ошибка в cuFFT\n");
  throw(-1);
}
void makeFFTz(float* buf, int Nx, int Ny, int Nz) {
try {
  cufftHandle plan;
  cufftComplex *dataC; cufftReal *dataR;
  cudaMalloc((void**)&dataC, sizeof(cufftComplex)*(Nz/2+1)*Nx*Ny);
  cudaMalloc((void**)&dataR, sizeof(cufftReal)*Nz*Nx*Ny);
  exit_if_ERR(cudaMemcpy(dataR, buf, 4*Nz*Nx*Ny, cudaMemcpyHostToDevice));
  exit_if_ERR(cufftPlan1d(&plan, Nz, CUFFT_R2C, Nx*Ny));
  exit_if_ERR(cufftExecR2C(plan, dataR, dataC));
  exit_if_ERR(cudaThreadSynchronize());
  cmplx2abs <<<Nx*Ny,Nz>>>(dataC, dataR);
  exit_if_ERR(cudaThreadSynchronize());
  exit_if_ERR(cudaMemcpy(buf, dataR, 4*Nz*Nx*Ny, cudaMemcpyDeviceToHost));
  exit_if_ERR(cufftDestroy(plan));
  exit_if_ERR(cudaFree(dataC)); exit_if_ERR(cudaFree(dataR));
} catch(...) {
  printf("Ошибка в makeFFTz.\n");
}
}
void im3D_pars::initCuda(Arr3D_pars& arr) {
  // create transfer function texture
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  //exit_if_ERR(cudaMalloc3DArray(&data3D_texArray, &channelDesc, make_cudaExtent(Nx,Ny,Nz)));
  cudaMemcpy3DParms myparms = {0};
  myparms.srcPos = make_cudaPos(0,0,0);
  myparms.dstPos = make_cudaPos(0,0,0);
  myparms.srcPtr = make_cudaPitchedPtr(arr.Arr3Dbuf, Nx*sizeof(float), Nx, Ny);
  myparms.dstArray = data3D_texArray;
  myparms.extent = make_cudaExtent(Nx,Ny,Nz);
  myparms.kind = arr.inGPUmem?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice;
  exit_if_ERR(cudaMemcpy3D(&myparms));
  //if(draw_edges_flag) draw_edges(imHost.fmax);
  data3D_tex.normalized = false;//true;
  data3D_tex.filterMode = cudaFilterModePoint;//Linear; //filter_pal?cudaFilterModePoint:cudaFilterModeLinear;
  data3D_tex.addressMode[0] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  data3D_tex.addressMode[1] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  data3D_tex.addressMode[2] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
  exit_if_ERR(cudaBindTextureToArray(data3D_tex, data3D_texArray));
}
void reset() {
  imHost.reset();
  imHost.set_lim(-1.f,1.f);
  imHost.draw_flag = imHost.negate_flag = imHost.centric_pal = true;
  imHost.cyclic_pal = false;
}
void im3D_pars::init3D(Arr3D_pars& arr) {
  ::reset();
  optfid = open(optfName, O_RDWR|O_CREAT, 0644);
  if(optfid<0) printf("Не могу открыть файл %s, сохранение/загрузка наборов опций визуализации невозможна\n", optfName);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  exit_if_ERR(cudaMalloc3DArray(&data3D_texArray, &channelDesc, make_cudaExtent(Nx,Ny,Nz)));
  initCuda(arr);
}
void im3D_pars::recalc_func() {
  if(recalc_always || recalc_at_once) {
    if(recalc_at_once) recalc_at_once=false;
    else xyz->step();
    cudaTimer tm; tm.start();
    switch(render_type) {
    case 2: recalc_im3D(); break;
    case 3: recalc3D_im3D(); break;
    }
    runTime=tm.stop(); SmoothFPS = 0.9*SmoothFPS+100./runTime;
    if(type_diag_flag>=2) printf("Frame %d (%.2f/%.2f fps), last run Times: %7.2f msec\n", imHost.nFrame, SmoothFPS, 1000./runTime, runTime);
  }
}
