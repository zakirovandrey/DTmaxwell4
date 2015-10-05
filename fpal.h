#ifndef FPAL_H
#define FPAL_H
#include <cuda.h>
#include <stdio.h>

// палитры творчески позаимствованы из aivlib
#define b 0.0001
#define q 0.25
#define h 0.75
#define H 0.5
const float fpal_array[] = { -1,
  1,0,H, h,0,0, 1,H,0, 1,1,0, 0,h,0, 0,0,0, 0,1,0, 0,0,1, H,0,1, 1,0,1, 0,1,1, -1,//hacked rainbow_pal
  //1,0,H, h,0,0, 1,H,0, 1,1,0, 0,h,0, 0,0,0, 0,0,1, 0,1,0, H,0,1, 1,0,1, 0,1,1, -1,//hacked rainbow_pal
  1,1,1, 1,0,H, h,0,0, 1,H,0, 1,1,0, 0,h,0, 0,1,1, 0,0,1, 1,0,1, H,0,1, 1,1,1, -1,//hacked rainbow_pal
  //H,H,H, 0,0,0, 1,0,0, 1,H,0, 1,1,0, 0,1,0, 0,1,1, 0,0,1, 1,0,1, 1,1,1, H,H,H, -1,//cyclic_pal
  1,0,H, h,0,0, 1,H,0, 1,1,0, 0,h,0, 0,0,1, H,0,1, 1,0,1, 0,1,1, 1,1,1, -1,//hacked rainbow_pal
  0,0,0, 1,0,0, 1,H,0, 1,1,0, 0,1,0, 0,1,1, 0,0,1, 1,0,1, 1,1,1, -1,//rainbow_pal
  1,1,1, 1,0,1, 0,0,1, 0,1,1, 0,1,0, 1,1,0, 1,H,0, 1,0,0, 0,0,0, -1,//inv_rainbow_pal
  q,0,0, 1,0,0, 1,H,0, 1,1,0, 1,1,1, 1,0,1, 0,0,1, 0,1,1, 0,1,0, -1,//neg_pos1_pal
  h,h,0, 1,1,0, 1,H,0, 1,0,0, 1,1,1, 0,1,0, 0,1,1, 0,0,1, 1,0,1, -1,//neg_pos2_pal
  1,1,1, 1,0,0, 1,H,0, 1,1,0, 0,1,0, 0,1,1, 0,0,1, 1,0,1, q,0,q, -1,//positive_pal
  1,1,1, 1,0,0, 1,H,0, 1,1,0, 0,1,0, 0,1,1, 0,0,1, 1,0,1, -1,//positive_pal
         1,0,0, 1,H,0, 1,1,0, 0,1,0, 0,1,1, 0,0,1, 1,0,1,        -1,//color_pal
         1,0,0, 1,1,0, 0,1,0, 0,1,1, 0,0,1, 1,0,1,        -1,//color_pal
         1,1,0, 1,0,0, 0,1,0, 0,0,1, 0,1,1,        -1,//color_pal
         1,0,0, 0,1,0, 0,0,1, 0,1,1,        -1,//color_pal
  0,0,1, 0,0,0, 1,0,0, -1,//blue_red_pal
  b,b,b, H,H,H, 1,1,1, -1,//grey_pal
  1,1,1, H,H,H, b,b,b, -1,//inv_grey_pal
  b,b,b, 1,1,1, 1,0,0, -1,//black_red_pal
  0,1,0, 1,1,1, 0,0,1, -1,//green_blue_pal
  1,0,H, h,0,0, 1,0,0, 1,H,0, 1,1,0, 0,h,0, 0,0,0, 0,1,0, 0,0,1, H,0,1, 1,0,1, 0,h,h, 0,1,1, -1,//super rainbow_pal
};
#undef H
#undef h
#undef q
#undef b
//прогрессия «в 10 раз за 10 кликов»:1.2   1.5     2     2.5     3      4      5       6      8     10
const float scale_step_array[] = { 6./5., 5./4., 4./3., 5./4., 6./5., 4./3., 5./4.,  6./5., 4./3., 5./4.};

//В нормальном коде так не делают, но в cuda текстуры должны иметь глобальные имена, так декларируем и соответствующие массивы так же
texture<float4, cudaTextureType1D, cudaReadModeElementType> fpal_col_tex;
texture<float, cudaTextureType1D> fpal_scale_tex;
cudaArray* fpal_col_texArray=0,* fpal_scale_texArray=0;

//------------------------------------
struct fpal_pars {
  float fmin, fmax, fscale, max_rgb;
  int start_pal; float pscale;
  bool cyclic_pal, centric_pal, filter_pal, draw_flag, negate_flag, logFlag, transparency_discrete_flag;
  int scale_step, gamma_step, max_rgb_step, brightness_coff_step, transparency_mode;
  float gamma_pal, brightness_coff;
 public:
  void reset() {
    cyclic_pal = centric_pal = filter_pal = negate_flag = transparency_discrete_flag = false;
    start_pal = 1; pscale = 1.0f;
    transparency_mode = 1;
    max_rgb = 1.0;
    draw_flag = false; logFlag = false;
    scale_step = gamma_step = max_rgb_step = 0; gamma_pal = 1.0;
    brightness_coff_step = 1; brightness_coff = scale_step_array[0];
  }
  void set_lim(float _fmin=0.0f, float _fmax=1.0f) {
    fmin = _fmin; fmax = _fmax;
    fscale = fmax>fmin?1.0f/(fmax-fmin):0.0;
  }
  void change_pal() {
    do { start_pal += 3; } while(fpal_array[start_pal] >= 0.0f);
    start_pal++;
    if(start_pal >= sizeof(fpal_array)/sizeof(float)) start_pal = 1;
  }
  void change_pal_back() {
    if(start_pal<=1) start_pal = sizeof(fpal_array)/sizeof(float)-1;
    else start_pal--;
    do { start_pal -= 3; } while(fpal_array[start_pal-1] >= 0.0f);
  }
  __device__ uchar4 invert_color(uchar4 c4) {
    return make_uchar4((c4.x+128)%256,(c4.y+128)%256,(c4.z+128)%256,255);
  }
  __device__ float4 get_color_norm_f4(float f, float br=1.0f) {
    f = 0.5f + pscale*f;
    return tex1D(fpal_col_tex,f);
  }
  __device__ float4 get_color_f4(float f) {
    float fn=(f-fmin)*fscale, br=1.0f;
    if(cyclic_pal && (fn<0.0f || fn>1.0f)) {
      float fi=floorf(fn);
      br = pow(brightness_coff,0.5f*fi);
      fn -= fi;
    }
    return get_color_norm_f4(0.01f*tex1D(fpal_scale_tex, 0.5f+100.0f*fn), br);
  }
  __device__ uchar4 get_color_norm(float f, float br=1.0) {
    f = 0.5f + pscale*f; br *= max_rgb;
    float4 col=tex1D(fpal_col_tex,f);
    return make_uchar4(__saturatef(br*col.x)*255, __saturatef(br*col.y)*255, __saturatef(br*col.z)*255, 255);
  }
  __device__ uchar4 get_color(float f) {
    float fn=(f-fmin)*fscale, br=1.0f;
    if(cyclic_pal && (fn<0.0f || fn>1.0f)) {
      float fi=floorf(fn);
      br = pow(brightness_coff,0.5f*fi);
      fn -= fi;
    }
    return get_color_norm(0.01f*tex1D(fpal_scale_tex, 0.5f+100.0f*fn), br);
  }
  void exit_if_ERR(cudaError_t rs) {
    if(rs == cudaSuccess) return;
    printf("Непонятная ошибка в fpal_pars, программа сейчас завершится,\nПроверяйте параметры:\n");
    print_help();
    throw(-1);
  }
  void bind2draw() {
    const int Nsc=100;
    float scale_data[Nsc+1], pal_data[128*4];
    if(centric_pal) { const float fN=0.5f*Nsc, dfN=1./fN;
      for(int i=0; i<=Nsc/2; i++) {
        float v=fN*pow(i*dfN,gamma_pal);
        scale_data[Nsc/2+i] = fN+v;
        scale_data[Nsc/2-i] = fN-v;
      }
    } else { const float fN=Nsc, dfN=1./fN;
      for(int i=0; i<=Nsc; i++) scale_data[i] = fN*pow(i*dfN,gamma_pal);
    }
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    exit_if_ERR(cudaMallocArray(&fpal_scale_texArray, &channelDesc, Nsc+1));
    exit_if_ERR(cudaMemcpyToArray(fpal_scale_texArray, 0, 0, scale_data, (Nsc+1)*sizeof(float), cudaMemcpyHostToDevice));
    fpal_scale_tex.normalized = false;
    fpal_scale_tex.filterMode =  cudaFilterModeLinear;
    fpal_scale_tex.addressMode[0] = cyclic_pal?cudaAddressModeBorder:cudaAddressModeClamp;
    exit_if_ERR(cudaBindTextureToArray(fpal_scale_tex, fpal_scale_texArray));
    int Nc; for(Nc=0; fpal_array[start_pal+3*Nc] >= 0.0f; Nc++);
    pscale = Nc-1.0f;
    for(int i=0; i<Nc; i++) {
      int ic=start_pal+3*i;
      if(!negate_flag || fpal_array[ic]+fpal_array[ic+1]+fpal_array[ic+2]==0.0) {
        pal_data[4*i] = fpal_array[ic]; pal_data[4*i+1] = fpal_array[ic+1]; pal_data[4*i+2] = fpal_array[ic+2]; pal_data[4*i+3] = 1.0f;
      } else {
        pal_data[4*i] = 1.0-fpal_array[ic]; pal_data[4*i+1] = 1.0-fpal_array[ic+1]; pal_data[4*i+2] = 1.0-fpal_array[ic+2]; pal_data[4*i+3] = 1.0f;
      }
    }
    //if(centric_pal) pal_data[4*(Nc/2)+3] = pal_data[4*((Nc-1)/2)+3] = 0.0f;
    //else pal_data[3] = pal_data[4*(Nc-1)+3] = 0.0f;
    if(transparency_discrete_flag) {
      for(int i=0; i<Nc; i++) pal_data[4*i+3] = (transparency_mode&(1<<i))==0?1.0f:0.0f;
    } else {
      float km=1.0*M_PI*transparency_mode, phi=0.0, inv=0.5;
      if(centric_pal) { km *= 2.0; phi = 0.5; }
      else if(transparency_mode<0) { phi = 1.0; inv = -inv; }
      if(transparency_mode>0) inv = -inv;
      for(int i=0; i<Nc; i++) pal_data[4*i+3] = pow(0.5+inv*cos(km*(i/(Nc-1.0)-phi)),brightness_coff);
    }
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    exit_if_ERR(cudaMallocArray(&fpal_col_texArray, &channelDesc2, Nc));
    exit_if_ERR(cudaMemcpyToArray(fpal_col_texArray, 0, 0, pal_data, Nc*sizeof(float4), cudaMemcpyHostToDevice));
    fpal_col_tex.normalized = false;//true;
    fpal_col_tex.filterMode = filter_pal?cudaFilterModePoint:cudaFilterModeLinear;
    fpal_col_tex.addressMode[0] = cudaAddressModeClamp;//cyclic_pal?cudaAddressModeWrap:cudaAddressModeClamp;
    exit_if_ERR(cudaBindTextureToArray(fpal_col_tex, fpal_col_texArray, channelDesc2));
  }
  void unbindAfterDraw() {
    exit_if_ERR(cudaFreeArray(fpal_scale_texArray));
    exit_if_ERR(cudaFreeArray(fpal_col_texArray));
  }
  void print_help() {
    printf("\
======= Палитра, цвета, масштабы:\n\
  p¦P \tВыбор палитры, в порядке уменьшения¦увеличения числа цветов (%.0f цвет[а/ов], начиная с поз. %d)\n\
«Ctl-p»\tПереключение вывода палитры в верх экрана (%d)\n\
  =¦- \tТочное (в 10 раз за 10 кликов) уменьшение¦увеличение пределов по цветовой оси (шаг %d)\n\
  +¦_ \tГрубое (в 10 раз) уменьшение¦увеличение пределов по цветовой оси (%g<f<%g)\n\
 0¦6¦7\tЦентрирование пределов палитры ¦ установка в 0 значения левого¦правого пределов\n\
  9¦8 \tУстановка пределов палитры в значения 1..9¦-1..1\n\
«Ctl-c»\tПереключение палитры из линейной в дискретную и обратно (%d). В частности - для уменьшения размера png\n\
  c¦C \tПереключение круговой¦центриванной палитры на ограниченную пределами и обратно (%d¦%d)\n\
  t¦T \tУвеличение¦уменьшение числа прозрачных цветов в палитре, (%d) д.б. делителем числа цветов в палитре (%d)\n\
«Ctl-t»\tПереключение на режим, в котором прозрачность каждого цвета в палитре кодируется битом числа %d (%d)\n\
  [¦] \tУменьшение¦увеличение показателя степени при нелинейном шкалировании по цветовой оси (%g, шаг %d)\n\
  /¦\\\tУменьшение¦увеличение яркости цветов в палитре, важно для круговой (до %g, шаг %d)\n\
  ?¦| \tУменьшение¦увеличение относительной яркости разных циклов в круговой палитре (%g раз, шаг %d)\n\
   i  \tСброс параметров в значения по умолчанию\n\
 «TAB»\tИнверсия цветов в палитре (кроме чёрного), для im3D решает проблему «грязи» на светлом фоне\n\
", pscale, start_pal, draw_flag, scale_step, fmin, fmax, filter_pal, cyclic_pal, centric_pal,
   transparency_mode, int(pscale)*(centric_pal?1:2), transparency_mode, transparency_discrete_flag,
   gamma_pal, gamma_step, max_rgb, max_rgb_step, pow(brightness_coff,0.5), brightness_coff_step);
  }
  bool key_func(unsigned char key, int x, int y) {
    float fc=centric_pal?0.5*(fmax+fmin):0.0;
    switch(key) {
    case 'i': reset(); break;
    case '-': set_lim(fc+(fmin-fc)*scale_step_array[scale_step], fc+(fmax-fc)*scale_step_array[scale_step]); scale_step = (scale_step+1)%10; break;
    case '=': scale_step = (scale_step+9)%10; set_lim(fc+(fmin-fc)/scale_step_array[scale_step], fc+(fmax-fc)/scale_step_array[scale_step]); break;
    case '_': set_lim(fc+(fmin-fc)*10.0f, fc+(fmax-fc)*10.0f); break;
    case '+': set_lim(fc+(fmin-fc)/10.0f, fc+(fmax-fc)/10.0f); break;
    case '8': set_lim(-1,1); break;
    case '9': set_lim(1,5); break;
    case '7': set_lim(fmin,0); break;
    case '6': set_lim(0,fmax); break;
    case '0': { float afmax=(fabs(fmin)>fabs(fmax))?fabs(fmin):fabs(fmax); set_lim(-afmax,afmax);} break;
    case 'c': cyclic_pal  ^= true; break;
    case 'C': centric_pal ^= true; break;
    case 't': transparency_mode++; break;
    case 'T': transparency_mode--; break;
    case  20: transparency_mode = 0; transparency_discrete_flag ^= true; break;
    case 3: filter_pal ^= true; break;
    case 'p': change_pal(); break;
    case 'P': change_pal_back(); break;
    case  16: draw_flag ^= true; break;
    case  9: negate_flag ^= true; break;
    case '\\':max_rgb *= scale_step_array[max_rgb_step]; max_rgb_step = (max_rgb_step+1)%10; break;
    case '/': max_rgb_step = (max_rgb_step+9)%10; max_rgb /= scale_step_array[max_rgb_step]; break;
    case '|': brightness_coff *= scale_step_array[brightness_coff_step]; brightness_coff_step = (brightness_coff_step+1)%10; break;
    case '?': brightness_coff_step = (brightness_coff_step+9)%10; brightness_coff /= scale_step_array[brightness_coff_step]; break;
    case 'l': logFlag=~logFlag; break;
    case 'L': logFlag=false; break;
    case '[': gamma_pal *= scale_step_array[gamma_step]; gamma_step = (gamma_step+1)%10; break;
    case ']': gamma_step = (gamma_step+9)%10; gamma_pal /= scale_step_array[gamma_step]; break;
    default : return false;
    } return true;
  }
};

//------------------------------------
struct image_pars: public fpal_pars {
  uchar4* bmp,* bmp4backgrownd;
  bool draw_bmp4backgrownd_flag;
  unsigned int nFrame;
  void reset() {
    fpal_pars::reset();
    nFrame = 0;
    bmp = bmp4backgrownd = 0;
    draw_bmp4backgrownd_flag = false;
  }
};
#endif//FPAL_H
