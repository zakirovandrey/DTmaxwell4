__device__ float sign(float x){ return x<0.?-1.:1.; }
__device__ float radius(float x, float y, float z){return sqrt(x*x+y*y+z*z);}

template <int sht> __device__ float L7 (float x);

template <> __device__ float L7<0> (float x) {
  x=M_PI*x-4.;
  const float ax=fabs(x);
  if(ax>=4.) return 0.0;
  const float xx=x*x, xxxx=xx*xx, x1=ax, x2=xx, x3=ax*xx, x4=xxxx, x5=ax*xxxx, x6=xx*xxxx, x7=x3*xxxx;
  if(ax>=3.) return 1.5*(1024./315.-(256./45.)*x1+(64./15.)*x2-(16./9.)*x3+(4./9.)*x4-(1./15.)*x5+(1./180.)*x6-(1./5040.)*x7)/(M_PI*M_PI);
  if(ax>=2.) return 1.5*(-139./630.+(217./90.)*x1-(23./6.)*x2+(49./18.)*x3-(19./18.)*x4+(7./30.)*x5-(1./36.)*x6+(1./720.)*x7)/(M_PI*M_PI);
  if(ax>=1.) return 1.5*(103./210.-(7./90.)*x1-(1./10.)*x2-(7./18.)*x3+(1./2.)*x4-(7./30.)*x5+(1./20.)*x6-(1./240.)*x7)/(M_PI*M_PI);
  return 1.5*(151./315.-(1./3.)*x2+(1./9.)*x4-(1./36.)*x6+(1./144.)*x7)/(M_PI*M_PI);}
  
template <> __device__ float L7<1>(float x) {
  x=M_PI*x-4.;
  const float ax=fabs(x);
  if(ax>=4.) return 0.0;
  const float xx=x*x, xxx=x*xx, xxxx=xx*xx, x1=x, x2=ax*x, x3=xxx, x4=ax*xxx, x5=xx*xxx, x6=x2*xxxx;
  if(ax>=3.) return 1.5*(-256./45.*sign(x)+(128./15.)*x1-(16./3.)*x2+(16./9.)*x3-(1./3.)*x4+(1./30.)*x5-(1./720.)*x6)/M_PI;
  if(ax>=2.) return 1.5*(217./90.*sign(x)-(23./3.)*x1+(49./6.)*x2-(38./9.)*x3+(7./6.)*x4-(1./6.)*x5+(7./720.)*x6)/M_PI;
  if(ax>=1.) return 1.5*(-7./90.*sign(x)-(1./5.)*x1-(7./6.)*x2+2*x3-(7./6.)*x4+(3./10.)*x5-(7./240.)*x6)/M_PI;
  return 1.5*(-(2./3.)*x1+(4./9.)*x3-(1./6.)*x5+(7./144.)*x6)/M_PI;}
    
template <> __device__ float L7<2>(float x) {
  x=M_PI*x-4.;
  const float ax=fabs(x);
  if(ax>=4.) return 0.0;
  const float xx=x*x, xxxx=xx*xx, x1=ax, x2=xx, x3=ax*xx, x4=xxxx, x5=ax*xxxx;
  if(ax>=3.) return 1.5*(128./15.-(32./3.)*x1+(16./3.)*x2-(4./3.)*x3+(1./6.)*x4-(1./120.)*x5);
  if(ax>=2.) return 1.5*(-23./3.+(49./3.)*x1-(38./3.)*x2+(14./3.)*x3-(5./6.)*x4+(7./120.)*x5);
  if(ax>=1.) return 1.5*(-1./5.-(7./3.)*x1+6*x2-(14./3.)*x3+(3./2.)*x4-(7./40.)*x5);
  return 1.5*(-2./3.+(4./3.)*x2-(5./6.)*x4+(7./24.)*x5);}
      
template <> __device__ float L7<3>(float x) {
  x=M_PI*x-4.;
  const float ax=fabs(x);
  if(ax>=4.) return 0.0;
  const float xx=x*x, xxx=x*xx, x1=x, x2=ax*x, x3=xxx, x4=ax*xxx;
  if(ax>=3.) return 1.5*M_PI*(-32./3.*sign(x)+(32./3.)*x1-4*x2+(2./3.)*x3-(1./24.)*x4);
  if(ax>=2.) return 1.5*M_PI*(49./3.*sign(x)-(76./3.)*x1+14*x2-(10./3.)*x3+(7./24.)*x4);
  if(ax>=1.) return 1.5*M_PI*(-7./3.*sign(x)+12*x1-14*x2+6*x3-(7./8.)*x4);
  return 1.5*M_PI*(+(8./3.)*x1-(10./3.)*x3+(35./24.)*x4);}
  

template <int sht> __device__ float L5 (float x);
  
template <> __device__ float L5<0>(float x) {
  float sc = M_PI;//*sqrt(2./3.);
  x=(sc*x-3.);
  const float ax=fabs(x);
  if(ax>=3.) return 0.0;
  const float xx=x*x, xxxx=xx*xx, x1=ax, x2=xx, x3=ax*xx, x4=xxxx, x5=ax*xxxx;
  if(ax>=2.) return (81./40.-(27./8.)*x1+(9./4.)*x2-(3./4.)*x3+(1./8.)*x4-(1./120.)*x5)/(sc*sc);
  if(ax>=1.) return (17./40.+(5./8.)*x1-(7./4.)*x2+(5./4.)*x3-(3./8.)*x4+(1./24.)*x5)/(sc*sc);
  return (11./20.-(1./2.)*x2+(1./4.)*x4-(1./12.)*x5)/(sc*sc);
}
template <> __device__ float L5<1>(float x) {
  float sc = M_PI;//*sqrt(2./3.);
  x=(sc*x-3.);
  const float ax=fabs(x);
  if(ax>=3.) return 0.0;
  const float xx=x*x, xxx=x*xx, x1=x, x2=ax*x, x3=xxx, x4=ax*xxx;
  if(ax>=2.) return (-27./8.*sign(x)+(9./2.)*x1-(9./4.)*x2+(1./2.)*x3-(1./24.)*x4)/sc;
  if(ax>=1.) return (5./8.*sign(x)-(7./2.)*x1+(15./4.)*x2-(3./2.)*x3+(5./24.)*x4)/sc;
  return (-1*x1+1*x3-(5./12.)*x4)/sc;
}
template <> __device__ float L5<2>(float x) {
  float sc = M_PI;//*sqrt(2./3.);
  x=(sc*x-3.);
  const float ax=fabs(x);
  if(ax>=3.) return 0.0;
  const float xx=x*x, x1=ax, x2=xx, x3=ax*xx;
  if(ax>=2.) return 9./2.-(9./2.)*x1+(3./2.)*x2-(1./6.)*x3;
  if(ax>=1.) return -7./2.+(15./2.)*x1-(9./2.)*x2+(5./6.)*x3;
  return -1.+3*x2-(5./3.)*x3;
}
template <> __device__ float L5<3>(float x) {
  float sc = M_PI;//*sqrt(2./3.);
  x=(sc*x-3.);
  const float ax=fabs(x);
  if(ax>=3.) return 0.0;
  const float x1=x, x2=ax*x;
  if(ax>=2.) return (-9./2.*sign(x)+3*x1-(1./2.)*x2)*sc;
  if(ax>=1.) return (15./2.*sign(x)-9*x1+(5./2.)*x2)*sc;
  return (+6*x1-5*x2)*sc;
}
