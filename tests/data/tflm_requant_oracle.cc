// Generator of tests/data/tflm_requant_golden.txt.
// Built against the real TFLite-micro sources (AdaWorldAPI/tflite-micro @ 2d8adcc,
// default build: TFLITE_SINGLE_ROUNDING undefined = double rounding):
//   g++ -std=c++17 -O1 -I<tflite-micro> tflm_requant_oracle.cc \
//     <tflite-micro>/tensorflow/lite/micro/kernels/internal/common.cc \
//     <tflite-micro>/tensorflow/lite/micro/kernels/internal/quantization_util.cc
//   ./a.out > tflm_requant_golden.txt
// Not built by cargo; kept so the fixture can be regenerated.
#include <cstdio>
#include <cstdint>
#include <limits>
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
using namespace tflite; using namespace tflite::micro;
static uint64_t s = 0x1200000000000019ull;
static uint64_t nx(){ s += 0x9E3779B97F4A7C15ull; uint64_t z=s; z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; return z^(z>>31);}
int main(){
  const int32_t E[] = {INT32_MIN, INT32_MIN+1, -(1<<30), -2,-1,0,1,2,(1<<30), INT32_MAX-1, INT32_MAX};
  // srdhm
  for (int32_t a: E) for (int32_t b: E) printf("S %d %d %d\n", a,b, SaturatingRoundingDoublingHighMul(a,b));
  for (int i=0;i<400;i++){ int32_t a=(int32_t)nx(), b=(int32_t)nx(); printf("S %d %d %d\n",a,b,SaturatingRoundingDoublingHighMul(a,b)); }
  for (int i=0;i<200;i++){ int16_t a=(int16_t)nx(), b=(int16_t)nx(); printf("H %d %d %d\n",a,b,(int)SaturatingRoundingDoublingHighMul(a,b)); }
  // rdpot incl ties
  for (int e=0;e<=31;e++){ for (int32_t x: E) printf("R %d %d %d\n",x,e,RoundingDivideByPOT(x,e));
    for (int i=0;i<8;i++){ int32_t x=(int32_t)nx(); if (e>0 && (i&1)) { x = (x & ~((int32_t)((1ll<<e)-1))) | (int32_t)(1ll<<(e-1)); } printf("R %d %d %d\n",x,e,RoundingDivideByPOT(x,e)); } }
  // mbqm double rounding: keep x<<left in range
  for (int i=0;i<600;i++){ int shift = (int)(nx()%62) - 31; int32_t m = (int32_t)(nx() & 0x7FFFFFFF); int left = shift>0?shift:0;
    int32_t x = (int32_t)nx(); x = (int32_t)((int64_t)x >> left);
    printf("M %d %d %d %d\n", x,m,shift, MultiplyByQuantizedMultiplier(x,m,shift)); }
  // int64 overload
  for (int i=0;i<300;i++){ int shift=(int)(nx()%39)-31; int32_t m=(int32_t)(nx()&0x7FFFFFFF); int64_t x=(int64_t)(nx()%(1ull<<48)) - (1ll<<47);
    int64_t red = m < 0x7FFF0000 ? ((m + (1 << 15)) >> 16) : 0x7FFF; (void)red;
    printf("L %lld %d %d %d\n",(long long)x,m,shift, MultiplyByQuantizedMultiplier(x,m,shift)); }
  // quantize_multiplier
  const double D[] = {0.0, 1.0, 0.5, 0.25, 3.7, 1e-10, 4.656612873077393e-10, 2.3283064365386963e-10, 0.999999999, 1.5, 123.456, 0.0078125, 0.75};
  for (double d: D){ int32_t q; int sh; QuantizeMultiplier(d,&q,&sh); printf("Q %.17g %d %d\n", d,q,sh); }
  for (int i=0;i<150;i++){ double d = (double)(nx()>>11) / 9007199254740992.0 * 8.0; int32_t q; int sh; QuantizeMultiplier(d,&q,&sh); printf("Q %.17g %d %d\n", d,q,sh); }
}
