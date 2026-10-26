// Independent oracle for TFLite Micro int8 TANH.
//
// Runs the real TFLM kernel (tflite::Register_TANH(): tanh.cc prepare + the
// reference integer_ops/tanh.h eval) through micro::KernelRunner, once per
// input calibration, over all 256 int8 input codes, and prints the output
// bytes as JSON. Nothing here re-implements the kernel; it only builds
// tensors and calls TFLM. It is compiled against a TFLite Micro checkout and
// linked with that checkout's libtensorflow-microlite.a (see README.md).
//
// The prepared parameters (input_multiplier, input_left_shift,
// input_range_radius) are private to tanh.cc. They are recomputed below with
// the same TFLM functions (frexp, TfLiteRound, CalculateInputRadius) purely
// for the record; they do not feed the output bytes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/cppmath.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"

namespace {

struct Calib {
  float scale;
  int zero_point;
};

// Legal int8 input calibrations: the TANH op constrains the output
// (scale 1/128, zero point 0), not the input. Chosen to cover a small, a
// unit-like and a large scale, zero points at both ends of the code range,
// and a scale whose radius saturates most codes.
constexpr Calib kInputs[] = {
    {1.0f / 128.0f, 0},  // input already in tanh's output units
    {0.05f, -3},
    {0.1f, 20},
    {0.0137f, -128},     // zero point at the lowest code
    {0.5f, 127},         // zero point at the highest code; most codes saturate
    {0.0001f, 0},        // tiny scale: near-linear region only
    {0.0234375f, 60},
};

constexpr float kOutScale = 1.0f / 128.0f;
constexpr int kOutZeroPoint = 0;

// Runs Register_TANH over all 256 codes. Returns the TfLiteStatus of
// InitAndPrepare (Invoke runs only if prepare succeeded).
TfLiteStatus RunTanh(const Calib& in, float out_scale, int out_zp, int8_t* out,
                     TfLiteStatus* invoke_status) {
  int8_t input[256];
  for (int i = 0; i < 256; ++i) input[i] = static_cast<int8_t>(i - 128);
  int dims_data[] = {2, 1, 256};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
  TfLiteTensor tensors[2] = {
      tflite::testing::CreateQuantizedTensor(input, dims, in.scale,
                                             in.zero_point),
      tflite::testing::CreateQuantizedTensor(out, dims, out_scale, out_zp)};
  int inputs_data[] = {1, 0};
  int outputs_data[] = {1, 1};
  const TFLMRegistration registration = tflite::Register_TANH();
  tflite::micro::KernelRunner runner(
      registration, tensors, 2, tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), nullptr);
  TfLiteStatus prep = runner.InitAndPrepare();
  *invoke_status = prep == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  return prep;
}

void PrintBytes(const int8_t* out) {
  std::printf("[");
  for (int i = 0; i < 256; ++i) std::printf(i ? ",%d" : "%d", out[i]);
  std::printf("]");
}

}  // namespace

int main() {
  std::printf("{\n  \"input_codes\": \"-128..127 in order; output[i] is for input code i-128\",\n");
  std::printf("  \"output_scale\": %.9g,\n  \"output_zero_point\": %d,\n",
              static_cast<double>(kOutScale), kOutZeroPoint);
  std::printf("  \"cases\": [\n");
  const int n = sizeof(kInputs) / sizeof(kInputs[0]);
  for (int c = 0; c < n; ++c) {
    const Calib& in = kInputs[c];
    int8_t out[256] = {0};
    TfLiteStatus inv;
    TfLiteStatus prep = RunTanh(in, kOutScale, kOutZeroPoint, out, &inv);
    // The record of what tanh.cc prepare computes (same TFLM functions).
    constexpr int kInputIntegerBits = 4;
    const double real = static_cast<double>(in.scale) *
                        static_cast<double>(1 << (31 - kInputIntegerBits));
    int shift = 0;
    const double q = std::frexp(real, &shift);
    const int32_t mult = static_cast<int32_t>(tflite::TfLiteRound(q * (1ll << 31)));
    const int32_t radius =
        tflite::CalculateInputRadius(kInputIntegerBits, shift, 31);
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &in.scale, sizeof scale_bits);
    std::printf("    {\"input_scale\": %.9g, \"input_scale_bits\": \"0x%08x\", "
                "\"input_zero_point\": %d, \"prepare_ok\": %s, \"invoke_ok\": %s,\n"
                "     \"input_multiplier\": %d, \"input_left_shift\": %d, "
                "\"input_range_radius\": %d,\n     \"output\": ",
                static_cast<double>(in.scale),
                scale_bits, in.zero_point,
                prep == kTfLiteOk ? "true" : "false",
                inv == kTfLiteOk ? "true" : "false", mult, shift, radius);
    PrintBytes(out);
    std::printf("}%s\n", c + 1 < n ? "," : "");
  }
  std::printf("  ],\n");
  // Admission probe: TFLM's int8 TANH prepare does not check the output
  // quantization. Declare a non-conforming output (scale 0.1, zp 5) and
  // record whether prepare accepts it and whether the bytes change.
  int8_t ref[256] = {0}, odd[256] = {0};
  TfLiteStatus inv_ref, inv_odd;
  RunTanh(kInputs[1], kOutScale, kOutZeroPoint, ref, &inv_ref);
  TfLiteStatus prep_odd = RunTanh(kInputs[1], 0.1f, 5, odd, &inv_odd);
  bool same = true;
  for (int i = 0; i < 256; ++i) same = same && ref[i] == odd[i];
  std::printf("  \"nonconforming_output_probe\": {\"input_case\": 1, "
              "\"declared_output_scale\": 0.1, \"declared_output_zero_point\": 5, "
              "\"prepare_ok\": %s, \"invoke_ok\": %s, \"bytes_identical_to_conforming\": %s}\n}\n",
              prep_odd == kTfLiteOk ? "true" : "false",
              inv_odd == kTfLiteOk ? "true" : "false", same ? "true" : "false");
  return 0;
}
