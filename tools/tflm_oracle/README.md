# TFLite Micro oracles

Independent reference outputs for TFLite Micro integer kernels. They are inputs
to future parity tests of native ports; nothing in this crate builds or runs
them. No native code was consulted to produce them.

## `tanh_int8_oracle.cc` → `tests/data/tflm/tanh_int8_8f6596d.json`

Runs the real `tflite::Register_TANH()` (prepare in `tanh.cc`, eval in
`integer_ops/tanh.h`) through `tflite::micro::KernelRunner` over all 256 int8
input codes, for seven input calibrations, with the TANH output contract
(scale 1/128, zero point 0). It also records one non-conforming output
declaration, because TFLM's int8 TANH prepare does not check the output
quantization.

The fixture carries its own provenance (TFLM commit, sha256 of the kernel
sources and of this harness, compiler, flags, third-party commits).

### Reproduce

```sh
git clone https://github.com/AdaWorldAPI/tflite-micro && cd tflite-micro
git checkout 8f6596d   # kernel tree identical to the fixture's checkout
# Dependencies: `make ... third_party_downloads` fetches GitHub archive zips.
# Where those are blocked, clone flatbuffers v25.9.23, kissfft v130 and
# eyalroz/printf f8ed5a9b into tensorflow/lite/micro/tools/make/downloads/,
# apply the patches named in download_dependencies.py, and write its stamp.
# The person_detection example needs Pillow and numpy on PATH's python3.
make -f tensorflow/lite/micro/tools/make/Makefile test_kernel_tanh_test
g++ -std=c++17 -Os -fno-rtti -fno-exceptions -fno-threadsafe-statics \
  -DTFLM_REMOVE_VIRTUAL_DELETE -DTF_LITE_DISABLE_X86_NEON \
  -DKERNELS_OPTIMIZED_FOR_SPEED -DTF_LITE_USE_CTIME -Werror -Wall -Wextra \
  -Wno-unused-parameter -I. -Itensorflow/lite/micro/tools/make/downloads \
  -Itensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -Itensorflow/lite/micro/tools/make/downloads/kissfft \
  -Igen/linux_x86_64_default_gcc/genfiles/ \
  <ndarray>/tools/tflm_oracle/tanh_int8_oracle.cc \
  gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a \
  -Wl,--gc-sections -lm -o tanh_oracle
./tanh_oracle   # the "cases" and "nonconforming_output_probe" of the fixture
```
