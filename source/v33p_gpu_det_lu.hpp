#pragma once

#include <cstddef>
#include <cstdint>

struct V33pComplex {
    double real;
    double imag;
};

int v33p_gpu_available();
int v33p_gpu_batched_determinants(int n,
                                  int batch,
                                  const V33pComplex* host_matrices,
                                  V33pComplex* host_determinants,
                                  std::uint64_t* device_bytes,
                                  char* error_message,
                                  std::size_t error_capacity);
