//
// Created by Sahar on 01/07/2022.
//

#pragma once


#include "cuda_runtime_api.h"
#include "cuda_runtime.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line); // Don't change signature.

