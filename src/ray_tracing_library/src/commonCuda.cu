//
// Created by Sahar on 01/07/2022.
//

#include <iostream>
#include "commonCuda.cuh"

void check_cuda(cudaError_t result, const char *const func, const char *const file, const int line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
//         Make sure we call CUDA Device Reset before exiting
        std::cerr << "CUDA cudaGetErrorString: " << cudaGetErrorString(result) << "\n";
        cudaDeviceReset();
        exit(99);
    }
}
