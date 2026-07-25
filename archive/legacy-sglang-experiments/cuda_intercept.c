/*
 * CUDA API Interceptor - Track kernel launches without GPU profiling permissions
 *
 * Compile: gcc -shared -fPIC -o libcudaintercept.so cuda_intercept.c -ldl
 * Use: LD_PRELOAD=./libcudaintercept.so python your_script.py
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

// File for logging
static FILE *log_file = NULL;

// Original CUDA launch function
typedef void* (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, void*);
static cudaLaunchKernel_t original_cudaLaunchKernel = NULL;

// Get kernel name from function pointer
static const char* get_kernel_name(const void *func) {
    Dl_info info;
    if (dladdr(func, &info) && info.dli_sname) {
        return info.dli_sname;
    }
    return "unknown";
}

// Initialize logging
static void init_logging() {
    if (!log_file) {
        const char *filename = getenv("CUDA_INTERCEPT_LOG");
        if (!filename) {
            filename = "cuda_kernels.log";
        }
        log_file = fopen(filename, "w");
        if (log_file) {
            fprintf(log_file, "CUDA Kernel Launches Log\n");
            fprintf(log_file, "========================\n");
            fflush(log_file);
        }
    }
}

// Intercepted cudaLaunchKernel
void* cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    void *stream
) {
    // Initialize on first call
    if (!original_cudaLaunchKernel) {
        original_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
        init_logging();
    }

    // Log kernel launch
    if (log_file) {
        const char *name = get_kernel_name(func);

        // Filter for interesting kernels (GEMM, nvjet, etc.)
        if (strstr(name, "nvjet") ||
            strstr(name, "gemm") ||
            strstr(name, "cutlass") ||
            strstr(name, "wmma") ||
            strstr(name, "mma")) {

            fprintf(log_file, "Kernel: %s\n", name);
            fprintf(log_file, "  Grid:  (%u, %u, %u)\n", gridDim.x, gridDim.y, gridDim.z);
            fprintf(log_file, "  Block: (%u, %u, %u)\n", blockDim.x, blockDim.y, blockDim.z);
            fprintf(log_file, "  Shared Memory: %zu bytes\n", sharedMem);
            fprintf(log_file, "\n");
            fflush(log_file);
        }
    }

    // Call original function
    return original_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

// Cleanup
__attribute__((destructor))
static void cleanup() {
    if (log_file) {
        fprintf(log_file, "\n=== End of Log ===\n");
        fclose(log_file);
    }
}
