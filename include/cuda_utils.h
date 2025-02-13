#pragma once

#include "fix.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/inner_product.h>
#include "time_measure_util.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline float __int_as_float_host(int a)
{
    union {int a; float b;} u;
    u.a = a;
    return u.b;
}

#define CUDART_INF_F_HOST __int_as_float_host(0x7f800000)

// copied from: https://github.com/treecode/Bonsai/blob/8904dd3ebf395ccaaf0eacef38933002b49fc3ba/runtime/profiling/derived_atomic_functions.h#L186
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long int ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long int old = ret;
        if((ret = atomicCAS((unsigned long long int *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long int ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long int old = ret;
        if((ret = atomicCAS((unsigned long long int *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// copied from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                    __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

inline int get_cuda_device()
{   
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

inline void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

inline void checkCudaStatus(std::string add_info = "")
{
    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n at: %s \n", cudaGetErrorString(error), add_info.c_str());
        exit(-1);
    }
}

inline void checkCudaError(cudaError_t status, std::string errorMsg)
{
    if (status != cudaSuccess) {
        std::cout << "CUDA error: " << errorMsg << ", status" <<cudaGetErrorString(status) << std::endl;
        throw std::exception();
    }
}

template<typename T>
inline mgxthrust::device_vector<T> repeat_values(const mgxthrust::device_vector<T>& values, const mgxthrust::device_vector<int>& counts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    mgxthrust::device_vector<int> counts_sum(counts.size() + 1);
    counts_sum[0] = 0;
    mgxthrust::inclusive_scan(counts.begin(), counts.end(), counts_sum.begin() + 1);
    
    int out_size = counts_sum.back();
    mgxthrust::device_vector<int> output_indices(out_size, 0);

    mgxthrust::scatter(mgxthrust::constant_iterator<int>(1), mgxthrust::constant_iterator<int>(1) + values.size(), counts_sum.begin(), output_indices.begin());

    mgxthrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin());
    mgxthrust::transform(output_indices.begin(), output_indices.end(), mgxthrust::make_constant_iterator(1), output_indices.begin(), mgxthrust::minus<int>());

    mgxthrust::device_vector<T> out_values(out_size);
    mgxthrust::gather(output_indices.begin(), output_indices.end(), values.begin(), out_values.begin());

    return out_values;
}

template<typename T>
inline mgxthrust::device_vector<T> concatenate(const mgxthrust::device_vector<T>& a, const mgxthrust::device_vector<T>& b)
{
    mgxthrust::device_vector<T> ab(a.size() + b.size());
    mgxthrust::copy(a.begin(), a.end(), ab.begin());
    mgxthrust::copy(b.begin(), b.end(), ab.begin() + a.size());
    return ab;
}

template<typename T>
struct add_noise_func
{
    unsigned int seed;
    T noise_mag;
    T* vec;

    __host__ __device__ void operator()(const unsigned int n)
    {
        mgxthrust::default_random_engine rng;
        mgxthrust::uniform_real_distribution<T> dist(-noise_mag, noise_mag);
        rng.discard(seed + n);
        vec[n] += dist(rng);
    }
};

template<typename T>
inline void add_noise(mgxthrust::device_ptr<T> v, const size_t num, const T noise_magnitude, const unsigned int seed)
{
    add_noise_func<T> add_noise({seed, noise_magnitude, mgxthrust::raw_pointer_cast(v)});
    mgxthrust::for_each(mgxthrust::make_counting_iterator<unsigned int>(0), mgxthrust::make_counting_iterator<unsigned int>(0) + num, add_noise);
}

template<typename T>
inline void print_vector(const mgxthrust::device_vector<T>& v, const char* name, const int num = 0)
{
    std::cout<<name<<": ";
    if (num == 0)
        mgxthrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    else
    {
        int size = std::distance(v.begin(), v.end());
        mgxthrust::copy(v.begin(), v.begin() + std::min(size, num), std::ostream_iterator<T>(std::cout, " "));
    }
    std::cout<<"\n";
}

template<typename T>
inline void print_vector(const mgxthrust::device_ptr<T>& v, const char* name, const int num)
{
    std::cout<<name<<": ";
    mgxthrust::copy(v, v + num, std::ostream_iterator<T>(std::cout, " "));
    std::cout<<"\n";
}

template<typename T>
inline void check_finite(const mgxthrust::device_ptr<T>& v, const size_t num)
{
    auto result = mgxthrust::minmax_element(v, v + num);
    assert(std::isfinite(*result.first));
    assert(std::isfinite(*result.second));
}

template<typename T>
inline void print_min_max(const mgxthrust::device_ptr<T>& v, const char* name, const size_t num)
{
    auto result = mgxthrust::minmax_element(v, v + num);
    std::cout<<name<<": min = "<<*result.first<<", max = "<<*result.second<<"\n";
}

template<typename T>
inline void print_norm(const mgxthrust::device_ptr<T>& v, const char* name, const size_t num)
{
    T result = std::sqrt(mgxthrust::inner_product(v, v + num, v, (T) 0.0));
    std::cout<<name<<": norm = "<<result<<"\n";
}

struct tuple_min
{
    template<typename REAL>
    __host__ __device__
    mgxthrust::tuple<REAL, REAL> operator()(const mgxthrust::tuple<REAL, REAL>& t0, const mgxthrust::tuple<REAL, REAL>& t1)
    {
        return mgxthrust::make_tuple(min(mgxthrust::get<0>(t0), mgxthrust::get<0>(t1)), min(mgxthrust::get<1>(t0), mgxthrust::get<1>(t1)));
    }
};

struct tuple_sum
{
    template<typename T>
    __host__ __device__
    mgxthrust::tuple<T, T> operator()(const mgxthrust::tuple<T, T>& t0, const mgxthrust::tuple<T, T>& t1)
    {
        return mgxthrust::make_tuple(mgxthrust::get<0>(t0) + mgxthrust::get<0>(t1), mgxthrust::get<1>(t0) + mgxthrust::get<1>(t1));
    }
};
