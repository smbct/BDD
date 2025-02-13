#include "lbfgs_impl.h"
#include "bdd_solver/bdd_cuda_parallel_mma.h"

namespace LPMP {
    template class lbfgs<bdd_cuda_parallel_mma<float>, mgxthrust::device_vector<float>, float, mgxthrust::device_vector<char>, true>;
    template class lbfgs<bdd_cuda_parallel_mma<double>, mgxthrust::device_vector<double>, double, mgxthrust::device_vector<char>, true>;
}