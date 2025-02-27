#pragma once

#include "fix.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_logging.h"
#include <limits>
#ifdef WITH_CUDA
#include <thrust/device_vector.h>
#endif

namespace LPMP
{

    template <class SOLVER, typename REAL>
    class subgradient : public SOLVER
    {
    public:
        subgradient(const BDD::bdd_collection &bdd_col) : SOLVER(bdd_col) {}

        subgradient(const BDD::bdd_collection& bdd_col, const std::vector<double>& costs)
        : SOLVER(bdd_col, costs)
        {}

        void iteration();

    private:
        void subgradient_step();
        void adaptive_subgradient_step(); // as in Section 7.5 of "MRF Energy Minimization and Beyond via Dual Decomposition" by Komodakis et al.
        
        REAL best_lb = -std::numeric_limits<REAL>::infinity();
        size_t iteration_ = 0;
        REAL ema_lb = -std::numeric_limits<REAL>::infinity();
        REAL step_size = 1.0;
    };

    template <class SOLVER, typename REAL>
    void subgradient<SOLVER, REAL>::iteration()
    {
        iteration_++;
        //subgradient_step();
        adaptive_subgradient_step();
    }

    template <class SOLVER, typename REAL>
    void subgradient<SOLVER, REAL>::adaptive_subgradient_step()
    {
        if (best_lb == -std::numeric_limits<REAL>::infinity())
            best_lb = this->lower_bound();
        if (ema_lb == -std::numeric_limits<REAL>::infinity())
            ema_lb = this->lower_bound();

        auto s = this->bdds_solution_vec();
        std::vector<REAL> g(s.begin(), s.end());
        this->make_dual_feasible(g);
        this->gradient_step(g, step_size);

        best_lb = std::max(REAL(this->lower_bound()), best_lb);

        constexpr static REAL ema_weight = 0.9;
        constexpr static REAL step_size_increase_factor = 1.1;
        constexpr static REAL step_size_decrease_factor = 0.9;

        ema_lb = ema_weight * ema_lb + (1.0 - ema_weight) * this->lower_bound();
        bdd_log << "[subgradient] step size: " << step_size << ", exp. moving average lb: " << ema_lb << "\n";
        if (ema_lb < this->lower_bound())
            step_size *= step_size_increase_factor;
        if(this->lower_bound() < best_lb)
            step_size *= step_size_decrease_factor;
    }

    template <class SOLVER, typename REAL>
    void subgradient<SOLVER, REAL>::subgradient_step()
    {
        auto s = this->bdds_solution_vec();
        std::vector<REAL> g(s.begin(), s.end());
        this->make_dual_feasible(g.begin(), g.end());
        //const REAL step_size = 1.0 / REAL(1 + 0.01*iteration_);
        const REAL step_size = 0.00072;
        bdd_log << "[subgradient] step size: " << step_size << "\n";
        this->gradient_step(g.begin(), g.end(), step_size);
    }

}