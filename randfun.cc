
//
// Various simplified variants of random number generators.
//
// Version: split off basefn
// Date: 11/09/2024
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "randfun.hpp"
#include <math.h>
#include <cmath>
#include <random>


double &randfill(double &res, char dt, int sv, double a, double b, int t, double p)
{
    static thread_local std::default_random_engine e1; //(r());

    switch ( dt )
    {
        case 'S':
        {
            // Seed the RNG with true random

            if ( sv == -2 )
            {
                std::random_device r;
                int rsv = r();

                e1.seed(rsv);
            }

            else
            {
                e1.seed(sv);
            }

            break;
        }

        case 'u': { std::uniform_real_distribution<double> rg(a,b); res = rg(e1); break; }

        case 'r': { std::bernoulli_distribution              rg(p);   res = rg(e1) ? 1 : -1; break; }
        case 'E': { std::bernoulli_distribution              rg(p);   res = rg(e1) ? 1 :  0; break; }
        case 'b': { std::binomial_distribution<int>          rg(t,p); res = rg(e1);          break; }
        case 'B': { std::negative_binomial_distribution<int> rg(t,p); res = rg(e1);          break; }
        case 'g': { std::geometric_distribution<int>         rg(p);   res = rg(e1);          break; }

        case 'p': { std::poisson_distribution<int>          rg(b);   res = rg(e1); break; }
        case 'e': { std::exponential_distribution<double>   rg(b);   res = rg(e1); break; }
        case 'G': { std::gamma_distribution<double>         rg(p,b); res = rg(e1); break; }
        case 'w': { std::weibull_distribution<double>       rg(p,b); res = rg(e1); break; }
        case 'x': { std::extreme_value_distribution<double> rg(a,b); res = rg(e1); break; }

        case 'n': { std::normal_distribution<double>        rg(a,b); res = rg(e1); break; }
        case 'l': { std::lognormal_distribution<double>     rg(a,b); res = rg(e1); break; }
        case 'c': { std::chi_squared_distribution<double>   rg(b);   res = rg(e1); break; }
        case 'C': { std::cauchy_distribution<double>        rg(a,b); res = rg(e1); break; }
        case 'f': { std::fisher_f_distribution<double>      rg(p,b); res = rg(e1); break; }
        case 't': { std::student_t_distribution<double>     rg(b);   res = rg(e1); break; }

        default:
        {
            throw("randfill distribution type not recognised");

            break;
        }
    }

    return res;
}

