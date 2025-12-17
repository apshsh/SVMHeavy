
//
// Optimisation test functions as per wikipedia (see opttest.pdf)
//
// Version: 
// Date: 1/12/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _opttest_h
#define _opttest_h

#include "vector.hpp"
#include "matrix.hpp"

#define NUMOPTTESTFNS 37


// fnnum values (acronyms as per DTLZ)
//
// * 1: Rastrigin function        - range -5.12   <= x_i <= 5.12 - opt f(0,...) = 0
// * 2: Ackley's function         - range -5      <= x_i <= 5    - opt f(0,...) = 0
// # 3: Sphere function           - range -inf    <= x_i <= inf  - opt f(0,...) = 0
// @ 4: Rosenbrock function       - range -inf    <= x_i <= inf  - opt f(1,...) = 0
//   5: Beale's function          - range -4.5    <= x,y <= 4.5  - opt f(3,0.5) = 0
//   6: Goldstein–Price function  - range -2      <= x,y <= 2    - opt f(0,-1)  = 3
// ! 7: Booth's function          - range -10     <= x,y <= 10   - opt f(1,3)   = 0
// * 8: Bukin function N.6        - range -15,-3  <= x,y <= -5,3 - opt f(-10,1) = 0
// ! 9: Matyas function           - range -10     <= x,y <= 10   - opt f(0,0)   = 0
// *10: Levi function N.13        - range -10     <= x,y <= 10   - opt f(1,1)   = 0
//  11: Himmelblau's function:    - range -5      <= x,y <= 5    - opt f(3,2)   = f(-2.805,3.131) = f(-3.779,-3.283) = f(3.584,-1.848) = 0
// @12: Three-hump camel function - range -5      <= x,y <= 5    - opt f(0,0)   = 0
// ~13: Easom function            - range -100    <= x,y <= 100  - opt f(pi,pi) = -1
// *14: Cross-in-tray function    - range -10     <= x,y <= 10   - opt f(+-1.34941,+-1.34941) = -2.06261
// *15: Eggholder function        - range -512    <= x,y <= 512  - opt f(512,404.2319) = -959.6407
// *16: Holder table function     - range -10     <= x,y <= 10   - opt f(+-8.05502,+-8.05502) = -19.2085
// !17: McCormick function        - range -1.5,-3 <= x,y <= 4,4  - opt f(-0.54719,-1.54719) = -1.9133
// *18: Schaffer function N. 2    - range -100    <= x,y <= 100  - opt f(0,0)   = 0
// *19: Schaffer function N. 4    - range -100    <= x,y <= 100  - opt f(0,1.25313) = 0.292579
//  20: Styblinski–Tang function  - range -5      <= x_i <= 5    - opt -39.16617n <= f(-2.903534,...) <= -39.16616n
//  21: Stability test function 1 - range 0       <= x_i <= 1    - opt f(0.5)   ~ 1.65 (unstable 2nd order)
//                                                                     f(1)     ~ 1.5  (unstable 1st order)
//                                                                     f(0.21)  ~ 1.3  (stable)
//  22: Stability test function 2 - range 0       <= x_i <= 1    - opt f(1)     = 4 (sharper)
//                                                                     f(0.5)   = 1 (blunter)
//  23: Test function 3           - range 0       <= x_i <= 1    - opt f(x) = sum_i a_{i,0} exp(-||x-x_{i,2:...}||_2^2/(2*a_{i,1}*a_{i,1}))
// *24: Drop-wave                 - range -5.12   <= x_i <= 5.12 - opt f(0,0)   = -1
// *25: Gramancy and Lee          - range 0.5     <= x_i <= 2.5  - opt f(0.548) = -0.95
// *26: Langermann function       - range 0       <= x_i <= 10   - opt ???
// *27: Griewank function         - range -600    <= x_i <= 600  - opt f(0,...) = 0
// *28: Levy function             - range -10     <= x_i <= 10   - opt f(1,...) = 0
// *29: Schwefel function         - range -500    <= x_i <= 500  - opt f(420.9687,...) = 0
// *30: Shubert function          - range -10     <= x_i <= 10   - opt f(?)     = -186.7309
// #31: Bohachevsky Function 1    - range -100    <= x,y <= 100  - opt f(0,0) = 0
// #32: Bohachevsky Function 2    - range -100    <= x,y <= 100  - opt f(0,0) = 0
// #33: Bohachevsky Function 3    - range -100    <= x,y <= 100  - opt f(0,0) = 0
// #34: Perm function 0,D,1       - range -2      <= x,y <= 2    - opt f(1,1/2,...,1/n) = 0
//  35: Currin (Kan4) multifid    - range 0       <= x,y,z <= 1  - opt f(0.217,0,-) = -13.8
//  35: Park multifid             - range 0       <= x,y,z <= 1  - opt f(0,0,1) = 0.25
//  37: Brannin multifid          - range 0       <= x,y,z <= 1  - opt ???
//
//
// *Many local minima
// #Bowl shape
// !Plate shape
// @Valley shape
// ~Steep ridge/drop
//
// Problems are arbitrary dimensional if defined in terms of x_i,
// finite (two) dimensional if defined in terms x,y.  The dim is
// implied by the dim of the argument x.
//
// a is a set of optional parameters
//
// fnnum in range 1xxx are normalised versions of xxx with -1 <= x_i <= 1, 0 <= f(x) <= 1
// fnnum in range 2xxx are normalised versions of xxx with  0 <= x_i <= 1, 0 <= f(x) <= 1


int evalTestFn(int fnnum, double &res, const Vector<double> &x, const Matrix<double> *a = nullptr);


#endif


