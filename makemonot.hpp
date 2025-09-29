
//
// Add inducing-point constraints to enforce monotonicity
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _makemonot_h
#define _makemonot_h

#include "ml_base.hpp"

// Given an ML, generate a bunch of random points in given range and
// add specified gradient constraints at these points, approximately
// enforcing monotonicity on the underlying ML.
//
// n:  number of points to add
// t:  0 grid of points (n rounded up)
//     1 random points
//     2 based on grid (see bayesopt)
// xb: x template for constraint to enforce. For example [ :: e ]
//     will place constraint on g(x \oplus xb) = e'd/dx g(x)
// xlb: lower x bound
// xub: upper x bound
// d:   constraint type
// y:   constraint level
//
// xlb <= x <= xub
//
// d=-1: g(x) <= y
// d=+1: g(x) >= y
// d=+2: g(x) == y
//
// so eg if xb = [ :: e ] then g(x \oplus xb) = e'.d/dx g(x)
//
// Notation: x = y \oplus z means first construct x from z, then add/set
// elements defined in xlb and xub. So for example if xlb = [ 0 0 ] and
// xub = [ 1 1 ] then x = [ .5 .5 ] fits in the range, and for example:
//
// [ .5 .5 ] \oplus [ :: e ] = [ .5 .5 :: e ]

ML_Base &makeMonotone(ML_Base &ml,
                      int n,
                      int t,
                      const SparseVector<gentype> &xb,
                      const SparseVector<double> &xlb,
                      const SparseVector<double> &xub,
                      int d,
                      gentype y,
                      double Cweigh = 1,
                      double epsweigh = 1,
                      const ML_Base *gridsrc = nullptr);

#endif
