
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
// xb: x template for constraint to enforce. For example [ :: e ]
//     will place constraint on e'd/dx g(x)
// xlb: lower x bound
// xub: upper x bound
// d:   constraint type
// y:   constraint level
//
// xlb <= x-xb <= xub
//
// d=-1: g(x) <= y
// d=+1: g(x) >= y
// d=+2: g(x) == y
//
// so eg if xb = [ :: e ] then g(x) = e'.d/dx g(x-xb)

ML_Base &makeMonotone(ML_Base &ml,
                      int n,
                      int t,
                      const SparseVector<gentype> &xb,
                      const SparseVector<double> &xlb,
                      const SparseVector<double> &xub,
                      int d,
                      gentype y,
                      double Cweigh = 1,
                      double epsweigh = 1);

#endif
