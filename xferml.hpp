
//
// Transfer learning setup
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

/*

Assume we have a bunch of kernel-dependent MLs (eg SVMs) and a core SVM from
which the MLs inherit their kernel (in a simple sense, kernel 801 - currently
this does not support kernel sums and other complicated inheritance types yet).
The kernel has the form:

K2(x,y) = sum_{i,j = 0,n-1} beta_i beta_j K4(z_i,z_j,x,y)

Objective:

R(beta) = sum_ik caseweight_k ek(x_i)

alphaRange: 0 - 0   <= beta_j <= 10
            1 - -10 <= beta_j <= 10

This is bi-quadratic optimisation in most case - for each "step" in beta_j,
the MLs are re-trained (inner loop), the gradients etc are calculated as/if
needed, and the process continues.  Note that caseweight can be negative for
anti-learning.

Randomisation details:

randtype: 0 - initialise z_i ~ N(0,v)
          1 - initialise z_i ~ U(0,v)
          2 - initialise z_i ~ {-v,v} (Rademacher)
          3 - initialise z_i ~ {-1,1} (Rademacher)
randvari: if randvari > 0 then v = randvari
          if randvari < 0 then v matches the per-element variance of the
                          training vectors in the MLs
useH01: 0 - all z_i are random
        1 - z_0 = 1, rest random

Beta regularisation details:

regtype: 0 - no regularisation
         1 - regularise beta to make kernel diag ~1 in MLs
         2 - regularise with C/2 sum_i beta_i^2
         3 - regularise with C sum_i |beta_i|
C is the regularisation magnitude

Optimisation details:

Training is via multi-quadratic optimisation, treating the MLs as "inner loop"
optimisations, and the beta's as outer loop based on the following details:

method:    0 - naive gradient descent.
           1 - Nelder-Mead (nlopt).
           2 - Subplex (nlopt).
           3 - SLQP (nlopt).
maxiter:   maximum iterations
maxtime:   maximum training time (0 for unlimited)
soltol:    solution tolerance.
lr:        learning rate.
usenewton: set to use Newton method for method 0.  Note that this assumes that
           the MLs are all SVM type (for now).








./svmheavyv7.exe -qw 3 -z d -ks 2 -Zx -ki 0 -kt 3 -kg 1 -Zx -ki 1 -kt 2 -kd 3 -Zx -TT 1 -Zx -qw 1 -Zx -R q -c 1 -kt 800 -ktx 3 -oM 1000 -ANe 0 500 -1 trainincome.txt -Zx -qw 2 -Zx -R q -c 1 -kt 800 -ktx 3 -oM 1000 -ANe 0 500 -1 traingender.txt -Zx -qw 3 -Zx -xl -1 -xs 0 -xr 2 -xrv -1 -xi 30 -xo 1 -xR 2 -xC 0 -x 2 [ 1 2 ] [ 1 1 ] -Zx -qw 3 -s temp3x0.svm -Zx -qw 1 -tr -s temp1x0.svm -Zx -qw 2 -tr -s temp2x0.svm
./svmheavyv7.exe -qw 3 -z d -kt 49 -kg 0.1 -kan 5 -Zx -qw 1 -Zx -R q -c 1 -kt 801 -ktx 3 -oM 1000 -ANe 0 50 -1 trainincome.txt -Zx -qw 2 -Zx -R q -c 1 -kt 801 -ktx 3 -oM 1000 -ANe 0 50 -1 traingender.txt -Zx -qw 3 -Zx -xl -1 -xs 0 -xr 2 -xrv -1 -xi 30 -xo 1 -xR 2 -xC 0.2 -x 20 [ 1 2 ] [ 1 1 ] -Zx -qw 3 -s temp3x0.svm -Zx -qw 1 -tr -s temp1x0.svm -Zx -qw 2 -tr -s temp2x0.svm

Assumptions:

- The core SVM is either SVM_Kfront or SVM_Scalar.

*/

#ifndef _xferml_h
#define _xferml_h

#include "svm_generic.hpp"
#include "ml_base.hpp"



int xferMLtrain(svmvolatile int &killSwitch,
                SVM_Generic &core, Vector<ML_Base *> &cases,
                int n, int maxiter, double maxtime, double soltol, const Vector<double> &caseweight, int usenewton, double lr,
                int randtype, int method, double C, int regtype, double randvari, int alphaRange, int useH01);

#endif

