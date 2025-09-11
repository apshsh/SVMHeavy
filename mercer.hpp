
//FIXME: dense derivatives/integrals need to be defined for *all* kernels if isprod set - 1xxx and 2xxx
//FIXME: for calcDenseDerivPair and calcDenseIntPair have dim arg, and extra pairs if dim = 1
//                       (also return symmetry argument to say if negate or don't negate)
//FIXME: define a related kernel for integral, one for derivative

 //     8  | Rational quadratic     | ( 1 + d/(2*r0*r0*r1) )^(-r1)                         (was 1 - d/(d+r0))
 //     9  | Multiquadratic%        | sqrt( d/(r0.r0) + r1^2 )
 //    10  | Inverse multiquadric   | 1/sqrt( d/(r0.r0) + r1^2 )
 //    11  | Circular*              | 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)
 //    12  | Sperical+              | 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3
 //    13  | Wave                   | sinc(sqrt(d)/r0)
 //    14  | Power                  | -sqrt(d/(r0.r0))^r1
 //    15  | Log#                   | -log(sqrt(d/(r0.r0))^r1 + 1)
 //    18  | Bessel^                | J_(i0+1) ( r1.sqrt(d)/r0) ) / ( (sqrt(d)/r0)^(-i0.(r1+1)) )
 //    19  | Cauchy                 | 1/(1+(d/(r0.r0)))
 //    23  | Generalised T-student  | 1/(1+(sqrt(d)/r0)^r1)
 //    25  | Weak fourier           | pi.cosh(pi-(sqrt(d)/r0))
 //    26  | Thin spline 1          | ((d/r0)^(r1+0.5))
 //    27  | Thin spline 2          | ((d/r0)^r1).ln(sqrt(d/r0))
 //    33  | Uniform                | 1/(2.r0) ( 1 if real(sqrt(d)) < r0, 0 otherwise )
 //    34  | Triangular             | (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
 //    35  | Even-integer Matern    | ((2^(1-i0))/gamma(i0)).((sqrt(2.i0).sqrt(d)/r0)^i0).K_r1(sqrt(2.i0).sqrt(d)/r0)
 //    37  | Half-integer Matern    | exp(-(sqrt(2.(i0+1/2))/r0).sqrt(d)) . (gamma(i0+1)/gamma((2.i0)+1)) . sum_{i=0,1,...,i0}( ((i0+1)!/(i!.(i0-i)!)) . pow((sqrt(8.(i0+1/2))/r0).sqrt(d),i0-i) )
 //    40  | 5/2-Matern             | (1+((sqrt(5)/r0).sqrt(d))+((5/(3.r0*r0))*d)) . exp(-(sqrt(5)/r0).sqrt(d))
 //    47  | Sinc Kernel (Tobar)    | sinc(sqrt(d)/r0).cos(2*pi*sqrt(d)/(r0.r1))
 //    49  | Gaussian Harmonic      | (1-r2)/(1-r2.exp(-d/(2.r0.r0)-r1))

//FIXME: do dense integral/derivative versions

 //    16  | Spline                 | prod_k ( 1 + (x_k/r0).(y_k/r0) + (x_k/r0).(y_k/r0).min(x_k/r0,y_k/r0) - ((x_k/r0+y_k/r0).min(x_k/r0,y_k/r0)^2)/2 + (min(x_k/r0,y_k/r0)^3)/3 )
 //    17  | B-Spline               | sum_k B_(2i0+1)(x_k/r0-y_k/r0)
 //    20  | Chi-square             | 1 - sum_k (2((x_k/r0).(y_k/r0)))/(x_k/r0+y_k/r0)
 //    21  | Histogram              | sum_k min(x_k/r0,y_k/r0)
 //    22  | Generalised histogram  | sum_k min(|x_k/r0|^r1,|y_k/r0|^r2)
 //    36  | Weiner                 | prod_i min(x_i/r0,y_i/r0)








//FIXME phantomxyzxyz - search and fixed inner normalisation (if xnorm is zero this means vector is zero, so remove from calculation altogether!)
                
/*
TO DO: have outer layer vector of gradient_order/direction pairs and weights, this defines an operator that applies to each variable in K, constructing a matrix kernel
K evaluation seems to return appropriate matrices
require farfar zero, then increment
scale up to largest order? (block diagonal?)

if farfar present:

A.(([ff'].[d])K) -> (A.[ff';ff';...]).[d].K

xaignorefarfar = 1 is the same as removing it, so no need to adjust x
xagradordadd = amount to be added to xagradOrder, use this rather than modify x
*/


//TO DO: need gradients for all KK1,KK2,KK3,KK4,KK6,KKm for at least two xagradOrder non-zero
//can do this easily enough for strictly inner-product kernels by direct-producting onto the result of K2 differentiation, so that must suffice for now

//TO DO: for non-mercer kernels, have krein-rectified mercer counterparts

//FIXME: setRandFeats needs draw NRFF variant

//FIXME: test gradient implementation

//FIXME: implement gradients and rank constraints on dK d2K etc forms
//FIXME: to do this, need to fix yyycK2 implementation to call further down the tree for d..K..del.. gradients

//FIXME: kernel chains, pass pxyprod in
//FIXME: kernel inheritance for K1,K3,Km odd
//FIXME: complex kernels for K1,K3,Km odd
//FIXME: kernel8xx for K1,K3,Km odd
//FIXME: do magterm for m != 2 + everything other than the most basic LL2,dLL2 functionality




//
// Basic kernel class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//NB - to add new kernel definitions search for ADDHERE
//KERNELSHERE - labels where kernel is actually evaluated


#ifndef _mercer_h
#define _mercer_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
//#ifdef ENABLE_THREADS
//#include <atomic>
//#include <mutex>
//#endif
#include "qswapbase.hpp"
#include "gentype.hpp"
#include "vector.hpp"
#include "sparsevector.hpp"
#include "matrix.hpp"
#include "numbase.hpp"
#include "mlcommon.hpp"



typedef const char * charptr;


//#define NUMMLINSTANCES             1024
#define STARTMLID 256

#define DEFAULTZEROPOINT -1.0

#define DEFAULT_VECT_INDEX -4
#define VECINFOSCRATCHSIZE 4*DEFAULT_NUM_TUPLES


#define DEFAULT_NUMKERNSAMP 10



// Note on kernel numbering
//
// In the 0-999 range:
//
// 0  -99  are regular kernels (which may or may not be Mercer)
// 100-199 are 0/1 neural network kernels (monotonic increasing functions of
//         x'y typically but not always with outputs ranging 0 to 1)
// 200-299 are -1/+1 neural network kernels (monotonic increasing functions of
//         x/y typically but not always with outputs ranging -1 to 1)
// 300-399 distance kernels (return -1/2 ||x-y|| for different norms).  Used
//         by KNN for example.
//
// 400-449 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic increasing, 0 < f(x) < 1, and f(0) = 1/2
// 450-499 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic decreasing, 0 < f(x) < 1, and f(0) = 1/2
//         (as per 400-449 but with x/y order reversed)
//
// 500-549 monotonic dense derivatives of kernels 400-449 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
// 550-599 monotonic dense derivatives of kernels 450-499 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
//
// 600-649 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic increasing, -1 < f(x) < 1, and f(0) = 0
// 650-699 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic decreasing, -1 < f(x) < 1, and f(0) = 0
//
// 700-749 monotonic dense derivatives of kernels 600-649 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
// 750-799 monotonic dense derivatives of kernels 650-699 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
//
// 800-899 use altcallback to evaluate kernel
//         f for corresponding monotonic density kernel
//
// 1xxx: dense derivative version (at least in the isprod case)
// 2xxx: dense integral version (at least in the isprod case)




//
// Kernel Descriptions
// ===================
//
// rj = real constant j
// ij = integer constant j
// var(0,0) (x) = a = x'x
// var(0,1) (y) = b = y'y
// var(0,2) (z) = z = x'y
// var(0,3) = d = ||x-y||_2^2 = a+b-2*z
// (var(0,3) is substituted out for var(0,0)+var(0,1)-2*var(0,2) at end)
//
//KERNELSHERE - labels where kernel is actually evaluated
//
//- r0 should be lengthscale always but isn't for these kernels
//
// Number | Name                   | K(x,y)
// -------+------------------------+------------------------------
//     0  | Constant               | r1
//     1  | Linear                 | z/(r0.r0)
//     2  | Polynomial             | ( r1 + z/(r0.r0) )^i0
//     3  | Gaussian***            | exp(-d/(2.r0.r0)-r1)
//     4  | Laplacian***           | exp(-sqrt(d)/r0-r1)
//     5  | Polynoise***           | exp(-sqrt(d)^r1/(r1*r0^r1)-r2)
//     6  | ANOVA                  | sum_k exp(-r4*((x_k/r0)^r1-(y_k/r0)^r1)^r2)^r3
//     7  | Sigmoid#               | tanh( z/(r0.r0) + r1 )
//     8  | Rational quadratic     | ( 1 + d/(2*r0*r0*r1) )^(-r1)                         (was 1 - d/(d+r0))
//     9  | Multiquadratic%        | sqrt( d/(r0.r0) + r1^2 )
//    10  | Inverse multiquadric   | 1/sqrt( d/(r0.r0) + r1^2 )
//    11  | Circular*              | 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)
//    12  | Sperical+              | 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3
//    13  | Wave                   | sinc(sqrt(d)/r0)
//    14  | Power                  | -sqrt(d/(r0.r0))^r1
//    15  | Log#                   | -log(sqrt(d/(r0.r0))^r1 + 1)
//    16  | Spline                 | prod_k ( 1 + (x_k/r0).(y_k/r0) + (x_k/r0).(y_k/r0).min(x_k/r0,y_k/r0) - ((x_k/r0+y_k/r0).min(x_k/r0,y_k/r0)^2)/2 + (min(x_k/r0,y_k/r0)^3)/3 )
//    17  | B-Spline               | sum_k B_(2i0+1)(x_k/r0-y_k/r0)
//    18  | Bessel^                | J_(i0+1) ( r1.sqrt(d)/r0) ) / ( (sqrt(d)/r0)^(-i0.(r1+1)) )
//    19  | Cauchy                 | 1/(1+(d/(r0.r0)))
//    20  | Chi-square             | 1 - sum_k (2((x_k/r0).(y_k/r0)))/(x_k/r0+y_k/r0)
//    21  | Histogram              | sum_k min(x_k/r0,y_k/r0)
//    22  | Generalised histogram  | sum_k min(|x_k/r0|^r1,|y_k/r0|^r2)
//    23  | Generalised T-student  | 1/(1+(sqrt(d)/r0)^r1)
//    24  | Vovk's real            | (1-((z/(r0.r0))^i0))/(1-(z/(r0.r0)))
//    25  | Weak fourier           | pi.cosh(pi-(sqrt(d)/r0))
//    26  | Thin spline 1          | ((d/r0)^(r1+0.5))
//    27  | Thin spline 2          | ((d/r0)^r1).ln(sqrt(d/r0))
//    28  | Generic                | (user defined)
//    29  | Arc-cosine~            | (1/pi) (r0.sqrt(a))^i0 (r0.sqrt(b))^i0 Jn(arccos(z/(sqrt(a).sqrt(b))))
//    30  | Chaotic logistic       | <phi_{sigma,n}(x/r0),phi_{sigma,n}(y/r0)>
//    31  | Summed chaotic logistic| sum_{0,n} Kn(x,y)
//    32  | Diagonal               | r1 if i == j >= 0, 0 otherwise
//    33  | Uniform                | 1/(2.r0) ( 1 if real(sqrt(d)) < r0, 0 otherwise )
//    34  | Triangular             | (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
//    35  | Even-integer Matern    | ((2^(1-i0))/gamma(i0)).((sqrt(2.i0).sqrt(d)/r0)^i0).K_r1(sqrt(2.i0).sqrt(d)/r0)
//    36  | Weiner                 | prod_i min(x_i/r0,y_i/r0)
//    37  | Half-integer Matern    | exp(-(sqrt(2.(i0+1/2))/r0).sqrt(d)) . (gamma(i0+1)/gamma((2.i0)+1)) . sum_{i=0,1,...,i0}( ((i0+1)!/(i!.(i0-i)!)) . pow((sqrt(8.(i0+1/2))/r0).sqrt(d),i0-i) )
//    38  | 1/2-Matern             | exp(-sqrt(d)/r0)
//    39  | 3/2-Matern             | (1+((sqrt(3)/r0).sqrt(d))) . exp(-(sqrt(3)/r0).sqrt(d))
//    40  | 5/2-Matern             | (1+((sqrt(5)/r0).sqrt(d))+((5/(3.r0*r0))*d)) . exp(-(sqrt(5)/r0).sqrt(d))
//    41  | RBF-rescale            | exp(log(z)/(2.r0.r0))
//    42  | Inverse Gudermannian   | igd(z/(r0.r0))
//    43  | Log ratio              | log((1+z/(r0.r0))/(1-z/(r0.r0)))
//    44  | Exponential***         | exp(z/(r0.r0)-r1)
//    45  | Hyperbolic sine        | sinh(z/(r0.r0))
//    46  | Hyperbolic cosine      | cosh(z/(r0.r0))
//    47  | Sinc Kernel (Tobar)    | sinc(sqrt(d)/r0).cos(2*pi*sqrt(d)/(r0.r1))
//    48  | LUT kernel             | r1((int) x, (int) y) if r1 is a matrix, otherwise (r1 if x != y, 1 if x == y)
//    49  | Gaussian Harmonic      | (1-r2)/(1-r2.exp(-d/(2.r0.r0)-r1))
//    50  | Alt arc-cosine         | pi - arccos(z/r0.r0)
//    51  | Vovk-like              | 1/(2-(z/r0.r0))
//    52  | Radius (see Bock)      | ((a.b)^(1/m))/(r0.r0) (just use a normalised linear kernel for the angular kernel).
//    53  | Radius (see Bock)      | (((1-(1-a^r1)^r2).(1-(1-b^r1)^r2))^(1/m))/(r0.r0) (just use a normalised linear kernel for the angular kernel).
//        |                        |
//   100  | Linear 0/1             | z/(r0*r0)
//   101  | Logistic 0/1           | 1/(1+exp(-z/(r0*r0)))
//   102  | Generalised logstic 0/1| 1/(1+r1*exp(-r2*(z-r3)/(r0*r0)))^(1/r2)
//   103  | Heavyside 0/1          | 0 if real(z) < 0, 1 otherwise
//   104  | ReLU 0/1               | 0 if real(z) < 0, z/(r0*r0) otherwise
//   105  | Softplus 0/1           | ln(r1+exp(z/(r0*r0)))
//   106  | Leaky ReLU 0/1         | r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise
//        |                        |
//   200  | Linear -1/1            | z/(r0*r0)-1
//   201  | Logistic -1/1          | 2/(1+exp(-z/(r0*r0))) - 1
//   202  | Generalised logstc -1/1| 2/(1+r1*exp(-r2*(z-r3)/(r0*r0)))^(1/r2) - 1
//   203  | Heavyside -1/1         | -1 if real(z) < 0, 1 otherwise
//   204  | Relu -1/1              | -1 if real(z) < 0, z/(r0*r0)-1 otherwise
//   205  | Softplus -1/1          | 2.ln(r1+exp(z/(r0*r0))) - 1
//        |                        |
//   300  | Euclidean distance$    | -1/2 d/(r0.r0)
//   301  | 1-norm distance$       | -1/2 ||x-y||_1^2/(r0.r0)
//   302  | inf-norm distance$     | -1/2 ||x-y||_inf^2/(r0.r0)
//   303  | 0-norm distance$       | -1/2 ||x-y||_0^2/(r0.r0)
//   304  | r0-norm distance$      | -1/2 ||x-y||_real(r1)^2/(r0.r0)
//        |                        |
//   400  | Monotnic 0/1 dense 1   | (K600(x,y)+1)/2
//   401  | Monotnic 0/1 dense 2   | (K601(x,y)+1)/2
//   402  | Monotnic 0/1 dense 3   | (K602(x,y)+1)/2
//   403  | Monotnic 0/1 dense 4   | (K603(x,y)+1)/2
//   404  | Monotnic 0/1 dense 5   | (K604(x,y)+1)/2
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| (K650(x,y)+1)/2
//   451  | Monotnic 0/1 dense 2rev| (K651(x,y)+1)/2
//   452  | Monotnic 0/1 dense 3rev| (K652(x,y)+1)/2
//   453  | Monotnic 0/1 dense 4rev| (K653(x,y)+1)/2
//   454  | Monotnic 0/1 dense 5rev| (K654(x,y)+1)/2
//        |                        |
//   500  | Monot dense deriv 1&   | K700(x,y)/2
//   501  | Monot dense deriv 2&   | K701(x,y)/2
//   502  | Monot dense deriv 3&'  | K702(x,y)/2
//   503  | Monot dense deriv 4&'  | K703(x,y)/2
//   504  | Monot dense deriv 5&'  | K704(x,y)/2
//        |                        |
//   550  | Monot dens deriv 1rev& | K750(x,y)/2
//   551  | Monot dens deriv 2rev& | K751(x,y)/2
//   552  | Monot dens deriv 3rev&`| K752(x,y)/2
//   553  | Monot dens deriv 5rev&`| K753(x,y)/2
//   554  | Monot dens deriv 5rev&`| K754(x,y)/2
//        |                        |
//   600  | Monot. -1/+1 density 1 | prod_k ( 2/(1+exp(-(x_k-y_k)/r0)) - r1 )
//   601  | Monot. -1/+1 density 2 | prod_k ( erf((x_k-y_k)/r0 - r1 )
//   602  | Monot. -1/+1 density 3 | 2/(1+exp(-min_k(x_k-y_k)/r0)) - r1
//   603  | Monot. -1/+1 density 4 | -1 if real(min_k(x_k-y_k)) < 0, 1 otherwise
//   604  | Monot. -1/+1 density 5 | max_k(x_k-y_k)/r0
//        |                        |
//   650  | Monot. -1/+1 dense 1rev| K600(y,x)
//   651  | Monot. -1/+1 dense 2rev| K601(y,x)
//   652  | Monot. -1/+1 dense 3rev| K602(y,x)
//   653  | Monot. -1/+1 dense 4rev| K603(y,x)
//   654  | Monot. -1/+1 dense 5rev| K604(y,x)
//        |                        |
//   700  | Mon -1+1 dens deriv 1& | prod_k ( (2/r0).exp(-(x_k-y_k)/r0)/(1+exp(-(x_k-y_k)/r0))^2 )
//   701  | Mon -1+1 dens deriv 2& | prod_k ((2/r0)/sqrt(pi))*exp(-((x_k-y_k)/r0)^2)
//   702  | Mon -1+1 dens deriv 3&'| (2/r0).exp(-min_k(x_k-y_k)/r0)/((1+exp(-max_k(x_k-y_k)/r0))^2)
//   703  | Mon -1+1 dens deriv 4&'| 0
//   704  | Mon -1+1 dens deriv 5&'| 1/r0
//        |                        |
//   750  | Mon dens+- deriv 1rev& | -K700(y,x)
//   751  | Mon dens+- deriv 2rev& | -K701(y,x)
//   752  | Mon dens+- deriv 3rev&`| -K702(y,x)
//   753  | Mon dens+- deriv 5rev&`| -K703(y,x)
//   754  | Mon dens+- deriv 5rev&`| -K704(y,x)
//        |                        |
//   8xx  | altcallback kernel eval| Uses altcallback to evaluate kernel.  Assumed symmetric
//        |                        |
//  1003  | Gaussian dense deriv   | exp(-r1).prod_k d/dx_k exp(-(x_k-y_k)^2/(2.r0.r0)) = exp(-r1).prod_k (-(x_k-y_k)/(r0.r0)) exp(-(x_k-y_k)^2/(2.r0.r0))
//        | (Dense deriv of RBF 3) |  = exp(-d/(2.r0.r0)-r1). prod_k ((x_k-y_k)/(r0.r0))
//  1038  | 1/2-Matern dense deriv | d/dx_0 exp(-|x_0-y_0|/r0)
//        | (isprod only)          |  = -sgn(x_0-y_0)/r0 exp(-|x_0-y_0|/r0)
//  1039  | 3/2-Matern dense deriv | d/dx_0 (1+((sqrt(3)/r0).|x_0-y_0|)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        | (isprod only)          |  =                                 (sqrt(3)/r0) . sgn(x_0-y_0) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |    -(1+((sqrt(3)/r0).|x_0-y_0|)) . (sqrt(3)/r0) . sgn(x_0-y_0) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |  = -(sqrt(3)/r0) . sgn(x_0-y_0) . exp(-(sqrt(3)/r0).|x_0-y_0|) . (1+((sqrt(3)/r0).|x_0-y_0|))
//        |                        |
//  2003  | Gaussian Dense Ingegral| exp(-r1).prod_k int_{x=-inf}^{x_k} exp(-(x-y_k)^2/(2.r0.r0)) dx
//        | (Dense integ of RBF 3) |  = exp(-r1).prod_k int_{x=-inf}^{x_k-y_k} exp(-(x/(sqrt(2).r0))^2) dx
//        |                        |  = exp(-r1).prod_k sqrt(2).r_0 int_{x=-inf}^{x_k-y_k} exp(-(x/(sqrt(2).r0))^2) d(x/sqrt(2).r_0)
//        |                        |  = exp(-r1).prod_k sqrt(2).r_0 sqrt(pi)/2 2/sqrt(pi) int_{x=-inf}^{(x_k-y_k)/(sqrt(2).r0)} exp(-x^2) dx
//        |                        |  = exp(-r1).prod_k sqrt(pi/2).r_0 ( 1 + erf((x_k-y_k)/(sqrt(2).r0)) )
//  2038  | 1/2-Matern dense integ | int_{-inf}^{x_0} exp(-|z-y_0|/r0) dz
//        | (isprod only)          |  = int_{-inf}^{x_0-y_0} exp(-|z|/r0) dz
//        |                        | if x_0<y_0: = int_{-inf}^{x_0-y_0} exp(z/r0) dz
//        |                        |             = r0 . ( exp((x_0-y_0)/r0) - exp(-inf) )
//        |                        |             = r0 . exp((x_0-y_0)/r0)
//        |                        |             = r0 . exp(-|x_0-y_0|/r0)
//        |                        | if x_0>y_0: = int_{-inf}^0 exp(z/r0) + inf_0^{x_0-y_0} exp(-z/r0) dz
//        |                        |             = r0 . ( exp(0) - exp(-inf) ) - r0 . ( exp(-(x_0-y_0)/r0) - exp(0) )
//        |                        |             = r0 . ( 2 - exp(-|x_0-y_0|/r0) )
//  2039  | 3/2-Matern dense integ | int_{-inf}^{x_0} (1+((sqrt(3)/r0).|x-y_0|)) . exp(-(sqrt(3)/r0).|z-y_0|) dz
//        | (isprod only)          |  = int_{-inf}^{x_0} exp(-(sqrt(3)/r0).|z-y_0|) dz
//        |                        |  + int_{-inf}^{x_0} ((sqrt(3)/r0).|z-y_0|) . exp(-(sqrt(3)/r0).|z-y_0|) dz
//        |                        |  = int_{-inf}^{x_0} exp(-(sqrt(3)/r0).|z-y_0|) dz
//        |                        |  + int_{-inf}^{x_0-y_0} ((sqrt(3)/r0).|z|) . exp(-(sqrt(3)/r0).|z|) dz
//        |                        |  = int_{-inf}^{x_0} exp(-(sqrt(3)/r0).|z-y_0|) dz
//        |                        |  + (r0/sqrt(3)) . int_{-inf}^{(sqrt(3)/r0).(x_0-y_0)} |z|.exp(-|z|) dz
//        |                        | if x_0<y_0: = (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             - (r0/sqrt(3)) . int_{-inf}^{-(sqrt(3)/r0).|x_0-y_0|} (-|z|).exp(-|z|) dz
//        |                        |             = (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             + (r0/sqrt(3)) . int_{-inf}^{-(sqrt(3)/r0).|x_0-y_0|} (-|z|).exp(-|z|) d(-|z|)
//        |                        |             = (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             + (r0/sqrt(3)) . int_{-inf}^{-(sqrt(3)/r0).|x_0-y_0|} z.exp(z) dz
//        |                        |             = (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             + (r0/sqrt(3)) . ( (-(sqrt(3)/r0).|x_0-y_0|-1).exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             = (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             - (r0/sqrt(3)) . (sqrt(3)/r0) . |x_0-y_0| . exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             - (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             = -|x_0-y_0| . exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        | if x_0>y_0: = (r0/sqrt(3)) . ( 2 - exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             - (r0/sqrt(3)) . int_{-inf}^{0} (-|z|).exp(-|z|) dz
//        |                        |             - (r0/sqrt(3)) . int_{0}^{(sqrt(3)/r0).|x_0-y_0|} (-|z|).exp(-|z|) dz
//        |                        |             = (r0/sqrt(3)) . ( 2 - exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             + (r0/sqrt(3)) . int_{-inf}^{0} (-|z|).exp(-|z|) d(-|z|)
//        |                        |             + (r0/sqrt(3)) . int_{0}^{(sqrt(3)/r0).|x_0-y_0|} (-z).exp(-z) d(-z)
//        |                        |             = (r0/sqrt(3)) . ( 2 - exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             + (r0/sqrt(3)) . int_{-inf}^{0} z.exp(z) dz
//        |                        |             + (r0/sqrt(3)) . int_{0}^{-(sqrt(3)/r0).|x_0-y_0|} z.exp(z) dz
//        |                        |             = (r0/sqrt(3)) . ( 2 - exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             + (r0/sqrt(3)) . ( (0-1).exp(0) - (-inf-1).exp(-inf) )
//        |                        |             + (r0/sqrt(3)) . ( ( -(sqrt(3)/r0).|x_0-y_0| - 1 ).exp(-(sqrt(3)/r0).|x_0-y_0|) - (0-1).exp(0) )
//        |                        |             = (r0/sqrt(3)) . ( 2 - exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             - (r0/sqrt(3))
//        |                        |             + (r0/sqrt(3)) . ( ( -(sqrt(3)/r0).|x_0-y_0| - 1 ).exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             + (r0/sqrt(3))
//        |                        |             = (r0/sqrt(3)) . ( 2 - exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             - (r0/sqrt(3)) . ( ( (sqrt(3)/r0).|x_0-y_0| + 1 ).exp(-(sqrt(3)/r0).|x_0-y_0|) )
//        |                        |             = (r0/sqrt(3)) . 2
//        |                        |             - (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             - (r0/sqrt(3)) . ( (sqrt(3)/r0).|x_0-y_0| ) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             - (r0/sqrt(3)) . exp(-(sqrt(3)/r0).|x_0-y_0|)
//        |                        |             = (r0/sqrt(3)) . ( 2 - ( ( 2 + ( (sqrt(3)/r0).|x_0-y_0| ) ) . exp(-(sqrt(3)/r0).|x_0-y_0|) ) )
//
// FIXME: x39 seem patently wrong!
//
// Notes: % non-mercer
//        * only positive definite in R^2
//        + only positive definite in R^3
//        # conditionally positive definite
//        ^ not yet implemented
//        ~ see Youngmin Cho, Lawrence K. Saul - Kernel Methods for Deep
//          Learning
//        $ note that K(x,x) + K(y,y) - 2.K(x,y) = ||x-y||_q^2 (q is relevant
//          norm)
//        & These are kernels 4xx with d/dx0 d/dx1 ... applied
//        ` See design decision in dense derivative
//        @ These are kernels 6xx with d/dx0 d/dx1 ... applied
//
// where: - B_i(z) = (1/i!) sum_{j=0 to i+1} (i+1)choose(j) (-1)^j max(0,(z + (i+1)/2 - j))^i
//        - Jn(x) = sin^(2n+1) (-1/sin(x) d/dx)^n (pi-x)/sin(x)
//        - phi_{sigma,n}(x) = phi_sigma(phi_sigma(...phi_sigma(x)))
//          (n repeats)
//        - phi_sigma(x) = ( sigma.x_0.(1-x_0) sigma.x_1.(1-x_1) ... )
//        - Kn is the Chaotic Logistic Kernel (case 30)
//        - K_r0 is the modified Bessel function
//        - If r1 = 3/2 the Matern kernel is
//          K(x,y) = ( 1 + sqrt(3).||x-y||/r0 ) exp(-sqrt(3).||x-y||/r0)
//          If r1 = 5/2 the Matern kernel is
//          K(x,y) = ( 1 + sqrt(5).||x-y||/r0 + 5.||x-y||^2/(3.r0^2) ) exp(-sqrt(5).||x-y||/r0)
//
// Generic kernel:
//
// Treating r10 as a function (which it is), evaluate:
//
// K(x,y) = r10(varxy)
//
// where varxy is:
//
// varxy(0,0) = m
// varxy(0,1) = x'y
// varxy(0,2) = y'x
// varxy(0,3) = (x-y)'(x-y)
// varxy(1,i) = ri (as is, not evaluated in any way, including r10)
// varxy(2,i) = Ki(x,y)  (evaluated on an is-used basis)
// varxy(3,.) = x
// varxy(4,.) = y
//
// Note that varxy(2,i) must be referenced directly or it will not
// be evaluated a-priori, leading to the wrong result.  So for example
// sum_i var(2,i) will not work.
//
// Note also that only x and y are available here
//
//
// Fourier transforms to be implemented for random sampling (non-unitary, angular frequency):
//
// 1-d:  K(x,x') = k(x-x')
//       k(y) = nu int_-inf^inf f(w) exp(-i yw) dw
//
// Number | Name                   | nu                        | f(w)
// -------+------------------------+---------------------------+--------------------------
//   ...  | ...                    | ...                       | ...
//     3  | Gaussian               | exp(-r1).2.pi             | r0/sqrt(2.pi).exp(-(1/2).(r0.r0.w.w))   (Normal distribution, a = mu = 0, b = sigma = 1/r0)
//     4  | Laplacian              | exp(-r1).2.pi             | r0/(pi.( 1 + r0.r0.w^2 ))               (Cauchy distribution, a = 0, b = 1/r0)
//   ...  | ...                    | ...                       | ...
//    13  | Wave (sinc)            | 2pi                       | r0/2pi rect(r0.w/(2*pi)) = 1 if |r0.w/(2*pi)| \leq 1/2, 0 otherwise  (Uniform distribution with a = -pi/r0, b = pi/r0)
//   ...  | ...                    | ...                       | ...
//    19  | Cauchy                 | pi                        | r0.exp(-r0.sqrt(d))                                                  (Exponential distribution, b = lambda = r0)


//
// Real constant derivatives
// =========================
//
//KERNELSHERE - labels where kernel is actually evaluated
//
// ... means calculate on the fly (if possible)
//
// Number | Name                   | K(x,y)
// -------+------------------------+------------------------------
//     0  | Constant               | ( 0 )
//        |                        | ( 1 )
//     1  | Linear                 | ( -2.z/(r0.r0.r0))
//     2  | Polynomial             | ( -2.i0.(x'y)/(r0.r0.r0) * ( r1 + x'y/(r0.r0) )^(i0-1) )
//        |                        | (    i0.                 * ( r1 + x'y/(r0.r0) )^(i0-1) )
//     3  | Gaussian               | ( (d/(r0*r0*r0)).exp(-d/(2*r0*r0)-r1) )
//     4  | Laplacian              | ( (sqrt(d)/(r0*r0)).exp(sqrt(d)/r0-r1) )
//     5  | Polynoise              | ( (sqrt(d)^r1)/(r0^(r1+1))                                                           exp(-sqrt(d)^r1/(r1*r0^r1)-r2) )
//        |                        | ( ( ((sqrt(d)^r1)/((r1^2).(r0^r1))) - (log(sqrt(d)/r0)/r1).exp(r1.log(sqrt(d)/r0)) ) exp(-sqrt(d)^r1/(r1*r0^r1)-r2) )
//     6  | ANOVA                  | ...
//     7  | Sigmoid                | ( -2.z/(r0.r0.r0) sech^2 ( z/(r0.r0) + r1 ) )
//        |                        | (                 sech^2 ( z/(r0.r0) + r1 ) )
//     8  | Rational quadratic     | ...                   (was ( d/((d+r0)^2) ))
//     9  | Multiquadratic         | ( -(2.d/(r0.r0.r0))/sqrt(d/(r0.r0)+r1^2) )
//        |                        | (                r1/sqrt(d/(r0.r0)+r1^2) )
//    10  | Inverse multiquadric   | ( (2.d/(r0.r0.r0))/(sqrt(d/(r0.r0)+r1^2))^3 )
//        |                        | (              -r1/(sqrt(d/(r0.r0)+r1^2))^3 )
//    11  | Circular               | ( -4/pi  (||x-y||^3/r0^3)/sqrt(1-(d/r0^2))  1/r0 )
//    12  | Sperical               | ( 3/2 (1-(d/r0^2)) sqrt(d)/r0^2 )
//    13  | Wave                   | ( -1/r0 ( cos(d/r0) - sinc(d/r0) ) )
//    14  | Power                  | ...
//    15  | Log                    | ...
//    16  | Spline                 | ...
//    17  | B-Spline               | ...
//    18  | Bessel                 | Not currently implemented
//    19  | Cauchy                 | ...
//    20  | Chi-square             | ...
//    21  | Histogram              | ...
//    22  | Generalised histogram  | ...
//    23  | Generalised T-student  | ...
//    24  | Vovk's real            | ...
//    25  | Weak fourier           | ...
//    26  | Thin spline 1          | ...
//    27  | Thin spline 2          | ...
//    28  | Generic                | ...
//    29  | Arc-cosine             | ...
//    30  | Chaotic logistic       | ...
//    31  | Summed chaotic logistic| ()
//    32  | Diagonal               |
//    33  | Uniform                | ...
//    34  | Triangular             | ...
//    35  | Matern                 | Not currently implemented
//    36  | Weiner                 | (-K/(r0.r0))
//    37  | Half-integer Matern    | ...
//    38  | 1/2-Matern             | ...
//    39  | 3/2-Matern             | ...
//    40  | 5/2-Matern             | ...
//    41  | RBF-rescale            | ...
//    42  | Inverse Gudermannian   | ...
//    43  | Log ratio              | ...
//    44  | Exponential            | ...
//    45  | Hyperbolic sine        | ...
//    46  | Hyperbolic cosine      | ...
//    47  | Sinc Kernel (Tobar)    | ...
//    48  | LUT kernel             | ...
//    49  | Gaussian Harmonic      | ...
//    50  | Alt arc-cosine         | Not implemented
//    51  | Vovk-like              | Not implemented
//    52  | Radius (see Bock)      | Not implemented
//    53  | Radius (see Bock)      | Not implemented
//        |                        |
//   100  | Linear 0/1             | ...
//   101  | Logistic 0/1           | ...
//   102  | Generalised logstic 0/1| ...
//   103  | Heavyside 0/1          | ( 0 )
//   104  | Rectifier 0/1          | ...
//   105  | Softplus 0/1           | ...
//   106  | Leaky rectifier 0/1    | ...
//        |                        |
//   200  | Linear -1/1            | ...
//   201  | Logistic -1/1          | ...
//   202  | Generalised logstc -1/1| ...
//   203  | Heavyside -1/1         | ( 0 )
//   204  | Rectifier -1/1         | ...
//   205  | Softplus -1/1          | ...
//        |                        |
//   300  | Euclidean distance$    | ...
//   301  | 1-norm distance$       | ...
//   302  | inf-norm distance$     | ...
//   304  | 0-norm distance$       | ...
//   305  | r0-norm distance$      | ...
//        |                        |
//   400  | Monotnic 0/1 dense 1   | ...
//   401  | Monotnic 0/1 dense 2   | ...
//   402  | Monotnic 0/1 dense 3   | ...
//   403  | Monotnic 0/1 dense 4   | ...
//   404  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| ...
//   451  | Monotnic 0/1 dense 2rev| ...
//   452  | Monotnic 0/1 dense 3rev| ...
//   453  | Monotnic 0/1 dense 4rev| ...
//   454  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   500  | Monot dense deriv 1&   | ...
//   501  | Monot dense deriv 2&   | ...
//   502  | Monot dense deriv 3&`  | ...
//   503  | Monot dense deriv 4&`  | ...
//   504  | Monot dense deriv 5&`  | ...
//        |                        |
//   550  | Monot dens deriv 1rev&`| ...
//   551  | Monot dens deriv 2rev&`| ...
//   552  | Monot dens deriv 3rev&`| ...
//   553  | Monot dens deriv 4rev&`| ...
//   554  | Monot dens deriv 5rev&`| ...
//        |                        |
//   600  | Monotnic 0/1 dense 1   | ...
//   601  | Monotnic 0/1 dense 2   | ...
//   602  | Monotnic 0/1 dense 3   | ...
//   603  | Monotnic 0/1 dense 4   | ...
//   604  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   650  | Monotnic 0/1 dense 1rev| ...
//   651  | Monotnic 0/1 dense 2rev| ...
//   652  | Monotnic 0/1 dense 3rev| ...
//   653  | Monotnic 0/1 dense 4rev| ...
//   654  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   700  | Monot dense deriv 1&   | ...
//   701  | Monot dense deriv 2&   | ...
//   702  | Monot dense deriv 3&`  | ...
//   703  | Monot dense deriv 4&`  | ...
//   704  | Monot dense deriv 5&`  | ...
//        |                        |
//   750  | Monot dens deriv 1rev&`| ...
//   751  | Monot dens deriv 2rev&`| ...
//   752  | Monot dens deriv 3rev&`| ...
//   753  | Monot dens deriv 4rev&`| ...
//   754  | Monot dens deriv 5rev&`| ...
//        |                        |
//   800  | altcallback kernel eval| ()
//        |                        |
//  1003  | Gaussian Dense Deriv   | Not implemented
//        |                        |
//  2003  | Gaussian Dense Int     | Not implemented
//


//
// Kernels derivatives
// ===================
//
//KERNELSHERE - labels where kernel is actually evaluated
//
// Number | Name                   | dK(x,y)/dz
// -------+------------------------+------------------------------
//     0  | Constant               | 0
//     1  | Linear                 | 1/(r0*r0)
//     2  | Polynomial             | i0/(r0.r0) * ( r1 + z/(r0.r0) )^(i0-1)
//     3  | Gaussian               | K(x,y)/(r0*r0)
//     4  | Laplacian              | K(x,y)/(r0*sqrt(d))
//        |                        | (arbitrarily 1 if x == y in line with RBF)
//     5  | Polynoise              | K(x,y) * ((sqrt(d)^(r1-2))/(r0^r1))
//        |                        | (arbitrarily 1 if x == y in line with RBF)
//     6  | ANOVA                  | ...
//     7  | Sigmoid                | 1/(r0.r0) * sech^2( z/(r0.r0) + r1 )
//     8  | Rational quadratic     | -(1/(2*r0*r0)).( 1 + d/(2*r0*r0*r1) )^(-r1-1)             was 2.r0/((d+r0)^2)
//     9  | Multiquadratic%        | -(1/(r0.r0))/K(x,y)
//    10  | Inverse multiquadric   | (1/(r0.r0)).K(x,y)^3
//    11  | Circular               | -4/(pi*r0^2) ( sqrt(diffis/(r0^2-diffis)) )
//    12  | Sperical               | 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3
//    13  | Wave                   | -2 (cos(sqrt(d)/r0) - sinc(sqrt(d)/r0))/sqrt(d)/r0 1/(2*r0^2*sqrt(d)/r0)
//    14  | Power                  | (r1 * (d/(r0.r0))^((r1/2)-1))/(r0.r0)
//    15  | Log#                   | (r1 * ((d/(r0.r0))^((r1/2)-1))/d^(r0/2) + 1))/(r0.r0)
//    16  | Spline                 | ...
//    17  | B-Spline               | ...
//    18  | Bessel^                | Not currently implemented
//    19  | Cauchy                 | 2/(r0.r0) * K(x,y)^2
//    20  | Chi-square             | ...
//    21  | Histogram              | ...
//    22  | Generalised histogram  | ...
//    23  | Generalised T-student  | ...
//    24  | Vovk's real            | ( ( -i0.((z/(r0.r0))^(i0-1)) + (1-((z/(r0.r0))^i0))/(1-(z/(r0.r0))) )/(1-(z/(r0.r0))) )/(r0.r0)
//        |                        | (ill-defined at z = 1)
//    25  | Weak fourier           | pi/r0 * sinh(pi-sqrt(d)/r0) / sqrt(d)
//    26  | Thin spline 1          | -2/r0 * (r1+0.5) * (d/r0)^(r1-0.5)
//    27  | Thin spline 2          | -(2.r1)/r0 * ( (d/r0)^r1 * ln(sqrt(d/r0)) + 1/2 ) / (d/r0)
//    28  | Generic                | ...
//    29  | Arc-cosine*            | ...
//    30  | Chaotic logistic       | ...
//    31  | Summed chaotic logistic| ...
//    32  | Diagonal               | ...
//    33  | Uniform                | 0.0
//    34  | Triangular             | (1-real(sqrt(d))/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
//    35  | Matern^                | Not currently implemented
//    36  | Weiner                 | ...
//    37  | Half-integer Matern    | Gradient not currently implemented
//    38  | 1/2-Matern             | Gradient not currently implemented
//    39  | 3/2-Matern             | Gradient not currently implemented
//    40  | 5/2-Matern             | Gradient not currently implemented
//    41  | RBF-rescale            | Gradient not currently implemented
//    42  | Inverse Gudermannian   | (1/(r0.r0)) sec^2(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) )
//    43  | Log ratio              | -(1/(r0.r0))^2/(1-z/(r0.r0))^2
//    44  | Exponential            | exp(z/(r0.r0)-r1)/(r0.r0)
//    45  | Hyperbolic sine        | sinh(z/(r0.r0))/(r0.r0)
//    46  | Hyperbolic cosine      | cosh(z/(r0.r0))/(r0.r0)
//    47  | Sinc Kernel (Tobar)    | ...
//    48  | LUT kernel             | ...
//    49  | Gaussian Harmonic      | ...
//    50  | Alt arc-cosine         | 1/(r0.r0.sqrt(1-(z/r0.r0)^2))
//    51  | Vovk-like              | 1/r0.r0.(2-(z/r0.r0))^2
//    52  | Radius (see Bock)      | Not implemented
//    53  | Radius (see Bock)      | Not implemented
//        |                        |
//   100  | Linear 0/1             | 1
//   101  | Logistic 0/1           | (K(x,y).(1-K(x,y)))/(r0*r0)
//   102  | Generalised logstic 0/1| (K(x,y).(1-K(x,y)^r2))/(r0*r0)
//   103  | Heavyside 0/1          | 0.0
//   104  | Rectifier 0/1          | 0 if real(z) < 0, 1/(r0*r0) otherwise
//   105  | Softplus 0/1           | (exp(r0*z)/(r1+exp(r0*z)))/(r0*r0)
//   106  | Leaky Rectifier 0/1    | r1/(r0*r0) if real(z) < 0, 1/(r0*r0) otherwise
//        |                        |
//   200  | Linear -1/1            | 1
//   201  | Logistic -1/1          | ((1-K(x,y)).(1+K(x,y))/2)/(r0*r0)
//   202  | Generalised logstc -1/1| 2.((1-K102(x,y)^r2).K102(x,y))/(r0*r0)
//   203  | Heavyside -1/1         | 0
//   204  | Rectifier -1/1         | 0 if real(z) < 0, 1/(r0*r0) otherwise
//   205  | Softplus -1/1          | 2.(exp(r0.z)/(r1+exp(r0.z)))/(r0*r0)
//        |                        |
//   300  | Euclidean distance     | ...
//   301  | 1-norm distance        | ...
//   302  | inf-norm distance      | ...
//   304  | 0-norm distance        | ...
//   305  | r0-norm distance       | ...
//        |                        |
//   400  | Monotnic 0/1 dense 1   | ...
//   401  | Monotnic 0/1 dense 2   | ...
//   402  | Monotnic 0/1 dense 3   | ...
//   403  | Monotnic 0/1 dense 4   | ...
//   404  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| ...
//   451  | Monotnic 0/1 dense 2rev| ...
//   452  | Monotnic 0/1 dense 3rev| ...
//   453  | Monotnic 0/1 dense 4rev| ...
//   454  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   500  | Monot dense deriv 1    | ...
//   501  | Monot dense deriv 2    | ...
//   502  | Monot dense deriv 3    | ...
//   503  | Monot dense deriv 4    | 0
//   504  | Monot dense deriv 5    | 0
//        |                        |
//   550  | Monot dense deriv 1    | ...
//   551  | Monot dense deriv 2    | ...
//   552  | Monot dens deriv 3rev  | ...
//   553  | Monot dens deriv 4rev  | 0
//   554  | Monot dens deriv 5rev  | 0
//        |                        |
//   600  | Monotnic 0/1 dense 1   | ...
//   601  | Monotnic 0/1 dense 2   | ...
//   602  | Monotnic 0/1 dense 3   | ...
//   603  | Monotnic 0/1 dense 4   | ...
//   604  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   650  | Monotnic 0/1 dense 1rev| ...
//   651  | Monotnic 0/1 dense 2rev| ...
//   652  | Monotnic 0/1 dense 3rev| ...
//   653  | Monotnic 0/1 dense 4rev| ...
//   654  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   700  | Monot dense deriv 1    | ...
//   701  | Monot dense deriv 2    | ...
//   702  | Monot dense deriv 3    | ...
//   703  | Monot dense deriv 4    | 0
//   704  | Monot dense deriv 5    | 0
//        |                        |
//   750  | Monot dense deriv 1    | ...
//   751  | Monot dense deriv 2    | ...
//   752  | Monot dens deriv 3rev  | ...
//   753  | Monot dens deriv 4rev  | 0
//   754  | Monot dens deriv 5rev  | 0
//        |                        |
//   800  | altcallback kernel eval| not defined
//        |                        |
//  1003  | Gaussian Dense Deriv   | Not implemented
//        |                        |
//  2003  | Gaussian Dense Integ   | Not implemented

// Notes: * The derivative of the arc-cosine kernel is not implemented
//          (although the arc-cosine kernel itself is).
//        ^ not yet implemented
//
//
// Working:
// ========
//
// OLD Case 8: Rational quadratic kernel:
// OLD
// OLD K(x,y) = 1 - ||x-y||^2/(||x-y||^2+r0)
// OLD        = 1 - ( a + b - 2z )/(( a + b - 2z )+r0)
// OLD
// OLD dK/dz = 2/((a+b-2z)+r0)  -  2.(a+b-2z)/(((a+b-2z)+r0)*((a+b-2z)+r0))
// OLD       = 2.( ((a+b-2z)+r0) - (a+b-2z) )/(((a+b-2z)+r0)*((a+b-2z)+r0))
// OLD       = 2.r0/(((a+b-2z)+r0)*((a+b-2z)+r0))
// OLD       = 2.r0/((diffis+r0)*(diffis+r0))
//
// Case 11: Circular kernel (ONLY POS DEFINITE IN R^2):
//
// K(x,y) = 2/pi * arccos(-||x-y||/r0) - 2/pi * ||x-y||/r0 * sqrt(1 - ||x-y||^2/r0^2)
//        = 2/pi * arccos(-sqrt(a+b-2z)/r0) - 2/pi * sqrt(a+b-2z)/r0 * sqrt(1 - (a+b-2z)/r0^2)
//
// q = diffis/r0^2
//
// K(diffis) = 2/pi * arccos(-sqrt(q))
//           - 2/pi * sqrt(q) * sqrt(1-q)
//
//           = 2/pi * ( arccos(-sqrt(q))
//                    - sqrt(q) * sqrt(1-q) )
//
// dK/dq = 2/pi * ( -1/2 * 1/sqrt(q) * -1/sqrt(1-q)
//                - 1/2 * 1/sqrt(q) * sqrt(1-q)
//                - sqrt(q) * -1/2 * 1/sqrt(1-q) )
//
//            = 2/pi * ( 1/2 * 1/sqrt(q) * 1/sqrt(1-q)
//                     - 1/2 * 1/sqrt(q) * sqrt(1-q)
//                     + 1/2 * sqrt(q) * 1/sqrt(1-q) )
//
//            = 1/pi * (       * 1/sqrt(q) * 1/sqrt(1-q)
//                     - (1-q) * 1/sqrt(q) * 1/sqrt(1-q)
//                     + q     * 1/sqrt(q) * 1/sqrt(1-q) )
//
//            = 2/pi * ( q * 1/sqrt(q) * 1/sqrt(1-q) )
//
//            = 2/pi * ( sqrt(q)/sqrt(1-q) )
//
// dK/dz = dK/dq * dq/ddiffis * ddiffis/dz
//       = dK/dq * 1/r0^2 * -2
//       = -2/r0^2 dK/dq
//       = -4/(pi*r0^2) ( sqrt(q)/sqrt(1-q) )
//       = -4/(pi*r0^2) ( sqrt(diffis)/sqrt(r0^2-diffis) )
//       = -4/(pi*r0^2) ( sqrt(diffis/(r0^2-diffis)) )
//
// Case 12: Spherical kernel (ONLY POS DEFINITE IN R^3):
//
// K(x,y) = 1 - 3/2 * ||x-y||/r0 + 1/2 * ||x-y||^3/r0^3
//        = 1 - 3/2 * sqrt( x'x + y'y - 2x'y )/r0 + 1/2 * ( x'x + y'y - 2x'y )^(3/2)/r0^3
//
// K(diffis) = 1 - 3/2 * sqrt(diffis)/r0 + 1/2 * sqrt(diffis)^3/r0^3
//
// dK/ddiffis = -3/2 1/2 1/sqrt(diffis) 1/r0 + 1/2 3/2 sqrt(diffis) 1/r0^3
// dK/ddiffis = -3/(4*r0) 1/sqrt(diffis) + 3/(4*r0^3) sqrt(diffis)
// dK/dz = -2 dK/ddiffis
//       = 3/(2*r0) 1/sqrt(diffis) - 3/(2*r0^3) sqrt(diffis)
//       = 3/(2*r0^3) ( r0*r0/sqrt(diffis) - sqrt(diffis) )
//
// Case 13: Wave kernel:
//
// K(x,y) = (r0/||x-y||).sin(||x-y||/r0)
//        = sinc(||x-y||/r0)
//        = (r0/sqrt( x'x + y'y - 2x'y )).sin(sqrt( x'x + y'y - 2x'y )/r0)
//
// K(diffis) = (r0/sqrt(diffis)).sin(sqrt(diffis)/r0)
//
// q = sqrt(diffis)/r0
//
// K(q) = sin(q)/q
//
// dK/dq = cos(q)/q - sin(q)/q^2
//       = (cos(q) - sin(q)/q)/q
//       = (cos(q) - sinc(q))/q
//
// dq/ddiffis = 1/(2*r0) * 1/sqrt(diffis)
//            = 1/(2*r0^2) * r0/sqrt(diffis)
//            = 1/(2*r0^2*q)
//
// ddiffis/dz = -2
//
// dK/dz = dK/dq dq/ddiffis ddiffis/dz
//       = -2 (cos(sqrt(diffis)/r0) - sinc(sqrt(diffis)/r0))/sqrt(diffis)/r0 1/(2*r0^2*sqrt(diffis)/r0)
//
// Case 19: Cauchy kernel:
//
// K(x,y) = 1/(1+((||x-y||^2)/r0))
//        = 1/(1+(( x'x + y'y - 2x'y )/r0))
//        = 1/(1 + ((a+b-2z)/r0) )
//
// dK/dz = d/dz 1/(1 + ((a+b-2z)/r0) )
//       = 2/r0 * 1/(1 + ((a+b-2z)/r0) )^2
//       = 2/r0 * K(diffis)^2
//
// Case 23: Generalised T-Student kernel:
//
// K(x,y) = 1/(1+(||x-y||/r0)^r1)
//        = 1/(1+(( x'x + y'y - 2x'y )/r0)^r1)
//        = 1/(1+ ((a+b-2z)/r0)^r1 )
//
// dK/dz = (2.r1/r0) * ((a+b-2z)/r0)^(r1-1)/(1+ ((a+b-2z)/r0)^r1 )^2
//       = (2.r1/r0) * diffis^(r1-1) * K(diffis)^2
//
// Case 24: Vovk's real polynomial:
//
// K(x,y) = (1-((x'y)^i0))/(1-(x'y))
//        = (1-(z^i0))/(1-z)
// (0 as z->1)
//
// dK/dz =  -i0.(z^(i0-1))/(1-z) + (1-(z^i0))/((1-z)^2)
// dK/dz =  ( -i0.(z^(i0-1)) + (1-(z^i0))/(1-z) )/(1-z)
// (ill-defined as z->1, so don't try)
//
// Case 25: Weak fourier kernel:
//
// K(x,y) = pi.cosh(pi-(||x-y||/r0))
//        = pi.cosh(pi-(sqrt(x'x + y'y - 2x'y)/r0))
//
// K(diffis) = pi.cosh(pi-sqrt(diffis)/r0)
//
// dK/ddiffis = pi * sinh(pi-sqrt(diffis)/r0) * -1/r0 * 1/2 * 1/sqrt(diffis)
//            = -pi/(2*r0) * sinh(pi-sqrt(diffis)/r0) / sqrt(diffis)
//
// dK/dz  = dK/ddiffis ddiffis/dz = -2 dK/ddiffis
//        = pi/r0 * sinh(pi-sqrt(diffis)/r0) / sqrt(diffis)
//
// Case 26: Thin spline (1):
//
// K(x,y) = ((||x-y||^2/r0)^(r1+0.5))
//        = (((x'x + y'y - 2x'y)/r0)^(r1+0.5))
//
// K(diffis) = (diffis/r0)^(r1+0.5)
//
// dK/ddiffis = 1/r0 * (r1+0.5) * (diffis/r0)^(r1-0.5)
//
// Case 27: Thin spline (2):
//
// K(x,y) = ((||x-y||^2/r0)^r1).ln(sqrt(||x-y||^2/r0))
//        = (((x'x + y'y - 2x'y)/r0)^r1).ln(sqrt((x'x + y'y - 2x'y)/r0))
//
// q = diffis/r0
//
// K(q) = (q^r1).ln(sqrt(q))
//
// dK/dq = r1 * q^(r1-1) * ln(sqrt(q))
//       + r1 * 1/sqrt(q) * 1/2 * 1/sqrt(q)
//       = r1 * ( q^(r1-1) * ln(sqrt(q)) + 1/2q )
//       = r1 * ( q^r1 * ln(sqrt(q)) + 1/2 ) / q
//
// dK/dz = dK/dq * dq/ddiffis * ddiffis/dz
//       = dK/dq * 1/r0 * -2
//       = -2/r0 dK/dq
//       = -(2.r1)/r0 * ( (d/r0)^r1 * ln(sqrt(d/r0)) + 1/2 ) / (d/r0)
//
// Case 29: Arccosine:
//
// K(x,y) = (1/pi) r0^2 ||x||^i0 ||y||^i0 Jn(arccos(x'y/(||x||.||y||)))
//        = (1/pi) r0^2 sqrt(a,b)^i0 Jn(arccos(z/sqrt(a.b)))
//
// dK/dz = (1/pi) r0^2 sqrt(a.b)^i0 dJn/dtheta(arccos(z/sqrt(a.b))) 1/sqrt(1-(z^2/(a.b))) 1/sqrt(a.b)
//       = (1/pi) r0^2 sqrt(a.b)^(i0-1) dJn/dtheta(arccos(z/sqrt(a.b))) 1/sqrt(1-(z^2/(a.b)))
//
// let q = z/sqrt(a.b)
//
// dK/dz = (1/pi) r0^2 sqrt(a.b)^(i0-1)/sqrt(1-q^2) dJn/dtheta(arccos(q))
//
// dK/da = (1/pi) r0^2 sqrt(b)^i0 Jn(arccos(z/sqrt(a.b)))
//
//FIXME: finish this derivation, implement it
//
// Case 34: Triangular kernel
//
// K(x,y) = (1-||x-y||/r0)/r0 if real(||x-y||) < r0, 0 otherwise )
//        = (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
//
// dK/dz = -2 dK/dd
//       = -2 -1/2r0 1/sqrt(d) 1/r0 if real(sqrt(d)) < r0, 0 otherwise
//       = 1/r0^2 1/sqrt(d) if real(sqrt(d)) < r0, 0 otherwise
//
// Case 501:
//
// K(d) = ((1.r0/sqrt(pi))^k)*exp(-r0*d)
//
// dK/dd = -r0.((1.r0/sqrt(pi))^k)*exp(-r0*d)
// dK/dz = -2 dK/dd = 2.r0.K(x,y)
//
// Case 701:
//
// K(d) = ((2.r0/sqrt(pi))^k)*exp(-r0*d)
//
// dK/dz = 2.r0.K(x,y)

// Second-order kernel derivatives
//
// q = a+b-2z
//
// Let e,f = a,b,z
//
// d2K/dede = d/de ( dK/de )
//          = dq/de d/dq ( dq/de dK/dq )
//          = dq/de ( de/dq d/de dq/de ) dK/dq + dq/de kq/de d2K/dqdq
//          = dq/de de/dq d2q/dede dK/dq + dq/de dq/de d2K/dqdq
//          = dq/de dq/de d2K/dqdq
//          = dz/de dz/de d2K/dzdz
//
// d2K/dedf = d/de ( dK/df )
//          = dq/de d/dq ( dq/df dK/dq )
//          = dq/de ( df/dq d/df dq/df ) dK/dq + dq/de dq/df d2K/dqdq
//          = dq/de df/dq d2q/dfdf dK/dq + dq/de dq/df d2K/dqdq
//          = dq/de dq/df d2K/dqdq
//          = dz/de dz/df d2K/dzdz
//
// So: need only /dqdq and /dzdz variants, rest can be calculated from that
//
//KERNELSHERE - labels where kernel is actually evaluated
//
// Number | Name                   | d2K(x,y)/dz2
// -------+------------------------+------------------------------
//     0  | Constant               | 0
//     1  | Linear                 | 0
//     2  | Polynomial             | i0.(i0-1)/(r0.r0.r0.r0) * ( r1 + z/(r0.r0) )^(i0-2)
//     3  | Gaussian               | K(x,y)/(r0*r0*r0*r0)
//     4  | Laplacian              | Not currently implemented
//     5  | Polynoise              | Not currently implemented
//     6  | ANOVA                  | ...
//     7  | Sigmoid                | Not currently implemented
//     8  | Rational quadratic     | Not currently implemented
//     9  | Multiquadratic%        | Not currently implemented
//    10  | Inverse multiquadric   | Not currently implemented
//    11  | Circular               | Not currently implemented
//    12  | Sperical               | Not currently implemented
//    13  | Wave                   | Not currently implemented
//    14  | Power                  | Not currently implemented
//    15  | Log#                   | Not currently implemented
//    16  | Spline                 | ...
//    17  | B-Spline               | ...
//    18  | Bessel^                | Not currently implemented
//    19  | Cauchy                 | Not currently implemented
//    20  | Chi-square             | ...
//    21  | Histogram              | ...
//    22  | Generalised histogram  | ...
//    23  | Generalised T-student  | Not currently implemented
//    24  | Vovk's real            | Not currently implemented
//    25  | Weak fourier           | Not currently implemented
//    26  | Thin spline 1          | Not currently implemented
//    27  | Thin spline 2          | Not currently implemented
//    28  | Generic                | ...
//    29  | Arc-cosine*            | ...
//    30  | Chaotic logistic       | ...
//    31  | Summed chaotic logistic| ...
//    32  | Diagonal               | 0
//    33  | Uniform                | 0
//    34  | Triangular             | Not currently implemented
//    35  | Matern^                | Not currently implemented
//    36  | Weiner                 | ...
//        |                        |
//   100  | Linear 0/1             | 0
//   101  | Logistic 0/1           | Not currently implemented
//   102  | Generalised logstic 0/1| Not currently implemented
//   103  | Heavyside 0/1          | 0
//   104  | Rectifier 0/1          | 0
//   105  | Softplus 0/1           | Not currently implemented
//   106  | Leaky rectifier 0/1    | 0
//        |                        |
//   200  | Linear -1/1            | 0
//   201  | Logistic -1/1          | Not currently implemented
//   202  | Generalised logstc -1/1| Not currently implemented
//   203  | Heavyside -1/1         | 0
//   204  | Rectifier -1/1         | 0
//   205  | Softplus -1/1          | Not currently implemented
//        |                        |
//   300  | Euclidean distance     | ...
//   301  | 1-norm distance        | ...
//   302  | inf-norm distance      | ...
//   304  | 0-norm distance        | ...
//   305  | r0-norm distance       | ...
//        |                        |
//   400  | Monotnic 0/1 dense 1   | ...
//   401  | Monotnic 0/1 dense 2   | ...
//   402  | Monotnic 0/1 dense 3   | ...
//   403  | Monotnic 0/1 dense 4   | ...
//   404  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| ...
//   451  | Monotnic 0/1 dense 2rev| ...
//   452  | Monotnic 0/1 dense 3rev| ...
//   453  | Monotnic 0/1 dense 4rev| ...
//   454  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   500  | Monot dense deriv 1    | ...
//   501  | Monot dense deriv 2    | ...
//   502  | Monot dense deriv 3    | ...
//   503  | Monot dense deriv 4    | 0
//   504  | Monot dense deriv 5    | 0
//        |                        |
//   550  | Monot dense deriv 1    | ...
//   551  | Monot dense deriv 2    | ...
//   552  | Monot dens deriv 3rev  | ...
//   553  | Monot dens deriv 4rev  | 0
//   554  | Monot dens deriv 5rev  | 0
//        |                        |
//   600  | Monotnic 0/1 dense 1   | ...
//   601  | Monotnic 0/1 dense 2   | ...
//   602  | Monotnic 0/1 dense 3   | ...
//   603  | Monotnic 0/1 dense 4   | ...
//   604  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   650  | Monotnic 0/1 dense 1rev| ...
//   651  | Monotnic 0/1 dense 2rev| ...
//   652  | Monotnic 0/1 dense 3rev| ...
//   653  | Monotnic 0/1 dense 4rev| ...
//   654  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   700  | Monot dense deriv 1    | ...
//   701  | Monot dense deriv 2    | ...
//   702  | Monot dense deriv 3    | ...
//   703  | Monot dense deriv 4    | 0
//   704  | Monot dense deriv 5    | 0
//        |                        |
//   750  | Monot dense deriv 1    | ...
//   751  | Monot dense deriv 2    | ...
//   752  | Monot dens deriv 3rev  | ...
//   753  | Monot dens deriv 4rev  | 0
//   754  | Monot dens deriv 5rev  | 0
//        |                        |
//   800  | altcallback kernel eval| not defined




#define BADZEROTOL 1e-12
#define BADVARTOL 1e-3

class MercerKernel;

inline std::ostream &operator<<(std::ostream &output, const MercerKernel &src );
inline std::istream &operator>>(std::istream &input,        MercerKernel &dest);

int operator==(const MercerKernel &leftop, const MercerKernel &rightop);






#define DEFAULT_XPROD_SIZE 2

class vecInfoBase;


inline void qswap(vecInfoBase &a, vecInfoBase &b);
inline vecInfoBase &setzero(vecInfoBase &a);

OVERLAYMAKEFNVECTOR(vecInfoBase)
OVERLAYMAKEFNVECTOR(Vector<vecInfoBase>)
OVERLAYMAKEFNVECTOR(SparseVector<vecInfoBase>)

class vecInfoBase
{
public:

    explicit vecInfoBase()
    {
        // Initialise values such that any normalisation has no effect

        xhalfmprod.resize(DEFAULT_XPROD_SIZE/2);

        xiseqn = 0;

//        xmean   = 0.0;
//        xmedian = 0.0;
//        xsqmean = 0.0;
//        xvari   = 1.0;
//        xstdev  = 1.0;
//        xmax    = 1.0;
//        xmin    = 0.0;
//        xrange  = 1.0;
//        xmaxabs = 1.0;

        xhalfinda = &xhalfindb;
        xhalfindb = &xhalfmprod;

        xusize = 1;

        hasbeenset = 0;
    }

    vecInfoBase(const vecInfoBase &src)
    {
        xhalfinda = &xhalfindb;
        xhalfindb = &xhalfmprod;

        *this = src;
    }

    vecInfoBase &operator=(const vecInfoBase &src)
    {
        xhalfmprod = src.xhalfmprod;

        xiseqn = src.xiseqn;

//        xmean   = src.xmean;
//        xmedian = src.xmedian;
//        xsqmean = src.xsqmean;
//        xvari   = src.xvari;
//        xstdev  = src.xstdev;
//        xmax    = src.xmax;
//        xmin    = src.xmin;
//        xrange  = src.xrange;
//        xmaxabs = src.xmaxabs;

        xusize = src.xusize;

        hasbeenset = src.hasbeenset;

        return *this;
    }

    Vector<gentype> xhalfmprod;

    int xiseqn;

//    gentype xmean;
//    gentype xmedian;
//    gentype xsqmean;
//    gentype xvari;
//    gentype xstdev;
//    gentype xmax;
//    gentype xmin;
//    gentype xrange;
//    gentype xmaxabs;

    int xusize;

    Vector<gentype> **xhalfinda;
    Vector<gentype> *xhalfindb;

    int hasbeenset;
};

COMMONOPDEF(vecInfoBase)
COMMONOPDEFPT(vecInfoBase)
COMMONOPDEFPT(const vecInfoBase)


inline vecInfoBase &setzero(vecInfoBase &a)
{
    a.xhalfmprod.resize(DEFAULT_XPROD_SIZE/2);

    a.xiseqn = 0;

//    a.xmean   = 0.0;
//    a.xmedian = 0.0;
//    a.xsqmean = 0.0;
//    a.xvari   = 1.0;
//    a.xstdev  = 1.0;
//    a.xmax    = 1.0;
//    a.xmin    = 0.0;
//    a.xrange  = 1.0;
//    a.xmaxabs = 1.0;

    a.xhalfinda = &(a.xhalfindb);
    a.xhalfindb = &(a.xhalfmprod);

    a.xusize = 1;

    a.hasbeenset = 0;

    return a;
}

inline void qswap(vecInfoBase &a, vecInfoBase &b)
{
    qswap(a.xhalfmprod,b.xhalfmprod);
    qswap(a.xiseqn    ,b.xiseqn    );

//    qswap(a.xmean  ,b.xmean  );
//    qswap(a.xmedian,b.xmedian);
//    qswap(a.xsqmean,b.xsqmean);
//    qswap(a.xvari  ,b.xvari  );
//    qswap(a.xstdev ,b.xstdev );
//    qswap(a.xmax   ,b.xmax   );
//    qswap(a.xmin   ,b.xmin   );
//    qswap(a.xrange ,b.xrange );
//    qswap(a.xmaxabs,b.xmaxabs);
//    qswap(a.xusize ,b.xusize );

    qswap(a.xusize    ,b.xusize    );
    qswap(a.hasbeenset,b.hasbeenset);
}

inline void qswap(SparseVector<vecInfoBase> *&a, SparseVector<vecInfoBase> *&b);
inline void qswap(SparseVector<vecInfoBase> *&a, SparseVector<vecInfoBase> *&b)
{
    SparseVector<vecInfoBase> *c(a); a = b; b = c;
}

class vecInfo;

inline void qswap(vecInfo &a, vecInfo &b);

OVERLAYMAKEFNVECTOR(vecInfo)
OVERLAYMAKEFNVECTOR(Vector<vecInfo>)
OVERLAYMAKEFNVECTOR(SparseVector<vecInfo>)

class vecInfo
{
public:
    explicit vecInfo()
    {
        scratch = nullptr;

        content.resize(2);

        MEMNEW(content("&",0),SparseVector<vecInfoBase>);
        MEMNEW(content("&",1),SparseVector<vecInfoBase>);

        setzero((*(content("&",0)))("&",0));
        setzero((*(content("&",1)))("&",0));

        isloc = 1;

        minind = 0;
        majind = 0;

        usize_overwrite = 0;
    }

    explicit vecInfo(const vecInfoBase &src)
    {
        scratch = nullptr;

        content.resize(2);

        MEMNEW(content("&",0),SparseVector<vecInfoBase>);
        MEMNEW(content("&",1),SparseVector<vecInfoBase>);

        setzero((*(content("&",0)))("&",0));
        setzero((*(content("&",1)))("&",0));

        isloc = 1;

        minind = 0;
        majind = 0;

        (*(content("&",0)))("&",0) = src;

        usize_overwrite = 0;
    }

    vecInfo(const vecInfo &src)
    {
        scratch = nullptr;

        content.resize(2);

        content("&",0) = nullptr;
        content("&",1) = nullptr;

        isloc = 0;

        minind = 0;
        majind = 0;

        usize_overwrite = 0;

        *this = src;
    }

    vecInfo &operator()(int _majind = -1, int _minind = -1, int xusize_overwrite = 0) const
    {
        int xmajind = ( _majind == -1 ) ? majind : _majind;
        int xminind = ( _minind == -1 ) ? minind : _minind;

        vecInfo &res = getvecscratch(xmajind,xminind);

        //(res.content).resize(2);

        if ( res.isloc )
        {
            MEMDEL((res.content)("&",0));
            MEMDEL((res.content)("&",1));

            res.content("&",0) = nullptr;
            res.content("&",1) = nullptr;

            (res.isloc) = 0;
        }

        (res.majind) = xmajind;
        (res.minind) = xminind;

        (res.content)("&",0) = content(0);
        (res.content)("&",1) = content(1);

        (res.usize_overwrite) = xusize_overwrite ? xusize_overwrite : usize_overwrite;

        return res;
    }

    vecInfo &operator=(const vecInfo &src)
    {
        content.resize(2);

        if ( isloc && src.isloc )
        {
            MEMDEL(content("&",0));
            MEMDEL(content("&",1));

            MEMNEW(content("&",0),SparseVector<vecInfoBase>);
            MEMNEW(content("&",1),SparseVector<vecInfoBase>);

            (*(content("&",0))) = (*((src.content(0))));
            (*(content("&",1))) = (*((src.content(1))));
        }

        else if ( !isloc && src.isloc )
        {
            MEMNEW(content("&",0),SparseVector<vecInfoBase>);
            MEMNEW(content("&",1),SparseVector<vecInfoBase>);

            (*(content("&",0))) = (*((src.content(0))));
            (*(content("&",1))) = (*((src.content(1))));
        }

        else if ( isloc && !(src.isloc) )
        {
            MEMDEL(content("&",0));
            MEMDEL(content("&",1));

            content("&",0) = src.content(0);
            content("&",1) = src.content(1);
        }

        else
        {
            content("&",0) = src.content(0);
            content("&",1) = src.content(1);
        }

        isloc = src.isloc;

        minind = src.minind;
        majind = src.majind;

        usize_overwrite = src.usize_overwrite;

        //if ( scratch )
        //{
        //    MEMDEL(scratch);
        //    scratch = nullptr;
        //}

        return *this;
    }

    ~vecInfo()
    {
        if ( isloc )
        {
            MEMDEL(content("&",0));
            MEMDEL(content("&",1));

            content("&",0) = nullptr;
            content("&",1) = nullptr;
        }

        if ( scratch )
        {
            MEMDEL(scratch);
            scratch = nullptr;
        }
    }

    const Vector<gentype> &xhalfmprod(void) const { return relbase().xhalfmprod; }

    int xiseqn(void) const { return relbase().xiseqn; }

//    const gentype &xmean  (void) const { return relbase().xmean;   }
//    const gentype &xmedian(void) const { return relbase().xmedian; }
//    const gentype &xsqmean(void) const { return relbase().xsqmean; }
//    const gentype &xvari  (void) const { return relbase().xvari;   }
//    const gentype &xstdev (void) const { return relbase().xstdev;  }
//    const gentype &xmax   (void) const { return relbase().xmax;    }
//    const gentype &xmin   (void) const { return relbase().xmin;    }
//    const gentype &xrange (void) const { return relbase().xrange;  }
//    const gentype &xmaxabs(void) const { return relbase().xmaxabs; }

    int xusize(void) const { return usize_overwrite ? usize_overwrite : relbase().xusize; }

    Vector<gentype> **xhalfinda(void) const { return relbase().xhalfinda; }
    Vector<gentype>  *xhalfindb(void) const { return relbase().xhalfindb; }

    int hasbeenset(void) const { return relbase().hasbeenset; }

    Vector<gentype> &xhalfmprod(void) { return relbase().xhalfmprod; }

    int &xiseqn(void) { return relbase().xiseqn; }

//    gentype &xmean  (void) { return relbase().xmean;   }
//    gentype &xmedian(void) { return relbase().xmedian; }
//    gentype &xsqmean(void) { return relbase().xsqmean; }
//    gentype &xvari  (void) { return relbase().xvari;   }
//    gentype &xstdev (void) { return relbase().xstdev;  }
//    gentype &xmax   (void) { return relbase().xmax;    }
//    gentype &xmin   (void) { return relbase().xmin;    }
//    gentype &xrange (void) { return relbase().xrange;  }
//    gentype &xmaxabs(void) { return relbase().xmaxabs; }

    int &xusize(void) { return usize_overwrite ? usize_overwrite : relbase().xusize; }

    int &hasbeenset(void) { return relbase().hasbeenset; }

    const vecInfoBase &relbase(void) const { return (*(content(    majind)))(    minind); }
          vecInfoBase &relbase(void)       { return (*(content("&",majind)))("&",minind); }

//private: - ok whatever, but don't use them

    Vector<SparseVector<vecInfoBase> *> content;

    int isloc;

    int minind;
    int majind; // 0 or 1

    int usize_overwrite;

    // Used to be a global, but not anymore as that clashed with multi-threaded operation

    mutable SparseVector<vecInfo> *scratch;

    vecInfo &getvecscratch(int xmajind, int xminind) const
    {
        if ( !scratch || !((*scratch).isindpresent((2*xmajind)+xminind)) )
        {
//#ifdef ENABLE_THREADS
//            vecinfoeyelock.lock();
//#endif

            if ( !scratch )
            {
                MEMNEW(scratch,SparseVector<vecInfo>);
            }

            if ( !((*scratch).isindpresent((2*xmajind)+xminind)) )
            {
                (*scratch)((2*xmajind)+xminind);
            }

//#ifdef ENABLE_THREADS
//            vecinfoeyelock.unlock();
//#endif
        }

        return (*scratch)("&",(2*xmajind)+xminind);
    }

//#ifdef ENABLE_THREADS
//    mutable std::mutex vecinfoeyelock;
//#endif
};

inline vecInfo &setzero(vecInfo &a) { vecInfo b; return a = b; }

COMMONOPDEF(vecInfo)
COMMONOPDEFPT(vecInfo)
COMMONOPDEFPT(const vecInfo)

inline void qswap(vecInfo &a, vecInfo &b)
{
    qswap(a.content        ,b.content        );
    qswap(a.isloc          ,b.isloc          );
    qswap(a.minind         ,b.minind         );
    qswap(a.majind         ,b.majind         );
    qswap(a.usize_overwrite,b.usize_overwrite);
    //qswap(a.scratch        ,b.scratch        );
}




// Kernel re-entry prototype: you can inherit from this and then overwrite
// K2xfer with some other function to evaluate the kernel.  Then set the
// pointers to force callback that over-rides ktype

class kernPrecursor;

COMMONOPDEFPT(kernPrecursor)

OVERLAYMAKEFNVECTOR(kernPrecursor)
OVERLAYMAKEFNVECTOR(Vector<kernPrecursor>)
OVERLAYMAKEFNVECTOR(SparseVector<kernPrecursor>)

inline std::ostream &operator<<(std::ostream &output, const kernPrecursor &src );
inline std::istream &operator>>(std::istream &input,        kernPrecursor &dest);



inline void qswap(kernPrecursor *&a, kernPrecursor *&b);

class kernPrecursor
{
public:
    explicit kernPrecursor()
    {
//#ifdef ENABLE_THREADS
//        kerneyelock.lock();
//#endif

        if ( fullmllist == nullptr )
        {
            SparseVector<kernPrecursor *> *locfullmllist = nullptr;

            MEMNEW(locfullmllist,SparseVector<kernPrecursor *>);

            NiceAssert(locfullmllist);

            fullmllist = locfullmllist;
        }

        SparseVector<kernPrecursor *> &themllist = *fullmllist;

        // Search for new unused slot in ML list
        {
            xmlid = xmlidcnt();

            while ( themllist.isindpresent(xmlid) )
            {
                xmlid = xmlidcnt();
            }
        }

        themllist("&",xmlid) = this;

//#ifdef ENABLE_THREADS
//        kerneyelock.unlock();
//#endif
    }

    virtual ~kernPrecursor()
    {
//#ifdef ENABLE_THREADS
//        kerneyelock.lock();
//#endif

        SparseVector<kernPrecursor *> &themllist = *fullmllist;

        themllist.remove(xmlid);

        if ( !themllist.indsize() )
        {
            MEMDEL(fullmllist);
            fullmllist = nullptr;
        }

//#ifdef ENABLE_THREADS
//        kerneyelock.unlock();
//#endif
    }

    kernPrecursor &operator=(const kernPrecursor &src)
    {
        assign(src);

        return *this;
    }

    //virtual void assign(const kernPrecursor &src, int onlySemiCopy = 0)
    virtual void assign(const kernPrecursor &, int = 0)
    {
    }

    //virtual void semicopy(const kernPrecursor &src)
    virtual void semicopy(const kernPrecursor &)
    {
    }

    virtual void qswapinternal(kernPrecursor &b)
    {
        int nv(xmlid); xmlid = b.xmlid; b.xmlid = nv;
    }

    virtual int isKVarianceNZ(void) const
    {
        return 0;
    }

    //
    // - resmode = 0: (default) the result is a number (or matrix or whatever)
    //   This is almost (but not quite) like the definition below, but with
    //   additional points at end.
    //
    // NB: - d2K support removed, placeholder resmode left
    //     - dkdr likewise removed, placeholder left
    //     - modes 16 and 32 also calculate result
    //
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | resmode | x,y    | integer consts | real consts | calculate | calculate | calculate | calculate | calculate  |
    // | resmode | subbed |     subbed     |    subbed   |   dk/dr   | dk/dxnorm | dk/dxyprod| d2k/dzdz  | K variance |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 0 deflt |   y    |        y       |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 1       |        |        y       |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 2       |   y    |                |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 3       |        |                |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 4       |   y    |        y       |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 5       |        |        y       |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 6       |   y    |                |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 7       |        |                |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 8       |   y    |        y       |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 9       |        |        y       |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 10      |   y    |                |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 11      |        |                |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 12      |   y    |        y       |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 13      |        |        y       |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 14      |   y    |                |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 15      |        |                |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 16      |   y    |        y       |      y      |           |     y     |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 32      |   y    |        y       |      y      |           |           |     y     |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 64      |   y    |        y       |      y      |           |           |           |     y     |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 128     |   y    |        y       |      y      |           |           |           |           |     y      |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+

//NB: templates cannot be made virtual, so need both versions

    //virtual void K0xfer(gentype &res, int &minmaxind, int typeis,
    //                    const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
    //                    int xdim, int densetype, int resmode, int mlid) const
    virtual void K0xfer(gentype &res, int &, int,
                        const gentype &, const gentype &, const gentype &,
                        int, int, int, int) const
    {
        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //NiceThrow("K0xfer not defined here for m = 4");

        res = 0.0;
    }

    virtual void K0xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K0xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    //virtual void K1xfer(gentype &res, int &minmaxind, int typeis,
    //                    const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
    //                    const SparseVector<gentype> &xa, 
    //                    const vecInfo &xainfo, 
    //                    int ia, 
    //                    int xdim, int densetype, int resmode, int mlid) const
    virtual void K1xfer(gentype &res, int &, int,
                        const gentype &, const gentype &, const gentype &,
                        const SparseVector<gentype> &, 
                        const vecInfo &, 
                        int, 
                        int, int, int, int) const
    {
        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //NiceThrow("K1xfer not defined here for m = 4");

        res = 0.0;
    }

    virtual void K1xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K1xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    //virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
    //                    const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
    //                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
    //                    const vecInfo &xainfo, const vecInfo &xbinfo,
    //                    int ia, int ib,
    //                    int xdim, int densetype, int resmode, int mlid) const
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &, int,
                        const gentype &, const gentype &, const gentype &,
                        const SparseVector<gentype> &, const SparseVector<gentype> &,
                        const vecInfo &, const vecInfo &,
                        int, int,
                        int, int, int, int) const
    {
        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //NiceThrow("K2xfer not defined here for m = 2");

        res = 0.0;

        dxyprod = 0.0;
        ddiffis = 0.0;
    }

    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;
        gentype tempdxyprod;
        gentype tempddiffis;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K2xfer(tempdxyprod,tempddiffis,tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

        res = (double) tempres;

        dxyprod = tempdxyprod;
        ddiffis = tempddiffis;
    }

    //virtual void K3xfer(gentype &res, int &minmaxind, int typeis,
    //                    const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
    //                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
    //                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
    //                    int ia, int ib, int ic, 
    //                    int xdim, int densetype, int resmode, int mlid) const
    virtual void K3xfer(gentype &res, int &, int,
                        const gentype &, const gentype &, const gentype &,
                        const SparseVector<gentype> &, const SparseVector<gentype> &, const SparseVector<gentype> &, 
                        const vecInfo &, const vecInfo &, const vecInfo &, 
                        int, int, int, 
                        int, int, int, int) const
    {
        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //NiceThrow("K3xfer not defined here for m = 4");

        res = 0.0;
    }

    virtual void K3xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K3xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    //virtual void K4xfer(gentype &res, int &minmaxind, int typeis,
    //                    const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
    //                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
    //                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
    //                    int ia, int ib, int ic, int id,
    //                    int xdim, int densetype, int resmode, int mlid) const
    virtual void K4xfer(gentype &res, int &, int,
                        const gentype &, const gentype &, const gentype &,
                        const SparseVector<gentype> &, const SparseVector<gentype> &, const SparseVector<gentype> &, const SparseVector<gentype> &,
                        const vecInfo &, const vecInfo &, const vecInfo &, const vecInfo &,
                        int, int, int, int,
                        int, int, int, int) const
    {
        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //NiceThrow("K4xfer not defined here for m = 4");

        res = 0.0;
    }

    virtual void K4xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K4xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    //virtual void Kmxfer(gentype &res, int &minmaxind, int typeis,
    //                    const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
    //                    Vector<const SparseVector<gentype> *> &x,
    //                    Vector<const vecInfo *> &xinfo,
    //                    Vector<int> &i,
    //                    int xdim, int m, int densetype, int resmode, int mlid) const
    virtual void Kmxfer(gentype &res, int &, int,
                        const gentype &, const gentype &, const gentype &,
                        Vector<const SparseVector<gentype> *> &,
                        Vector<const vecInfo *> &,
                        Vector<int> &,
                        int, int, int, int, int) const
    {
        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
/*
        if ( m == 0 )
        {
            K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);
        }

        else if ( m == 1 )
        {
            K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*xinfo(0),i(0),xdim,densetype,resmode,mlid);
        }

        else if ( m == 2 )
        {
            gentype dummy;

            K2xfer(dummy,dummy,res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*xinfo(0),*xinfo(1),i(0),i(1),xdim,densetype,resmode,mlid);
        }

        else if ( m == 3 )
        {
            K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*x(2),*xinfo(0),*xinfo(1),*xinfo(2),i(0),i(1),i(2),xdim,densetype,resmode,mlid);
        }

        else if ( m == 4 )
        {
            K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*x(2),*x(3),*xinfo(0),*xinfo(1),*xinfo(2),*xinfo(3),i(0),i(1),i(2),i(3),xdim,densetype,resmode,mlid);
        }

        else
*/
        {
            res = 0.0;

            // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
            //NiceThrow("Kmxfer not defined here for m > 4");
        }
    }

    virtual void Kmxfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &i,
                        int xdim, int m, int densetype, int resmode, int mlid) const
    {
        gentype tempres(res);

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        Kmxfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

        res = (double) tempres;
    }





    // Kernel transfer switching stuff.  All MLs are registered so you
    // can switch between them (kernel transfer).
    //
    // MLid(): unique ID for this this ML.
    // setMLid(): MLid is default = > STARTMLID - use this to set to more sensible
    //            value.  Return 0 on success, nz otherwise.
    // getaltML(): get reference to ML with given ID.  Return 0 on success, 1 if nullptr.

    virtual int MLid(void) const
    {
        return xmlid;
    }

    virtual int setMLid(int nv)
    {
        int res = 0;

//#ifdef ENABLE_THREADS
//        kerneyelock.lock();
//#endif

        SparseVector<kernPrecursor *> &themllist = *fullmllist;

        //if ( ( nv < 0 ) || ( nv > NUMMLINSTANCES ) )
        if ( nv < 0 )
        {
            res = 1;
        }

        else if ( themllist.isindpresent(nv) )
        {
            res = 2;
        }

        else
        {
            xmlid = nv;

            themllist("&",xmlid) = this;
        }

//#ifdef ENABLE_THREADS
//        kerneyelock.unlock();
//#endif

        return res;
    }

    virtual int getaltML(kernPrecursor *&res, int altMLid) const
    {
//#ifdef ENABLE_THREADS
//        kerneyelock.lock();
//#endif

        SparseVector<kernPrecursor *> &themllist = *fullmllist;

        int ires = 1;
        res = nullptr;

        if ( themllist.isindpresent(altMLid) )
        {
            ires = 0;
            res = themllist("&",altMLid);
        }

//#ifdef ENABLE_THREADS
//        kerneyelock.unlock();
//#endif

        return ires;
    }

    int mllistsize(void)
    {
//#ifdef ENABLE_THREADS
//        svm_mutex_lock(kerneyelock);
//#endif
        int res = (*fullmllist).indsize();
//#ifdef ENABLE_THREADS
//        svm_mutex_unlock(kerneyelock);
//#endif
        return res;
    }

    int mllistind(int i)
    {
//#ifdef ENABLE_THREADS
//        kerneyelock.lock();
//#endif
        int res = (*fullmllist).ind(i);
//#ifdef ENABLE_THREADS
//        kerneyelock.unlock();
//#endif
        return res;
    }

    int mllistisindpresent(int i)
    {
//#ifdef ENABLE_THREADS
//        kerneyelock.lock();
//#endif
        int res = (*fullmllist).isindpresent(i);
//#ifdef ENABLE_THREADS
//        kerneyelock.unlock();
//#endif
        return res;
    }

    int xmlid;

    virtual std::ostream &printstream(std::ostream &output, int dep) const { (void) dep; return output; }
    virtual std::istream &inputstream(std::istream &input ) { return input; }

    virtual int type(void) const { return -2; }
    virtual int subtype(void) const { return 0; }

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const { (void) ind; (void) val; (void) xa; (void) ia; (void) xb; (void) ib; desc = ""; return 0; }
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const { (void) ind; (void) val; (void) xa; (void) ia; (void) xb; (void) ib;            return 0; }

    //const svmvolatile SparseVector<kernPrecursor *> &getmllist(void) const { return *fullmllist; }



    // x data

    virtual const SparseVector<gentype> &x(int i)              const { (void) i; NiceThrow("Not there yet"); const static SparseVector<gentype> dummy; return dummy;  }
    virtual const SparseVector<gentype> &x(int i, int altMLid) const { kernPrecursor *tmp = nullptr; getaltML(tmp,altMLid);  NiceAssert(tmp); return (*tmp).x(i); }

private:

    int xmlidcnt(void)
    {
//#ifdef ENABLE_THREADS
//        static std::atomic<int> loccnt(STARTMLID); //NUMMLINSTANCES/2);
//#endif
//#ifndef ENABLE_THREADS
        static thread_local int loccnt(STARTMLID); //NUMMLINSTANCES/2);
//#endif

        return (int) ++loccnt;
    }

    static thread_local SparseVector<kernPrecursor *>* fullmllist;
//#ifdef ENABLE_THREADS
//    static std::mutex kerneyelock;
//#endif
};

inline std::ostream &operator<<(std::ostream &output, const kernPrecursor &src)
{
    return src.printstream(output,0);
}

inline std::istream &operator>>(std::istream &input, kernPrecursor &dest)
{
    return dest.inputstream(input);
}








//
// Kernel information structure: this stores information about what info
// the kernel function uses to evaluate.  Adding these gives the flags if
// the 

class kernInfo;

OVERLAYMAKEFNVECTOR(kernInfo)
OVERLAYMAKEFNVECTOR(Vector<kernInfo>)
OVERLAYMAKEFNVECTOR(SparseVector<kernInfo>)

inline void qswap(kernInfo *&a, kernInfo *&b);

class kernInfo
{
public:

    kernInfo()
    {
        usesDiff    = 0;
        usesInner   = 0;
        usesNorm    = 0;
        usesVector  = 0;
        usesMinDiff = 0;
        usesMaxDiff = 0;
    }

    kernInfo(const kernInfo &src)
    {
        usesDiff    = src.usesDiff;
        usesInner   = src.usesInner;
        usesNorm    = src.usesNorm;
        usesVector  = src.usesVector;
        usesMinDiff = src.usesMinDiff;
        usesMaxDiff = src.usesMaxDiff;
    }

    kernInfo &operator=(const kernInfo &src)
    {
        usesDiff    = src.usesDiff;
        usesInner   = src.usesInner;
        usesNorm    = src.usesNorm;
        usesVector  = src.usesVector;
        usesMinDiff = src.usesMinDiff;
        usesMaxDiff = src.usesMaxDiff;

        return *this;
    }

    int numflagsset(void) const
    {
        return usesDiff+usesInner+usesNorm+usesVector+usesMinDiff+usesMaxDiff;
    }

    kernInfo &zero(void)
    {
        usesDiff    = 0;
        usesInner   = 0;
        usesNorm    = 0;
        usesVector  = 0;
        usesMinDiff = 0;
        usesMaxDiff = 0;

        return *this;
    }

    unsigned int usesDiff    : 1; // set if kernel uses ||x-y||^2 explicitly
    unsigned int usesInner   : 1; // set if kernel uses x'y explicitly
    unsigned int usesNorm    : 1; // set if kernel uses ||x||^2 and ||y||^2 explicitly
    unsigned int usesVector  : 1; // set if kernel uses x and y vectors explicitly
    unsigned int usesMinDiff : 1; // set if kernel uses min(x-y) explicitly
    unsigned int usesMaxDiff : 1; // set if kernel uses max(x-y) explicitly
};

//
// +=: this is defined so that summing a vector of kernInfo works as OR
// ==: equivalence operator
// <<: output stream
// >>: input strea,
// setzero: sets all flags 0
// qswap:   standard quickswap operation
//

kernInfo &operator+=(kernInfo &a, const kernInfo &b);
int operator==(const kernInfo &a, const kernInfo &b);

std::ostream &operator<<(std::ostream &output, const kernInfo &src);
std::istream &operator>>(std::istream &input, kernInfo &dest);

inline kernInfo &setzero(kernInfo &a);
inline void qswap(kernInfo &a, kernInfo &b);

COMMONOPDEF(kernInfo)
COMMONOPDEFPT(kernInfo)
COMMONOPDEFPT(const kernInfo)

inline kernInfo &setzero(kernInfo &a)
{
    return a.zero();
}

inline void qswap(kernInfo &a, kernInfo &b)
{
    kernInfo c(a); a = b; b = c;
}






std::ostream &operator<<(std::ostream &output, const vecInfoBase &src);
std::istream &operator>>(std::istream &input, vecInfoBase &dest);

std::ostream &operator<<(std::ostream &output, const vecInfo &src);
std::istream &operator>>(std::istream &input, vecInfo &dest);

// Swap function

inline void qswap(MercerKernel &a, MercerKernel &b);

OVERLAYMAKEFNVECTOR(MercerKernel)
OVERLAYMAKEFNVECTOR(Vector<MercerKernel>)
OVERLAYMAKEFNVECTOR(SparseVector<MercerKernel>)

class MercerKernel : public kernPrecursor
{
    friend int operator==(const MercerKernel &leftop, const MercerKernel &rightop);
    friend void qswap(MercerKernel &a, MercerKernel &b);

public:

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input)                 override;

    virtual int type   (void) const override { return -3; }
    virtual int subtype(void) const override { return 0; }

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    // Constructors and assignment operators

    explicit MercerKernel();
             MercerKernel(const MercerKernel &src);

    ~MercerKernel();

    MercerKernel &operator=(const MercerKernel &src);

    // Overall Information:
    //
    // isAltDiff: 0:   ||x-y||_2^2    -> ||x||_m^m + ||x'||_m^m + ... - m.<<x,x',...>>_m
    //            1:   ||x-y||_2^2    -> ||x||_2^2 + ||x'||_2^2 + ... - 2.<<x,x',...>>_m (default)
    //            2:   2*(||x-y||_2^2 -> ||x||_2^2 + ||x'||_2^2 + ... - (1/m).(sum_{ij} <xi,xj>))
    //                 (the RBF has additional scaling as per paper - see Kbase)
    //            5:   ||x-y||_2^2    -> ||x-x'||_2^2 + ||x''-x'''||_2^2 + ...
    //                                 = ||x||_2^2 + ||x'||_x^2 + ... - 2<x,x'> - 2<x'',x'''> - ...
    //                                 = ||x||_2^2 + ||x'||_x^2 + ... - 2 sum_{i=0,2,...} <x_i,x_{i+1}>
    //            103: K(...) -> 1/2^{m-1} \sum_{s = [ +-1 +-1 ... ] \in R^m : |i:si=+1| + |i:si=-1| \in 4Z_+} K(||sum_i s_i x_i ||_2^2)
    //            104: K(...) -> 1/m!      \sum_{s = [ +-1 +-1 ... ] \in R^m : |i:si=+1| = |i:si=-1|         } K(||sum_i s_i x_i ||_2^2)
    //            203: like 103, but kernel expansion occurs over first kernel in chain only
    //            204: like 104, but kernel expansion occurs over first kernel in chain only
    //            300: true moment-kernel expension to 2-kernels

    bool unadornedRBFKernel(int allowsymm = 0) const { return isVeryTrivialKernel(allowsymm) && !sizeLinConstr() && !sizeLinParity() && ( cType() == 3 ) && ( ((double) cWeight()) == 1.0 ) && ( ((double) cRealConstants()(1)) == 0.0 ); }

    const Vector<int> &defindKey(void) const { return xdefindKey; } // on the training set at least
    int basexdim(void) const { return defindKey().size(); }

    int isSymmSet        (void) const { return issymmset;                 } // symmetrise setwise kernel evaluation
    int isFullNorm       (void) const { return isfullnorm;                } // normalise at outermost
    int isProd           (void) const { return isprod;                    } // K(x,y) = prod_i K(x_i,y_i)
    int isIndex          (void) const { return isind;                     } // x'z is indexed (x'z -> sum_{i in S} x_i.y_i)
    int isShifted        (void) const { return isshift & 1;               } // vectors shifted (x -> (x-sh))
    int isScaled         (void) const { return isshift & 2;               } // vectors scaled (x -> diag(sc).x)
    int isShiftedScaled  (void) const { return isshift == 3;              } // vectors shifted and scaled (x -> diag(sc).(x-sh))
    int isLeftPlain      (void) const { return leftplain;                 } // no normalisation applied to x in K(x,y)
    int isRightPlain     (void) const { return rightplain;                } // no normalisation applied to y in K(x,y)
    int isLeftRightPlain (void) const { return leftplain && rightplain;   } // no normalisation applied to x and y in K(x,y)
    int isLeftNormal     (void) const { return !leftplain;                } // normalisation may be applied to x in K(x,y)
    int isRightNormal    (void) const { return !rightplain;               } // normalisation may be applied to y in K(x,y)
    int isLeftRightNormal(void) const { return !leftplain && !rightplain; } // normalisation may be applied to x and y in K(x,y)
    int isPartNormal     (void) const { return !leftplain || !rightplain; } // normalisation may be applied to x or y in K(x,y)
    int isAltDiff        (void) const { return isdiffalt;                 } // see above notes
    int needsmProd       (void) const { return needsInner(-1,4);          } // m-kernel calculation requires <<x,y,...>>_m
    int wantsXYprod      (void) const { return needsMatDiff();            } // providing an xy matrix will result in speedup
    int suggestXYcache   (void) const { return xsuggestXYcache;           } // suggest to user that passing xy matrix will help even in 2-kernel cache (purely advisory, can be ignored)
    int isIPdiffered     (void) const { return xisIPdiffered;             } // inner-products have changes (0 means *probably* unchanged)
    int rankType         (void) const { return xranktype;                 } // how ranks are translated. 0: phi(x,x') = phi(x)-phi(x'), 1: phi(x,x') = phi(x)+phi(x'), 2: phi(x,x') = phi(x) \otimes phi(x') - phi(x') \otimes phi(x), 3: phi(x,x') = phi(x) \otimes phi(x') - phi(x') \otimes phi(x)

    double denseZeroPoint(void) const { return xdenseZeroPoint; } // zero point (beta) for dense integration (beta = r0.xdenseZeroPoint)

    int size       (void) const { return isnorm.size(); } // number of kernel "blocks" in total kernel
    int getSymmetry(void) const;                          // 1 for symmetric, -1 for anti, 0 for none

    const       Vector<int>     &cIndexes(void) const { return dIndexes; } // index vector S if used
    const SparseVector<gentype> &cShift  (void) const { return dShift;   } // shift used for normalisation
    const SparseVector<gentype> &cScale  (void) const { return dScale;   } // scale used for normalisation

    // Linear constraint information
    //
    // As per Jidling, you can enforce linear constraints on the RKHS by modifying
    // the kernel with the null-space operator of the form:
    //
    // O_0 O_1 ... O_{m-1} Km(x_0,x_1,...,x_{m-1})
    //
    // where O_i is an operator of the form:
    //
    // O_i = M_i (d/dx)^{\otimes p_i} (kronecker power)
    //
    // where p_i is an integer (order) and M_i is a matrix.

    int sizeLinConstr(void) const { return linGradOrd.size(); } // number of terms in linear constraint (see Jidling), 0 of no constraints

    const Vector<int>             &getlinGradOrd (void) const { return linGradOrd;  } // order of linear gradient constraints (see Jidling)
    const Vector<Matrix<double> > &getlinGradScal(void) const { return linGradScal; } // get matrix part of linear gradient constraints (see Jidling)

    // Linear parity constraint
    //
    // Replace x = [ x0 x1 x2 ... ] by multiplying xi0 xi1 xi2 ... by sgn(xi0.xi1.xi2...).

    int sizeLinParity(void) const { return linParity.size(); }

    const Vector<int>     &getlinParity    (void) const { return linParity;     }
    const Vector<gentype> &getlinParityOrig(void) const { return linParityOrig; }

    // Kernel "block" information
    //
    // Constant overwrites let you take the value for real (integer)
    // constants from the input vectors x and y.  For example, if
    // cRealOverwrite(q) = ( 0:2 1:10 ) then:
    //
    // realConstant(0) -> x(2)*y(2)    (x(2) is rightPlain, y(2) if leftPlain)
    // realConstant(1) -> x(10)*y(10)  (x(2) is rightPlain, y(2) if leftPlain)

    const gentype &cWeight     (int q = 0) const { return dRealConstants(q)(0); } // weight w_q for K_q
    int            cType       (int q = 0) const { return dtype(q);             } // type of K_q
    int            isNormalised(int q = 0) const { return isnorm(q);            } // set if K_q is scaled/shifted (x -> diag(sc).(x-sh))
    int            isMagTerm   (int q = 0) const { return ismagterm(q);         } // K(x,y,...)=K(x,x,...).K(y,y,...)....

          int              numSamples        (void) const { return xnumsamples; } // used when interpretting functions as distributions
    const Vector<gentype> &sampleDistribution(void) const { return xsampdist;   } // used when interpretting functions as distributions
    const Vector<int>     &sampleIndices     (void) const { return xindsub;     } // used when interpretting functions as distributions

    const kernPrecursor *getAltCall(int q = 0, int currml = -1) const { (void) currml; NiceAssert( currml != altcallback(q) ); kernPrecursor *res = nullptr; int ires = getaltML(res,altcallback(q)); NiceAssert( !ires ); (void) ires; return res; }

    const Vector<gentype>   &cRealConstants(int q = 0) const { return dRealConstants(q)(1,1,dRealConstants(q).size()-1,cRealConstantsTmp); } // real constants for K_q
    const Vector<int>       &cIntConstants (int q = 0) const { return dIntConstants(q);                                                    } // int constants for K_q
    const SparseVector<int> &cRealOverwrite(int q = 0) const { return dRealOverwrite(q);                                                   } // real constant overwrites for K_q
    const SparseVector<int> &cIntOverwrite (int q = 0) const { return dIntOverwrite(q);                                                    } // int constant overwrites for K_q

    const gentype &getRealConstZero(int q = 0) const { return cRealConstants(q)(0); }
          int      getIntConstZero (int q = 0) const { return cIntConstants(q)(0);  }

    double effweight(int q = 0) const; // return effective weight for kernel starting at q (ie product of is isSplit == 1 terms)

    // Nominal bounds on constants (used in tuneKernel)

    int isNomConst(int q = 0) const { return disNomConst(q); }

    const gentype &cWeightLB(int q = 0) const { return dRealConstantsLB(q)(0); }

    const Vector<gentype> &cRealConstantsLB(int q = 0) const { return dRealConstantsLB(q)(1,1,dRealConstantsLB(q).size()-1,cRealConstantsTmp); }
    const Vector<int>     &cIntConstantsLB (int q = 0) const { return dIntConstantsLB(q);                                                      }

    const gentype &getRealConstZeroLB(int q = 0) const { return cRealConstantsLB(q)(0); }
          int      getIntConstZeroLB (int q = 0) const { return cIntConstantsLB(q)(0);  }

    const gentype &cWeightUB(int q = 0) const { return dRealConstantsUB(q)(0); }

    const Vector<gentype> &cRealConstantsUB(int q = 0) const { return dRealConstantsUB(q)(1,1,dRealConstantsUB(q).size()-1,cRealConstantsTmp); }
    const Vector<int>     &cIntConstantsUB (int q = 0) const { return dIntConstantsUB(q);                                                      }

    const gentype &getRealConstZeroUB(int q = 0) const { return cRealConstantsUB(q)(0); }
          int      getIntConstZeroUB (int q = 0) const { return cIntConstantsUB(q)(0);  }

    // Random features
    //
    // getRandFeats: random features for kernel q, if any
    // isRandFeatReOnly: 0 for real and imaginary parts, 1 for real only, -1 for imaginary only
    // isRandFeatNoAngle: 0 for random angles, 1 for zero angle.

          int                             getnumRandFeats  (int q = 0) const { return randFeats(q).size(); }
    const Vector<SparseVector<gentype> > &getRandFeats     (int q = 0) const { return randFeats(q);        }
    const Vector<double>                 &getRandFeatAngle (int q = 0) const { return randFeatAngle(q);    }
          int                             isRandFeatReOnly (int q = 0) const { return randFeatReOnly(q);   }
          int                             isRandFeatNoAngle(int q = 0) const { return randFeatNoAngle(q);  }

    // Details on how the kernel "blocks" are put together
    //
    // Chaining: normally the total kernel function is:
    //
    // K = K_0 + K_1 + K_2 + K_3 + K_4 ...
    //
    // chaining involves taking the output of one kernel and feeding to the
    // input of the next kernel.  So if, for example, isChained(1) &&
    // isChained(2):
    //
    // K(...) = K_0(...) + K_3(K_2(K_1(...))) + K_4(...) + ...
    //
    // Note that chaining will only work for kernels that do not explicitly
    // use x,y (rather than x'x, x'y and y'y).  Kernels for which
    // !isKkitchensink(q) are fine.
    //
    //
    // Splitting: By splitting you can serparate kernels into multiple parts.
    // For example is isSplit(1) is set then:
    //
    // K(a,b,c,d,e) = K01(a,b).K23...(c,d,e)
    //
    // where K01 = K0+K1 (or different depending on chaining etc) and
    // K23... = K2+K3+....
    //
    // if issplit == 2 then this becomes *additive* rather than
    // multiplicative.
    //
    //
    // Multiplicative splitting: In this case the kernel can be built
    // as (in this example isMulSplit(2) = 1, rest 0):
    //
    // K(x,y) = K012(x,y).K34...(x,y)
    //
    // where K012 = K0+K1+K2 (or dfiferent depending on chaining etc) and
    // K23... = K2+K3+....
    //
    // if mulsplit == 2 then this becomes *additive* rather than
    // multiplicative.
    //
    //
    // The order of these is: first do mulsplit, then do split, then
    // do chaining.

    int isChained   (int q = 0) const { return ischain(q);   } // see above
    int isSplit     (int q = 0) const { return issplit(q);   } // see above
    int isMulSplit  (int q = 0) const { return mulsplit(q);  } // see above

    int numSplits   (void) const { return xnumSplits;    } // total number of splits
    int numMulSplits(void) const { return xnumMulSplits; } // total number of multiplicative splits

    // isKVarianceNZ: does K have variance (ie. inheritted from some GP or similar that uses averaging)

    int isKVarianceNZ(void) const override
    {
        int res = 0;

        //FIXME
        /*
        if ( size() )
        {
            int i;

            for ( i = 0 ; i < size() ; ++i )
            {
                if ( ( cType(i) >= 800 ) && ( cType(i) <= 899 ) )
                {
                    NiceAssert( MLid() != altcallback(i) );

                    if ( (*(getAltCall(i,mlid))).isKVarianceNZ(void) )
                    {
                        res = 1;
                        break;
                    }
                }
            }
        }
        */

        return res;
    }

    // Individual kernel block information.  Note that this does
    // not take norming into account.

    const kernInfo &kinf(int q) const { return kernflags(q); }
    const kernInfo  kinf(void)  const { return sum(kernflags); }

    // Vector forms
    //
    // getHyper: return weights and hyper-parameters: [ [ w_0 rp_{0,0} rp_{0,1} ... ] ; [ w_1 rp_{1,0} rp_{1,1} ... ] ; ... ]

    const Vector<int>               &getTypes         (void) const { return dtype;          }
    const Vector<Vector<gentype>>   &getHyper         (void) const { return dRealConstants; }
    const Vector<Vector<int>>       &getIntConstants  (void) const { return dIntConstants;  }
    const Vector<SparseVector<int>> &getRealOverwrites(void) const { return dRealOverwrite; }
    const Vector<SparseVector<int>> &getIntOverwrites (void) const { return dIntOverwrite;  }
    const Vector<int>               &getIsNormalised  (void) const { return isnorm;         }
    const Vector<int>               &getIsMagTerm     (void) const { return ismagterm;      }
    const Vector<int>               &getIsNomConst    (void) const { return disNomConst;    }

    const Vector<int> &getChained (void) const { return ischain;  }
    const Vector<int> &getSplit   (void) const { return issplit;  }
    const Vector<int> &getMulSplit(void) const { return mulsplit; }

    const Vector<Vector<gentype>> &getHyperLB(void) const { return dRealConstantsLB; }
    const Vector<Vector<gentype>> &getHyperUB(void) const { return dRealConstantsUB; }

    const Vector<Vector<int>> &getIntConstantsLB(void) const { return dIntConstantsLB; }
    const Vector<Vector<int>> &getIntConstantsUB(void) const { return dIntConstantsUB; }

    // Modifiers:

    MercerKernel &add   (int q);
    MercerKernel &remove(int q);
    MercerKernel &resize(int nsize);

    MercerKernel &setdefindKey(const Vector<int> &nv) { xdefindKey = nv; recalcRandFeats(-1); return *this; }

    MercerKernel &setSymmSet        (void) {                                                                                                      issymmset  = 1;                                  recalcRandFeats(-1); return *this; }
    MercerKernel &setNoSymmSet      (void) {                                                                                                      issymmset  = 0;                                  recalcRandFeats(-1); return *this; }
    MercerKernel &setFullNorm       (void) {                                                                                                      isfullnorm = 1;                                  recalcRandFeats(-1); return *this; }
    MercerKernel &setNoFullNorm     (void) {                                                                                                      isfullnorm = 0;                                  recalcRandFeats(-1); return *this; }
    MercerKernel &setProd           (void) {                    xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isprod     = 1;                                  recalcRandFeats(-1); return *this; }
    MercerKernel &setnonProd        (void) {                    xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isprod     = 0;                                  recalcRandFeats(-1); return *this; }
    MercerKernel &setLeftPlain      (void) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain  = 1;                  fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setRightPlain     (void) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; rightplain = 1;                  fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setLeftRightPlain (void) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain  = 1;  rightplain = 1; fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setLeftNormal     (void) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain  = 0;                  fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setRightNormal    (void) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; rightplain = 0;                  fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setLeftRightNormal(void) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain  = 0;  rightplain = 0; fixShiftProd(); recalcRandFeats(-1); return *this; }

    MercerKernel &setdenseZeroPoint(double nv) { xdenseZeroPoint = nv; return *this; }

    MercerKernel &setSymmSet        (int nv) {                                                                                                      issymmset       = nv;                 recalcRandFeats(-1); return *this; }
    MercerKernel &setFullNorm       (int nv) {                                                                                                      isfullnorm      = nv;                 recalcRandFeats(-1); return *this; }
    MercerKernel &setProd           (int nv) {                    xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isprod          = nv;                 recalcRandFeats(-1); return *this; }
    MercerKernel &setLeftPlain      (int nv) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain       = nv; fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setRightPlain     (int nv) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; rightplain      = nv; fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setAltDiff        (int nv) {                    xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isdiffalt       = nv;                 recalcRandFeats(-1); return *this; }
    MercerKernel &setsuggestXYcache (int nv) {                                                                                                      xsuggestXYcache = nv;                 recalcRandFeats(-1); return *this; }
    MercerKernel &setIPdiffered     (int nv) {                                                                                                      xisIPdiffered   = nv;                 recalcRandFeats(-1); return *this; }
    MercerKernel &setrankType       (int nv) {                    xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; xranktype       = nv;                                      return *this; }

    MercerKernel &setnumSamples        (int                    nv) { xnumsamples = nv; return *this; }
    MercerKernel &setSampleDistribution(const Vector<gentype> &nv) { xsampdist   = nv; return *this; }
    MercerKernel &setSampleIndices     (const Vector<int>     &nv) { xindsub     = nv; return *this; }

    MercerKernel &setIndexes(const Vector<int> &ndIndexes) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isind = 1; dIndexes = ndIndexes; fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setUnIndex(void)                         { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isind = 0;                       fixShiftProd(); recalcRandFeats(-1); return *this; }

    MercerKernel &setShift(const SparseVector<gentype> &ndShift) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isshift |= 1; dShift = ndShift; dShift.makealtcontent(); fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setScale(const SparseVector<gentype> &ndScale) { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isshift |= 2; dScale = ndScale; dScale.makealtcontent(); fixShiftProd(); recalcRandFeats(-1); return *this; }
    MercerKernel &setUnShiftedScaled(void)                       { xisIPdiffered = 1; xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isshift  = 0;                                            fixShiftProd(); recalcRandFeats(-1); return *this; }

    MercerKernel &setChained   (int q = 0)         { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ischain  ("&",q) = 1;                                     recalcRandFeats(q); return *this; }
    MercerKernel &setNormalised(int q = 0)         { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isnorm   ("&",q) = 1;                                     recalcRandFeats(q); return *this; }
    MercerKernel &setSplit     (int q = 0)         { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; issplit  ("&",q) = 1; xnumSplits    = calcnumSplits();    recalcRandFeats(q); return *this; }
    MercerKernel &setSplitAdd  (int q = 0)         { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; issplit  ("&",q) = 2; xnumSplits    = calcnumSplits();    recalcRandFeats(q); return *this; }
    MercerKernel &setMulSplit  (int q = 0)         { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; mulsplit ("&",q) = 1; xnumMulSplits = calcnumMulSplits(); recalcRandFeats(q); return *this; }
    MercerKernel &setAddSplit  (int q = 0)         { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; mulsplit ("&",q) = 2; xnumMulSplits = calcnumMulSplits(); recalcRandFeats(q); return *this; }
    MercerKernel &setMagTerm   (int q = 0)         { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ismagterm("&",q) = 1;                                     recalcRandFeats(q); return *this; }

    MercerKernel &setUnChained   (int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ischain  ("&",q) = 0;                                     recalcRandFeats(q); return *this; }
    MercerKernel &setUnNormalised(int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isnorm   ("&",q) = 0;                                     recalcRandFeats(q); return *this; }
    MercerKernel &setUnSplit     (int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; issplit  ("&",q) = 0; xnumSplits    = calcnumSplits();    recalcRandFeats(q); return *this; }
    MercerKernel &setUnMulSplit  (int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; mulsplit ("&",q) = 0; xnumMulSplits = calcnumMulSplits(); recalcRandFeats(q); return *this; }
    MercerKernel &setUnMagTerm   (int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ismagterm("&",q) = 0;                                     recalcRandFeats(q); return *this; }

    MercerKernel &setisNormalised(int nv, int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isnorm("&",q)    = nv; recalcRandFeats(q); return *this; }
    MercerKernel &setisMagTerm   (int nv, int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ismagterm("&",q) = nv; recalcRandFeats(q); return *this; }

    MercerKernel &setWeight (const gentype &nw, int q = 0) { dRealConstants("&",q)("&",0) = nw; recalcRandFeats(q); return *this; }
    MercerKernel &setType   (int ndtype,        int q = 0);
    MercerKernel &setAltCall(int newMLid,       int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; altcallback("&",q) = newMLid; recalcRandFeats(q); return *this; }

    MercerKernel &setisChained (int nv, int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ischain ("&",q) = nv;                                     recalcRandFeats(q); return *this; }
    MercerKernel &setisSplit   (int nv, int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; issplit ("&",q) = nv; xnumSplits    = calcnumSplits();    recalcRandFeats(q); return *this; }
    MercerKernel &setisMulSplit(int nv, int q = 0) { xisfast = -1; xneedsInnerm2 = xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; mulsplit("&",q) = nv; xnumMulSplits = calcnumMulSplits(); recalcRandFeats(q); return *this; }

    MercerKernel &setRealConstants(const Vector<gentype>   &ndRealConstants, int q = 0) { NiceAssert( dRealConstants(q).size()-1 == ndRealConstants.size() ); retVector<gentype> tmpva; dRealConstants("&",q)("&",1,1,dRealConstants(q).size()-1,tmpva) = ndRealConstants; recalcRandFeats(q); return *this; }
    MercerKernel &setIntConstants (const Vector<int>       &ndIntConstants,  int q = 0) { NiceAssert( dIntConstants(q).size()    == ndIntConstants.size()  );                           dIntConstants("&",q)                                            = ndIntConstants;  recalcRandFeats(q); return *this; }
    MercerKernel &setRealOverwrite(const SparseVector<int> &ndRealOverwrite, int q = 0) { dRealOverwrite("&",q) = ndRealOverwrite; fixcombinedOverwriteSrc(); recalcRandFeats(q); return *this; }
    MercerKernel &setIntOverwrite (const SparseVector<int> &ndIntOverwrite,  int q = 0) { dIntOverwrite("&",q)  = ndIntOverwrite;  fixcombinedOverwriteSrc(); recalcRandFeats(q); return *this; }

    MercerKernel &setRealConstZero(double nv, int q = 0) { dRealConstants("&",q)("&",1) = nv; recalcRandFeats(q); return *this; }
    MercerKernel &setIntConstZero (int    nv, int q = 0) { dIntConstants("&",q)("&",0)  = nv; recalcRandFeats(q); return *this; }

    // Nominal bounds on constants (used in tuneKernel)

    MercerKernel &setisNomConst(int nv, int q = 0) { disNomConst("&",q) = nv; return *this; }

    MercerKernel &setWeightLB(const gentype &nwLB, int q = 0) { dRealConstantsLB("&",q)("&",0) = nwLB; return *this; }

    MercerKernel &setRealConstantsLB(const Vector<gentype> &ndRealConstantsLB, int q = 0) { NiceAssert( dRealConstantsLB(q).size()-1 == ndRealConstantsLB.size() ); retVector<gentype> tmpva; dRealConstantsLB("&",q)("&",1,1,dRealConstantsLB(q).size()-1,tmpva) = ndRealConstantsLB; return *this; }
    MercerKernel &setIntConstantsLB (const Vector<int>     &ndIntConstantsLB,  int q = 0) { NiceAssert( dIntConstantsLB(q).size()  == ndIntConstantsLB.size() ); dIntConstantsLB("&",q) = ndIntConstantsLB; return *this; }

    MercerKernel &setRealConstZeroLB(double nvLB, int q = 0) { dRealConstantsLB("&",q)("&",1) = nvLB; return *this; }
    MercerKernel &setIntConstZeroLB (int    nvLB, int q = 0) { dIntConstantsLB("&",q)("&",0)  = nvLB; return *this; }

    MercerKernel &setWeightUB(const gentype &nwUB, int q = 0) { dRealConstantsUB("&",q)("&",0) = nwUB; return *this; }

    MercerKernel &setRealConstantsUB(const Vector<gentype> &ndRealConstantsUB, int q = 0) { NiceAssert( dRealConstantsUB(q).size()-1 == ndRealConstantsUB.size() ); retVector<gentype> tmpva; dRealConstantsUB("&",q)("&",1,1,dRealConstantsUB(q).size()-1,tmpva) = ndRealConstantsUB; return *this; }
    MercerKernel &setIntConstantsUB (const Vector<int>     &ndIntConstantsUB,  int q = 0) { NiceAssert( dIntConstantsUB(q).size()  == ndIntConstantsUB.size() ); dIntConstantsUB("&",q) = ndIntConstantsUB; return *this; }

    MercerKernel &setRealConstZeroUB(double nvUB, int q = 0) { dRealConstantsUB("&",q)("&",1) = nvUB; return *this; }
    MercerKernel &setIntConstZeroUB (int    nvUB, int q = 0) { dIntConstantsUB("&",q)("&",0)  = nvUB; return *this; }

    // setnumRandFeats triggers calculation/drawing of features.
    // setRandFeats and sertRandFeatAngle does this manually.

    MercerKernel &setnumRandFeats  (int nv,                             int q = 0) { recalcRandFeats(q,nv);       return *this; }
    MercerKernel &setRandFeats     (Vector<SparseVector<gentype> > &nv, int q = 0) { randFeats("&",q)       = nv; return *this; }
    MercerKernel &setRandFeatAngle (Vector<double>                 &nv, int q = 0) { randFeatAngle("&",q)   = nv; return *this; }
    MercerKernel &setRandFeatReOnly(int nv,                             int q = 0) { randFeatReOnly("&",q)  = nv; return *this; }
    MercerKernel &setRandFeatNoAngle(int nv,                            int q = 0) { randFeatNoAngle("&",q) = nv; return *this; }

    MercerKernel &setlinGradOrd (const Vector<int>             &nv) { linGradOrd  = nv; fixhaslinconstr();   return *this; }
    MercerKernel &setlinGradScal(const Vector<Matrix<double> > &nv) { linGradScal = nv; fixlingradscaltsp(); return *this; }

    MercerKernel &setlinGradOrd (int i,       int             nv) { linGradOrd("&",i)  = nv; fixhaslinconstr();    return *this; }
    MercerKernel &setlinGradScal(int i, const Matrix<double> &nv) { linGradScal("&",i) = nv; fixlingradscaltsp(i); return *this; }

    MercerKernel &setlinParity(const Vector<int> &nv) { linParity        = nv; return *this; }
    MercerKernel &setlinParity(int i,       int   nv) { linParity("&",i) = nv; return *this; }

    MercerKernel &setlinParityOrig(const Vector<gentype> &nv) { linParityOrig        = nv; return *this; }
    MercerKernel &setlinParityOrig(int i, const gentype  &nv) { linParityOrig("&",i) = nv; return *this; }

    // Vector forms
    //
    // getHyper: return weights and hyper-parameters: [ [ w_0 rp_{0,0} rp_{0,1} ... ] ; [ w_1 rp_{1,0} rp_{1,1} ... ] ; ... ]

    MercerKernel &setTypes         (const Vector<int>               &nv);
    MercerKernel &setHyper         (const Vector<Vector<gentype>>   &nv);
    MercerKernel &setIntConstantss (const Vector<Vector<int>>       &nv);
    MercerKernel &setRealOverwrites(const Vector<SparseVector<int>> &nv);
    MercerKernel &setIntOverwrites (const Vector<SparseVector<int>> &nv);
    MercerKernel &setIsNormalised  (const Vector<int>               &nv);
    MercerKernel &setIsMagTerm     (const Vector<int>               &nv);
    MercerKernel &setIsNomConst    (const Vector<int>               &nv) { resize(nv.size()); disNomConst = nv; return *this; }

    MercerKernel &setChained (const Vector<int> &nv);
    MercerKernel &setSplit   (const Vector<int> &nv);
    MercerKernel &setMulSplit(const Vector<int> &nv);

    MercerKernel &setHyperLB(const Vector<Vector<gentype>> &nv) { resize(nv.size()); dRealConstantsLB = nv; return *this; }
    MercerKernel &setHyperUB(const Vector<Vector<gentype>> &nv) { resize(nv.size()); dRealConstantsUB = nv; return *this; }

    MercerKernel &setIntConstantssLB(const Vector<Vector<int>> &nv) { resize(nv.size()); dIntConstantsLB = nv; return *this; }
    MercerKernel &setIntConstantssUB(const Vector<Vector<int>> &nv) { resize(nv.size()); dIntConstantsUB = nv; return *this; }

    // Element retrieval
    //
    // Sets res = x(i).direcref(j) (shifted/scaled if shifting/scaling is turned on)
    // and returns reference to this.

    gentype &xelm(gentype &res, const SparseVector<gentype> &x, int i, int j) const;
    int xindsize(const SparseVector<gentype> &x, int i) const;
    //const SparseVector<gentype> &getx(const SparseVector<gentype> &x, int i) const { (void) i; return x; }

    // Kernel-space distance
    //
    // ||x-y|| = K(x,x)+K(y,y)-2K(x,y)
    //
    // but may be accelerated for some cases (kernels 300-399)

    double distK(const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int assumreal = 0) const;
    void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int assumreal = 0) const;

    // Evaluate kernel K(x,y).
    //
    // Optional arguments: xnorm = x'x
    //                     ynorm = y'y
    //
    // If normalisation on then returns K(x,y)/sqrt(K(x,x)*K(y,y))
    //
    // Matrix arguments:
    //
    // - rows of left-hand matrix argument are x vectors
    // - columns of right-hand matrix argument are x vectors
    // - allrow forms assume arguments are rows in both left and right hand
    // - allcol forms assume arguments are columns in both left and right hand
    //
    // Biased forms:
    //
    // - bias is added to the inner product
    // - BiasedR implies bias vector is replaced by 1*b'
    // - BiasedL implies bias vector is replaced by b*1'
    //
    // Vector-less forms:
    //
    // - Only work for kernels that do not require explicit use of vectors
    //   x and y.  All pre-normalisation etc is assumed taken care of.
    // - biased: it is assumed that the bias HAS NOT BEEN ADDED to either
    //   xyprod or yxprod.
    // - real vectorless forms are very fast for isSimpleNNKernel types.
    //   These are the only K functions optimised for this type of kernel.
    //
    // Indexing:
    //
    // If xconsist is set then it is assumed that x and y share the same
    // indexes, which allows certain optimisations to occur (namely inner
    // products can be more quickly calculated).
    //
    // pxyprod: - if xyprod and diffis known then make this an array where
    //            pxyprod[0] points to xyprod and pxyprod[1] points to diffis
    //          - if xyprod is known only then " but pxyprod[1] = nullptr
    //          - if diffis is known only then " but pxyprod[0] = nullptr
    //          - otherwise set nullptr
    //          - for 8xx kernel pxyprod is the result
    //          - for simple chained 8xx kernels pxyprod is the result of the first part
    //
    //
    // Return as equation option (Keqn):
    //
    // - resmode = 0: (default) the result is a number (or matrix or whatever)
    //   K variance: if K is inherited then it can have inherent variance.
    //
    // +---------+--------+----------------+-------------+-----------+------------+
    // | resmode | x,y    | integer consts | real consts | calculate | calculate  |
    // | resmode | subbed |     subbed     |    subbed   |   dk/dr   | K variance |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 0 deflt |   y    |        y       |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 1       |        |        y       |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 2       |   y    |                |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 3       |        |                |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 4       |   y    |        y       |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 5       |        |        y       |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 6       |   y    |                |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 7       |        |                |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 8       |   y    |        y       |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 9       |        |        y       |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 10      |   y    |                |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 11      |        |                |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 12      |   y    |        y       |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 13      |        |        y       |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 14      |   y    |                |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 15      |        |                |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 128     |   y    |        y       |      y      |           |     y      |
    // +---------+--------+----------------+-------------+-----------+------------+
    //
    // - in equation: var(0,0) = x'x (if !(resmode|1))
    //                var(0,1) = y'y (if !(resmode|1))
    //                var(0,2) = x'y (if !(resmode|1))
    //                var(2,i) = ij  (if !(resmode|2))
    //                var(1,j) = rj  (if !(resmode|4))
    // - this can also do real constant gradients when resmode|8.  In this
    //   case the result is a vector of the required dimension, the elements
    //   of which may or may not be equations depending on the settings of
    //   the three LSB of resmode (see above table).

    gentype &Keqn(gentype &res, int resmode = 1) const
    {
        const static SparseVector<gentype> x;
        const static SparseVector<gentype> y;

        const static vecInfo xinfo;
        const static vecInfo yinfo;

//        K2(res,x,y,xinfo,yinfo,defaultgentype(),nullptr,DEFAULT_VECT_INDEX,DEFAULT_VECT_INDEX,0,0,resmode,0,nullptr,nullptr,nullptr,0);
        K2(res,x,y,xinfo,yinfo,0_gent,nullptr,DEFAULT_VECT_INDEX,DEFAULT_VECT_INDEX-1,0,0,resmode,0,nullptr,nullptr,nullptr,0);

        return res;
    }

    // Kernels for different norms.
    //
    // NB: - odd-order kernels implemented for fast kernels only.
    //     - The vectorial form can speed up k2xfer forms
    //     - for 2-norm forms:
    //     - you can do gradients directly through here (see mlinter.cc), but
    //       note these assume non-sparse formats.
    //
    // xy matrix stores either inner products [ <x,x> <x,y> ; <y,x> <y,y> ] or their transferred
    // equivalents [ K2xfer(x,x) K2xfer(x,y) ; K2xfer(y,x) K2xfer(y,y) ].
    //
    // xconsist:  set if we can assume the indices of x,y, etc and scale/shift are all the same
    // assumreal: set 1 if we assume x is real-valued to speed things up (call before doing kernel eval)

    gentype &K0(gentype &res, const gentype &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal = 0) const;
    double   K0(                    double   bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal = 0) const;

    gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, const gentype &bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, int assumreal = 0) const;
    double   K1(              const SparseVector<gentype> &xa, const vecInfo &xainfo,       double   bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, int assumreal = 0) const;

    gentype &K2(gentype &res, const SparseVector<gentype> &xa,  const SparseVector<gentype> &xb,  const vecInfo &xainfo, const vecInfo &xbinfo, const gentype &bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int assumreal = 0) const;
    double   K2(              const SparseVector<gentype> &xa,  const SparseVector<gentype> &xb,  const vecInfo &xainfo, const vecInfo &xbinfo,       double   bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int assumreal = 0) const;

    gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int k = DEFAULT_VECT_INDEX-2, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, const double *xy20 = nullptr, const double *xy21 = nullptr, const double *xy22 = nullptr, int assumreal = 0) const;
    double   K3(              const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int k = DEFAULT_VECT_INDEX-2, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, const double *xy20 = nullptr, const double *xy21 = nullptr, const double *xy22 = nullptr, int assumreal = 0) const;

    gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int k = DEFAULT_VECT_INDEX-2, int l = DEFAULT_VECT_INDEX-3, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, const double *xy20 = nullptr, const double *xy21 = nullptr, const double *xy22 = nullptr, const double *xy30 = nullptr, const double *xy31 = nullptr, const double *xy32 = nullptr, const double *xy33 = nullptr, int assumreal = 0) const;
    double   K4(              const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int k = DEFAULT_VECT_INDEX-2, int l = DEFAULT_VECT_INDEX-3, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, const double *xy20 = nullptr, const double *xy21 = nullptr, const double *xy22 = nullptr, const double *xy30 = nullptr, const double *xy31 = nullptr, const double *xy32 = nullptr, const double *xy33 = nullptr, int assumreal = 0) const;

    gentype &Km(int m, gentype &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const gentype &bias, Vector<int> &i, const gentype **pxyprod = nullptr, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const Matrix<double> *xy = nullptr, int assumreal = 0) const;
    double   Km(int m,               Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo,       double   bias, Vector<int> &i, const gentype **pxyprod = nullptr, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const Matrix<double> *xy = nullptr, int assumreal = 0) const;

    // 2-kernel product.
    //
    // K2x2(x,xa,xb) = K2(x,xa).K2(x,xb)
    //
    // The trick here is that this does the right thing with dense integrals on x,
    // allowing them to be incorporated for a limited set of kernels without the
    // need for numerical integration.

    gentype &K2x2(gentype &res, const SparseVector<gentype> &x,  const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,  const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo, const gentype &bias, int i = DEFAULT_VECT_INDEX, int ia = DEFAULT_VECT_INDEX-1, int ib = DEFAULT_VECT_INDEX-2, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, const double *xy20 = nullptr, const double *xy21 = nullptr, const double *xy22 = nullptr, int assumreal = 0) const;
    double   K2x2(              const SparseVector<gentype> &x,  const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,  const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo,       double   bias, int i = DEFAULT_VECT_INDEX, int ia = DEFAULT_VECT_INDEX-1, int ib = DEFAULT_VECT_INDEX-2, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, const double *xy20 = nullptr, const double *xy21 = nullptr, const double *xy22 = nullptr, int assumreal = 0) const;

    // Density function (if defined for RFF):

    double density(const SparseVector<gentype> &xa, const vecInfo &xainfo,       double   bias, int ia = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;

    // Feature maps
    //
    // phim: returns the image of x in feature space.  This may be finite dimensional if possible
    //       (and allowfinite = 1), but otherwise infinite dimensional.  Does not include bias term.
    // phidim: return -1 if feature map infinite dimensional, >= 0 otherwise

    int phi1(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(1,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }
    int phi1(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(1,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }

    int phi2(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(2,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }
    int phi2(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(2,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }

    int phi3(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(3,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }
    int phi3(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(3,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }

    int phi4(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(4,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }
    int phi4(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(4,res,x,xinfo,i,allowfinite,xdim,xconsist,assumreal); }

    int phim(int m, Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const;
    int phim(int m, Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const;

    int phidim(int allowfinite = 1, int xdim = 0) const { SparseVector<gentype> xdummy; vecInfo xinfodummy; Vector<double> resdummy; return phim(-1,resdummy,xdummy,xinfodummy,0,allowfinite,xdim); }

    // Inner-product calculation forms
    //
    // These just calculate the inner product, not the kernel.  If *simple* kernel transfer
    // is enabled then the inner product is actually the result of evaluating the transfered
    // kernel.  This operation is not well defined for non-simple kernels.

    double K0ip(double bias, const gentype **pxyprod, int xdim, int xconsist, int mlid, int assumreal) const;
    double K1ip(const SparseVector<gentype> &xa, const vecInfo &xainfo, double bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    double K2ip(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, double bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    double K3ip(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, double bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX-1, int ic = DEFAULT_VECT_INDEX-2, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    double K4ip(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, double bias, const gentype **pxyprod = nullptr, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX-1, int ic = DEFAULT_VECT_INDEX-2, int id = DEFAULT_VECT_INDEX-3, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    double Kmip(int m, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, double bias, const gentype **pxyprod = nullptr, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;

    // 2-norm derivatives
    //
    // dK2delx: calculates derivative w.r.t. x vector.  Result is xscaleres.x + yscaleres.y.
    // d2K2delxdelx: calculates 2nd derivative d/dx d/dx K.  Result is xxscaleres.x.x' + yyscaleres.y.y' + xyscaleres.x.y' + yxscaleres.y.x' + constres.I
    // d2K2delxdely: calculates 2nd derivative d/dx d/dy K.  Result is xxscaleres.x.x' + yyscaleres.y.y' + xyscaleres.x.y' + yxscaleres.y.x' + constres.I
    // dnK2del: nth derivative (currently only for RBF kernel).  Result is an array:
    //
    // dnK/dx_q0.dx_q1... sum_i sc_i kronProd_{j=0,1,...} [ x{n_ij}   if n_ij = 0,1
    //                                                    [ kd{n_ij}  if n_ij < 0
    //
    // where: x{0} = x
    //        x{1} = y
    //        kd{a} ... kd{a} = kronecker-delta (vectorised identity matrix) on indices
    //
    // If minmaxind >= 0 then derivative is only with respect to element minmaxind
    // of vectors x,y (so result is xscaleres.x(minmaxind) + yscaleres.y(minmaxind)).
    //
    // NB: this is actually the derivative wrt dScale.*(x-dShift) if shifting and/or
    //     scaling is present, so factor this in when calculating results.  That is
    //     d/dx_i => dScale_i d/dx_i etc

    void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int assumreal = 0) const;
    void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int assumreal = 0) const;

    void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDerive = 0, int assumreal = 0) const;
    void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDerive = 0, int assumreal = 0) const;

    void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDerive = 0, int assumreal = 0) const;
    void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDerive = 0, int assumreal = 0) const;

    void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDerive = 0, int assumreal = 0) const;
    void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDerive = 0, int assumreal = 0) const;

    // 2-norm derivatives (alternative form - deprecated)
    //
    // dK: assuming kernel can be written K(<x,y>,||x||^2,||y||^2), with symmetry in x,y, this
    //     returns the derivatives with respect to the first two arguments.
    //     If K is a simple transfer kernel then this derivative is with respect to the the
    //     arguments in the form K(K2xfer(x,y),K2xfer(x,x),K2xfer(y,y)) (the derivative
    //     dK/dK2xfer(x,y), xK/dK2xfer(x,x)).  This behaviour can be changed by setting the
    //     arguments deepDeriv to 1, which will recurse down to <x,y>,||x||^2,||y||^2 where
    //     possible.  Behaviour for non-simple transfer kernels is ill-defined (unless
    //     deepDeriv is set to 1).

    void dK(gentype &xygrad, gentype &xnormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDeriv = 0, int assumreal = 0) const;
    void dK(double  &xygrad, double  &xnormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDeriv = 0, int assumreal = 0) const;

    // Note: dk(x,y)/dynorm = dk(y,x)/dxnorm etc, standard assumptions necessary

    void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDeriv = 0, int assumreal = 0) const;
    void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo,       double   bias, const gentype **pxyprod = nullptr, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = nullptr, const double *xy10 = nullptr, const double *xy11 = nullptr, int deepDeriv = 0, int assumreal = 0) const;

    // "Reversing" functions.
    //
    // For speed of operation it is sometimes helpful to retrieve either the
    // inner product or distance from an evaluated kernel.  These functions
    // let you do that
    //
    // isReversible: test if kernel is reversible.  Output is:
    //     0: kernel cannot be reversed
    //     1: kernel can be reversed to produce <x,y>+bias
    //     2: kernel can be reversed to produce ||x-y||^2
    //
    // reverseK: reverse kernel as described by isReversible
    //
    // The result so produced can be fed back in via the pxyprod argument
    // (appropriately set) to speed up calculation of results.  Use case
    // could be quickly changing kernel parameters with minimal recalculation.
    //
    // As a general rule these only work with isSimpleFastKernel or
    // isSimpleKernelChain, and then in limited cases.  For the chain case
    // the result is the relevant (processed) output of the first layer.

    int isReversible(void) const;
    gentype &reverseK(gentype &res, const gentype &Kval) const;
    double  &reverseK(double &res,        double   Kval) const;

    // Evaluate kernel gradient dK/dx(x,y) and dK/dy(x,y)
    //
    // FIXME: at present this assumes everything is real-valued.  The more
    // general case is a little more difficult.
    //
    // Product kernels are not dealt with at present
    //
    // The returned value is in terms of x and y scales.  The gradient so
    // represented is of the form (x gradient case):
    //
    // dK/dx = xscaleres.x + yscaleres.y
    //
    // densedKdx: for product kernels only this calculates:
    //            d/dx0 d/dx1 ... K(x,y) = \prod_j dK(xj,yj)/dxj
    //            *This will disable callback for inner product calculation*
    // denseintK: reverse of densedKdx
    //
    // Design decision: for the kernel defined on max(x_k-y_k) the dense
    // derivative is simply the derivative on this axis.  This enables us
    // to estimate variance on the pareto frontier.
    //
    // At present this makes the following assumptions:
    //
    // - the caller is aware of any indexing tricks
    // - kernels are either inner product or norm difference kernels
    //
    // Biased gradients: these actually return gradients for the vectors
    //
    // ( x )  and  ( y    )
    // ( 1 )       ( bias )
    //
    // (so xscaleres refers to the scale for the augmented vector, and like-
    // wise yscaleres).  This makes surprisingly little difference if you
    // want the gradients for x and y as the scale factors are the same
    // (dxaug/dx = diag(I,0)).  To calculate the bias gradient, note that:
    //
    // dK/dbias = dK/dxaug dxaug/dbias + dK/dyaug dyaug/dbias
    //          = dK/dyaug dyaug/dbias
    //          = dyxscaleres + bias.dyyscaleres
    //
    // minmaxind: -1  if gradient is for whole x/y
    //            >=0 if gradient is for just one element of min/max(x-y)
    //
    // For Km gradients: xyscaleres refers to scaling factor on x(0).^{m-1} (for dx) or x(1).^{m-1} (for dy) (.^ is the elementwise power)
    //                   zscaleres refers to scaling factor on x(i0)*x(i1)*...x(i{m-2}) (elementwise product of all x except x(0) (for dx) or x(1) (for dy))
    //
    // Currently second order gradients are not implemented for Km kernels.  If
    // you want to implement them bear in mind that you will need xxscaleres for
    // weighting (m-1)-order terms (and yyscaleres etc), plus constres, but also
    // weight terms for (m-2) order terms, so things will get a bit tricky.
    //
    // Only a very limited subset of second order derivatives are defined!

    // Dense derivatives and integrals

    void densedKdx(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, double bias, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    void denseintK(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, double bias, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX-1, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;

    // Get vector information, taking into account indexing.
    //
    // scratch: may or may not be used for something or other, saves on allocs and statics
    //
    // xmag: 2-norm, if known

    vecInfo     &getvecInfo(vecInfo     &res, const SparseVector<gentype> &x, const gentype *xmag = nullptr, int xconsist = 0, int assumreal = 0) const;
    vecInfoBase &getvecInfo(vecInfoBase &res, const SparseVector<gentype> &x, const gentype *xmag = nullptr, int xconsist = 0, int assumreal = 0) const;

    const gentype &getmnorm(const vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist = 0, int assumreal = 0) const;
          gentype &getmnorm(      vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist = 0, int assumreal = 0) const;

    // Kernel scaling

    MercerKernel &operator*=(double sf)
    {
        int q;
        int goon = 0;

        for ( q = 0 ; q < size() ; ++q )
        {
            if ( !isChained(q) || isSplit(q) || isMulSplit(q) )
            {
                dRealConstants("&",q)("&",0) *= ( goon ? sf : 1.0 );
            }

            if ( isSplit(q) == 1 )
            {
                goon = 0;
            }

            if ( isSplit(q) == 2 )
            {
                goon = 1;
            }

            if ( isMulSplit(q) == 1 )
            {
                goon = 0;
            }

            if ( isMulSplit(q) == 2 )
            {
                goon = 1;
            }
        }

        recalcRandFeats(-1);

        return *this;
    }

private:

    int calcnumSplits(int indstart = 0, int indend = -1) const
    {
        if ( indend == -1 )
        {
            indend = size()-1;
        }

        if ( indend <= 0 )
        {
            return 0;
        }

        int i,res = 0;

        for ( i = indstart ; i < indend ; ++i )
        {
            res += ( issplit(i) ? 1 : 0 );
        }

        return res;
/*
        retVector<int> tmp;

        if ( indend == -1 )
        {
            indend = size()-1;
        }

        return ( indend <= 0 ) ? 0 : sum(issplit(indstart,1,indend-1,tmp));
*/
    }

    int calcnumMulSplits(int indstart = 0, int indend = -1) const
    {
        if ( indend == -1 )
        {
            indend = size()-1;
        }

        if ( indend <= 0 )
        {
            return 0;
        }

        int i,res = 0;

        for ( i = indstart ; i < indend ; ++i )
        {
            res += ( mulsplit(i) ? 1 : 0 );
        }

        return res;
    }

    // Terms used:
    //
    // - normalised:         Kn(x,y) = K(x,y)/sqrt(|K(x,x)|.|K(y,y)|)
    // - shifted and scaled: Ks(x,y) = K((x+shift).*scale,(y+shift).*scale)
    //
    // dtype: kernel type vector.
    // isprod:     0 = normal, 1 = K(x,y) = prod_i K(x_i,y_i)
    // isnorm:     0 = normal, 1 = normalisation on.
    // isdiffalt:  see previous
    // ischain:    0 = normal, 1 = this kernel is then chained into next kernel.
    // issplit:    0 = normal, 1 = this kernel (for this part of x) stops here, next kernel (for this part of x) starts, 2 means additive split
    // mulsplit:   0 = normal, 1 = this kernel (for all x) stops here, next kernel (for all x) starts, 2 for additive split.
    // ismagtern   0 = normal, 1 = use K(x,x).K(y,y) rather than K(x,y).
    // isshift:    0 = normal, 1 = use shifting only, 2 = use scaling only, 3 = use shifting and scaling
    // isind:      0 = normal, 1 = use indexed products
    // isfullnorm: 0 = normal, 1 = normalise at outermost
    // leftplain:  0 = normal, 1 = don't shift-scale left-hand argument in K
    // rightplain: 0 = normal, 1 = don't shift-scale right-hand argument in K
    // weight: weight factor kernel is multiplied by
    //         (now stored as index 0 of real constants)
    // dIntConstants: integer constants in kernel
    // dRealConstants: real constants in kernel
    // dIntConstantsLB: lower bound integer constants in kernel
    // dRealConstantsLB: lower bound real constants in kernel
    // dIntConstantsUB: upper bound integer constants in kernel
    // dRealConstantsUB: upper bound real constants in kernel
    // dIntOverwrite: selects which variables will be overwritten by which x(i)*y(i)
    // dRealOverwrite: selects which variables will be overwritten by which x(i)*y(i)
    // dIndexes: indices used in index products
    // dShift: shift factor
    // dScale: scale factor
    // dShiftProd: ||dShift.*dScale||_2^2

    Vector<int> xdefindKey;

    mutable int isind;
    int isshift;
    int leftplain;
    int rightplain;
    int isprod;
    int isdiffalt;
    int xproddepth;
    int enchurn; // set if kernel reversal is enabled.
    int xsuggestXYcache;
    int xisIPdiffered;
    int isfullnorm;
    int issymmset;
    int xnumSplits;
    int xnumMulSplits;
    int xranktype;
    double xdenseZeroPoint;

    Vector<int> dtype;
    Vector<int> isnorm;
    Vector<int> ischain;
    Vector<int> issplit;
    Vector<int> mulsplit;
    Vector<int> ismagterm;
    mutable Vector<int> dIndexes;
    Vector<kernInfo> kernflags;
    mutable Vector<Vector<gentype> > dRealConstants;
    mutable Vector<Vector<int> > dIntConstants;
    mutable Vector<int> disNomConst;
    mutable Vector<Vector<gentype> > dRealConstantsLB;
    mutable Vector<Vector<int> > dIntConstantsLB;
    mutable Vector<Vector<gentype> > dRealConstantsUB;
    mutable Vector<Vector<int> > dIntConstantsUB;
    Vector<SparseVector<int> > dRealOverwrite;
    Vector<SparseVector<int> > dIntOverwrite;
    Vector<int> altcallback;
    Vector<Vector<SparseVector<gentype> > > randFeats;
    Vector<Vector<double> > randFeatAngle;
    Vector<int> randFeatReOnly; // 0 both real and im, 1 re only, -1 im only
    Vector<int> randFeatNoAngle; // 0 for random angle, 1 for no angle.

    SparseVector<gentype> dShift;
    SparseVector<gentype> dScale;
    gentype dShiftProd;
    gentype dShiftProdNoConj;
    gentype dShiftProdRevConj;

    mutable retVector<gentype> cRealConstantsTmp;
    mutable retVector<int> cRealConstantsTmpb;

    Vector<int> linGradOrd;
    Vector<Matrix<double> > linGradScal;
    Vector<Matrix<double> > linGradScalTsp;
    bool haslinconstr;

    Vector<int> linParity;
    Vector<gentype> linParityOrig;

    // random features calculation stub.  If numFeats = -1 then just
    // fix the features that exist, otherwise start from scratch with
    // the specified number of random features.

    void recalcRandFeats(int q, int numFeats = -1);

    // Feature no longer used, assume enchurn == 0
    //
    // churnInner:   set 1 if we want to attempt to retrieve and reuse
    //               inner products <x,y>+b and distances ||x-y||^2 when
    //               changing the kernel (see prepareKernel in ml_base).
    //               This does not guarantee that retrieval will occur, but
    //               only that if it is possible and implemented then it will
    //               be attempted when feasible.  Only really speeds things up
    //               when you're using kernel inheritance.
    //

    int churnInner(void) const { return enchurn; }
    MercerKernel &setChurnInner(int nv) { enchurn = nv; return *this; }

    // Distribution kernel information:
    //
    // xnumSamples: number of samples to estimate the kernel
    // xindsub:     indices of variables in distributions that are substituted
    // xsampdist:   sample distribution for these variables
    //
    // That is, we are estimating:
    //
    // E_a[K(x(a),y(a))]
    //
    // where x and y are (or contain) *distributions*, so
    //
    // x \sim dist(a)
    //
    // and a itself if drawn from sampdist, and has indices xindsub.

    int xnumsamples;
    Vector<int> xindsub;
    Vector<gentype> xsampdist;

    // Call tree to calculate kernel.
    //
    // - public version calculates xyprod etc and calls first version
    //
    // - yyyK version does preprocessing on [ xa ~ xb ... ] forms
    // - next levels goes through kernel structure and calls second version
    //   to evaluate individual kernels in structure (indexed by q)
    // - this calls unnorm form and applies normalisation if needed
    // - unnorm form calculates diffis (||x-y||^2) if needed and then
    //   calls Kbase
    // - Kbase does the actual work.
    // - resmode = 0: standard evaluation
    //   resmode = 1: return equation, including constants
    //   resmode = 2: return equation, integers substituted out
    //   resmode = 3: return equation, integers and reals subbed out
    //     var(0,0) = x'x
    //     var(0,1) = y'y
    //     var(0,2) = x'y
    //     var(1,j) = rj (real constants) (if resmode == 1,2)
    //     var(2,i) = ij (integer constants) (if resmode == 1)
    //
    // K4 and Km are similar but work with 4 and m norms before finally
    // converging onto the same Kbase for final evaluation
    //
    // densetype = 0: normal operation
    //             1: calculate dense derivative d/dx0 d/dx1 ... K(x,y)
    //             2: calculate dense integral int_x0 ind_x1 ... K(x,y)
    //            -1: calculate dense derivative d/dy0 d/dy1 ... K(x,y)
    //            -2: calculate dense integral int_y0 ind_y1 ... K(x,y)
    //
    // iset: controls how distributions in the data are treated.
    //     0 = standard behaviour, distributions are treated as such, result is average of samples
    //     1 = distributions represent draws from infinite sets, result is largest (most similar) evaluation with draws from given distribution(s)
    //
    // The actual dense derivative/integral operations work by pairing
    // kernels.  That is, if K is the kernel then the dense derivative will
    // find the kernel corresponding to the derivative and then call that
    // instead.  Only works if required pair is defined.
    //
    // FIXME: will also only work for simple kernels
    //
    //
    //
    //
    // ------------------------------------------------------------------
    //
    // More detail on the yyyK... levels
    //
    //    inline gentype &K2(gentype &res, const SparseVector<gentype> &x,  const SparseVector<gentype> &y,  const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = nullptr, int i = DEFA$
    //    inline double  &K2(double  &res, const SparseVector<gentype> &x,  const SparseVector<gentype> &y,  const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = nullptr, int i = DEFA$
    // these translate to gentype and call direct to...
    //
    //     template <class T> T &yyyK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, $ 
    // this does nothing (for now) but calls direct to...
    //
    //     template <class T> T &yyyaK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim,$ 
    // *data dependent operation*
    // if diagonal kernel: return diagonal
    // if simple evaluation: jump to xKKK2
    // otherwise calls to...
    //
    //     template <class T> T &yyybK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim,$ 
    // *data dependent operation*
    // if far reference present then evaluate as rank kernel by computing difference...
    // evaluation by calling...
    //
    //     template <class T> T &yyycK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim,$ 
    // *data dependent operation*
    // does gradients if required by branching to gradient functions (K2 only).
    // otherwise calls to...
    //
    //    template <class T> T &yyyKK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim,$ 
    // *data dependent operation*
    // branching for setwise kernels.  For example K2([x~y],[a~b~c]) = K5(x,y,a,b,c).
    // based on nupsize calls to xkkk2, xkkk3, ...,
    // but usually goes through to...
    //
    //     template <class T> T &xKKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int $ 
    // *kernel dependent operation*
    // processes mulSplits, which are points in the kernel dictionary that evaluate as <left of split>*<right of split>
    // each "split" evaluated by calling...
    //
    //     template <class T> T &xKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int x$ 
    // *kernel dependent operation*
    // processes normalisation and splits.
    // normalisation is when K(x,y) -> K(x,y)/sqrt(K(x,x)*K(y,y)).
    // designed to allow for changes made in yyykk2 function!
    // does this by calling (in the most basic path)...
    //
    //     template <class T> T &KK2(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **px$ 
    // *data dependent operation*
    // does kernels over distributions.  If arguments are functions then this does monte-carlo integration on an even grid.
    // does this by calling...
    //
    //     template <class T> T &LL2(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **px$ 
    // *kernel and data dependent operation*
    // actual kernel evaluation begins.
    // ip evaluation branches here
    // isprod evaluation recurses from here
    // kernel evaluation for relatively simple kernels goes here.
    // Otherwise falls through to K2i which does the full tree as described previously.

    double   yyyK2x2(              const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo,       double   bias, int i, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    gentype &yyyK2x2(gentype &res, const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo, const gentype &bias, int i, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;

    template <class T> T &yyyaaK2x2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo, int xignorefarfar, int xaignorefarfar, int xbignorefarfar, int xignorefarfarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xgradordadd, int xagradordadd, int xbgradordadd, int xgradordaddR, int xagradordaddR, int xbgradordaddR, const T &bias, int i, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK2x2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo, int xignorefarfar, int xaignorefarfar, int xbignorefarfar, int xignorefarfarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xgradordadd, int xagradordadd, int xbgradordadd, int xgradordaddR, int xagradordaddR, int xbgradordaddR, const T &bias, int i, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    template <class T> T &yyybK2x2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo, int xgradOrder, int xagradOrder, int xbgradOrder, int xgradOrderR, int xagradOrderR, int xbgradOrderR, int iupm, int iaupm, int ibupm, int xfarpresent, int xafarpresent, int xbfarpresent, double xrankw, double xarankw, double xbrankw, double xArankw, double xBrankw, int xfarfarpresent, int xafarfarpresent, int xbfarfarpresent, int xfarfarfarpresent, int xafarfarfarpresent, int xbfarfarfarpresent, int xgradup, int xagradup, int xbgradup, int xgradupR, int xagradupR, int xbgradupR, int xignorefarfar, int xaignorefarfar, int xbignorefarfar, int xignorefarfarfar, int xaignorefarfarfar, int xbignorefarfarfar, const T &bias, int i, int ia, int ib, int xdim, int xbonsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iset, int iaset, int ibset, int assumreal, int justcalcip, int densetype, int adensetype, int bdensetype) const;
    template <class T> T &yyycK2x2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo, int xgradOrder, int xagradOrder, int xbgradOrder, int xgradup, int xagradup, int xbgradup, int iupm, int iaupm, int ibupm, const SparseVector<gentype> &xff, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const T &bias, int i, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iset, int iaset, int ibset, int assumreal, int justcalcip, int densetype, int adensetype, int bdensetype) const;
    template <class T> T &yyyKK2x2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xvinfo, int xgradOrder, int xagradOrder, int xbgradOrder, int xgradup, int xagradup, int xbgradup, int iupm, int iaupm, int ibupm, const SparseVector<gentype> &xff, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const T &bias, int i, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iset, int iaset, int ibset, int assumreal, int justcalcip, int densetype, int adensetype, int bdensetype) const;

    double yyyK0(       double bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    double yyyK1(       const SparseVector<gentype> &xa, const vecInfo &xainfo, double bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int resmode, int mlid, const double *xy, int assumreal, int justcalcip) const;
    double yyyK2(       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, double bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const;
    double yyyK3(       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, double bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    double yyyK4(       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, double bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const;
    double yyyKm(int m, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, double bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, int assumreal, int justcalcip) const;

    gentype &yyyK0(       gentype &res, const gentype &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    gentype &yyyK1(       gentype &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, const gentype &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int resmode, int mlid, const double *xy, int assumreal, int justcalcip) const;
    gentype &yyyK2(       gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const gentype &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const;
    gentype &yyyK3(       gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const gentype &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    gentype &yyyK4(       gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const;
    gentype &yyyKm(int m, gentype &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const gentype &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, int assumreal, int justcalcip) const;

    template <class T> T &yyyaaK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyyaaK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xaignorefarfar, int xaignorefarfarfar, int xagradordadd, int xagradordaddR, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int resmode, int mlid, const double *xy, int assumreal, int justcalcip) const;
    template <class T> T &yyyaaK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int xaignorefarfar, int xbignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xagradordadd, int xbgradordadd, int xagradordaddR, int xbgradordaddR, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const;
    template <class T> T &yyyaaK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xaignorefarfar, int xbignorefarfar, int xcignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xcignorefarfarfar, int xagradordadd, int xbgradordadd, int xcgradordadd, int xagradordaddR, int xbgradordaddR, int xcgradordaddR, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    template <class T> T &yyyaaK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xaignorefarfar, int xbignorefarfar, int xcignorefarfar, int xdignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xcignorefarfarfar, int xdignorefarfarfar, int xagradordadd, int xbgradordadd, int xcgradordadd, int xdgradordadd, int xagradordaddR, int xbgradordaddR, int xcgradordaddR, int xdgradordaddR, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const;
    template <class T> T &yyyaaKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &ignorefarfar, Vector<int> &ignorefarfarfar, Vector<int> &xgradordadd, Vector<int> &xgradordaddR, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, int assumreal, int justcalcip) const;

    template <class T> T &yyyaK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xaignorefarfar, int xaignorefarfarfar, int xagradordadd, int xagradordaddR, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int resmode, int mlid, const double *xy, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int xaignorefarfar, int xbignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xagradordadd, int xbgradordadd, int xagradordaddR, int xbgradordaddR, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xaignorefarfar, int xbignorefarfar, int xcignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xcignorefarfarfar, int xagradordadd, int xbgradordadd, int xcgradordadd, int xagradordaddR, int xbgradordaddR, int xcgradordaddR, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xaignorefarfar, int xbignorefarfar, int xcignorefarfar, int xdignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xcignorefarfarfar, int xdignorefarfarfar, int xagradordadd, int xbgradordadd, int xcgradordadd, int xdgradordadd, int xagradordaddR, int xbgradordaddR, int xcgradordaddR, int xdgradordaddR, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const;
    template <class T> T &yyyaKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &ignorefarfar, Vector<int> &ignorefarfarfar, Vector<int> &xgradordadd, Vector<int> &xgradordaddR, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, int assumreal, int justcalcip) const;

    template <class T> T &yyybK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyybK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xagradOrder, int xagradOrderR, int iaupm, int xafarpresent, double xarankw, double xArankw, int xafarfarpresent, int xafarfarfarpresent, int xagradup, int xagradupR, int xaignorefarfar, int xaignorefarfarfar, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int resmode, int mlid, const double *xy, int iaset, int assumreal, int justcalcip) const;
    template <class T> T &yyybK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int xagradOrder, int xbgradOrder, int xagradOrderR, int xbgradOrderR, int iaupm, int ibupm, int xafarpresent, int xbfarpresent, double xarankw, double xbrankw, double xArankw, double xBrankw, int xafarfarpresent, int xbfarfarpresent, int xafarfarfarpresent, int xbfarfarfarpresent, int xagradup, int xbgradup, int xagradupR, int xbgradupR, int xaignorefarfar, int xbignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal, int justcalcip, int adensetype, int bdensetype) const;
    template <class T> T &yyybK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xagradOrderR, int xbgradOrderR, int xcgradOrderR, int iaupm, int ibupm, int icupm, int xafarpresent, int xbgrapresent, int xcfarpresent, double xarankw, double xbrankw, double xcrankw, double xArankw, double xBrankw, double xCrankw, int xafarfarpresent, int xbfarfarpresent, int xcfarfarpresent, int xafarfarfarpresent, int xbfarfarfarpresent, int xcfarfarfarpresent, int xagradup, int xbgradup, int xcgradup, int xagradupR, int xbgradupR, int xcgradupR, int xaignorefarfar, int xbignorefarfar, int xcignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xcignorefarfarfar, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const;
    template <class T> T &yyybK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xdgradOrder, int xagradOrderR, int xbgradOrderR, int xcgradOrderR, int xdgradOrderR, int iaupm, int ibupm, int icupm, int idupm, int xafarpresent, int xbfarpresent, int xcfarpresent, int xdfarpresent, double xarankw, double xbrankw, double xcrankw, double xdrankw, double xArankw, double xBrankw, double xCrankw, double xDrankw, int xafarfarpresent, int xbfarfarpresent, int xcfarfarpresent, int xdfarfarpresent, int xafarfarfarpresent, int xbfarfarfarpresent, int xcfarfarfarpresent, int xdfarfarfarpresent, int xagradup, int xbgradup, int xcgradup, int xdgradup, int xagradupR, int xbgradupR, int xcgradupR, int xdgradupR, int xaignorefarfar, int xbignorefarfar, int xcignorefarfar, int xdignorefarfar, int xaignorefarfarfar, int xbignorefarfarfar, int xcignorefarfarfar, int xdignorefarfarfar, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const;
    template <class T> T &yyybKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &xgradOrder, Vector<int> &xgradOrderR, Vector<int> &iupm, Vector<int> &xfarpresent, Vector<double> &xxrankw, Vector<double> &xXrankw, Vector<int> &sfarfarpresent, Vector<int> &sfarfarfarpresent, Vector<int> &xgradup, Vector<int> &xgradupR, Vector<int> &ignorefarfar, Vector<int> &ignorefarfarfar, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iset, int assumreal, int justcalcip) const;
    template <class T> T &yyybKmb(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const SparseVector<gentype> *> &xn, Vector<const SparseVector<gentype> *> &xf, Vector<const SparseVector<gentype> *> &xff, Vector<const SparseVector<gentype> *> &xfff, Vector<const vecInfo *> &xinfo, Vector<int> &xgradOrder, Vector<int> &xgradOrderR, Vector<int> &iupm, Vector<int> &xfarpresent, Vector<double> &xxrankw, Vector<double> &xXrankw, Vector<int> &xgradup, Vector<int> &xgradupR, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iiset, int assumreal, int justcalcip) const;

    template <class T> T &yyycK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyycK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xagradOrder, int xagradup, int iaupm, const SparseVector<gentype> &xaff, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int resmode, int mlid, const double *xy, int iaset, int assumreal, int justcalcip) const;
    template <class T> T &yyycK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &ybinfo, int xagradOrder, int xbgradOrder, int xagradup, int xbgradup, int iaupm, int ibupm, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal, int justcalcip, int adensetype, int bdensetype) const;
    template <class T> T &yyycK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xagradup, int xbgradup, int xcgradup, int iaupm, int ibupm, int icupm, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const;
    template <class T> T &yyycK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xdgradOrder, int xagradup, int xbgradup, int xcgradup, int xdgradup, int iaupm, int ibupm, int icupm, int idupm, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const SparseVector<gentype> &xdff, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const;
    template <class T> T &yyycKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &xgradOrder, Vector<int> &xgradup, Vector<int> &iupm, Vector<const SparseVector<gentype> *> &xff, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iset, int assumreal, int justcalcip) const;

    template <class T> T &yyyKK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyyKK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xagradOrder, int xagradup, int iaupm, const SparseVector<gentype> &xaff, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int resmode, int mlid, const double *xy, int iaset, int assumreal, int justcalcip) const;
    template <class T> T &yyyKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int xagradOrder, int xbgradOrder, int xagradup, int xbgradup, int iaupm, int ibupm, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal, int justcalcip, int adensetype, int bdensetype) const;
    template <class T> T &yyyKK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xagradup, int xbgradup, int xcgradup, int iaupm, int ibupm, int icupm, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const;
    template <class T> T &yyyKK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xdgradOrder, int xagradup, int xbgradup, int xcgradup, int xdgradup, int iaupm, int ibupm, int icupm, int idupm, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const SparseVector<gentype> &xdff, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const;
    template <class T> T &yyyKKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const Vector<int> &xgradOrder, const Vector<int> &xgradup, const Vector<int> &xupm, Vector<const SparseVector<gentype> *> &xff, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iset, int assumreal, int justcalcip) const;

    template <class T> T &xKKK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip) const;
    template <class T> T &xKKK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xagradOrder, const SparseVector<gentype> &xaff, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int iaset) const;
    template <class T> T &xKKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int xagradOrder, int xbgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int justcalcip, int iset, int jset, int adensetype, int bdensetype) const;
    template <class T> T &xKKK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int justcalcip, int iaset, int ibset, int icset) const;
    template <class T> T &xKKK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xdgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const SparseVector<gentype> &xdff, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int justcalcip, int iaset, int ibset, int icset, int idset) const;
    template <class T> T &xKKKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const Vector<int> &xgradOrder, Vector<const SparseVector<gentype> *> &xff, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, int justcalcip, const Vector<int> *iset) const;

    template <class T> T &xKK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, int indstart, int indend, int ns) const;
    template <class T> T &xKK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xagradOrder, const SparseVector<gentype> &xaff, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int iaset, int indstart, int indend, int ns) const;
    template <class T> T &xKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int xagradOrder, int xbgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double * xy10, const double *xy11, int justcalcip, int iset, int jset, int indstart, int indend, int ns, int adensetype, int bdensetype) const;
    template <class T> T &xKK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int justcalcip, int iaset, int ibset, int icset, int indstart, int indend, int ns) const;
    template <class T> T &xKK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xdgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const SparseVector<gentype> &xdff, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int indstart, int indend, int ns) const;
    template <class T> T &xKKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const Vector<int> &xgradOrder, Vector<const SparseVector<gentype> *> &xaff, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, int justcalcip, const Vector<int> *iset, int indstart, int indend, int ns) const;

    template <class T> T &KK0(T &res, T &logres, int &logresvalid, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, int indstart, int indend, int skipbias = 0) const;
    template <class T> T &KK1(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const vecInfo &xainfo, int xagradOrder, const SparseVector<gentype> &xaff, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int indstart, int indend, int iaset, int skipbias = 0, int skipxa = 0) const;
    template <class T> T &KK2(int adensetype, int bdensetype, T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int xagradOrder, int xbgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int justcalcip, int indstart, int indend, int iset, int jset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;
    template <class T> T &KK3(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const Vector<int> *s, int justcalcip, int indstart, int indend, int iaset, int ibset, int icset, int skipbias = 0, int skipxa = 0, int skipxb = 0, int skipxc = 0) const;
    template <class T> T &KK4(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xdgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const SparseVector<gentype> &xdff, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const Vector<int> *s, int justcalcip, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int indstart, int indend, int iaset, int ibset, int icset, int idset, int skipbias = 0, int skipxa = 0, int skipxb = 0, int skipxc = 0, int skipxd = 0) const;
    template <class T> T &KK6(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const SparseVector<gentype> &xe, const SparseVector<gentype> &xf, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const vecInfo &xeinfo, const vecInfo &xfinfo, int xagradOrder, int xbgradOrder, int xcgradOrder, int xdgradOrder, int xegradOrder, int xfgradOrder, const SparseVector<gentype> &xaff, const SparseVector<gentype> &xbff, const SparseVector<gentype> &xcff, const SparseVector<gentype> &xdff, const SparseVector<gentype> &xeff, const SparseVector<gentype> &xf4, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int ie, int jf, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend, int iaset, int ibset, int icset, int idset, int ieset, int ifset) const;
    template <class T> T &KKm(int m, T &res, T &logres, int &logresvalid, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const Vector<int> &xgradOrder, Vector<const SparseVector<gentype> *> &xff, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend, const Vector<int> *iset, int skipbias = 0, int skipx = 0) const;

    template <class T> T &LL0(T &res, T &logres, int &logresvalid, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL1(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const vecInfo &xainfo, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL2(int adensetype, int bdensetype, T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL3(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const Vector<int> *s, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL4(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const Vector<int> *s, int justcalcip, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int indstart, int indend) const;
    template <class T> T &LLm(int m, T &res, T &logres, int &logresvalid, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend) const;

    double LL2fast(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, double bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, const double *xy00, const double *xy10, const double *xy11) const;

    // phim evaluation tree is similar:
    //
    // - phim(m,...)
    // calls direct to:
    //
    // - yyyphim(m,...)
    // calls direct to:
    //
    // - yyyaphim(m,...)
    // *data dependent operation*
    // assert that this is not a diagonal.
    // call direct to:
    //
    // - yyybphim(m,...)
    // *data dependent operation*
    // if far reference present then return difference of feature maps for near and far,
    // evaluation by calling:
    //
    // - yyycphim(m,...)
    // *data dependent operation*
    // extract order of gradient and add as parameter,
    // then call to:
    //
    // - yyyPphim(m,...,gradOrder)
    // *data dependent operation*
    // if setwise then evaluate next level as elementwise product of each call to,
    // evaluation by calling:
    //
    // - xPPphim(m,...,gradOrder,usize)
    // *kernel dependent operation*
    // if there are multiplicative splits then return kronprod of each,
    // then call:
    //
    // - xPphim(m,...,gradOrder,usize)
    //     template <class T> T &xKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int x$ 
    // *kernel dependent operation*
    // normalise if needed, dividing by appropriate root of normalisation constant, throw if there is non-trivial normalisation due to setwise operation,
    // then call to:
    //
    // - Pphim(m,...,gradOrder)
    //     template <class T> T &KK2(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **px$ 
    // *data dependent operation*
    // If arguments are functions then this computes the average on an even grid,
    // calling to:
    //
    // - Qqhim(m,...,gradOrder)
    // *kernel and data dependent operation*
    // assert !isprod
    // if not a simple kernel sum then throw an exception
    // do sum of kernels by appending vectors to each other
    // weighting by multiplying by appropriate root of weight
    // individual kernels by usual reference.
    // gradients done at this level.
    // inner normalisation here with appropriate square-root

    int yyyphim(int m, Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int allowfinite, int xdim, int xconsist, int assumreal) const;
    int yyyphim(int m, Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int allowfinite, int xdim, int xconsist, int assumreal) const;

    template <class T> int yyyaaphim(int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int xaignorefarfar, int xaignorefarfarfar, int xagradordadd, int xagradordaddR, int i, int allowfinite, int xdim, int xconsist, int assumreal) const;
    template <class T> int yyyaphim(int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int xaignorefarfar, int xaignorefarfarfar, int xagradordadd, int xagradordaddR, int i, int allowfinite, int xdim, int xconsist, int assumreal) const;
    template <class T> int yyybphim(int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int xaignorefarfar, int xaignorefarfarfar, int i, int xagradOrder, int xagradOrderR, int iaupm, int xafarpresent, int xafarfarpresent, int xafarfarfarpresent, int xagradup, int xagradupR, int allowfinite, int xdim, int xconsist, int assumreal, int iaset) const;
    template <class T> int yyycphim(int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int xagradOrder, int iaupm, int xagradup, const SparseVector<gentype> &xaff, int allowfinite, int xdim, int xconsist, int assumreal, int iaset) const;
    template <class T> int yyyPphim(int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int allowfinite, int xdim, int xconsist, int assumreal, int iaset, int gradOrder, int xagradup, int iaupm) const;
    //template <class T> int xPPphim (int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int allowfinite, int xdim, int xconsist, int assumreal, int iaset, int gradOrder, int xagradup) const;
    //template <class T> int xPphim  (int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int allowfinite, int xdim, int xconsist, int assumreal, int iaset, int gradOrder, int xagradup, int indstart, int indend, int ns) const;
    template <class T> int Pphim   (int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int allowfinite, int xdim, int xconsist, int assumreal, int iaset, int gradOrder, int xagradup, int indstart, int indend, int xaskip = 0) const;
    template <class T> int Qqhim   (int m, Vector<T> &res, const SparseVector<gentype> &x, const vecInfo &xinfo, int i, int allowfinite, int xdim, int xconsist, int assumreal,            int gradOrder, int xagradup, int indstart, int indend) const;






    // Analogous trees for derivative evaluation

    template <class T> void yyydKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyyd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyydnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;

    template <class T> void yyyadKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyyad2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyyadnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;

    template <class T> void yyybdKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyybd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyybdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;

    template <class T> void yyycdKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyycd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyycdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;

    template <class T> void qqqdK2delx(T &xscaleres, T &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal) const;
    template <class T> void qqqd2K2delxdelx(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDerive, int iaset, int ibset, int assumreal) const;
    template <class T> void qqqd2K2delxdely(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDerive, int iaset, int ibset, int assumreal) const;
    template <class T> void qqqdnK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDerive, int iaset, int ibset, int assumreal) const;

    template <class T> void xdKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void xd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void xdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;

    template <class T> void dKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;
    template <class T> void d2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;
    template <class T> void dnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;

    template <class T> void dLL2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void d2LL2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void dnLL2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;




    template <class T> int  KKpro(  T &res, const T &xyprod, const T &diffis, int *i, int locindstart, int locindend, int xdim, int m, T &logres, const T *xprod) const;
    template <class T> void dKKpro( T &xygrad, T &xnormgrad, T &res, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, const T &xxprod, const T &yyprod) const;
    template <class T> void d2KKpro(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, T &res, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, const T &xxprod, const T &yyprod) const;
    template <class T> void dnKKpro(T &res, const Vector<int> &gd, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, int isfirstcalc, T &scratch) const;


    template <class T> int  KKprosingle(  T &res, const T &xyprod, const T &diffis, int *i, int xdim, int m, T &logres, const T *xprod, int ktype, int &logresvalid, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;
    template <class T> void dKKprosingle( T &xygrad, T &diffgrad, T &xnormonlygrad, T &res, const T &xyprod, const T &diffis, int i, int j, int xdim, int m, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;
    template <class T> void d2KKprosingle(T &xygrad, T &diffgrad, T &xnormonlygrad, T &xyxygrad, T &diffdiffgrad, T &xnormxnormonlygrad, T &xnormynormonlygrad, T &ynormynormonlygrad, T &res, const T &xyprod, const T &diffis, int i, int j, int xdim, int m, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;
//    template <class T> void dnKKprosingle(T &res, const Vector<int> &gd, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, int isfirstcalc, T &scratch, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;

    template <class T> int  KKprosinglediffiszero(T &res, const T &xyprod, int ia, int ib, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic) const;



    template <class T> int QQpro      (int m, Vector<T> &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int allowfinite, int xdim, int xconsist, int assumreal, int gradOrder, int xagradup, int indstart, int indend) const;
    template <class T> int QQprosingle(int m, Vector<T> &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int allowfinite, int xdim, int xconsist, int assumreal, int gradOrder, int xagradup, int ind, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic) const;


    void K0i(     gentype &res,        const gentype &xyprod,                                    int xdim, int resmode, int mlid, int indstart, int indend) const;
    void K0(      gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, int xdim, int resmode, int mlid) const;
    void K0unnorm(gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, int xdim, int resmode, int mlid) const;


    void K2i(     gentype &res,        const gentype &xyprod, const gentype &yxprod,                                    const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int adensetype, int bdensetype, int resmode, int mlid, int indstart, int indend, int assumreal) const;
    void K2(      gentype &res, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int adensetype, int bdensetype, int resmode, int mlid) const;
    void K2unnorm(gentype &res, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int adensetype, int bdensetype, int resmode, int mlid) const;


    void K4i(     gentype &res,        const gentype &xyprod,                                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int i, int j, int k, int l, int xdim, int resmode, int mlid, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s, int indstart, int indend, int assumreal) const;
    void K4(      gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int i, int j, int k, int l, int xdim, int resmode, int mlid, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const;
    void K4unnorm(gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int i, int j, int k, int l, int xdim, int resmode, int mlid, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const;


    void Kmi(     gentype &res,        const gentype &xyprod,                                    Vector<const vecInfo *> &xinfo, Vector<const gentype *> &xnorm, Vector<const SparseVector<gentype> *> &x, Vector<int> &i, int xdim, int m, int resmode, int mlid, const Matrix<double> &xy, const Vector<int> *s, int indstart, int indend, int assumreal) const;
    void Km(      gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, Vector<const vecInfo *> &xinfo, Vector<const gentype *> &xnorm, Vector<const SparseVector<gentype> *> &x, Vector<int> &i, int xdim, int m, int resmode, int mlid, const Matrix<double> &xy, const Vector<int> *s) const;
    void Kmunnorm(gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, Vector<const vecInfo *> &xinfo, Vector<const gentype *> &xnorm, Vector<const SparseVector<gentype> *> &x, Vector<int> &i, int xdim, int m, int resmode, int mlid, const Matrix<double> &xy, const Vector<int> *s) const;


    void Kbase(gentype &res, int q, int typeis,
               const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
               Vector<const SparseVector<gentype> *> &x,
               Vector<const vecInfo *> &xinfo,
               Vector<const gentype *> &xnorm,
               Vector<int> &i,
               int xdim, int m, int adensetype, int bdensetype, int resmode, int mlid) const;

    // Kernel normalisation constants (altdiff 2,3)

    double AltDiffNormConst(int xdim, int m, double gamma) const
    {
        return ( ( m == 0 ) || ( m == 2 ) || ( !xdim ) || ( isAltDiff() != 2 ) ) ? 1 : (pow(2.0/m,xdim/2.0)*pow(2.0/(NUMBASE_PI*gamma*gamma),(xdim/2.0)*((m/2.0)-1)));
    }

    // Derivative tree - first order
    //
    // in dKmdx: x is the first argument, "y" is the elementwise product of all other arguments

    void dKdaz(gentype &resda, gentype &resdz, int &minmaxind, const gentype &xyprod, const gentype &yxprod, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid, int assumreal) const;

    void dKda(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;
    void dKdz(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;

    void dKunnormda(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;
    void dKunnormdz(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;


    void dKdaBase(gentype &res, int &minmaxind, int q, 
                  const gentype &xyprod, const gentype &yxprod, const gentype &diffis, 
                  Vector<const SparseVector<gentype> *> &x,
                  Vector<const vecInfo *> &xinfo,
                  Vector<const gentype *> &xnorm,
                  Vector<int> &i,
                  int xdim, int m, int mlid) const;

    void dKdzBase(gentype &res, int &minmaxind, int q, 
                  const gentype &xyprod, const gentype &yxprod, const gentype &diffis, 
                  Vector<const SparseVector<gentype> *> &x,
                  Vector<const vecInfo *> &xinfo,
                  Vector<const gentype *> &xnorm,
                  Vector<int> &i,
                  int xdim, int m, int mlid) const;


    // Kernel fall-through for 800-series kernels

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                   const SparseVector<gentype> &xa,
                   const vecInfo &xainfo,
                   int ia,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                   int ia, int ib, int ic,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                   int ia, int ib, int ic, int id,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                   Vector<const SparseVector<gentype> *> &x,
                   Vector<const vecInfo *> &xinfo,
                   Vector<int> &i,
                   int xdim, int m, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   double xyprod, double yxprod, double diffis,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   double xyprod, double yxprod, double diffis,
                   const SparseVector<gentype> &xa,
                   const vecInfo &xainfo,
                   int ia,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   double xyprod, double yxprod, double diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   double xyprod, double yxprod, double diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                   int ia, int ib, int ic,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   double xyprod, double yxprod, double diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                   int ia, int ib, int ic, int id,
                   int xdim, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   double xyprod, double yxprod, double diffis,
                   Vector<const SparseVector<gentype> *> &x,
                   Vector<const vecInfo *> &xinfo,
                   Vector<int> &i,
                   int xdim, int m, int resmode, int mlid) const;

    void dkernel8xx(int q, gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                   const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int resmode, int mlid) const;

    void dkernel8xx(int q, double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                   double xyprod, double yxprod, double diffis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int resmode, int mlid) const;




    // Final destination inner products, where redirection occurs from if
    // redirction is turned on.
    //
    // inding: 0 - not indexed
    //         1 - indexed
    // conj: 0 - no conj
    //       1 - normal conj operation
    //       2 - reversed conj
    // scaling: 0 - no scale
    //          1 - left scale
    //          2 - right scale
    //          3 - left/right scale
    //
    // Return value: 0 if result is not an equation or distribution (uses isValEqn)
    //               nz otherwise (if res has type != gentype then result *not* set)

    int innerProductDiverted       (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;
    int innerProductDivertedRevConj(gentype &res, const gentype &xyres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;

    int innerProductDiverted       (double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;
    int innerProductDivertedRevConj(double &res, double xyres, const SparseVector<gentype> &, const SparseVector<gentype> &, int xconsist = 0, int assumreal = 0) const { (void) xconsist; (void) assumreal; res = xyres; return 0; }

    int oneProductDiverted  (       gentype &res, const SparseVector<gentype> &a, int xconsist = 0, int assumreal = 0) const;
    int twoProductDiverted  (       gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;
    int threeProductDiverted(       gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, int xconsist = 0, int assumreal = 0) const;
    int fourProductDiverted (       gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d, int xconsist = 0, int assumreal = 0) const;
    int mProductDiverted    (int m, gentype &res, const Vector<const SparseVector<gentype> *> &a, int xconsist = 0, int assumreal = 0) const;

    int oneProductDiverted  (       double &res, const SparseVector<gentype> &a, int xconsist = 0, int assumreal = 0) const;
    int twoProductDiverted  (       double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;
    int threeProductDiverted(       double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, int xconsist = 0, int assumreal = 0) const;
    int fourProductDiverted (       double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d, int xconsist = 0, int assumreal = 0) const;
    int mProductDiverted    (int m, double &res, const Vector<const SparseVector<gentype> *> &a, int xconsist = 0, int assumreal = 0) const;

    // Further in, deeper down

    void getOneProd  (gentype &res, const SparseVector<gentype> &xa, int inding, int scaling, int xconsist, int assumreal) const;
    void getTwoProd  (gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, int inding, int conj, int scaling, int xconsist, int assumreal) const;
    void getThreeProd(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, int inding, int scaling, int xconsist, int assumreal) const;
    void getFourProd (gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int inding, int scaling, int xconsist, int assumreal) const;
    void getmProd    (gentype &res, const Vector<const SparseVector<gentype> *> &x, int inding, int scaling, int xconsist, int assumreal) const;

    double getOneProd  (const SparseVector<gentype> &xa, int inding, int scaling, int xconsist, int assumreal) const;
    double getTwoProd  (const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, int inding, int conj, int scaling, int xconsist, int assumreal) const;
    double getThreeProd(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, int inding, int scaling, int xconsist, int assumreal) const;
    double getFourProd (const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int inding, int scaling, int xconsist, int assumreal) const;
    double getmProd    (const Vector<const SparseVector<gentype> *> &x, int inding, int scaling, int xconsist, int assumreal) const;

    void fixShiftProd(void);

    SparseVector<gentype> &preShiftScale(SparseVector<gentype> &res, const SparseVector<gentype> &x) const;

    // Note: diff0norm and diff1norm evaluate to zero in all cases, so we bypass them here
    // Note: because xyprod etc can be infinite for sets, need to take in ia,ib,... and set res = 0 if ia == ib == ic ... for diff kernels to work

    void diff0norm(gentype &res,                                 const gentype &xyprod) const { (void) xyprod; res = 0; }
    void diff1norm(gentype &res, int ia,                         const gentype &xyprod, const gentype &xanorm) const { (void) ia; (void) xyprod; (void) xanorm; res = 0; }
    void diff2norm(gentype &res, int ia, int ib,                 const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb) const;
    void diff3norm(gentype &res, int ia, int ib, int ic,         const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s = nullptr) const;
    void diff4norm(gentype &res, int ia, int ib, int ic, int id, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s = nullptr) const;
    void diffmnorm(int m, gentype &res, const Vector<int> &i, const gentype &xyprod, const Vector<const gentype *> &xanorm, const Matrix<double> &xy, const Vector<int> *s = nullptr) const;

    void diff0norm(double &res,                                 double xyprod) const { (void) xyprod; res = 0; }
    void diff1norm(double &res, int ia,                         double xyprod, double xanorm) const { (void) ia; (void) xyprod; (void) xanorm; res = 0; }
    void diff2norm(double &res, int ia, int ib,                 double xyprod, double xanorm, double xbnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb) const;
    void diff3norm(double &res, int ia, int ib, int ic,         double xyprod, double xanorm, double xbnorm, double xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s = nullptr) const;
    void diff4norm(double &res, int ia, int ib, int ic, int id, double xyprod, double xanorm, double xbnorm, double xcnorm, double xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s = nullptr) const;
    void diffmnorm(int m, double &res, const Vector<int> &i, double xyprod, const Vector<const double *> &xanorm, const Matrix<double> &xy, const Vector<int> *s = nullptr) const;

    void diff0norm(gentype &res,                                 double xyprod) const { diff0norm(res.force_double(),xyprod); }
    void diff1norm(gentype &res, int ia,                         double xyprod, double xanorm) const { diff1norm(res.force_double(),ia,xyprod,xanorm); }
    void diff2norm(gentype &res, int ia, int ib,                 double xyprod, double xanorm, double xbnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb) const { diff2norm(res.force_double(),ia,ib,xyprod,xanorm,xbnorm,xa,xb); }
    void diff3norm(gentype &res, int ia, int ib, int ic,         double xyprod, double xanorm, double xbnorm, double xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s = nullptr) const { diff3norm(res.force_double(),ia,ib,ic,xyprod,xanorm,xbnorm,xcnorm,xy00,xy10,xy11,xy20,xy21,xy22,s); }
    void diff4norm(gentype &res, int ia, int ib, int ic, int id, double xyprod, double xanorm, double xbnorm, double xcnorm, double xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s = nullptr) const { diff4norm(res.force_double(),ia,ib,ic,id,xyprod,xanorm,xbnorm,xcnorm,xdnorm,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s); }
    void diffmnorm(int m, gentype &res, const Vector<int> &i, double xyprod, const Vector<const double *> &xanorm, const Matrix<double> &xy, const Vector<int> *s = nullptr) const { diffmnorm(m,res.force_double(),i,xyprod,xanorm,xy,s); }

    // If optionCache set then dereferences this, otherwise resizes altres to m*m and fills it will <x,y> products.

    void fillXYMatrix(double &altxyr00, double &altxyr10, double &altxyr11, double &altxyr20, double &altxyr21, double &altxyr22, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int doanyhow = 0, int assumreal = 0) const;
    void fillXYMatrix(double &altxyr00, double &altxyr10, double &altxyr11, double &altxyr20, double &altxyr21, double &altxyr22, double &altxyr30, double &altxyr31, double &altxyr32, double &altxyr33, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int doanyhow = 0, int assumreal = 0) const;

    const Matrix<double> &fillXYMatrix(int m, Matrix<double> &altres, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const Matrix<double> *optionCache = nullptr, int doanyhow = 0, int assumreal = 0) const;

    // Function to overwrite constants (integer and real) from source vectors

    Vector<int> combinedOverwriteSrc;

    mutable int backupisind; // don't worry that this is not initialised - it is
                             // always set when required.  Also no need to save this.
    mutable Vector<int> backupdIndexes;

    void processOverwrites(int q, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const;
    void fixcombinedOverwriteSrc(void);
    void addinOverwriteInd(const SparseVector<gentype> &v) const;
    void addinOverwriteInd(const SparseVector<gentype> &x, const SparseVector<gentype> &y) const;
    void addinOverwriteInd(const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x) const;
    void addinOverwriteInd(const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const;
    void addinOverwriteInd(const Vector<const SparseVector<gentype> *> &a) const;
//    void addinOverwriteInd(const Vector<gentype> &x, const Vector<gentype> &y) const;
//    void addinOverwriteInd(const SparseVector<double> &x, const SparseVector<double> &y) const;
//    void addinOverwriteInd(const Vector<double> &x, const Vector<double> &y) const;
    void addinOverwriteInd(void) const;
    void removeOverwriteInd(void) const;

    int arexysimple(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd) const
    {
        if ( !arexysimple(xa,xb) )
        {
            return 0;
        }

        if ( !arexysimple(xa,xc) )
        {
            return 0;
        }

        if ( !arexysimple(xa,xd) )
        {
            return 0;
        }

        return 1;
    }

    int arexysimple(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc) const
    {
        if ( !arexysimple(xa,xb) )
        {
            return 0;
        }

        if ( !arexysimple(xa,xc) )
        {
            return 0;
        }

        return 1;
    }

    int arexysimple(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb) const
    {
        if ( ( xa.nindsize() == 0 ) && ( xb.nindsize() == 0 ) )
        {
            return 1;
        }

        if ( ( xa.nindsize() == 1 ) && ( xb.nindsize() == 1 ) )
        {
            if ( xa.ind(0) == xb.ind(0) )
            {
                return 1;
            }
        }

        return 0;
    }

    int arexysimple(const SparseVector<gentype> &x) const
    {
        if ( ( x.nindsize() == 0 ) || ( x.nindsize() == 1 ) )
        {
             return 1;
        }

        return 0;
    }

    int arexysimple(int m, const Vector<const SparseVector<gentype> *> &x) const
    {
        NiceAssert( m <= x.size() );

        if ( m > 1 )
        {
            int i = 0;

            for ( i = 1 ; i < m ; ++i )
            {
                if ( !arexysimple(*(x(0)),*(x(i))) )
                {
                    return 0;
                }
            }
        }

        return 1;
    }

    // The only relevant part of indres is the index vector

    void combind(SparseVector<gentype> &indres, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const
    {
        indres.resize(0);

        if ( x.nindsize() )
        {
            int i;

            for ( i = 0 ; i < x.nindsize() ; ++i )
            {
                indres("&",x.ind(i)) = x.direcref(i);
            }
        }

        if ( y.nindsize() )
        {
            int i;

            for ( i = 0 ; i < y.nindsize() ; ++i )
            {
                indres("&",y.ind(i)) = y.direcref(i);
            }
        }
    }

    void combind(int m, SparseVector<gentype> &indres, const Vector<const SparseVector<gentype> *> &x) const
    {
        indres.resize(0);

        int i,j;

        if ( m )
        {
            for ( j = 0 ; j < m ; ++j )
            {
                if ( (*(x(j))).nindsize() )
                {
                    for ( i = 0 ; i < (*(x(j))).nindsize() ; ++i )
                    {
                        indres("&",(*(x(j))).ind(i)) = (*(x(j))).direcref(i);
                    }
                }
            }
        }
    }

    // Given the result of kernel evaluation, this function will attempt
    // to calculate xyprod and yxprod.  Returns 0 on success, nz on fail.

    int reverseEngK(gentype &res, const vecInfo &xinfo, const vecInfo &yinfo, const SparseVector<gentype> &x, const SparseVector<gentype> &y, double Kres) const;
    int reverseEngK(gentype &res, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, double Kres) const;
    int reverseEngK(int m, gentype &res, const Vector<const vecInfo *> &xinfo, const Vector<const SparseVector<gentype> *> &x, double Kres) const;

    int isKernelDerivativeEasy(void) const
    {
//        return !isProd() &&
//               !isIndex() &&
//               !isShifted() &&
//               !isScaled() &&
//               ( size() == 1 ) &&
//               !isNormalised(0) &&
//               ( kinf(0).numflagsset() <= 1 ) &&
//               ( ( kinf(0).numflagsset() == 0 ) || kinf(0).usesDiff || kinf(0).usesInner || kinf(0).usesMinDiff || kinf(0).usesMaxDiff );
        return !isProd() &&
               ( size() == 1 ) &&
               !isNormalised(0) &&
               ( kinf(0).numflagsset() <= 1 ) &&
               ( ( kinf(0).numflagsset() == 0 ) || kinf(0).usesDiff || kinf(0).usesInner || kinf(0).usesMinDiff || kinf(0).usesMaxDiff );
    }

    // Function to tell us if kerrnel index actually exists

    int iskern(int potind) const;

    // This function returns true if, despite redirected inner products, the
    // kernel will still require the actual vectors themselves.  Note that
    // evaluation may still throw an exception if overwrites must be processed
    // or pre-processing on vector basis is requested.

    int needExplicitVector(void) const
    {
        return kinf(0).usesVector || isProd();
    }

    // combine minmaxadd

    int combineminmaxind(int aminmaxind, int bminmaxind) const
    {
        if ( aminmaxind == -2 )
        {
            return ( bminmaxind == -2 ) ? -1 : bminmaxind;
        }

        if ( bminmaxind == -2 )
        {
            return ( aminmaxind == -2 ) ? -1 : aminmaxind;
        }

        if ( ( aminmaxind == -1 ) && ( bminmaxind == -1 ) )
        {
            return -1;
        }

        NiceThrow("Incompatible gradient indices error");

        return -3;
    }





    void fixhaslinconstr()
    {
        if ( sizeLinConstr() )
        {
            haslinconstr = false;

            int q;

            for ( q = 0 ; q < sizeLinConstr() ; ++q )
            {
                if ( !linGradOrd(q) )
                {
                    haslinconstr = true;
                    break;
                }
            }
        }

        else
        {
            haslinconstr = true;
        }

        return;
    }

    void fixlingradscaltsp(int i = -1)
    {
        int q;

        if ( i == -1 )
        {
            linGradScalTsp.resize(linGradScal.size());
        }

        for ( q = ( ( i >= 0 ) ? i : 0 ) ; q < ( ( i >= 0 ) ? (i+1) : (linGradScal.size()) ) ; ++q )
        {
            linGradScalTsp("&",q) = linGradScal(q);
            linGradScalTsp("&",q).transpose();
        }

        return;
    }







    // xisfast: -1 if unknown
    //          0  if kernel is not fast and full calculation is required
    //          1  if completely unchained kernel where all kernels are either inner-product or diff
    //          2  if completely chained kernel where kernel 0 is either inner-product or diff, and remaining kernels are inner-product (no splits allowed or magterms)
    //          3  if completely chained kernel where kernel 0 is kernel transfer, kernel 1 is either inner-product or diff, and remaining kernels are inner-product (no splits or magterms allowed)
    // xneedsInner: needs inner product to calculate
    // xneedsDiff:  needs inner product to calculate
    // xneedsNorm:  needs norms of vectors to calculate

    mutable int xisfast;
    mutable int xneedsInner;
    mutable int xneedsInnerm2;
    mutable int xneedsDiff;
    mutable int xneedsNorm;

    // needsInner:   returns 1 if inner (m) product is required in this kernel (-1 for all parts)
    // needsDiff:    returns 1 if diff  (m) product is required in this kernel (-1 for all parts)
    // needsMatDiff: returns 1 if diff  (m) product is required and matrix accelerated works
    //               returns 2 if diff  (m) product is required, matrix accel works, but we can get away with only pairwise parts.
    // needsNorm:    returns 1 if ubber (m) norm(s) is(are) required in this kernel (-1 for all parts)

    int needsMatDiff(int q = -1) const
    {
        if ( needsDiff(q) && isAltDiff() == 5 )
        {
            return -1;
        }

        return needsDiff(q) && ( ( isAltDiff() == 2   ) ||
                                 ( isAltDiff() == 102 ) || ( isAltDiff() == 103 ) || ( isAltDiff() == 104 ) ||
                                 ( isAltDiff() == 202 ) || ( isAltDiff() == 203 ) || ( isAltDiff() == 204 ) ||
                                 ( isAltDiff() == 300 )    );
    }

//ADDHERE
    int isFastKernelType(int ind) const
    {
        return ( ( cType(ind) == 0   ) ||
                 ( cType(ind) == 1   ) ||
                 ( cType(ind) == 2   ) ||
                 ( cType(ind) == 3   ) ||
                 ( cType(ind) == 4   ) ||
                 ( cType(ind) == 5   ) ||
                 ( cType(ind) == 7   ) ||
                 ( cType(ind) == 8   ) ||
                 ( cType(ind) == 9   ) ||
                 ( cType(ind) == 10  ) ||
                 ( cType(ind) == 11  ) ||
                 ( cType(ind) == 12  ) ||
                 ( cType(ind) == 13  ) ||
                 ( cType(ind) == 14  ) ||
                 ( cType(ind) == 15  ) ||
                 ( cType(ind) == 19  ) ||
                 ( cType(ind) == 23  ) ||
                 ( cType(ind) == 24  ) ||
                 ( cType(ind) == 25  ) ||
                 ( cType(ind) == 26  ) ||
                 ( cType(ind) == 27  ) ||
                 ( cType(ind) == 32  ) ||
                 ( cType(ind) == 33  ) ||
                 ( cType(ind) == 34  ) ||
                 ( cType(ind) == 38  ) ||
                 ( cType(ind) == 39  ) ||
                 ( cType(ind) == 42  ) ||
                 ( cType(ind) == 43  ) ||
                 ( cType(ind) == 44  ) ||
                 ( cType(ind) == 45  ) ||
                 ( cType(ind) == 46  ) ||
                 ( cType(ind) == 47  ) ||
                 ( cType(ind) == 49  ) ||
                 ( cType(ind) == 50  ) ||
                 ( cType(ind) == 51  ) ||
                 ( cType(ind) == 52  ) ||
                 ( cType(ind) == 53  ) ||
                 ( cType(ind) == 100 ) ||
                 ( cType(ind) == 103 ) ||
                 ( cType(ind) == 104 ) ||
                 ( cType(ind) == 106 ) ||
                 ( cType(ind) == 200 ) ||
                 ( cType(ind) == 203 ) ||
                 ( cType(ind) == 204 ) ||
                 ( cType(ind) == 206 )    );
    }

    int isfast(void) const
    {
        int res = xisfast;

        if ( xisfast == -1 )
        {
             res = isfastunsafe();
        }

        return res;
    }

    int needsInner(int q = -1, int m = 2) const
    {
        NiceAssert( q < size() );
        NiceAssert( q >= -1 );

        int res = ( m == 2 ) ? xneedsInnerm2 : xneedsInner;

        if ( q >= 0 )
        {
            res = kinf(q).usesInner || ( kinf(q).usesDiff && ( ( m == 2 ) || ( isAltDiff() <= 1 ) ) );
        }

        else if ( res == -1 )
        {
            res = needsInnerunsafe(m);
        }

        return res;
    }

    int needsDiff(int q = -1) const
    {
        NiceAssert( q < size() );
        NiceAssert( q >= -1 );

        int res = xneedsDiff;

        if ( q >= 0 )
        {
            res = kinf(q).usesDiff;
        }

        else if ( xneedsDiff == -1 )
        {
            res = needsDiffunsafe();
        }

        return res;
    }

    int needsNorm(int q = -1) const
    {
        NiceAssert( q < size() );
        NiceAssert( q >= -1 );

        int res = xneedsNorm;

        if ( q >= 0 )
        {
            res = needsDiff(q) || ( needsInner(q) && isMagTerm(q) );
        }

        else if ( xneedsNorm == -1 )
        {
            res = needsNormunsafe();
        }

        return res;
    }

    int isfastunsafe(void) const
    {
        // xisfast: -1 if unknown

        if ( xisfast == -1 )
        {
          //static svm_mutex eyelock; - assume mercer object is single-thread only
          //svm_mutex_lock(eyelock);

          if ( xisfast == -1 )
          {
            xisfast = 0;

            //          0  if kernel is not fast and full calculation is required

            retVector<int> tmpva;

//errstream() << "phantomxyzxyz 42: numSplits() = " << numSplits() << "\n";
//errstream() << "phantomxyzxyz 42: numMulSplits() = " << numMulSplits() << "\n";
//errstream() << "phantomxyzxyz 42: size() = " << size() << "\n";
//errstream() << "phantomxyzxyz 42: isnorm(size()-1) = " << isnorm(size()-1) << "\n";
//errstream() << "phantomxyzxyz 42: isAltDiff() = " << isAltDiff() << "\n";
//errstream() << "phantomxyzxyz 42: ischain(0,1,size()-2) = " << ischain(0,1,size()-2,tmpva) << "\n";
//errstream() << "phantomxyzxyz 42: cType(0) = " << cType(0) << "\n";
//errstream() << "phantomxyzxyz 42: big clause = " << ( ( !numSplits() && !numMulSplits() && 
//                      ( ( size() >= 1 ) && ( ( isnorm(size()-1) == 0 ) || ( isAltDiff() <= 99 ) ) 
//                                        && (    ( ( size() == 1 ) || ( ischain(0,1,size()-2,tmpva) == 1 ) ) 
//                                             && (    ( ( cType(0) >= 800 ) && ( cType(0) <= 829 ) ) ) ) ) ) ? 1 : 0 ) << "\n";

            if ( ( size() <= 1 ) || ( isnorm(0,1,size()-2,tmpva) == 0 ) )
            {
                if ( ( size() >= 1 ) && ( isnorm(size()-1) == 0 ) && ( ischain(0,1,size()-2,tmpva) == 0 ) )
                {
                    // Could be xisfast == 1
                    //          1  if completely unchained kernel where all kernels are either inner-product or diff

                    xisfast = 1;

                    int i;

                    for ( i = 0 ; i < size() ; ++i )
                    {
                        if ( !isFastKernelType(i) )
                        {
                            xisfast = 0;
                            break;
                        }
                    }
                }

                // Important: can't use else if here, as might be xisfast == 2

                if ( ( xisfast == 0 ) && !numSplits() && !numMulSplits() && 
                      ( ( size() >= 1 ) && ( ( isnorm(size()-1) == 0 ) || ( isAltDiff() <= 99 ) ) 
                                        && (    ( ( size() == 1 ) || ( ischain(0,1,size()-2,tmpva) == 1 ) ) 
                                             && ( cType(0) < 800 ) ) ) )
                {
                    // Could be xisfast == 2
                    //          2  if completely chained kernel where kernel 0 is either inner-product or diff, and remaining kernels are inner-product (no splits allowed or magterms)

                    xisfast = 2;

                    int i;

                    for ( i = 0 ; i < size() ; ++i )
                    {
                        if ( !isFastKernelType(i) || ( i && needsDiff(i) ) || isMagTerm(i) )
                        {
                            xisfast = 0;
                            break;
                        }
                    }
                }

                // Important: can't use else if here, as might be xisfast == 3

                if ( ( xisfast == 0 ) && !numSplits() && !numMulSplits() && 
                      ( ( size() >= 1 ) && ( ( isnorm(size()-1) == 0 ) || ( isAltDiff() <= 99 ) ) 
                                        && (    ( ( size() == 1 ) || ( ischain(0,1,size()-2,tmpva) == 1 ) ) 
                                             && (    ( ( cType(0) >= 800 ) && ( cType(0) <= 829 ) ) ) ) ) )
                {
                    // Could be xisfast == 3
                    //          3  if completely chained kernel where kernel 0 is kernel transfer not requiring inner product, norm or diff; kernel 1 is either inner-product or diff; and remaining kernels are inner-product (no splits or magterms allowed)

                    xisfast = 3;

                    int i;

                    for ( i = 0 ; i < size() ; ++i )
                    {
                        if ( ( i && ( !isFastKernelType(i) || ( ( i > 1 ) && needsDiff(i) ) ) ) || isMagTerm(i) )
                        {
                            xisfast = 0;
                            break;
                        }
                    }
                }
            }
          }

          //svm_mutex_unlock(eyelock);
        }

        return xisfast;
    }

    int needsInnerunsafe(int m) const
    {
        if ( ( xneedsInner == -1 ) || ( xneedsInnerm2 == -1 ) )
        {
            //static svm_mutex eyelock;
            //svm_mutex_lock(eyelock);

            int usesInner = 0;
            int usesDiff  = 0;

            if ( ( xneedsInner == -1 ) || ( xneedsInnerm2 == -1 ) )
            {
                if ( size() )
                {
                    int q;

                    for ( q = 0 ; q < size() ; ++q )
                    {
                        usesInner |= kinf(q).usesInner;
                        usesDiff  |= kinf(q).usesDiff;
                    }
                }

                if ( xneedsInner == -1 )
                {
                    xneedsInner = ( usesInner || ( usesDiff && ( isAltDiff() <= 1 ) ) ) ? 1 : 0;
                }

                if ( xneedsInnerm2 == -1 )
                {
                    xneedsInnerm2 = ( usesInner || usesDiff ) ? 1 : 0;
                }
            }

            //svm_mutex_unlock(eyelock);
        }

        return ( m == 2 ) ? xneedsInnerm2 : xneedsInner;
    }

    int needsDiffunsafe(void) const
    {
        if ( xneedsDiff == -1 )
        {
            //static svm_mutex eyelock;
            //svm_mutex_lock(eyelock);

            if ( xneedsDiff == -1 )
            {
                int usesDiff  = 0;

                if ( size() )
                {
                    int q;

                    for ( q = 0 ; q < size() ; ++q )
                    {
                        usesDiff  |= kinf(q).usesDiff;
                    }
                }

                xneedsDiff = usesDiff ? 1 : 0;
            }

            //svm_mutex_unlock(eyelock);
        }

        return xneedsDiff;
    }

    int needsNormunsafe(void) const
    {
        if ( xneedsNorm == -1 )
        {
            //static svm_mutex eyelock;
            //svm_mutex_lock(eyelock);

            if ( xneedsNorm == -1 )
            {
                int usesNorm = 0;

                if ( size() )
                {
                    int q;

                    for ( q = 0 ; q < size() ; ++q )
                    {
                        usesNorm |= ( needsDiff(q) || ( needsInner(q) && isMagTerm(q) ) );
                    }
                }

                xneedsNorm = usesNorm ? 1 : 0;
            }

            //svm_mutex_unlock(eyelock);
        }

        return xneedsNorm;
    }

    // Sampling functions for distribution kernels - return 0 if nothing is changed by sampling, >0 otherwise

    int subSample(SparseVector<SparseVector<gentype> > &subval, SparseVector<gentype> &x, vecInfo &xinfo) const;
    int subSample(SparseVector<SparseVector<gentype> > &subval, gentype &b) const;
    int subSample(SparseVector<SparseVector<gentype> > &subval, double  &b) const;

    // Various short-circuited kernels
    //
    // isSimpleKernel:           size 1, no normalisation, no chaining
    // isSimpleBasicKernel:      isSimpleKernel, and type   0-99  (NN kernel)
    // isSimpleNNKernel:         isSimpleKernel, and type 100-299 (NN kernel)
    // isSimpleDistKernel:       isSimpleKernel, and type 300-399 (-ve dist kernel)
    // isSimpleXferKernel:       isSimpleKernel, and type 800-899
    // isSimpleKernelChain:      size 2, no normalisation, chained, with kernel 0 being a kernel transfer

public:
    int isSimpleKernel     (void) const { return ( ( size() == 1 ) && !isNormalised() && !isChained() && !isSplit() && !isMulSplit() && !isMagTerm() ); }
    int isSimpleBasicKernel(void) const { return ( isSimpleKernel() && ( cType() >=   0 ) && ( cType() <  100 ) ); }
    int isSimpleNNKernel   (void) const { return ( isSimpleKernel() && ( cType() >= 100 ) && ( cType() <  300 ) ); }
    int isSimpleDistKernel (void) const { return ( isSimpleKernel() && ( cType() >= 300 ) && ( cType() <  400 ) ); }
    int isSimpleXferKernel (void) const { return ( isSimpleKernel() && ( ( cType() >= 800 ) && ( cType() <= 829 ) ) ); }
    int isSimpleKernelChain(void) const { return ( ( size() == 2 )  && ( ( cType() >= 800 ) && ( cType() <= 829 ) )
                                          && !isNormalised(0) && !isNormalised(1) && isChained(0) && !isSplit(0) && !isMulSplit(0) && !isMagTerm() ); }
    int isTrivialKernel    (int allowsymm = 0) const { const static gentype tempsampdist("[ ]"); return ( ( size() == 1 ) && !isFullNorm() && ( allowsymm || !isSymmSet() ) && !isProd() && !isIndex() && !isShifted() && !isScaled() && !isLeftPlain() &&
                                          !isRightPlain() && ( isAltDiff() == 1 ) && !isNormalised() && !isChained() && !isSplit() && !isMulSplit() && !isMagTerm() &&
                                          ( numSamples() == DEFAULT_NUMKERNSAMP ) && ( sampleDistribution() == tempsampdist ) && ( sampleIndices().size() == 0 ) &&
                                          ( cRealOverwrite().indsize() == 0 ) && ( cIntOverwrite().indsize() == 0 ) ); }
    int isVeryTrivialKernel(int allowsymm = 0) const { return ( ( !xdefindKey.size() || ( xdefindKey(xdefindKey.size()-1) == xdefindKey.size()-1 ) ) && isTrivialKernel(allowsymm) && ( !sizeLinConstr() ) && ( !sizeLinParity() ) ); }

    int isFastKernelSum  (void) const { return ( isfast() == 1 ); }
    int isFastKernelChain(void) const { return ( isfast() == 2 ); } // No splits or magnitude terms allowed
    int isFastKernelXfer (void) const { return ( isfast() == 3 ); } // No splits or magnitude terms allowed

    int isSimpleLinearKernel(void) const { return ( isSimpleKernel() && ( cType() == 1 ) ); }

private:
    Vector<gentype> &local_makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a, int m) const;
    Vector<double> &local_makeanRKHSVector(Vector<double> &res, const MercerKernel &, const SparseVector<gentype> &, const gentype &, int) const;
};

inline std::ostream &operator<<(std::ostream &output, const MercerKernel &src)
{
    return src.printstream(output,0);
}

inline std::istream &operator>>(std::istream &input, MercerKernel &dest)
{
    return dest.inputstream(input);
}


inline void qswap(MercerKernel &a, MercerKernel &b)
{
    qswap(a.xdefindKey,b.xdefindKey);

    qswap(a.isprod              ,b.isprod              );
    qswap(a.isind               ,b.isind               );
    qswap(a.isfullnorm          ,b.isfullnorm          );
    qswap(a.issymmset           ,b.issymmset           );
    qswap(a.isshift             ,b.isshift             );
    qswap(a.leftplain           ,b.leftplain           );
    qswap(a.rightplain          ,b.rightplain          );
    qswap(a.isdiffalt           ,b.isdiffalt           );
    qswap(a.xproddepth          ,b.xproddepth          );
    qswap(a.enchurn             ,b.enchurn             );
    qswap(a.xsuggestXYcache     ,b.xsuggestXYcache     );
    qswap(a.xisIPdiffered       ,b.xisIPdiffered       );
    qswap(a.xnumSplits          ,b.xnumSplits          );
    qswap(a.xnumMulSplits       ,b.xnumMulSplits       );
    qswap(a.xdenseZeroPoint     ,b.xdenseZeroPoint     );

    qswap(a.dtype               ,b.dtype               );
    qswap(a.isnorm              ,b.isnorm              );
    qswap(a.ischain             ,b.ischain             );
    qswap(a.issplit             ,b.issplit             );
    qswap(a.mulsplit            ,b.mulsplit            );
    qswap(a.ismagterm           ,b.ismagterm           );
    qswap(a.xranktype           ,b.xranktype           );
    qswap(a.dIndexes            ,b.dIndexes            );
    qswap(a.kernflags           ,b.kernflags           );
    qswap(a.dRealConstants      ,b.dRealConstants      );
    qswap(a.dIntConstants       ,b.dIntConstants       );
    qswap(a.disNomConst         ,b.disNomConst         );
    qswap(a.dRealConstantsLB    ,b.dRealConstantsLB    );
    qswap(a.dIntConstantsLB     ,b.dIntConstantsLB     );
    qswap(a.dRealConstantsUB    ,b.dRealConstantsUB    );
    qswap(a.dIntConstantsUB     ,b.dIntConstantsUB     );
    qswap(a.dRealOverwrite      ,b.dRealOverwrite      );
    qswap(a.dIntOverwrite       ,b.dIntOverwrite       );
    qswap(a.altcallback         ,b.altcallback         );
    qswap(a.randFeats           ,b.randFeats           );
    qswap(a.randFeatAngle       ,b.randFeatAngle       );
    qswap(a.randFeatReOnly      ,b.randFeatReOnly      );
    qswap(a.randFeatNoAngle     ,b.randFeatNoAngle     );

    qswap(a.dShift              ,b.dShift              );
    qswap(a.dScale              ,b.dScale              );
    qswap(a.dShiftProd          ,b.dShiftProd          );
    qswap(a.dShiftProdNoConj    ,b.dShiftProdNoConj    );
    qswap(a.dShiftProdRevConj   ,b.dShiftProdRevConj   );

    qswap(a.linGradOrd          ,b.linGradOrd          );
    qswap(a.linGradScal         ,b.linGradScal         );
    qswap(a.linGradScalTsp      ,b.linGradScalTsp      );
    qswap(a.haslinconstr        ,b.haslinconstr        );

    qswap(a.linParity           ,b.linParity           );
    qswap(a.linParityOrig       ,b.linParityOrig       );

    qswap(a.xnumsamples         ,b.xnumsamples         );
    qswap(a.xindsub             ,b.xindsub             );
    qswap(a.xsampdist           ,b.xsampdist           );

    qswap(a.combinedOverwriteSrc     ,b.combinedOverwriteSrc     );
    qswap(a.backupisind              ,b.backupisind              );
    qswap(a.backupdIndexes           ,b.backupdIndexes           );

    qswap((a.xisfast)      ,(b.xisfast)      );
    qswap((a.xneedsInner)  ,(b.xneedsInner)  );
    qswap((a.xneedsInnerm2),(b.xneedsInnerm2));
    qswap((a.xneedsDiff)   ,(b.xneedsDiff)   );
    qswap((a.xneedsNorm)   ,(b.xneedsNorm)   );
}


#endif

