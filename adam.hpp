
//
// Rudimentary implementation of ADAM
//
// Version: 7
// Date: 30/04/2021
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _adam_h
#define _adam_h

#include "vector.hpp"
#include "sQbase.hpp"

// Return: 0  on success
//         -1 if timeout/too many iterations
//         >0 error in objective evaluation
//
// x:       startpoint passed in, result passed out
// calcObj: calculate the objective and it's gradient
//          You can also pass in the second derivative
//          (diagonal) here, and it will be used to
//          approximate a Newton step.  This should only
//          be used if there is something like 1/x, where
//          the curvature tends large.  The result doesn't
//          have to be exact, but should be proportional.
//          If not used set gradgradx = 1.
//          The final int * can be either ignored, or set
//          1 if the gradient returned is not reliable and
//          so should not be used for termination.
//
// usestring:  string describing what is being optimised
// killSwitch: set 1 at any time (eg in another thread) to end optimisation
// lr:         learning rate
// Opttol:     max gradient error for optimality.
//
// useadam:      0 for gradient descent, 1 for ADAM
// maxitcnt:     max iterations (default 0 for unlimited)
// maxtraintime: max training time (default 0 for unlimited)
// traintimeend: absolute train end time (relative, seconds, -1 for none)
//
// abeta1: ADAM parameter
// abeta2: ADAM parameter
// aeps:   ADAM parameter
//
// xsgn: 0  -> x = 0
//       -1 -> x < 0
//       +1 -> x > 0
//        2 -> x unconstrained
//        3 -> x >= minv
//        4 -> 0 < x <= 1/minv
// nsgn: number of elements effected by xsgn (first nsgn, -1 for all)

#define USE_ADAM              1
#define XDEFAULT_MAXITCNT     0
#define XDEFAULT_MAXTRAINTIME 0
#define XDEFAULT_TRAINTIMEEND -1

////#define ADAM_BETA1 0.7
////#define ADAM_BETA2 0.8
//#define ADAM_BETA1 0.9
//#define ADAM_BETA2 0.999
#define ADAM_BETA1 0.99
#define ADAM_BETA2 0.999
////#define ADAM_EPS   1e-8
//#define ADAM_EPS   0.5
#define ADAM_EPS   0.1
#define ZSTOL 1e-4
#define SCHEDCONST 0.3

// lr is scaled by 1/t^schedconst
// normmax > 0 means, after each step, scale to ensure that ||w||_2 <= normmax
// normmax < 0 means, after each step, scale to ensure that ||wq||_2 <= normmax, where w = [ wq ; 0 ] (ignore last element in magnitude calculation


class ADAMscratch;

inline int ADAMopt(double &objres, Vector<double> &x, int (*calcObj)(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs),
            const char *usestring, svmvolatile int &killSwitch, double lr, void *objargs, int useadam,
            const stopCond &sc, //double Opttol = DEFAULT_OPTTOL, int maxitcnt = XDEFAULT_MAXITCNT, double maxtraintime = XDEFAULT_MAXTRAINTIME, double traintimeend = XDEFAULT_TRAINTIMEEND,
            double abeta1 = ADAM_BETA1, double abeta2 = ADAM_BETA2, double aeps = ADAM_EPS, int xsgn = 1, double schedconst = SCHEDCONST, double normmax = 0, double minv = 1, int nsgn = -1,
            double (*stepscale)(const Vector<double> &x, const Vector<double> &dx, void *) = nullptr, void *stepscalearg = nullptr);
int ADAMopt(double &objres, Vector<double> &x, int (*calcObj)(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs),
            const char *usestring, svmvolatile int &killSwitch, double lr, void *objargs, ADAMscratch &scratchpad, int useadam,
            stopCond sc, //double Opttol = DEFAULT_OPTTOL, int maxitcnt = XDEFAULT_MAXITCNT, double maxtraintime = XDEFAULT_MAXTRAINTIME, double traintimeend = XDEFAULT_TRAINTIMEEND,
            double abeta1 = ADAM_BETA1, double abeta2 = ADAM_BETA2, double aeps = ADAM_EPS, int xsgn = 1, double schedconst = SCHEDCONST, double normmax = 0, double minv = 1, int nsgn = -1,
            double (*stepscale)(const Vector<double> &x, const Vector<double> &dx, void *) = nullptr, void *stepscalearg = nullptr);





class ADAMscratch
{
    public:

    ADAMscratch() { ; }
    ADAMscratch(int xdim) : gradx(xdim), gradgradx(xdim), dx(xdim), xbest(xdim), am(xdim), av(xdim), amhat(xdim), avhat(xdim), xold(xdim) { ; }

    void resize(int size)
    {
        gradx.resize(size);
        gradgradx.resize(size);
        dx.resize(size);
        xbest.resize(size);
        am.resize(size);
        av.resize(size);
        amhat.resize(size);
        avhat.resize(size);
        xold.resize(size);
    }

    Vector<double> gradx;
    Vector<double> gradgradx;
    Vector<double> dx;
    Vector<double> xbest;

    Vector<double> am;
    Vector<double> av;

    Vector<double> amhat;
    Vector<double> avhat;

    Vector<double> xold;
};

inline int ADAMopt(double &objres, Vector<double> &x, int (*calcObj)(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs),
            const char *usestring, svmvolatile int &killSwitch, double lr, void *objargs, int useadam,
            const stopCond &sc, //double Opttol, int maxitcnt, double maxtraintime, double traintimeend,
            double abeta1, double abeta2, double aeps, int xsgn, double schedconst, double normmax, double minv, int nsgn,
            double (*stepscale)(const Vector<double> &x, const Vector<double> &dx, void *), void *stepscalearg)
{
    ADAMscratch scratchpad(x.size());

    return ADAMopt(objres,x,calcObj,usestring,killSwitch,lr,objargs,scratchpad,useadam,sc,abeta1,abeta2,aeps,xsgn,schedconst,normmax,minv,nsgn,stepscale,stepscalearg);
}


#endif
