
//
// Sparse quadratic solver - gradient descent
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQgradesc_h
#define _sQgradesc_h

#include "sQbase.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "optstate.hpp"

//
// Uses gradient descent to solve general convex optimisation problems.  Basic 
// method is projected gradient descent.  For simplicity the inner (step 
// calculating) "loop" uses sQsLsAsWs to solve (where Gn should probably be
// zero):
//
// [ I    Gpn ] [ dalpha ] + [ e ] = [ 0 ]
// [ Gpn' Gn  ] [ dbeta  ]   [ 0 ]   [ 0 ]
//
// Notes:
//
// - stepscalefactor is replaced by lr, which only affects outer loop
// - higherorderterms (fixOptState) function should be set up update optState.
// - usels:
//   bit 1: controls whether line-search is used (1 for line-search)
//   bit 2: gradient descent (0) or Netwon (1)
// - GpnRowTwoMag does not work here.
//



class fullOptStateGradDesc : public fullOptState<double,double>
{
public:

    fullOptStateGradDesc(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = DEFAULTOUTERSTEPSCALE,
        double _lrback = DEFAULTOUTERSTEPBACK, double _delta = DEFAULTOUTERDELTA, int _usels = DEFAULTUSELS)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        lr     = _stepscalefactor;
        lrback = _lrback;
        delta  = _delta;
        usels  = _usels;

        if ( !fixHigherOrderTerms )
        {
            fixHigherOrderTerms =  fullfixbasea;
        }

        return;
    }

    fullOptStateGradDesc(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = DEFAULTOUTERSTEPSCALE,
        double _lrback = DEFAULTOUTERSTEPBACK, double _delta = DEFAULTOUTERDELTA, int _usels = DEFAULTUSELS)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        lr     = _stepscalefactor;
        lrback = _lrback;
        delta  = _delta;
        usels  = _usels;

        if ( !fixHigherOrderTerms )
        {
            fixHigherOrderTerms =  fullfixbasea;
        }

        return;
    }

    fullOptStateGradDesc(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_qGpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = DEFAULTOUTERSTEPSCALE,
        double _lrback = DEFAULTOUTERSTEPBACK, double _delta = DEFAULTOUTERDELTA, int _usels = DEFAULTUSELS)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_qGpnRowTwoMag,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        lr     = _stepscalefactor;
        lrback = _lrback;
        delta  = _delta;
        usels  = _usels;

        if ( !fixHigherOrderTerms )
        {
            fixHigherOrderTerms =  fullfixbasea;
        }

        return;
    }

    virtual ~fullOptStateGradDesc() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState<double,double> *gencopy(int _chistart,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_qGpnRowTwoMag)
    {
        fullOptStateGradDesc *res;

        MEMNEW(res,fullOptStateGradDesc(x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_qGpnRowTwoMag));

        copyvars(res,_chistart);

        return static_cast<fullOptState<double,double> *>(res);
    }

    double lr;     // learning rate
    double lrback; // lr rollback factor
    double delta;  // delta factor
    int usels;     // use line-search

//private:

    // Actual quadratic optimiser

    virtual int solve(svmvolatile int &killSwitch);

    virtual void copyvars(fullOptState<double,double> *dest, int _chistart)
    {
        fullOptState<double,double>::copyvars(dest,_chistart);

        fullOptStateGradDesc *ddest = static_cast<fullOptStateGradDesc *>(dest);

        ddest->lr     = lr;
        ddest->lrback = lrback;
        ddest->usels  = usels;

        return;
    }
};





#endif
