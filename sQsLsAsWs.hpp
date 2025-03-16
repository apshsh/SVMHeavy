
//
// Sparse quadratic solver - large scale, active set, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQsLsAsWs_h
#define _sQsLsAsWs_h

#include "sQbase.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "optstate.hpp"


class fullOptStateActive : public fullOptState<double,double>
{
public:

    fullOptStateActive(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( x.keepfact() );
        NiceAssert( !_fixHigherOrderTerms );

        linbreak = 0;

        return;
    }

    fullOptStateActive(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( x.keepfact() );
        NiceAssert( !_fixHigherOrderTerms );

        linbreak = 0;

        return;
    }

    fullOptStateActive(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_qGpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_qGpnRowTwoMag,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( x.keepfact() );
        NiceAssert( !_fixHigherOrderTerms );

        linbreak = 0;

        return;
    }

    virtual ~fullOptStateActive() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState<double,double> *gencopy(int _chistart,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_qGpnRowTwoMag)
    {
        fullOptStateActive *res;

        MEMNEW(res,fullOptStateActive(x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_qGpnRowTwoMag));

        copyvars(res,_chistart);

        res->linbreak = linbreak;

        return static_cast<fullOptState<double,double> *>(res);
    }

private:

    // Actual quadratic optimiser

    virtual int solve(svmvolatile int &killSwitch);

    virtual void copyvars(fullOptState<double,double> *dest, int _chistart)
    {
        fullOptState<double,double>::copyvars(dest,_chistart);

        return;
    }

public:
    int linbreak; // set 1 to stop if linear step occurs
};

#endif
