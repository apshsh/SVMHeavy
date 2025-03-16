
//
// Sparse quadratic solver - large scale, d2c based, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQd2c_h
#define _sQd2c_h

#include "sQbase.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "optstate.hpp"

//
// Pretty self explanatory: you give is a state and relevant matrices and it
// will solve:
//
// [ alpha ]' [ Gp   Gpn ] [ alpha ] + [ alpha ]' [ gp ] + | alpha' |' [ hp ]
// [ beta  ]  [ Gpn' Gn  ] [ beta  ]   [ beta  ]  [ gn ]   | beta   |  [ 0  ]
//
// to within precision optsol.  It is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gpn is a size(Gp)*1 matrix of all 1s (the only effect is in sQsmo.cc, trial_step_SMO)
// - Gn is a 1*1 zero matrix
// - GpnRowTwoSigned = 0
//
// Will return 0 on success or an error code otherwise
//



class fullOptStateD2C : public fullOptState<double,double>
{
public:

    fullOptStateD2C(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        return;
    }

    fullOptStateD2C(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        return;
    }

    fullOptStateD2C(optState<double,double> &_x,
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_qGpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = nullptr, void *_htArg = nullptr,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_qGpnRowTwoMag,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        return;
    }

    virtual ~fullOptStateD2C() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState<double,double> *gencopy(int _chistart, 
        const Matrix<double> &_qGp, const Matrix<double> &_qGpsigma, const Matrix<double> &_qGn, Matrix<double> &_qGpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_qGpnRowTwoMag)
    {
        fullOptStateD2C *res;

        MEMNEW(res,fullOptStateD2C(x,_qGp,_qGpsigma,_qGn,_qGpn,_gp,_gn,_hp,_lb,_ub,_qGpnRowTwoMag));

        copyvars(res,_chistart);

        return static_cast<fullOptState<double,double> *>(res);
    }

//private:

    // Actual quadratic optimiser

    virtual int solve(svmvolatile int &killSwitch);

    virtual void copyvars(fullOptState<double,double> *dest, int _chistart)
    {
        fullOptState<double,double>::copyvars(dest,_chistart);

        return;
    }
};





#endif
