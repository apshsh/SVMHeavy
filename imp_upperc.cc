
//
// Upper Confidence Bound
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "imp_upperc.hpp"


IMP_UpperC::IMP_UpperC(int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(nullptr);

    return;
}

IMP_UpperC::IMP_UpperC(const IMP_UpperC &src, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(nullptr);

    assign(src,0);

    return;
}

IMP_UpperC::IMP_UpperC(const IMP_UpperC &src, const ML_Base *xsrc, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,-1);

    return;
}

IMP_UpperC::~IMP_UpperC()
{
    return;
}

std::ostream &IMP_UpperC::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Upper confidence bound block\n";

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_UpperC::inputstream(std::istream &input )
{
    //wait_dummy dummy;

    return IMP_Generic::inputstream(input);
}

int IMP_UpperC::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !retaltg );

    (void) retaltg;
    (void) pxyprodi;

    int res = imp(resg,x(i),nullgentype());

    resh = resg;

    return res;
}











int IMP_UpperC::train(int &res, svmvolatile int &killSwitch)
{
    (void) res;

    NiceAssert( xspaceDim() <= 1 );

    int retval = 0;

    if ( !isTrained() )
    {
        retval = IMP_Generic::train(res,killSwitch);
    }

    return retval;
}

int IMP_UpperC::imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const
{
    NiceAssert( isTrained() );

    const gentype &muy = xxmean(0);
    const gentype &sigmay = xxvar;

    // NB: following calculation actually for increase, not decrease, so we
    //     have an infestation of negatives.

    double res = 0;

    if ( !(N()-NNC(0)) )
    {
        res = -( (double) muy );
    }

    else if ( (double) sigmay > zerotol() )
    {
        double eps  = 2*log(powf(((double) N())+1,(2+(((double) ucbvdim())/2)))*(NUMBASE_PI*NUMBASE_PI/(3*ucbdelta())));
        double beta = ucbnu()*eps;

        res = -( (double) muy ) + ( sqrt(beta) * ( (double) sigmay ) );
    }

    else
    {
        res = -( (double) muy );
    }

    resi.force_double() = -res;

    return ( res == 0 ) ? 0 : ( ( -res > 0 ) ? +1 : -1 );
}
