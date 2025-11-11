
//
// 1-norm 1-class Pareto SVM measure
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
#include "imp_parsvm.hpp"




IMP_ParSVM::IMP_ParSVM(int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(nullptr);





    return;
}

IMP_ParSVM::IMP_ParSVM(const IMP_ParSVM &src, int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(nullptr);




    assign(src,0);

    return;
}

IMP_ParSVM::IMP_ParSVM(const IMP_ParSVM &src, const ML_Base *xsrc, int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(xsrc);




    assign(src,-1);

    return;
}

IMP_ParSVM::~IMP_ParSVM()
{






    return;
}

std::ostream &IMP_ParSVM::printstream(std::ostream &output,int dep) const
{
    repPrint(output,'>',dep) << "SVM Pareto improvement block\n";

    repPrint(output,'>',dep) << "Pareto SVM: " << content << "\n";

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_ParSVM::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> content;

    return IMP_Generic::inputstream(input);
}







int IMP_ParSVM::imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxxstddev) const
{
    // Things can get a little confusing here.  We are trying to
    // minimise a function overall, but locally (ie in this context)
    // the problem is re-framed as *maximisation*.  The outputs we
    // see are all greater than zref(), and we want to maximise them.

    NiceAssert( isTrained() );

    gentype tempresh;

    SparseVector<gentype> xx(xxmean);

    gentype gzref(zref());

    xx -= gzref;

    gh(tempresh,resi,xx);

    //SparseVector<gentype> mgrad;
    //const static gentype dummy('N');
    //
    //dgg(mgrad,dummy,xxmean);

    gentype mgrad;
    SparseVector<gentype> &xxmeangrad = xx;

    double &resvv = resv.force_double();

    if ( xxxstddev.isValNull() )
    {
        resvv = 0.0;
    }

    if ( resvv > zerotol() )
    {
        resvv = ((const Vector<double> &) xxxstddev)(0); // FIXME: there is a big and rather obvious assumption here

        xxmeangrad.f4("&",6).force_int() = 1;
        gg(mgrad,xxmeangrad);

        resv *= abs2(mgrad);
        resvv *= resvv;
    }

    return ( ((double) resi) == 0 ) ? 0 : ( ( -((double) resi) > 0 ) ? +1 : -1 );
}
