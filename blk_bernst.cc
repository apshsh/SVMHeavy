
//
// Bernstain polynomial block
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
#include "blk_bernst.hpp"
#include "gpr_generic.hpp"

BLK_Bernst::BLK_Bernst(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(nullptr);

    localygood = 0;

    locsampleMode = 0;
    locxsampType  = 3;
    locNsamp      = -1;
    locsampSplit  = 1;
    locsampType   = 0;
    locsampScale  = 1.0;

    return;
}

std::ostream &BLK_Bernst::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Bernstein polynomial block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Bernst::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}

const Vector<gentype> &BLK_Bernst::y(void) const
{
    Vector<gentype> &res = localy;

    if ( !localygood )
    {
        NiceAssert( locxmax.size() == locxmin.size() );

        // Generate x grid

        Vector<SparseVector<gentype> > xgrid;
        static thread_local GPR_Generic sampler;
        sampler.genSampleGrid(xgrid,locxmin,locxmax,locNsamp,locsampSplit,locxsampType,locsampSlack);

        int totsamp = xgrid.size();

        // Pre-calculate y vector

        res.resize(totsamp);

        for ( int jj = 0 ; jj < totsamp ; ++jj )
        {
            gg(res("&",jj),xgrid(jj));
        }

        localygood = 1;
    }

    return res;
}

int BLK_Bernst::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !retaltg );

    (void) retaltg;
    (void) pxyprodi;

    const gentype &n = bernDegree();
    const gentype &j = bernIndex();

    int res = 0;

    const SparseVector<gentype> &xx = x(i);

//errstream() << "phantomxyzqw 0\n";
    if ( xx.size() == 0 )
    {
        NiceAssert( n.isValNull() );
        NiceAssert( j.isValNull() );

        // By default we work on the assumption that the empty product = 1

        resg = 1.0;
        resh = resg;
//errstream() << "phantomxyzqw 1\n";
    }

    else if ( xx.size() == 1 )
    {
//errstream() << "phantomxyzqw 2\n";
        // "Standard" Bernstein basis polynomials.

        NiceAssert( n.isCastableToIntegerWithoutLoss() );
        NiceAssert( j.isCastableToIntegerWithoutLoss() );

        int nn = (int) n;
        int jj = (int) j;

//errstream() << "phantomxyzqw 3: " << nn << "\n";
//errstream() << "phantomxyzqw 4: " << jj << "\n";
        NiceAssert( nn >= 0 );
        NiceAssert( jj <= nn );

        gentype nnn(nn);
        gentype jjj(jj);

//errstream() << "phantomxyzqw 5: " << nnn << "\n";
//errstream() << "phantomxyzqw 6: " << jjj << "\n";
        const gentype &xxx = xx.direcref(0);
        const gentype ov(1.0);

        resg = (pow(xxx,jjj)*pow(ov-xxx,nnn-jjj)*((double) xnCr(nn,jj)));
//errstream() << "phantomxyzqw 7: pow(" << xxx << "," << jjj << " = " << pow(xxx,jjj) << "\n";
//errstream() << "phantomxyzqw 8: pow(" << ov-xxx << "," << nnn-jjj << ") = " << pow(ov-xxx,nnn-jjj) << "\n";
//errstream() << "phantomxyzqw 9: xnCr(" << nn << "," << jj << ") = " << xnCr(nn,jj) << "\n";
//errstream() << "phantomxyzqw 10: " << resg << "\n";
        resh = resg;
    }

    else
    {
        // See http://www.iue.tuwien.ac.at/phd/heitzinger/node17.html

        NiceAssert( n.isValVector() );
        NiceAssert( j.isValVector() );

        NiceAssert( n.size() == xx.size() );
        NiceAssert( j.size() == xx.size() );

        const Vector<gentype> &nn = (const Vector<gentype> &) n;
        const Vector<gentype> &jj = (const Vector<gentype> &) j;

        resg = 1.0;

        int i;

        for ( i = 0 ; i < xx.size() ; ++i )
        {
            const gentype &nnn = nn(i);
            const gentype &jjj = jj(i);

            NiceAssert( (int) nnn >= 0 );
            NiceAssert( (int) jjj <= (int) nnn );

            const gentype &xxx = xx(i);
            const gentype ov(1.0);

            resg *= (pow(xxx,jjj)*pow(ov-xxx,nnn-jjj)*((double) xnCr((int) nnn,(int) jjj)));
        }

        resh = resg;
    }

    return res;
}


