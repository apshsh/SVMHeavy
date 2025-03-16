
//
// Test function access block
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
#include "blk_testfn.hpp"
#include "opttest.hpp"
#include "paretotest.hpp"


std::ostream &operator<<(std::ostream &output, const BLK_Testfn &src)
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input, BLK_Testfn &dest)
{
    return dest.inputstream(input);
}

BLK_Testfn::BLK_Testfn(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(nullptr);

    localygood = 0;

    locsampleMode = 0;
    locNsamp      = -1;
    locsampType   = 0;
    locsampScale  = 1.0;

    return;
}

std::ostream &BLK_Testfn::printstream(std::ostream &output) const
{
    output << "Test function access block\n";

    return BLK_Generic::printstream(output);
}

std::istream &BLK_Testfn::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}

const Vector<gentype> &BLK_Testfn::y(void) const
{
    Vector<gentype> &res = localy;

    if ( !localygood )
    {
        int jj,kk;

        int allowGridSample = 1;
        int dim = locxmin.size();
        int totsamp = allowGridSample ? ( dim ? pow(locNsamp,dim) : 0 ) : locNsamp;

        NiceAssert( locxmax.size() == locxmin.size() );

        gentype xxmin(locxmin);
        gentype xxmax(locxmax);

        res.resize(totsamp);

        if ( allowGridSample && ( dim == 1 ) )
        {
            SparseVector<gentype> xq;

            for ( jj = 0 ; jj < locNsamp ; ++jj )
            {
                xq("&",0)  = (((double) jj)+0.5)/(((double) locNsamp));
                xq("&",0) *= (locxmax(0)-locxmin(0));
                xq("&",0) += locxmin(0);

                gg(res("&",jj),xq);
            }
        }

        else if ( allowGridSample )
        {
            Vector<int> jjj(dim);

            jjj = 0;

            for ( jj = 0 ; jj < totsamp ; ++jj )
            {
                SparseVector<gentype> xq;

                for ( kk = 0 ; kk < dim ; ++kk )
                {
                    xq("&",kk)  = (((double) jjj(kk))+0.5)/(((double) locNsamp));
                    xq("&",kk) *= (locxmax(kk)-locxmin(kk));
                    xq("&",kk) += locxmin(kk);
                }

                gg(res("&",jj),xq);

                for ( kk = 0 ; kk < dim ; ++kk )
                {
                    ++(jjj("&",kk));

                    if ( jjj(kk) >= locNsamp )
                    {
                        jjj("&",kk) = 0;
                    }

                    else
                    {
                        break;
                    }
                }
            }
        }

        else
        {
            for ( jj = 0 ; jj < totsamp ; ++jj )
            {
                gentype xx = urand(xxmin,xxmax);
                SparseVector<gentype> xq = (const Vector<gentype> &) xx;

                gg(res("&",jj),xq);
            }
        }

        localygood = 1;
    }

    return res;
}

int BLK_Testfn::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !retaltg );

    (void) retaltg;
    (void) pxyprodi;

    const SparseVector<gentype> &xx = x(i);
    Vector<double> xxx(xx.size());

    int j,res = 0;

    for ( j = 0 ; j < xx.size() ; ++j )
    {
        xxx("&",j) = (double) xx(j);
    }

    if ( testFnType() == 0 )
    {
        res = evalTestFn(testFnNum(),resg.force_double(),xxx,&testFnA());
    }

    else
    {
        Vector<gentype> &resx = resg.force_vector(testFnType());
        Vector<double> resxx(testFnType());

        res = evalTestFn(testFnNum(),xxx.size(),testFnType(),resxx,xxx,testFnAlpha());

        for ( j = 0 ; j < testFnType() ; ++j )
        {
            resx("&",j) = resxx(j);
        }
    }

    resh = resg;

    return res;
}


