
//
// Expected improvement (EHI for multi-objective)
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
#include "imp_expect.hpp"
#include "hyper_alt.hpp"
#include "hyper_base.hpp"


IMP_Expect::IMP_Expect(int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(nullptr);

    xmaxval = 0.0;
    hc      = nullptr;
    X       = nullptr;

    return;
}

IMP_Expect::IMP_Expect(const IMP_Expect &src, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(nullptr);




    assign(src,0);

    return;
}

IMP_Expect::IMP_Expect(const IMP_Expect &src, const ML_Base *xsrc, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(xsrc);




    assign(src,-1);

    return;
}

IMP_Expect::~IMP_Expect()
{
    untrain(); //this will delete hc and X if required.





    return;
}

std::ostream &IMP_Expect::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Expected improvement block\n";

    repPrint(output,'>',dep) << "x minima: " << xmaxval << "\n";

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_Expect::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xmaxval;

    if ( xspaceDim() > 1 )
    {
        untrain();
    }

    return IMP_Generic::inputstream(input);
}

int IMP_Expect::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !retaltg );

    (void) retaltg;
    (void) pxyprodi;

    gentype dummyresv;

    int res = imp(resg,dummyresv,x(i),nullgentype());

    resh = resg;

    return res;
}












int IMP_Expect::train(int &res, svmvolatile int &killSwitch)
{
    // Things can get a little confusing here.  We are trying to
    // minimise a function overall, but locally (ie in this context)
    // the problem is re-framed as *maximisation*.  The outputs we
    // see are all greater than zref(), and we want to maximise them.

    (void) res;

    int retval = 0;

    incgvernum();

    if ( !isTrained() )
    {
        retval = IMP_Generic::train(res,killSwitch);

        xmaxval = 0.0;

        if ( N()-NNC(0) )
        {
            if ( xspaceDim() <= 1 )
            {
                NiceAssert( xspaceDim() == 1 );

                int i;
                gentype temp;

                xelm(xmaxval,0,0);
                xmaxval -= zref();

                for ( i = 1 ; ( i < N() ) && !killSwitch ; ++i )
                {
                    if ( isenabled(i) )
                    {
                        xelm(temp,i,0);
                        temp -= zref();

                        if ( temp < xmaxval )
                        {
                            xmaxval = temp;
                        }
                    }
                }
            }

            else
            {
                int M = N()-NNC(0);
                int n = xspaceDim();
                gentype temp;

                MEMNEWARRAY(X,double *,M+1);

                int i,j,k;

                for ( i = 0, j = 0 ; i < N() ; ++i )
                {
                    if ( isenabled(i) )
                    {
                        MEMNEWARRAY(X[j],double,xspaceDim());

                        for ( k = 0 ; k < xspaceDim() ; ++k )
                        {
                            xelm(temp,i,k);
                            X[j][k] = (((double) temp)-zref());
                        }

                        ++j;
                    }
                }

                if ( ehimethod() <= 1 )
                {
                    hc = make_cache(X,M,ehimethod() ? -n : n);
                }
            }
        }
    }

    return retval;
}

void IMP_Expect::untrain(void)
{
    incgvernum();

    if ( hc )
    {
        del_cache(hc);
        hc = nullptr;
    }

    if ( X )
    {
        int j;

        for ( j = 0 ; j < N() ; ++j )
        {
            MEMDELARRAY(X[j]);
        }

        MEMDELARRAY(X);
        X = nullptr;
    }

    IMP_Generic::untrain();

    return;
}

int IMP_Expect::imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxsigma) const
{
    // Things can get a little confusing here.  We are trying to
    // minimise a function overall, but locally (ie in this context)
    // the problem is re-framed as *maximisation*.  The outputs we
    // see are all greater than zref(), and we want to maximise them.

    NiceAssert( isTrained() );

    double &res = resi.force_double();
    double &vres = resv.force_double();

    vres = 0.0;

    if ( !(N()-NNC(0)) )
    {
        res = 0.0;

        if ( xxmean.size() )
        {
            res = 1.0;

            int j;

            for ( j = 0 ; j < xxmean.size() ; ++j )
            {
                res *= (((double) xxmean(j))-zref());
            }
        }
    }

    else if ( xspaceDim() <= 1 )
    {
        gentype altvar(0.0);

        const gentype &muy = xxmean(0);
        const gentype &sigmay = xxsigma.isValNull() ? altvar : xxsigma(0);

        double ymax = (double) xmaxval;
        double yval = ((double) muy)-zref();
        double sigmaval = (double) sigmay;

        if ( sigmaval > zerotol() )
        {
            double z = (yval-ymax)/sigmaval;
            double Phiz = 0.5 + (0.5*erf(z*NUMBASE_SQRT1ON2));
            double phiz = exp(-z*z/2)/2.506628;

            res = ((yval-ymax)*Phiz) + (sigmaval*phiz);
        }

        else
        {
            // if muy > ymax then z = +infty, so Phiz = +1, phiz = 0
            // if muy < ymax then z = -infty, so Phiz = 0,  phiz = infty
            // assume lim_{z->-infty} sigmay.phiz = 0

            res = ( yval > ymax ) ? (yval-ymax) : 0.0;
        }
    }

    else
    {
//errstream() << "phantomxyz: " << x() << "\n";
//for ( int j = 0 ; j < N() ; j++ )
//{
//for ( int jj = 0 ; jj < 2 ; jj++ )
//{
//errstream() << "phantomxyz X[" << j << "][" << jj << "] = " << X[j][jj] << "\n";
//}
//}
        int j;

        double *mu;
        double *s;

        bool svalzero = true;

        MEMNEWARRAY(mu,double,xspaceDim());
        MEMNEWARRAY(s ,double,xspaceDim());

        for ( j = 0 ; j < xspaceDim() ; ++j )
        {
            mu[j] = (((double) xxmean(j))-zref());
            s[j] = xxsigma.isValNull() ? 0.0 : ((double) xxsigma(j));

            svalzero = ( svalzero && ( s[j] < zerotol() ) ) ? true : false;
        }

        if ( svalzero )
        {
            res = hi(X,N(),xspaceDim(),mu);
        }

        else
        {
            switch ( ehimethod() )
            {
                case 0:
                case 1:
                {
                    NiceAssert( hc );
                    res = ehi(mu,s,hc);
                    break;
                }

                case 2:
                {
                    res = ehi(X,N(),xspaceDim(),mu,s);
                    break;
                }

                case 3:
                {
                    res = ehi_hup(X,N(),xspaceDim(),mu,s);
                    break;
                }

                default:
                {
                    NiceAssert( ehimethod() == 4 );
                    res = ehi_cou(X,N(),xspaceDim(),mu,s);
                    break;
                }
            }
        }

        MEMDELARRAY(mu);
        MEMDELARRAY(s);
    }

//errstream() << "phantomxyz res = " << res << "\n";
//errstream() << "phantomxyz zbox = " << zbox << "\n";

    return ( res == 0 ) ? 0 : ( ( -res > 0 ) ? +1 : -1 );
}
