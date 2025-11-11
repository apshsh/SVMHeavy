
//
// Random linear scalarisation
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
#include "imp_rlsamp.hpp"
#include "hyper_alt.hpp"
#include "hyper_base.hpp"
#include "randfun.hpp"


IMP_RLSamp::IMP_RLSamp(int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(nullptr);





    return;
}

IMP_RLSamp::IMP_RLSamp(const IMP_RLSamp &src, int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(nullptr);




    assign(src,0);

    return;
}

IMP_RLSamp::IMP_RLSamp(const IMP_RLSamp &src, const ML_Base *xsrc, int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(xsrc);




    assign(src,-1);

    return;
}

IMP_RLSamp::~IMP_RLSamp()
{






    return;
}

std::ostream &IMP_RLSamp::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Expected improvement block\n";

    repPrint(output,'>',dep) << "scalarisation: " << randlinscal << "\n";

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_RLSamp::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> randlinscal;

    return IMP_Generic::inputstream(input);
}

int IMP_RLSamp::gh(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    gentype dummyresv;

    int res = imp(resg,dummyresv,x(i),nullgentype());

    resh = resg;

    return res;
}












int IMP_RLSamp::train(int &res, svmvolatile int &killSwitch)
{
    int retval = IMP_Generic::train(res,killSwitch);

    if ( xdim() > 0 )
    {
        randlinscal.resize(xspaceDim());

        for ( int i = 0 ; i < xspaceDim() ; i++ )
        {
            randufill(randlinscal("&",i),0,1);
        }

        randlinscal *= 1/sum(randlinscal);
    }

    return retval;
}

void IMP_RLSamp::untrain(void)
{
    randlinscal.resize(0);

    return;
}

int IMP_RLSamp::imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxvar) const
{
    // Things can get a little confusing here.  We are trying to
    // minimise a function overall, but locally (ie in this context)
    // the problem is re-framed as *maximisation*.  The outputs we
    // see are all greater than zref(), and we want to maximise them.

    NiceAssert( isTrained() );

    double &res = resi.force_double();
    double &vres = resv.force_double();

    res  = 0.0;
    vres = 0.0;

    NiceAssert( xxmean.size() == xspaceDim() );

    switch ( scaltype() )
    {
        case 0:
        {
            // Linear scalarisation

            if ( xxmean.size() )
            {
                int j;
                double jres,jvres;

                for ( j = 0 ; j < xxmean.size() ; ++j )
                {
                    jres  = randlinscal(j)*(((double) xxmean(j)) - zref());
                    jvres = xxvar.isValNull() ? 0.0 : (randlinscal(j)*((double) xxvar(j))*((double) xxvar(j)));

                    res  += jres;
                    vres += jvres;
                }
            }

            break;
        }

        case 1:
        {
            // Chebyshev scalarisation

            if ( xxmean.size() )
            {
                int j;
                double jres,jvres;

                for ( j = 0 ; j < xxmean.size() ; ++j )
                {
                    jres  = randlinscal(j)*(((double) xxmean(j)) - zref());
                    jvres = xxvar.isValNull() ? 0.0 : (randlinscal(j)*((double) xxvar(j))*((double) xxvar(j)));

                    if ( !j || ( jres < res ) )
                    {
                        res  = jres;
                        vres = jvres;
                    }
                }
            }

            break;
        }

        case 2:
        {
            // Augmenteed Chebyshev scalarisation

            if ( xxmean.size() )
            {
                int j;
                double jres,jvres;
                double augres  = 0;
                double augvres = 0;

                for ( j = 0 ; j < xxmean.size() ; ++j )
                {
                    jres  = randlinscal(j)*(((double) xxmean(j)) - zref());
                    jvres = xxvar.isValNull() ? 0.0 : (randlinscal(j)*((double) xxvar(j))*((double) xxvar(j)));

                    augres  += scalalpha()*fabs(((double) xxmean(j)) - zref());
                    augvres += xxvar.isValNull() ? 0.0 : (scalalpha()*((double) xxvar(j))*((double) xxvar(j)));

                    if ( !j || ( jres < res ) )
                    {
                        res  = jres;
                        vres = jvres;
                    }
                }

                res  += augres;
                vres += augvres;
            }

            break;
        }

        case 3:
        {
            // Modified Chebyshev scalarisation

            if ( xxmean.size() )
            {
                int j;
                double jres,jvres;
                double augres  = 0;
                double augvres = 0;

                for ( j = 0 ; j < xxmean.size() ; ++j )
                {
                    jres  = randlinscal(j)*(((double) xxmean(j)) - zref());
                    jvres = xxvar.isValNull() ? 0.0 : (randlinscal(j)*((double) xxvar(j))*((double) xxvar(j)));

                    augres  += scalalpha()*fabs(((double) xxmean(j)) - zref());
                    augvres += xxvar.isValNull() ? 0.0 : (scalalpha()*((double) xxvar(j))*((double) xxvar(j)));

                    if ( !j || ( jres+augres < res ) )
                    {
                        res  = jres+augres;
                        vres = jvres+augvres;
                    }
                }
            }

            break;
        }

        default:
        {
            NiceThrow("Unknown scalarisation");

            break;
        }
    }

    return ( res == 0 ) ? 0 : ( ( res > 0 ) ? +1 : -1 );
}
