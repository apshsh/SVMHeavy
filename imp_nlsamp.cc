
//
// Random thompson sample form GP scalarisation
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
#include "imp_nlsamp.hpp"
#include "hyper_alt.hpp"
#include "hyper_base.hpp"


IMP_NLSamp::IMP_NLSamp(int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(nullptr);

    randscal = nullptr;
    dbias    = 0;
    dscale   = 0;


    return;
}

IMP_NLSamp::IMP_NLSamp(const IMP_NLSamp &src, int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(nullptr);

    randscal = nullptr;
    dbias    = 0;
    dscale   = 0;

    assign(src,0);

    return;
}

IMP_NLSamp::IMP_NLSamp(const IMP_NLSamp &src, const ML_Base *xsrc, int _isIndPrune) : IMP_Generic(_isIndPrune)
{
    setaltx(xsrc);

    randscal = nullptr;
    dbias    = 0;
    dscale   = 0;

    assign(src,-1);

    return;
}

IMP_NLSamp::~IMP_NLSamp()
{
    if ( randscal )
    {
        MEMDEL(randscal); randscal = nullptr;
    }

    return;
}

std::ostream &IMP_NLSamp::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Expected improvement block\n";

    repPrint(output,'>',dep) << "scalarisation template: " << randscaltemplate << "\n";
    repPrint(output,'>',dep) << "bias correction:        " << dbias            << "\n";
    repPrint(output,'>',dep) << "scale correction:       " << dscale           << "\n";

    if ( randscal )
    {
        repPrint(output,'>',dep) << "sampled scalarisation: " << *randscal << "\n";
    }

    else
    {
        repPrint(output,'>',dep) << "sampled scalarisation: none\n";
    }

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_NLSamp::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> randscaltemplate;
    input >> dummy; input >> dbias;
    input >> dummy; input >> dscale;

    if ( randscal )
    {
        MEMDEL(randscal); randscal = nullptr;
    }

    return IMP_Generic::inputstream(input);
}

/*
int IMP_NLSamp::gh(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    gentype dummyresv;

    int res = imp(resg,dummyresv,x(i),realgentype());
    resh = resg;

    return res;
}

int IMP_NLSamp::gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg, const vecInfo *xinf, gentype ***pxyprodx) const
{
    (void) retaltg;
    (void) xinf;
    (void) pxyprodx;

    gentype dummyresv;

    int res = imp(resg,dummyresv,x,realgentype());

    resh = resg;

    return res;
}
*/












int IMP_NLSamp::train(int &res, svmvolatile int &killSwitch)
{
    untrain();

    int retval = IMP_Generic::train(res,killSwitch);

errstream() << "Sample scalarisation model... \n";
    if ( xdim() > 0 )
    {
        MEMNEW(randscal,GPR_Scalar(randscaltemplate));

        int i,moodim = xspaceDim();

        Vector<gentype> ymin(moodim);
        Vector<gentype> ymax(moodim);

        for ( i = 0 ; i < moodim ; i++ )
        {
            ymin("&",i) = 0.0;
            ymax("&",i) = 1.0;
        }

        int sampSplit = 1;        // Default
        int sampleTSmode = 1;     // Random simple sample
        int sampleType = 100;     // Standard type with direct squaring
        //int sampleType = 7;       // alpha from uniform distribution
        int xsampleType = 0;      // True random grid
        double sampleScale = 1.0; // No scaling for now

        (*randscal).setSampleMode(sampleTSmode,ymin,ymax,Nsamp(),sampSplit,sampleType,xsampleType,sampleScale,sampSlack());

        SparseVector<gentype> ymintest(ymin);
        SparseVector<gentype> ymaxtest(ymax);

//        ymintest.f4("&",11) = 1;
//        ymintest.f4("&",13) = -sampSlack();

//        ymaxtest.f4("&",11) = 1;
//        ymaxtest.f4("&",13) = -sampSlack();

        gentype dummy,minres,maxres,sf,dummyb('N');

        dbias = 0.0;
        dscale = 1.0;

        //(*randscal).gh(dummy,minres,ymintest,2);
        //(*randscal).gh(dummy,maxres,ymaxtest,2);

        imp(minres,dummy,ymintest,dummyb);
        imp(maxres,dummy,ymaxtest,dummyb);

        dbias  = -(double) minres;
        dscale = 1/((double) (maxres-minres));

//errstream() << "phantomxyz trained: " << *randscal << "\n";
//errstream() << "phantomxyz bias = " << dbias << "\n";
//errstream() << "phantomxyz scale = " << dscale << "\n";
//errstream() << "phantomxyz minres = " << minres << "\n";
//errstream() << "phantomxyz maxres = " << maxres << "\n";

//{
//        SparseVector<gentype> yymin(ymin);
//        SparseVector<gentype> yymax(ymax);
//gentype ddum,impres,varres;
//imp(impres,varres,yymin,ddum);
//errstream() << "phantomxyz 0: imp(" << yymin << ") = " << impres << " with var " << varres << "\n";
//imp(impres,varres,yymax,ddum);
//errstream() << "phantomxyz 1: imp(" << yymax << ") = " << impres << " with var " << varres << "\n";
//}
    }

    return retval;
}

void IMP_NLSamp::untrain(void)
{
    if ( randscal )
    {
        MEMDEL(randscal); randscal = nullptr;
    }

    dbias  = 0;
    dscale = 0;

    return;
}

int IMP_NLSamp::imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxxstddev) const
{
    // Things can get a little confusing here.  We are trying to
    // minimise a function overall, but locally (ie in this context)
    // the problem is re-framed as *maximisation*.  The outputs we
    // see are all greater than zref(), and we want to maximise them.

    // NB: in TS the inputs often fall outside this range.  To prevent
    //     the global optimiser getting lost in flatland (and it will:
    //     if the function is too flat then global optimisers tend to
    //     fail, so e.g. DIReCT will just suggest something random like
    //     [ 0.5 0.5 ] or similar), linearly reward if inputs are too big,
    //     linearly penalise if too small.

    double outpenalty = 0;

    resi.force_double() = 0.0;
    resv.force_double() = 0.0;

    NiceAssert( isTrained() );
    NiceAssert( xxmean.size() == xspaceDim() );

    SparseVector<gentype> xx(xxmean);

    for ( int i = 0 ; i < xx.indsize() ; i++ )
    {
        xx.direref(i).force_double() -= zref();

        double xvalhere = xx.direcref(i);

        outpenalty += ( xvalhere > 1 ) ? (xvalhere-1) : 0;
        outpenalty += ( xvalhere < 0 ) ? xvalhere : 0;
    }

    xx.f4("&",11) = 1;

    gentype dummy;

    getQconst().gh(dummy,resi,xx); //,2);

//errstream() << "phantomxyz resi = " << resi << "\n";
    // If used outside of a TS framework we also need to approximate the variance

    resv.force_double() = 0.0;

    if ( !xxxstddev.isValNull() && ( ((double) norm2(xxxstddev)) > zerotol() ) )
    {
        gentype mgrad;

        // dg(x)^2/dx = 2g(x) dg(x)/dx

        SparseVector<gentype> xxx(xxmean);

        for ( int i = 0 ; i < xxmean.indsize() ; i++ )
        {
            xxx.direref(i).force_double() -= zref();

            // gradient in direction of standard deviation - very specific here

            xxx.f2("&",i) = ((double) xxxstddev(i))*((double) xxxstddev(i));
        }

        xxx.f4("&",11).force_int() = 1; // dense integral

//FIXME: need to implement gradient in lsv_scalar at ghTraningVector level (currently won't work)
        gg(resv,xxx); // variance projected in direction of gradient scaled by slope

        resv = emul(resv,resv);
    }

    resi += dbias;
    resi *= dscale;

    resi += outpenalty; // avoid the flatlands
//errstream() << "phantomxyz (resi+bias)*scale = " << resi << "\n";

    resv *= dscale;
    resv *= dscale;

    return ( ((double) resi) == 0 ) ? 0 : ( ( ((double) resi) > 0 ) ? +1 : -1 );
}
