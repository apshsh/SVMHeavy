
//
// Scalar RFF regression GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_scalar_rff.hpp"

GPR_Scalar_rff::GPR_Scalar_rff() : GPR_Generic()
{
    setsigma(DEFAULT_SIGMA);

    setZeromuBias();

    setaltx(nullptr);

    xNaiveConst = 0;

    return;
}

GPR_Scalar_rff::GPR_Scalar_rff(const GPR_Scalar_rff &src) : GPR_Generic()
{
    setsigma(DEFAULT_SIGMA);

    setZeromuBias();

    setaltx(nullptr);

    xNaiveConst = 0;

    assign(src,0);

    return;
}

GPR_Scalar_rff::GPR_Scalar_rff(const GPR_Scalar_rff &src, const ML_Base *srcx) : GPR_Generic()
{
    setsigma(DEFAULT_SIGMA);

    setZeromuBias();

    setaltx(srcx);

    xNaiveConst = 0;

    assign(src,-1);

    return;
}


int GPR_Scalar_rff::setNaiveConst(void)
{
    int res = 0;

    if ( !xNaiveConst )
    {
        res = 1;
        xNaiveConst = 1;

        getQ().setd(d()); // make d in Q directly match real d
    }

    return res;
}

int GPR_Scalar_rff::setEPConst(void)
{
    int res = 0;

    if ( xNaiveConst )
    {
        res = 1;
        xNaiveConst = 0;

        Vector<int> dd(d());

        for ( int i = 0 ; i < N() ; i++ )
        {
            dd("&",i) = dd(i) ? 2 : 0;
        }

        getQ().setd(dd); // treat d=+-1 as special cases
    }

    return res;
}





int GPR_Scalar_rff::gg(gentype &resg, const SparseVector<gentype> &x, const vecInfo *xinf, gentype ***pxyprodx) const
{
    int res = 0;

    if ( 3 == isSampleMode() )
    {
        gentype resv;
        var(resv,resg,x,xinf,pxyprodx);

        res = ( ((double) resg) > 0 ) ? +1 : -1;
    }

    else
    {
        res = getQconst().gg(resg,x,xinf,pxyprodx);
    }

    return res;
}

int GPR_Scalar_rff::hh(gentype &resh, const SparseVector<gentype> &x, const vecInfo *xinf, gentype ***pxyprodx) const
{
    int res = 0;

    if ( 3 == isSampleMode() )
    {
        gentype resv;
        var(resv,resh,x,xinf,pxyprodx);

        res = ( ((double) resh) > 0 ) ? +1 : -1;
    }

    else
    {
        res = getQconst().hh(resh,x,xinf,pxyprodx);
    }

    return res;
}

int GPR_Scalar_rff::gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg, const vecInfo *xinf, gentype ***pxyprodx) const
{
    int res = 0;

    if ( 3 == isSampleMode() )
    {
        gentype resv;
        var(resv,resg,x,xinf,pxyprodx);

        resh = resg;
        res = ( ((double) resg) > 0 ) ? +1 : -1;
    }

    else
    {
        res = getQconst().gh(resh,resg,x,retaltg,xinf,pxyprodx);
    }

    return res;
}












// Close-enough-to-infinity for variance

#define CETI 1e8
#define CUTI 1e-8

int GPR_Scalar_rff::train(int &res, svmvolatile int &killSwitch)
{
    if ( isLocked )
    {
        return 0;
    }

    int Nineq = NNC(-1)+NNC(+1);

    int locres = 0;

    if ( !Nineq || isNaiveConst() )
    {
        locres = getQ().train(res,killSwitch);
    }

    else if ( N() == 1 )
    {
        int dval = d()(0);

//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
        getQ().setd(0,dval);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
        locres = getQ().train(res,killSwitch);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
        getQ().setd(0,2);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
    }

    else
    {
        int i,j;

        // Get indices of inequality constraints

        Vector<int> indin(Nineq);

        for ( i = 0, j = 0 ; i < N() ; ++i )
        {
            if ( ( d()(i) == +1 ) || ( d()(i) == -1 ) )
            {
                indin("&",j) = i;

                ++j;
            }
        }

        // Start with "base" mean/variance

        retVector<double> tmpva;
        retVector<double> tmpvb;
        retVector<gentype> tmpvc;
        retVector<gentype> tmpvd;

        Vector<gentype> bmean(y()(indin,tmpvc));
        Vector<double> bvar(sigmaweight()(indin,tmpvb));

        bvar *= sigma();

        Vector<double> nutilde(Nineq);
        Vector<double> tautilde(Nineq);

//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
        getQ().sety(indin,bmean);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
        getQsetsigmaweight(indin,bvar/sigma());
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";

        locres = getQ().train(res,killSwitch);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";

        // Main EP loop

        Vector<gentype> smean(bmean);
        Vector<double> svar(bvar);

        Vector<gentype> pmean(smean);
        Vector<double> pvar(svar);

        int isdone = 0;


// Notation: - mutilde_i, vartilde_i = Sigmatilde_ii = sigmatilde_i^2 are the mean and variance
//             of the pseudo-observation of y (that we use to enforce the real observation/constraint)
//           - ...negi means "without observation i" - that is, set d(i) zero before training.
//           - {\bg{\Sigma}} = ( {\bf K}^{-1} + {\tilde{\bg{\Sigma}}}^{-1} )^{-1}
//                           = {\bf K} - {\bf K} ( {\bf K} + {\tilde{\bg{\Sigma}}} )^{-1} {\bf K}
//             so mu_i, vari = Sigma_ii = sigma_i^2 are the posterior mean/variance when evaluating f

        while ( !isdone )
        {
            // Record current mean/variance (for convergence testing)

            pmean = smean;
            pvar  = svar;

            // Loop through constraints

            for ( j = 0 ; j < Nineq ; ++j )
            {
                i = indin(j);

                // Disable vector and update

                getQ().setd(i,0);
//errstream() << "indin(" << j << ") = i = " << i << "\n";
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
                locres |= getQ().train(res,killSwitch);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";

                // (3.56) in Ras2

                gentype gvari,gmeani;

                getQconst().varTrainingVector(gvari,gmeani,i);
//errstream() << "phantomxyz var,mean " << i << " = " << gvari << ", " << gmeani << "\n";

                double munegi  = (double) gmeani;
                double varnegi = (double) gvari;
//errstream() << "phantomxyz munegi raw " << munegi  << "\n";
//errstream() << "phantomxyz varegmi    " << varnegi << "\n";

                double targi = (double) y()(i);
                double yi = (double) d()(i);
//errstream() << "phantomxyz targi " << targi << "\n";
//errstream() << "phantomxyz yi    " << yi    << "\n";

                // Factor out target, which is zero in Ras2

                munegi -= targi;
//errstream() << "phantomxyz mui " << munegi << "\n";

                // (3.58) in Ras2

                //double zi = yi*munegi/sqrt(1+varnegi); // yi*mui/sqrt(nu+vari); - this was the old varsion.  Why nu not 1?
                //double phizionPhizi = normphionPhi(zi);
                double zi = yi*munegi/sqrt(1+varnegi); // yi*mui/sqrt(nu+vari); - this was the old varsion.  Why nu not 1?
                double phizionPhizi = normphionPhi(zi);
//errstream() << "phantomxyz zi " << zi << "\n";
//errstream() << "phantomxyz phizionPhizi " << phizionPhizi << "\n";

                //double varhati = vari - (((vari*vari*phizi)/((nu+vari)*Phizi))*(zi+(phizi/Phizi))); - this was the old varsion.  Why nu not 1?
                //double muhati  = mui  + ((yi*vari*phizi)/(sqrt(nu+vari)*Phizi)); - this was the old varsion.  Why nu not 1?
                double muhati  = ( munegi + (yi               *varnegi*phizionPhizi/sqrt(1+varnegi)) );
                double varhati = ( 1      - ((zi+phizionPhizi)*varnegi*phizionPhizi/    (1+varnegi)) ) * varnegi;
//errstream() << "phantomxyz muhati  " << muhati  << "\n";
//errstream() << "phantomxyz varhati " << varhati << "\n";

                muhati = ( muhati >  CETI ) ?  CETI : muhati;
                muhati = ( muhati < -CETI ) ? -CETI : muhati;

                varhati = ( varhati >  CETI ) ?  CETI : varhati;
                varhati = ( varhati < -CETI ) ? -CETI : varhati;

                varhati = ( ( varhati <  0 ) && ( varhati > -CUTI ) ) ? -CUTI : varhati;
                varhati = ( ( varhati >= 0 ) && ( varhati <  CUTI ) ) ?  CUTI : varhati;
//errstream() << "phantomxyz muhati bounded  " << muhati  << "\n";
//errstream() << "phantomxyz varhati bounded " << varhati << "\n";

                // (3.59) in Ras2
                //
                // NB: for numerical reasons we limit how bit vartildei can be
                //     (remember that this is the offset on the diagonal of the
                //      Hessian, so if it's too small then the Cholesky is likely
                //      to be numerically unstable).

                //double deltautilde = 1/varhati - taunegi - tautilde(j);

                double vartildei =         1/( (     1/varhati) - (     1/varnegi) );
                double mutildei  = vartildei/( (muhati/varhati) - (munegi/varnegi) );
//errstream() << "phantomxyz mutildei  " << mutildei  << "\n";
//errstream() << "phantomxyz vartildei " << vartildei << "\n";

                mutildei = ( mutildei >  CETI ) ?  CETI : mutildei;
                mutildei = ( mutildei < -CETI ) ? -CETI : mutildei;

                vartildei = ( vartildei >  CETI ) ?  CETI : vartildei;
                vartildei = ( vartildei < -CETI ) ? -CETI : vartildei;

                vartildei = ( ( vartildei <  0 ) && ( vartildei > -CUTI ) ) ? -CUTI : vartildei;
                vartildei = ( ( vartildei >= 0 ) && ( vartildei <  CUTI ) ) ?  CUTI : vartildei;
//errstream() << "phantomxyz mutildei bounded  " << mutildei  << "\n";
//errstream() << "phantomxyz vartildei bounded " << vartildei << "\n";

                // Factor target back in

                mutildei  += targi;
//errstream() << "phantomxyz mutildei " << mutildei << "\n";

                // Update stuff.

                smean("&",j) = mutildei;
                svar("&",j)  = vartildei;

                // Re-enable vector and update (no need to train)

                getQ().setd(i,2);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
            }

            // Set new mean/var (no need to train)

//errstream() << "phantomxyz smean = " << smean << "\n";
//errstream() << "phantomxyz svar/sigma = " << svar << "/" << sigma() << " = " << svar/sigma() << "\n";
            getQ().sety(indin,smean);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
            getQsetsigmaweight(indin,svar/sigma());
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";

            // test convergence

            // Want to test convergence in normalised domain
            // pmean-smean/pmean+smean

            double stepsize = norm2(pvar(indin,tmpva)-svar(indin,tmpvb));

            for ( j = 0 ; j < Nineq ; j++ )
            {
                stepsize += norm2((((double) pmean(indin(j)))-((double) smean(indin(j))))/(fabs((double) pmean(indin(j)))+fabs((double) smean(indin(j)))));
            }
//errstream() << "phantomxyz pmean " << pmean << "\n";
//errstream() << "phantomxyz smean " << smean << "\n";
//errstream() << "phantomxyz pvar  " << pvar  << "\n";
//errstream() << "phantomxyz svar  " << svar  << "\n";
errstream() << "!" << stepsize << "!";

//FIXME: may need better convergence test.
            isdone = ( stepsize <= 1e-3 ) ? 1 : 0;
        }

        // Final training update

//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
        locres |= getQ().train(res,killSwitch);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
    }

    return locres;
}

std::ostream &GPR_Scalar_rff::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "GPR (Scalar/Real RFF)\n";

    GPR_Generic::printstream(output,dep+1);

    return output;
}

std::istream &GPR_Scalar_rff::inputstream(std::istream &input )
{
    GPR_Generic::inputstream(input);

    return input;
}

