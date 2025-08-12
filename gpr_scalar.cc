
//
// Scalar regression GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_scalar.hpp"

GPR_Scalar::GPR_Scalar() : GPR_Generic()
{
    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(nullptr);

    xNaiveConst  = 0; // Laplace by default
    xEPorLaplace = 1; // Laplace by default (EP is buggy)
    isLocked     = 0;

    return;
}

GPR_Scalar::GPR_Scalar(const GPR_Scalar &src) : GPR_Generic()
{
    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(nullptr);

    xNaiveConst  = 0;
    xEPorLaplace = 1;
    isLocked     = 0;

    assign(src,0);

    return;
}

GPR_Scalar::GPR_Scalar(const GPR_Scalar &src, const ML_Base *srcx) : GPR_Generic()
{
    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(srcx);

    xNaiveConst  = 0;
    xEPorLaplace = 1;
    isLocked     = 0;

    assign(src,-1);

    return;
}


int GPR_Scalar::setNaiveConst(void)
{
    int res = 0;

    if ( !xNaiveConst )
    {
        res = 1;

        xNaiveConst  = 1;
        xEPorLaplace = 1;

        getQ().setd(d()); // make d in Q directly match real d
    }

    return res;
}

int GPR_Scalar::setEPConst(void)
{
    int res = 0;

    if ( xNaiveConst )
    {
        res = 1;

        xNaiveConst  = 0; // Not naive
        xEPorLaplace = 0; // EP

        Vector<int> dd(d());

        for ( int i = 0 ; i < N() ; i++ )
        {
            dd("&",i) = dd(i) ? 2 : 0;
        }

        getQ().setd(dd); // treat d=+-1 as special cases
    }

    return res;
}

int GPR_Scalar::setLaplaceConst(void)
{
    int res = 0;

    if ( xNaiveConst )
    {
        res = 1;

        xNaiveConst  = 0; // Not naive
        xEPorLaplace = 1; // Laplace

        Vector<int> dd(d());

        for ( int i = 0 ; i < N() ; i++ )
        {
            dd("&",i) = dd(i) ? 2 : 0;
        }

        getQ().setd(dd); // treat d=+-1 as special cases
    }

    return res;
}









/*

Notes on Laplace approximation

We follow Rassmusen's book \cite{Ras2}, around page 43. We use p(f|y) = Phi(d\frac{f-yreal}{sigma}),
where d=+1 for f >= y, d=-1 for f <= y, and sigma is the variance given by the user. Working from
this, it isn't too hard to show that:

f^next = K.inv(K+inv(W)).(f + grad log p(y|f))
E(f(x)) = k(x)^T.inv(K+inv(W)).(f + grad log p(y|f))
W = -grad^2 log p(y|f)

So basically we start with the target:

t=yreal (t is target y for the underlying LSV, yreal is the true target in the dataset)
and then update:
f = outputs given training vectors X

and at each iteration update:

t = f + grad log p(y|f)
W = -grad^2 log p(y|f)
and then update:
f = outputs given training vectors X

and recurse until convergence.

Note that W is 1/sigma, so we use setC(W) rather than setsigma as C=1/sigma

For our log likelihood we use (Ras2, top of page 43):

log p(y_i|f_i) = log Phi(d_i\frac{f_i-yreal_i}{sigma_i})

where Phi is the cdf for the normal distribution; and so:

grad log p(y_i|f_i) = \frac{d_i}{sigma_i} \frac{phi(d_i\frac{f_i-yreal_i}{sigma_i})}{Phi(d_i\frac{f_i-yreal_i}{sigma_i})}
grad^2 log p(y_i|f_i) = -\frac{1}{sigma_i^2} \frac{phi(d_i\frac{f_i-yreal_i}{sigma_i})}{Phi(d_i\frac{f_i-yreal_i}{sigma_i})} [ \frac{phi(d_i\frac{f_i-yreal_i}{sigma_i})}{Phi(d_i\frac{f_i-yreal_i}{sigma_i})} - d_i \frac{f_i-yreal_i}{sigma_i} ]

where phi is the pdf for the normal distribution.
*/


// Close-enough-to-infinity for variance

//#define CETI 1e8
//#define CUTI 1e-8
#define CETI 1e25
#define CUTI 1e-25
#define MAXEPITCNT 4

#define LAPSTEPSTOP 1e-3
#define MAXLAPLACEITCNT 10

int GPR_Scalar::train(int &res, svmvolatile int &killSwitch)
{
    if ( isLocked )
    {
        return 0;
    }

    int Nineq = NNC(-1)+NNC(+1);
errstream() << "phantomxyz Nineq = " << Nineq << "\n";

    int locres = 0;

    if ( !Nineq || isNaiveConst() )
    {
//FIXME: tune eps based on sigma here if required (method between naive and full EP)
        locres = getQ().train(res,killSwitch);
errstream() << "phantomxyz method naive\n";
    }

//    else if ( N() == 1 )
//    {
//        int dval = d()(0);
//
////errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
//        getQ().setd(0,dval);
////errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
//        locres = getQ().train(res,killSwitch);
////errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
//        getQ().setd(0,2);
////errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
//    }

    else if ( isEPConst() )
    {
errstream() << "phantomxyz method EP\n";
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
        int itcnt = 0;
        int maxitcnt = MAXEPITCNT;

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
                //int di = d()(i);

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
                double zi = yi*munegi/sqrt(1+varnegi);
                //double zi = yi*mui/sqrt((nu*nu)+vari); - this is the Riihimaki version
                double phizionPhizi = normphionPhi(zi);
//errstream() << "phantomxyz zi " << zi << "\n";
//errstream() << "phantomxyz phizionPhizi " << phizionPhizi << "\n";

                //double varhati = vari - (((vari*vari*phizi)/((nu+vari)*Phizi))*(zi+(phizi/Phizi))); - this was the old varsion.  Why nu not 1?
                //double muhati  = mui  + ((yi*vari*phizi)/(sqrt(nu+vari)*Phizi)); - this was the old varsion.  Why nu not 1?
                double muhati  = ( munegi + (yi*varnegi*phizionPhizi/sqrt(1+varnegi)) );
                //double muhati  = ( munegi + (yi*varnegi*phizionPhizi/(nu*sqrt(1+(varnegi/(nu*nu))))) ); - this is the Riihimaki version
                double varhati = varnegi - ((zi+phizionPhizi)*varnegi*varnegi*phizionPhizi/(1+varnegi));
                //double varhati = varnegi - ((zi+phizionPhizi)*varnegi*varnegi*phizionPhizi/((nu*nu)+varnegi)); - this is the Riihimaki version
//errstream() << "phantomxyz muhati  " << muhati  << "\n";
//errstream() << "phantomxyz varhati " << varhati << "\n";

                muhati = ( muhati >  CETI ) ?  CETI : muhati;
                muhati = ( muhati < -CETI ) ? -CETI : muhati;

                varhati = ( varhati >  CETI ) ?  CETI : varhati;
                varhati = ( varhati < -CETI ) ? -CETI : varhati;

                //varhati = ( ( varhati <  0 ) && ( varhati > -CUTI ) ) ? -CUTI : varhati; - negative variance is nonsensical
                //varhati = ( ( varhati >= 0 ) && ( varhati <  CUTI ) ) ?  CUTI : varhati;
                varhati = ( varhati < CUTI ) ?  CUTI : varhati;
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

                //vartildei = ( ( vartildei <  0 ) && ( vartildei > -CUTI ) ) ? -CUTI : vartildei; - negative variance is nonsensical
                //vartildei = ( ( vartildei >= 0 ) && ( vartildei <  CUTI ) ) ?  CUTI : vartildei;
                vartildei = ( vartildei <  CUTI ) ?  CUTI : vartildei;
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
                //getQ().setd(i,di);
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
            ++itcnt;
            isdone = ( ( stepsize <= 1e-3*Nineq ) || ( itcnt >= maxitcnt ) ) ? 1 : 0;
        }

        // Final training update

//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
        locres |= getQ().train(res,killSwitch);
//errstream() << "phantomxyz line " << __LINE__ << " alpha = " << getQ().alphaVal(0) << "\n";
    }


    else
    {
errstream() << "phantomxyz method Laplace\n";
        int i,j;

        // Get indices of inequality constraints

        Vector<int> indin(Nineq);  // Index vector for inequality constraints
        Vector<int> twovec(Nineq); // Vector of 2s
        Vector<int> dreal(Nineq);

        for ( i = 0, j = 0 ; i < N() ; ++i )
        {
            if ( ( d()(i) == +1 ) || ( d()(i) == -1 ) )
            {
                indin("&",j)  = i;
                twovec("&",j) = 2;
                dreal("&",j)  = d()(i);

                ++j;
            }
        }

        // Start with "base" mean/variance

        retVector<int> tmpva;
        retVector<double> tmpvb;
        retVector<gentype> tmpvc;

        Vector<gentype> yreal(y()(indin,tmpvc));              // This is the underlying target
        Vector<double> sigrealsq(sigmaweight()(indin,tmpvb)); // This is the underlying variance

        sigrealsq *= sigma();

        Vector<gentype> f(Nineq);
        Vector<gentype> t(Nineq);
        Vector<double> W(Nineq);

        Vector<gentype> fnext(Nineq);

        // Initial train with *only* equality constraints (ie naive method)

        getQ().setd(indin,dreal);
        getQ().sety(indin,yreal);
        getQ().setsigmaweight(indin,sigrealsq/sigma());
        getQ().train(res,killSwitch);
        getQ().setd(indin,twovec);

        // Startup

        for ( j = 0 ; j < Nineq ; ++j )
        {
            f("&",j).force_double() = yreal(j);
        }

        // Laplace loop

        bool isdone = false;

        int itcnt = 0;
        int maxitcnt = MAXLAPLACEITCNT;

errstream() << "entry alpha = " << getQ().alphaVal() << "\n";
        while ( !isdone )
        {
            // Work out W and pseudo-targets

errstream() << "phantomxyz f = " << f << "\n";
            for ( j = 0 ; j < Nineq ; ++j )
            {
                double fmod     = dreal(j)*(((double) f(j))-((double) yreal(j)))/sqrt(sigrealsq(j));
                double phival   = normphi(fmod);
                double Phival   = normPhi(fmod);
                double phionPhi = phival/Phival;

                W("&",j)                = (1/sigrealsq(j))*phionPhi*(phionPhi+fmod);
                t("&",j).force_double() = ((double) f(j)) + (((double) dreal(j))*phionPhi/(sqrt(sigrealsq(j))*W(j)));
            }

            // Set W (which corresponds to C) and pseudo-targets and train

errstream() << "phantomxyz t = " << t << "\n";
errstream() << "phantomxyz W = " << W << "\n";
            getQ().sety(indin,t);
            getQ().setCweight(indin,W*sigma());
            getQ().train(res,killSwitch);
errstream() << "phantomxyz alpha = " << getQ().alphaVal() << "\n";

            // Termination conditions

            ++itcnt;

            if ( itcnt >= maxitcnt )
            {
                isdone = true;
            }

            else
            {
                for ( j = 0 ; j < Nineq ; ++j )
                {
                    getQ().ggTrainingVector(fnext("&",j),indin(j));
                }

                f -= fnext;

errstream() << "phantomxyz stepsize = " << abs2(f) << "\n";
                if ( abs2(f) <= LAPSTEPSTOP )
                {
                    isdone = true;
                }

                else
                {
                    f = fnext;
                }
            }
        }
    }

    return locres;
}

std::ostream &GPR_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "GPR (Scalar/Real)\n";

    repPrint(output,'>',dep) << "Base training inequality method: " << xNaiveConst  << "\n";
    repPrint(output,'>',dep) << "Base training EP or Laplace:     " << xEPorLaplace << "\n";

    GPR_Generic::printstream(output,dep+1);

    return output;
}

std::istream &GPR_Scalar::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xNaiveConst;
    input >> dummy; input >> xEPorLaplace;

    GPR_Generic::inputstream(input);

    return input;
}

