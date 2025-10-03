
//FIXME: need to add a further function to the final (squared) result to enforce constraints on the derivative on the dense integral!
//This may violate the increasing property required?
//Funciton must have integral form, N^2 points hidden in rank constraints a_17,18


// FIXME: use a_0, a_1 to speed up data doubling.
// FIXME: alphaType code needs to be re-included (even if not used)

// DESIGN DECISION: just draw whatever, then construct a "correcting" function so that g(x)+h(x) is the final result
// EXCEPTION: trivial (g(x)=y) parameters are treated normally
// REASON: found that other, "proper" methods failed to converge or had alpha so large it would clip, or elements of the cholesky factorisation ->inf
//         due to poor conditioning.  When we're doing constraints on GPs we're going outside of guassians to approximations anyhow, so just
//         adding a correction is fine I guess.

//
// Gaussian Process (GP) base class
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
#include "gpr_generic.hpp"
#include "ml_mutable.hpp"
#include "randfun.hpp"

GPR_Generic::GPR_Generic() : ML_Base_Deref()
{
    dsigma     = DEFAULT_SIGMA;
    dsigma_cut = DEFAULT_SIGMA_CUT;

    sampleMode  = 0;
    sampleScale = 1.0;

    setaltx(nullptr);

    xsigmaclass.resize(4) = 1.0;

    Nnc.resize(4);
    Nnc = 0;

    return;
}

GPR_Generic::GPR_Generic(const GPR_Generic &src) : ML_Base_Deref()
{
    dsigma     = DEFAULT_SIGMA;
    dsigma_cut = DEFAULT_SIGMA_CUT;

    sampleMode  = 0;
    sampleScale = 1.0;

    setaltx(nullptr);

    xsigmaclass.resize(4) = 1.0;

    Nnc.resize(4);
    Nnc = 0;

    assign(src,0);

    return;
}

GPR_Generic::GPR_Generic(const GPR_Generic &src, const ML_Base *srcx) : ML_Base_Deref()
{
    dsigma     = DEFAULT_SIGMA;
    dsigma_cut = DEFAULT_SIGMA_CUT;

    sampleMode  = 0;
    sampleScale = 1.0;

    setaltx(srcx);

    xsigmaclass.resize(4) = 1.0;

    Nnc.resize(4);
    Nnc = 0;

    assign(src,-1);

    return;
}

int GPR_Generic::prealloc(int expectedN)
{
    dy.prealloc(expectedN);
    dyR.prealloc(expectedN);
    dyA.prealloc(expectedN);
    dyV.prealloc(expectedN);
    dsigmaweight.prealloc(expectedN);
    dCweight.prealloc(expectedN);
    xd.prealloc(expectedN);
    getQ().prealloc(expectedN);

    return 0;
}

std::ostream &GPR_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Measurement sigma:         " << dsigma       << "\n";
    repPrint(output,'>',dep) << "Measurement sigma_cut:     " << dsigma_cut   << "\n";
    repPrint(output,'>',dep) << "Measurement sigma weights: " << dsigmaweight << "\n";
    repPrint(output,'>',dep) << "Measurement C weights:     " << dCweight     << "\n";
    repPrint(output,'>',dep) << "Measurement sigma class:   " << xsigmaclass  << "\n";
    repPrint(output,'>',dep) << "Local (actual) y:          " << dy           << "\n";
    repPrint(output,'>',dep) << "Local (actual) yR:         " << dyR          << "\n";
    repPrint(output,'>',dep) << "Local (actual) yA:         " << dyA          << "\n";
    repPrint(output,'>',dep) << "Local (actual) yV:         " << dyV          << "\n";
    repPrint(output,'>',dep) << "Local (actual) d:          " << xd           << "\n";
    repPrint(output,'>',dep) << "Class counts Nnc:          " << Nnc          << "\n";
    repPrint(output,'>',dep) << "Sample mode:               " << sampleMode   << "\n";
    repPrint(output,'>',dep) << "Sample scale:              " << sampleScale  << "\n";
    repPrint(output,'>',dep) << "Underlying LS-SVM:         " << getQconst()  << "\n\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &GPR_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dsigma;
    input >> dummy; input >> dsigma_cut;
    input >> dummy; input >> dsigmaweight;
    input >> dummy; input >> dCweight;
    input >> dummy; input >> xsigmaclass;
    input >> dummy; input >> dy;
    input >> dummy; input >> dyR;
    input >> dummy; input >> dyA;
    input >> dummy; input >> dyV;
    input >> dummy; input >> xd;
    input >> dummy; input >> Nnc;
    input >> dummy; input >> sampleMode;
    input >> dummy; input >> sampleScale;
    input >> dummy; input >> getQ();

    ML_Base::inputstream(input);

    return input;
}

int GPR_Generic::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( epsweigh = 1 );

    dsigmaweight.add(i);  dsigmaweight("&",i) = 1/Cweigh;
    dCweight.add(i);      dCweight("&",i)     = Cweigh;
    xd.add(i);            xd("&",i)           = dval; //2;
    dy.add(i);            dy("&",i)           = y;
    dyR.add(i);           dyR("&",i)          = (double) y;
    dyA.add(i);           dyA("&",i)          = (const d_anion &) y;
    dyV.add(i);           dyV("&",i)          = (const Vector<double> &) y;

    ++(Nnc("&",xd(i)+1));

    return getQ().addTrainingVector(i,y,x,Cweigh,epsweigh,dval);
}

int GPR_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( epsweigh = 1 );

    dsigmaweight.add(i);  dsigmaweight("&",i) = 1/Cweigh;
    dCweight.add(i);      dCweight("&",i)     = Cweigh;
    xd.add(i);            xd("&",i)           = dval; //2;
    dy.add(i);            dy("&",i)           = y;
    dyR.add(i);           dyR("&",i)          = (double) y;
    dyA.add(i);           dyA("&",i)          = (const d_anion &) y;
    dyV.add(i);           dyV("&",i)          = (const Vector<double> &) y;

    ++(Nnc("&",xd(i)+1));

    return getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh,dval);
}

int GPR_Generic::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; ++j )
        {
            res |= addTrainingVector(i+j,y(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int GPR_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; ++j )
        {
            res |= qaddTrainingVector(i+j,y(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int GPR_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    --(Nnc("&",xd(i)+1));

    y = dy(i);

    dsigmaweight.remove(i);
    dCweight.remove(i);
    xd.remove(i);
    dy.remove(i);
    dyR.remove(i);
    dyA.remove(i);
    dyV.remove(i);

    gentype dummy;

    return getQ().removeTrainingVector(i,dummy,x);
}

int GPR_Generic::removeTrainingVector(int i, int num)
{
    int res = 0;
    gentype y;
    SparseVector<gentype> x;

    while ( num )
    {
        --num;
        res |= removeTrainingVector(i+num,y,x);
    }

    return res;
}

int GPR_Generic::sety(int i, const gentype &nv)
{
    dy("&",i) = nv;

    dyR("&",i) = (double) nv;
    dyA("&",i) = (const d_anion &) nv;
    dyV("&",i) = (const Vector<double> &) nv;

    return getQ().sety(i,nv);
}

int GPR_Generic::sety(const Vector<int> &i, const Vector<gentype> &nv)
{
    for ( int ii = 0 ; ii < i.size() ; ++ii )
    {
        dy("&",i(ii)) = nv(ii);

        dyR("&",i(ii)) = (double) nv(ii);
        dyA("&",i(ii)) = (const d_anion &) nv(ii);
        dyV("&",i(ii)) = (const Vector<double> &) nv(00);
    }

    return getQ().sety(i,nv);
}

int GPR_Generic::sety(const Vector<gentype> &nv)
{
    for ( int ii = 0 ; ii < N() ; ++ii )
    {
        dy("&",ii) = nv(ii);

        dyR("&",ii) = (double) nv(ii);
        dyA("&",ii) = (const d_anion &) nv(ii);
        dyV("&",ii) = (const Vector<double> &) nv(00);
    }

    return getQ().sety(nv);
}

int GPR_Generic::sety(int i, double nv)
{
    dy("&",i) = nv;

    dyR("&",i) = nv;
    dyA("&",i) = nv;
    dyV("&",i) = nv;

    return getQ().sety(i,nv);
}

int GPR_Generic::sety(const Vector<int> &i, const Vector<double> &nv)
{
    for ( int ii = 0 ; ii < i.size() ; ++ii )
    {
        dy("&",i(ii)) = nv(ii);

        dyR("&",i(ii)) = nv(ii);
        dyA("&",i(ii)) = nv(ii);
        dyV("&",i(ii)) = nv(00);
    }

    return getQ().sety(i,nv);
}

int GPR_Generic::sety(const Vector<double> &nv)
{
    for ( int ii = 0 ; ii < N() ; ++ii )
    {
        dy("&",ii) = nv(ii);

        dyR("&",ii) = nv(ii);
        dyA("&",ii) = nv(ii);
        dyV("&",ii) = nv(00);
    }

    return getQ().sety(nv);
}

int GPR_Generic::sety(int i, const Vector<double> &nv)
{
    gentype n(nv);

    return sety(i,n);
}

int GPR_Generic::sety(const Vector<int> &i, const Vector<Vector<double> > &nv)
{
    Vector<gentype> n(i.size());

    for ( int ii = 0 ; ii < i.size() ; ++ii )
    {
        n("&",ii) = nv(ii);

        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
    }

    return sety(i,n);
}

int GPR_Generic::sety(const Vector<Vector<double> > &nv)
{
    Vector<gentype> n(N());

    for ( int ii = 0 ; ii < N() ; ++ii )
    {
        n("&",ii) = nv(ii);

        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
    }

    return sety(n);
}

int GPR_Generic::sety(int i, const d_anion &nv)
{
    gentype n(nv);

    return sety(i,n);
}

int GPR_Generic::sety(const Vector<int> &i, const Vector<d_anion> &nv)
{
    Vector<gentype> n(i.size());

    for ( int ii = 0 ; ii < i.size() ; ++ii )
    {
        n("&",ii) = nv(ii);

        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
    }

    return sety(i,n);
}

int GPR_Generic::sety(const Vector<d_anion> &nv)
{
    Vector<gentype> n(N());

    for ( int ii = 0 ; ii < N() ; ++ii )
    {
        n("&",ii) = nv(ii);

        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
        n("&",ii) = nv(ii);
    }

    return sety(n);
}

int GPR_Generic::setsigmaweight(int i, double nv)
{
    dsigmaweight("&",i) = nv;
    dCweight("&",i) = 1/nv;

    return getQ().setCweight(i,1/nv);
}

int GPR_Generic::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i.size() == nv.size() );

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            dCweight("&",i(j)) = 1/nv(j);
            dsigmaweight("&",i(j)) = nv(j);
        }
    }

    retVector<double> tmpva;

    return getQ().setCweight(i,dCweight(i,tmpva));
}

int GPR_Generic::setsigmaweight(const Vector<double> &nv)
{
    NiceAssert( N() == nv.size() );

    if ( nv.size() )
    {
        int j;

        for ( j = 0 ; j < nv.size() ; ++j )
        {
            dCweight("&",j) = 1/nv(j);
            dsigmaweight("&",j) = nv(j);
        }
    }

    return getQ().setCweight(dCweight);
}




int GPR_Generic::setCweight(int i, double nv)
{
    dCweight("&",i) = nv;
    dsigmaweight("&",i) = 1/nv;

    return getQ().setCweight(i,nv);
}

int GPR_Generic::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i.size() == nv.size() );

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            dCweight("&",i(j)) = nv(j);
            dsigmaweight("&",i(j)) = 1/nv(j);
        }
    }

    return getQ().setCweight(i,nv);
}

int GPR_Generic::setCweight(const Vector<double> &nv)
{
    NiceAssert( N() == nv.size() );

    if ( nv.size() )
    {
        int j;

        for ( j = 0 ; j < nv.size() ; ++j )
        {
            dCweight("&",j) = nv(j);
            dsigmaweight("&",j) = 1/nv(j);
        }
    }

    return getQ().setCweight(nv);
}

int GPR_Generic::scaleCweight(double s)
{
    dsigmaweight *= 1/s;
    dCweight *= s;

    return getQ().scaleCweight(s);
}

int GPR_Generic::scalesigmaweight(double s)
{
    dsigmaweight *= s;
    dCweight *= 1/s;

    return getQ().scaleCweight(1/s);
}

int GPR_Generic::setd(int i, int nd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    // for anions and vectors setting d == -1,+1 makes no sense

    NiceAssert( ( nd == 0 ) || ( nd == 2 ) || ( type() != 401 ) || ( type() != 402 ) );

    --(Nnc("&",xd(i)+1));
    ++(Nnc("&",nd+1));

    xd("&",i) = nd;

    if ( isNaiveConst() )
    {
        return getQ().setd(i,nd);
    }

    // the LSV layer only sees 0 (constrained) or 2 (unconstrained)

    return getQ().setd(i,nd ? 2 : 0);
}

int GPR_Generic::setd(const Vector<int> &i, const Vector<int> &nd)
{
    NiceAssert( i.size() == nd.size() );

    int res = 0;

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ++ii )
        {
            res |= setd(i(ii),nd(ii));
        }
    }

    if ( isNaiveConst() )
    {
        return getQ().setd(i,nd);
    }

    return res;
}

int GPR_Generic::setd(const Vector<int> &nd)
{
    NiceAssert( N() == nd.size() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; ++i )
        {
            res |= setd(i,nd(i));
        }
    }

    return res;
}












int GPR_Generic::noisevar(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xvar, int u, const vecInfo *xainf, gentype ***pxyprodx, gentype **pxyprodxx) const
{
// NB: code is repeated in GPR_Generic

    int res = var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx); // Use local version (difference is here)

    SparseVector<gentype> ximod(xa);

    ximod.f4("&",6) = ((int) ximod.f4(6))+1; // gradient is in addition to what is already there!

    int ui = ( u < 0 ) ? ximod.nupsize() : 1;

    int umin = ( u == -1 ) ? 0 : ( ( u == -2 ) ? 1 : u );
    int umax = ( u < 0 ) ? ui-1 : u;

    for ( u = umin ; u <= umax ; ++u )
    {
        if ( xvar.nupsize(u) && ximod.nupsize(u) )
        {
            // Gradient wrt xi

            ximod.f4("&",9) = u; // with respect to xu in [ x0 ~ x1 ~ ... ]

            gentype gradiu;

            res |= getQconst().gg(gradiu,ximod); // Note use of getQconst() here!

            // Assuming non-sparse data here!

            retVector<gentype> tmpvi;
            retVector<gentype> tmpva;

            const Vector<gentype> &xvaru = xvar.nup(u)(tmpva);
            const Vector<gentype> &gradiuv = (gradiu.cast_vector())(0,1,xvaru.size()-1,tmpvi);

            gentype tmp;

            resv += threeProduct(tmp,gradiuv,xvaru,gradiuv);
        }
    }

    return res;
}

int GPR_Generic::noisecov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xvar, int u, const vecInfo *xainf, const vecInfo *xbinf, gentype ***pxyprodx, gentype ***pxyprody, gentype **pxyprodxy) const
{
// NB: code is repeated in GPR_Generic

    int res = cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodxy); // Use local version (difference is here)

    SparseVector<gentype> ximod(xa);
    SparseVector<gentype> xjmod(xb);

    ximod.f4("&",6) = ((int) ximod.f4(6))+1; // gradient is in addition to what is already there!
    xjmod.f4("&",6) = ((int) xjmod.f4(6))+1;

    int ui = ( u == -1 ) ? 1 : ximod.nupsize();
    int uj = ( u == -1 ) ? 1 : xjmod.nupsize();

    int uij = ( ui > uj ) ? ui : uj;

    int umin = ( u == -1 ) ? 0 : u;
    int umax = ( u == -1 ) ? uij-1 : u;

    for ( u = umin ; u <= umax ; ++u )
    {
        if ( xvar.nupsize(u) && ximod.nupsize(u) && xjmod.nupsize(u) )
        {
            // Gradient wrt xi

            ximod.f4("&",9) = u; // with respect to xu in [ x0 ~ x1 ~ ... ]
            xjmod.f4("&",9) = u;

            gentype gradiu;
            gentype gradju;

            res |= getQconst().gg(gradiu,ximod); // Note use of getQconst() here!
            res |= getQconst().gg(gradju,xjmod); // Note use of getQconst() here!

            // Assuming non-sparse data here!

            retVector<gentype> tmpva;

            const Vector<gentype> &gradiuv = gradiu.cast_vector();
            const Vector<gentype> &gradjuv = gradju.cast_vector();
            const Vector<gentype> &xvaru = xvar.nup(u)(tmpva);

            gentype tmp;

            resv += threeProduct(tmp,gradiuv,xvaru,gradjuv);
        }
    }

    return res;
}

int GPR_Generic::scale(double _sf)
{
    int res = 0;
    gentype sf(_sf);

    dy *= sf;
    res |= sety(dy);

    getKernel_unsafe() *= sf;
    res |= resetKernel(0);

    return res;
}

int GPR_Generic::gg(gentype &resg, const SparseVector<gentype> &x, const vecInfo *xinf, gentype ***pxyprodx) const
{
    if ( ( 3 == isSampleMode() ) || ( 4 == isSampleMode() ) || ( 5 == isSampleMode() ) )
    {
        gentype resv,resmu;
        var(resv,resmu,x,xinf,pxyprodx);
    }

    return getQconst().gg(resg,x,xinf,pxyprodx);
}

int GPR_Generic::hh(gentype &resh, const SparseVector<gentype> &x, const vecInfo *xinf, gentype ***pxyprodx) const
{
    if ( ( 3 == isSampleMode() ) || ( 4 == isSampleMode() ) || ( 5 == isSampleMode() ) )
    {
        gentype resv,resmu;
        var(resv,resmu,x,xinf,pxyprodx);
    }

    return getQconst().hh(resh,x,xinf,pxyprodx);
}

int GPR_Generic::gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg, const vecInfo *xinf, gentype ***pxyprodx) const
{
    if ( ( 3 == isSampleMode() ) || ( 4 == isSampleMode() ) || ( 5 == isSampleMode() ) )
    {
        gentype resv,resmu;
        var(resv,resmu,x,xinf,pxyprodx);
    }

    return getQconst().gh(resh,resg,x,retaltg,xinf,pxyprodx);
}

int GPR_Generic::cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf, const vecInfo *xbinf, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    int res = 0;

    if ( ( 3 == isSampleMode() ) || ( 4 == isSampleMode() ) || ( 5 == isSampleMode() ) )
    {
        gentype resmean;
        res = getQconst().cov(resv,resmean,xa,xb,xainf,xbinf,pxyprodi,pxyprodj,pxyprodij);

        // see comments in var function

        //double sigma_cut = sigma(); ///1000; //1.05*sigma(); //SIGMA_ADD;

//FIXME: this doesn't work for vectors, anions etc
        if ( ((double) resv) > sigma()*sigma_cut() )
        {
            // Not optimal but good enough

            var(resv,resmu,xa,xainf);
            var(resv,resmu,xb,xbinf);

            getQconst().gg(resmu,xa,xainf,pxyprodi);

            if ( 3 == isSampleMode() )
            {
                randnfill(resmu.force_double(),(double) resmean,sampleScale*sqrt((double) resv));
            }

            else if ( 4 == isSampleMode() )
            {
                resmu.force_double() = -1;

                while ( ((double) resmu) < 0 )
                {
                    randnfill(resmu.force_double(),(double) resmean,sampleScale*sqrt((double) resv));
                }
            }

            else if ( 5 == isSampleMode() )
            {
                resmu.force_double() = +1;

                while ( ((double) resmu) > 0 )
                {
                    randnfill(resmu.force_double(),(double) resmean,sampleScale*sqrt((double) resv));
                }
            }
        }

        else
        {
            resmu = resmean;
        }

        resv = 0.0;
    }

    else if ( isSampleMode() )
    {
        // Covariance is now zero by definition

        resv = 0.0;
        getQconst().gg(resmu,xa,xainf,pxyprodi);
    }

    else
    {
        res = getQconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodi,pxyprodj,pxyprodij);
    }

    return res;
}

int GPR_Generic::covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &xx) const
{
    int res = 0;

    if ( ( 3 == isSampleMode() ) || ( 4 == isSampleMode() ) || ( 5 == isSampleMode() ) )
    {
        int m = xx.size();
        int res = 0;

        gentype resvv,dummy,dummyb;

        // Sample along the diagonal

        for ( int ii = 0 ; ii < m ; ++ii )
        {
            res |= var(dummy,dummyb,xx(ii));
        }

        // Covariance is now zero by definition

        resv.resize(m,m) = ( dummy = 0.0 );
    }

    else if ( isSampleMode() )
    {
        // Covariance is now zero by definition

        gentype dummy;
        resv.resize(xx.size(),xx.size()) = ( dummy = 0.0 );
    }

    else
    {
        res = getQconst().covar(resv,xx);
    }

    return res;
}

int GPR_Generic::var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf, gentype ***pxyprodx, gentype **pxyprodxx) const
{
    int res = 0;

    if ( ( 3 == isSampleMode() ) || ( 4 == isSampleMode() ) || ( 5 == isSampleMode() ) )
    {
        gentype resmean;
        res = getQconst().var(resv,resmean,xa,xainf,pxyprodx,pxyprodxx);

        int dummy = 0;
        int killSwitch = 0;

        // Draw mu

            if ( 3 == isSampleMode() )
            {
                randnfill(resmu.force_double(),(double) resmean,sampleScale*sqrt((double) resv));
            }

            else if ( 4 == isSampleMode() )
            {
                resmu.force_double() = -1;

                while ( ((double) resmu) < 0 )
                {
                    randnfill(resmu.force_double(),(double) resmean,sampleScale*sqrt((double) resv));
                }
            }

            else if ( 5 == isSampleMode() )
            {
                resmu.force_double() = +1;

                while ( ((double) resmu) > 0 )
                {
                    randnfill(resmu.force_double(),(double) resmean,sampleScale*sqrt((double) resv));
                }
            }

//FIXME: this doesn't work for vectors, anions etc
        if ( ((double) resv) > sigma()*sigma_cut() )
        {
            GPR_Generic &that = const_cast<GPR_Generic &>(*this);
            int NN = that.N();

            // Add to training set
//FIXME: potential bug here using const_cast, c++ might not "see" the change!
            that.addTrainingVector(NN,resmu,xa,( ( (sigma()/SIGMA_ADD) > (1.0/sigma_cut()) ) ? (sigma()/SIGMA_ADD) : (1.0/sigma_cut()) ),0);
            that.train(dummy,killSwitch);
        }

        resv = 0.0;
    }

    else if ( isSampleMode() )
    {
        resv = 0.0;
        getQconst().gg(resmu,xa,xainf,pxyprodx);
    }

    else
    {
        res = getQconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx);
    }

    return res;
}

















int GPR_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    int k,res = 0;
    const char *dummy = nullptr;

    NiceAssert( xa.size() == xb.size() );

    val.resize(xa.size());

    for ( k = 0 ; k < xa.size() ; ++k )
    {
        res |= getparam(ind,val("&",k),xa(k),ia,xb(k),ib,dummy);
    }

    return res;
}

int GPR_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    int res = 0;

    desc = "";

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 2000: { val = muWeight();     desc = "GPR_Generic::muWeight"; break; }
        case 2001: { val = muBias();       desc = "GPR_Generic::muBias"; break; }
        case 2002: { val = isZeromuBias(); desc = "GPR_Generic::isZeromuBias"; break; }
        case 2003: { val = isVarmuBias();  desc = "GPR_Generic::isVarmuBias"; break; }
        case 2004: { val = isSampleMode(); desc = "GPR_Generic::isSampleMode"; break; }
        case 9070: { val = gprGp();        desc = "GPR_Generic::gprGp"; break; }

        default:
        {
            isfallback = 1;
            res = ML_Base::getparam(ind,val,xa,ia,xb,ib,desc);

            break;
        }
    }

    if ( ( ia || ib ) && !isfallback )
    {
        val.force_null();
    }

    return res;
}


















//NB ************* THIS IS IMPORTANT ************************
//
// For speed, we need a *consistent*, *linear* sampling for x.  The reason for this
// is that, in smboopt (sequential model-based optimisation) the model must
// compute inner products between sampled models, and, while that can be
// done autonomously using global functions, that is too slow.  However the y()
// vector effectively represents points at which g(x) has been (effectively)
// pre-computed, so if y() corresponds to the same x for two different GPRs
// then we can simply grab y() and use this, which saves a *lot* of time.
// This is why grids are generally preferable to random samples or JIT.

// Step size for hard constraints
#define INGSTEP 0.3
// Convergence for hard constraints
#define CONVPT 1e-5

#undef SIGMA_ADD
#define SIGMA_ADD 1e-3


int GPR_Generic::setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double sampScale, double sampSlack)
{
    //int debugit = 0; //1; // set 1 in gdb to turn on feedback

    int postType  = (sampType/100);
    int alphaType = (sampType%100)/10;
    int yType     = (sampType%10);

    setNaiveConst(); //NiceAssert( isNaiveConst() );

    sampSplit = sampSplit ? sampSplit : 1;

    NiceAssert( !nv || ( nv >= 3 ) || ( xmin.size() == xmax.size() ) );
    NiceAssert( !nv || ( nv >= 3 ) || ( sampSplit >= 0 ) );
    NiceAssert(        ( nv >= 3 ) || ( yType >= 0 ) );
    NiceAssert(        ( nv >= 3 ) || ( yType <= 5 ) );
    NiceAssert(        ( nv <  3 ) || !sampType );
    NiceAssert( alphaType >= 0 );
    NiceAssert( alphaType <= 2 );
    NiceAssert( postType >= 0 );
    NiceAssert( postType <= 3 );

    int i,j,k;

    // IMPORTANT: if this is already a sample then we actually DO NOT want to re-sample as this might
    // subtly change it, and our BO method (maybe) relies on the assumption that it does not get changed

    static thread_local Vector<int>    intype("&",2);
    static thread_local Vector<double> insx("&",2);
    static thread_local Vector<int>    indval("&",2);

    static thread_local GPR_Generic *prior = nullptr;

    static thread_local int Npresamp = 0;
    static thread_local int Ntotsamp = 0;
    static thread_local int Nnontriv = 0;
    static thread_local int Nhard    = 0;

    if ( ( nv == 0 ) || ( nv == 3 ) || ( nv == 4 ) || ( nv == 5 ) )
    {
        // No existing sampling, so need to initialise L etc
    }

    else if ( ( nv == 1 ) && ( sampleMode == 0 ) )
    {
        // No presampling done yet.  Our approach is to pre-sample x
        // and then do the actual Thompson sampling.

        setSampleMode(2,xmin,xmax,Nsamp,sampSplit,sampType,xsampType,sampScale,sampSlack); // presample
        setSampleMode(1,xmin,xmax,Nsamp,sampSplit,sampType,xsampType,sampScale,sampSlack); // actual sample
    }

    else if ( ( nv == 2 ) && ( sampleMode == 0 ) )
    {
        Npresamp = N();
        Nnontriv = 0;
        Nhard    = 0;

        intype.resize(0);
        insx.resize(0);
        indval.resize(0);

        if ( ( postType == 0 ) || ( postType == 1 ) )
        {
            // Non-trivial priors on g(x) are dealt with after post-draw

            for ( i = 0 ; i < Npresamp ; i++ )
            {
                gentype realy = y()(i); // right-side of original constraint

                if ( d()(i) == -1 )
                {
                    // -1 is a bound g(x) <= y

                    Nnontriv++;

                    intype.add(i); intype("&",i) = -1;
                    insx.add(i);   insx("&",i)   = realy;
                    indval.add(i); indval("&",i) = d()(i);
                }

                else if ( d()(i) == +1 )
                {
                    // -2 is a bound g(x) >= y

                    Nnontriv++;

                    intype.add(i); intype("&",i) = -2;
                    insx.add(i);   insx("&",i)   = realy;
                    indval.add(i); indval("&",i) = d()(i);
                }

                else
                {
                    intype.add(i); intype("&",i) = 0;
                    insx.add(i);   insx("&",i)   = realy;
                    indval.add(i); indval("&",i) = d()(i);
                }
            }
        }

        else
        {
            // All priors on g(x)^2 are dealt with after post-draw

            errstream() << "PreProcess constraints\n";

            for ( i = 0 ; i < Npresamp ; i++ )
            {
                int xitang     = (xtang()(i));
                gentype realy  = y()(i); // right-side of original constraint
                gentype effy   = y()(i); // effective right-side of constraint
                double Cweight = 1/sigmaweight()(i);

                if ( d()(i) )
                {
                    // loctype = 0: simple constraint on g(x)
                    //           1: constraint on dg(x)
                    //           2: constraint on g(x)-g(z)
                    //           3: constraint on dg(x)-g(z)
                    //           4: constraint on g(x)-dg(z)
                    //           5: constraint on dg(x)-dg(z)
                    //
                    // xtang options:
                    //
                    // 0:    xnear{info} points to x vector {info}, inear is index (xnear always set)
                    // 1:    xfar{info} and ifar refer to "other side" of rank constraint
                    // 2:    xfarfar refers to direction of gradient constraint, gradOrder > 0 *or* xfarfarfar refers to direction of gradient constraint, gradOrderR > 0
                    // 4:    gradOrder > 0, but xfarfar not set *or* gradOrderR > 0 but xfarfarfar not set
                    // 8:    treat distributions as sample from sets, allowing whole-set constraints
                    // 16:   idiagr set
                    // 32:   evaluate dense integral
                    // 64:   evaluate dense derivative
                    // 128:  xfarfar refers to direction of gradient constraint, gradOrder > 0
                    // 256:  gradOrder > 0, but xfarfar not set
                    // 512:  xfarfarfar refers to direction of gradient constraint, gradOrderR > 0
                    // 1024: gradOrderR > 0, but xfarfarfar not set
                    // 2048: actually a weighted sum of vectors
                    // 3,5,9,10,11,12,13,16,32,64: allowable combinations (128,256,512,1024 also follow if 2,4 set)
                    // NOT ANY MORE: -1: idiagr set (simple version only)

                    int loctype = 0;

                    SparseVector<gentype> realx = x()(i);

                    if ( xitang == 0 )
                    {
                        loctype = 0;
                    }

                    else if ( xitang == 2+128 )
                    {
                        loctype = 1;
                    }

                    else if ( xitang == 1 )
                    {
                        loctype = 2;
                    }

                    //else if ( xitang == 1+2+128 )
                    //{
                    //    loctype = 3;
                    //}

                    //else if ( xitang == 1+2+512 )
                    //{
                    //    loctype = 4;
                    //}

                    //else if ( xitang == 1+2+128+512 )
                    //{
                    //    loctype = 5;
                    //}

                    else
                    {
                        NiceThrow("Observation type not suppored when sampling g(x)^2 using postType 2xx,3xx");
                    }

                    if ( ( loctype == 0 ) && ( d()(i) == -1 ) )
                    {
                        Nnontriv++;

                        //          1 is a bound g(x)^2 <= y, cast as g(x) <=  sqrt(y)
                        //          1000                              g(x) >= -sqrt(y))

                        effy = sqrt((double) realy);

                        sety(i,effy);

                        intype.add(i); intype("&",i) = 1;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        effy = -sqrt((double) realy);

                        addTrainingVector(i,effy,realx,Cweight,0);
                        setd(i,+1);

                        intype.add(i); intype("&",i) = 1000;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 0 ) && ( d()(i) == 1 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //          2 is a bound g(x)^2 >= y, cast as gx.g(x) >= y

                        intype.add(i); intype("&",i) = 2;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx,Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 102;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 0 ) && ( d()(i) == 2 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //          3 is a bound g(x)^2 == y, cast as gx.g(x) == y

                        intype.add(i); intype("&",i) = 3;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx,Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 103;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 1 ) && ( d()(i) == -1 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //          11 is a bound dg(x)^2 <= y, cast as 2gx.dg(x) <= y

                        intype.add(i); intype("&",i) = 11;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.n(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 111;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx,Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 311;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 1 ) && ( d()(i) == +1 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //          12 is a bound dg(x)^2 >= y, cast as 2gx.dg(x) >= y

                        intype.add(i); intype("&",i) = 12;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.n(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 112;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx,Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 312;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 1 ) && ( d()(i) == 2 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //          13 is a bound dg(x)^2 == y, cast as 2gx.dg(x) == y

                        intype.add(i); intype("&",i) = 13;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.n(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 113;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx,Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 313;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 2 ) && ( d()(i) == -1 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //         21 is a bound g(x)^2-g(z)^2 <= y, cast as gx.g(x)-gz.g(z) <= y

                        intype.add(i); intype("&",i) = 21;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.n(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 121;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.f1(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 221;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 2 ) && ( d()(i) == 1 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //         22 is a bound g(x)^2-g(z)^2 >= y, cast as gx.g(x)-gz.g(z) >= y

                        intype.add(i); intype("&",i) = 22;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.n(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 122;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.f1(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 222;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }

                    else if ( ( loctype == 2 ) && ( d()(i) == 2 ) )
                    {
                        Nnontriv++;
                        Nhard++;

                        //         23 if a bound g(x)^2-g(z)^2 == y, cast as gx.g(x)-gz.g(z) == y

                        intype.add(i); intype("&",i) = 23;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.n(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 123;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);

                        i++;
                        Npresamp++;

                        addTrainingVector(i,realy,realx.f1(),Cweight,0);
                        setd(i,0);

                        intype.add(i); intype("&",i) = 223;
                        insx.add(i);   insx("&",i)   = realy;
                        indval.add(i); indval("&",i) = d()(i);
                    }
                }

                else
                {
                    intype.add(i); intype("&",i) = 0;
                    insx.add(i);   insx("&",i)   = realy;
                    indval.add(i); indval("&",i) = d()(i);
                }
            }
        }

        // Pre-sample by setting up grid, adding to model and working out partial Cholesky factorisation

        Vector<SparseVector<gentype> > xx;

        if ( Nsamp )
        {
            GPR_Generic::genSampleGrid(xx,xmin,xmax,Nsamp,sampSplit,xsampType,sampSlack);
        }

        else
        {
            // Auto case, note negated sampSplit bodge

            // This is a terrible hack

            int effdim = (int) std::sqrt((xmin.size())*(xmin.size()));
            int basemul = 200;

            if      ( effdim == 1 ) { basemul = 200; }
            else if ( effdim == 2 ) { basemul = 100; }
            else if ( effdim == 3 ) { basemul = 50;  }
            else if ( effdim == 4 ) { basemul = 25;  }
            else if ( effdim == 5 ) { basemul = 12;  }
            else                    { basemul = 10;  }

//            GPR_Generic::genSampleGrid(xx,xmin,xmax,-(10*(N() ? N() : 1)*(xmin.size())*(xmin.size())),sampSplit,xsampType,sampSlack);
            GPR_Generic::genSampleGrid(xx,xmin,xmax,basemul*(xmin.size())*(xmin.size()),sampSplit,xsampType,sampSlack);
        }

        Ntotsamp = xx.size();

        // Pre-emptively set up kernel now rather than later

        if ( postType )
        {
            getKernel_unsafe().setSymmSet();
            resetKernel();
        }

        // Add vectors to model, with d = 0 for now (do this preemptively to avoid double-calculation)
        //
        // NB: we set y = muBias() as a sort of minimal baseline to prevent numerical difficulties

        double sigweightval = ( ( (sigma()/SIGMA_ADD) > (1.0/sigma_cut()) ) ? (sigma()/SIGMA_ADD) : (1.0/sigma_cut()) );
        //double sigweightval = sigma()/SIGMA_ADD;

        for ( i = 0 ; i < Ntotsamp ; ++i )
        {
            qaddTrainingVector(Npresamp+i,muBias(),xx("&",i),sigweightval,0); // This destroys x(i), but that doesn't matter as we don't need it after this
            setd(Npresamp+i,0); // disable until after we have trained on the prior
        }

        // "Prior" is the post-correction function.  We want to set d=0 in *this
        // for all non-trivial priors, and d=0 in prior for all trivial priors
        // and sample data.

        prior = dynamic_cast<GPR_Generic *>(makeDupML(*this));

        for ( i = 0 ; i < Npresamp ; ++i )
        {
            if ( !intype(i) )
            {
                // Want h(x) = 0 for these points so g(x)+h(x) satisfies relevant constraint

                prior->setd(i,2);
                prior->sety(i,0.0_gent);
            }

            //else if ( ( intype(i) == -1 ) || ( intype(i) == -2 ) )
            //{
            //    // Enforce g(x) == y and h(x) >= 0 (or <= 0, depending)
            //
            //    setd(i,2);
            //    prior->sety(i,0.0_gent);
            //}

            else
            {
                // Non-trivial prior

                setd(i,0);
            }
        }

        // Both this and prior have d=0 for the grid, so no need to fix
        // (We will re-enable grid in *this later!)

        // Pre-train this to fix prior mean and variance for trivial constraints

        {
            int dummyres = 0;
            svmvolatile int killSwitch = 0;

            errstream() << "Pre-train for trivial priors\n";
            train(dummyres,killSwitch);
        }
    }

    else if ( ( nv == 1 ) && ( sampleMode == 2 ) )
    {
        // Generate random vector

        if ( yType <= 5 )
        {
            // Kept thread-local static for speed (no new/free, just resize as needed)

            static thread_local Matrix<double> vv; // We use these to avoid constant delete/alloc
            static thread_local Matrix<double> L;
            static thread_local Vector<double> m("&",2);
            static thread_local Vector<int>    p("&",2);
            static thread_local int            s;
            static thread_local Vector<double> rr("&",2);
            static thread_local Vector<double> yr("&",2);

            vv.resize(Ntotsamp,Ntotsamp);
            L.resize(Ntotsamp,Ntotsamp);
            m.resize(Ntotsamp);
            p.resize(Ntotsamp);
            rr.resize(Ntotsamp);
            yr.resize(Ntotsamp);

            s = 0;

            // Draw random vector

            for ( i = 0 ; i < Ntotsamp ; ++i )
            {
                randnfill(rr("&",i));
            }

            //if ( debugit ) { errstream() << "phantomxyz rr = " << rr << "\n"; }

            // Main sample loop

            retVector<int> tmpvb;
            retVector<int> tmpvc;
            retVector<double> tmpva;
            retVector<double> tmpvd;
            retMatrix<double> tmpma;

            gentype altvs,ms;

            {
                // Calculate prior mean/variance (recall that *this is currently only trained on the prior)

                vv = 0.0;
                m  = 0.0;

                for ( i = 0 ; i < Ntotsamp ; ++i )
                {
                    for ( j = 0 ; j <= i ; ++j )
                    {
                        covTrainingVector(altvs,ms,Npresamp+i,Npresamp+j);

                        vv("&",i,j) = ( (double) altvs ) + ( ( i == j ) ? SIGMA_ADD : 0.0 );

                        if ( i == j )
                        {
                            m("&",i) = (double) ms;
                        }
                    }
                }

                //if ( debugit ) { errstream() << "phantomxyz prior covariance = " << vv << "\n"; }
                //if ( debugit ) { errstream() << "phantomxyz prior mean = " << m << "\n"; }

                // Calculate targets

                vv.naivepartChol(L,p,s);

                yr("&",p,tmpva) = (L(p,p(0,1,s-1,tmpvb),tmpma))*rr(p(0,1,s-1,tmpvc),tmpvd); // Note that p includes all variables, so no need to pre-zero yr
                yr += m;

                //if ( debugit ) { errstream() << "phantomxyz yr = " << yr << "\n"; }

                // Process y if required

                if ( yType )
                {
                    if ( ( sampSplit == 1 ) || ( sampSplit == 0 ) )
                    {
                        if ( yType == 1 )
                        {
                            for ( i = 0 ; i < Ntotsamp ; ++i )
                            {
                                yr("&",i) = ( yr(i) >= 0.0 ) ? yr(i) : 0.0;
                            }
                        }

                        else if ( yType == 2 )
                        {
                            for ( i = 0 ; i < Ntotsamp ; ++i )
                            {
                                yr("&",i) = ( yr(i) >= 0.0 ) ? yr(i) : -yr(i);
                            }
                        }

                        else if ( yType == 3 )
                        {
                            for ( i = 0 ; i < Ntotsamp ; ++i )
                            {
                                yr("&",i) = ( yr(i) <= 0.0 ) ? yr(i) : 0.0;
                            }
                        }

                        else if ( yType == 4 )
                        {
                            for ( i = 0 ; i < Ntotsamp ; ++i )
                            {
                                yr("&",i) = ( yr(i) <= 0.0 ) ? yr(i) : -yr(i);
                            }
                        }
                    }

                    else if ( sampSplit == 2 )
                    {
                        int matsize = (int) sqrt(Ntotsamp);

                        Matrix<double> yreal(matsize,matsize);

                        // Convert vectorised matrix to actual matrix for projection
                        //
                        // Also symmetrise at the same time

                        for ( i = 0 ; i < matsize ; ++i )
                        {
                            for ( j = 0 ; j <= i ; ++j )
                            {
                                yreal("&",j,i) = ( yreal("&",i,j) = (yr(i+(j*matsize))+yr(j+(i*matsize)))/2 );
                            }
                        }

                        Matrix<double> yyreal(yreal);

                        if ( yType == 1 )
                        {
                            Vector<double> fv1;
                            Matrix<double> fv2;
                            Vector<double> fv3;

                            int ierr = yreal.projpsd(yyreal,fv1,fv2,fv3,0);

                            (void) ierr;

                            NiceAssert( !ierr );
                        }

                        else if ( yType == 2 )
                        {
                            Vector<double> fv1;
                            Matrix<double> fv2;
                            Vector<double> fv3;

                            int ierr = yreal.projpsd(yyreal,fv1,fv2,fv3,1);

                            (void) ierr;

                            NiceAssert( !ierr );
                        }

                        else if ( yType == 3 )
                        {
                            Vector<double> fv1;
                            Matrix<double> fv2;
                            Vector<double> fv3;

                            int ierr = yreal.projnsd(yyreal,fv1,fv2,fv3,0);

                            (void) ierr;

                            NiceAssert( !ierr );
                        }

                        else if ( yType == 4 )
                        {
                            Vector<double> fv1;
                            Matrix<double> fv2;
                            Vector<double> fv3;

                            int ierr = yreal.projnsd(yyreal,fv1,fv2,fv3,1);

                            (void) ierr;

                            NiceAssert( !ierr );
                        }

                        // Convert back to vectorised form

                        for ( i = 0 ; i < matsize ; ++i )
                        {
                            for ( j = 0 ; j < matsize ; ++j )
                            {
                                yr("&",i+(j*matsize)) = yyreal(i,j);
                            }
                        }
                    }
                }

                //if ( debugit ) { errstream() << "phantomxyz yr post-processing = " << yr << "\n"; }

                // Set target in *this and re-enable grid

                for ( i = 0 ; i < Ntotsamp ; ++i )
                {
                    gentype yrr(yr(i));
                    setd(Npresamp+i,2);
                    sety(Npresamp+i,yrr);
                }

                // Train to complete sample

                {
                    int dummy = 0;
                    svmvolatile int killSwitch = 0;

                    errstream() << "Train for Thompson sample\n";
                    train(dummy,killSwitch);
                }

                //if ( debugit ) { errstream() << "phantomxyz muWeight = " << muWeight() << "\n"; }
            }

            if ( Nnontriv )
            {
                static thread_local Vector<double> gx("&",2);
                static thread_local Vector<double> gz("&",2);

                gx.resize(Npresamp);
                gz.resize(Npresamp);

                gx = 0.0;
                gz = 0.0;

                static thread_local Vector<double> dgx("&",2);
                static thread_local Vector<double> dgz("&",2);

                dgx.resize(Npresamp);
                dgz.resize(Npresamp);

                dgx = 0.0;
                dgz = 0.0;

                // Deal with non-trivial priors by constructing a correcting function
                //
                // This is done by constructing h(x) so the g(x)+h(x) satisfies relevant constraints

                // Adjust targets and x-weights on h(x) constraints
                // To do this, we first fix the target value, then train, then jump-start hx and hz values
                //
                // Note that trivial constraints are forced to h(x)=0 here

                gentype yval;

                for ( i = 0 ; i < Npresamp ; i++ )
                {
                    if ( ( intype(i) == -1 ) || ( intype(i) == -2 ) )
                    {
                        ggTrainingVector(ms,i);

                        // -1 is a bound g(x)+h(x) <= y, recast as h(x) <= y-gx
                        // -2 is a bound g(x)+h(x) >= y, recast as h(x) >= y-gx

                        yval = insx(i)-((double) ms);
                        prior->sety(i,yval);
                    }

                    else if ( intype(i) == 1 )
                    {
                        ggTrainingVector(ms,i);

                        //         1 is a bound (g(x)+h(x))^2 <= y, recast as h(x) <=  sqrt(y)-gx
                        //         1000                                       h(x) >= -sqrt(y)-gx

                        yval = sqrt(insx(i))-((double) ms);
                        prior->sety(i,yval);
                        yval = -sqrt(insx(i))-((double) ms);
                        prior->sety(i+1,yval);
                    }

                    else if ( ( intype(i) == 2 ) || ( intype(i) == 3 ) )
                    {
                        ggTrainingVector(ms,i+1);
                        gx("&",i) = (double) ms;

                        //          2 is a bound (g(x)+h(x))^2 >= y, recast as (2gx+hx).h(x) >= y-gx^2
                        //          3 is a bound (g(x)+h(x))^2 == y, recast as (2gx+hx).h(x) == y-gx^2
                        //
                        // (so g(x)^2 + 2g(x).h(x) + h(x).h(x) >= y, so (2gx+hx).h(x) >= y-gx^2)

                        yval = insx(i)-(gx(i)*gx(i));
                        prior->sety(i,yval);
                        prior->setd(i,0); // disable for now
                    }

                    else if ( ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) )
                    {
//FIXME
                        ggTrainingVector(ms,i+1);
                        gx("&",i) = (double) ms;
                        ggTrainingVector(ms,i+2);
                        dgx("&",i) = (double) ms;

                        //          11 is a bound d(g(x)+h(x))^2 <= y, cast as 2(gx+hx).dh(x) <= y-2(gx+hx)dgx
                        //          12 is a bound d(g(x)+h(x))^2 >= y, cast as 2(gx+hx).dh(x) >= y-2(gx+hx)dgx
                        //          13 is a bound d(g(x)+h(x))^2 == y, cast as 2(gx+hx).dh(x) == y-2(gx+hx)dgx
                        //
                        // (so 2(g(x)+h(x))(dg(x)+dh(x)) <= y, so 2(gx+hx).dh(x) <= y-2(gx+hx)dgx)

                        yval = insx(i)-(2*gx(i)*dgx(i));
                        prior->sety(i,yval);
                        prior->setd(i,0); // disable for now
                    }

                    else if ( ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                    {
                        ggTrainingVector(ms,i+1);
                        gx("&",i) = (double) ms;
                        ggTrainingVector(ms,i+2);
                        gz("&",i) = (double) ms;

                        //         21 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 <= y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) <= y-gx^2+gz^2
                        //         22 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 >= y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) >= y-gx^2+gz^2
                        //         23 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 == y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) == y-gx^2+gz^2

                        yval = insx(i)-(gx(i)*gx(i))+(gz(i)*gz(i));
                        prior->sety(i,yval);
                        prior->setd(i,0); // disable for now
                    }
                }

                //if ( debugit ) { errstream() << "phantomxyz gx = " << gx << "\n"; }
                //if ( debugit ) { errstream() << "phantomxyz gz = " << gz << "\n"; }

                {
                    int dummy = 0;
                    svmvolatile int killSwitch = 0;

                    errstream() << "Pre-train correction function\n";
                    prior->train(dummy,killSwitch);
                }

                // Only proceed further if we have "hard" constraints

                if ( Nhard )
                {
                    // Grab local copy of prior x

                    retVector<SparseVector<gentype> > tmpvg;

                    Vector<SparseVector<gentype> > xloc = x()(0,1,Npresamp-1,tmpvg);

                    double yv,xr,zr;
                    double wx,wz;

                    static thread_local Vector<double> hx("&",2);
                    static thread_local Vector<double> hz("&",2);

                    hx.resize(Npresamp);
                    hz.resize(Npresamp);

                    hx = 0.0;
                    hz = 0.0;

                    static thread_local Vector<double> dhx("&",2);
                    static thread_local Vector<double> dhz("&",2);

                    dhx.resize(Npresamp);
                    dhz.resize(Npresamp);

                    dhx = 0.0;
                    dhz = 0.0;

                    // Evaluate initial values for h(x)

                    for ( i = 0 ; i < Npresamp ; i++ )
                    {
                        if ( ( intype(i) == 2 ) || ( intype(i) == 3 ) || ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) || ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                        {
                            prior->ggTrainingVector(ms,i+1);
                            hx("&",i) = (double) ms;
                        }

                        if ( ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) )
                        {
                            prior->ggTrainingVector(ms,i+2);
                            dhx("&",i) = (double) ms;
                        }

                        if ( ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                        {
                            prior->ggTrainingVector(ms,i+2);
                            hz("&",i) = (double) ms;
                        }
                    }

                    //if ( debugit ) { errstream() << "phantomxyz hx = " << hx << "\n"; }
                    //if ( debugit ) { errstream() << "phantomxyz hz = " << hz << "\n"; }

                    // Pre-set kernel weights to initial values

                    for ( i = 0 ; i < Npresamp ; i++ )
                    {
                        if ( ( intype(i) == 2 ) || ( intype(i) == 3 ) )
                        {
                            //          2 is a bound (g(x)+h(x))^2 >= y, recast as (2gx+hx).h(x) >= y-gx^2
                            //          3 is a bound (g(x)+h(x))^2 == y, recast as (2gx+hx).h(x) == y-gx^2

                            yv = insx(i)-(gx(i)*gx(i));
                            wx = (2*gx(i))+hx(i);

                            xr = fabs(wx);

                            if ( xr > 1 )
                            {
                                yval = yv/xr;
                                xloc("&",i).f4("&",13) = wx/xr;
                            }

                            else
                            {
                                yval = yv;
                                xloc("&",i).f4("&",13) = wx;
                            }

                            // We will update x,y in *this later!

                            prior->sety(i,yval);
                            prior->setx(i,xloc(i));
                            prior->setd(i,indval(i));
                        }

                        else if ( ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) )
                        {
                            //          11 is a bound d(g(x)+h(x))^2 <= y, cast as 2(gx+hx).dh(x) <= y-2(gx+hx)dgx
                            //          12 is a bound d(g(x)+h(x))^2 >= y, cast as 2(gx+hx).dh(x) >= y-2(gx+hx)dgx
                            //          13 is a bound d(g(x)+h(x))^2 == y, cast as 2(gx+hx).dh(x) == y-2(gx+hx)dgx

                            yv = insx(i)-(2*(gx(i)+hx(i))*dgx(i));
                            wx = 2*(gx(i)+hx(i));

                            xr = fabs(wx);

                            if ( xr > 1 )
                            {
                                yval = yv/xr;
                                xloc("&",i).f4("&",13) = wx/xr;
                            }

                            else
                            {
                                yval = yv;
                                xloc("&",i).f4("&",13) = wx;
                            }

                            // We will update x,y in *this later!

                            prior->sety(i,yval);
                            prior->setx(i,xloc(i));
                            prior->setd(i,indval(i));
                        }

                        else if ( ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                        {
                            //         21 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 <= y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) <= y-gx^2+gz^2
                            //         22 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 >= y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) >= y-gx^2+gz^2
                            //         23 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 == y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) == y-gx^2+gz^2

                            yv = insx(i)-(gx(i)*gx(i))+(gz(i)*gz(i));
                            wx = (2*gx(i))+hx(i);
                            wz = (2*gz(i))+hz(i);

                            xr = fabs(wx);
                            zr = fabs(wz);

                            if ( ( xr > 1 ) && ( xr > zr ) )
                            {
                                yval = yv/xr;
                                xloc("&",i).f4("&",13) = wx/xr;
                                xloc("&",i).f4("&",14) = wz/xr;
                            }

                            else if ( zr > 1 )
                            {
                                yval = yv/zr;
                                xloc("&",i).f4("&",13) = wx/zr;
                                xloc("&",i).f4("&",14) = wz/zr;
                            }

                            else
                            {
                                yval = yv;
                                xloc("&",i).f4("&",13) = wx;
                                xloc("&",i).f4("&",14) = wz;
                            }

                            // We will update x,y in *this later!

                            prior->sety(i,yval);
                            prior->setx(i,xloc(i));
                            prior->setd(i,indval(i));
                        }
                    }

                    // Storage previous hx,hz values

                    static thread_local Vector<double> hxprev("&",2);
                    static thread_local Vector<double> hzprev("&",2);

                    hxprev.resize(Npresamp);
                    hzprev.resize(Npresamp);

                    static thread_local Vector<double> hxstep("&",2);
                    static thread_local Vector<double> hzstep("&",2);

                    hxstep.resize(Npresamp);
                    hzstep.resize(Npresamp);

                    static thread_local Vector<double> dhxprev("&",2);
                    static thread_local Vector<double> dhzprev("&",2);

                    dhxprev.resize(Npresamp);
                    dhzprev.resize(Npresamp);

                    static thread_local Vector<double> dhxstep("&",2);
                    static thread_local Vector<double> dhzstep("&",2);

                    dhxstep.resize(Npresamp);
                    dhzstep.resize(Npresamp);

                    bool isConverged = false;

                    while ( !isConverged )
                    {
                        hxprev = hx;
                        hzprev = hz;

                        dhxprev = dhx;
                        dhzprev = dhz;

                        // Train prior

                        {
                            int dummy = 0;
                            svmvolatile int killSwitch = 0;

                            errstream() << "~";
                            prior->train(dummy,killSwitch);
                        }

                        // Calculate h(x)

                        for ( i = 0 ; i < Npresamp ; i++ )
                        {
                            if ( ( intype(i) == 2 ) || ( intype(i) == 3 ) || ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) || ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                            {
                                prior->ggTrainingVector(ms,i+1);
                                hx("&",i) = (double) ms;
                            }

                            if ( ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) )
                            {
                                prior->ggTrainingVector(ms,i+2);
                                dhx("&",i) = (double) ms;
                            }

                            if ( ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                            {
                                prior->ggTrainingVector(ms,i+2);
                                hz("&",i) = (double) ms;
                            }
                        }

                        //if ( debugit ) { errstream() << "phantomxyz hx = " << hx << "\n"; }
                        //if ( debugit ) { errstream() << "phantomxyz hz = " << hz << "\n"; }
                        //if ( debugit ) { Vector<double> ghx(hx); ghx += gx; ghx *= ghx; errstream() << "phantomxyz (hx+gx)^2 = " << ghx << "\n"; }
                        //if ( debugit ) { Vector<double> ghz(hz); ghz += gz; ghz *= ghz; errstream() << "phantomxyz (hz+gz)^2 = " << ghz << "\n"; }

                        // Calculate step

                        hxstep = hx;
                        hzstep = hz;

                        hxstep -= hxprev;
                        hzstep -= hzprev;

                        dhxstep = dhx;
                        dhzstep = dhz;

                        dhxstep -= dhxprev;
                        dhzstep -= dhzprev;

                        // Test if hx,hz changed significantly, set isConverged if not

                        isConverged = true;

                        for ( i = 0 ; i < Npresamp ; i++ )
                        {
                            if ( ( intype(i) == 2 ) || ( intype(i) == 3 ) || ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) || ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                            {
                                double prevmag = (hxprev(i)+gx(i))*(hxprev(i)+gx(i));
                                double nowmag = (hxprev(i)+hxstep(i)+gx(i))*(hxprev(i)+hxstep(i)+gx(i));
                                double locstep = fabs(prevmag-nowmag);

                                if ( prevmag > 1 )
                                {
                                    locstep /= prevmag;
                                }

                                if ( locstep > CONVPT )
                                {
                                    isConverged = false;
                                    break;
                                }
                            }

                            if ( ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) )
                            {
                                double prevmag = (dhxprev(i)+dgx(i))*(dhxprev(i)+dgx(i));
                                double nowmag = (dhxprev(i)+dhxstep(i)+dgx(i))*(dhxprev(i)+dhxstep(i)+dgx(i));
                                double locstep = fabs(prevmag-nowmag);

                                if ( prevmag > 1 )
                                {
                                    locstep /= prevmag;
                                }

                                if ( locstep > CONVPT )
                                {
                                    isConverged = false;
                                    break;
                                }
                            }

                            if ( ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                            {
                                double prevmag = (hzprev(i)+gz(i))*(hzprev(i)+gz(i));
                                double nowmag = (hzprev(i)+hzstep(i)+gz(i))*(hzprev(i)+hzstep(i)+gz(i));
                                double locstep = fabs(prevmag-nowmag);

                                if ( prevmag > 1 )
                                {
                                    locstep /= prevmag;
                                }

                                if ( locstep > CONVPT )
                                {
                                    isConverged = false;
                                    break;
                                }
                            }
                        }

                        // Scale step

                        hxstep *= INGSTEP;
                        hzstep *= INGSTEP;

                        dhxstep *= INGSTEP;
                        dhzstep *= INGSTEP;

                        if ( !isConverged )
                        {
                            hx = hxprev;
                            hz = hzprev;

                            hx += hxstep;
                            hz += hzstep;

                            dhx = dhxprev;
                            dhz = dhzprev;

                            dhx += dhxstep;
                            dhz += dhzstep;

                            // Update x weights

                            for ( i = 0 ; i < Npresamp ; i++ )
                            {
                                if ( ( intype(i) == 2 ) || ( intype(i) == 3 ) )
                                {
                                    //          2 is a bound (g(x)+h(x))^2 >= y, recast as (2gx+hx).h(x) >= y-gx^2
                                    //          3 is a bound (g(x)+h(x))^2 == y, recast as (2gx+hx).h(x) == y-gx^2

                                    yv = insx(i)-(gx(i)*gx(i));
                                    wx = (2*gx(i))+hx(i);

                                    xr = fabs(wx);

                                    if ( xr > 1 )
                                    {
                                        yval = yv/xr;
                                        xloc("&",i).f4("&",13) = wx/xr;
                                    }

                                    else
                                    {
                                        yval = yv;
                                        xloc("&",i).f4("&",13) = wx;
                                    }

                                    // We will update x,y in *this later!

                                    prior->sety(i,yval);
                                    prior->setx(i,xloc(i));
                                }

                                else if ( ( intype(i) == 11 ) || ( intype(i) == 12 ) || ( intype(i) == 13 ) )
                                {
                                    //          11 is a bound d(g(x)+h(x))^2 <= y, cast as 2(gx+hx).dh(x) <= y-2(gx+hx)dgx
                                    //          12 is a bound d(g(x)+h(x))^2 >= y, cast as 2(gx+hx).dh(x) >= y-2(gx+hx)dgx
                                    //          13 is a bound d(g(x)+h(x))^2 == y, cast as 2(gx+hx).dh(x) == y-2(gx+hx)dgx

                                    yv = insx(i)-(2*(gx(i)+hx(i))*dgx(i));
                                    wx = 2*(gx(i)+hx(i));

                                    xr = fabs(wx);

                                    if ( xr > 1 )
                                    {
                                        yval = yv/xr;
                                        xloc("&",i).f4("&",13) = wx/xr;
                                    }

                                    else
                                    {
                                        yval = yv;
                                        xloc("&",i).f4("&",13) = wx;
                                    }

                                    // We will update x,y in *this later!

                                    prior->sety(i,yval);
                                    prior->setx(i,xloc(i));
                                }

                                else if ( ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                                {
                                    //         21 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 <= y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) <= y-gx^2+gz^2
                                    //         22 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 >= y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) >= y-gx^2+gz^2
                                    //         23 is a bound (g(x)+h(x))^2-(g(z)+h(z))^2 == y, cast as (2gx+hx).h(x)-(2gz+hz).h(z) == y-gx^2+gz^2

                                    yv = insx(i)-(gx(i)*gx(i))+(gz(i)*gz(i));
                                    wx = (2*gx(i))+hx(i);
                                    wz = (2*gz(i))+hz(i);

                                    xr = fabs(wx);
                                    zr = fabs(wz);

                                    if ( ( xr > 1 ) && ( xr > zr ) )
                                    {
                                        yval = yv/xr;
                                        xloc("&",i).f4("&",13) = wx/xr;
                                        xloc("&",i).f4("&",14) = wz/xr;
                                    }

                                    else if ( zr > 1 )
                                    {
                                        yval = yv/zr;
                                        xloc("&",i).f4("&",13) = wx/zr;
                                        xloc("&",i).f4("&",14) = wz/zr;
                                    }

                                    else
                                    {
                                        yval = yv;
                                        xloc("&",i).f4("&",13) = wx;
                                        xloc("&",i).f4("&",14) = wz;
                                    }

                                    // We will update x,y in *this later!

                                    prior->sety(i,yval);
                                    prior->setx(i,xloc(i));
                                }
                            }
                        }
                    }

                    errstream() << "\n";
                }

                // Merge prior with *this

                static thread_local Vector<gentype> muwval("&",2);

                muwval.resize(Npresamp+Ntotsamp);

                for ( i = 0 ; i < Npresamp ; ++i )
                {
                    if ( intype(i) )
                    {
                        muwval("&",i) = prior->muWeight()(i);
                        setd(i,indval(i));
                    }

                    else if ( prior->d()(i) )
                    {
                        // Have to parts in this weight: g(x) in the original draw, and the forcing values to ensure h(x)=0 there

                        muwval("&",i) = (prior->muWeight()(i))+(muWeight()(i));
                    }

                    else
                    {
                        muwval("&",i) = muWeight()(i);
                    }
                }

                for ( i = 0 ; i < Ntotsamp ; ++i )
                {
                    muwval("&",Npresamp+i) = muWeight()(Npresamp+i);
                }

                for ( i = 0 ; i < Npresamp ; i++ )
                {
                    if ( ( intype(i) == 1 ) || ( intype(i) == -1 ) || ( intype(i) == -2 ) )
                    {
                        sety(i,prior->y()(i));
                    }

                    else if ( ( intype(i) == 2 ) || ( intype(i) == 3 ) || ( intype(i) == 21 ) || ( intype(i) == 22 ) || ( intype(i) == 23 ) )
                    {
                        sety(i,prior->y()(i));
                        setx(i,prior->x()(i));
                    }
                }

                setmuWeight(muwval);

                //if ( debugit )
                //{
                //    gentype mq;
                //
                //    for ( i = 0 ; i < Npresamp ; i++ )
                //    {
                //        ggTrainingVector(mq,i);
                //        errstream() << "phantomxyz pre g+h(" << i << ") = " << mq << "\t\t squared = " << (mq*mq) << "\n";
                //    }
                //
                //    for ( i = 0 ; i < Ntotsamp ; i++ )
                //    {
                //        ggTrainingVector(mq,i+Npresamp);
                //        errstream() << "phantomxyz tot g+h(" << i << ") = " << mq << "\t\t squared = " << (mq*mq) << "\n";
                //    }
                //}
                //if ( debugit ) { errstream() << "phantomxyz Gp(prior) = " << prior->gprGp() << "\n"; }
                //if ( debugit ) { errstream() << "phantomxyz Gp(this) = " << gprGp() << "\n"; }
                //if ( debugit ) { errstream() << "phantomxyz prior = " << *prior << "\n"; }
                //if ( debugit ) { errstream() << "phantomxyz *this = " << *this << "\n"; }
            }

            // Delete prior

            MEMDEL(prior);
            prior = nullptr;
        }

        else
        {
            NiceAssert( !postType );
            NiceAssert( !Npresamp );

            // Generate random alpha

            Vector<gentype> aa(muWeight());

            for ( i = 0 ; i < Ntotsamp ; i++ )
            {
                if ( yType == 6 )
                {
                    randnfill(aa("&",Npresamp+i).force_double(),0,1);
                }

                else if ( yType == 7 )
                {
                    randufill(aa("&",Npresamp+i).force_double(),-1,1);
                }
            }

            // Enable grid

            for ( i = 0 ; i < Ntotsamp ; ++i )
            {
                setd(Npresamp+i,2);
            }

            // Set alpha to rectified value

            setmuWeight(aa);

            // No need to worry about y as no post-training is allowed
        }

        // Rectify based on alphaType if required

        if ( ( alphaType == 1 ) || ( alphaType == 2 ) )
        {
            // Grab unrectified alpha

            Vector<gentype> aa(muWeight());

            // Rectify based on alphaType

            for ( i = 0 ; i < Npresamp+Ntotsamp ; i++ )
            {
                if ( ( ((double) aa(i)) < 0 ) && ( alphaType == 1 ) )
                {
                    setnegate(aa("&",i));
                }

                if ( ( ((double) aa(i)) > 0 ) && ( alphaType == 2 ) )
                {
                    setnegate(aa("&",i));
                }
            }

            // Set alpha to rectified value

            setmuWeight(aa);
        }

        // Post-processing, if required

        if ( ( postType == 1 ) || ( postType == 2 ) || ( postType == 3 ) )
        {
            // Pre-record alpha values

            static thread_local Vector<gentype> alphaxform("&",2);

            alphaxform.resize(Npresamp+Ntotsamp);

            for ( i = 0 ; i < Npresamp+Ntotsamp ; i++ )
            {
                alphaxform("&",i) = muWeight()(i);
            }

            // Begin by nominally separating rank-type constraints into pairs
            // of datapoints (one left, one right).  The target is meaningless
            // here, but that really doesn't matter as we don't train after this.
            // Also factor out kernel weights for simplicity.

            SparseVector<gentype> xL;
            SparseVector<gentype> xR;

            errstream() << "Split rank constraints\n";

            for ( i = Npresamp-1 ; i >= 0 ; i-- )
            {
                if ( d()(i) )
                {
                    if ( !( xtang()(i) & 1 ) )
                    {
                        xL = x()(i);

                        if ( xL.isf4indpresent(13) )
                        {
                            alphaxform("&",i) *= xL.f4(13);
                            xL.zerof4i(13);
                            setx(i,xL);
                        }

                        setd(i,2);
                    }

                    else
                    {
                        alphaxform.add(i+1);
                        alphaxform("&",i+1) = -alphaxform(i); // NB negation is important, this is a rank constraint!

                        xL = x()(i).n();
                        xR = x()(i).f1();

                        if ( x()(i).isf4indpresent(13) ) { alphaxform("&",i)   *= x()(i).f4(13); }
                        if ( x()(i).isf4indpresent(14) ) { alphaxform("&",i+1) *= x()(i).f4(14); }

                        setx(i,xL);
                        setd(i,2);

                        Npresamp++;

                        addTrainingVector(i+1,y()(i),xR,1/sigmaweight()(i),0);
                        //setd(i,2);
                    }
                }
            }

            //if ( debugit )
            //{
            //    setmuWeight(alphaxform);
            //
            //    gentype mq;
            //
            //    for ( i = 0 ; i < Npresamp ; i++ )
            //    {
            //        ggTrainingVector(mq,i);
            //        errstream() << "phantomxyz pre g+h(" << i << ") = " << mq << "\t\t squared = " << (mq*mq) << "\n";
            //    }
            //
            //    for ( i = 0 ; i < Ntotsamp ; i++ )
            //    {
            //        ggTrainingVector(mq,i+Npresamp);
            //        errstream() << "phantomxyz tot g+h(" << i << ") = " << mq << "\t\t squared = " << (mq*mq) << "\n";
            //    }
            //}
            //if ( debugit ) { errstream() << "phantomxyz split this = " << *this << "\n"; }
            //if ( debugit ) { errstream() << "muWeight 2 = " << muWeight() << "\n"; }

            if ( ( postType == 1 ) || ( postType == 2 ) )
            {
                // Actually we're sampling g(x), then recasting as g(x)^2 using a kernel trick

                int Npostsamp = (Npresamp+Ntotsamp)*(Npresamp+Ntotsamp);
                int offstart = Npresamp+Ntotsamp;

                double ccweight = ( ( (sigma()/SIGMA_ADD) > (1.0/sigma_cut()) ) ? (sigma()/SIGMA_ADD) : (1.0/sigma_cut()) );
                double eeweight = 0.0;

                errstream() << "Generate and add doubled dataset\n";

                SparseVector<gentype> xxx;

                for ( i = 0, k = 0 ; i < Npresamp+Ntotsamp ; ++i )
                {
                    for ( j = 0 ; j < Npresamp+Ntotsamp ; ++j, ++k )
                    {
                        xxx.zero();
                        xxx.overwriten(x(i),0);
                        xxx.overwriten(x(j),1);
                        //FIXME - what about gradients?

                        qaddTrainingVector(offstart+k,muBias(),xxx,ccweight,eeweight); // use qadd, we don't need to preserve xxx
                    }
                }

                // Square the function by kronprodding alpha into NpostSamp

                //if ( debugit ) { errstream() << "phantomxyz Qd << " << getQ().d() << "\n"; }
                //if ( debugit ) { errstream() << "phantomxyz Qy << " << getQ().y() << "\n"; }
                //if ( debugit ) { errstream() << "phantomxyz Qalpha << " << getQ().alphaVal() << "\n"; }

                errstream() << "Calculating doubled alpha weights\n";

                alphaxform.addpad(alphaxform.size(),Npostsamp);

                retVector<gentype> tmpve;
                retVector<gentype> tmpvf;

                kronpow(alphaxform("&",offstart,1,offstart+Npostsamp-1,tmpve),alphaxform(0,1,offstart-1,tmpvf),2);
                alphaxform("&",0,1,offstart-1,tmpve) = 0.0_gent;

                //if ( debugit ) { errstream() << "muWeight 5 = " << muWeight() << "\n"; }
            }

            setmuWeight(alphaxform);

            //if ( debugit ) { errstream() << "muWeight 6 = " << muWeight() << "\n"; }

            //if ( debugit )
            //{
            //    gentype mq;
            //
            //    for ( i = 0 ; i < Npresamp ; i++ )
            //    {
            //        ggTrainingVector(mq,i);
            //        errstream() << "phantomxyz pre g+h(" << i << ") = " << mq << "\t\t squared = " << (mq*mq) << "\n";
            //    }
            //
            //    for ( i = 0 ; i < Ntotsamp ; i++ )
            //    {
            //        ggTrainingVector(mq,i+Npresamp);
            //        errstream() << "phantomxyz tot g+h(" << i << ") = " << mq << "\t\t squared = " << (mq*mq) << "\n";
            //    }
            //}

            errstream() << "Done generating sample\n";
        }

        // Lock to prevent re-training (which will either fail and mess up the sample or, in best case, just waste time)

        isLocked = 1;
    }

    sampleMode  = nv;
    sampleScale = sampScale;

    return sampleMode | ML_Base::setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType,xsampType,sampScale,sampSlack);
}

// Nsamp:     number of samples
// sampSplit: sample split
// xsampType: method for generating x sample
//            0 - "true" pseudo-random
//            1 - pre-defined sequence of random samples, generated sequentially
//            2 - pre-defined sequence of random samples, same everytime
//            3 - grid of Nsamp^dim samples

int GPR_Generic::genSampleGrid(Vector<SparseVector<gentype> > &res, const Vector<gentype> &xqmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int xsampType, double sampSlack)
{
    int dim = xqmin.size();
    int allowGridSample  = ( xsampType == 3 ) ? 1 : 0; // 0 is implicitly random
    int isautorand       = ( xsampType == 1 ) ? 0 : 1;
    int enabletruerandom = ( xsampType == 0 ) ? 1 : 0;

//    Nsamp     = ( Nsamp     > 0 ) ? Nsamp     : -Nsamp;
//    sampSplit = ( sampSplit > 0 ) ? sampSplit : ( ( sampSplit < 0 ) ? -sampSplit : 1 );

    NiceAssert( xqmin.size() == xmax.size() );
    NiceAssert( Nsamp > 0 );
    NiceAssert( sampSplit > 0 );
    NiceAssert( !(dim%sampSplit) );

    Vector<gentype> xamin(xmax);
    xamin.negate();
    xamin.scale(sampSlack);

    const Vector<gentype> *xbmin = ( ( sampSlack == 0 ) ? &xqmin : &xamin );
    const Vector<gentype> &xmin = *xbmin;

/*
    if ( allowGridSample || enabletruerandom )
    {
        // In this case we want to include "unset" training data in the set of seed points.
        // Note that we can't include these in the pseudo-random case as we want comparability.

        int oldNsamp = Nsamp;

        Nsamp = ( Nsamp >= (NNC(0)) ) ? (Nsamp-NNC(0)) : 0;
        Nsamp = ( Nsamp >= (oldNsamp/2) ) ? Nsamp : (oldNsamp/2); // still want at least half generated randomly
    }
*/

    int totsamp = allowGridSample ? ( dim ? (int) pow(Nsamp,dim) : 0 ) : Nsamp;
    int sampPerSplit = dim/sampSplit;

    gentype xxmin(xmin);
    gentype xxmax(xmax);

    // Take uniform samples on res

    int i,jj,kk;

    res.resize(totsamp);

    if ( allowGridSample && ( dim == 1 ) )
    {
        for ( jj = 0 ; jj < totsamp ; ++jj )
        {
            SparseVector<gentype> &xq = res("&",jj);

            xq("&",0)  = (((double) jj)+0.5)/(((double) Nsamp));
            xq("&",0) *= (xmax(0)-xmin(0));
            xq("&",0) += xmin(0);
        }

/*
        // Also make sure we include "unset" training data in seeds

        if ( NNC(0) )
        {
            for ( jj = 0 ; jj < N() ; ++jj )
            {
                if ( !d()(jj) )
                {
                    res.add(res.size());
                    res("&",res.size()-1) = x(jj);

                    ++totsamp;
                }
            }
        }
*/
    }

    else if ( allowGridSample )
    {
        Vector<int> jjj(dim);

        jjj = 0;

        for ( jj = 0 ; jj < totsamp ; ++jj )
        {
            SparseVector<gentype> &xq = res("&",jj);

            for ( kk = 0 ; kk < dim ; ++kk )
            {
                xq("&",kk%sampPerSplit,kk/sampPerSplit)  = (((double) jjj(kk))+0.5)/(((double) Nsamp));
                xq("&",kk%sampPerSplit,kk/sampPerSplit) *= (xmax(kk)-xmin(kk));
                xq("&",kk%sampPerSplit,kk/sampPerSplit) += xmin(kk);
            }

            for ( kk = 0 ; kk < dim ; ++kk )
            {
                ++(jjj("&",kk));

                if ( jjj(kk) >= Nsamp )
                {
                    jjj("&",kk) = 0;
                }

                else
                {
                    break;
                }
            }
        }

/*
        // Also make sure we include "unset" training data in seeds

        if ( NNC(0) )
        {
            for ( jj = 0 ; jj < N() ; ++jj )
            {
                if ( !d()(jj) )
                {
                    res.add(res.size());
                    res("&",res.size()-1) = x(jj);

                    ++totsamp;
                }
            }
        }
*/
    }

    else if ( enabletruerandom )
    {
        for ( jj = 0 ; jj < totsamp ; ++jj )
        {
            SparseVector<gentype> &xq = res("&",jj);

            for ( kk = 0 ; kk < dim ; ++kk )
            {
                xq("&",kk%sampPerSplit,kk/sampPerSplit)  = ((double) (rand()%INT_MAX))/INT_MAX;
                xq("&",kk%sampPerSplit,kk/sampPerSplit) *= (xmax(kk)-xmin(kk));
                xq("&",kk%sampPerSplit,kk/sampPerSplit) += xmin(kk);
            }
        }

/*
        // Also make sure we include "unset" training data in seeds

        if ( NNC(0) )
        {
            for ( jj = 0 ; jj < N() ; ++jj )
            {
                if ( !d()(jj) )
                {
                    res.add(res.size());
                    res("&",res.size()-1) = x(jj);

                    ++totsamp;
                }
            }
        }
*/
    }

    else
    {
        // We want the *same* random samples (x grid) every time for easy comparison
        // between different draws.  We also don't want to disturb our actually
        // random draws (y).  So we just use a pre-generated set of 10000 "random"
        // (uniform) values.  Code used to generate this is included.

/*
#include <stdlib.h>
#include <iostream>
#include <climits>

int main()
{
    int i;

    for ( i = 0 ; i < 10000 ; i++ )
    {
        std::cout.precision(15);
        std::cout << ((double) (rand()%INT_MAX))/INT_MAX << ",\n";
    }

    return 0;
}
*/

        static const double xrandvals[10000] = {
0.84018771715471,
0.394382926819093,
0.783099223758606,
0.798440033476073,
0.911647357936784,
0.197551369293384,
0.335222755714889,
0.768229594811904,
0.277774710803188,
0.553969955795431,
0.47739705186216,
0.628870924761924,
0.364784472791843,
0.513400910195616,
0.952229725174713,
0.916195068003701,
0.635711727959901,
0.717296929432683,
0.141602555355803,
0.606968876257059,
0.0163005716243296,
0.242886770629737,
0.137231576786019,
0.80417675422699,
0.156679089254085,
0.400944394246183,
0.129790446781456,
0.108808802025769,
0.998924518003559,
0.218256905310907,
0.512932394404398,
0.839112234692607,
0.612639832595661,
0.296031617697343,
0.637552267703019,
0.524287190066784,
0.493582986990727,
0.97277502388357,
0.292516784413027,
0.771357697793915,
0.526744979213339,
0.769913836275187,
0.400228622090178,
0.891529452005182,
0.283314746005142,
0.352458347264891,
0.807724520008883,
0.919026473965042,
0.0697552762319126,
0.949327075364686,
0.525995350222101,
0.0860558478562421,
0.192213845994423,
0.66322692700812,
0.890232602548894,
0.348892935248508,
0.0641713207886421,
0.0200230488646883,
0.457701737274277,
0.0630958383265398,
0.238279954175595,
0.970634131678675,
0.902208073484808,
0.850919786771256,
0.266665749376018,
0.539760340722166,
0.375206976372379,
0.760248736366745,
0.512535364140074,
0.667723760785406,
0.53160643416066,
0.0392803433534132,
0.437637596594932,
0.931835056250838,
0.930809795358595,
0.720952343065735,
0.284293403050068,
0.738534314901817,
0.639978816565116,
0.354048679747641,
0.687861390266503,
0.165974166321556,
0.440104527603884,
0.880075236260926,
0.829201093329676,
0.330337129687116,
0.228968171043772,
0.893372414583979,
0.350360178551804,
0.686669908318049,
0.956468252910519,
0.588640133193061,
0.657304039531063,
0.858676325929666,
0.439559919498656,
0.923969788907082,
0.398436666651832,
0.814766896336697,
0.684218525273827,
0.910972030791907,
0.482490656656442,
0.215824958968826,
0.95025237414532,
0.920128253717035,
0.147660014754003,
0.881062169503915,
0.641080596317109,
0.431953418269732,
0.619596483940071,
0.281059412416564,
0.786002098017373,
0.307457873740912,
0.447033579203781,
0.226106625155595,
0.187533109536177,
0.276234672067796,
0.556443755308373,
0.416501280579949,
0.169607086186114,
0.906803933860177,
0.103171188432337,
0.126075339096633,
0.495444066587577,
0.760475228429062,
0.9847516650263,
0.935003986551894,
0.684445016870482,
0.38318833121247,
0.749770882422929,
0.368663541678648,
0.294160362004377,
0.232261538613709,
0.584488500647474,
0.244412735684036,
0.152389791865083,
0.732148515867138,
0.12547490472229,
0.793470388182192,
0.164101933671209,
0.745071389128022,
0.0745298005987563,
0.950104031688582,
0.0525292624032727,
0.521563379802538,
0.176210656378516,
0.240062372405111,
0.797798051870334,
0.732654411686889,
0.65656365298506,
0.96740513852211,
0.639458345547066,
0.759734841883059,
0.0934804771530817,
0.134902411668982,
0.52021006984646,
0.0782321417137199,
0.0699063977552142,
0.204655086251281,
0.461420473391852,
0.819677280178143,
0.57331862839559,
0.755580835396229,
0.0519388187918527,
0.157807128577403,
0.999993571080264,
0.204328610656936,
0.889955644444542,
0.125468475802554,
0.99779899930479,
0.0540575776500896,
0.870539864930576,
0.0723287994378846,
0.00416160887301043,
0.923069127333848,
0.593892179240422,
0.180372265717188,
0.163131499273298,
0.391690230645095,
0.913026677404077,
0.81969515272402,
0.359095368701543,
0.552485022485482,
0.579429994141418,
0.452575845854625,
0.687387434620125,
0.099640063522216,
0.530807988034006,
0.757293832375339,
0.304295149773497,
0.992228461425858,
0.576971112553482,
0.877613778169087,
0.747809296356425,
0.628909931345335,
0.0354209067464904,
0.747802866971028,
0.833238542002271,
0.925376551191032,
0.873271342773582,
0.831037540841399,
0.979434129306783,
0.743811207238497,
0.903366340279284,
0.983595738179794,
0.666880334572345,
0.497258519054045,
0.16396800343132,
0.830011833845643,
0.888948750164802,
0.0769946808353973,
0.649706986104002,
0.248044118400684,
0.629479703320879,
0.229136979779758,
0.70061996472097,
0.316867137475343,
0.328777043301974,
0.231427952289315,
0.0741609698506822,
0.633072193541132,
0.223656413249511,
0.651132082404165,
0.510685971244558,
0.971465709605937,
0.280042013283839,
0.546106877991048,
0.719268576111304,
0.113280554820448,
0.47148342918208,
0.592539918884886,
0.944318096127509,
0.450917558023202,
0.336351125657722,
0.847684435941132,
0.434513295737334,
0.00323145976440583,
0.344942954995177,
0.598481299634316,
0.83324329407571,
0.233891704694318,
0.675475980469713,
0.482950279714051,
0.481935823095001,
0.304955683324931,
0.71208725995947,
0.18255578735031,
0.621822821265935,
0.0408643027957828,
0.413983739639625,
0.695983791116618,
0.673936496336915,
0.637640152889136,
0.347115873055121,
0.184622467581473,
0.609105862029412,
0.627157886338959,
0.730729345572521,
0.328374438140716,
0.740438441159408,
0.20221277428894,
0.920914357025602,
0.684756536821256,
0.653130332312142,
0.257265482217663,
0.532440972762388,
0.0876436280494759,
0.26049694244773,
0.877383927757565,
0.686124927683792,
0.0937402360577789,
0.111275631986221,
0.361600907687843,
0.57669051577183,
0.593211455081222,
0.666556591012774,
0.288777775265639,
0.775767242897194,
0.288379411813048,
0.329642078527083,
0.189750982071157,
0.984363202929666,
0.00357857439833627,
0.827391135425955,
0.331479075519125,
0.188201041979809,
0.436496996989705,
0.958636961858085,
0.918930388017991,
0.764871435130421,
0.699075403017493,
0.121143161841269,
0.685785791690362,
0.383831939373087,
0.774273494619072,
0.943051274373686,
0.916272912135475,
0.861917122668548,
0.203548216355754,
0.793656839427378,
0.548042049886678,
0.297288452413533,
0.9049324714136,
0.909642957574522,
0.873978968185363,
0.498143926494822,
0.576199548121635,
0.162756743451001,
0.273911168926354,
0.864578960400344,
0.492398821978084,
0.463662150997511,
0.848942162864349,
0.49597739637642,
0.291053285957805,
0.180421238383474,
0.684178438821891,
0.727550283413171,
0.139058199775898,
0.60310882637422,
0.492421718077931,
0.838133602793391,
0.724251988215489,
0.178207509768292,
0.221965542166478,
0.4985254823689,
0.121258783676316,
0.138238453836291,
0.360442604571787,
0.324807000032071,
0.931895293729331,
0.908484654458465,
0.622095452445604,
0.836827764677269,
0.818127611567326,
0.496074420165305,
0.33497169070643,
0.39432715968896,
0.658831163616307,
0.608882859632784,
0.258906119623643,
0.15122998512873,
0.0725450106302952,
0.107848282022331,
0.647207381970811,
0.3635982965881,
0.288269520405805,
0.331385820327041,
0.0911485795356094,
0.427327720647365,
0.934494646701261,
0.583570298079201,
0.265461322975094,
0.658746634451089,
0.761777807847493,
0.487426865141572,
0.157272116819989,
0.88303659152381,
0.625665319443525,
0.517714721391776,
0.207843591090219,
0.557560612707194,
0.426199375850241,
0.829939043535823,
0.394388376918802,
0.244326986951906,
0.326013463235466,
0.729360068090893,
0.638654146640866,
0.984844627317434,
0.338242927258016,
0.89756026626451,
0.136074611980503,
0.410787937888311,
0.00540854828684057,
0.783281993951314,
0.774386234942072,
0.293678068692646,
0.114667813812693,
0.865534814477682,
0.721005789340011,
0.049162460048293,
0.449105112091221,
0.986467112780766,
0.707909094965043,
0.210882919473053,
0.473893977456677,
0.865181211785032,
0.0939195105312017,
0.09955929643454,
0.382895932711147,
0.301763101621421,
0.657119909141734,
0.809095308561388,
0.131702144691582,
0.0515082860605364,
0.0534222955132938,
0.45771560839271,
0.78086835415143,
0.69207644215416,
0.442560235244483,
0.119111280943785,
0.58963670841867,
0.578634847224986,
0.529899219297757,
0.59504525670551,
0.3619168411763,
0.304285453774168,
0.888723325398156,
0.476584654988993,
0.169820267786188,
0.609729114738167,
0.525747115502948,
0.618925380343071,
0.596196227053272,
0.233656210002329,
0.829808299816124,
0.0700902040442872,
0.0988374213217001,
0.923727810812987,
0.169649500944488,
0.481733354032847,
0.225490911968747,
0.826769410086223,
0.290828662594235,
0.35719305712599,
0.878277696146759,
0.344250958107529,
0.8149086655187,
0.659146049832528,
0.0363273997960274,
0.257468900297521,
0.778257331241973,
0.625964108214697,
0.836103747522507,
0.308156550074069,
0.221009364454546,
0.198020588233145,
0.612442003848237,
0.109732689852702,
0.6746052436878,
0.782262271634425,
0.719461804590869,
0.200352358725086,
0.401187651511835,
0.315658031178479,
0.434008568727415,
0.230995951327959,
0.385748235222766,
0.532845990049115,
0.154723761675285,
0.555397736167255,
0.0145793436163009,
0.380214673644032,
0.382167146253477,
0.305408006210536,
0.737407730770022,
0.260444841934575,
0.649658964318064,
0.55231639582306,
0.919590892232764,
0.685986364579753,
0.809785296120581,
0.697848223009076,
0.311950472328789,
0.645889043643088,
0.00600477261748387,
0.532959837248996,
0.843909631876233,
0.618446776465721,
0.642692527101698,
0.518514875098371,
0.400709047634485,
0.362154331226905,
0.718867233823457,
0.801896699611981,
0.677812362405384,
0.152875802085211,
0.0328926504742786,
0.0635605971624891,
0.685721792134327,
0.187616412149564,
0.618958333795405,
0.700301136216289,
0.567831085793595,
0.00112547958322125,
0.00570914196116344,
0.305238816563617,
0.261570321517796,
0.655368106744889,
0.857555212386677,
0.181161213284899,
0.341354470858981,
0.667340508507258,
0.879009436293975,
0.65330494318777,
0.313229551684684,
0.885014208911459,
0.186264779971104,
0.157139183095255,
0.503460984911519,
0.828957307072802,
0.675654058659288,
0.904170033011665,
0.191111637834046,
0.394521292017084,
0.706066732157984,
0.86892400023943,
0.547397094567957,
0.738959382632263,
0.93248459786758,
0.233118886236622,
0.926575794781826,
0.551442931197324,
0.933420022452911,
0.49440688010976,
0.552568410780546,
0.939129164879736,
0.799645696673377,
0.814138732298342,
0.594497271158964,
0.657200909060054,
0.995299946048902,
0.935851742017945,
0.32454141710165,
0.874309381877216,
0.589156684740054,
0.637770968786334,
0.759323590788675,
0.77542146517682,
0.79491015188159,
0.262784575234533,
0.60437877178396,
0.470564210075216,
0.166954607780536,
0.795490410083668,
0.865085502557962,
0.87302133993852,
0.664414409857436,
0.412482596660258,
0.611980722105122,
0.596899007259355,
0.64560148289688,
0.538556516421287,
0.148341937991018,
0.579021505349791,
0.032963396531047,
0.700910348771564,
0.518150669763866,
0.832609093204424,
0.515049081069906,
0.112647940457169,
0.489810001798817,
0.510349026653147,
0.0484996820094528,
0.814351418900467,
0.384658408064702,
0.637656366749507,
0.45212238722114,
0.143981998387716,
0.413077831460665,
0.24703253910273,
0.40676657408791,
0.0174566032446253,
0.717596749177946,
0.573721181868446,
0.812947013328293,
0.582682251270247,
0.446742521806966,
0.477361422720068,
0.995164847930504,
0.0587232434464261,
0.0742604299794233,
0.640766330827384,
0.597279760333374,
0.222602367970442,
0.219787835711515,
0.630243156864421,
0.923512717207667,
0.737938505475381,
0.462852249603184,
0.438561797811911,
0.85058644593255,
0.952662251867662,
0.948910824465058,
0.899086127942002,
0.767013670302468,
0.333569232064099,
0.536742494225848,
0.219136057523608,
0.477551230917476,
0.949820326152174,
0.466168596626338,
0.884317805005385,
0.967276929396799,
0.183765345338623,
0.45803898640817,
0.780223942259431,
0.766447597074531,
0.904781508215136,
0.257585364513837,
0.761612444539374,
0.963504752127223,
0.331845794493261,
0.402378774901097,
0.560784511994936,
0.554448162929364,
0.622166610612612,
0.191027668393696,
0.477960879671369,
0.360105115622331,
0.65387991799688,
0.916522677483281,
0.210691561089219,
0.606542169398881,
0.865433501482677,
0.10977768856556,
0.373555839701349,
0.199002733081115,
0.64652018325707,
0.592691897224957,
0.676553963998591,
0.596340508943582,
0.0588604933856337,
0.560871768538315,
0.56361743787472,
0.242625838724256,
0.0189107549464846,
0.34384137966849,
0.00907343533312596,
0.92369226316162,
0.601426744182327,
0.770685880338161,
0.887197014823182,
0.933272539141249,
0.173064654773597,
0.447981526352457,
0.487720701604952,
0.79523126585187,
0.639009194746152,
0.965681581276321,
0.15533638100854,
0.292889112743032,
0.88220425829394,
0.36602794256342,
0.899431282141912,
0.747637759310956,
0.475805631128981,
0.2729871213776,
0.946640492392071,
0.122325813920389,
0.865679018602557,
0.623194456390661,
0.718666322863971,
0.924539511988191,
0.184066224463315,
0.28228376027303,
0.167165350712447,
0.202976979409799,
0.62612513994152,
0.176238786045573,
0.12666924257142,
0.227551884123847,
0.946924666383734,
0.0138662569289404,
0.160824422799435,
0.119989321157331,
0.461847783747058,
0.648545124404386,
0.915220587009201,
0.100856978027549,
0.614226705215046,
0.0705569675520793,
0.393746090770581,
0.496430963043324,
0.436584910115499,
0.293177372912493,
0.244068721888619,
0.912390541710141,
0.566164494290093,
0.190709213815028,
0.0347163551648689,
0.43184351289265,
0.81390367020569,
0.753382678494501,
0.35638302441518,
0.997969895134666,
0.0356664383018699,
0.523548375127627,
0.200946874078804,
0.661791578709051,
0.699787161638861,
0.327616116650224,
0.889343462832898,
0.646711827556934,
0.341482374044826,
0.0501678851666711,
0.766701148714265,
0.803330157791884,
0.698713009571057,
0.681921735257805,
0.904187135819433,
0.312939714320442,
0.752478703275546,
0.297933226590014,
0.809370677363766,
0.189063612925384,
0.591110599502507,
0.0534393987867233,
0.101454154169864,
0.157275093326939,
0.244148612601752,
0.136170509800394,
0.589118606219589,
0.0580522828074416,
0.889553188294895,
0.94550163110043,
0.0560221774764462,
0.925219626596765,
0.469050005762395,
0.25696905155525,
0.587011204840155,
0.168837166935595,
0.584585168671135,
0.476354667207391,
0.815548994958191,
0.926067542715961,
0.526522552374062,
0.582250143206795,
0.729397700042183,
0.225235561479458,
0.2641718784646,
0.633584835395955,
0.5381752757999,
0.0166505812744846,
0.931518061985969,
0.347545952698004,
0.205714194199869,
0.522628661022814,
0.400985351484728,
0.307168348835394,
0.679903754815414,
0.645133964552141,
0.443338858635788,
0.269022360569342,
0.703186247359582,
0.332892046465023,
0.214523991204111,
0.759208424836029,
0.258111672596127,
0.683573997432168,
0.0161774759256176,
0.845122877901943,
0.852411164367763,
0.600762644596753,
0.321477544643673,
0.667960158860292,
0.526830186847053,
0.848000097483397,
0.250210302067087,
0.256227886423575,
0.0732356584971937,
0.514382180531687,
0.889812722285191,
0.611410934762755,
0.531032761806172,
0.821330783805498,
0.958956887460759,
0.73674695600604,
0.343959444828313,
0.359942238945487,
0.043915304375773,
0.0238631991780657,
0.00507620303196656,
0.487254163011561,
0.292885559747408,
0.708262450391549,
0.820146209476584,
0.50740955141718,
0.467470874761916,
0.0782578820727104,
0.190983548383686,
0.483648350687534,
0.923380759974653,
0.0433947122857881,
0.0844109952842868,
0.244858304152665,
0.711354871611742,
0.611241182131339,
0.0928584011704002,
0.961565173678829,
0.867469069020575,
0.166094060133255,
0.475947353744855,
0.757281790840105,
0.77750499489601,
0.0069801150853653,
0.578612574179942,
0.736461881891108,
0.743727071091406,
0.922572019008255,
0.0964041203709338,
0.78764237593284,
0.94643521818632,
0.1014803234029,
0.27489653847874,
0.239320777933728,
0.809742773794449,
0.0950427479553235,
0.746730329350908,
0.277213648090704,
0.173300630028034,
0.937713877734595,
0.760861999243899,
0.0966813895370259,
0.981108590020383,
0.845272994528186,
0.341539693689691,
0.692463461166464,
0.456514176193864,
0.434398095325752,
0.654028634379631,
0.323983244748778,
0.600492155459007,
0.129975987658825,
0.0812650351232221,
0.377997149889356,
0.13695610274419,
0.659877609768825,
0.114459031314803,
0.880683174301257,
0.582449628311419,
0.210863152151398,
0.668325549768436,
0.528884846497739,
0.312343475554298,
0.943222088247175,
0.768205624431467,
0.122086249348748,
0.0382648357368376,
0.514935953316715,
0.399299897439452,
0.211565465764872,
0.452649830585648,
0.16016189621769,
0.308246855301897,
0.43375842014037,
0.00543489028021455,
0.649786549457249,
0.126221881306834,
0.46194906693974,
0.0841846443173404,
0.780250515686465,
0.785932311688518,
0.684676799776348,
0.910226503810951,
0.867197347277402,
0.0626739492000425,
0.0471826060894796,
0.527074956580566,
0.177132980514845,
0.927865780390737,
0.109524584426323,
0.387996132666243,
0.596191329693511,
0.638409430924062,
0.700339608686203,
0.539413417940686,
0.406615054889868,
0.822425858034951,
0.577678253677524,
0.921551008206583,
0.221725755008741,
0.789243719908057,
0.37420083832657,
0.381887651692092,
0.0974905747442928,
0.807959258932601,
0.387322541972307,
0.747277124201542,
0.934181140239435,
0.849271608912047,
0.831461768518883,
0.714431655460238,
0.635203920600565,
0.516138567829569,
0.624658158805528,
0.502401267412306,
0.578812517029612,
0.671840765360669,
0.0294762235272099,
0.755945498010118,
0.599706545285744,
0.139000807953533,
0.1439416302107,
0.195897874979255,
0.777410239343257,
0.844281238896903,
0.735311292919941,
0.184025293767464,
0.666707096466193,
0.312989546131803,
0.105576301974047,
0.888432851940595,
0.102233265574199,
0.479777140300617,
0.270320503167026,
0.199723840784153,
0.287736398767557,
0.657643045139333,
0.947000964985695,
0.221917538541331,
0.50691465405138,
0.778462733504578,
0.93634919446723,
0.142118574186283,
0.294601300868486,
0.561007352807097,
0.644519841598589,
0.873413818363759,
0.232848117702104,
0.673996065125799,
0.629359315908216,
0.832554663453509,
0.812996873079332,
0.773300946118916,
0.0284525379671029,
0.590407111956928,
0.617582185015819,
0.763763830887044,
0.774432406190053,
0.28428928101635,
0.0767533765531859,
0.8800087081641,
0.172722132491284,
0.178986642593046,
0.359785847999056,
0.44304263565831,
0.378710483377199,
0.647522247232274,
0.100685680797643,
0.325711447897233,
0.869439785773605,
0.607600334849023,
0.104174180936149,
0.805788979775174,
0.749718909035306,
0.398775482270296,
0.366796332582271,
0.394238750168234,
0.272189300168394,
0.599644450284375,
0.0682348148283711,
0.901548616076609,
0.432199113272223,
0.881231687907703,
0.674849561729864,
0.460651651239326,
0.471638799864631,
0.292431746280022,
0.224415481660709,
0.246071205589022,
0.576721027762033,
0.301168858679556,
0.126079913287461,
0.749443160253318,
0.480155501272602,
0.485865761286517,
0.192485795911628,
0.858865984649801,
0.13338800805313,
0.293171476709271,
0.184577432081372,
0.00282779382673455,
0.900771811558293,
0.288751613483183,
0.808616773601909,
0.650490720127938,
0.687527095753479,
0.175413105718518,
0.04472946983051,
0.959716395921873,
0.775057556468555,
0.112964284658881,
0.861265011532821,
0.207256669275116,
0.994195972566584,
0.536114572797024,
0.667908320980104,
0.465834771965553,
0.828546319542707,
0.892323802640812,
0.711905977554576,
0.405267346839079,
0.193492660854707,
0.837985890842036,
0.154710506626736,
0.673648162127309,
0.323851652128553,
0.347196302538363,
0.532514146311448,
0.457239660181682,
0.640367779247634,
0.717091578858482,
0.460067454008417,
0.541139590340266,
0.00584319187600314,
0.268684227610326,
0.191630310002542,
0.693370287629482,
0.444097333328844,
0.236359779833052,
0.653086683085694,
0.219154889331737,
0.349324064491933,
0.514351694152854,
0.426411559072515,
0.343520037058517,
0.0504662664842169,
0.0943198795869573,
0.809354809024071,
0.879012586026924,
0.98664368222777,
0.521260786578646,
0.284279932400342,
0.180136343082477,
0.359246676955021,
0.438990439492739,
0.853784505209785,
0.683098329083574,
0.786186742031102,
0.386298651521233,
0.140337989265257,
0.426554521278736,
0.103390229914053,
0.600405443273674,
0.967694111619002,
0.109233421790057,
0.869089670883999,
0.159324421621544,
0.802603709419539,
0.313187004212843,
0.395684201454597,
0.455690392039572,
0.53234189354458,
0.745008266412191,
0.970042086192426,
0.958753452617095,
0.0885283030050473,
0.0205083522109819,
0.0530733317383906,
0.897883112029118,
0.899520938703567,
0.0397170139661604,
0.419143898142103,
0.183800870638248,
0.219853357048637,
0.778390575562786,
0.622791310130987,
0.0736378617927608,
0.461488904180699,
0.408978052162089,
0.459936513313994,
0.601826893445955,
0.835532573440826,
0.563326743228048,
0.202232336719629,
0.803226684594167,
0.672560165018104,
0.0713220071379663,
0.962551106215711,
0.475163873971982,
0.384509011350809,
0.358235307670308,
0.930854266011554,
0.91685090536105,
0.103243573616838,
0.900896351738319,
0.875604357512484,
0.191771876621885,
0.921404704414962,
0.928677689716536,
0.0896549881853419,
0.820925642652868,
0.968394703682696,
0.508798886793106,
0.00472651282545482,
0.188248060265671,
0.287189461890231,
0.627517823422103,
0.261885922058432,
0.748678366070929,
0.0364958751185312,
0.721822435838088,
0.350505259516884,
0.872028448559357,
0.285149178600474,
0.552737596236513,
0.675255132687863,
0.957709344084239,
0.624059603374479,
0.637806238903574,
0.43287321759056,
0.00856861472528829,
0.996041546573882,
0.363727483602114,
0.925419520086339,
0.0992851197250584,
0.264623834874771,
0.801023877133161,
0.291056996346944,
0.186028538824072,
0.729701566384035,
0.380711984532285,
0.0069541810112792,
0.69809626960107,
0.889510871325392,
0.0116806943023953,
0.886344329866741,
0.176700332749961,
0.639198517724498,
0.148230251925173,
0.925378699286552,
0.67569439284303,
0.870052687763261,
0.275883958337774,
0.547722840936725,
0.155201865898074,
0.828621554574287,
0.222977973158927,
0.112911209516652,
0.452681157483105,
0.860784212062501,
0.545784427572873,
0.461249772208393,
0.856825758170721,
0.909511911174986,
0.386669291829071,
0.956110877895779,
0.174135745584097,
0.187693168962231,
0.247167873777062,
0.36016428487383,
0.917394735346266,
0.627879858775008,
0.367118465885109,
0.615491004947336,
0.517390729634739,
0.378799160187505,
0.501835334348416,
0.694091062850361,
0.0179976774463419,
0.650065586273589,
0.619469761671251,
0.693692070289372,
0.520118273571189,
0.895353720009026,
0.241414910760436,
0.675320139934924,
0.723975274117652,
0.464392884385024,
0.788231349451575,
0.176656431135096,
0.325177095981863,
0.334015776558787,
0.637906203809151,
0.182002854152584,
0.243527687268112,
0.02457549517256,
0.138113731582702,
0.41766343331787,
0.212268664134791,
0.385281605825425,
0.7778277181917,
0.129663399481058,
0.0131614641347721,
0.144946183611148,
0.745154404428394,
0.530552193769511,
0.523745344264314,
0.246989738311148,
0.224643256154211,
0.541743021710656,
0.897055325050398,
0.844113017825462,
0.235435092000028,
0.417173598155926,
0.739466737368827,
0.476850002760463,
0.0924937376251881,
0.463442011020818,
0.941242887145487,
0.880725087542425,
0.640098442621575,
0.26641998312735,
0.21474086363555,
0.278004645965064,
0.448422837279934,
0.458268551369323,
0.302580141137624,
0.586536568862636,
0.875931984687192,
0.514848805738077,
0.971818174688061,
0.653759702413231,
0.644512205219135,
0.984979638822833,
0.798705886024379,
0.389666609181867,
0.515531832592344,
0.322451229823032,
0.636656347493015,
0.740175088746555,
0.864194251999349,
0.533711672077752,
0.584288106572017,
0.0996293435337159,
0.95088527069934,
0.323754843475183,
0.576479346294179,
0.0433790078588664,
0.787196854961662,
0.517722233439666,
0.924104095401291,
0.427295297117576,
0.784142216567016,
0.138844959036841,
0.70529994308264,
0.232565053381289,
0.597113510406163,
0.00788008375460286,
0.819101622243925,
0.473045494627694,
0.52272888949268,
0.790919796931986,
0.126805196575264,
0.167241094246153,
0.775899435289157,
0.925511082599643,
0.556907703428021,
0.29143126741584,
0.247962312422675,
0.193564050921036,
0.0316063561623946,
0.112156563956363,
0.727275722998788,
0.615894462734412,
0.211785907490079,
0.678160993232467,
0.939649306209595,
0.788265253784258,
0.721540001556994,
0.726846160705595,
0.305987486758264,
0.645644096492624,
0.15414145735751,
0.0901297028596186,
0.784489055529465,
0.85944140044015,
0.322694756240907,
0.381602565469967,
0.867321484660414,
0.141796378484832,
0.854648060097661,
0.390050373687432,
0.932716175416818,
0.981453256672925,
0.557291468399247,
0.708615610240314,
0.906964339272568,
0.114199171361606,
4.68776561538119e-05,
0.154926651229582,
0.307763222282642,
0.0316532338185484,
0.267083215185945,
0.0350389452814306,
0.64754769655296,
0.478869122676025,
0.713199938513897,
0.587197002296893,
0.267134375994622,
0.43473993960523,
0.314043163002489,
0.573121863218547,
0.0803840360978544,
0.468184620359998,
0.663251566078165,
0.864873091627319,
0.327626020800148,
0.985946322784734,
0.246475657097285,
0.194947504994901,
0.127742700803905,
0.101123716729285,
0.584997878682333,
0.0604588757550618,
0.08257697340221,
0.142289346615919,
0.769074485995376,
0.989541312674778,
0.256488518443186,
0.76912136365153,
0.144467963438699,
0.564251740725828,
0.800774597470078,
0.411551178624645,
0.599290686007259,
0.448322293557377,
0.890420301300669,
0.312490624521156,
0.0355192958542701,
0.15755467682963,
0.747230564126387,
0.349562458856759,
0.730676540048177,
0.827614600224241,
0.817747079216757,
0.393928106126342,
0.69248769185156,
0.145373099551244,
0.379874428445415,
0.938963348948845,
0.340320604546145,
0.50761712924932,
0.0400870652124691,
0.925318483694139,
0.568076005004382,
0.122664038614679,
0.0676078298443965,
0.337150490534096,
0.112205350823796,
0.324096348287583,
0.106271854185626,
0.256673314262495,
0.888348089479072,
0.907046451655704,
0.66822449288714,
0.48763877502067,
0.355368745213081,
0.558644793722147,
0.800129399541826,
0.390888041067351,
0.716199471017439,
0.547359963668212,
0.740450499924109,
0.446876010599954,
0.374974563426792,
0.558197578675206,
0.840804116726296,
0.0674622548126906,
0.703570678692111,
0.22067854470605,
0.0064256032958746,
0.0438912827725947,
0.72829567395537,
0.046512668974005,
0.969209766466734,
0.29637167849409,
0.169176707588684,
0.0368175963111304,
0.633522169493848,
0.281382058878141,
0.360913944598713,
0.739794023679473,
0.538055373140637,
0.249262033612124,
0.646840474869516,
0.206279866027776,
0.736900809098455,
0.00220921961693523,
0.764924659749924,
0.537030208174619,
0.393097260684286,
0.481124130301701,
0.08439017137717,
0.133547760142734,
0.928000141367316,
0.459364735269623,
0.691745339283601,
0.768804257627951,
0.526826990082314,
0.395316017510051,
0.989482802799662,
0.53325259384385,
0.439207300282646,
0.71777847628937,
0.579765262817855,
0.40841706674938,
0.0141501547834604,
0.748941970406539,
0.44523466306051,
0.647672324277308,
0.030324028819019,
0.806148607659223,
0.38746634749112,
0.568379402425317,
0.0554106412713465,
0.0343068218949748,
0.774659268453093,
0.792311450369801,
0.0365160419775713,
0.539583927737355,
0.329341658078759,
0.429613302661857,
0.0207080580390562,
0.41373182992159,
0.563161063270253,
0.948708199406373,
0.873096565191213,
0.254906402088192,
0.717512456568662,
0.399923554807866,
0.650222419598243,
0.706995258902663,
0.933176148651715,
0.0894297194152277,
0.424773735192033,
0.512941411003909,
0.497846786164607,
0.438923889975494,
0.261883381410448,
0.943081449225117,
0.0865962137871404,
0.292207410229467,
0.74923005688434,
0.474062561278261,
0.860586812654783,
0.804640698155687,
0.508369383173235,
0.635246080642215,
0.596952148059826,
0.544885425150807,
0.17483000837957,
0.926293806138585,
0.974498728278325,
0.195538066418626,
0.340025635594514,
0.537659791082917,
0.144246265359338,
0.213122200320066,
0.792566193171109,
0.861758721928,
0.613045755593593,
0.442788612303691,
0.568753980830663,
0.546221903779647,
0.532218331718919,
0.993527716022696,
0.0591633147835561,
0.0300651178835263,
0.432451605532529,
0.321046696194004,
0.973146567108644,
0.519047819319669,
0.61325410642347,
0.722376623527322,
0.99311038059793,
0.473840919078254,
0.527017321217348,
0.501479763305504,
0.109086999254807,
0.123969468811513,
0.0463651884563105,
0.283917007634378,
0.0502632749500979,
0.0208639162689745,
0.479455074053004,
0.390288910544612,
0.558523707351891,
0.623701339412342,
0.603411111330339,
0.351089900057339,
0.485460061340341,
0.216456866458271,
0.79387851236103,
0.054214041705343,
0.762678770237918,
0.326096843614288,
0.0477417572623779,
0.821842085021474,
0.356161961497814,
0.480193362794907,
0.142888780749817,
0.329308528606458,
0.999241182114576,
0.756142887638948,
0.0516851516681189,
0.992351562246844,
0.229983806251541,
0.578702473351128,
0.493831325086686,
0.339070805506348,
0.70267194216264,
0.540196513542997,
0.622987813606387,
0.752935217112738,
0.561060429811971,
0.10244288719373,
0.14322412765735,
0.119584136698201,
0.726144227071732,
0.746635238987689,
0.47067403675554,
0.211604287946412,
0.963092105445961,
0.264552548650909,
0.265818329651755,
0.725770875218218,
0.590649392730859,
0.313560086914133,
0.547612960239692,
0.946811354228673,
0.79375344970904,
0.690501740989509,
0.27611988236947,
0.792994631357954,
0.446644628162796,
0.32780503450325,
0.785346193139137,
0.676628434414337,
0.906507507854378,
0.279177517760162,
0.0156992399206847,
0.609179449551357,
0.81937403176882,
0.638687053527072,
0.362114666664095,
0.38043446111513,
0.741129940720801,
0.505338794321445,
0.500018598278993,
0.467274167326872,
0.251974032843473,
0.970692635034534,
0.678878455273285,
0.215066137823773,
0.235245183685443,
0.94469678492504,
0.94083701304199,
0.825894576416302,
0.258256871373512,
0.488449972816021,
0.772705930179314,
0.0520103206168908,
0.17895171380553,
0.0488258125487835,
0.845004951974845,
0.625596341968326,
0.376630847052033,
0.630351144648321,
0.302224776382663,
0.28313835444075,
0.909528662874144,
0.317924016303347,
0.892317803992106,
0.728902694177303,
0.956611069830419,
0.25443247019054,
0.109337154826772,
0.697741010085559,
0.759771264511985,
0.609355753105765,
0.165015177412431,
0.0117452968897975,
0.580048388140299,
0.843893632685716,
0.22681143471357,
0.815293571825742,
0.788590417610756,
0.16764844775556,
0.641188147776382,
0.0468472885186073,
0.656098420571582,
0.413894077490035,
0.0988576096011594,
0.835050134377112,
0.462719890038818,
0.943862561576004,
0.460646476345438,
0.839350737090852,
0.574213706224325,
0.762871252728101,
0.12248909106594,
0.483742368632808,
0.0807952685657867,
0.0148068950580465,
0.212645062344449,
0.0374063379305444,
0.269239365248587,
0.321982217171221,
0.735147348016103,
0.029010629760572,
0.931337970742648,
0.900162525428535,
0.0407559266503695,
0.511386358417285,
0.744056158114251,
0.267567361829601,
0.326679929777365,
0.532646575259346,
0.435215809585161,
0.967868077553747,
0.579493863777953,
0.0913142296910818,
0.381762154578121,
0.678351473379113,
0.926364364533855,
0.8444820450826,
0.622214034955117,
0.387010840413631,
0.68383278170779,
0.196427740713781,
0.149882092676071,
0.806321873239391,
0.680170109346588,
0.230677361241857,
0.821128768297438,
0.892815171691037,
0.268083699172402,
0.0903681330803633,
0.214797388396597,
0.0032310467228438,
0.119378762840935,
0.146135358673583,
0.90339357261704,
0.160134689956966,
0.657521717090868,
0.647449730265629,
0.427702051786567,
0.984201646868233,
0.180096305059314,
0.862917861371729,
0.952069723956319,
0.759590168837267,
0.95423209106281,
0.33383187853444,
0.43794164221638,
0.880596455131004,
0.178313923151378,
0.0601556767058352,
0.267607295078974,
0.862146704859169,
0.256583417419616,
0.417489387755044,
0.668468577632899,
0.936753526766204,
0.648166748996902,
0.489597345464675,
0.82956869799158,
0.916250448169303,
0.5799654790107,
0.044366085922516,
0.919481495357808,
0.699344241851635,
0.190501445061761,
0.822875067509187,
0.859478931808602,
0.848023162152629,
0.470324797309155,
0.287180983129507,
0.832224809020862,
0.650421102368469,
0.150098844035575,
0.78429453251152,
0.410011270740075,
0.104330935098385,
0.118126410580299,
0.847952912956454,
0.984927390229389,
0.296440333731678,
0.908108589662289,
0.252534685308363,
0.158587038590846,
0.164692006616244,
0.670024073063407,
0.827055616223745,
0.101445532916787,
0.318190822060309,
0.31665296168842,
0.931014230908367,
0.234441269763951,
0.89661844069912,
0.975380317296544,
0.153922764656098,
0.595962682085094,
0.165881761892644,
0.976797832165285,
0.455441613893696,
0.0139049235796113,
0.447122629008779,
0.742622597023203,
0.846129732600474,
0.0975437309115863,
0.892721441524439,
0.630424265111994,
0.507555002117322,
0.997052376622824,
0.748550675692293,
0.355507914608115,
0.981979766386552,
0.0449910094239707,
0.263616504270405,
0.234514451229253,
0.203578048014817,
0.428308510886649,
0.904538524758322,
0.0306336637729004,
0.529754044269097,
0.222729346352969,
0.347286625461321,
0.460768274711803,
0.45717061611692,
0.24390506569478,
0.436148591542686,
0.611093381238679,
0.839867748245535,
0.602030353435329,
0.587891212938303,
0.29530936167357,
0.615935277480602,
0.0350138419470814,
0.037931958231112,
0.462065009615414,
0.132557572858668,
0.930653399755551,
0.0924892742617471,
0.64011257497599,
0.927705775912714,
0.841039950419702,
0.995620490049767,
0.909685542299266,
0.886030959843672,
0.25923699385451,
0.144199993062858,
0.0896090073928279,
0.687545504741159,
0.0487385173555177,
0.120242671165728,
0.217299548544595,
0.271467863708487,
0.46752929709271,
0.678067823722059,
0.728638480291068,
0.71143436278749,
0.114216414799083,
0.339731861064086,
0.551302110567364,
0.716246768700074,
0.927623074002388,
0.846611472240934,
0.332182045715014,
0.96263691594947,
0.884543430937707,
0.794247055330429,
0.0951944888081376,
0.815196830227597,
0.886736330057837,
0.735307063784128,
0.742902606140311,
0.727776280011878,
0.730927553368233,
0.652588147973916,
0.613807239389889,
0.990164547222743,
0.796788141036773,
0.703416246782717,
0.677710051963902,
0.845526658857952,
0.823658917948445,
0.895009600508497,
0.116994522100778,
0.291188214575494,
0.573077423764894,
0.845633002391845,
0.00262257736298376,
0.687293839029639,
0.18536486299027,
0.553924687930348,
0.403540607264051,
0.112987936992658,
0.400536160171282,
0.735722652979066,
0.0756248524764668,
0.285079590643327,
0.529969707843833,
0.170819341284604,
0.100276420405263,
0.416706037436009,
0.906126405534393,
0.843179026545574,
0.144482316982226,
0.637053958436965,
0.495767174053829,
0.758289556372114,
0.627218505659708,
0.292555315090602,
0.461705802689169,
0.304928557157949,
0.138081973482893,
0.285364720637614,
0.199938157200785,
0.25507649558367,
0.576552935213108,
0.77301558143134,
0.100709497975516,
0.579175512576092,
0.460309419995318,
0.286074360965786,
0.13310020050644,
0.863850027259369,
0.399062297958444,
0.533636360677721,
0.599572679772774,
0.474687150900572,
0.818715951321049,
0.129542387150946,
0.645506492185177,
0.918992371726312,
0.546248425052617,
0.551632897253909,
0.762171398271886,
0.690730742034843,
0.188686855690874,
0.257938571860054,
0.449020298406957,
0.815905361350582,
0.550493886950656,
0.910726101096126,
0.120833918042869,
0.688575860433549,
0.196090821268079,
0.320772075243654,
0.94365235648288,
0.772643756946849,
0.0937876562093327,
0.044361853992735,
0.351819269057279,
0.554097076204651,
0.330436214958521,
0.484919469563719,
0.417947102998359,
0.729498513382626,
0.0185558297757785,
0.0175197823054715,
0.204185663817537,
0.837271781096827,
0.147062169922079,
0.849692156002713,
0.756264152823139,
0.693310594974696,
0.401325053256622,
0.518435550629364,
0.384041336543877,
0.590011908947496,
0.776374122489418,
0.833061634950834,
0.405917269832416,
0.326868009440074,
0.74378773604696,
0.526751187875285,
0.0154438694079611,
0.93987855731504,
0.847523263118939,
0.959096225890841,
0.712522313796227,
0.941310919793933,
0.00345807941791512,
0.0643415828535061,
0.495407995532922,
0.333894294842097,
0.549261052417225,
0.913355098996942,
0.0633928077590618,
0.567816882658665,
0.930874881302414,
0.267578471576599,
0.40508866328983,
0.0779370507588317,
0.117270627113651,
0.161352815647308,
0.771247645733528,
0.518595680370273,
0.679788366276672,
0.155288982277405,
0.108607588852107,
0.45616248876609,
0.988350617228239,
0.514524858684523,
0.783030498206164,
0.732138352809538,
0.0412760460941475,
0.798474367614125,
0.672016910124578,
0.888799309678748,
0.757570593504966,
0.384539223455144,
0.83011022900702,
0.761028672922882,
0.44888080630865,
0.325518224074281,
0.0949229672993175,
0.998141859191536,
0.238873322605562,
0.158315775058379,
0.565958741384539,
0.169748203907976,
0.425894246634978,
0.97104740467437,
0.247685254666808,
0.54316487421429,
0.132400220321678,
0.0189329004003354,
0.0617605541189018,
0.81218858659835,
0.17422188267774,
0.17036814343667,
0.268351074898779,
0.162572499440318,
0.684893002121194,
0.0513815726392817,
0.894710852715518,
0.726169048681003,
0.849855940719068,
0.566727762374435,
0.614968357894089,
0.607426533758373,
0.951266985829578,
0.445078586435448,
0.368455206215594,
0.400147792138228,
0.770596810509729,
0.463378173980572,
0.398289650864103,
0.00947013311529073,
0.621693949038952,
0.964248392248642,
0.179218337023267,
0.0475881956739296,
0.93529579645735,
0.426903591690074,
0.59075306988822,
0.0676960163133666,
0.44583649209041,
0.652513624007122,
0.879884603377378,
0.62005837476815,
0.822881767443792,
0.148235677810496,
0.78263087467413,
0.507774769564986,
0.199617250915439,
0.677341726923986,
0.233943817780327,
0.0494731911688453,
0.244069488832759,
0.848912175674416,
0.656899724927218,
0.195336474196676,
0.293990761644203,
0.0253549306771508,
0.595484266334904,
0.06458757168827,
0.488733104657723,
0.993773917199007,
0.0740577048035607,
0.110427053696675,
0.958022308981988,
0.253276041826827,
0.158015249370604,
0.893318105439338,
0.680179633982563,
0.748768319258824,
0.961014121752705,
0.126016125607312,
0.401281942800284,
0.840898724664421,
0.746074500841123,
0.224163710244076,
0.989134402940578,
0.528705375049592,
0.731938479809062,
0.188751653390355,
0.206047101507917,
0.965882297589389,
0.238224844559201,
0.450116590340676,
0.814794472798144,
0.895124569486419,
0.645453064537352,
0.108785233976685,
0.92047950016357,
0.240937330872257,
0.173372805664955,
0.409212604821293,
0.234711247605603,
0.247430510934177,
0.519639658517968,
0.192733556587591,
0.500706552761005,
0.677654907888572,
0.0860516615612673,
0.180886186277907,
0.426423226681735,
0.0470657828483106,
0.30690231235088,
0.827705169482019,
0.887964507978393,
0.0529768127263416,
0.0518688792604343,
0.87709891045331,
0.581682187775933,
0.783807359069496,
0.0658505633780037,
0.78772928928385,
0.749689656193224,
0.304075407937204,
0.237845879158865,
0.564484128525706,
0.199199976957962,
0.883298943696217,
0.673269362502391,
0.11967947665587,
0.124236274102813,
0.846642168633008,
0.528892081477163,
0.358947522174077,
0.0940726791015233,
0.0485317395294699,
0.551681078761667,
0.594779231862528,
0.726186647418042,
0.637732740322935,
0.775665418606096,
0.152609873634116,
0.684798523171245,
0.082567730491314,
0.980315043581796,
0.572763030683977,
0.135544543217656,
0.0321839223765693,
0.449861940671625,
0.717226730993589,
0.815991281911727,
0.515712504049629,
0.504956019811777,
0.565680937639289,
0.819787911986833,
0.742801898970642,
0.130165066164995,
0.018987888479134,
0.626100842201198,
0.803434428667386,
0.138667365600666,
0.750337116769672,
0.650076596834733,
0.667559447077829,
0.109284638478088,
0.744149275936256,
0.71609118707296,
0.660965717239755,
0.338928507798784,
0.442277834025341,
0.298698457097029,
0.114593925939218,
0.594887708125118,
0.983496980268274,
0.197161656430532,
0.575202751241253,
0.556260010952251,
0.332706199648188,
0.607386673617822,
0.00612195115821527,
0.0499329301761151,
0.423377955063888,
0.521834455673506,
0.554888949987893,
0.989058893168838,
0.341622367194678,
0.297690848492873,
0.119223958868172,
0.360610256139473,
0.923791690694071,
0.922658387535558,
0.499277621740139,
0.674128806998082,
0.572734984370291,
0.166837068352307,
0.783413445476169,
0.316884259840885,
0.882928255425267,
0.444379162250263,
0.655812767639669,
0.325206089450608,
0.743077619347292,
0.770406693578887,
0.920093797575726,
0.726574599615566,
0.967568350009419,
0.495296548351318,
0.282834610102156,
0.300274549191945,
0.102683221503479,
0.288956561260371,
0.350207479368061,
0.526061177033028,
0.810791016933877,
0.905096429355953,
0.515120069736205,
0.152413384128554,
0.202787277383165,
0.634344028604377,
0.513023640268028,
0.126578968077236,
0.557002415674274,
0.0123012615425052,
0.800707775075318,
0.129737399578904,
0.179138329894812,
0.584121220551488,
0.44662165988545,
0.062066585320079,
0.0285003823360896,
0.102434427059458,
0.387272674770687,
0.771578002149043,
0.872841121104006,
0.307366471880752,
0.498152601298947,
0.840409470647764,
0.80266302023207,
0.780987211401103,
0.14068401983971,
0.905346241735549,
0.0699437721958122,
0.49089149920777,
0.431407418302916,
0.88073478959535,
0.395987928563723,
0.946527488039121,
0.0331481732582432,
0.598775205946888,
0.580871516177837,
0.546171813526271,
0.725354174024125,
0.137873931852111,
0.558473075068776,
0.526061949099443,
0.267611331431014,
0.737611404963588,
0.110183169185269,
0.714232991316464,
0.799677990283667,
0.138683551521359,
0.816667418841583,
0.186950664588693,
0.910261553670402,
0.689508539479928,
0.494317136469445,
0.408414154503687,
0.529918009662031,
0.296980156235853,
0.189401365439129,
0.670602029501741,
0.202326397971402,
0.259345138100602,
0.161493528709511,
0.633733816274318,
0.140079927230291,
0.557481457273234,
0.580261304313439,
0.173228100488534,
0.156256662754461,
0.161132820025614,
0.719399914014805,
0.881610837244247,
0.299006751877725,
0.27787298861792,
0.407672785878029,
0.5666180837744,
0.0154843931158466,
0.517855955063298,
0.280851074625203,
0.815162383865175,
0.656539506584657,
0.0975184930011251,
0.00211304798820664,
0.566801060255059,
0.787027032481053,
0.496430184923313,
0.975215214758746,
0.316945042143085,
0.793410341159166,
0.164616580197875,
0.987547071644825,
0.995736739130568,
0.423961718298477,
0.149040599888675,
0.629470555404886,
0.564041645528768,
0.70652205716191,
0.209731859252663,
0.737269746017302,
0.862778720382032,
0.370864679278277,
0.456669659566446,
0.744389557160618,
0.669871431621663,
0.734542648184366,
0.152062342572986,
0.236489514930402,
0.750027041300212,
0.669918297636285,
0.517340589555605,
0.565189424699726,
0.32645780375528,
0.614859083022391,
0.567302473153594,
0.893258864010339,
0.401886115037783,
0.0637326576112456,
0.868474078769085,
0.718831157180868,
0.857142998770412,
0.0330906585012985,
0.706378228825693,
0.85287973790098,
0.457052376799775,
0.855418828714368,
0.482350292840204,
0.0210940218628822,
0.561940885876278,
0.692082152092868,
0.758363767880184,
0.424719605792649,
0.0629468309054835,
0.215033426980969,
0.169109162487606,
0.732818262527147,
0.949576075165335,
0.321171505060592,
0.969307777457548,
0.699603115999886,
0.991089802696877,
0.486648367013153,
0.264792540699613,
0.317547605986496,
0.101507449569883,
0.832095013853207,
0.210806469996835,
0.503393564607666,
0.895827671464452,
0.0792805483002591,
0.222224721788533,
0.752970669769203,
0.112371206801558,
0.928602950614226,
0.605850407204521,
0.569423583601333,
0.784021778862934,
0.0882006995790641,
0.590517605464215,
0.34596266427355,
0.780282851671932,
0.348881372878738,
0.770682270066199,
0.843229683043077,
0.563914799859708,
0.939791432553805,
0.576047945104562,
0.513490874559381,
0.260962937148736,
0.54535572256211,
0.213093990093607,
0.252052739379952,
0.0320040891096015,
0.477886530793219,
0.56960034583211,
0.133511538679484,
0.309981544180765,
0.780406815828945,
0.63690510328715,
0.205809215179556,
0.859687364129204,
0.859129825075683,
0.95877988541442,
0.972058570930762,
0.787732775224248,
0.564630292153279,
0.541482154066433,
0.571754554087182,
0.652830992198005,
0.131999759064987,
0.917717218360732,
0.433113843404275,
0.480881131943726,
0.688399488426931,
0.276343525981691,
0.0447959313377719,
0.628190920515075,
0.852391471551914,
0.558286805897153,
0.889153858129472,
0.397747193648362,
0.77138079599076,
0.141206597043763,
0.429751282757964,
0.249267326783979,
0.710806942875873,
0.563262821437448,
0.559248870964744,
0.491213758239156,
0.200167924258936,
0.765058086609961,
0.350901121902699,
0.0592977493346193,
0.723837971558719,
0.322959692367799,
0.847030524558868,
0.288468263246337,
0.864441846434233,
0.418785078180388,
0.941299255444342,
0.99644160549922,
0.33650229654112,
0.374413098848617,
0.477322736977284,
0.0249017845023897,
0.650756624830308,
0.522118668315056,
0.653092705017465,
0.50314809591656,
0.0804054737465482,
0.542246562681276,
0.900895289564922,
0.851786270202969,
0.6834531601907,
0.330646571857224,
0.101053596521287,
0.394260102600912,
0.893909393294672,
0.660302467486031,
0.885473861305729,
0.0940773175536084,
0.42536055363033,
0.236374982742767,
0.153375066888228,
0.149198524723388,
0.559334675576228,
0.000405590981433909,
0.437666788435386,
0.423776521544799,
0.419190669627483,
0.378966043414067,
0.420218127044019,
0.755692966168603,
0.753379142262684,
0.897540864021304,
0.780594750670993,
0.404135767092992,
0.41965953233636,
0.433687455222796,
0.907283863009551,
0.500065006082908,
0.975934018369733,
0.808179152574474,
0.351851275820216,
0.659387178094772,
0.138825723966037,
0.452904872341503,
0.0536472802300226,
0.032735117260709,
0.113207339827534,
0.939121141535752,
0.126812434814317,
0.538567893457863,
0.175496124278519,
0.280187501702545,
0.687766418646912,
0.734830799854747,
0.280593092683979,
0.125433206616637,
0.158607320933886,
0.699783762311462,
0.504399250030704,
0.578825447977905,
0.455476728014404,
0.257778392293387,
0.476366311999208,
0.236071478219736,
0.661914159386379,
0.896025844335568,
0.669758933908194,
0.569198021930269,
0.396090849952815,
0.645692951812266,
0.377377174039081,
0.747942126238692,
0.305080129441377,
0.516202898470779,
0.200846998114533,
0.358727409671399,
0.548938015731488,
0.314054337942067,
0.297848551207152,
0.675750450545806,
0.852622231865591,
0.473344675485671,
0.955937952248351,
0.540388650046842,
0.208175474874757,
0.236531044466668,
0.665821856663479,
0.366782795808643,
0.936314807243792,
0.170221106228522,
0.945608244252209,
0.391791534792535,
0.427999498521909,
0.421974555785756,
0.627863013477932,
0.0899136574426264,
0.318000399655663,
0.297621946920465,
0.659111679838557,
0.714091249608477,
0.943314898732731,
0.0364888534119766,
0.462033375381508,
0.248395027708446,
0.552691751882756,
0.662880373961702,
0.607122437379846,
0.101629767148583,
0.976934711903769,
0.904970988586997,
0.777380217694389,
0.829556943303699,
0.378315663607007,
0.733318169477078,
0.36994559288488,
0.586491138481764,
0.969849214409408,
0.0357674490826984,
0.953273934756068,
0.906164021187538,
0.205988555776881,
0.898882178542615,
0.297955555514412,
0.63398805429879,
0.32085673386271,
0.925818568992344,
0.723901712207078,
0.638857133518372,
0.223440515447147,
0.383013391579973,
0.352948382661188,
0.166755413714217,
0.41950224499195,
0.814981758508357,
0.415150441422663,
0.972193997340367,
0.477862132004398,
0.0222728788025085,
0.073823764023289,
0.454796843442506,
0.927243867389506,
0.851203982183339,
0.284353786746205,
0.305559530996512,
0.584522151194756,
0.654299379631085,
0.892050669478276,
0.554371365138502,
0.690066829179445,
0.845324603768682,
0.46053538586038,
0.896055384956326,
0.744206781845636,
0.758490941840453,
0.530043438789455,
0.0650635152426844,
0.684309510367135,
0.253945150530872,
0.703920648761057,
0.907750026279944,
0.636958542110845,
0.0568690314222449,
0.0745054395284995,
0.0564607871027946,
0.871850789930602,
0.489655881416824,
0.0286547839775005,
0.349712921469339,
0.511928760219332,
0.102478548000789,
0.804509764911844,
0.439172627143177,
0.953682530184129,
0.0888635511923877,
0.744732158139689,
0.538204681378885,
0.743162931289134,
0.636782827617965,
0.0925760460517258,
0.433229760002918,
0.482107430920986,
0.553111432377766,
0.329285144493582,
0.226314212766622,
0.311602373752558,
0.859328583283037,
0.291377728009307,
0.995911884119693,
0.113273733813909,
0.995298377236025,
0.903661909933976,
0.750232275924754,
0.0521674081926082,
0.978167349928137,
0.806693063027548,
0.92401819812321,
0.467823230879299,
0.835347847005049,
0.273731119126888,
0.979751991098631,
0.937826395471499,
0.0782408835730706,
0.418924617776146,
0.891508925189967,
0.16710443523112,
0.163656775915835,
0.42971360610319,
0.910267366520254,
0.800439603533801,
0.522289652620577,
0.34349712605751,
0.282547033989125,
0.0754010845326824,
0.672782270551092,
0.508861246755748,
0.38700345828524,
0.532110853368468,
0.800238975230716,
0.382915341939272,
0.645384587182377,
0.795537352001079,
0.286577251873248,
0.39561686310713,
0.847704760193687,
0.264744601335723,
0.202309925669017,
0.771722957851236,
0.732567832215022,
0.0376577726740659,
0.0454540765124625,
0.712319822847992,
0.975484168145565,
0.123694960085533,
0.131244440158477,
0.866993092869871,
0.290799395316653,
0.294901216074313,
0.296706698973061,
0.201066761371245,
0.0953408191424519,
0.818996351593638,
0.544563887428755,
0.377887853597239,
0.89439743612632,
0.217346157514186,
0.886749100352986,
0.281400893945899,
0.749457011348315,
0.686988075118041,
0.664316235885171,
0.394841598065031,
0.482525426653458,
0.950893487758419,
0.790458461172161,
0.330230186381484,
0.215638088628481,
0.992768387306839,
0.101953143767059,
0.948205920843503,
0.030426159515244,
0.147407220279522,
0.660525743225834,
0.00591032719514814,
0.271102180830716,
0.791770183849973,
0.872903420065019,
0.561901576147369,
0.0866713994586241,
0.169610118572419,
0.762968337984275,
0.182012219066737,
0.988606470166057,
0.307532224947369,
0.559900072663976,
0.883003905826716,
0.524878382927216,
0.446649172551301,
0.164404799306954,
0.27433539380987,
0.133637247669342,
0.828721035657786,
0.6691769918749,
0.6161626743228,
0.779614522950544,
0.459635453047061,
0.946392861169946,
0.995252612044687,
0.452403839888239,
0.0483460044713439,
0.943458532422529,
0.482829999403483,
0.195753225216527,
0.603984275648363,
0.488740326598631,
0.466855406047243,
0.395754459032675,
0.361643746197989,
0.028756981728951,
0.482425858491299,
0.531253865236069,
0.791725319713226,
0.664438077558036,
0.519860334936464,
0.099257544660595,
0.224338149756351,
0.40286424076318,
0.624135927587811,
0.670987322773313,
0.567269040070134,
0.89847132139768,
0.804624570442654,
0.395990075262259,
0.567648312806919,
0.420787244299793,
0.175604598212803,
0.0272837653883192,
0.367180105004078,
0.170857209791829,
0.479687605276558,
0.415526109941083,
0.114315741748696,
0.962517604680042,
0.61127933515761,
0.718300017397059,
0.451257930813012,
0.0781347407391922,
0.114054475964072,
0.812901677476662,
0.106891722468143,
0.596480334921032,
0.34415554224707,
0.898617042647031,
0.260918412013407,
0.864015877183534,
0.997874587307626,
0.485256562235419,
0.266880117481053,
0.622010514429775,
0.156243884543071,
0.834149158016848,
0.520481835361794,
0.960868454985725,
0.230139232813446,
0.0881301477030526,
0.381655698819857,
0.405743831026249,
0.115413913557033,
0.748835804289596,
0.576601040818077,
0.595101518833591,
0.164361913765018,
0.690916783032434,
0.557619123513633,
0.775641248922628,
0.409216799963832,
0.00887705386098337,
0.85377598966182,
0.523271276393566,
0.821778731337645,
0.960667712129963,
0.119751610848937,
0.165934273119054,
0.859284754311333,
0.380670022862344,
0.0299501503025881,
0.857159341153297,
0.865926585097763,
0.296830267783641,
0.479169855117411,
0.0221704691751723,
0.130979425334828,
0.999651690479206,
0.983038924160897,
0.361118658613934,
0.0877818381822583,
0.364694622980754,
0.766862489640183,
0.203195751739291,
0.113530426804689,
0.34346353045826,
0.798297270572883,
0.277892340569707,
0.0343803130250332,
0.355916393620854,
0.0535335890266735,
0.443597112988866,
0.364793447947499,
0.907309578688494,
0.966868389382431,
0.186572178819483,
0.867977290352796,
0.0866199997657072,
0.352506452404198,
0.727262044664129,
0.467290023093713,
0.382456602706787,
0.584421385351765,
0.333216607725814,
0.679286870490428,
0.063591240469176,
0.355387077366648,
0.810266296290917,
0.0632429304827205,
0.338426001061884,
0.17138495443919,
0.151024768664979,
0.703120624042638,
0.938247444079373,
0.35422052040427,
0.816651050847327,
0.281710974071972,
0.152517790977153,
0.0945433909513724,
0.316091287097005,
0.508434184598007,
0.148076979978046,
0.759688400551532,
0.873227632545506,
0.0553865582008784,
0.726556789468302,
0.0597998113649896,
0.923363849019336,
0.81317678969967,
0.412306263769188,
0.650625893217803,
0.280466812327722,
0.794762866475975,
0.235047278103906,
0.613683420053536,
0.474049736500741,
0.298638518573082,
0.969070497420184,
0.284316032325996,
0.361881449521464,
0.307496498482067,
0.455700986765186,
0.512906218186443,
0.010617122059044,
0.393948430844559,
0.867126739056374,
0.827268173372032,
0.67565940491653,
0.0196445295678659,
0.921811564323405,
0.991750692479196,
0.528078714165873,
0.0698885443014505,
0.751439092565067,
0.40130634671138,
0.125275102502329,
0.477995881567707,
0.461106158076369,
0.0486389510560031,
0.291172670801716,
0.873412421845557,
0.699264844273806,
0.571639483129438,
0.668175287855871,
0.934312122843374,
0.185322903182974,
0.14222502389095,
0.232950640950795,
0.154393400137496,
0.426541056682608,
0.594832090472259,
0.461889898619563,
0.882242043447793,
0.107738308658702,
0.472507021144269,
0.276190473826691,
0.974865047715076,
0.29977519405064,
0.951849879208882,
0.994509577282942,
0.221586757908383,
0.943600571222417,
0.522588290983154,
0.291475302209833,
0.695039663321823,
0.923894637694533,
0.416750405177823,
0.173035544423869,
0.385000795305241,
0.465389356233827,
0.464208215691246,
0.258413216685137,
0.164654200507633,
0.0358476983550227,
0.926588504541008,
0.0989663228853449,
0.221170601537996,
0.0688135284319583,
0.33191696383614,
0.375564002141153,
0.495354585114566,
0.926749054774059,
0.837453900760717,
0.377596628096698,
0.0344873629670997,
0.309960921439324,
0.65378710238905,
0.00935241021651421,
0.609736115489964,
0.605636981132271,
0.00386198703379463,
0.831322873864008,
0.549237551889027,
0.526450278016948,
0.12279817560818,
0.244277214745189,
0.450344915711482,
0.539548580786003,
0.417312759169057,
0.835345711016723,
0.00493793701982961,
0.881520974860304,
0.0937589277018602,
0.169592137527462,
0.917368673680988,
0.0203474317772069,
0.268558460412807,
0.138539274753322,
0.0891609602091652,
0.600475424248947,
0.514103276894476,
0.584515545323731,
0.527224478557345,
0.351557177655193,
0.962112173886091,
0.561711841524444,
0.661518099094517,
0.61589927580948,
0.571064251740959,
0.27125421458448,
0.221536256476089,
0.574926238774753,
0.102577087982826,
0.770773808365117,
0.101376516791702,
0.225375263591006,
0.0150510226446441,
0.551721432503183,
0.76492384484267,
0.432363781813701,
0.387067143054245,
0.7698617818625,
0.313884756674005,
0.480826070756105,
0.939453919389962,
0.231253429889331,
0.501173502998973,
0.208012379337108,
0.369792704642654,
0.590334463208138,
0.808487803586055,
0.883895982002791,
0.174850008531869,
0.335712282143399,
0.235453159192322,
0.136962181952299,
0.897424123667844,
0.896971258286839,
0.752861457761778,
0.468488375408802,
0.168225472405658,
0.974397714237868,
0.0434146137178944,
0.270802560388484,
0.745171522137323,
0.144791130509596,
0.496177824445152,
0.760222544781967,
0.696512563012779,
0.261101668822161,
0.192586326130008,
0.0835797060670237,
0.0309634502189995,
0.506471082804013,
0.564405776823128,
0.970417369608962,
0.737724512693344,
0.0655792793564402,
0.178429748480408,
0.107517217335998,
0.65591374303024,
0.986917552066463,
0.991413199338789,
0.830763751562109,
0.322629833744201,
0.22686635806545,
0.967725933514408,
0.220053957412045,
0.123837615886628,
0.720587390810525,
0.688542332820847,
0.292063088757947,
0.694985104582731,
0.731956946538741,
0.562865649146431,
0.440156626254393,
0.876748077513998,
0.0590434731259213,
0.200379170570699,
0.573260640061116,
0.320145141948082,
0.392965496700707,
0.65684034612814,
0.351108592167082,
0.899436579970381,
0.221246122951268,
0.321525961310382,
0.637161092198063,
0.286825402307708,
0.49995570979079,
0.744678309534061,
0.942739145337948,
0.486873261857253,
0.736091508407188,
0.773502896434396,
0.809503095601454,
0.962957866472638,
0.741228829483143,
0.0295570525478372,
0.0867954823592657,
0.461816219828006,
0.718099385368684,
0.378858571117212,
0.156801323945076,
0.450056331907425,
0.941724220263643,
0.596957950199469,
0.326804408955762,
0.00076769338956461,
0.797337120770168,
0.900065049016878,
0.320912835337647,
0.190302617470875,
0.556905395145018,
0.67202142797039,
0.089739196975594,
0.778151518096287,
0.993547389280772,
0.726900289173657,
0.0649769199383338,
0.493503099071562,
0.471578598707718,
0.00771606527628194,
0.980376360928815,
0.207670106649245,
0.781218961710678,
0.789879456064608,
0.170627973121883,
0.52244779119382,
0.819436509078106,
0.257423455481149,
0.984264011021826,
0.537535893981129,
0.636282026598361,
0.141065334966902,
0.987592225888554,
0.578006246396343,
0.738023285166371,
0.314396634378655,
0.578773939785908,
0.535360405936539,
0.214461683395534,
0.899686775589216,
0.725663023407414,
0.771367078540552,
0.571708203093944,
0.815402220383008,
0.549518596171177,
0.565255591909055,
0.542302509091004,
0.614495516109511,
0.0587586905149551,
0.0138811073330609,
0.622211581385793,
0.0391350509781088,
0.221551214447967,
0.403430543096471,
0.829014507508378,
0.392179187569851,
0.925878334290291,
0.648451016120823,
0.649602643051,
0.910142344846457,
0.18598690963629,
0.2858846691837,
0.0512076793476975,
0.173579135524844,
0.863890915580043,
0.78923096497973,
0.4879757699035,
0.442664855365951,
0.324591370450608,
0.702437453299033,
0.342351630489506,
0.050254393392361,
0.473804531373924,
0.91405983358345,
0.865656613775369,
0.0233231270794399,
0.479315425026843,
0.407959122400712,
0.637818643654612,
0.538074116007459,
0.421840230199434,
0.260030224574744,
0.577209166985568,
0.643391444647401,
0.663460767671215,
0.406223674028285,
0.0355706317515907,
0.589339101495845,
0.054674689683446,
0.68517327480259,
0.499481446342301,
0.240661599785398,
0.97105794398629,
0.550689125689999,
0.414240735310242,
0.834948859100672,
0.339920090204068,
0.902216505213741,
0.277613714000962,
0.664511460654676,
0.604653958512775,
0.619965344490467,
0.714765854047037,
0.0784584894210373,
0.534025177608256,
0.580422467356744,
0.101781616966138,
0.0133406026350989,
0.988381590223118,
0.739600260620751,
0.551414718642558,
0.41022181995689,
0.999630485195494,
0.128623885162465,
0.0536132641386302,
0.663091252866709,
0.534847559656411,
0.089183895890221,
0.252430353896893,
0.589522249339857,
0.774357170692811,
0.751911800239194,
0.830183849125255,
0.74541511421344,
0.302600925929193,
0.244424583969835,
0.580363972848451,
0.64252101613326,
0.146641089183577,
0.857977687315074,
0.307032476787936,
0.751295047696352,
0.47794303133988,
0.0217983303693115,
0.829753537117389,
0.0119682089481355,
0.602220798191717,
0.931535154083527,
0.0253088115832344,
0.590602387949174,
0.671135414238617,
0.576723530225793,
0.000824207440402455,
0.670765899434111,
0.705347415388258,
0.0544374715790327,
0.333857151835159,
0.240194974579008,
0.143621367469254,
0.586287505732052,
0.829717224384526,
0.917978538162065,
0.338199305971246,
0.65990107304412,
0.663393651909844,
0.640800231900439,
0.904325657479617,
0.243757624292633,
0.283321247568038,
0.0509667461975323,
0.101735311142046,
0.590353724355974,
0.802261793893884,
0.579678342947587,
0.612152055190947,
0.632015330545611,
0.591646551895722,
0.214372852917003,
0.563550484629139,
0.616955363478956,
0.804975240866176,
0.234685898402094,
0.193678893239088,
0.805799448306579,
0.905451797836205,
0.899026308627346,
0.860236919885612,
0.239308949205703,
0.139221283206354,
0.00385828688920396,
0.825596455403416,
0.96893850759088,
0.921836825051269,
0.163795760909,
0.628839580169338,
0.585230476495452,
0.804595992809439,
0.533165237183294,
0.828988100788085,
0.0879172399118157,
0.584131983380826,
0.930723412395792,
0.678270964733451,
0.386393776809049,
0.510401754877717,
0.290423019458737,
0.0184091073546601,
0.102048306307778,
0.50479587237574,
0.581959591983799,
0.719003669786734,
0.309771112776255,
0.816645490385893,
0.912682563025822,
0.115570560617172,
0.722097288222098,
0.811708871187507,
0.975807480502784,
0.9614062374278,
0.95093015439386,
0.979665767391988,
0.787002692365555,
0.919868661519079,
0.901502591977596,
0.950798453274555,
0.548708241688417,
0.486733068007386,
0.755394445618333,
0.08187347840605,
0.315721168329809,
0.84331168599581,
0.666005462252537,
0.24644458025994,
0.521582650263599,
0.0523992385959249,
0.756846335137657,
0.812005669722336,
0.070808345950585,
0.858894641445435,
0.316801541632415,
0.652767937934384,
0.577898310766508,
0.626572654408669,
0.469413428320276,
0.490580873326669,
0.742143215025842,
0.191510716076712,
0.302289744514176,
0.717950695062964,
0.152916953038851,
0.253219898442374,
0.697616461989291,
0.939919645870067,
0.173088559961453,
0.599119053501226,
0.89071809867896,
0.721796801649871,
0.0858521210429501,
0.646112544297293,
0.803670280055921,
0.40157328937276,
0.489424229827441,
0.469675741842797,
0.648017870098361,
0.0110068796253795,
0.522074980904383,
0.404864204770357,
0.823012549347716,
0.592883326854968,
0.263758846215791,
0.139814090514469,
0.245651264789352,
0.841657156982299,
0.766386744923138,
0.715064693109628,
0.332238030308968,
0.508529959483319,
0.906575409186341,
0.634527774823144,
0.226480654080622,
0.0594923617595306,
0.887747673265519,
0.924097116069913,
0.999412007629598,
0.0608362327613105,
0.523216169105478,
0.890130106308558,
0.782633034411181,
0.609068290148428,
0.53624265014019,
0.586303314467102,
0.0106415790555261,
0.02566687950197,
0.0559790558442376,
0.658659449153887,
0.0366737591273495,
0.578054036748621,
0.0635236539242434,
0.859686308475065,
0.170937363603589,
0.327282500140035,
0.999500398989534,
0.416588628392941,
0.168939656656673,
0.765887143447011,
0.131653321036908,
0.501177686965641,
0.274417102464669,
0.0382287297575868,
0.135705461323124,
0.500897756545291,
0.0977210915171174,
0.0234531345886426,
0.424994872149543,
0.0971330991467149,
0.0842893673499531,
0.948211041255021,
0.987263205455273,
0.866922402226796,
0.557279330937788,
0.523505855129802,
0.453225716228236,
0.567920909993314,
0.549172734631772,
0.509204772072474,
0.226580359147201,
0.585846493759121,
0.0872588088210946,
0.290104013071444,
0.445532801768525,
0.258196172424683,
0.617386513211479,
0.445033200292398,
0.674784800817624,
0.786326169868152,
0.210920343273748,
0.806438121854532,
0.287503856833793,
0.485337445738417,
0.844666851612119,
0.423209318156917,
0.986235202283708,
0.942387943129236,
0.44666245274556,
0.41123007396759,
0.0395210418102895,
0.530951820561174,
0.35944111475695,
0.0267842467999012,
0.397874222322309,
0.916720445694737,
0.550290101929703,
0.851099938550545,
0.48464135522239,
0.0994628360958131,
0.360304710157357,
0.71122171436959,
0.685309329854934,
0.447563518978452,
0.00132572697537287,
0.130842131157798,
0.705759691403136,
0.618712240186852,
0.575875331450196,
0.380544491755098,
0.405038410055003,
0.786795674723944,
0.186982613143969,
0.692542266888796,
0.272133119996699,
0.0316494642904259,
0.115751584580052,
0.258368321814746,
0.974037407885323,
0.562414037791274,
0.669598395782336,
0.0135584492299512,
0.0933658578867865,
0.0290395100736243,
0.0403426964955138,
0.491240080209095,
0.945759955768362,
0.590632798425217,
0.342340018293979,
0.430401310990751,
0.690095634986691,
0.702644728451336,
0.14162302489468,
0.375404964375964,
0.150208247429788,
0.142948752335715,
0.506247095999423,
0.855967938832924,
0.761660992522566,
0.0821224269839574,
0.236512430588022,
0.166699402111908,
0.868918102173562,
0.423495043731991,
0.859241669000705,
0.1410512217046,
0.455144508488078,
0.974993254046418,
0.399419543985007,
0.42918191590774,
0.537407291372031,
0.0690179393016817,
0.442740365137691,
0.630773149258817,
0.0980574498409673,
0.483083061633205,
0.122013229002251,
0.0438174051436677,
0.0737158600584212,
0.46435324729623,
0.474218716134419,
0.763811495045112,
0.166997975281904,
0.615841741494761,
0.139216458955415,
0.317206222711693,
0.758790493830475,
0.645463554954838,
0.173174161544616,
0.52045148588738,
0.727585982404456,
0.409686592132638,
0.68715088846495,
0.596504084112357,
0.833181635864629,
0.546392556999993,
0.737555305816957,
0.288326143887046,
0.52138581058075,
0.136974849336303,
0.717508059794785,
0.0587931014871193,
0.205992789103646,
0.160248424932476,
0.689566250745936,
0.304050238944613,
0.643331486565681,
0.811579479748187,
0.347867644088281,
0.717047346624102,
0.275932726578756,
0.822086360688361,
0.480858841203553,
0.442930702326321,
0.437928101717461,
0.620075300624629,
0.760136925038014,
0.196718595082275,
0.265538855113806,
0.93331108658263,
0.717170080969655,
0.993124837518262,
0.342997678249608,
0.404320968968943,
0.589628921164958,
0.176179314114237,
0.950713526434597,
0.327184226981916,
0.464505458001283,
0.472099336549686,
0.464159076318219,
0.182013517796068,
0.530892438502466,
0.670151865421865,
0.342261942728545,
0.220458688782742,
0.974202104366479,
0.985593429294225,
0.032038168530929,
0.32206974845476,
0.702640775918328,
0.307970895109685,
0.14415610867746,
0.183499616656219,
0.750901597436006,
0.582084210394921,
0.803574917280849,
0.51103852247402,
0.778802805477196,
0.0691137723946542,
0.444349608590989,
0.495972885981189,
0.0622386094472551,
0.787347287306258,
0.900293855415794,
0.651867531077875,
0.963526601420495,
0.85100738138473,
0.97905175805979,
0.428032058956116,
0.323106717468755,
0.443210833912348,
0.610045576752185,
0.853999155971221,
0.113362699334213,
0.952307519480729,
0.0744578447539629,
0.0875648032350302,
0.937900948774955,
0.106496013284892,
0.40963455168979,
0.640541724227621,
0.414466908394576,
0.55379066036725,
0.824041340883841,
0.165368505830582,
0.135874870296509,
0.627616258164689,
0.676407028304603,
0.914677675773705,
0.696730030559343,
0.12075663642993,
0.410650561754895,
0.758968640006598,
0.908103923736189,
0.310944416705027,
0.410836170618812,
0.871630524691022,
0.161951797624096,
0.389887928212941,
0.299662583647139,
0.485058515092851,
0.83309876259095,
0.909708160399323,
0.339057671064072,
0.946461461925163,
0.862015679880053,
0.413515515818035,
0.0340262646945316,
0.799916628189346,
0.520011529102927,
0.443660816849983,
0.440458351951306,
0.934478437497503,
0.997451477217233,
0.264499692369485,
0.0998469428624245,
0.133326347513742,
0.892115950534174,
0.776253971167027,
0.0480040228217859,
0.588845980627856,
0.897010608062619,
0.458654584576681,
0.347814620634455,
0.805114531333146,
0.769599001281708,
0.758650791253266,
0.676745056024168,
0.931550798905804,
0.148538719466207,
0.976407639671307,
0.416609313998655,
0.981637482057157,
0.886115799604969,
0.755666985062727,
0.928098943516658,
0.74813147901936,
0.169182500415101,
0.96212520821119,
0.548048106743045,
0.689194029518028,
0.405786024595511,
0.988506458694351,
0.62367246654987,
0.403237501812744,
0.253006151063836,
0.723519409877956,
0.536563849326486,
0.145122101132349,
0.499773380579321,
0.584567872148272,
0.733968082225867,
0.396783988176279,
0.0432224567249522,
0.08178270239466,
0.201898519509425,
0.81282145800666,
0.840433493647926,
0.878643575533593,
0.744372256912464,
0.988972213114133,
0.855051214739239,
0.160981570445458,
0.970609694705629,
0.741167013878546,
0.916648555508185,
0.898708637756625,
0.489298492897907,
0.0858310554576251,
0.860833845967815,
0.0373465991752905,
0.775025084975653,
0.266619870097665,
0.0258530578696416,
0.398697551059861,
0.669857371910409,
0.278859208933478,
0.122216960472156,
0.206421220771233,
0.423981310065827,
0.621990341517139,
0.790989093385166,
0.157949391826032,
0.0187743292277559,
0.834211550110118,
0.239732094220692,
0.220672848737181,
0.647033007651117,
0.0801655878686186,
0.0993164238051122,
0.39140526409792,
0.0691378005170905,
0.954367638544351,
0.552386834543378,
0.0397474952227191,
0.695534652422897,
0.469035389585903,
0.938456132979345,
0.184833144855142,
0.554866445043528,
0.799289978481499,
0.222179744030433,
0.329891529553519,
0.0659098485791636,
0.248032801900074,
0.728589081079042,
0.735767220489572,
0.526892010833552,
0.850806041551198,
0.942188441726467,
0.95087332136504,
0.472796382602675,
0.733177534645972,
0.108822712725411,
0.491570712296092,
0.567389084290429,
0.348554807411765,
0.712243561033273,
0.214422091475884,
0.428720395280384,
0.811559985304046,
0.605827356039466,
0.497858195797474,
0.765927623382736,
0.158214190117183,
0.537605691020193,
0.461462275339971,
0.627249580168747,
0.476061823999538,
0.646295420195114,
0.182116024746614,
0.275351802015375,
0.868475164691208,
0.512007554765794,
0.341261650594539,
0.116507966125621,
0.240596635379175,
0.0770288710841112,
0.643399977424834,
0.0914026764647116,
0.0192173123449168,
0.594273298324213,
0.564199059533048,
0.752394846990888,
0.703096011049624,
0.055769771363479,
0.319783930815656,
0.0516508179957284,
0.768013332396752,
0.53420602229154,
0.480371213276112,
0.579573317235137,
0.140033377865345,
0.978229409539248,
0.345500940617873,
0.298247568448189,
0.51583510009378,
0.806963215957844,
0.925497148616937,
0.991896924093318,
0.453258636152958,
0.107613172897889,
0.267248726108693,
0.321733800378504,
0.619620727663683,
0.608510376703232,
0.438241766504125,
0.860217363042858,
0.685539247787343,
0.0816417434632973,
0.951620039973231,
0.70475656013226,
0.67591504178751,
0.515819099040618,
0.457151406657487,
0.379011052837135,
0.571588870404097,
0.776935337473143,
0.430661870832863,
0.339602202335187,
0.311141359764683,
0.911033084574637,
0.919175520035986,
0.451174737630028,
0.889262493648223,
0.264676460188197,
0.749422306078217,
0.405097593276341,
0.0716396756803802,
0.674919454229492,
0.396994517369659,
0.524898311833338,
0.782532627593043,
0.664243243478352,
0.846632112211842,
0.402153354791064,
0.272753619715922,
0.284873878715967,
0.262370717833923,
0.958292867968927,
0.366515622179264,
0.213990757341492,
0.663049427635525,
0.0424306639667743,
0.72980985638211,
0.120200834293012,
0.421441716803909,
0.301398726320546,
0.897136171766155,
0.852103587636772,
0.641000928655733,
0.208277531065176,
0.763136671745748,
0.560176448226057,
0.659452269160865,
0.652399164928309,
0.824852908414254,
0.408874574773421,
0.0574967582046505,
0.896492584560296,
0.0837940290029133,
0.454491275574309,
0.421390895927973,
0.866326656595956,
0.118734518587,
0.268023008139815,
0.268480010921359,
0.391488138302922,
0.552896886855782,
0.530850728755282,
0.349781005806188,
0.919412509035046,
0.744841486096774,
0.0128304334417127,
0.96184317300182,
0.474651342013223,
0.133031267734725,
0.383284889340068,
0.776050068333768,
0.0301674390352179,
0.23538847697684,
0.417050996989501,
0.238444970566055,
0.998525148722588,
0.977227445215558,
0.89789723972692,
0.650924313650897,
0.802080353629813,
0.30677181403468,
0.708421071855547,
0.698572937724447,
0.390565843037593,
0.162912346964196,
0.119963833186759,
0.256892499167888,
0.281646865551196,
0.387986841326573,
0.525372510554908,
0.673135003854118,
0.940883728182355,
0.0562232388445284,
0.0229160096603055,
0.8602962372174,
0.801064725406964,
0.0357464431020182,
0.822139409753559,
0.275716066954525,
0.168777710836743,
0.205424299093627,
0.0517661352882935,
0.198945149871961,
0.440812776070467,
0.468817132277795,
0.437390120438016,
0.439337924327393,
0.446044577027692,
0.335287359699275,
0.0902622375126287,
0.248124930191843,
0.642059174199616,
0.798683309368176,
0.946697867916291,
0.032625016771548,
0.961595656332372,
0.0666617011030492,
0.289517516405097,
0.243242521417906,
0.454648542429622,
0.814890026960005,
0.916377525737685,
0.395532270611977,
0.871113265804534,
0.93929353539799,
0.255828507363716,
0.672177990745836,
0.975039978500009,
0.077967916651614,
0.947894058166022,
0.14381768887109,
0.283392215745241,
0.999660193454316,
0.342762838743051,
0.724204991815707,
0.468477325266449,
0.780152959646728,
0.1635429161431,
0.914521902294141,
0.115440318880342,
0.253805153655729,
0.162646832485984,
0.757499493079958,
0.0524884630239049,
0.109344699936614,
0.790124510317167,
0.0140841188906152,
0.176006401039663,
0.0796420262566032,
0.257326640774182,
0.630654943934947,
0.894532053216608,
0.173704166046206,
0.0261872140812628,
0.76564531855548,
0.112997700978535,
0.282015721444979,
0.437823309301316,
0.0880376790128824,
0.359983638096593,
0.385717367001678,
0.231855367883973,
0.643375854307495,
0.385377559990332,
0.574618206627023,
0.367580845657541,
0.853854885256782,
0.354771165808091,
0.531123761800641,
0.768376787085262,
0.470211485154094,
0.784928915922031,
0.931023619571246,
0.227710977768391,
0.837417378945936,
0.0403683195078598,
0.0178354876198971,
0.851501497836551,
0.216374720547523,
0.0974775138765003,
0.108828138145072,
0.84702966448247,
0.992009567093109,
0.282532304191278,
0.873216878563732,
0.757654885648589,
0.395530005169814,
0.15523259954305,
0.195478194484244,
0.483567684182696,
0.515216238105305,
0.581195561485922,
0.715423052066668,
0.158592091947138,
0.966573121476254,
0.290041258693692,
0.526172937604679,
0.820428006267374,
0.644812424501782,
0.0572966994053203,
0.588804792886975,
0.115023909190215,
0.842225615327352,
0.519828412458221,
0.342734887424267,
0.679642993807626,
0.56019673196608,
0.360570375044164,
0.531144491178516,
0.776571452513603,
0.458047889386326,
0.63997262978925,
0.623601116996073,
0.450057456013773,
0.922504933980529,
0.496817995094144,
0.207712341196701,
0.318034939150342,
0.652050595102855,
0.403190535680945,
0.801602623333038,
0.167266832742499,
0.984386097166867,
0.517025675399707,
0.325858924689637,
0.95095921817746,
0.807066934093398,
0.852031862294316,
0.771387223979173,
0.45187935812952,
0.909328561699636,
0.360192016866148,
0.566903267785396,
0.751554176561327,
0.880020429324368,
0.909638155209663,
0.431197169903292,
0.440217160824787,
0.270208529788167,
0.96234166154747,
0.216788612872729,
0.728256419174493,
0.602314290871059,
0.840389729868802,
0.178313875188266,
0.524819224385926,
0.337207724962946,
0.386026216384967,
0.842854163536268,
0.989258320065801,
0.789216752531574,
0.644456786869307,
0.156525152342639,
0.773602849232779,
0.161482461803352,
0.482384077032276,
0.724562067410239,
0.96854939589675,
0.33441593886093,
0.495949290923751,
0.420428753560609,
0.243744500560567,
0.856141307789898,
0.987332021346005,
0.995298677121894,
0.736161736648605,
0.896970176090007,
0.426495847025186,
0.176378897007731,
0.167178705878173,
0.388837508106994,
0.393167510346122,
0.895435125052666,
0.991151798978053,
0.233557239749263,
0.0737489997752705,
0.515971022898318,
0.570764964712209,
0.459775216625899,
0.358825186434586,
0.560023284312348,
0.248991968691811,
0.00328197283823135,
0.716548436654987,
0.0225948174589289,
0.164764434641583,
0.198932513221601,
0.747156884869168,
0.133313830072672,
0.533348452548193,
0.243106175792919,
0.553742583633281,
0.77709295310876,
0.0992474831171555,
0.541074604979285,
0.772391629764993,
0.835409220231422,
0.438044780603631,
0.198887476324517,
0.0117881167734918,
0.605223486481804,
0.587724984431511,
0.404955627119614,
0.50065861153447,
0.578876782943903,
0.638512867334538,
0.57440761130974,
0.0948478058422207,
0.209277831581085,
0.0341828274699779,
0.453672992276807,
0.769301116359095,
0.283174796161789,
0.456954965115038,
0.48584955254842,
0.305769614086379,
0.621719399756621,
0.684782066235683,
0.0529264984898858,
0.755033229829293,
0.218130518318215,
0.296032674282804,
0.308775812996913,
0.995223471426975,
0.395280157865621,
0.849850417976198,
0.767615100726306,
0.230689377631382,
0.287895198579829,
0.966502577516484,
0.242477494404874,
0.893118685061633,
0.554227561482334,
0.647433121990148,
0.393777296130442,
0.133104344426237,
0.285945988859025,
0.968184907905844,
0.227952150268458,
0.49522382044011,
0.00236773491016018,
0.681625142545265,
0.264524936333543,
0.28554253153761,
0.138580107194642,
0.750374489347625,
0.591312145623989,
0.760299506951263,
0.435156555117647,
0.644238644113875,
0.515332736314895,
0.653287073435861,
0.940271318862341,
0.824108549311808,
0.648510544397175,
0.335551476262301,
0.673958967288006,
0.416125644657819,
0.566240853893683,
0.961854165867834,
0.382628221708642,
0.808718348298556,
0.854972850463806,
0.936855783656638,
0.456151469823044,
0.248750146594248,
0.0699601276172139,
0.742097458682068,
0.21693505403443,
0.297912277885672,
0.237321279122178,
0.21930278894459,
0.979537420430937,
0.501846215455721,
0.504845320482201,
0.118117527159917,
0.252220704337685,
0.0961574656405288,
0.87841703411118,
0.687377259455331,
0.740396110220065,
0.393749769960414,
0.340664332425531,
0.680667428616745,
0.21785831880656,
0.989174876822706,
0.0162189044133848,
0.891817286094566,
0.405300521480525,
0.582459758307068,
0.853671451496739,
0.787928743189168,
0.391178106139963,
0.708644301960545,
0.724784526380144,
0.847329576428667,
0.957394448554793,
0.794744653997358,
0.589427034645075,
0.174329502123561,
0.0926569314173688,
0.826748313767253,
0.393632291068152,
0.0721943513826441,
0.328594529222974,
0.898477612016014,
0.190311878542561,
0.580815233560659,
0.994635077656542,
0.0687289121880796,
0.26819249301599,
0.735031187410947,
0.462478682148493,
0.608856825441521,
0.41569861556203,
0.680337001420714,
0.598031702264227,
0.431917519975415,
0.572154287049619,
0.0033322232790907,
0.0143772778168215,
0.425825738546357,
0.791260966468258,
0.405555384422445,
0.134470040041241,
0.516045492848403,
0.252884960385452,
0.0918644881303722,
0.3107901463801,
0.842311995030526,
0.266193990253934,
0.40344707826313,
0.669060308797779,
0.659826281322085,
0.475641429645774,
0.997654838020753,
0.558303892872438,
0.665953308653996,
0.57847007111575,
0.55293897052898,
0.734682220842076,
0.84666256413174,
0.287970157474266,
0.197160902990569,
0.455519389573261,
0.703668773501957,
0.877497904411283,
0.0535510913718264,
0.135586293011711,
0.449652190995241,
0.0568833146509171,
0.149963571294194,
0.875477929541598,
0.848144281119175,
0.555518955716639,
0.00994796911717764,
0.364189773501917,
0.808403916102091,
0.10181245724755,
0.674979920347678,
0.650715910666956,
0.368006447501484,
0.0784269981451458,
0.319776218999073,
0.027832728823569,
0.554068427790919,
0.317431056554164,
0.586136621696007,
0.220021735979254,
0.895901128135576,
0.139075591759326,
0.954703957286991,
0.742563691801654,
0.427045749699253,
0.151864859811899,
0.198083080909254,
0.130714522735548,
0.0293627637575207,
0.25163417228108,
0.26630081574726,
0.479014954752761,
0.308517486931997,
0.416264387041453,
0.354492884294359,
0.156661768051172,
0.971783342758093,
0.364440853411537,
0.520851541553089,
0.780187258394522,
0.466253311124748,
0.195831461435105,
0.430903169061478,
0.834259758626232,
0.274258459580251,
0.750679388060551,
0.862092487449801,
0.828326887836832,
0.0681104446147151,
0.448229109145808,
0.048348623350425,
0.964011572750291,
0.587304700905133,
0.00305258017175485,
0.706575264086283,
0.0143504501387246,
0.154917439983654,
0.904658345461198,
0.145064972874273,
0.184280203741174,
0.156292517276617,
0.411365789087194,
0.663295158959597,
0.464810004674275,
0.827630176128647,
0.0177880427882951,
0.621471772725448,
0.79941351888674,
0.382228896199832,
0.142323313812876,
0.579600776815601,
0.84848220732458,
0.338154775713642,
0.0105039454114176,
0.682741965950812,
0.612413235293894,
0.76118333393763,
0.544834452934952,
0.440740122665064,
0.829293778552345,
0.993063562080759,
0.489088746481151,
0.793305350836974,
0.580368262985893,
0.492141326652906,
0.499880614457596,
0.594718713124617,
0.64705876710222,
0.404538959453133,
0.739783686464552,
0.831338970843395,
0.560831477195412,
0.151149475086084,
0.49463412933733,
0.0256414814040258,
0.978779651214732,
0.512422172125625,
0.647113254129474,
0.778193169635811,
0.894651068791119,
0.78943656794235,
0.357793946451412,
0.743133275650038,
0.127591343190331,
0.368297891862829,
0.425875241135189,
0.740004578949886,
0.129481225334798,
0.970709694535802,
0.180744701149289,
0.958775003887143,
0.9637732561509,
0.66983344763044,
0.752080354258456,
0.544141518671131,
0.161974774283345,
0.251960968716052,
0.138860231795749,
0.809033541385566,
0.656499928169185,
0.878643918260301,
0.640372511763299,
0.217331404898936,
0.0297933928807235,
0.13500664110063,
0.242972886302961,
0.00857304362979394,
0.647428813226255,
0.890086140432435,
0.786766213731266,
0.542079881551713,
0.679522708374785,
0.144560159717016,
0.285213157201751,
0.807114051565115,
0.512858051579845,
0.71108839833694,
0.54711863004934,
0.642339277380304,
0.68179809240708,
0.72786333166429,
0.601114280801785,
0.64557134855798,
0.397696778829068,
0.353194635060241,
0.189712866763451,
0.559671553112414,
0.605155603776293,
0.328573098559199,
0.368705094032318,
0.261655531479817,
0.207217016353839,
0.00907760579561703,
0.478986936844414,
0.237010409234562,
0.144084246896247,
0.721959823147375,
0.245583453330017,
0.791513060122502,
0.61204596357981,
0.0323496665956218,
0.333592941674214,
0.291568671488934,
0.176909826312638,
0.618806098875965,
0.0986827225883877,
0.689767877892483,
0.329894496747243,
0.645801353103389,
0.332107154807126,
0.0116925891543238,
0.373664684302017,
0.933221436074572,
0.657263937712304,
0.771361463131086,
0.286416070669152,
0.846976804475755,
0.331033016243499,
0.891571674445445,
0.175549903034954,
0.699738110275817,
0.1532272054596,
0.382766919388793,
0.708815716071434,
0.632214142304014,
0.619777329089016,
0.852899962967681,
0.354173965451389,
0.865360782419034,
0.644413023090182,
0.966219929031199,
0.897710449014656,
0.978005964764397,
0.257788600054471,
0.074620274861632,
0.5968120631747,
0.356471322642859,
0.764388152754115,
0.926706559921944,
0.0022726752805862,
0.0964953075612408,
0.938399149076268,
0.375937359582604,
0.0297167431701518,
0.595663086322911,
0.147298822713689,
0.316132813839304,
0.442639890798666,
0.478331838957189,
0.207704487819087,
0.61818979383362,
0.178069948767344,
0.360931693278687,
0.000956712756751437,
0.88688566530444,
0.993145836048362,
0.620734041845768,
0.739785627806459,
0.347319801034089,
0.48609482379914,
0.38419865043098,
0.313539729599627,
0.383805272348134,
0.362204614729716,
0.571328329654098,
0.458425547209766,
0.959016677904416,
0.927799652296956,
0.222813699963882,
0.88572323782636,
0.930072328043204,
0.319309007525122,
0.824122386436966,
0.306009687160146,
0.349025750695274,
0.419785472759877,
0.453308509873836,
0.665158564534578,
0.862425363558543,
0.931640348831024,
0.872863052353665,
0.480615156926501,
0.109710297598369,
0.23379474516669,
0.481571869683253,
0.996595962902808,
0.226940580749391,
0.10230591152902,
0.736381590243607,
0.57426038178348,
0.588400735328161,
0.120580240674587,
0.887800111383107,
0.972206008141956,
0.482784855404303,
0.459128440571543,
0.430631554886061,
0.441801533308719,
0.386928092868499,
0.653445254849943,
0.327524770669418,
0.317000420446042,
0.972754262375065,
0.151647156640723,
0.623010107606188,
0.321780012604678,
0.5714326294006,
0.0763186174800241,
0.986938577139256,
0.433857992493481,
0.00795896584538695,
0.859801629027259,
0.914473149419982,
0.117669263443756,
0.0935963741939498,
0.396045019103235,
0.114265225880903,
0.320536954943341,
0.498350930632255,
0.850646816590171,
0.894797337192482,
0.0867516654947548,
0.971227057264758,
0.782597448109927,
0.0589576731710498,
0.454011912203399,
0.24172588868147,
0.489589228522773,
0.895813445512119,
0.62865398154997,
0.143034482907054,
0.223338215715875,
0.945654401996012,
0.115788745282119,
0.374985372822259,
0.568664509136539,
0.437568757886798,
0.946418002222859,
0.644983126616563,
0.424507335026053,
0.380275994716341,
0.652942092927612,
0.284308963587651,
0.294749143670662,
0.770611356371367,
0.377905337781601,
0.690794162773897,
0.884876582717931,
0.698442293190603,
0.189145092940491,
0.735523398842441,
0.593239629917424,
0.275896758900907,
0.706750455641537,
0.37583707756169,
0.334854432071957,
0.160762367379275,
0.61756296624316,
0.82444366059473,
0.0565758128913938,
0.246216947327469,
0.967478143967445,
0.279914028607269,
0.19187134885782,
0.0832668887839033,
0.654899401429528,
0.76053585846002,
0.520835646670701,
0.601317403652387,
0.405518984610922,
0.945342981696754,
0.981593398368728,
0.0584610770728723,
0.229651945284406,
0.276342541573729,
0.829072433909901,
0.607557283066007,
0.967136704813287,
0.713949016162171,
0.305999575790949,
0.156281797288117,
0.44947241453895,
0.899239205708373,
0.432178556189024,
0.156222869714826,
0.275076283270063,
0.767032988726642,
0.316985237094101,
0.892639249513223,
0.591476648855711,
0.373561049985495,
0.138856196840692,
0.558954792357494,
0.653475079058425,
0.330727545698512,
0.642221681141398,
0.308374480022292,
0.0912634036928711,
0.163057327812099,
0.909691883674679,
0.496782388769455,
0.108400309043192,
0.891285281577746,
0.555243465842327,
0.338052254327597,
0.167627823151475,
0.384315899286566,
0.945609537859265,
0.1347645274991,
0.0982649149830755,
0.251609113184553,
0.291046324787217,
0.547737329522025,
0.150848318892926,
0.723224881441903,
0.703960199236851,
0.425924602162989,
0.490257869702884,
0.0209454363309524,
0.318563851210551,
0.0817345180929333,
0.394506486316447,
0.457420048051244,
0.640689310916089,
0.0479815649092112,
0.788147594215417,
0.282910991591826,
0.356356045397164,
0.879410997908288,
0.445968319403924,
0.266047928606182,
0.376193386212081,
0.554368628912777,
0.157333210183928,
0.931436852520069,
0.892420883240374,
0.324961033335403,
0.315752751340974,
0.838030420633978,
0.459725560834504,
0.414017666789711,
0.0896395338185316,
0.750771885621721,
0.961754996311737,
0.240487852711458,
0.473996766597962,
0.665715195548588,
0.666412454874447,
0.964254636300846,
0.68666063187954,
0.984976306084998,
0.0459891543937797,
0.0811671177303265,
0.442396354136242,
0.686678465309869,
0.129148683105199,
0.230543947885998,
0.969589456901694,
0.485504728502363,
0.109954945328624,
0.415557776305619,
0.751552657108546,
0.486148332006367,
0.969926405218396,
0.908885867292474,
0.417585184060775,
0.862347287993109,
0.233846900162216,
0.73333793540175,
0.700377708627087,
0.693572460996719,
0.1473556017258,
0.790017242445618,
0.44434434661844,
0.109110598037537,
0.0305050946914149,
0.918341113216403,
0.774825793586125,
0.696917549565862,
0.882595749517249,
0.461486425000004,
0.681893855185199,
0.928584903911028,
0.54265354273033,
0.124290208855779,
0.615263368755236,
0.671802225835529,
0.354834156741777,
0.584852825191269,
0.157306953872231,
0.464789102536062,
0.000410601031226386,
0.908859611446438,
0.950937434542429,
0.970337006249622,
0.817745478273251,
0.368522618137543,
0.83268429424273,
0.0515923784354666,
0.101860553073632,
0.533062002404156,
0.745164839432186,
0.249216155265093,
0.323079244384113,
0.189509185584965,
0.358326753302629,
0.353584339075528,
0.107850298801367,
0.133152546423093,
0.0505018881757287,
0.990446048318616,
0.594638971423097,
0.732395743360927,
0.919030951763983,
0.137292513687765,
0.856685952682367,
0.534294320053558,
0.809094739988956,
0.211520108958483,
0.119147144779166,
0.966401693861187,
0.676309211494545,
0.119557746276053,
0.875261304841964,
0.627246645571313,
0.0898947520600142,
0.693006783115215,
0.995769263708857,
0.922579046302745,
0.744599161550682,
0.0976298167824884,
0.45564104824124,
0.489764000517206,
0.346845972047581,
0.778720292625353,
0.679273186567832,
0.705172725350211,
0.13230463123522,
0.7871234853692,
0.838325271773304,
0.182806519410948,
0.777569533222154,
0.432964242730739,
0.915202263237537,
0.696600484520477,
0.570256756418504,
0.771888215454243,
0.230894804108373,
0.379351495941799,
0.983408324412726,
0.350041948887539,
0.345753189802986,
0.659717535907271,
0.469599695163593,
0.221014494179289,
0.286964181012923,
0.559494447689268,
0.914021277294504,
0.28273344472178,
0.482073493526351,
0.658620438379524,
0.380363261504268,
0.937714542233252,
0.148384438896731,
0.727209233551849,
0.716434834392944,
0.827657625464563,
0.432381958436399,
0.848739466093825,
0.614781110368101,
0.270707229744041,
0.0315459850391121,
0.392350643124595,
0.70367147247478,
0.946748248276649,
0.0889511271794099,
0.273928228427623,
0.718636463265231,
0.319845931287783,
0.653279724835083,
0.702044787677957,
0.669887880175322,
0.999032914638069,
0.361762323119567,
0.139487575338915,
0.220047408351697,
0.64872650413249,
0.698982023028183,
0.134068685646201,
0.931459948854269,
0.181055516088873,
0.792689124025725,
0.311823209892876,
0.118770057856464,
0.941073562922456,
0.039032442979064,
0.835204892715069,
0.768731187921358,
0.471414401415463,
0.683944358343233,
0.383512297823798,
0.742121631159504,
0.715490343382345,
0.775862940948393,
0.445793103168622,
0.662238591658994,
0.864814068127803,
0.719721332061906,
0.380875054458564,
0.184659998949924,
0.373001056431328,
0.0829198416708595,
0.854547879590908,
0.372033970603735,
0.444682164790426,
0.994035454929823,
0.592081378955432,
0.0934086689229164,
0.693017477492344,
0.726150064601633,
0.0248686173115245,
0.874072993581217,
0.518839188627358,
0.336691827670062,
0.992843051903342,
0.459912751084153,
0.375724270649126,
0.82804794415275,
0.228643939005511,
0.84713867253025,
0.511992302030321,
0.612156236829309,
0.589260303224093,
0.227482644947005,
0.388019177777702,
0.0350534063927147,
0.889721236605999,
0.252833245439843,
0.754774738454621,
0.270596291064562,
0.437493244855429,
0.127775794420287,
0.353516132735422,
0.292041123980675,
0.499809765024022,
0.79819829799151,
0.286076578444837,
0.0918911435137927,
0.891606966914426,
0.979094055937181,
0.818041208581087,
0.916475584225951,
0.853167049052737,
0.336880396742784,
0.253167411430351,
0.846010100490418,
0.796793147826937,
0.628891682545138,
0.674058044177507,
0.0254370863667862,
0.476030354609727,
0.186050345742167,
0.637593323661756,
0.0652906573681583,
0.413532991154833,
0.0256125009737967,
0.100344063760873,
0.303254227295171,
0.27844574641364,
0.855118802215494,
0.573850518359733,
0.715938991269068,
0.982894596635781,
0.927366651560816,
0.00798011478408245,
0.482704361194141,
0.725564949086665,
0.294056693228919,
0.574595505173595,
0.617171915535429,
0.273150748700439,
0.392636713289021,
0.533647499295719,
0.126317797753177,
0.729517110031805,
0.786814911191731,
0.972327898243595,
0.52631025739308,
0.415706593271208,
0.646385942421102,
0.551747344225527,
0.891736947880935,
0.832436288163269,
0.189340667421623,
0.957027605249094,
0.245969278852441,
0.214953168395419,
0.0573716690099666,
0.549223506613273,
0.493398914809059,
0.91249047122546,
0.123074024507345,
0.209337906078127,
0.895385067861241,
0.0504406756024997,
0.21731802086221,
0.378089428589721,
0.776005624689164,
0.51137471455679,
0.952684933763316,
0.393177539758933,
0.78452546325723,
0.345321646586676,
0.926825039520313,
0.910843261010406,
0.0748387561528193,
0.713639950246382,
0.883171159254001,
0.601149014011561,
0.129346543051929,
0.529557101209442,
0.152896357771427,
0.0210834904672035,
0.36199338890705,
0.342237025193049,
0.978111095716297,
0.607962668225152,
0.557190193588468,
0.0354827642606025,
0.157186174372763,
0.0505891079318659,
0.947973235951724,
0.280260198880108,
0.259927014009993,
0.843358303347304,
0.330700874948269,
0.477245035337864,
0.221447731471363,
0.106706499171772,
0.988619749894654,
0.17413266523468,
0.499884039396366,
0.773145212686223,
0.519454311821356,
0.426709078451017,
0.683988473696629,
0.594293068439836,
0.140349028231739,
0.567159632484969,
0.195442081985735,
0.269695571283668,
0.096716733228749,
0.348338439757162,
0.290779061750872,
0.458710122135798,
0.690575464950211,
0.268890157001508,
0.066672789895289,
0.247765658073018,
0.304372921727771,
0.223858964268052,
0.298354766004884,
0.252346157213834,
0.504119163613822,
0.558281780480538,
0.0957044600954766,
0.834820038562091,
0.0355268153527411,
0.31715219156684,
0.941526537733863,
0.0241465647817341,
0.49128485680152,
0.441410576664568,
0.797291777467957,
0.0107391686228752,
0.868119655115586,
0.481280250698924,
0.605032237062711,
0.00846868288166294,
0.0484398827182315,
0.800474319048447,
0.278164254165331,
0.14515661594698,
0.148812758339947,
0.568943315916203,
0.60386673854844,
0.839388223290158,
0.837833473383371,
0.670539528443729,
0.0871538808975154,
0.142206394645482,
0.894398493177443,
0.385508646902399,
0.394552551859316,
0.398517656325604,
0.943790427382938,
0.490257011954792,
0.233337694422033,
0.979317242735679,
0.807409203521632,
0.174864231690236,
0.00346380705175167,
0.298694060323152,
0.616274808820465,
0.80075558498537,
0.309433228946027,
0.484394463470389,
0.282035835218633,
0.914465466008738,
0.492863146817714,
0.330475718402525,
0.714939784591524,
0.771027400983045,
0.475632334349506,
0.863752542931471,
0.339970716899247,
0.0794990724322847,
0.703140765755969,
0.177804189816957,
0.750038601341675,
0.790294646653484,
0.320010584462439,
0.644437094053457,
0.175803293090222,
0.714563136321755,
0.0429547499133995,
0.11959372047316,
0.204820147810886,
0.276292444335433,
0.0989109627431775,
0.0122293508668567,
0.451156676025669,
0.10237477026059,
0.310923411190008,
0.0674314843804722,
0.90313035524596,
0.620356640136035,
0.551825948316523,
0.185166189998931,
0.534822105679112,
0.0446890946685751,
0.515641908401457,
0.249761889804975,
0.81571649565162,
0.991274243216624,
0.113514432270785,
0.155687212085206,
0.0707733151832471,
0.816655198026754,
0.333491401902163,
0.820811916524922,
0.606949844214576,
0.653501986364602,
0.465249010112718,
0.78275313777046,
0.368065122220696,
0.508203760026118,
0.902346858243619,
0.572885270031581,
0.784496204361551,
0.00125782052113573,
0.585114621364099,
0.235652880387219,
0.103632590781726,
0.896038032554108,
0.303084364767691,
0.00676294556202504,
0.516394672690143,
0.854910313084214,
0.191929136026618,
0.0512167779035944,
0.899599407752789,
0.707571044428074,
0.300978668174231,
0.715315903404409,
0.698845287179037,
0.414493100445016,
0.871003115489615,
0.769618602827945,
0.231148298471769,
0.204494517391778,
0.590430518887206,
0.838098142686346,
0.85799650375638,
0.0556795289999244,
0.620851279991144,
0.226061625977075,
0.563883289026042,
0.523198137769102,
0.798946896008657,
0.348379493387593,
0.524455958755899,
0.384061516907095,
0.584032373774812,
0.628088549537625,
0.280099549461202,
0.887116738542503,
0.634851495565312,
0.796494222151346,
0.742027051626717,
0.826780631591929,
0.84771100005494,
0.641626458913845,
0.534351675554342,
0.14868966776351,
0.356942361852593,
0.233196962733379,
0.563182768674187,
0.227945477342208,
0.00281556509566287,
0.794331067145956,
0.432439994733986,
0.593246083982869,
0.632429209366641,
0.290436498024704,
0.648925612982793,
0.253280489357785,
0.51649812400178,
0.212808902008836,
0.776478627126887,
0.315445020010436,
0.561188395396428,
0.300934585417125,
0.699506536917531,
0.145220768705579,
0.929023135420411,
0.979606086378734,
0.0323375067824207,
0.563874630520062,
0.776100308064418,
0.774364558409138,
0.39065526164633,
0.623811308119358,
0.415991017322983,
0.925006937200672,
0.772500975882868,
0.772933379175576,
0.15820389946839,
0.335683744091393,
0.000878856052122477,
0.161019464564053,
0.130014810771688,
0.433318850786108,
0.754265549012583,
0.76244402060399,
0.723755349276473,
0.403191161529716,
0.0157245094961135,
0.240253472812592,
0.616000063538551,
0.792203136623,
0.555698492823028,
0.177188458469318,
0.093137722040125,
0.255205029740559,
0.322409227174897,
0.0221608569948752,
0.234811115653632,
0.354746734422979,
0.586035487514937,
0.0109114237180499,
0.129111292366456,
0.976690749161267,
0.634722731837408,
0.545102309689439,
0.901697685896278,
0.407223707254615,
0.318035688865015,
0.0599015853646684,
0.742907451811669,
0.318914544917138,
0.220921049928722,
0.872922262583357,
0.752233396168907,
0.975186598941305,
0.635366282721686,
0.475988744979719,
0.378377760471021,
0.6510907922178,
0.716242217792311,
0.994377824009572,
0.4432939288408,
0.271940710615339,
0.171566282478891,
0.536431650880925,
0.527145740355898,
0.493975509653788,
0.5585925078758,
0.76195685600953,
0.848722244076767,
0.144627994925076,
0.77286827972758,
0.977833536908884,
0.121318743620682,
0.407591011099327,
0.522935846132662,
0.0230164290512988,
0.814814718819603,
0.840971534997677,
0.0829180144159673,
0.557722170165611,
0.159886079914815,
0.30383906481035,
0.430644432283307,
0.912119476083722,
0.279025663285994,
0.066010715004993,
0.388108220597779,
0.657403423757015,
0.717101507222793,
0.104350437924429,
0.651781247766587,
0.160395435597932,
0.376291148539768,
0.823347530245477,
0.696827086478857,
0.903436888895667,
0.317323039433604,
0.255419593888996,
0.665393744905197,
0.166045283510371,
0.400047588814072,
0.438262024167116,
0.143878819953594,
0.521366332434754,
0.845853035732104,
0.666814666086256,
0.544382761951714,
0.660667754086045,
0.507786201083933,
0.627300776367681,
0.218389923785995,
0.667672280998748,
0.931139841178032,
0.649034356069301,
0.579791756616808,
0.210165504464025,
0.715045071074294,
0.967899977214588,
0.86756892822104,
0.432146578297087,
0.0722504146733556,
0.519350175521965,
0.592542013895019,
0.448541563678785,
0.342697705301781,
0.289369100373876,
0.35197845210879,
0.660020745201046,
0.544788694262872,
0.017372196548326,
0.826066028711417,
0.944836283542605,
0.455634221181103,
0.969944848665011,
0.466202615511698,
0.301487256447546,
0.636759514285605,
0.0105853769977509,
0.962155010533591,
0.144545714903877,
0.637886153831094,
0.180544933853925,
0.812217995902625,
0.569025994543464,
0.829579289923226,
0.392009752053772,
0.779191499007489,
0.544624360997521,
0.359909728802699,
0.646760427228529,
0.976770939294608,
0.432160143941715,
0.166110602284833,
0.569312952723966,
0.8807017076205,
0.508808308052276,
0.858682053097841,
0.23268015926363,
0.16882905278766,
0.403470747360713,
0.250052356277617,
0.994895081499077,
0.348307030437657,
0.70568657745872,
0.964839929698426,
0.814509645949355,
0.00717383344060454,
0.60159944351837,
0.825095023412767,
0.969328843974196,
0.746145158887908,
0.4629811767782,
0.149873777362459,
0.558363154324872,
0.0320071708560023,
0.979453067285685,
0.950372906844305,
0.811198670329153,
0.524077427817545,
0.310282635181343,
0.457959097092021,
0.500848366646491,
0.742442779123058,
0.624069699376854,
0.0701613193704567,
0.623144486277897,
0.132878006963468,
0.928843372468298,
0.855824646007188,
0.301707059751128,
0.33231411936335,
0.105877001819144,
0.296602140784544,
0.680621149801007,
0.811563579277863,
0.261442070017309,
0.495130795750362,
0.818737412718468,
0.86304151400134,
0.320225818697468,
0.788066256227003,
0.609186672423587,
0.783206995475668,
0.937940033589462,
0.167549826282798,
0.81521416633167,
0.917393100409486,
0.117922732661442,
0.626412836195162,
0.441470528227031,
0.428205368308446,
0.0843719328215215,
0.942318894873522,
0.170648146965843,
0.708441632664037,
0.0124802137783171,
0.793792633709401,
0.841319639627505,
0.941323586712276,
0.649617279250928,
0.143026699378633,
0.273637705609965,
0.755494281070071,
0.439628840163177,
0.954258855876633,
0.567057859882274,
0.701070910646148,
0.449389651161334,
0.38579527213508,
0.564112424181827,
0.769615469858803,
0.173861527896422,
0.173299096139753,
0.552822464868809,
0.111801561020222,
0.340848922422551,
0.368036630734818,
0.0291946609640469,
0.458771655549655,
0.994449467395642,
0.470665189191077,
0.886977023858101,
0.0788213997515018,
0.412984084064599,
0.0576251703582821,
0.787263032415539,
0.425464297842916,
0.851417804067683,
0.628582672043044,
0.366787884089531,
0.50103508285295,
0.771609371421677,
0.640425590165158,
0.25652936345736,
0.211238211119193,
0.59468444557613,
0.823587223339634,
0.91230912176534,
0.0440740962718027,
0.209382495009053,
0.476421545481506,
0.813689566130605,
0.383244022905475,
0.649720641621258,
0.366512030533753,
0.495045583925697,
0.990569564509471,
0.734548661268572,
0.524240245355405,
0.449341219593464,
0.728998128198552,
0.994905434546483,
0.336318242985903,
0.807819528415715,
0.40788951814542,
0.393943413344186,
0.595082560365592,
0.833353816453998,
0.245361217411869,
0.223665231942975,
0.200141700077868,
0.746396300264819,
0.995274603364651,
0.840567290243026,
0.00292566372217874,
0.206512814483844,
0.435251735353494,
0.826512887061812,
0.118821935783523,
0.479325831625297,
0.0358953820708652,
0.59524348173069,
0.293015397290241,
0.41913940497634,
0.244964122886287,
0.659527427823994,
0.914184989367698,
0.235533686930097,
0.394076089092566,
0.438425234257442,
0.684874906523561,
0.123074216825456,
0.433330668338263,
0.0211931490438027,
0.930893745241172,
0.841220186949344,
0.41513656285365,
0.525976305606764,
0.674574002937681,
0.660497780265518,
0.749641537549739,
0.874715703015549,
0.406894080064676,
0.74491614091439,
0.715282993258574,
0.409819743786855,
0.951428955398234,
0.150534728146407,
0.236332630848667,
0.0702508907160959,
0.629860560237365,
0.272228012919532,
0.665494372446786,
0.922875957527606,
0.691367417895872,
0.910458495798734,
0.5824033853516,
0.605552406797908,
0.14599218226317,
0.976479474444166,
0.0439776405896887,
0.830867089252391,
0.0995536908039608,
0.477308309393613,
0.852060238296194,
0.0304474360451323,
0.318528495877296,
0.267196800684182,
0.556423741651896,
0.993102498814977,
0.927694580949701,
0.306065278735974,
0.867818201364865,
0.334588661014377,
0.0509814191847022,
0.583101194157778,
0.744408404801231,
0.00241037411727494,
0.733635922769846,
0.980741035649898,
0.0726612652990321,
0.36349648254155,
0.252969048103769,
0.738155637745818,
0.286372439603495,
0.944336466465302,
0.648614133078891,
0.868775824955095,
0.549888872797549,
0.794606315807722,
0.845255298933599,
0.593866513852899,
0.625473404594452,
0.944808990203221,
0.0711748227808507,
0.477533642424985,
0.975256426248353,
0.389703318658147,
0.744730443574828,
0.531680167434588,
0.382805817007462,
0.672425024058868,
0.837745446170562,
0.250624017906666,
0.00701368460758295,
0.888726865820925,
0.833725212530105,
0.751422089874475,
0.8911372399382,
0.56736113483429,
0.732163125058712,
0.963798505237232,
0.93085761737584,
0.985132173628142,
0.70195414298305,
0.217230056513674,
0.929468639627783,
0.35056827559628,
0.0860058814687682,
0.479357511959671,
0.14517459093834,
0.931261180402367,
0.0732240253469087,
0.770647995532792,
0.876070170139927,
0.144398848127759,
0.248181637957777,
0.851326595922619,
0.534102166785906,
0.992912081532605,
0.383006762891545,
0.916907983793369,
0.665337105125811,
0.220752209062107,
0.167532001700034,
0.672350790199056,
0.109479074417371,
0.00125721376447809,
0.42377287960787,
0.000616313889909682,
0.568618348598768,
0.15593600420092,
0.964414819592803,
0.499475965508947,
0.141068177363401,
0.666368962110192,
0.716706022022621,
0.070536816525523,
0.0169372372408105,
0.802711903491389,
0.549894328950855,
0.162111828644812,
0.733973083893756,
0.623118354297764,
0.932759824177604,
0.610043253568021,
0.767517202891185,
0.18094146166972,
0.46136984949064,
0.301619369211429,
0.173853542736663,
0.844376612382185,
0.218527353004798,
0.839190648328136,
0.0651288209786307,
0.386059354704832,
0.51154143806153,
0.174607895396002,
0.38731656846931,
0.9353143176694,
0.175224209751573,
0.955934917068079,
0.0912503214046593,
0.139639028878714,
0.455410882111364,
0.232318499233722,
0.806007990988907,
0.172116903668324,
0.302855315759245,
0.822945228229717,
0.974828807625374,
0.8527496447101,
0.985057056874529,
0.708801891053469,
0.475867999007864,
0.917816881052133,
0.318845144155829,
0.243385201433387,
0.0987583422561913,
0.780214993646469,
0.545004570644817,
0.272611885458516,
0.624591606028653,
0.763531923649615,
0.111802533320991,
0.689720427007284,
0.149591277888786,
0.623343971382521,
0.864328322868947,
0.536907846358096,
0.55865828858626,
0.0395525321548584,
0.492842762960513,
0.649908609990919,
0.179191561033573,
0.948253645071878,
0.882227109224641,
0.985199552022479,
0.120370548740202,
0.185082424983886,
0.808144779786535,
0.0951993558999148,
0.0378320692283251,
0.793201836661064,
0.804001246953384,
0.513700068236189,
0.711018717247536,
0.122846390643551,
0.757085269669576,
0.809777059503727,
0.903061384755681,
0.302089840314393,
0.0823889444965818,
0.527652990318673,
0.0656217634983462,
0.194191477817572,
0.217373416860296,
0.215213041852793,
0.817535449200093,
0.0817017392635819,
0.752120888210889,
0.376193737320692,
0.12125427141844,
0.244963651171403,
0.0261023473116114,
0.300445832452013,
0.193217295777619,
0.908329456536253,
0.285645384008831,
0.313587844517821,
0.0934118810544777,
0.093790163795366,
0.408787200417736,
0.131243950748464,
0.88699200045643,
0.212788446905459,
0.644944018984653,
0.598010717238305,
0.335634838014671,
0.402029288654229,
0.40778777627637,
0.238696222304691,
0.704119128968622,
0.490176721238613,
0.766349212623364,
0.769740892466968,
0.684368199056186,
0.98372262948366,
0.984953934319762,
0.501903648256279,
0.065424368747242,
0.737074822530651,
0.878097385576971,
0.186678640165682,
0.982038473702054,
0.904199732888583,
0.487124473083357,
0.175255769014012,
0.812529189424836,
0.772769857092188,
0.488843613997495,
0.905941070479313,
0.866560020887554,
0.897630814415231,
0.0371850207621162,
0.753552020878322,
0.110419261320689,
0.682129040212431,
0.351562737650966,
0.44605409933536,
0.0841583284009985,
0.759350514392997,
0.684750321640051,
0.788277457369621,
0.24952723516595,
0.451099533797754,
0.558018349370928,
0.933895434222135,
0.434822162815753,
0.542972283690689,
0.435799082012754,
0.500246531562994,
0.280047105755679,
0.313896467589725,
0.686925172194338,
0.262085578992071,
0.218096200012647,
0.174049644812033,
0.437341348006083,
0.0306253889718211,
0.946819501904221,
0.926184962003578,
0.936566459916796,
0.813379522791775,
0.823815776418809,
0.973751480678912,
0.566931543204436,
0.934235037739498,
0.655880520425681,
0.918494281321063,
0.380289136609197,
0.74003884882668,
0.677844795248399,
0.065039457783587,
0.528316305730639,
0.927372030414348,
0.516138991581341,
0.0863346551015669,
0.861267464170823,
0.950961154397093,
0.629306938792256,
0.297066546183576,
0.451207685960088,
0.909354044547935,
0.610963013773301,
0.138132857688765,
0.171439623074345,
0.829059213785948,
0.312182502500798,
0.608780971080428,
0.85968460322343,
0.259002003939358,
0.534965933084006,
0.796251062674565,
0.072381526265471,
0.358781709037154,
0.770002542887816,
0.639313069935568,
0.293016746310991,
0.425883063313497,
0.55780735079097,
0.673305882920188,
0.165921911674515,
0.235652145573707,
0.738345340703775,
0.694238217870816,
0.163024175522395,
0.254484331819454,
0.780572872972383,
0.0242916392275559,
0.205445485750886,
0.409879811298977,
0.321358185411132,
0.656653171710974,
0.319233855381251,
0.932321199184433,
0.794786029399739,
0.490673478455596,
0.761380412970381,
0.106968531434875,
0.0994544490703635,
0.62106501572815,
0.365970535839894,
0.63442038215437,
0.417316077937054,
0.438352062105365,
0.993202091191524,
0.187318620359208,
0.077665131575272,
0.286218837036853,
0.613201683672705,
0.635472482366242,
0.959524719957041,
0.779123595812881,
0.871124627939949,
0.697870060195155,
0.473361813218036,
0.0341488029966824,
0.952354392014609,
0.253934685724757,
0.0584404426898996,
0.157799877765495,
0.663814497023735,
0.379798628101032,
0.814453049476469,
0.983048352404986,
0.312119827285465,
0.609239078876208,
0.473721830394921,
0.0735002397901845,
0.716207610311083,
0.573176279465284,
0.694565255518335,
0.0821781456853161,
0.207596661619654,
0.111881332989727,
0.520530208256343,
0.200798752345517,
0.299199953348935,
0.598195339831615,
0.487017589848031,
0.912401637487301,
0.233667822197856,
0.446542309339411,
0.69152523283452,
0.104792449672144,
0.144412369534565,
0.164887045586895,
0.138941253134488,
0.0967667610835129,
0.418821731311652,
0.197381695824387,
0.254566638849008,
0.082636227869725,
0.577180323925419,
0.0690196883254776,
0.0656845798090494,
0.889300151210884,
0.678258767201686,
0.53940641020397,
0.962800391001068,
0.394466377047108,
0.112582689203593,
0.657365646053741,
0.476644523198085,
0.320179350823247,
0.769246979043468,
0.997174731454428,
0.520978103634426,
0.068446932392403,
0.595370070820381,
0.0079956930167953,
0.980848569879703,
0.829037893018237,
0.454538002356206,
0.672373802248563,
0.933830343156043,
0.598950371890771,
0.837260847835457,
0.0727715958248692,
0.695717133439946,
0.256082578681448,
0.270153291649256,
0.950283772288954,
0.338718806551173,
0.847333615574675,
0.0193034601487701,
0.404403386360222,
0.736633766785559,
0.697562227350456,
0.943809796564192,
0.699434157320966,
0.0920286043975636,
0.0563924853021244,
0.356799803374708,
0.568673127595649,
0.376571836591033,
0.126046781952515,
0.565847858584415,
0.897549940225459,
0.194493714344918,
0.161217928939135,
0.905545633242254,
0.17534228375896,
0.990255822423033,
0.360083635132799,
0.847716086007522,
0.924086165113415,
0.959034007489231,
0.684976933377318,
0.996857760938284,
0.654751140463516,
0.941059512058766,
0.267011052121879,
0.605034912286808,
0.279778318144278,
0.114344667230893,
0.62433837290124,
0.6841817045045,
0.850978434016452,
0.321900599786034,
0.627991500603031,
0.550412591337418,
0.413929204183598,
0.684383986370817,
0.907212394712126,
0.982602331779246,
0.0609558224961887,
0.0332591761989795,
0.548450189898,
0.958505762721647,
0.227752891009558,
0.709668119302796,
0.86405139549824,
0.403095174768518,
0.699923941260168,
0.224135030165378,
0.250811260776041,
0.624010105907922,
0.183169037188948,
0.935788194153359,
0.620867866380544,
0.837920177652463,
0.876847706212126,
0.887878918502423,
0.442955089939272,
0.156626023890742,
0.00222358526765536,
0.0672934623748499,
0.840807728860903,
0.853202019749769,
0.389194062160884,
0.468799228998273,
0.403614610621526,
0.803123266810143,
0.153183214903429,
0.310827004867991,
0.785725598123728,
0.214139037399618,
0.34408618106697,
0.334175788021728,
0.172644799655604,
0.571839072076529,
0.0438439068588633,
0.0366961946881824,
0.974934247310708,
0.743767848119032,
0.260831225319221,
0.225745507621088,
0.367777953561292,
0.444000262508169,
0.161533701308786,
0.988645819941836,
0.281920440160632,
0.0383814070552501,
0.876524737978598,
0.724875530099904,
0.195007431411653,
0.878748323246254,
0.792168992474754,
0.0358151598068956,
0.731950342530362,
0.181363054169977,
0.504614388805169,
0.135564952686226,
0.98448632098012,
0.657797604174259,
0.446391957554217,
0.770211919103848,
0.871936641573876,
0.790478139086849,
0.104387706659915,
0.0445814412294801,
0.362317210697717,
0.148231613518778,
0.0812776359176625,
0.337251457542764,
0.89199946163781,
0.342108861236884,
0.562996965163852,
0.25977741473344,
0.786109124210714,
0.724530666938299,
0.248423234209615,
0.068029563905685,
0.762912073993549,
0.124947971722553,
0.792905094005589,
0.957919505405202,
0.00369629450314506,
0.585074086014682,
0.993734665212098,
0.735646637499168,
0.766437140184658,
0.498349054017267,
0.871211590185394,
0.750923461164778,
0.156146657725864,
0.317603547739612,
0.521135379802964,
0.0280832988340796,
0.1080816863608,
0.625523086462879,
0.0726647400635596,
0.470398897058516,
0.773754699981657,
0.153942376446883,
0.807650355066941,
0.665754161153805,
0.496051237683767,
0.370647319765132,
0.925531575887246,
0.28216036142882,
0.095177986237769,
0.1739548096312,
0.350189925334505,
0.858090060696979,
0.298902781353752,
0.143095018874432,
0.81600956563652,
0.302599075856898,
0.728169104889114,
0.809744230848618,
0.0382457128904041,
0.494606245073772,
0.308093284400223,
0.90945730354146,
0.245529705772889,
0.464239942126088,
0.22706085081541,
0.766665085575853,
0.492323240960167,
0.33514253717621,
0.392188171573071,
0.564987981489388,
0.805541434234726,
0.165942871089067,
0.718930357936272,
0.613191788836006,
0.831697032242872,
0.214981595154377,
0.983839109066799,
0.757228607664457,
0.497141957048858,
0.0790170948389066,
0.931183417295657,
0.847331882383363,
0.937107155535886,
0.230086198183748,
0.990426901723457,
0.753116721172406,
0.532685274040646,
0.71859600614691,
0.562860951555362,
0.570930987396711,
0.213202250755021,
0.870954235955586,
0.480388290472509,
0.458731956527909,
0.335194177616012,
0.70744914128792,
0.225397041638101,
0.82751741857618,
0.0425916779984681,
0.617585213211172,
0.392505399599907,
0.848133112233194,
0.783528084300239,
0.111435757070517,
0.4613249010692,
0.61522511607745,
0.326417352690556,
0.445164009670338,
0.372453723276245,
0.823559309739414,
0.524181104509244,
0.303637140106241,
0.670891191657116,
0.46128826004513,
0.533723338289989,
0.661318092914912,
0.214404980751874,
0.0664086123306344,
0.379914099061822,
0.777265932307237,
0.637339599727345,
0.593116349816842,
0.648220167797161,
0.117727889734194,
0.0518483063447514,
0.983414345413174,
0.825177031022113,
0.277245347982852,
0.810931763523692,
0.867768709020581,
0.894830561659686,
0.203437163123599,
0.715901820788114,
0.678358645494263,
0.314872920194116,
0.177226721391653,
0.293583761571713,
0.641290272884672,
0.622390731061991,
0.666037484847958,
0.464849582158425,
0.146571835571235,
0.96967462541986,
0.135740773815541,
0.607860095616365,
0.503397963244188,
0.797058866730453,
0.822265076368239,
0.569806575574822,
0.176972965326613,
0.599531008209815,
0.207146174836506,
0.770089315609117,
0.247751175541315,
0.3248740645707,
0.821937621953868,
0.231165520488827,
0.150051095127152,
0.0991829694710593,
0.0420972835468581,
0.017819803682072,
0.994013531130745,
0.245534446670457,
0.733721624935848,
0.672372176625008,
0.560407367330234,
0.910948346327501,
0.965955938196721,
0.201697639749245,
0.533339077389491,
0.631993422579017,
0.666547222373331,
0.679910912960726,
0.601668047533216,
0.802287996188872,
0.28777100811143,
0.105066010777404,
0.599346862453663,
0.110036084014008,
0.674872586352226,
0.776319828245938,
0.709567092223823,
0.882018761654393,
0.546409143389393,
0.957318267765138,
0.206892825759432,
0.3683467648776,
0.188483787788303,
0.356943921352245,
0.467529734348659,
0.230581071335162,
0.374763725034317,
0.461543265479404,
0.47611551847128,
0.108485349504503,
0.133915441638751,
0.0365228853358528,
0.0194336958320037,
0.0998713793698099,
0.238220525085097,
0.552772773221495,
0.731864801948827,
0.904767747458428,
0.23268368571656,
0.333532849482043,
0.707055743181638,
0.52045469382799,
0.438598860259447,
0.306402605635301,
0.630490777841998,
0.113471446146011,
0.0827224334155779,
0.340057869600159,
0.995490207800404,
0.629131576804971,
0.297376136899635,
0.202383033559836,
0.997478341682571,
0.485859924687939,
0.559326954912081,
0.465008075565569,
0.7164409960231,
0.934090679946398,
0.926551341044973,
0.192556514028719,
0.0425760294509009,
0.0604667822180627,
0.229079399364572,
0.0620097252829046,
0.160338161587873,
0.46729992491533,
0.6147824985044,
0.892202964002361,
0.372067671908097,
0.84746618422096,
0.225735813018743,
0.0791234150897355,
0.367920877583288,
0.66433467327819,
0.385526020725037,
0.998411655425286,
0.777806119424201,
0.468248454140615,
0.338469524559784,
0.773296327224605,
0.0973800304799248,
0.635845661459419,
0.975679360784441,
0.0948583716968346,
0.121705585681696,
0.535006315230861,
0.559866447262404,
0.838146581704797,
0.469096994711597,
0.486417788307377,
0.0307030957335155,
0.511673024162498,
0.54688457052544,
0.259782495098087,
0.573682749445402,
0.707222732578974,
0.727082420013418,
0.188465247484141,
0.599425696115673,
0.0991500919215149,
0.0359314312394389,
0.825161509134416,
0.17827350701125,
0.403852308822727,
0.489496181946945,
0.563799527736287,
0.402263963782351,
0.267302300905484,
0.032047981411241,
0.740733488342135,
0.0405986276644276,
0.129428011891166,
0.376579149335892,
0.0162779879832072,
0.224286383588,
0.498284735017589,
0.551284303214068,
0.784152830850404,
0.336431316722385,
0.0203812979256647,
0.27057061869212,
0.367134412455901,
0.532054322088163,
0.817455189683221,
0.626916907553988,
0.105737071533565,
0.524677921796533,
0.353999327567406,
0.294202319017706,
0.124103617446545,
0.453149419488921,
0.330133750722806,
0.949265126580962,
0.631422926500171,
0.733986059545533,
0.438761308062245,
0.195222453770797,
0.136250023327884,
0.706063608967729,
0.227270435182038,
0.876983511670019,
0.746662237097818,
0.356698447073204,
0.253562661005912,
0.762940225081025,
0.580984830661204,
0.7518473960235,
0.314224528295093,
0.365137661045947,
0.0882787122802244,
0.334605826220757,
0.635708280203728,
0.455413124736125,
0.86666014830892,
0.453163469421288,
0.0823300322901132,
0.972397219842485,
0.977841391217821,
0.436329359857519,
0.26659953886019,
0.101945008198705,
0.889478779346439,
0.596733289582996,
0.0512101343140053,
0.520901705380949,
0.330719348662868,
0.48997144237625,
0.716124159151746,
0.466969371990752,
0.196035050878318,
0.943394594333784,
0.343952883660771,
0.942697287976135,
0.300093040941326,
0.597515544666683,
0.70563751305716,
0.88107787160253,
0.349362940224522,
0.0198620408865912,
0.246215532182816,
0.437641652970408,
0.354467867107348,
0.881923812386544,
0.893054777706533,
0.221128014950607,
0.335087281342171,
0.975384809996646,
0.193525234793092,
0.312928672094331,
0.411714169388504,
0.460124773653282,
0.414873680293036,
0.301192948269282,
0.0568580627706172,
0.466083814607041,
0.822094653650231,
0.387577411433485,
0.956055256983291,
0.538218812336316,
0.854546783889898,
0.152090307861609,
0.481613406204438,
0.198499667085009,
0.0947875953720825,
0.781706447145765,
0.796015211751692,
0.800425108429242,
0.662784318282634,
0.145378151510553,
0.820287149315834,
0.908999850931112,
0.58301980448096,
0.174755015957521,
0.790923662851995,
0.476074582187494,
0.395883031373789,
0.126010944194166,
0.451459391718479,
0.589408266166881,
0.438939616288496,
0.863173561572644,
0.0495330393545018,
0.853813297047193,
0.164366509376264,
0.106391102125119,
0.319897111188572,
0.986461163492157,
0.493968513558604,
0.275952368171863,
0.524679975362811,
0.348515296982841,
0.428042676033472,
0.00629338156724972,
0.54701496406785,
0.522830271405554,
0.787999828713015,
0.34303017535388,
0.323255379834797,
0.450784146995649,
0.488408327330094,
0.143542528684969,
0.359783997461099,
0.0714281313453932,
0.318297545108151,
0.150707659847433,
0.547502713532887,
0.71418057648194,
0.276718604041598,
0.998962105717027,
0.30358884218316,
0.715658220795755,
0.862135666824009,
0.353121881537662,
0.569471517377287,
0.0265021757346122,
0.459512983662781,
0.889368628565859,
0.0129633387611077,
0.953481497221385,
0.165320996272061,
0.53764331458958,
0.301996794204226,
0.593363672305534,
0.54393669615683,
0.849011758272076,
0.116193943711088,
0.331936524404183,
0.192041933625956,
0.439449323545885,
0.782720671399832,
0.68045026095605,
0.582991852230854,
0.14250466839527,
0.751878392301443,
0.901289397339006,
0.293212328708364,
0.29938110583433,
0.615469973355285,
0.569930932749962,
0.298343211085695,
0.919058815538445,
0.285589153080056,
0.160478877444043,
0.272180696610445,
0.855060670457343,
0.186981053178655,
0.731693680273226,
0.744429299023202,
0.199944392405424,
0.685175177494611,
0.909750295295264,
0.737587706995005,
0.987171971698837,
0.503113967600797,
0.281524402686173,
0.836183729505252,
0.619307911311885,
0.613460927090357,
0.0282256626655467,
0.058757234392109,
0.396181598490189,
0.708675923621597,
0.641749086622963,
0.538686266885459,
0.46055431592304,
0.543038483961969,
0.831898595593823,
0.75993542175737,
0.158508456851593,
0.401829528343784,
0.0582786323774041,
0.0775672723900374,
0.68741868142384,
0.218757509821447,
0.349747969000483,
0.542479351881183,
0.405738563000103,
0.0814416492737092,
0.286908650438724,
0.605682955405527,
0.76661682676832,
0.196658945268327,
0.343270661934871,
0.753788798001496,
0.699772912869124,
0.624795064621044,
0.589972527041087,
0.319080823715348,
0.2382559917114,
0.618198190172295,
0.377838058107457,
0.634437590201589,
0.326874113328231,
0.0195871447304204,
0.173123856621386,
0.787428429251271,
0.562625628692389,
0.00502245221520888,
0.547363850542979,
0.721134085543982,
0.406851980558993,
0.605642482920383,
0.798701357934019,
0.0942706615171724,
0.824399992741831,
0.148449326934502,
0.636750013398356,
0.230138555741933,
0.229890976208211,
0.92365866383708,
0.835821511147461,
0.996507802976532,
0.120317609105407,
0.179092173082331,
0.750296600512367,
0.820090521974531,
0.803887237703375,
0.340269127087793,
0.139171345689879,
0.0421432289491143,
0.958467317260088,
0.517009403797337,
0.676580819150703,
0.285341430588319,
0.536596548527757,
0.849704676237751,
0.072769859373928,
0.099222176754485,
0.85472712845296,
0.620133710382568,
0.820356262764128,
0.261579108546292,
0.225776192837291,
0.619057620232486,
0.355849770063464,
0.0501761855791212,
0.767506947166988,
0.992599783927481,
0.280314741321055,
0.9973979233752,
0.9162584472989,
0.116136252002854,
0.99390572588607,
0.0365760559386462,
0.295228425085185,
0.744202325932776,
0.856666578378839,
0.0991156627885605,
0.0844714530205687,
0.995837924068718,
0.141258891737675,
0.0429387698149955,
0.512847327400394,
0.817839711354039,
0.328280200403314,
0.0494438759281504,
0.667544387126129,
0.401050060242903,
0.148666052682635,
0.522271515113428,
0.0211837701598107,
0.969022315446763,
0.78385062365972,
0.246959962997101,
0.588079935679249,
0.139700393257523,
0.297136148576222,
0.355586882380576,
0.132300176719343,
0.577450889897277,
0.352984805755776,
0.0485586240182438,
0.693587142365792,
0.346890531176185,
0.08513467995689,
0.988815567450978,
0.0910928571089603,
0.941801258335729,
0.0879312297738768,
0.175564310129529,
0.937639181938786,
0.229190121977213,
0.218503079944524,
0.450486509339179,
0.0470298328655911,
0.5467832808135,
0.49993038526733,
0.71457421999172,
0.947833341056403,
0.648596437949965,
0.236845734639487,
0.969017111216214,
0.617618753396728,
0.0206963578335458,
0.215977073747654,
0.205698688610317,
0.160396751091069,
0.513113222789538,
0.561285570990893,
0.292696928276074,
0.0905641122211535,
0.914270376746669,
0.341255552294317,
0.784151254586946,
0.261160907922853,
0.426390232251207,
0.772966822037924,
0.352253765031814,
0.368191490586936,
0.8608980518118,
0.527818075161342,
0.305830672060061,
0.090088173323352,
0.746321155105867,
0.75631718139924,
0.137118006188943,
0.293104435453706,
0.256247566200908,
0.851692226180663,
0.240937776044448,
0.904844004616534,
0.0885379603544893,
0.209954886795001,
0.522462757547602,
0.109234318188035,
0.425931961008316,
0.728161446157918,
0.269631069744765,
0.939045183797854,
0.28944701668315,
0.562327998020839,
0.0296092955533458,
0.203717393429818,
0.903583550315156,
0.813760550605953,
0.464878301352671,
0.329973782100703,
0.586727372178215,
0.817132066384485,
0.698165272687639,
0.447625423524354,
0.344950141080166,
0.0039959447476994,
0.537713597313368,
0.091271295720372,
0.760313126146939,
0.674831603502311,
0.384375731639739,
0.0165606923478472,
0.526523829682974,
0.625313507684187,
0.921404696964382,
0.615061790037463,
0.835268394944849,
0.443867454046322,
0.72429610869116,
0.261200355487503,
0.172028899738579,
0.993927178435925,
0.200245538819696,
0.461475916421728,
0.556255175991103,
0.229854834373041,
0.665193309851546,
0.459838725840598,
0.0436153845133332,
0.130071610738557,
0.7898125079413,
0.630342756691548,
0.947203677123042,
0.487977780628939,
0.0779681802159027,
0.292153817737547,
0.491973725376638,
0.61568177752927,
0.38342511392358,
0.252286851057916,
0.29051338056592,
0.767800845563319,
0.268847543405764,
0.817037210248894,
0.393114352781844,
0.190252239904484,
0.432099000286357,
0.228382747261032,
0.634119693950806,
0.156395108511855,
0.489583102748535,
0.806148593689384,
0.150322286482119,
0.689828641568231,
0.267624510111112,
0.706577462473222,
0.919683476406933,
0.932817819962659,
0.166416187848158,
0.963298860920267,
0.0628894307012155,
0.956228695789459,
0.593641617611815,
0.0100931073585959,
0.444206475952736,
0.671609797827718,
0.302246925561804,
0.936180201329375,
0.287291574891327,
0.685672039485384,
0.188467052387291,
0.577804955457246,
0.453472884583041,
0.457314595793055,
0.39484216570614,
0.846587237364886,
0.6475668361632,
0.826941165992497,
0.0749699846259178,
0.281686529648344,
0.983336274504352,
0.564553087374453,
0.0878351233377285,
0.13365856052081,
0.254381728942684,
0.355459633448841,
0.840236022994032,
0.174065204883956,
0.288277452945838,
0.00665221037652912,
0.137364065338561,
0.351166883647054,
0.962880906165988,
0.731005682950376,
0.361259991471311,
0.407087382118724,
0.402615480312433,
0.663506917033115,
0.343267582982437,
0.68990705520376,
0.349178956052837,
0.531734635369729,
0.267712010661006,
0.802651840635879,
0.989049231628445,
0.662554176367146,
0.649239077535104,
0.636616067325983,
0.489495341893982,
0.724209062161021,
0.918302596974327,
0.472831615932673,
0.288762149535474,
0.00613771984639471,
0.606490176453483,
0.543143878478158,
0.361597353295236,
0.446726198981854,
0.717209083362114,
0.649874806706735,
0.453378409358383,
0.854573148700676,
0.00104168988812794,
0.41625931505871,
0.585578831651052,
0.362301681359439,
0.823346697177434,
0.988194311963485,
0.0258085979268926,
0.166614280159871,
0.678101367167244,
0.37498755397973,
0.6983489155296,
0.94581337782825,
0.177639394149948,
0.687398146692383,
0.608367553729735,
0.826878472150713,
0.324014213552705,
0.0978628951580557,
0.551087533846073,
0.242316810527033,
0.570694511090729,
0.839849683381547,
0.248454530373427,
0.177184687078551,
0.382993561394044,
0.610051884134324,
0.623910886060405,
0.100202644290497,
0.259926690375398,
0.0772892949531271,
0.954775793456834,
0.260968380263526,
0.493548610011837,
0.540354624642224,
0.623270062088626,
0.316895307189271,
0.528548936140048,
0.649078660015519,
0.483509587349142,
0.206650302841631,
0.0240662139952491,
0.18185850241308,
0.15246368020422,
0.201705608145197,
0.869256649571125,
0.760831233933955,
0.0285840798302479,
0.193270862658169,
0.858694129092011,
0.579671614141982,
0.435587673185201,
0.429388639717078,
0.419521297057868,
0.68404220402429,
0.606573326795629,
0.802514858917573,
0.294094087692953,
0.230484212390373,
0.90271750320807,
0.554020778068351,
0.3077735073435,
0.857493296199242,
0.814989158331877,
0.801322117820998,
0.397847920375805,
0.438259219954842,
0.118217424544607,
0.926396856981514,
0.0873378799703614,
0.601727011893749,
0.133047159357484,
0.111404093965611,
0.783585514306829,
0.285510840027365,
0.313109702110807,
0.652842163412293,
0.0463420734956591,
0.341693782406716,
0.846113026536122,
0.905036203053331,
0.921365396548698,
0.281700699255662,
0.334424842304748,
0.340886693140905,
0.965742903279952,
0.940998169566038,
0.143401551592816,
0.259836990507244,
0.171482381490749,
0.0461190548008862,
0.813857768575595,
0.47925588929991,
0.903612351000128,
0.628846926907472,
0.280578006655247,
0.301460271375934,
0.0671061463966529,
0.398795431199854,
0.227857127891787,
0.154444026367014,
0.000522442627941464,
0.360904287249271,
0.265848120332625,
0.784107956934771,
0.646415127276636,
0.578957822909093,
0.436950120347063,
0.692757201237957,
0.92065160531581,
0.283063146417524,
0.597793403825626,
0.842017001398847,
0.564763845673187,
0.932218246130374,
0.18290369407409,
0.530506748953139,
0.873216415230751,
0.326305246132568,
0.790343739460382,
0.0446987967215007,
0.372424300933454,
0.604201508035977,
0.523954686021411,
0.276036651467922,
0.233048434477788,
0.804532692676658,
0.577496922843855,
0.300154580874441,
0.203328123410851,
0.805354050735642,
0.454598607707116,
0.203850566038792,
0.166258337984913,
0.720446728039741,
0.987958522973563,
0.812673465261549,
0.299404550483173,
0.424908642854965,
0.505430666033845,
0.220056155333322,
0.707971789272489,
0.10322406939381,
0.0620731562665073,
0.272735634945676,
0.0354423155241843,
0.244976850806259,
0.803242383898814,
0.908658730754936,
0.571282096938827,
0.593586122893536,
0.953357527476436,
0.943706397872281,
0.197787630463851,
0.477312213032186,
0.219743048874542,
0.430836064941639,
0.281844905243183,
0.797239972184058,
0.730990646281741,
0.485173028654034,
0.602594022454039,
0.185589253523196,
0.689023594692826,
0.768852360438952,
0.906035981562937,
0.676982117666389,
0.581525825700502,
0.20544053204611,
0.101890760055692,
0.0869564912686853,
0.425496687379431,
0.809862549793843,
0.190180561128157,
0.4875698441116,
0.0825981842738568,
0.225622876652341,
0.732546694917859,
0.885840568172671,
0.134281606941615,
0.303828791391025,
0.479426690600545,
0.0876391344180513,
0.247535188797645,
0.677214321530058,
0.564951347450237,
0.467278237672187,
0.108050386006036,
0.846796253159082,
0.264518209390584,
0.839041032287777,
0.331969281347454,
0.867112232310284,
0.0246302853453114,
0.0209928760402803,
0.635964592283575,
0.930666267373909,
0.697974993706669,
0.217490417518416,
0.136106798954358,
0.799865753762361,
0.304446909252762,
0.561603486333789,
0.609728303090543,
0.494627470380919,
0.0491733299797277,
0.6923264873644,
0.72025034703326,
0.781720024897587,
0.578167055071409,
0.854531953974875,
0.08554881582295,
0.0575937456719548,
0.942171088392926,
0.333084004620595,
0.734808067202013,
0.507122435843163,
0.800362242758443,
0.842858453208049,
0.353918688536584,
0.064880451683365,
0.681899485495826,
0.685887969884038,
0.931992683993649,
0.706529770841137,
0.706880845924318,
0.567957276277224,
0.637196037749385,
0.404855839165326,
0.78544769379564,
0.773302836703743,
0.204721592927688,
0.0898946025827409,
0.334906323037532,
0.81444989601823,
0.58452207296366,
0.38407965301726,
0.50677638338263,
0.304772419531258,
0.165799677914847,
0.0849434379883778,
0.159304373506133,
0.251348493737797,
0.142537183660333,
0.101475461433397,
0.584432498824053,
0.877345250862346,
0.608597897276561,
0.384794741116834,
0.720203704070395,
0.962516585813144,
0.449675192800199,
0.402103189100559,
0.648404555697182,
0.381667876793848,
0.108632959476036,
0.355285401155839,
0.949625153071073,
0.745828997691082,
0.760141240786827,
0.735072846401051,
0.519131833929164,
0.964862833714514,
0.824967449449453,
0.854038156966697,
0.779312729267083,
0.409489521947452,
0.238117809983957,
0.286089112184052,
0.714261941944371,
0.403917487898803,
0.371032550638091,
0.873566315450504,
0.6552659816366,
0.513569734298424,
0.975041776883901,
0.239698479994991,
0.390914985160769,
0.583639674160462,
0.624493221111825,
0.111118688765503,
0.546156259507945,
0.0741684139120245,
0.513221877866062,
0.194560814739466,
0.455836290705873,
0.621854837342098,
0.549846216360967,
0.405461443311284,
0.367683834567519,
0.309987456682132,
0.140534289246674,
0.886815668962344,
0.274850289930985,
0.965501738696127,
0.74085382546338,
0.0541630187324076,
0.374991260643579,
0.978971635447336,
0.340252131382121,
0.0892532021222884,
0.382889122880478,
0.711284682020212,
0.962819517572792,
0.0381551045170776,
0.224854415852974,
0.937861293991032,
0.277853584512069,
0.615769401013744,
0.521500967685832,
0.902346806089555,
0.726888089779247,
0.0676572271937771,
0.97651522000158,
0.240109967179648,
0.262218041933243,
0.432351510241791,
0.861964804987407,
0.81206425829421,
0.837812953553075,
0.229648639089264,
0.122051714510681,
0.978347242799749,
0.116464307585947,
0.396902004441666,
0.943848981495876,
0.857318133514988,
0.451065023639735,
0.318840241673794,
0.836289768496663,
0.791317155021856,
0.408093443796082,
0.219178891377141,
0.502601836576407,
0.370912960903213,
0.257333995894219,
0.727456252429381,
0.308774254428583,
0.535187580406287,
0.343225653443125,
0.830275222580077,
0.437534386030181,
0.0701137427567103,
0.897932449773854,
0.4140496055661,
0.310223710402019,
0.160150491241436,
0.846401115807891,
0.172188514923765,
0.972214750001307,
0.684214068895306,
0.401837154013029,
0.0942664640463267,
0.662561311695055,
0.518301462064638,
0.491168468953654,
0.60641029272527,
0.375619595113965,
0.94223349259339,
0.925250534399064,
0.211909363144966,
0.733550647149585,
0.333343977729484,
0.431088254522108,
0.23615248326033,
0.704256938632697,
0.688422250416326,
0.963608735689711,
0.0130311925956193,
0.223609830822614,
0.306834388667175,
0.843306415175696,
0.661144216852795,
0.376948131889546,
0.741238864483889,
0.0751938224188955,
0.687171842291566,
0.901389356190986,
0.921594938226787,
0.85936035721533,
0.873604105726631,
0.605809007122092,
0.261197510762698,
0.967870569772958,
0.268370318351486,
0.779498972827335,
0.459039038260951,
0.874780611076756,
0.155118567475639,
0.40127253038868,
0.800031145010158,
0.367027931086266,
0.134823177072603,
0.133375122273981,
0.798116185608374,
0.370975660332933,
0.837632060906679,
0.4865384360247,
0.334584396022644,
0.850663253502298,
0.710148266847314,
0.641418784689819,
0.693969668677994,
0.371292483234449,
0.0183669161137039,
0.435208532696221,
0.446486305653344,
0.705538758405269,
0.336597888421546,
0.368081243880131,
0.564899115154938,
0.210201993682516,
0.973890251002223,
0.826096626383297,
0.178072563455474,
0.242260568888048,
0.605595598744971,
0.637111601716425,
0.117041179499142,
0.760714166686271,
0.0383841321051047,
0.9170723245093,
0.127742097306876,
0.173207309177708,
0.0504474463176203,
0.925858282915251,
0.544182969976302,
0.888079507224299,
0.41239671847429,
0.878767365998946,
0.738742760260935,
0.122544984855943,
0.520186150223103,
0.432712428473268,
0.493837468556053,
0.538553066802469,
0.86792096163515,
0.940323774209397,
0.244091824742077,
0.204518849591035,
0.308405017623866,
0.808990940362676,
0.414720843739212,
0.282295268160428,
0.635087566280313,
0.592793407194686,
0.524555837048477,
0.240683164559623,
0.22990500844545,
0.641597016547619,
0.00139733078023294,
0.268289140550554,
0.558669340591258,
0.129139428087109,
0.441496450193923,
0.609116786908878,
0.05499771100236,
0.985679420170225,
0.497196293667516,
0.46739442947665,
0.86444678570351,
0.235939053928451,
0.589939414798254,
0.384632935460952,
0.668651482401719,
0.0837768828886453,
0.92318600226342,
0.536572443571208,
0.0241006566323809,
0.167277827005497,
0.741091293627904,
0.332505674721909,
0.976268767368174,
0.155812136901455,
0.614800942882337,
0.611356333182825,
0.748605544096141,
0.139356779930814,
0.852039497742448,
0.978510552541591,
0.780953796478433,
0.853436828988342,
0.246799693092145,
0.339623137069691,
0.982576257075451,
0.688296143286068,
0.948739923978569,
0.0375739676121501,
0.673975562990632,
0.445936217646085,
0.504968397554461,
0.53842234822848,
0.681875271574536,
0.0949078118870537,
0.923055283689431,
0.350526753510594,
0.178684694775699,
0.846241285952852,
0.887099197547463,
0.20278535140808,
0.0135191124926876,
0.628190490709706,
0.535291026129988,
0.989787879860861,
0.784002627611161,
0.150091969012326,
0.601144212578025,
0.532608171241641,
0.289448748943139,
0.453183709854811,
0.51111872331757,
0.0704025449559104,
0.306620538377492,
0.757918416409715,
0.410025682025601,
0.289196795452943,
0.446214559230122,
0.35876560600417,
0.326770763065094,
0.120190121755092,
0.804701823650255,
0.831739160619555,
0.658612469983572,
0.48657709475913,
0.926646972506608,
0.581667753673004,
0.837103848269723,
0.105331666816646,
0.427909039160194,
0.724203045351525,
0.308117018690387,
0.441428151652882,
0.35239353559557,
0.843408044820376,
0.431216031048082,
0.136396162741071,
0.993500013832701,
0.0323602431604454,
0.669004333982712,
0.282948762310179,
0.485543953015257,
0.180123057300282,
0.353351307731751,
0.79216449185841,
0.938041473709997,
0.763376989757352,
0.0813612868456921,
0.38425603294012,
0.122142595295861,
0.408132049910786,
0.504446154695212,
0.926844418946115,
0.23987121053034,
0.163058624678784,
0.413421513239584,
0.166518182571287,
0.744726378351788,
0.250525361509307,
0.271849849853595,
0.172635417046321,
0.974728406860832,
0.579966868543982,
0.614063568699203,
0.327121942456403,
0.423374912898696,
0.0452795992816238,
0.463518105197473,
0.416874926731398,
0.0776398424420691,
0.132522439180185,
0.699823689041577,
0.563183795457326,
0.312645496480468,
0.0531749963076669,
0.355348286850074,
0.250686969724804,
0.81655198653068,
0.436709573695767,
0.634943002664923,
0.938694581826541,
0.844841624072214,
0.139389157360135,
0.865539000772656,
0.0847128341368925,
0.30244778203892,
0.278960513546579,
0.25123101670818,
0.0471741599250464,
0.529485875055886,
0.523080866561775,
0.219809576971368,
0.504214281916718,
0.103047734640095,
0.833873145670571,
0.831336224373121,
0.526422648004453,
0.879152744952195,
0.294854329104933,
0.943297574735851,
0.956792587394264,
0.427376768285118,
0.643121263311767,
0.519976382385928,
0.740022264765586,
0.696296260085095,
0.875324669701664,
0.990709234956051,
0.512848246150114,
0.312034242931769,
0.625652237155313,
0.451542827510993,
0.156875866538322,
0.765041394515448,
0.317081827817988,
0.241588700675214,
0.0674891760887062,
0.596042341830229,
0.492819717849055,
0.114663336013753,
0.125528216420453,
0.0159005839451685,
0.33447291298512,
0.629742498337171,
0.118948319050925,
0.16834605819003,
0.461078722244631,
0.645370967055378,
0.0474988026765635,
0.755933051815225,
0.588668541325568,
0.00429138960516611,
0.183309819634682,
0.231789804171673,
0.524267772456756,
0.92333208486593,
0.928086064256768,
0.399592441692759,
0.914041319356319,
0.44093430994122,
0.711626684624528,
0.539693556045971,
0.892477137917875,
0.868502551628511,
0.304734950561419,
0.209558965270202,
0.110091251838063,
0.372224126650125,
0.805601307100431,
0.602910969687119,
0.486887463129539,
0.931129523986545,
0.618811554097948,
0.821360376114659,
0.560872021858055,
0.737759873148873,
0.989706434770351,
0.0219507441026861,
0.38313083973859,
0.0372052369812528,
0.777883795917911,
0.971799381064158,
0.0414966270520802,
0.961193615552594,
0.203589184770169,
0.565764399508836,
0.884525699952862,
0.131675249026937,
0.965356841201594,
0.79856701884352,
0.572609558968157,
0.676983525360461,
0.338260574889491,
0.465086696420371,
0.54548607652331,
0.642995525450909,
0.674645662156234,
0.655577328827035,
0.0152196516353728,
0.480246968791004,
0.258488298048493,
0.502107114764912,
0.411376492311888,
0.877299852146441,
0.323467490879571,
0.972248514635604,
0.615059724829653,
0.31317392518426,
0.99419925873829,
0.998190564568243,
0.350379162165513,
0.77208305419054,
0.96998994516674,
0.391875789217593,
0.733276669277473,
0.173579129936909,
0.957640188726429,
0.617802369230335,
0.305254378963846,
0.922997029462362,
0.416369387608194,
0.877863937932003,
0.599980554822823,
0.754629962497684,
0.342950634352374,
0.145466630880473,
0.397625487482932,
0.0175962960429472,
0.801043959707508,
0.412845139583966,
0.497843264833951,
0.0595322577560005,
0.914952254348878,
0.9092197576115,
0.936832109902442,
0.238419744762787,
0.881468271781443,
0.551891834732095,
0.551593669947048,
0.875667530054072,
0.550082398834677,
0.901972832578222,
0.647750583778951,
0.520072344001416,
0.293848621330153,
0.381027252590762,
0.693651473938325,
0.251488810056582,
0.998829621821097,
0.998905852902171,
0.174485839053283,
0.41519900942929,
0.876769790368513,
0.774466393876107,
0.169828971461313,
0.219720424255226,
0.919933025222241,
0.567454459409907,
0.237316720298173,
0.720976984464087,
0.980299598993873,
0.735159985597786,
0.780509242220088,
0.89525185287709,
0.644379742743624,
0.717341352122529,
0.133671597639877,
0.525848014059406,
0.269233186388963,
0.685265267586925,
0.401515543647816,
0.819315585223639,
0.587238099699485,
0.0492661269611056,
0.339387928759394,
0.8810867214953,
0.430293379551868,
0.0330394022320581,
0.132575531086221,
0.429123001372965,
0.0319452546685679,
0.307061370139504,
0.844322010802255,
0.908715045502742,
0.0815277640156112,
0.014150981797907,
0.128435469292307,
0.00146078877219036,
0.581605441207814,
0.365752190056142,
0.722437773236277,
0.561905039736026,
0.100912175188266,
0.502947015456365,
0.457156892613115,
0.74529191793189,
0.220288367113233,
0.590828490252992,
0.271139931525635,
0.489521553502195,
0.276093757374256,
0.672655475173451,
0.308837138260173,
0.863331857539402,
0.721921602134556,
0.648225067485229,
0.744418578569041,
0.152214981220763,
0.681264469717287,
0.876994109655262,
0.581337982593727,
0.713209724851516,
0.184055479329105,
0.425659992930321,
0.621924769888597,
0.265583243344716,
0.439810975193889,
0.750360239180904,
0.267044032116907,
0.0214164159360418,
0.116112428771384,
0.989481805818845,
0.583321456137729,
0.21702460395965,
0.492428820809549,
0.0404783482851825,
0.96231652189154,
0.712717187922782,
0.631306838538175,
0.233456452951513,
0.202238740959316,
0.907400595912431,
0.906111928124964,
0.511075879219489,
0.770732452986172,
0.628033529793859,
0.159300946239056,
0.515151031089551,
0.780248511480283,
0.840565416422005,
0.392145140279152,
0.361586493608349,
0.553775140807859,
0.576200620073919,
0.787246487004332,
0.175699910230795,
0.841783863418635,
0.22705746173256,
0.926060149411699,
0.108827895535542,
0.248473877668601,
0.0421725781830831,
0.0983097008887258,
0.83179533380633,
0.259197182142733,
0.590738521698275,
0.872273682091513,
0.221513704034273,
0.303455709155395,
0.503580520164026,
0.454970156985787,
0.505694450114711,
0.410981116076457,
0.361082085110751,
0.0167703288685392,
0.181713568596967,
0.98911561490461,
0.176071275573257,
0.69686460015218,
0.769364125919232,
0.0166366915296003,
0.0890097399656706,
0.130950619527581,
0.57041183233746,
0.665210360039589,
0.918197106531913,
0.746111742568254,
0.506994223458224,
0.145254567798811,
0.672171891514292,
0.615822118993766,
0.393728445467413,
0.714344469697375,
0.714131819882492,
0.225523779273743,
0.97354165230577,
0.304870341115105,
0.097797460899594,
0.195055355874382,
0.608326050270501,
0.601377981529281,
0.650025512860169,
0.114020499919551,
0.0123590971400771,
0.0111075975052582,
0.13079082878809,
0.194072665737044,
0.00022321240986847,
0.306862104361347,
0.890937265889224,
0.769587338329101,
0.323498795890947,
0.979947006320556,
0.900537957856682,
0.893910628228407,
0.645157365894484,
0.818735063922934,
0.640022370331,
0.152151588887047,
0.963989631721745,
0.312194261845292,
0.767973707880813,
0.357718077189158,
0.0265387310770055,
0.482105527297643,
0.5832418564629,
8.03829171137805e-05,
0.786975868412748,
0.681039317362494,
0.195135738791495,
0.395301918217587,
0.282417298426115,
0.845161252117325,
0.509322418137138,
0.294776395566192,
0.856268849622583,
0.640113247390889,
0.488849061768897,
0.856492062032452,
0.946975351752236,
0.37978632719246,
0.626079400361553,
0.270474147643183,
0.359733333047355,
0.526617357752573,
0.164384775405929,
0.00489069894183925,
0.345352421675507,
0.80440714620259,
0.157042287828886,
0.309342052931591,
0.11660140758222,
0.92501599617536,
0.667060130120749,
0.143140138659226,
0.407121523007341,
0.250301986117988,
0.143220522042001,
0.194097391420089,
0.931341303946144,
0.338356260833496,
0.589399309637677,
0.213758601906597,
0.18351751248516,
0.0987217277748146,
0.50853499793845,
0.0397863621077437,
0.738834975165704,
0.997384059707347,
0.896278424140196,
0.685810326452279,
0.377170386434146,
0.522357824036087,
0.956284474095462,
0.736903719947163,
0.0489751817886602,
0.120669249501391,
0.741794418889002,
0.394327603464167,
0.925076395703981,
0.898836706717888,
0.703669656395758,
0.0416778028205399,
0.823852702427587,
0.370729786516507,
0.184817941479766,
0.230974225434928,
0.621031772634495,
0.328038463521767,
0.425071616855018,
0.552373076114977,
0.666394724820924,
0.014470926027033,
0.766131678487236,
0.849912237306085,
0.113192653801848,
0.274666675960024,
0.889698599413828,
0.852027628967551,
0.27205073520171,
0.785977023554024,
0.53783795541983,
0.649221121635857,
0.308334847124449,
0.494122429049631,
0.386124841117358,
0.35731002891311,
0.614791678551022,
0.127919259540699,
0.751637632377277,
0.539868073789341,
0.0267559662585873,
0.455307288773035,
0.581545876609881,
0.850608668686174,
0.826037075289542,
0.766363818555308,
0.0815828936554412,
0.447068847458376,
0.0944022816114138,
0.506654510510459,
0.999441924039015,
0.760797006432338,
0.521125437003153,
0.765573602060589,
0.610709243738423,
0.634318090805001,
0.0402402775549517,
0.50040784268659,
0.486345719772552,
0.312291012756662,
0.286384865774952,
0.0241836747267207,
0.961512134392519,
0.594719713365063,
0.518306103776352,
0.347636975509877,
0.952029742278172,
0.133097782327373,
0.475556235050576,
0.703667374655449,
0.672965856116715,
0.502312201309163,
0.158974662962823,
0.254511732726596,
0.352920869995337,
0.985011738252366,
0.0208755508162433,
0.434503763650779,
0.432080585245081,
0.115277832427657,
0.941158274626899,
0.431522508818434,
0.876074839325657,
0.462283711164391,
0.197096110413361,
0.486784082598418,
0.0966018015037299,
0.237336387968313,
0.987191925285008,
0.582947521276282,
0.549627400724975,
0.27357679105996,
0.607131196003003,
0.511139535117494,
0.868296504425023,
0.125437299779354,
0.858776510627371,
0.820326246237534,
0.258535082106728,
0.334332745212285,
0.523993620427322,
0.931500938223442,
0.83664494698711,
0.682968283390146,
0.186012670484377,
0.189565816516786,
0.66798002117685,
0.206888221300621,
0.624069580167564,
0.100060606421931,
0.322166054193939,
0.565227854328802,
0.531583115240365,
0.198240893053934,
0.027511565027531,
0.728679226119388,
0.685024975652352,
0.124113366996922,
0.966015614087701,
0.672216900471699,
0.707060888273204,
0.515643014812676,
0.945793691531659,
0.314192084276207,
0.0267825494645082,
0.814090195491021,
0.439629384055561,
0.885559060091879,
0.634416441728555,
0.698164466162289,
0.219891804838503,
0.158410061690216,
0.62966540392007,
0.0565367513599511,
0.841378345546023,
0.815678074870109,
0.246102567876737,
0.509358366257212,
0.0225662957050681,
0.870172148509963,
0.609418972679143,
0.344732349899007,
0.435400002373103,
0.141002087919508,
0.542973242952942,
0.462911567866295,
0.869681314038896,
0.227998218139633,
0.587024934863218,
0.835696927660935,
0.900215119076993,
0.29408582267076,
0.35133994200795,
0.846008810142991,
0.608277906946967,
0.378122491472458,
0.660099005634011,
0.0479072905368671,
0.263681551098675,
0.294515446896905,
0.746071756699156,
0.483573356402839,
0.452925508587121,
0.375737160619226,
0.54011010776279,
0.294303853667483,
0.191415235023673,
0.786212676105189,
0.803662220390356,
0.213981530728741,
0.65638482414949,
0.413081192603838,
0.55871388109341,
0.0917848260569316,
0.554083280523346,
0.10168712358069,
0.554696393923227,
0.423764594096581,
0.329685342185984,
0.141721328320783,
0.259461521757516,
0.229900460797316,
0.435807151457205,
0.610801463765465,
0.075909270474645,
0.044085057938511,
0.988923955703584,
0.736008276108656,
0.0919923489410395,
0.252605506336598,
0.0305237225398997,
0.838064105640195,
0.736178862739438,
0.483449231592682,
0.21380126579376,
0.276288970502228,
0.777753085260165,
0.405216500817433,
0.0625016461417552,
0.58141530518486,
0.619198032011836,
0.718886470291245,
0.994496498254359,
0.177911912639584,
0.810671296348177,
0.548579778312044,
0.279599036220274,
0.365367690271403,
0.972344372874286,
0.609284378406258,
0.507089018592187,
0.231805894166141,
0.839184839203574,
0.942896170049392,
0.842607357931606,
0.91509411014388,
0.986981228453564,
0.831531313169529,
0.651102385786875,
0.0789735769289423,
0.0841368195061278,
0.681626108792436,
0.917037682569138,
0.820315682245565,
0.165075339919457,
0.130838948362898,
0.0966046522821321,
0.942828425179621,
0.536055449180331,
0.159106298423887,
0.524243730364481,
0.155253480726505,
0.877992768715132,
0.518740228153179,
0.33316539336609,
0.688664065063309,
0.0673200059995614,
0.612764430052025,
0.0540317548690512,
0.0396643784081863,
0.222048807992623,
0.561120773461238,
0.271470272574327,
0.0612336471961968,
0.50401694351063,
0.114077630505933,
0.976327757340077,
0.490998171498532,
0.945608943675463,
0.627430142661291,
0.569971748427475,
0.0297457627159291,
0.309056250988066,
0.487009430530951,
0.850061445427156,
0.474131590907523,
0.617848378893849,
0.946666097709288,
0.416960016087145,
0.15390382807418,
0.105772396133175,
0.941203746451626,
0.309157308800685,
0.983765164848308,
0.459943974139143,
0.642322702632436,
0.672429229445955,
0.527263980138704,
0.2550871322188,
0.726460984315006,
0.566928359012552,
0.477135940211423,
0.287581757776244,
0.838398631586879,
0.53836958740762,
0.791598701286874,
0.952476262092812,
0.514697344282035,
0.282596872319745,
0.898085205768275,
0.142127486943327,
0.85256862074722,
0.927830968484204,
0.451183737931393,
0.33957805081251,
0.777892413445698,
0.925315329304577,
0.95742643017202,
0.724558511154986,
0.342275344926061,
0.111330257780538,
0.830330907288162,
0.283479090912025,
0.420487566581223,
0.814096071670808,
0.743423065051168,
0.0628102687479976,
0.486525301116763,
0.270687044724211,
0.317897400966798,
0.212986284966109,
0.837615403736762,
0.795033341643882,
0.500568042742353,
0.676014035323641,
0.33340292858584,
0.292166743563566,
0.628490296950792,
0.848100273333537,
0.574763615883311,
0.526575502253406,
0.990227760276863,
0.42733223616487,
0.454406470737609,
0.441411497742595,
0.766910287443041,
0.232298883717646,
0.366726826581511,
0.7243367171494,
0.956857394872633,
0.709002171507572,
0.835666974929938,
0.787188301695133,
0.992481262419597,
0.256154541045499,
0.601284373365941,
0.735904327005103,
0.318964810259158,
0.0878096740170427,
0.00659137172931403,
0.636862211225956,
0.300795958983151,
0.844206775466076,
0.431895552404176,
0.801364002191165,
0.520220810324056,
0.765298481455677,
0.0935307452890699,
0.148711107274849,
0.613398754323553,
0.668294361638042,
0.675286609528254,
0.603626514134755,
0.0956265973372509,
0.129693079800202,
0.0450380114116883,
0.862536884780292,
0.36199196398351,
0.411764838458861,
0.58687360146403,
0.318849358390481,
0.120767009500771,
0.422540575928306,
0.106037660085614,
0.113248271920368,
0.678695117439467,
0.707322033451554,
0.849152598925472,
0.997659927698625,
0.795131707468597,
0.855743970654786,
0.63452213845892,
0.095927665986087,
0.699950746120862,
0.066417690863096,
0.897291668177252,
0.220171555979257,
0.831716172318773,
0.990822413931984,
0.368882663254106,
0.445114926176665,
0.659116775104365,
0.0441692727823599,
0.0487414398457582,
0.754743372441616,
0.173862352582562,
0.0937794512574465,
0.617280257221907,
0.535854316566072,
0.505544289716307,
0.204153858220276,
0.854703675422214,
0.62631129968274,
0.626694434614244,
0.960741335507827,
0.739559571603108,
0.30538955158805,
0.668063368493721,
0.588712170062918,
0.303049478821014,
0.463195075496656,
0.444456140717704,
0.937571617279934,
0.559122741482743,
0.144406886372905,
0.00398930767736831,
0.456414409659996,
0.364578442352162,
0.835705479996142,
0.447236823126318,
0.733461106071929,
0.280820405707145,
0.106353597765022,
0.777630378854289,
0.329561845552904,
0.861096970206637,
0.951492731436851,
0.423341297276011,
0.478377226962884,
0.487347048002922,
0.928885586992319,
0.682531085648821,
0.342050722959475,
0.555196886209397,
0.309225519797404,
0.302792058001641,
0.294756457346844,
0.614615071385454,
0.970855426495362,
0.883468627875423,
0.917664550206468,
0.434050501526357,
0.327924768127466,
0.85523616702074,
0.9931732430091,
0.472331654500371,
0.85922547516377,
0.449587652203435,
0.836910096852533,
0.69493095469425,
0.896824475329753,
0.5703712024588,
0.975751360867057,
0.00317807262911372,
0.348001580847428,
0.305313205954299,
0.864275043301412,
0.299494312284279,
0.72865450323031,
0.342652269798635,
0.786841360287201,
0.657540090222629,
0.0251833549817947,
0.128892082781015,
0.212736975966365,
0.334408874779199,
0.431684140782656,
0.50749343377887,
0.949023946164653,
0.402539566812357,
0.390962061188632,
0.866688495905459,
0.836590068338714,
0.718886829316098,
0.721924662926199,
0.829763310882153,
0.191218483350807,
0.581150137624308,
0.279350963085588,
0.0281285797376784,
0.276081091852896,
0.17617543794968,
0.59849978266214,
0.251832452254292,
0.179353511044455,
0.946501363509568,
0.557145658674252,
0.0436285538802056,
0.245995675328186,
0.285800161438901,
0.38628082367884,
0.0328370351497256,
0.94334025166153,
0.411464179126296,
0.161729117930741,
0.156077227627894,
0.745873053905495,
0.593413258713397,
0.663570661406764,
0.694897000070148,
0.995952825525754,
0.0545327221297346,
0.561585495509946,
0.832542893398806,
0.773419551445832,
0.283510157970484,
0.662306204280959,
0.964638034796639,
0.864660295594791,
0.941657167366546,
0.992766614534318,
0.140741387447688,
0.117832604850564,
0.591266396730797,
0.392573839701979,
0.297186115895019,
0.537767760240365,
0.949719498376231,
0.340814669775225,
0.78376343556855,
0.235519659815132,
0.727095493919726,
0.816600471183937,
0.178859911011001,
0.138559672580361,
0.978329589114678,
0.334937138638896,
0.884432726485856,
0.571742847828075,
0.99850780004566,
0.579329726090343,
0.567695672888167,
0.053040521709733,
0.140915221600288,
0.400238566286973,
0.826460073155565,
0.424425379570772,
0.0625447701022703,
0.791098107486543,
0.289085675165563,
0.00420193700315521,
0.783864722020861,
0.42982706261325,
0.12203454185372,
0.375131118285996,
0.82240090231523,
0.4192206582144,
0.912898878526361,
0.772120400691461,
0.760035327989624,
0.696662314094912,
0.00764006004093218,
0.487130821443689,
0.513262784813187,
0.186499971051933,
0.625690494024051,
0.491592373462204,
0.52143711015649,
0.510123220509907,
0.0633352208246175,
0.519944909736488,
0.0894529461345882,
0.631030894178446,
0.572985431911883,
0.230368167734876,
0.0312694599997576,
0.399445504601787,
0.654793547771309,
0.0938142301020279,
0.19054361208833,
0.943879222936872,
0.0980161671051831,
0.974408334109191,
0.373706285084461,
0.220050708958903,
0.349539451929526,
0.196107186934029,
0.639271367173303,
0.262438330455887,
0.96822758762549,
0.399306695162927,
0.959100644550799,
0.975867647666423,
0.886437516606617,
0.472363428898325,
0.162367618718356,
0.512128010630667,
0.963955802360529,
0.683804728874846,
0.0222512306749128,
0.0272910231851465,
0.203749638145673,
0.111704176809501,
0.658321917363592,
0.776735070057556,
0.342072345010039,
0.68959137736335,
0.176180574659342,
0.996865892781348,
0.783405607465378,
0.366724186747672,
0.940745115252559,
0.881421774570561,
0.341132520391202,
0.314451399871358,
0.101472483063802,
0.690671972320728,
0.510558586805388,
0.740743850702766,
0.953110302776615,
0.478786174430878,
0.140050545400032,
0.912210946861753,
0.454653821631639,
0.0264880615409874,
0.384574375294416,
0.617021440349995,
0.538616072171655,
0.348530177654945,
0.300826168759179,
0.560867302846567,
0.375821200840092,
0.504575807370514,
0.67257148012173,
0.034143117738023,
0.281310876962408,
0.014643824666107,
0.723734495101373,
0.45749145162175,
0.0115097169817936,
0.50714010210109,
0.824215638369422,
0.952254832234352,
0.388561876205989,
0.165348158294963,
0.266706231640049,
0.490034359269791,
0.856020131081352,
0.777264818445437,
0.230778209506896,
0.809130433392306,
0.256050992410654,
0.370828754906929,
0.721341380254059,
0.710704814507954,
0.397316816913577,
0.105915755082814,
0.327726254392288,
0.935932889085232,
0.454445932737759,
0.628552423617128,
0.496800191931799,
0.830267133577851,
0.133128230521981,
0.169371671587868,
0.864410251315874,
0.414439107484389,
0.184015496253975,
0.588144745951586,
0.8719305595718,
0.195525213235768,
0.095284847587014,
0.696146197475561,
0.147780045004459,
0.483846723793003,
0.861494356236185,
0.414486276644508,
0.973881083528456,
0.717514486851876,
0.191751094624284,
0.204659292569691,
0.526644919778521,
0.447802087500599,
0.575488047942281,
0.247986299566918,
0.158506901542892,
0.972804864855858,
0.353902055115393,
0.48623315593518,
0.908737753475428,
0.808347987853153,
0.114785579086647,
0.405537944941566,
0.638615121431004,
0.247913809608628,
0.574909616529434,
0.503025372281217,
0.662352917558678,
0.758925112783408,
0.0911701182328025,
0.534283476664817,
0.954450326019176,
0.186454965819817,
0.230429673674716,
0.102230370557974,
0.670301690078481,
0.0919240294452403,
0.516716647202483,
0.644182773141275,
0.809438516297116,
0.708467742292428,
0.848842065710967,
0.336083436075637,
0.156269829327366,
0.424330113187586,
0.584069735642555,
0.314776730870258,
0.397134977577783,
0.937971790757948,
0.801009886805438,
0.305872731053211,
0.746319778611101,
0.915795466357747,
0.711410675994777,
0.384934899576444,
0.163709275500714,
0.286320292524211,
0.88796027185766,
0.826062193059391,
0.0452454048419583,
0.979130390090463,
0.360345669258547,
0.999695731326796,
0.165585355910279,
0.590775343398925,
0.101926101419109,
0.83588704598876,
0.682699372844165,
0.618642749087253,
0.480069818664375,
0.492137889141281,
0.32711049091402,
0.32891188390968,
0.828221325216918,
0.483380320241386,
0.753241997562927,
0.412291060393812,
0.798157051111645,
0.150376974675049,
0.35026285115176,
0.599166937451422,
0.45624970572826,
0.0965826292971999,
0.514962403343507,
0.167660381723037,
0.481517528873644,
0.678671679309882,
0.453980674247248,
0.369477800265643,
0.504733871903612,
0.499226079089207,
0.348608190356106,
0.86507954162782,
0.498921809950341,
0.514193546266385,
0.455854884561084,
0.600847911835112,
0.350080591789484,
0.138554256939587,
0.219490660456703,
0.830150410453859,
0.630692146080868,
0.546601151370723,
0.159062293897878,
0.458913470832125,
0.0299814711464483,
0.912304291460805,
0.871204531691598,
0.828138522258093,
0.0626812661358534,
0.221467382377697,
0.427305459243853,
0.518930971864113,
0.318050011674897,
0.942267863053022,
0.68659135358715,
0.79956754054854,
0.620939541897243,
0.140572027368737,
0.169045340348522,
0.125673413335194,
0.639798106923605,
0.517653530704627,
0.990752954963014,
0.138719916408285,
0.0318470765053514,
0.446607839058436,
0.739567828243397,
0.381927668294836,
0.585162096463685,
0.9590584887001,
0.212078078283033,
0.215854242078892,
0.505659639605162,
0.371140372646572,
0.674767713376678,
0.535641110751611,
0.283444663641716,
0.545972244602615,
0.363779632544042,
0.346125929777569,
0.767439626980312,
0.791085092253557,
0.865056902107344,
0.0854896381895475,
0.733352954840918,
0.551648255228832,
0.885057178738088,
0.354292496272499,
0.692220283063231,
0.0541025186209486,
0.479965909607693,
0.332018389521175,
0.571756049791237,
0.470718864570707,
0.470738305929461,
0.603603126296589,
0.917326703629143,
0.210306134172858,
0.985530795057086,
0.502488799627167,
0.169364622407297,
0.197608872874458,
0.71834304217172,
0.675024262478121,
0.56874924552103,
0.393110755082737,
0.21066537276407,
0.852193909628407,
0.939082999685352,
0.574445005773774,
0.198319838940315,
0.706522626200003,
0.36553009756167,
0.0633767405819971,
0.79201226438955,
0.0988830519369259,
0.615024996276491,
0.677069442661977,
0.453175548209425,
0.307245278874061,
0.731171961282926,
0.933141457817117,
0.639263668395236,
0.302928010608502,
0.403860321922163,
0.110001973859035,
0.906531137370752,
0.321187025551306,
0.320308108031893,
0.892061931962176,
0.823675825178472,
0.489672730904851,
0.0896708043709727,
0.542018866884531,
0.164696992917311,
0.658420050357664,
0.935129621967268,
0.375362365681381,
0.51061395952041,
0.874212621186959,
0.949807371455155,
0.708933798460725,
0.580735246921301,
0.315337468551163,
0.772310539508383,
0.37274751084519,
0.414220520488089,
0.387335535319213,
0.0498169530415055,
0.867396068697514,
0.694580814193273,
0.780988914790092,
0.800537526514632,
0.333844482122848,
0.0839169249329329,
0.204397847971133,
0.443846455981883,
0.990448062303685,
0.525584873522439,
0.764154564479438,
0.8825099938002,
0.349260698700911,
0.253827294918628,
0.972180798636834,
0.891279565585442,
0.418524287835939,
0.630600848528836,
0.82640918755271,
0.79388665351732,
0.141214807583585,
0.700621808274007,
0.743694024972475,
0.85014860604431,
0.281357055195308,
0.0590314930579772,
0.622459145087031,
0.654104566040497,
0.473252014011728,
0.00979467994058257,
0.703921519547664,
0.34064808224358,
0.704375494133856,
0.484910433872096,
0.141185608292551,
0.0382199757910427,
0.568827358805028,
0.345583456729345,
0.482066432238587,
0.559275421108713,
0.871168330251783,
0.246220996252364,
0.441785414443251,
0.220429028487033,
0.500048291170992,
0.413966212614424,
0.111708594072474,
0.918572579006931,
0.0445670606775987,
0.938117781625184,
0.71245923205859,
0.185781868261184,
0.63873958943353,
0.456153256565404,
0.035930474305493,
0.920096644628838,
0.515184750089042,
0.658389619392524,
0.574201210669335,
0.98843676410077,
0.668184299798768,
0.278122729751339,
0.329084845878689,
0.372559793466963,
0.763033163623434,
0.470270454636901,
0.410779769723667,
0.331860521962801,
0.815853911366246,
0.892846201962254,
0.891135943071514,
0.687022241152368,
0.139067197748957,
0.332921357514766,
0.907451270105062,
0.639115488919949,
0.746887570129189,
0.0191598637118749,
0.557688067461219,
0.791454631272449,
0.957277645337059,
0.270147299054147,
0.977236499533633,
0.596017234304928,
0.726300556085212,
0.0131669733734648,
0.516113878933766,
0.241485305708593,
0.67155659323165,
0.0903150891374401,
0.229922069343702,
0.339740892564757,
0.368437818888779,
0.559006915222391,
0.71230068603172,
0.131470982046552,
0.0292773693936306,
0.123080455289725,
0.463331504009353,
0.845131280759876,
0.0159266567863182,
0.354467447080867,
0.532153521912244,
0.154993854535275,
0.687388804595633,
0.439604791551644,
0.794109343455224,
0.434276374259161,
0.458764655263519,
0.351797410450781,
0.225731005065949,
0.416042300134917,
0.62194470997059,
0.202967504599582,
0.0120595339741835,
0.348245265590141,
0.216134477973047,
0.528173412907949,
0.589730571298735,
0.887691071204697,
0.61848850204539,
0.819652640642436,
0.227431963303793,
0.986926320934168,
0.378659555399166,
0.939732649801174,
0.118397302515058,
0.407936925258458,
0.0628131046252386,
0.581728806990072,
0.253068205552673,
0.0787397618772182,
0.93619625407094,
0.785221727464917,
0.233733616412493,
0.623585058200911,
0.2248265185509,
0.0278429598677172,
0.0578614319944109,
0.683591173814419,
0.379640370318499,
0.283592437526021,
0.0996334734836749,
0.00158507982342741,
0.486559942125604,
0.11169300792352,
0.349830345413569,
0.70269442009865,
0.639866420831469,
0.939560916712303,
0.590385491303348,
0.258354922876859,
0.759213556889078,
0.817817454607141,
0.245281243345365,
0.137873112288245,
0.757550103942654,
0.363678546326085,
0.545810037546703,
0.820363209033554,
0.945407353316158,
0.798878243099376,
0.899102970910772,
0.881603606921436,
0.584099970564293,
0.132836586857604,
0.505188664656686,
0.808926489115193,
0.160679546725321,
0.563050096651097,
0.492517662929612,
0.540319917509481,
0.846642534177118,
0.592151136413287,
0.541904997332909,
0.33320247583706,
0.703844144336807,
0.891735343212139,
0.0358968959357109,
0.343710565168276,
0.83129625945878,
0.626282387239059,
0.602065488045134,
0.590509816347859,
0.444099841380538,
0.8473467313905,
0.728382928636103,
0.201649945323192,
0.211025277250924,
0.274192965717145,
0.0220131538910853,
0.15643263010142,
0.0730712083508592,
0.921116124801857,
0.0380362365571951,
0.657171178915152,
0.0539527111938003,
0.543224901213881,
0.466097668030345,
0.214632258384783,
0.106274997864978,
0.958615330959957,
0.754952175894264,
0.952917532042096,
0.550766466907582,
0.296857172761512,
0.286120007879157,
0.254610611244389,
0.188592515507989,
0.322016903814868,
0.598321176412665,
0.0198887749667693,
0.948299291053926,
0.200386663992138,
0.610398591314628,
0.392399131968803,
0.0477333949169765,
0.33878151948507,
0.594049077291996,
0.258758672633562,
0.612974485202215,
0.616062231183081,
0.415191302734982,
0.686045693553074,
0.537178355519277,
0.453227539757838,
0.343216872468226,
0.591131067178739,
0.996452440971719,
0.809314540498571,
0.805763325563522,
0.102727438371036,
0.767929870992866,
0.560715500992125,
0.0556449704131321,
0.318696337434787,
0.857572673753636,
0.341764978292289,
0.573306948679177,
0.0461651892616251,
0.663781882107156,
0.17162812462618,
0.0660539642283944,
0.612081172695421,
0.372014788618318,
0.676452555543023,
0.00448030419856324,
0.419748184000956,
0.0152340745624313,
0.59852938195622,
0.678506856634518,
0.628208559764646,
0.21459161267364,
0.0936981589038382,
0.314254252852059,
0.751769968658579,
0.546925698661677,
0.657471125320285,
0.342901035371656,
0.543378139633396,
0.466785665353195,
0.148664360469517,
0.646105578004432,
0.2347155358804,
0.709379861461641,
0.701750548417564,
0.553411873780848,
0.566952534749616,
0.0435155262441912,
0.126718821994364,
0.613117724011241,
0.707297408351347,
0.298346947086205,
0.679171688239636,
0.319378580581107,
0.670361735704523,
0.355624243316997,
0.323858885245332,
0.0901099192398181,
0.370858317879428,
0.922388267201552,
0.768616775874336,
0.999066877644075,
0.136979879409531,
0.862314935243835,
0.313321130496134,
0.88874984806811,
0.40924063343985,
0.970792255816419,
0.231650882974104,
0.952618773073246,
0.437577921169613,
0.380315243443621,
0.598724351077678,
0.672293457050013,
0.0896951044396009,
0.30047489902958,
0.2257053303652,
0.656647639654878,
0.343990425739433,
0.352424152825225,
0.269765363200458,
0.0512878336251191,
0.65077109991143,
0.948937051905755,
0.370666414671888,
0.321132835150293,
0.304561294757091,
0.69452529991722,
0.411242754855772,
0.675419613102181,
0.616913566653111,
0.179859530264446,
0.674486490280594,
0.753893446062642,
0.0421744650426202,
0.987807620776728,
0.642643294130751,
0.451415098948132,
0.958599876593146,
0.874294177104856,
0.404033871555717,
0.396177797297099,
0.254609420548477,
0.00275822216773323,
0.0684712538814504,
0.344304524988078,
0.303233121662975,
0.294176584712312,
0.000952164177294897,
0.647223547402408,
0.646600737537537,
0.270717527843415,
0.698511381027527,
0.297371836983306,
0.219654579283509,
0.0691777952337534,
0.618504672133599,
0.5242158740406,
0.763703095150973,
0.0297474265237094,
0.199635486677119,
0.380616661338423,
0.209606957253817,
0.874121977423374,
0.134510107401065,
0.251781422296437,
0.861929597734441,
0.777153401531816,
0.703196521244569,
0.820529473861926,
0.651447578171011,
0.107230392800286,
0.216707270693363,
0.906056998719488,
0.109988614968019,
0.285178525040475,
0.250361523707566,
0.413221736630994,
0.579355109752787,
0.251313687884861,
0.0604452835677402,
0.225955846824663,
0.522031215728275,
0.758956665060928,
0.523327683807969,
0.741685795011784,
0.828134460294682,
0.141832355475906,
0.265901669052384,
0.591837555445655,
0.171579782465277,
0.465537155729503,
0.972454216784078,
0.381186739719094,
0.339659132687216,
0.106964323719481,
0.632968162015531,
0.201588730421657,
0.884117725251297,
0.3361646832601,
0.0221182038179218,
0.535565303422308,
0.443395076060386,
0.238825474976946,
0.441622301676135,
0.553383691028405,
0.524004000017421,
0.6919838253837,
0.96660542812506,
0.103359109304547,
0.943297513734222,
0.0270507112271389,
0.32931495612921,
0.465328728996836,
0.786007376288067,
0.852642639937178,
0.20701452400862,
0.614141836582749,
0.994474995413085,
0.472916193061004,
0.205979391562743,
0.166054777412701,
0.938453348790507,
0.178433607881159,
0.547241517131795,
0.278112481477723,
0.285397932066302,
0.180209679147326,
0.47970121189938,
0.169515656851938,
0.516374362407427,
0.501819415717302,
0.705080960274246,
0.959769438467812,
0.740644890694248,
0.146703261950381,
0.513153129030556,
0.264648890246008,
0.838687087334081,
0.479758556689955,
0.368007999550555,
0.781984600602642,
0.506809268382755,
0.697322955679764,
0.247313329599478,
0.292816644205161,
0.549965595151282,
0.454327853608098,
0.90695848078791,
0.544440590564367,
0.927244046669101,
0.112937871884991,
0.710495367977068,
0.865697394993947,
0.291371480231812,
0.257736885108863,
0.143809876006008,
0.576769412298114,
0.437946564256189,
0.623511087905388,
0.746285069150052,
0.954320926663615,
0.12533050362269,
0.451366029424298,
0.914090364665766,
0.865975394316937,
0.598069291374678,
0.427243493230661,
0.130624284562946,
0.436756378243098,
0.907002050386277,
0.4986322841135,
0.21874097884574,
0.413811318303371,
0.195955239793265,
0.466054308445218,
0.706627962508531,
0.745920834944547,
0.920382162053316,
0.613586443296441,
0.290361425043252,
0.847626208256756,
0.726524315181432,
0.000856793020319563,
0.713323602785041,
0.0178957949475831,
0.258593678129182,
0.857133479256711,
0.594665207245697,
0.696540242385371,
0.480644566696437,
0.340950276395748,
0.650861168583325,
0.605975070319127,
0.792316305820046,
0.56495153278343,
0.471950464636064,
0.390385596729063,
0.992195026014091,
0.60257474919901,
0.827141974972161,
0.899197075934707,
0.101207032846849,
0.0458829533522404,
0.313008393772416,
0.297162272640114,
0.511937261797459,
0.0196363562809473,
0.0430831075846605,
0.432319423385113,
0.633222799577388,
0.333444532627912,
0.279945631176208,
0.35974711429316,
0.334301325648232,
0.993269234426911,
0.377642909706404,
0.592895003777414,
0.85040271321796,
0.972308116952101,
0.289435245697124,
0.331047279448736,
0.313258392882188,
0.940296414280448,
0.937022350233524,
0.105574698236573,
0.505247946598217,
0.408972814403927,
0.495960294965636,
0.497442972612308,
0.0115475631372759,
0.323102269937798,
0.396640048081353,
0.112754595984125,
0.368985223290038,
0.709648442319431,
0.4099168690899,
0.880922485553158,
0.729284798600378,
0.45299997667456,
0.31324190847261,
0.362507597712105,
0.786444509302473,
0.593187540114479,
0.722254712005264,
0.120745834950705,
0.586456774075728,
0.0998976212460071,
0.713640838728119,
0.436859486828027,
0.0722057381981079,
0.00307608442524266,
0.767906766276763,
0.385464131080296,
0.943372498705691,
0.704929116044626,
0.49103882978253,
0.448620445303908,
0.113901929982892,
0.986999124748166,
0.946063417916216,
0.125449493120168,
0.310101394220303,
0.342703465531908,
0.238204089569954,
0.679086617976002,
0.0523519073856771,
0.648120958659854,
0.560009103063498,
0.781636705986055,
0.101120934868753,
0.873251011536108,
0.144144303232499,
0.887565444171226,
0.466438551184925,
0.866399015703424,
0.00831127865627002,
0.0528953247949925,
0.966296636949431,
0.72195211785005,
0.48975481162302,
0.0385023746818781,
0.725028202275293,
0.257661577899783,
0.423966506227835,
0.668400700515323,
0.962590693944409,
0.915005336010365,
0.117021145353569,
0.0764926239273011,
0.90200446029287,
0.0630845628041237,
0.201942117047469,
0.212105854513173,
0.405788028801693,
0.440146206617423,
0.891192472489175,
0.45813993618737,
0.0882671648116164,
0.451201575087012,
0.239776642173425,
0.18938809968037,
0.324452586157458,
0.383920945405923,
0.0769535438515961,
0.790891137808045,
0.250319960643686,
0.0852648225078661,
0.843786462603037,
0.216616597593118,
0.807216940357916,
0.333541274226057,
0.255118972274996,
0.532245142167548,
0.59120285212584,
0.679085478502831,
0.200645842682871,
0.553793545604587,
0.594090814047535,
0.31766698803644,
0.630286169531888,
0.496095274340406,
0.380751551306225,
0.832228287045019,
0.708201128853579,
0.786539580107918,
0.272374493196781,
0.599393600877092,
0.244679515829626,
0.360641658474059,
0.0505951754984423,
0.484456158003051,
0.550029758154428,
0.375047761655901,
0.868377103874636,
0.626983302006025,
0.165938898998284,
0.118697064052661,
0.712248124979552,
0.00972536160132166,
0.335313661645778,
0.519465064871807,
0.343266635827379,
0.590432634386435,
0.0517102070393554,
0.934469487953218,
0.269518112423605,
0.252356049722226,
0.488263033092144,
0.863608926936802,
0.570023037758667,
0.118549202624033,
0.359704200811546,
0.950774589064892,
0.950777489669052,
0.0679053291994638,
0.737314168707148,
0.223151982400171,
0.667298930076556,
0.981993685002436,
0.58379364087423,
0.717894105574998,
0.466449842539825,
0.133823399028659,
0.0929418667652374,
0.3348269459488,
0.760806701034683,
0.258880766229183,
0.453524010467121,
0.473054825548574,
0.268606127830505,
0.7888376721129,
0.992519890886042,
0.611872763657883,
0.379270306033674,
0.0442300974597363,
0.54634225114544,
0.64878841892294,
0.296586147181963,
0.0346052837719234,
0.512397345394081,
0.866609184940629,
0.153154486395956,
0.872101546205628,
0.817383774005521,
0.103931975599347,
0.940006875405091,
0.554697942247008,
0.327083958465179,
0.607305805015986,
0.536691626783782,
0.91087759933941,
0.325199910125323,
0.00314146932360785,
0.0447009979024068,
0.41814177689056,
0.337968415272407,
0.805507699402751,
0.677022543119743,
0.791492425739529,
0.278562524485664,
0.945628670950247,
0.580330097852429,
0.271082414906045,
0.557501434142469,
0.959600403886102,
0.315312512365781,
0.103843684822248,
0.608388822343382,
0.611898659547744,
0.138448969059833,
0.120786167271801,
0.478507844488373,
0.291603455455789,
0.992887713477429,
0.295891618028233,
0.395535431520797,
0.932894588416859,
0.850589560275241,
0.722619389985976,
0.540200392967183,
0.387281187059023,
0.633496988859725,
0.865400303092506,
0.390422656382631,
0.678197987227793,
0.283542079983066,
0.728391071655038,
0.483705686164883,
0.960564623102809,
0.519883497394567,
0.762268210650547,
0.906193294053056,
0.100213594781334,
0.0333506255565913,
0.463694727729864,
0.0598139982017754,
0.348663137922372,
0.567538413017773,
0.668202821010818,
0.960561797935777,
0.705987382077606,
0.78898898828262,
0.439069641958489,
0.997590837533395,
0.781876701760048,
0.734961259986722,
0.393126268588531,
0.714771289711246,
0.585550819796301,
0.115745658108846,
0.254971682678429,
0.972832006855324,
0.749242647434232,
0.120371985305274,
0.363254662772294,
0.427440634196364,
0.40391406528834,
0.0916457344273318,
0.911146320361247,
0.364478688391148,
0.611529231821899,
0.673414530546132,
0.270671981978543,
0.711742826603233,
0.706765156102723,
0.734366709708407,
0.771556824805009,
0.0554282940250953,
0.301905122260519,
0.439759645350165,
0.0159900914952113,
0.00789250387246371,
0.228748633632785,
0.4550597334537,
0.00548334140585891,
0.0106253349271721,
0.19002099297476,
0.39860960999439,
0.725396624638418,
0.775571812771061,
0.514355268568897,
0.980368307316847,
0.748403819626385,
0.263597915537468,
0.100740292622121,
0.111658481933017,
0.691038549733832,
0.504654357910461,
0.203304216360349,
0.602184869629417,
0.869133046301609,
0.814833448182248,
0.275599399709888,
0.139805027814491,
0.52657627431982,
0.982364556278272,
0.874171737988559,
0.298133099124829,
0.0377928498377059,
0.176076859783417,
0.737892744474994,
0.0537829413329172,
0.183969364121542,
0.966641378107779,
0.508842674786617,
0.189452705527401,
0.977266713034951,
0.698863667761378,
0.588062315521791,
0.70266333767337,
0.474435480066778,
0.102417583625027,
0.683031644524555,
0.222839299227502,
0.366015499162495,
0.783771937146677,
0.334497781626181,
0.0570540484306654,
0.288426295057138,
0.53780199798653,
0.659238918060082,
0.157559340893086,
0.352635446168778,
0.934838318235631,
0.297364369173238,
0.879211720488598,
0.917202874048242,
0.171536106696136,
0.177344819147766,
0.954995723885948,
0.347612966479553,
0.915237564088422,
0.00877866475320359,
0.531582330601095,
0.88187894173054,
0.517621339539821,
0.721035036128496,
0.859145654765491,
0.216485006835538,
0.309097351650287,
0.561808991973199,
0.690920486902315,
0.411514935275314,
0.244840636032094,
0.913759786595479,
0.77753043490347,
0.0286125731787703,
0.248257567755998,
0.834584483334135,
0.317038868235908,
0.786059565742528,
0.493823401394218,
0.474598209128994,
0.138695011445645,
0.428661719164188,
0.771962578302232,
0.0179067319342432,
0.345864592746768,
0.943498684998368,
0.195251551082009,
0.300860316167055,
0.291111651477922,
0.11048911470477,
0.309638980920258,
0.822693982079017,
0.992368056900971,
0.827260320460079,
0.543729018207513,
0.851513711200801,
0.0437453268299556,
0.8528263698578,
0.413322702708339,
0.734665813732271,
0.264341304667453,
0.658163338740432,
0.648425599862088,
0.0418717391052617,
0.686775911919203,
0.896683167618086,
0.876456222905059,
0.00381477968944925,
0.682742733360614,
0.370279623833615,
0.478412989284104,
0.821437744806259,
0.798941342997803,
0.250375567120675,
0.839344476740502,
0.14480593527891,
0.193874252119043,
0.0345960278225113,
0.445666251445965,
0.484985903596965,
0.145085142527281,
0.755305232366223,
0.30767988521032,
0.13745319896259,
0.582565552360642,
0.851408903417834,
0.988966910163391,
0.626310879190597,
0.704235272809973,
0.402289612406068,
0.360976692457207,
0.968576577943087,
0.0604529511465006,
0.00940229231929513,
0.0104483165826873,
0.747228863065703,
0.906085459937381,
0.886904539487746,
0.751043643220814,
0.588828192832334,
0.2571841628557,
0.229456632039257,
0.410265937638593,
0.0561255053878415,
0.479832199159931,
0.249610413913434,
0.200931440666752,
0.673706451278975,
0.284206441735945,
0.646597692112717,
0.158692354410278,
0.429291584728887,
0.401902924013279,
0.466372240086259,
0.566744783691477,
0.98446847637392,
0.317781143038432,
0.555711693854868,
0.610779355098856,
0.0220164158484044,
0.958001306260937,
0.971756048021724,
0.990592993791491,
0.0184542569417759,
0.98115834034102,
0.00104130990851732,
0.76568312047314,
0.88724379981274,
0.887945849861924,
0.516726763228293,
0.476071992645074,
0.145130012251963,
0.74618339526755,
0.886337930283667,
0.201255518105466,
0.22601559396182,
0.135948343731439,
0.402186958772217,
0.899722045706456,
0.420154785933045,
0.0487846508849341,
0.0584143996510722,
0.849446370661932,
0.450687574898213,
0.524786639737332,
0.416191153887748,
0.435156051272133,
0.842567783241425,
0.971902847742617,
0.0459354059053284,
0.864584199089829,
0.929904154003553,
0.0176914534613916,
0.855177192415659,
0.948358410945329,
0.998849793802411,
0.856218502324176,
0.714041530952808,
0.886093593615151,
0.744164351720439,
0.23076829371544,
0.362165585794563,
0.889294364438064,
0.976951688982989,
0.248503515612568,
0.090549882077868,
0.202967282944809,
0.384451859344007,
0.492736840850085,
0.102689328185604,
0.804606645277053,
0.54152149173502,
0.161103727836676,
0.654053015473323,
0.992209067098894,
0.685890368039669,
0.0702441693610717,
0.427365117905366,
0.528458150815432,
0.042147016638027,
0.473300523810694,
0.3930423494396,
0.97205117064158,
0.490991977737747,
0.248219541389597,
0.920409581586909,
0.489841771074497,
0.104438043248112,
0.634451112074056,
0.375935364223987,
0.848602395434213,
0.865219406255157,
0.73810095001855,
0.737896759406615,
0.842171094772486,
0.986604465631118,
0.828446641484483,
0.0451383772516336,
0.371056324975126,
0.321183481868908,
0.147827705437237,
0.175662969786517,
0.862704974069588,
0.308931433739574,
0.829715985725502,
0.854914040702821,
0.994821801779243,
0.899960155086573,
0.282279158142525,
0.523279952129014,
0.9421071717246,
0.75557968195322,
0.916322301568613,
0.914158342366181,
0.246571659225305,
0.16454184249255,
0.834567923487429,
0.736413430765464,
0.268979885740662,
0.469019035095823,
0.112348794523789,
0.117582280709214,
0.33423844088532,
0.850449745008,
0.855479040115829,
0.176409535657805,
0.837054210173457,
0.683925681134651,
0.221547912909439,
0.208110534682921,
0.00510916300355883,
0.369375618812337,
0.383773504935099,
0.867814137073147,
0.678307052551912,
0.21348949019494,
0.722728177310307,
0.673128853865493,
0.113449644815852,
0.00500733498717069,
0.196408805528846,
0.0555568160747908,
0.76058701694039,
0.112731106631798,
0.969715158440971,
0.00715867616569562,
0.277272949124348,
0.804283081462739,
0.743572106931159,
0.54625283486501,
0.273302116558562,
0.855920901454948,
0.663835116039885,
0.607540557443882,
0.706370645997287,
0.519314155690053,
0.783950093101687,
0.543424856170744,
0.203239836824704,
0.00549800601112563,
0.751535390853665,
0.208348999828263,
0.374873624823463,
0.135308895323104,
0.0761631364357486,
0.0531806769097134,
0.348798385518043,
0.798891313746055,
0.726309530775207,
0.462248030333895,
0.803898648733226,
0.922718336304053,
0.517804846874347,
0.564485665207955,
0.0354494424701899,
0.487520004849657,
0.57164434137365,
0.312722391594538,
0.291803086312396,
0.315216447839149,
0.858975226459547,
0.565105202870958,
0.171137349294097,
0.522810342033771,
0.172645759849178,
0.877507995291384,
0.0421244977238236,
0.956595853416527,
0.420932850996467,
0.245364334548528,
0.962093859427652,
0.172468241850132,
0.45371333437679,
0.336967483785454,
0.307777137173236,
0.529876470812539,
0.390148160695167,
0.656575523156941,
0.328767784092933,
0.116457691004713,
0.118823553025175,
0.132666432360497,
0.0391760268431045,
0.636628399899522,
0.697152097568452,
0.0746254693132944,
0.124148404749179,
0.268796438942103,
0.387347860907832,
0.415951491061575,
0.584012886781252,
0.246323086901718,
0.981056693932534,
0.755150236075348,
0.769133429401151,
0.153702453316051,
0.632658231366732,
0.811257927124974,
0.110298306266916,
0.0535910818975377,
0.0566222612078405,
0.0723921652289071,
0.22605932374767,
0.510335595584631,
0.409359649014361,
0.533836461386567,
0.0402120659315083,
0.799507809709528,
0.190411984077847,
0.368979850024441,
0.915965500714241,
0.309235537103021,
0.501646282384939,
0.955141527557346,
0.945863937468205,
0.19879837948773,
0.0297669964049789,
0.0700123417517228,
0.467594818429833,
0.417114857312811,
0.485963832813298,
0.0516077052110842,
0.663437944680191,
0.467020526280171,
0.806757941286433,
0.43257137361568,
0.620722979596221,
0.439416172187504,
0.243829300274993,
0.731021286328799,
0.493007254550703,
0.300451561482833,
0.803413451557706,
0.719066578298373,
0.810787157067464,
0.212773100572067,
0.252903039219278,
0.850999222998973,
0.012280909815934,
0.443315023297125,
0.219979072557753,
0.928246410995837,
0.752550560865808,
0.721625354942691,
0.883387938087521,
0.698414497868351,
0.920423734430421,
0.913154934958161,
0.768426839620074,
0.388018552860254,
0.330269791805311,
0.254390671967711,
0.439626258071338,
0.993707736485502,
0.721411198247881,
0.246384198892109,
0.42627910963552,
0.342134177378441,
0.685800371545274,
0.670108409910513,
0.0731554632415788,
0.178807625630315,
0.970559971393347,
0.876568915264946,
0.897874203928688,
0.78134712799515,
0.0893420153713515,
0.150777243147966,
0.632346350528461,
0.101622925187285,
0.594092266445091,
0.852325423086214,
0.0298693357174608,
0.346642826845237,
0.573950777563244,
0.913257274270643,
0.0450573242479271,
0.494374511993665,
0.826412208763143,
0.813484163868001,
0.882393064853918,
0.156682000102793,
0.0678748353700502,
0.322019322459595,
0.150389736588294,
0.789286033617931,
0.568403521351704,
0.576668846223815,
0.131420210996373,
0.254203892431317,
0.246777256134328,
0.204575674237951,
0.433011518061632,
0.217337227062014,
0.0811445890372361,
0.33088572199032,
0.998684355522825,
0.170486604408588,
0.481662965138286,
0.631030705585624,
0.272109530061534,
0.0757552311177157,
0.483356128671838,
0.301978865778995,
0.422398058428614,
0.0573069057694203,
0.215236139583977,
0.467455382676542,
0.551681417763085,
0.0416483478814589,
0.280939546544542,
0.434074482151342,
0.198330348449913,
0.348814381914593,
0.756093804610937,
0.348720085038208,
0.138100415532524,
0.324497325962641,
0.925388931262022,
0.269520626528897,
0.578701218393958,
0.172166186930689,
0.474096300766848,
0.0117127364555899,
0.389503414458364,
0.555240890269746,
0.34259845844591,
0.388187769515527,
0.725727494678333,
0.824261423584196,
0.0192184746354904,
0.997837024739868,
0.900016654701911,
0.502574603307329,
0.299815890518863,
0.322414712664864,
0.55988150954241,
0.51505203010284,
0.789870095807067,
0.111562926839834,
0.55670037844996,
0.0708096418859482,
0.545637409456837,
0.755030726899873,
0.419624023800541,
0.301731213602112,
0.103750811472419,
0.557724439333065,
0.626228539564753,
0.0291397422687801,
0.827245065861961,
0.204929757958711,
0.201305929665131,
0.30134136662881,
0.216642494414301,
0.590809344123495,
0.856582256898555,
0.559240952860211,
0.978997113639022,
0.582309751111227,
0.383502375978745,
0.998215588274512,
0.580146775851095,
0.283519030214995,
0.500790191581841,
0.879962666369957,
0.60593374334552,
0.0606717006585895,
0.395014696007136,
0.395803838686926,
0.172234627498423,
0.951715074457095,
0.466613480572874,
0.71787203695526,
0.706745800891307,
0.886237504373415,
0.019603250557372,
0.810496612363726,
0.44396194370648,
0.645831790122125,
0.839636355098168,
0.27120700910278,
0.850761548080836,
0.0409422842976368,
0.572548375731589,
0.0674040420294758,
0.631751628421131,
0.429130632164483,
0.626644994889686,
0.610748741594492,
0.0114403832757102,
0.0101473704027698,
0.608964329403343,
0.591587159126805,
0.293666401083426,
0.109754520519522,
0.471549825031101,
0.899600144428946,
0.170426221178112,
0.866564521038236,
0.295403982650211,
0.342660849142196,
0.818279595495332,
0.762017463223086,
0.0605328856317945,
0.525025395920978,
0.648254967130839,
0.0801361361891665,
0.335522008284704,
0.0922169103716579,
0.725967926776953,
0.17515836291721,
0.363423919940099,
0.576729474392128,
0.216100647214847,
0.935972295671688,
0.644133516887265,
0.847852275635978,
0.365102927836172,
0.27077851131129,
0.458601016764809,
0.376543311111882,
0.280925882179721,
0.0675653457024905,
0.968130470238687,
0.574592283263147,
0.177319866687674,
0.439680294804126,
0.474192427226432,
0.347746087865786,
0.306244815376701,
0.769596409876643,
0.690406937007982,
0.124524410406372,
0.531613872634067,
0.750939823105438,
0.649549806793011,
0.179868839299245,
0.831075959294604,
0.985071815077715,
0.272085750136564,
0.557043885605896,
0.160230177529263,
0.635509670076663,
0.133773359998024,
0.37633082474411,
0.571481965282691,
0.777906876885289,
0.224183099914427,
0.936584893118863,
0.0486853877309176,
0.682784116679236,
0.313128203765083,
0.329611269910639,
0.750349462381727,
0.281258673538109,
0.904203553173786,
0.927669329069401,
0.720938968342235,
0.378395979934556,
0.275415416935187,
0.027183783718936,
0.147992389345538,
0.965822353943168,
0.151708194125308,
0.679606261979605,
0.716762176582945,
0.801258000918319,
0.859475101744512,
0.547838135877549,
0.786329815530372,
0.131560851415415,
0.104882021017783,
0.946559993059635,
0.767070521492078,
0.238655381015807,
0.322890817338084,
0.338552486309108,
0.0165622574354346,
0.547073917252512,
0.27513737942797,
0.0652476456320135,
0.229858033466087,
0.588265583193053,
0.394858915542652,
0.980207496313475,
0.869524257196823,
0.299062468716438,
0.907876824917214,
0.590463225073397,
0.677458448650995,
0.18329224138674,
0.617647008792333,
0.825450838462194,
0.149114595329908,
0.769355203383302,
0.505057099976138,
0.865876771912853,
0.570613203835959,
0.364532201254988,
0.41371490732474,
0.356943019366331,
0.496093052670403,
0.518596928808185,
0.303503011960305,
0.26316357369682,
0.757252309823992,
0.62639382976405,
0.601716060471589,
0.773814567725088,
0.173467746550901,
0.876853439899559,
0.839062213357101,
0.403325780482649,
0.465119022626951,
0.233921128434092,
0.383533276330462,
0.334643279358113,
0.53298359715053,
0.291410100782015,
0.925106504897171,
0.210442045801525,
0.474702342634416,
0.542753513223842,
0.0358928837980576,
0.623816937964324,
0.312108716141483,
0.540949983774195,
0.489693709411516,
0.882721920443104,
0.905482185494845,
0.903408617201917,
0.239664939343773,
0.401575237699587,
0.422005545544441,
0.543167951304078,
0.664738811862068,
0.179257854902771,
0.169561780602467,
0.266454871867995,
0.953072422627859,
0.343029527619029,
0.143308311301893,
0.792134635519299,
0.746355308101678,
0.608427333928844,
0.0260557639533914,
0.129888583966479,
0.943070613752618,
0.559039361103922,
0.421298684748494,
0.868177118184127,
0.769481406905447,
0.896001027382911,
0.410930630942309,
0.805374290703505,
0.519817964881574,
0.723039347549453,
0.3463242744777,
0.00951167429308951,
0.605761267526895,
0.251806459506884,
0.912920291495007,
0.845426206870669,
0.65338169720647,
0.334925836573786,
0.388594157709085,
0.318120508602876,
0.514183691476557,
0.558155938777214,
0.584575380470872,
0.467256114104416,
0.901185466396243,
0.727883691772764,
0.259390749158054,
0.64754077403226,
0.336311025701608,
0.285446513111445,
0.777429357998739,
0.279381638988565,
0.844485874681028,
0.198728042281572,
0.147558756707031,
0.613967281120814,
0.0947290696644825,
0.558489387649339,
0.419341571824319,
0.614547034546056,
0.281528734733131,
0.765665846302018,
0.624058708839146,
0.887290002260027,
0.0174723053432406,
0.536978999868491,
0.732716208665034,
0.670854002549711,
0.871904836442277,
0.121310366374119,
0.988974511618248,
0.386088527918834,
0.679466305151333,
0.573549891623459,
0.853344642023251,
0.580651771081915,
0.301433583396223,
0.112735390715643,
0.228192544648513,
0.637744609097831,
0.39818190429275,
0.00562190218159086,
0.917126248086396,
0.242667778508117,
0.204349944463163,
0.0646850043277652,
0.856635059628931,
0.299079014127645,
0.623174391977105,
0.275976630987589,
0.913626049139363,
0.904703127175897,
0.0416424768239457,
0.537684757512847,
0.791993128970262,
0.0591147821671864,
0.0746637573813385,
0.524709337635296,
0.729968785182558,
0.946568593823616,
0.646019704009415,
0.718943296335146,
0.332657121276789,
0.325486008695087,
0.292493187492943,
0.186001762834378,
0.906137779777002,
0.593926770889166,
0.298737154015683,
0.134330323959855,
0.231671379521336,
0.696919058308433,
0.139952226141445,
0.148797627142071,
0.939586836816551,
0.344302170604608,
0.213482631469836,
0.796221895979821,
0.643381185197915,
0.836657023912602,
0.0721985269674093,
0.557007233871616,
0.741360150622837,
0.113841003791355,
0.0946919909188021,
0.533353279127438,
0.172955786424203,
0.169355748300141,
0.058062616297073,
0.902924571606761,
0.115924342123756,
0.704082320306488,
0.621867867476245,
0.448581463400545,
0.0295683285359146,
0.914361054969188,
0.634583226700585,
0.935706108312917,
0.508287825858355,
0.933320380716268,
0.0700364318071103,
0.739959205379691,
0.63023943855904,
0.209988657948556,
0.888756832987422,
0.56982627490993,
0.554290829018825,
0.102239463991597,
0.366048170424089,
0.197672013751078,
0.938896487904199,
0.438246697391498,
0.754679247622694,
0.680256638061375,
0.552087701648515,
0.849371238541497,
0.213609917188813,
0.725043488072717,
0.0187269868416372,
0.271672533485886,
0.627968059213817,
0.134651328965394,
0.975754854258036,
0.249835926224401,
0.583232792365939,
0.00532318232828899,
0.164196980727928,
0.217816018600862,
0.941029291106867,
0.672484806586283,
0.151136398851469,
0.0110657224483163,
0.412444011965973,
0.78137583741051,
0.221054380862533,
0.301200844487735,
0.351202111854778,
0.775345209881358,
0.403440308479332,
0.717250282744528,
0.973017223632437,
0.34233679638353,
0.155496979670365,
0.72769647078947,
0.0225934339792437,
0.707584681318879,
0.577067708865305,
0.236203351168057,
0.432628168925935,
0.595794695706942,
0.507875885119604,
0.0605962276740913,
0.730446024672336,
0.483630738911978,
0.310432153898493,
0.313678817038275,
0.488953921240267,
0.474629135092082,
0.531494835639137,
0.429983211881473,
0.147113941212703,
0.682631234956268,
0.441048934795451,
0.559557953178677,
0.464007071901116,
0.662103315657984,
0.860758797666411,
0.815209184221555,
0.437448525073681,
0.264199106145743,
0.532459466500422,
0.410465748240457,
0.606535902529273,
0.687956446170787,
0.138162218564265,
0.629129336508517,
0.395541127489666,
0.715229927429571,
0.865332688142235,
0.828169296415602,
0.311024623136513,
0.373208572796177,
0.888765524555354
        };
        static const int numrands = 10000;
        static thread_local int randpos = 0; // Thread-local to let asyTS-BO work!

        for ( i = 0 ; i < totsamp ; ++i )
        {
            SparseVector<gentype> &xq = res("&",i);

            int kk;

            for ( kk = 0 ; kk < dim ; ++kk, ++randpos )
            {
                //randufill(xq("&",kk%sampPerSplit,kk/sampPerSplit),0,1);
                // Generate the SAME sequence normally, generate SEEDED sequence for automatic method
                xq("&",kk%sampPerSplit,kk/sampPerSplit) = xrandvals[((i*dim)+((isautorand ? randpos : kk))%numrands)];
                xq("&",kk%sampPerSplit,kk/sampPerSplit) *= (xmax(kk)-xmin(kk));
                xq("&",kk%sampPerSplit,kk/sampPerSplit) += xmin(kk);
            }
        }
    }

    return totsamp;
}
