//FIXME: putting a 0.5 min on lengthscale for task relatedness seemed to help?

/*
Wmethodkey 4: set if gradorder > 0.  For everything but K2 assert that if 4 set then 2 should be set
FIXME: when igradorder > 0 but no farfar then need to return a vector or matrix of appropriate size.  Basic method is to use
a helper to try to (a) turn T into vector/matrix (throw if not gentype) and (b) access element i (or i,j) of T (throw if not gentype).
Then jam the vectorised result (or de-vectorised if it's a full covariance) into this.  This result then naturally filters through
ghTrainingVector(-1) of various models (though need to make sure ghTrainginVector doesn't *assume* K double for -1 as per
svm_scalar) and the result naturally comes out at the end.  Also note in instructions that setting a_6 without gradient will
give a vector or matrix, which could potentially be used as a matrix-valued kernel.
*/
//
// ML (machine learning) base type
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
#include "mercer.hpp"
#include "ml_base.hpp"
#include "randfun.hpp"
//#ifdef ENABLE_THREADS
//#include <mutex>
//#endif
#include "nlopt_direct.hpp"

#define LARGE_TRAIN_BOUNDARY       5000
// SEE ALSO KCACHE_H

thread_local SparseVector<int> ML_Base::xvernumber;
thread_local SparseVector<int> ML_Base::gvernumber;
//#ifdef ENABLE_THREADS
//std::mutex ML_Base::mleyelock;
//#endif

//The original version used fixed-length arrays.  This is a very bad idea and the version numbers quite quickly overflow!
//std::atomic<int> *ML_Base::xvernumber = nullptr;
//std::atomic<int> *ML_Base::gvernumber = nullptr;


// Training vector conversion

int convertSetToSparse(SparseVector<gentype> &res, const gentype &srrc, int idiv)
{
    if ( srrc.isValSet() )
    {
        const Set<gentype> &src = srrc.cast_set(2); // Note use of 2 here to finalise globals but not randoms!

        res.zero();

        NiceAssert( src.size() <= 5 );

        int i = 0;
        int j;

        if ( src.size() >= 1 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; ++j )
                {
                    res("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            ++i;
        }

        if ( src.size() >= 2 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; ++j )
                {
                    res.f1("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            ++i;
        }

        if ( src.size() >= 3 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; ++j )
                {
                    res.f2("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            ++i;
        }

        if ( src.size() >= 4 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; ++j )
                {
                    res.f4("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            ++i;
        }
    }

    else
    {
//        const Vector<gentype> &src = (const Vector<gentype> &) srrc;
//
//        res = src;
        res = srrc.cast_vector(0); // Note use of 2 here to finalise globals but not randoms!
    }

    if ( idiv > 0 )
    {
        res.f4("&",6) += idiv;
    }

//    return ( srrc.isValEqn() & 0x10 ) && !(srrc.scalarfn_isscalarfn());
//    return srrc.isValEqn() && !(srrc.scalarfn_isscalarfn());
    return srrc.scalarfn_isscalarfn() ? 0 : ( srrc.isValEqn() & 8 );
}

int convertSparseToSet(gentype &rres, const SparseVector<gentype> &src)
{
    if ( src.indsize() == src.nindsize() )
    {
        Set<gentype> &res = rres.force_set();

        res.zero();

        int doit = 0;

        Vector<gentype> temp;
        gentype nulldummy('N');

             if ( src.f4indsize() ) { doit = 4; }
        else if ( src.f2indsize() ) { doit = 3; }
        else if ( src.f1indsize() ) { doit = 2; }
        else if ( src.nindsize()  ) { doit = 1; }

        if ( doit >= 1 )
        {
            if ( src.nindsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.n()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }

        if ( doit >= 2 )
        {
            if ( src.f1indsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.f1()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }

        if ( doit >= 3 )
        {
            if ( src.f2indsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.f2()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }

        if ( doit >= 4 )
        {
            if ( src.f4indsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.f4()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }
    }

    else
    {
        Vector<gentype> &res = rres.force_vector();

        retVector<gentype> tmpva;

        res = src(tmpva);
    }

    return rres.isValEqn() && !(rres.scalarfn_isscalarfn());
}



// First we define those functions that never need to be polymorphed, and do
// any changes via functions that may be polymorphed.

ML_Base::ML_Base(int _isIndPrune) : kernPrecursor()
{
    {
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif

        //if ( xvernumber == nullptr )
        //{
        //    xvernumber = new std::atomic<int>[NUMMLINSTANCES];
        //    gvernumber = new std::atomic<int>[NUMMLINSTANCES];
        //
        //    int i;
        //
        //    for ( i = 0 ; i < NUMMLINSTANCES ; ++i )
        //    {
        //        xvernumber[i] = 0;
        //        gvernumber[i] = 0;
        //    }
        //}

        xvernumber("&",MLid()) = 1; // non-zero to indicate existence
        gvernumber("&",MLid()) = 1; // non-zero to indicate existence

//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
    }

    assumeReal       = 1;
    trainingDataReal = 1;

    wildxaReal       = 1;
    wildxbReal       = 1;
    wildxcReal       = 1;
    wildxdReal       = 1;
    wildxxReal       = 1;

    wildxinfoa = nullptr;
    wildxinfob = nullptr;
    wildxinfoc = nullptr;
    wildxinfod = nullptr;

    (allocxinfoa) = nullptr;
    (allocxinfob) = nullptr;
    (allocxinfoc) = nullptr;
    (allocxinfod) = nullptr;

    (wildxtanga) = 0;
    (wildxtangb) = 0;
    (wildxtangc) = 0;
    (wildxtangd) = 0;

    xpreallocsize = 0;

    UUoutkernel.setType(1);
    RFFkernel.setType(1);

    isBasisUserUU = 0;
    defbasisUU = -1;
    isBasisUserVV = 0;
    defbasisVV = -1;
    UUcallback = UUcallbackdef;
    VVcallback = VVcallbackdef;

    (wildxgenta) = nullptr;
    (wildxgentb) = nullptr;
    (wildxgentc) = nullptr;
    (wildxgentd) = nullptr;
    (wildxxgent) = nullptr;

    (wildxdim) = 0;

    (wildxdima) = 0;
    (wildxdimb) = 0;
    (wildxdimc) = 0;
    (wildxdimd) = 0;
    (wildxxdim) = 0;

    altxsrc = nullptr;

    MEMNEW(that,ML_Base *);
    NiceAssert(that);
    *that = this;

    isIndPrune      = _isIndPrune;
    xassumedconsist = 0;
    xconsist        = 1;

    xdzero = 0;

    locsigma = 1.0/DEFAULT_C;

    loclr  = DEFAULT_LR;
    loclrb = DEFAULT_LR;
    loclrc = DEFAULT_LR;
    loclrd = DEFAULT_LR;

    globalzerotol = DEFAULT_ZTOL;

    xmuprior    = 0;
    xmuprior_gt = 0_gent;
    xmuprior_ml = nullptr;

    return;
}

ML_Base::~ML_Base()
{
    {
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif

        xvernumber("&",MLid()) = 0;
        gvernumber("&",MLid()) = 0;

        xvernumber.zero(MLid());
        gvernumber.zero(MLid());

//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
    }

    MEMDEL(that);

    return;
}

int ML_Base::xvernum(void) const
{
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif
        int res = xvernumber.isindpresent(MLid()) ? xvernumber(MLid()) : 0;
//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
        return res;
}

int ML_Base::gvernum(void) const
{
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif
        int res = gvernumber.isindpresent(MLid()) ? gvernumber(MLid()) : 0;
//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
        return res;
}

int ML_Base::xvernum(int altMLid) const
{
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif
        int res = xvernumber.isindpresent(altMLid) ? xvernumber(altMLid) : 0;
//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
        return res;
}

int ML_Base::gvernum(int altMLid) const
{
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif
        int res = gvernumber.isindpresent(altMLid) ? gvernumber(altMLid) : 0;
//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
        return res;
}

int ML_Base::incxvernum(void)
{
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif

        if ( !xvernumber.isindpresent(MLid()) )
        {
            xvernumber("&",MLid()) = 0;
        }

        int res = ++xvernumber("&",MLid());
//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
        return res;
}

int ML_Base::incgvernum(void)
{
//#ifdef ENABLE_THREADS
//        mleyelock.lock();
//#endif
        if ( !gvernumber.isindpresent(MLid()) )
        {
            gvernumber("&",MLid()) = 0;
        }

        int res = ++gvernumber("&",MLid());
//#ifdef ENABLE_THREADS
//        mleyelock.unlock();
//#endif
        return res;
}

const SparseVector<gentype> &ML_Base::xsum(SparseVector<gentype> &res) const
{
    res.zero();

    if ( N() )
    {
        // Use x so function polymorphs correctly

        res.nearassign(x(0));

        //if ( N() > 1 )
        {
            for ( int i = 1 ; i < N() ; ++i )
            {
                res.nearadd(x(i));
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xsqsum(SparseVector<gentype> &res) const
{
    res.zero();

    if ( N() )
    {
        // Use x so function polymorphs correctly

        res.nearassign(x(0));

        int i,j,k;

        for ( i = 0 ; i < N() ; ++i )
        {
            const SparseVector<gentype> &xx = x(i);

            if ( xx.nindsize() )
            {
                for ( j = 0 ; j < xx.nindsize() ; ++j )
                {
                    k = xx.ind(j);

                    if ( xx(k).isValNull()    ||
                         xx(k).isValInteger() ||
                         xx(k).isValReal()    ||
                         xx(k).isValAnion()   ||
                         xx(k).isValVector()  ||
                         xx(k).isValMatrix()     )
                    {
                        if ( i )
                        {
                            res("&",k) += outerProd(xx(k),xx(k));
                        }

                        else
                        {
                            res("&",k) = outerProd(xx(k),xx(k));
                        }
                    }

                    else
                    {
                        res("&",k) = "\"wtf\"";
                    }
                }
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmean(SparseVector<gentype> &res) const
{
    // NB: these are designed with data normalisation in mind!!!  Hence
    // setting things to zero (or one) when the sum doesn't make sense,
    // and saying the inverse of 0 is 1 for the invstddev

    xsum(res);

    if ( xspaceDim() )
    {
        int i;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; ++i )
        {
            if ( res(indKey()(i)).isValNull()    ||
                 res(indKey()(i)).isValInteger() ||
                 res(indKey()(i)).isValReal()    ||
                 res(indKey()(i)).isValAnion()   ||
                 res(indKey()(i)).isValVector()  ||
                 res(indKey()(i)).isValMatrix()     )
            {
                res("&",indKey()(i)) *= (1.0/(indkeyscale*indKeyCount()(i)));
            }

            else
            {
                res("&",indKey()(i)) = 0;
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xsqmean(SparseVector<gentype> &res) const
{
    xsqsum(res);

    if ( xspaceDim() )
    {
        int i;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; ++i )
        {
            if ( res(indKey()(i)).isValNull()    ||
                 res(indKey()(i)).isValInteger() ||
                 res(indKey()(i)).isValReal()    ||
                 res(indKey()(i)).isValAnion()   ||
                 res(indKey()(i)).isValVector()  ||
                 res(indKey()(i)).isValMatrix()     )
            {
                res("&",indKey()(i)) *= (1.0/(indkeyscale*indKeyCount()(i)));
            }

            else
            {
                res("&",indKey()(i)) = 0;
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmeansq(SparseVector<gentype> &res) const
{
    xmean(res);

    //if ( res.nindsize() )
    {
        for ( int i = 0 ; i < res.nindsize() ; ++i )
        {
            if ( res.direcref(i).isValNull()    ||
                 res.direcref(i).isValInteger() ||
                 res.direcref(i).isValReal()    ||
                 res.direcref(i).isValAnion()   ||
                 res.direcref(i).isValVector()  ||
                 res.direcref(i).isValMatrix()     )
            {
                res.direref(i) = outerProd(res.direcref(i),res.direcref(i));
            }

            else
            {
                res.direref(i) = 0;
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmedian(SparseVector<gentype> &res) const
{
    res.zero();

    if ( xspaceDim() && N() )
    {
        int i,j,k;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; ++i )
        {
            Vector<gentype> featvec(indkeyscale*indKeyCount()(i));

            k = 0;

            for ( j = 0 ; j < N() ; ++j )
            {
                const SparseVector<gentype> &xx = x(j);

                if ( xx.isindpresent(indKey()(i)) )
                {
                    featvec("&",k) = xx(indKey()(i));
                    ++k;
                }
            }

            gentype featveccomp(featvec);

            res("&",indKey()(i)) = median(featveccomp);
        }
    }

    //if ( res.nindsize() )
    {
        for ( int i = 0 ; i < res.nindsize() ; ++i )
        {
            if ( !( res.direcref(i).isValNull()    ||
                    res.direcref(i).isValInteger() ||
                    res.direcref(i).isValReal()    ||
                    res.direcref(i).isValAnion()   ||
                    res.direcref(i).isValVector()  ||
                    res.direcref(i).isValMatrix()     ) )
            {
                res.direref(i) = 0;
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xvar(SparseVector<gentype> &res) const
{
    // var(x) = 1/N sum_i (x_i-m)^2
    //          1/N sum_i x_i^2 + 1/N sum_i m^2 - 2/N sum_i x_i 1/N sum_j x_j
    //          1/N sum_i x_i^2 + m^2 - 2 1/N sum_i x_i 1/N sum_j x_j
    //          1/N sum_i x_i^2 + m^2 - 2 m^2
    //          1/N sum_i x_i^2 - m^2 
    //          xsqmean - xmeansq

    SparseVector<gentype> xmeansqval;

    xsqmean(res);
    xmeansq(xmeansqval);

    xmeansqval.negate();

    res += xmeansqval;

    return res;
}

const SparseVector<gentype> &ML_Base::xstddev(SparseVector<gentype> &res) const
{
    xvar(res);

    //if ( res.nindsize() )
    {
        for ( int i = 0 ; i < res.nindsize() ; ++i )
        {
            if ( res.direcref(i).isValNull()    ||
                 res.direcref(i).isValInteger() ||
                 res.direcref(i).isValReal()    ||
                 res.direcref(i).isValAnion()       )
            {
                if ( ( (double) norm2(res.direcref(i)) ) >= zerotol() )
                {
                    res.direref(i) = sqrt(res.direcref(i));
                }

                else
                {
                    res.direref(i) = 1;
                }
            }

            else if ( res.direcref(i).isValMatrix() )
            {
                if ( ( (double) det(res.direcref(i)) ) >= zerotol() )
                {
                    Matrix<gentype> temp((const Matrix<gentype> &) res.direcref(i));

                    ((const Matrix<gentype> &) res.direref(i)).naiveChol(temp,1);
                    res.direref(i) = temp;
                }

                else
                {
                    res.direref(i) = 1;
                }
            }

            else
            {
                res.direref(i) = "\"wtf\"";
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmax(SparseVector<gentype> &res) const
{
    res.zero();

    if ( xspaceDim() && N() )
    {
        int i,j,k;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; ++i )
        {
            Vector<gentype> featvec(indkeyscale*indKeyCount()(i));

            k = 0;

            for ( j = 0 ; j < N() ; ++j )
            {
                SparseVector<gentype> xx = x(j);

                if ( xx.isindpresent(indKey()(i)) )
                {
                    featvec("&",k) = xx(indKey()(i));
                    ++k;
                }
            }

            gentype featveccomp(featvec);

            res("&",indKey()(i)) = max(featveccomp);
        }
    }

    //if ( res.nindsize() )
    {
        for ( int i = 0 ; i < res.nindsize() ; ++i )
        {
            if ( !( res.direcref(i).isValNull()    ||
                    res.direcref(i).isValInteger() ||
                    res.direcref(i).isValReal()    ||
                    res.direcref(i).isValAnion()   ||
                    res.direcref(i).isValVector()  ||
                    res.direcref(i).isValMatrix()     ) )
            {
                res.direref(i) = 0;
            }
        }
    }

    return res;
}


const SparseVector<gentype> &ML_Base::xmin(SparseVector<gentype> &res) const
{
    res.zero();

    if ( xspaceDim() && N() )
    {
        int i,j,k;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; ++i )
        {
            Vector<gentype> featvec(indkeyscale*indKeyCount()(i));

            k = 0;

            for ( j = 0 ; j < N() ; ++j )
            {
                const SparseVector<gentype> &xx = x(j);

                if ( xx.isindpresent(indKey()(i)) )
                {
                    featvec("&",k) = xx(indKey()(i));
                    ++k;
                }
            }

            gentype featveccomp(featvec);

            res("&",indKey()(i)) = min(featveccomp);
        }
    }

    //if ( res.nindsize() )
    {
        for ( int i = 0 ; i < res.nindsize() ; ++i )
        {
            if ( !( res.direcref(i).isValNull()    ||
                    res.direcref(i).isValInteger() ||
                    res.direcref(i).isValReal()    ||
                    res.direcref(i).isValAnion()   ||
                    res.direcref(i).isValVector()  ||
                    res.direcref(i).isValMatrix()     ) )
            {
                res.direref(i) = 0;
            }
        }
    }

    return res;
}

int ML_Base::normKernelNone(void)
{
    int res = 0;

    if ( N() )
    {
        res = 1;

        getKernel_unsafe().setUnShiftedScaled();
        resetKernel(1);
    }

    return res;
}

int ML_Base::normKernelZeroMeanUnitVariance(int flatnorm, int noshift)
{
    // Normalise to zero mean, unit variance

    int res = 0;

    if ( N() )  
    {
        res = 1;

        SparseVector<gentype> xmeanis;
        SparseVector<gentype> xstddevis;

        // Calculate mean and variance

        xmean(xmeanis);
        xstddev(xstddevis);

        errstream() << ".";

        // Calculate shift and scale from mean and variance
        // Also sanitise by removing non-numeric entries

        SparseVector<gentype> xshift;
        SparseVector<gentype> xscale;

        xshift = xmeanis;
        xshift.negate();
        xscale = xstddevis;

        if ( flatnorm )
        {
            gentype totalscale;
            int iii;

            totalscale = min(xscale,iii);

            xscale = totalscale;
        }

        if ( noshift )
        {
            xshift.zero();
        }

        // Remove errors from shift and scale (probably related to
        // non-numeric features).

        errstream() << "Shift: " << xshift << "\n";
        errstream() << "Scale: " << xscale << "\n";

        getKernel_unsafe().setShift(xshift);
        getKernel_unsafe().setScale(xscale);
        resetKernel(1);
    }

    return res;
}

int ML_Base::normKernelZeroMedianUnitVariance(int flatnorm, int noshift)
{
    // Normalise to zero median, unit variance

    int res = 0;

    if ( N() )  
    {
        res = 1;

        SparseVector<gentype> xmedianis;
        SparseVector<gentype> xstddevis;

        // Calculate median and variance

        xmedian(xmedianis);
        xstddev(xstddevis);

        errstream() << ".";

        // Calculate shift and scale from median and variance
        // Also sanitise by removing non-numeric entries

        SparseVector<gentype> xshift;
        SparseVector<gentype> xscale;

        xshift = xmedianis;
        xshift.negate();
        xscale = xstddevis;

        if ( flatnorm )
        {
            gentype totalscale;
            int iii;

            totalscale = min(xscale,iii);

            xscale = totalscale;
        }

        if ( noshift )
        {
            xshift.zero();
        }

        // Remove errors from shift and scale (probably related to
        // non-numeric features).

        errstream() << "Shift: " << xshift << "\n";
        errstream() << "Scale: " << xscale << "\n";

        getKernel_unsafe().setShift(xshift);
        getKernel_unsafe().setScale(xscale);
        resetKernel(1);
    }

    return res;
}


int ML_Base::normKernelUnitRange(int flatnorm, int noshift)
{
    // Normalise range to 0-1

    int res = 0;

    if ( N() )  
    {
        res = 1;

        SparseVector<gentype> xminis;
        SparseVector<gentype> xmaxis;

        xmin(xminis);
        xmax(xmaxis);

        // Calculate shift and scale

        SparseVector<gentype> xshift(xminis);
        SparseVector<gentype> xscale(xmaxis);

        xshift.negate();
        xscale -= xminis;

        //if ( xscale.nindsize() )
        {
            for ( int j = 0 ; j < xscale.nindsize() ; ++j )
            {
                if ( abs2((double) xscale.direcref(j)) > 0 )
                {
                    ;
                }

                else
                {
                    xscale.direref(j) = 1;
                }
            }
        }

        if ( flatnorm )
        {
            gentype totalscale;
            int iii;

            totalscale = min(xscale,iii);

            xscale = totalscale;
        }

        if ( noshift )
        {
            xshift.zero();
        }

        // Shift and scale

        errstream() << "Shift: " << xshift << "\n";
        errstream() << "Scale: " << xscale << "\n";

        getKernel_unsafe().setShift(xshift);
        getKernel_unsafe().setScale(xscale);
        resetKernel(1);
    }

    return res;
}

SparseVector<gentype> &ML_Base::xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype> &src) const
{
        NiceAssert( src.size() == xspaceDim() );

        dest.zero();

        //if ( xspaceDim() )
        {
            for ( int i = 0 ; i < xspaceDim() ; ++i )
            {
                dest("&",indKey()(i)) = src(i);
            }
        }

        return dest;
}

SparseVector<gentype> &ML_Base::xlateToSparse(SparseVector<gentype> &dest, const Vector<double> &src) const
{
        NiceAssert( src.size() == xspaceDim() );

        dest.zero();

        //if ( xspaceDim() )
        {
            for ( int i = 0 ; i < xspaceDim() ; ++i )
            {
                dest("&",indKey()(i)) = src(i);
            }
        }

        return dest;
}

SparseVector<gentype> &ML_Base::xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const
{
    dest.indalign(src);

    NiceAssert( dest.nindsize() == src.nindsize() );

    //if ( dest.nindsize() )
    {
        for ( int i = 0 ; i < dest.nindsize() ; ++i )
        {
            dest.direref(i) = src.direcref(i);
        }
    }

    return dest;
}

Vector<gentype> &ML_Base::xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const
{
        dest.resize(xspaceDim());
        dest.zero();

        //if ( src.nindsize() )
        {
            int ikInd = 0;

            for ( int i = 0 ; i < src.nindsize() ; ++i )
            {
                int oob = 1;

                if ( ikInd < xspaceDim() )
                {
                    oob = 0;

                    while ( indKey()(ikInd) < src.ind(i) )
                    {
                        ++ikInd;

                        if ( ikInd >= xspaceDim() )
                        {
                            oob = 1;
                            break;
                        }
                    }
                }

                (void) oob;
                // Design change: don't add unknown indices at all.  That
                // way if you are using real vectors to represent sparse
                // (for example in ml_serial) then then the inner product
                // calls will have aligned sizes, and hence won't throw
                // an exception.  This is OK, as the indices that are not
                // in the ML already will simply be multiplied by zero and
                // hence leaving them out should technically make no diff.

                //if ( oob )
                //{
                //    // General rule: put unknown indices at the end
                //
                //    int j;
                //
                //    for ( j = i ; j < src.nindsize() ; ++j )
                //    {
                //        dest.add(dest.size());
                //        dest("&",dest.size()-1) = src.direcref(j);
                //    }
                //
                //    return dest;
                //}
                //
                //if ( indKey()(ikInd) != src.ind(i) )
                //{
                //    dest.add(dest.size());
                //    dest("&",dest.size()-1) = src.direcref(i);
                //}
                //
                //else
                {
                    dest("&",ikInd) = src.direcref(i);
                }
            }
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const SparseVector<gentype> &src) const
{
        dest.resize(xspaceDim());
        dest.zero();

        //if ( src.nindsize() )
        {
            int ikInd = 0;

            for ( int i = 0 ; i < src.nindsize() ; ++i )
            {
                int oob = 1;

                if ( ikInd < xspaceDim() )
                {
                    oob = 0;

                    while ( indKey()(ikInd) < src.ind(i) )
                    {
                        ++ikInd;

                        if ( ikInd >= xspaceDim() )
                        {
                            oob = 1;
                            break;
                        }
                    }
                }

                (void) oob;
                // Design change: don't add unknown indices at all.  That
                // way if you are using real vectors to represent sparse
                // (for example in ml_serial) then then the inner product
                // calls will have aligned sizes, and hence won't throw
                // an exception.  This is OK, as the indices that are not
                // in the ML already will simply be multiplied by zero and
                // hence leaving them out should technically make no diff.

                //if ( oob )
                //{
                //    // General rule: put unknown indices at the end
                //
                //    int j;
                //
                //    for ( j = i ; j < src.nindsize() ; ++j )
                //    {
                //        dest.add(dest.size());
                //        dest("&",dest.size()-1) = (double) src.direcref(j);
                //    }
                //
                //    return dest;
                //}
                //
                //if ( indKey()(ikInd) != src.ind(i) )
                //{
                //    dest.add(dest.size());
                //    dest("&",dest.size()-1) = (double) src.direcref(i);
                //}
                //
                //else
                {
                    dest("&",ikInd) = (double) src.direcref(i);
                }
            }
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const SparseVector<double> &src) const
{
        dest.resize(xspaceDim());
        dest.zero();

        //if ( src.nindsize() )
        {
            int ikInd = 0;

            for ( int i = 0 ; i < src.nindsize() ; ++i )
            {
                int oob = 1;

                if ( ikInd < xspaceDim() )
                {
                    oob = 0;

                    while ( indKey()(ikInd) < src.ind(i) )
                    {
                        ++ikInd;

                        if ( ikInd >= xspaceDim() )
                        {
                            oob = 1;
                            break;
                        }
                    }
                }

                (void) oob;
                // Design change: don't add unknown indices at all.  That
                // way if you are using real vectors to represent sparse
                // (for example in ml_serial) then then the inner product
                // calls will have aligned sizes, and hence won't throw
                // an exception.  This is OK, as the indices that are not
                // in the ML already will simply be multiplied by zero and
                // hence leaving them out should technically make no diff.

                //if ( oob )
                //{
                //    // General rule: put unknown indices at the end
                //
                //    int j;
                //
                //    for ( j = i ; j < src.nindsize() ; ++j )
                //    {
                //        dest.add(dest.size());
                //        dest("&",dest.size()-1) = (double) src.direcref(j);
                //    }
                //
                //    return dest;
                //}
                //
                //if ( indKey()(ikInd) != src.ind(i) )
                //{
                //    dest.add(dest.size());
                //    dest("&",dest.size()-1) = (double) src.direcref(i);
                //}
                //
                //else
                {
                    dest("&",ikInd) = (double) src.direcref(i);
                }
            }
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const Vector<gentype> &src) const
{
        dest.resize(src.size());

        for ( int i = 0 ; i < src.size() ; ++i )
        {
            dest("&",i) = src(i);
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const Vector<double> &src) const
{
        dest = src;

        return dest;
}

SparseVector<gentype> &ML_Base::makeFullSparse(SparseVector<gentype> &dest) const
{
        dest.zero();

        //if ( xspaceDim() )
        {
            for ( int i = 0 ; i < xspaceDim() ; ++i )
            {
                dest("&",indKey()(i));  // No need to actually give a value, we just want the index to exist
            }
        }

        return dest;
}


















int ML_Base::setMLid(int nv)
{
//#ifdef ENABLE_THREADS
//    mleyelock.lock();
//#endif

    int oldMLid = MLid();
    int res = kernPrecursor::setMLid(nv);

    NiceAssert( !res );

    //if ( !res )
    {
        NiceAssert( !xvernumber.isindpresent(nv) || ( xvernumber(nv) == 0 ) );
        NiceAssert( !gvernumber.isindpresent(nv) || ( gvernumber(nv) == 0 ) );

        int xvn = xvernumber(oldMLid);
        int gvn = gvernumber(oldMLid);

        xvernumber("&",oldMLid) = 0;
        gvernumber("&",oldMLid) = 0;

        xvernumber.zero(oldMLid); // Indicate no longer present
        gvernumber.zero(oldMLid); // Indicate no longer present

        xvernumber("&",nv) = xvn;
        gvernumber("&",nv) = gvn;
    }

//#ifdef ENABLE_THREADS
//    mleyelock.unlock();
//#endif

    return res;
}

































// private functions that are only called locally

void ML_Base::unfillIndex(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < indexKey(0).size() );

    if ( !altxsrc )
    {
        if ( indexKeyCount(0)(i) )
        {
            int j,k,m,n;

            for ( j = 0 ; ( j < ML_Base::N() ) && indexKeyCount(0)(i) ; ++j )
            {
                SparseVector<gentype> &xx = allxdatagent("&",j);

                m = xx.nupsize();

                for ( n = 0 ; n < m ; ++n )
                {
                    if ( xx.isnindpresent(indexKey(0)(i),n) )
                    {
                        k = gettypeind(xx.n(indexKey(0)(i),n));

                        xx.zero(indexKey(0)(i),n);
                        --(indexKeyCount("&",0)("&",i));
                        --(typeKeyBreak("&",0)("&",i)("&",k));
                    }
                }

                m = xx.f1upsize();

                for ( n = 0 ; n < m ; ++n )
                {
                    if ( xx.isf1indpresent(indexKey(0)(i),n) )
                    {
                        k = gettypeind(xx.f1(indexKey(0)(i),n));

                        xx.zerof1i(indexKey(0)(i),n);
                        --(indexKeyCount("&",0)("&",i));
                        --(typeKeyBreak("&",0)("&",i)("&",k));
                    }
                }
            }
        }

        typeKey("&",0)("&",i) = 1;
    }

    calcSetAssumeReal();

    return;
}

void ML_Base::fillIndex(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < indexKey(0).size() );

    if ( !altxsrc )
    {
        //if ( indexKeyCount(i) < ML_Base::N() ) - can exceed N if multivectors and near/far are present!
        {
            int j,m,n,k = 1;

            //for ( j = 0 ; ( j < ML_Base::N() ) && ( indexKeyCount(i) < ML_Base::N() ) ; ++j )
            for ( j = 0 ; j < ML_Base::N() ; ++j )
            {
                SparseVector<gentype> &xx = allxdatagent("&",j);

                m = xx.nupsize();

                for ( n = 0 ; n < m ; ++n )
                {
                    if ( !(xx.isnindpresent(indexKey(0)(i),n)) )
                    {
                        //k = 1;

                        (xx("&",indexKey(0)(i),n)).makeNull();
                        ++(indexKeyCount("&",0)("&",i));
                        ++(typeKeyBreak("&",0)("&",i)("&",k));
                    }
                }

                m = xx.f1upsize();

                for ( n = 0 ; n < m ; ++n )
                {
                    if ( !(xx.isf1indpresent(indexKey(0)(i),n)) )
                    {
                        //k = 1;

                        (xx.f1("&",indexKey(0)(i),n)).makeNull();
                        ++(indexKeyCount("&",0)("&",i));
                        ++(typeKeyBreak("&",0)("&",i)("&",k));
                    }
                }
            }
        }
    }

    calcSetAssumeReal();

    return;
}

void ML_Base::addToIndexKeyAndUpdate(const SparseVector<gentype> &newz, int u)
{
    int indKeyChanged = 0;

    if ( ( newz.nupsize() > 1 ) || newz.isf1offindpresent() || newz.isf2offindpresent() || newz.isf4offindpresent() )
    {
        NiceAssert( u == -1 );

        int s;

        s = newz.nupsize();

        for ( u = 0 ; u < s ; ++u )
        {
            addToIndexKeyAndUpdate(newz.nup(u),-1);
            addToIndexKeyAndUpdate(newz.nup(u),u);
        }

        s = newz.f1upsize();

        for ( u = 0 ; u < s ; ++u )
        {
            addToIndexKeyAndUpdate(newz.f1up(u),-1);
            addToIndexKeyAndUpdate(newz.f1up(u),u);
        }
    }

    else
    {
        if ( !(indexKey.isindpresent(u+1)) )
        {
            Vector<int> dummy;
            Vector<Vector<int> > dummyb;

            indexKey("&",u+1)      = dummy;
            indexKeyCount("&",u+1) = dummy;
            typeKey("&",u+1)       = dummy;
            typeKeyBreak("&",u+1)  = dummyb;
        }

        if ( newz.nindsize() )
        {
            int zInd;
            int ikInd = indexKey(u+1).size()-1;
            int needaddxspacedim;

            for ( zInd = newz.nindsize()-1 ; zInd >= 0 ; --zInd )
            {
                needaddxspacedim = 0;

                // Find index zInd in new vector
                // Set ikInd such that indexKey(ikInd) == newz.ind(zInd).
                // If ikInd == -1 or indexKey(ikInd) != newz.ind(zInd) then
                // we need to add the new index.

                if ( ikInd >= 0 )
                {
                    while ( indexKey(u+1)(ikInd) > newz.ind(zInd) )
                    {
                        --ikInd;

                        if ( ikInd < 0 )
                        {
                            break;
                        }
                    }
                }

                if ( ikInd == -1 )
                {
                    addInd:

                    // Index not found, so add it.

                    ++ikInd;

                    indexKey("&",u+1).add(ikInd);
                    indexKey("&",u+1)("&",ikInd) = newz.ind(zInd);
                    indexKeyCount("&",u+1).add(ikInd);
                    indexKeyCount("&",u+1)("&",ikInd) = 0;
                    typeKey("&",u+1).add(ikInd);
                    typeKey("&",u+1)("&",ikInd) = 1;
                    typeKeyBreak("&",u+1).add(ikInd);
                    typeKeyBreak("&",u+1)("&",ikInd).resize(NUMXTYPES);
                    typeKeyBreak("&",u+1)("&",ikInd) = 0;

                    needaddxspacedim = 1;
                    indKeyChanged = 1;
                }

                else if ( indexKey(u+1)(ikInd) != newz.ind(zInd) )
                {
                    goto addInd;
                }

                NiceAssert( indexKey(u+1)(ikInd) == newz.ind(zInd) );

                // Update index information

                ++(indexKeyCount("&",u+1)("&",ikInd));

                int indType = gettypeind(newz.direcref(zInd));

                NiceAssert( indType );

                ++(typeKeyBreak("&",u+1)("&",ikInd)("&",indType));
                ++(typeKeyBreak("&",u+1)("&",ikInd)("&",0));

                int indchange = 1;

                if ( indType == typeKey(u+1)(ikInd) )
                {
                    indchange = 0;
                }

                else if ( ( indType <= typeKey(u+1)(ikInd) ) && ( typeKey(u+1)(ikInd) <= 5 ) )
                {
                    indchange = 0;
                    indType = typeKey(u+1)(ikInd);
                }

                if ( indchange )
                {
                    int runsum = typeKeyBreak(u+1)(ikInd)(1);

                    if ( typeKeyBreak(u+1)(ikInd)(0) == runsum )
                    {
                        // null
                        indType = 1;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(2) ) )
                    {
                        // null or binary
                        indType = 2;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(3) ) )
                    {
                        // null or binary or integer
                        indType = 3;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(4) ) )
                    {
                        // null or binary or integer or double
                        indType = 4;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(5) ) )
                    {
                        // null or binary or integer or double or anion
                        indType = 5;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(6) ) )
                    {
                        // null or vector
                        indType = 6;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(7) ) )
                    {
                        // null or matrix
                        indType = 7;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(8) ) )
                    {
                        // null or set
                        indType = 8;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(9) ) )
                    {
                        // null or dgraph
                        indType = 9;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(10) ) )
                    {
                        // null or string
                        indType = 10;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(11) ) )
                    {
                        // null or string
                        indType = 11;
                    }

                    else
                    {
                        // unknown
                        indType = 0;
                    }
                }

                typeKey("&",u+1)("&",ikInd) = indType;

                if ( indPrune() && ( indexKeyCount(u+1)(ikInd) < ML_Base::N() ) )
                {
                    // Make sure index is in all vectors, as required when
                    // pruning is set.

                    fillIndex(ikInd);
                }

                if ( needaddxspacedim && ( u == -1 ) )
                {
                    addxspaceFeat(ikInd);
                }
            }
        }

        // Need this to make sure the newly added training vector has
        // all relevant indices

        if ( indexKey(u+1).size() && indPrune() )
        {
            for ( int i = 0 ; i < indexKey(u+1).size() ; ++i )
            {
                if ( indPrune() && ( indexKeyCount(u+1)(i) < ML_Base::N() ) )
                {
                    fillIndex(i);
                }
            }
        }
    }

    if ( indKeyChanged )
    {
        calcSetXdim();
    }

    calcSetAssumeReal();

    return;
}

void ML_Base::removeFromIndexKeyAndUpdate(const SparseVector<gentype> &oldz, int u)
{
    int indKeyChanged = 0;

    if ( ( oldz.nupsize() > 1 ) || ( oldz.f1upsize() > 1 ) || oldz.isf1offindpresent() || oldz.isf2offindpresent() || oldz.isf4offindpresent() )
    {
        NiceAssert( u == -1 );

        int s;

        s = oldz.nupsize();

        for ( u = 0 ; u < s ; ++u )
        {
            removeFromIndexKeyAndUpdate(oldz.nup(u),-1);
            removeFromIndexKeyAndUpdate(oldz.nup(u),u);
        }

        s = oldz.f1upsize();

        for ( u = 0 ; u < s ; ++u )
        {
            removeFromIndexKeyAndUpdate(oldz.f1up(u),-1);
            removeFromIndexKeyAndUpdate(oldz.f1up(u),u);
        }
    }

    else
    {
        if ( oldz.nindsize() )
        {
            int zInd;
            int ikInd = indexKey(u+1).size()-1;

            for ( zInd = oldz.nindsize()-1 ; zInd >= 0 ; --zInd )
            {
                while ( indexKey(u+1)(ikInd) > oldz.ind(zInd) )
                {
                    --ikInd;
                    NiceAssert( ikInd >= 0 );
                }

                int indType = gettypeind(oldz.direcref(zInd));

                NiceAssert( indType );

                typeKeyBreak("&",u+1)("&",ikInd)("&",indType)--;
                typeKeyBreak("&",u+1)("&",ikInd)("&",0)--;

                int indchange = 0;

                if ( typeKeyBreak(u+1)(ikInd)(indType) != typeKeyBreak(u+1)(ikInd)(0) )
                {
                    indchange = 1;
                }

                if ( indchange )
                {
                    int runsum = typeKeyBreak(u+1)(ikInd)(1);

                    if ( typeKeyBreak(u+1)(ikInd)(0) == runsum )
                    {
                        // null
                        indType = 1;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(2) ) )
                    {
                        // null or binary
                        indType = 2;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(3) ) )
                    {
                        // null or binary or integer
                        indType = 3;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(4) ) )
                    {
                        // null or binary or integer or double
                        indType = 4;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( runsum += typeKeyBreak(u+1)(ikInd)(5) ) )
                    {
                        // null or binary or integer or double or anion
                        indType = 5;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(6) ) )
                    {
                        // null or vector
                        indType = 6;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(7) ) )
                    {
                        // null or matrix
                        indType = 7;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(8) ) )
                    {
                        // null or set
                        indType = 8;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(9) ) )
                    {
                        // null or dgraph
                        indType = 9;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(10) ) )
                    {
                        // null or string
                        indType = 10;
                    }

                    else if ( typeKeyBreak(u+1)(ikInd)(0) == ( typeKeyBreak(u+1)(ikInd)(0) + typeKeyBreak(u+1)(ikInd)(11) ) )
                    {
                        // null or string
                        indType = 11;
                    }

                    else
                    {
                        // unknown
                        indType = 0;
                    }
                }

                typeKey("&",u+1)("&",ikInd) = indType;

                NiceAssert( indexKey(u+1)(ikInd) == oldz.ind(zInd) );

                --(indexKeyCount("&",u+1)("&",ikInd));

                if ( indPrune() && ( typeKey(u+1)(ikInd) == 1 ) )
                {
                    // Remove null feature as required.  Next if statement
                    // will finish operation

                    unfillIndex(ikInd);
                }

                if ( !indexKeyCount(u+1)(ikInd) )
                {
                    // Remove unused feature from indexing

                    removexspaceFeat(ikInd);
                    indexKey("&",u+1).remove(ikInd);
                    indexKeyCount("&",u+1).remove(ikInd);

                    if ( ikInd > indexKey(u+1).size()-1 )
                    {
                        ikInd = indexKey(u+1).size()-1;
                    }

                    indKeyChanged = 1;
                }
            }
        }
    }

    if ( indKeyChanged )
    {
        calcSetXdim();
    }

    calcSetAssumeReal();

    return;
}

int ML_Base::gettypeind(const gentype &y) const
{
        int indType = 0; // unknown type

             if ( y.isValNull()    ) { indType = 1;  }
        else if ( y.isValInteger() ) { indType = 3;  }
        else if ( y.isValReal()    ) { indType = 4;  }
        else if ( y.isValAnion()   ) { indType = 5;  }
        else if ( y.isValVector()  ) { indType = 6;  }
        else if ( y.isValMatrix()  ) { indType = 7;  }
        else if ( y.isValSet()     ) { indType = 8;  }
        else if ( y.isValDgraph()  ) { indType = 9;  }
        else if ( y.isValString()  ) { indType = 10; }
        else if ( y.isValEqnDir()  ) { indType = 11; }

        if ( y.isValInteger() && ( ( ( (int) y ) == 0 ) ||
                                   ( ( (int) y ) == 1 )    ) )
        {
            indType = 2;
        }

        return indType;
}

























































// Functions that *should* be polymorphed.

int ML_Base::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &xx, double nCweight, double nepsweight, int dval)
{
    if ( ML_Base::N()+1 > LARGE_TRAIN_BOUNDARY )
    {
        disableAltContent();
    }

    if ( i != ML_Base::N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        addToBasisUU(i,y);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        addToBasisVV(i,y);
        isBasisUserVV = 1;
    }

    xd.add(i);           xd("&",i)           = dval; //2;
    xCweight.add(i);     xCweight("&",i)     = nCweight;
    xCweightfuzz.add(i); xCweightfuzz("&",i) = 1.0;
    xepsweight.add(i);   xepsweight("&",i)   = nepsweight;

    SparseVector<gentype> xxqq(xx);

    allxdatagent.add(i);   qswap(allxdatagent("&",i),xxqq);

    alltraintarg.add(i);   alltraintarg("&",i)   = y;
    alltraintargR.add(i);  alltraintargR("&",i)  = (double) y;
    alltraintargA.add(i);  alltraintargA("&",i)  = (const d_anion &) y;
    alltraintargV.add(i);  alltraintargV("&",i)  = (const Vector<double> &) y;

    static thread_local Vector<double> empvec;

    alltraintargp.add(i);  calcprior(alltraintargp("&",i),allxdatagent(i));
    alltraintargpR.add(i); alltraintargpR("&",i) = ( gOutType() == 'R' ) ? ( (double) alltraintargp(i) ) : 0.0;
    alltraintargpA.add(i); alltraintargpA("&",i) = ( gOutType() == 'A' ) ? ( (const d_anion &) alltraintargp(i) ) : 0_gent;
    alltraintargpV.add(i); alltraintargpV("&",i) = ( gOutType() == 'V' ) ? ( (const Vector<double> &) alltraintargp(i) ) : empvec;

    if ( !(allxdatagent(i).altcontent) && !(allxdatagent(i).altcontentsp) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    if ( !isXAssumedConsistent() || !i )
    {
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }
    }

    // Note that we do this after we have done everything else.  This is
    // because x(i) may be redirected through any number of caches and
    // misdirections, which we need to have set up before we can proceed.
    // Note also the use of getKernel(), which is presumed to have been
    // appropriately polymorphed by inheriting class(es).

    traininfo.add(i);
    traintang.add(i);
    xalphaState.add(i);

    // If x() is not accurate don't worry as it will be done via callback
    getKernel().getvecInfo(traininfo("&",i),x(i),nullptr,isXConsistent(),assumeReal); //x()(i));
    traintang("&",i)   = detangle_x(i);
    xalphaState("&",i) = dval ? 1 : 0; //1;

//FIXME    allxdatagentp.add(i);
//FIXME    traininfop.add(i);

//FIXME    allxdatagentp("&",i) = &allxdatagent(i);
//FIXME    traininfop("&",i)    = &traininfo(i);

    return 0;
}

int ML_Base::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &xx, double nCweight, double nepsweight, int dval)
{
    if ( ML_Base::N()+1 > LARGE_TRAIN_BOUNDARY )
    {
        disableAltContent();
    }

    if ( i != ML_Base::N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        addToBasisUU(i,y);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        addToBasisVV(i,y);
        isBasisUserVV = 1;
    }

    xd.add(i);           xd("&",i)           = dval; //2;
    xCweight.add(i);     xCweight("&",i)     = nCweight;
    xCweightfuzz.add(i); xCweightfuzz("&",i) = 1.0;
    xepsweight.add(i);   xepsweight("&",i)   = nepsweight;

    allxdatagent.add(i);   qswap(allxdatagent("&",i),xx);

    alltraintarg.add(i);   alltraintarg("&",i)   = y;
    alltraintargR.add(i);  alltraintargR("&",i)  = (double) y;
    alltraintargA.add(i);  alltraintargA("&",i)  = (const d_anion &) y;
    alltraintargV.add(i);  alltraintargV("&",i)  = (const Vector<double> &) y;

    static thread_local Vector<double> empvec;

    alltraintargp.add(i);  calcprior(alltraintargp("&",i),allxdatagent(i));
    alltraintargpR.add(i); alltraintargpR("&",i) = ( gOutType() == 'R' ) ? ( (double) alltraintargp(i) ) : 0.0;
    alltraintargpA.add(i); alltraintargpA("&",i) = ( gOutType() == 'A' ) ? ( (const d_anion &) alltraintargp(i) ) : 0_gent;
    alltraintargpV.add(i); alltraintargpV("&",i) = ( gOutType() == 'V' ) ? ( (const Vector<double> &) alltraintargp(i) ) : empvec;

    if ( !(allxdatagent(i).altcontent) && !(allxdatagent(i).altcontentsp) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    if ( !isXAssumedConsistent() || !i )
    {
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }
    }

    // Note that we do this after we have done everything else.  This is
    // because x(i) may be redirected through any number of caches and
    // misdirections, which we need to have set up before we can proceed.
    // Note also the use of getKernel(), which is presumed to have been
    // appropriately polymorphed by inheriting class(es).

    traininfo.add(i);
    traintang.add(i);
    xalphaState.add(i);

    // If x() is not accurate don't worry as it will be done via callback
    getKernel().getvecInfo(traininfo("&",i),x(i),nullptr,isXConsistent(),assumeReal); //x()(i));
    traintang("&",i)   = detangle_x(i);
    xalphaState("&",i) = dval ? 1 : 0; //1;

//FIXME    allxdatagentp.add(i);
//FIXME    traininfop.add(i);

//FIXME    allxdatagentp("&",i) = &allxdatagent(i);
//FIXME    traininfop("&",i)    = &traininfo(i);

    return 0;
}

int ML_Base::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &xx, const Vector<double> &nCweigh, const Vector<double> &nepsweigh)
{
    int Nadd = y.size();

    if ( ML_Base::N()+Nadd > LARGE_TRAIN_BOUNDARY )
    {
        disableAltContent();
    }

    if ( i != ML_Base::N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );
    NiceAssert( Nadd == xx.size() );
    NiceAssert( Nadd == nCweigh.size() );
    NiceAssert( Nadd == nepsweigh.size() );

    for ( int j = 0 ; j < Nadd ; ++j )
    {
        if ( isBasisUserUU )
        {
            isBasisUserUU = 0;
            addToBasisUU(i,y(i+j));
            isBasisUserUU = 1;
        }

        if ( isBasisUserVV )
        {
            isBasisUserVV = 0;
            addToBasisVV(i,y(i+j));
            isBasisUserVV = 1;
        }
    }

    retVector<int> tmpvint;
    retVector<double> tmpvdbl;

    xd.addpad(i,Nadd);           xd("&",i,1,i+Nadd-1,tmpvint)           = 2;
    xCweight.addpad(i,Nadd);     xCweight("&",i,1,i+Nadd-1,tmpvdbl)     = nCweigh;
    xCweightfuzz.addpad(i,Nadd); xCweightfuzz("&",i,1,i+Nadd-1,tmpvdbl) = 1.0;
    xepsweight.addpad(i,Nadd);   xepsweight("&",i,1,i+Nadd-1,tmpvdbl)   = nepsweigh;

    for ( int j = 0 ; j < Nadd ; ++j )
    {
        allxdatagent.add(i+j);   allxdatagent("&",i+j)   = xx(j);

        alltraintarg.add(i+j);   alltraintarg("&",i+j)   = y(j);
        alltraintargR.add(i+j);  alltraintargR("&",i+j)  = (double) y(j);
        alltraintargA.add(i+j);  alltraintargA("&",i+j)  = (const d_anion &) y(j);
        alltraintargV.add(i+j);  alltraintargV("&",i+j)  = (const Vector<double> &) y(j);

        static thread_local Vector<double> empvec;

        alltraintargp.add(i+j);  calcprior(alltraintargp("&",i+j),allxdatagent(i+j));;
        alltraintargpR.add(i+j); alltraintargpR("&",i+j) = ( gOutType() == 'R' ) ? ( (double) alltraintargp(i+j) ) : 0.0;
        alltraintargpA.add(i+j); alltraintargpA("&",i+j) = ( gOutType() == 'A' ) ? ( (const d_anion &) alltraintargp(i+j) ) : 0_gent;
        alltraintargpV.add(i+j); alltraintargpV("&",i+j) = ( gOutType() == 'V' ) ? ( (const Vector<double> &) alltraintargp(i+j) ) : empvec;

        if ( !(allxdatagent(i+j).altcontent) && !(allxdatagent(i+j).altcontentsp) )
        {
            allxdatagent("&",i+j).makealtcontent();
        }

        if ( !isXAssumedConsistent() || !(i+j) )
        {
            addToIndexKeyAndUpdate(x(i+j));

            if ( xconsist || !(i+j) )
            {
                xconsist = testxconsist();
            }
        }

        // Note that we do this after we have done everything else.  This is
        // because x(i) may be redirected through any number of caches and
        // misdirections, which we need to have set up before we can proceed.
        // Note also the use of getKernel(), which is presumed to have been
        // appropriately polymorphed by inheriting class(es).

        traininfo.add(i+j);
        traintang.add(i+j);
        xalphaState.add(i+j);

        // If x() is not accurate don't worry as it will be done via callback
        getKernel().getvecInfo(traininfo("&",i+j),x(i+j),nullptr,isXConsistent(),assumeReal); //()(i+j));
        traintang("&",i+j)   = detangle_x(i+j);
        xalphaState("&",i+j) = 1;

//FIXME            allxdatagentp.add(i+j);
//FIXME            traininfop.add(i+j);

//FIXME            allxdatagentp("&",i+j) = &allxdatagent(i+j);
//FIXME            traininfop("&",i+j)    = &traininfo(i+j);
    }

    return 0;
}

int ML_Base::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &xx, const Vector<double> &nCweigh, const Vector<double> &nepsweigh)
{
    int Nadd = y.size();

    if ( ML_Base::N()+Nadd > LARGE_TRAIN_BOUNDARY )
    {
        disableAltContent();
    }

    if ( i != ML_Base::N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );
    NiceAssert( Nadd == xx.size() );
    NiceAssert( Nadd == nCweigh.size() );
    NiceAssert( Nadd == nepsweigh.size() );

    for ( int j = 0 ; j < Nadd ; ++j )
    {
        if ( isBasisUserUU )
        {
            isBasisUserUU = 0;
            addToBasisUU(i,y(i+j));
            isBasisUserUU = 1;
        }

        if ( isBasisUserVV )
        {
            isBasisUserVV = 0;
            addToBasisVV(i,y(i+j));
            isBasisUserVV = 1;
        }
    }

    retVector<int> tmpvint;
    retVector<double> tmpvdbl;

    xd.addpad(i,Nadd);           xd("&",i,1,i+Nadd-1,tmpvint)           = 2;
    xCweight.addpad(i,Nadd);     xCweight("&",i,1,i+Nadd-1,tmpvdbl)     = nCweigh;
    xCweightfuzz.addpad(i,Nadd); xCweightfuzz("&",i,1,i+Nadd-1,tmpvdbl) = 1.0;
    xepsweight.addpad(i,Nadd);   xepsweight("&",i,1,i+Nadd-1,tmpvdbl)   = nepsweigh;

    for ( int j = 0 ; j < Nadd ; ++j )
    {
        allxdatagent.add(i+j);   qswap(allxdatagent("&",i+j),xx("&",j));

        alltraintarg.add(i+j);   alltraintarg("&",i+j)   = y(j);
        alltraintargR.add(i+j);  alltraintargR("&",i+j)  = (double) y(j);
        alltraintargA.add(i+j);  alltraintargA("&",i+j)  = (const d_anion &) y(j);
        alltraintargV.add(i+j);  alltraintargV("&",i+j)  = (const Vector<double> &) y(j);

        static thread_local Vector<double> empvec;

        alltraintargp.add(i+j);  calcprior(alltraintargp("&",i+j),allxdatagent(i+j));;
        alltraintargpR.add(i+j); alltraintargpR("&",i+j) = ( gOutType() == 'R' ) ? ( (double) alltraintargp(i+j) ) : 0.0;
        alltraintargpA.add(i+j); alltraintargpA("&",i+j) = ( gOutType() == 'A' ) ? ( (const d_anion &) alltraintargp(i+j) ) : 0_gent;
        alltraintargpV.add(i+j); alltraintargpV("&",i+j) = ( gOutType() == 'V' ) ? ( (const Vector<double> &) alltraintargp(i+j) ) : empvec;

        if ( !(allxdatagent(i+j).altcontent) && !(allxdatagent(i+j).altcontentsp) )
        {
            allxdatagent("&",i+j).makealtcontent();
        }

        if ( !isXAssumedConsistent() || !(i+j) )
        {
            addToIndexKeyAndUpdate(x(i+j));

            if ( xconsist || !(i+j) )
            {
                xconsist = testxconsist();
            }
        }

        // Note that we do this after we have done everything else.  This is
        // because x(i) may be redirected through any number of caches and
        // misdirections, which we need to have set up before we can proceed.
        // Note also the use of getKernel(), which is presumed to have been
        // appropriately polymorphed by inheriting class(es).

        traininfo.add(i+j);
        traintang.add(i+j);
        xalphaState.add(i+j);

        // If x() is not accurate don't worry as it will be done via callback
        getKernel().getvecInfo(traininfo("&",i+j),x(i+j),nullptr,isXConsistent(),assumeReal); //()(i+j));
        xalphaState("&",i+j) = 1;
        traintang("&",i+j) = detangle_x(i+j);

//FIXME            allxdatagentp.add(i+j);
//FIXME            traininfop.add(i+j);

//FIXME            allxdatagentp("&",i+j) = &allxdatagent(i+j);
//FIXME            traininfop("&",i+j)    = &traininfo(i+j);
    }

    return 0;
}

int ML_Base::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i <  ML_Base::N() );

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        removeFromBasisUU(i);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        removeFromBasisVV(i);
        isBasisUserVV = 1;
    }

    xdzero -= xd(i) ? 0 : 1;

    if ( !isXAssumedConsistent() || !i )
    {
        removeFromIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }
    }

    qswap(xx,allxdatagent("&",i));
    qswap(y,alltraintarg("&",i));
//    qswap(y,alltraintargR("&",i));
//    qswap(y,alltraintargA("&",i));
//    qswap(y,alltraintargV("&",i));
//    qswap(y,alltraintargp("&",i));
//    qswap(y,alltraintargpR("&",i));
//    qswap(y,alltraintargpA("&",i));
//    qswap(y,alltraintargpV("&",i));

    allxdatagent.remove(i);
    alltraintarg.remove(i);
    alltraintargR.remove(i);
    alltraintargA.remove(i);
    alltraintargV.remove(i);
    alltraintargp.remove(i);
    alltraintargpR.remove(i);
    alltraintargpA.remove(i);
    alltraintargpV.remove(i);

    traininfo.remove(i);
    traintang.remove(i);
    xalphaState.remove(i);

    xd.remove(i);
    xCweight.remove(i);
    xCweightfuzz.remove(i);
    xepsweight.remove(i);

//FIXME    allxdatagentp.remove(i);
//FIXME    traininfop.remove(i);

    return 0;
}

int ML_Base::removeTrainingVector(int i, int num)
{
    if ( !num )
    {
        return 0;
    }

    NiceAssert( i < ML_Base::N() );
    NiceAssert( num >= 0 );
    NiceAssert( num <= ML_Base::N()-i );

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






















int ML_Base::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < ML_Base::N() );

    calcSetAssumeReal();

    int res = 1;

    if ( ML_Base::N() && ( onlyChangeRowI == -1 ) && updateInfo )
    {
        res = 1;

        for ( int i = 0 ; i < ML_Base::N() ; ++i )
	{
            if ( modind )
	    {
                // If x() is not accurate don't worry as it will be done via callback
                getKernel().getvecInfo(traininfo("&",i),x(i),nullptr,isXConsistent(),assumeReal); //()(i));
                traintang("&",i) = detangle_x(i);
	    }
	}
    }

    else if ( ( onlyChangeRowI >= 0 ) && updateInfo )
    {
        res = 1;

        if ( modind )
        {
            // If x() is not accurate don't worry as it will be done via callback
            getKernel().getvecInfo(traininfo("&",onlyChangeRowI),x(onlyChangeRowI),nullptr,isXConsistent(),assumeReal); //()(onlyChangeRowI));
            traintang("&",onlyChangeRowI) = detangle_x(onlyChangeRowI);
        }
    }

    if ( onlyChangeRowI >= 0 )
    {
        int i = onlyChangeRowI;

        static thread_local Vector<double> empvec;

        calcprior(alltraintargp("&",i),allxdatagent(i));
        alltraintargpR("&",i) = ( gOutType() == 'R' ) ? ( (double) alltraintargp(i) ) : 0.0;
        alltraintargpA("&",i) = ( gOutType() == 'A' ) ? ( (const d_anion &) alltraintargp(i) ) : 0_gent;
        alltraintargpV("&",i) = ( gOutType() == 'V' ) ? ( (const Vector<double> &) alltraintargp(i) ) : empvec;
    }

    else
    {
        static thread_local Vector<double> empvec;

        for ( int i = 0 ; i < ML_Base::N() ; i++ )
        {
            calcprior(alltraintargp("&",i),allxdatagent(i));
            alltraintargpR("&",i) = ( gOutType() == 'R' ) ? ( (double) alltraintargp(i) ) : 0.0;
            alltraintargpA("&",i) = ( gOutType() == 'A' ) ? ( (const d_anion &) alltraintargp(i) ) : 0_gent;
            alltraintargpV("&",i) = ( gOutType() == 'V' ) ? ( (const Vector<double> &) alltraintargp(i) ) : empvec;
        }
    }

    return res;
}

int ML_Base::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    kernel = xkernel;

    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < ML_Base::N() );

    calcSetAssumeReal();

    int res = 1;

    if ( ML_Base::N() && ( onlyChangeRowI == -1 ) )
    {
        res = 1;

        for ( int i = 0 ; i < ML_Base::N() ; ++i )
	{
            if ( modind )
	    {
                // If x() is not accurate don't worry as it will be done via callback
                getKernel().getvecInfo(traininfo("&",i),x(i),nullptr,isXConsistent(),assumeReal); //()(i));
                traintang("&",i) = detangle_x(i);
	    }
	}
    }

    else if ( onlyChangeRowI >= 0 )
    {
        res = 1;

        // If x() is not accurate don't worry as it will be done via callback
        getKernel().getvecInfo(traininfo("&",onlyChangeRowI),x(onlyChangeRowI),nullptr,isXConsistent(),assumeReal); //()(onlyChangeRowI));
        traintang("&",onlyChangeRowI) = detangle_x(onlyChangeRowI);
    }

    return res;
}

int ML_Base::setx(int i, const SparseVector<gentype> &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i < ML_Base::N() );

    int res = 0;

    if ( !isXAssumedConsistent() || !i )
    {
        removeFromIndexKeyAndUpdate(x(i));
        allxdatagent("&",i) = xx;
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }

        res = 1;
    }

    else
    {
        allxdatagent("&",i) = xx;
    }

    if ( !(allxdatagent(i).altcontent) && !(allxdatagent(i).altcontentsp) && ( preallocsize() < LARGE_TRAIN_BOUNDARY ) && ( ML_Base::N() < LARGE_TRAIN_BOUNDARY ) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    getKernel_unsafe().setIPdiffered(1);

    return res |= resetKernel(1,i);
}

int ML_Base::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( i.size() == xx.size() );

    int res = 0;

    for ( int j = 0 ; j < i.size() ; ++j )
    {
        if ( !isXAssumedConsistent() || !i(j) )
        {
            removeFromIndexKeyAndUpdate(x(i(j)));
            allxdatagent("&",i(j)) = xx(j);
            addToIndexKeyAndUpdate(x(i(j)));

            if ( xconsist || !i(j) )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            allxdatagent("&",i(j)) = xx(j);
        }

        if ( !(allxdatagent(i(j)).altcontent) && !(allxdatagent(i(j)).altcontentsp) && ( preallocsize() < LARGE_TRAIN_BOUNDARY ) && ( ML_Base::N() < LARGE_TRAIN_BOUNDARY ) )
        {
            allxdatagent("&",i(j)).makealtcontent();
        }

        getKernel_unsafe().setIPdiffered(1);

        res |= resetKernel(1,i(j));
    }

    return res;
}

int ML_Base::setx(const Vector<SparseVector<gentype> > &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( xx.size() == ML_Base::N() );

    int res = 0;

    for ( int j = 0 ; j < ML_Base::N() ; ++j )
    {
        if ( !isXAssumedConsistent() || !j )
        {
            removeFromIndexKeyAndUpdate(x(j));
            allxdatagent("&",j) = xx(j);
            addToIndexKeyAndUpdate(x(j));

            if ( xconsist || !j )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            allxdatagent("&",j) = xx(j);
        }
    }

    getKernel_unsafe().setIPdiffered(1);

    return res |= resetKernel(1,-1);
}

int ML_Base::qswapx(int i, SparseVector<gentype> &xx, int dontupdate)
{
    incxvernum();
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i < ML_Base::N() );

    int res = 0;

    if ( !dontupdate && ( !isXAssumedConsistent() || !i ) )
    {
        removeFromIndexKeyAndUpdate(x(i));
        qswap(allxdatagent("&",i),xx);
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }

        res = 1;
    }

    else
    {
        qswap(allxdatagent("&",i),xx);
    }

    if ( !(allxdatagent(i).altcontent) && !(allxdatagent(i).altcontentsp) && ( preallocsize() < LARGE_TRAIN_BOUNDARY ) && ( ML_Base::N() < LARGE_TRAIN_BOUNDARY ) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    if ( !dontupdate )
    {
        getKernel_unsafe().setIPdiffered(1);
        res |= resetKernel(1,i);
    }

    return res;
}

int ML_Base::qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &xx, int dontupdate)
{
    incxvernum();
    incgvernum();

    NiceAssert( i.size() == xx.size() );

    int res = 0;

    for ( int j = 0 ; j < i.size() ; ++j )
    {
        if ( !dontupdate && ( !isXAssumedConsistent() || !i(j) ) )
        {
            removeFromIndexKeyAndUpdate(x(i(j)));
            qswap(allxdatagent("&",i(j)),xx("&",j));
            addToIndexKeyAndUpdate(x(i(j)));

            if ( xconsist || !i(j) )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            qswap(allxdatagent("&",i(j)),xx("&",j));
        }

        if ( !(allxdatagent(i(j)).altcontent) && !(allxdatagent(i(j)).altcontentsp) && ( preallocsize() < LARGE_TRAIN_BOUNDARY ) && ( ML_Base::N() < LARGE_TRAIN_BOUNDARY ) )
        {
            allxdatagent("&",i(j)).makealtcontent();
        }

        if ( !dontupdate )
        {
            getKernel_unsafe().setIPdiffered(1);
            res |= resetKernel(1,i(j));
        }
    }

    return res;
}

int ML_Base::qswapx(Vector<SparseVector<gentype> > &xx, int dontupdate)
{
    incxvernum();
    incgvernum();

    NiceAssert( xx.size() == ML_Base::N() );

    int res = 0;

    for ( int j = 0 ; j < ML_Base::N() ; ++j )
    {
        if ( !dontupdate && ( !isXAssumedConsistent() || !j ) )
        {
            removeFromIndexKeyAndUpdate(x(j));
            qswap(allxdatagent("&",j),xx("&",j));
            addToIndexKeyAndUpdate(x(j));

            if ( xconsist || !j )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            qswap(allxdatagent("&",j),xx("&",j));
        }

        if ( !(allxdatagent(j).altcontent) && !(allxdatagent(j).altcontentsp) && ( preallocsize() < LARGE_TRAIN_BOUNDARY ) && ( ML_Base::N() < LARGE_TRAIN_BOUNDARY ) )
        {
            allxdatagent("&",j).makealtcontent();
        }
    }

    if ( !dontupdate )
    {
        getKernel_unsafe().setIPdiffered(1);
        res |= resetKernel(1,-1);
    }

    return res;
}

int ML_Base::sety(int i, const gentype &y)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < ML_Base::N() );
    //NiceAssert( ( i < ML_Base::N() ) || altxsrc );
//if ( !( i < ML_Base::N() ) )
//{
//if ( altxsrc )
//{
//char buffer[2048];
//sprintf(buffer,"phantomxyz 0: (%p): %d,%d, %d, %p\n",this,i,ML_Base::N(),altxsrc->N(),altxsrc);
//errstream() << buffer;
//}
//else
//{
//char buffer[2048];
//sprintf(buffer,"phantomxyz 0: (%p): %d,%d, %d, %p\n",this,i,ML_Base::N(),N(),altxsrc);
//errstream() << buffer;
//}
//}

    if ( isBasisUserUU && ( i >= 0 ) )
    {
        isBasisUserUU = 0;
        setBasisUU(i,y);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV && ( i >= 0 ) )
    {
        isBasisUserVV = 0;
        setBasisVV(i,y);
        isBasisUserVV = 1;
    }

    if ( i >= 0 )
    {
        alltraintarg("&",i)   = y;
        alltraintargR("&",i)  = (double) y;
        alltraintargA("&",i)  = (const d_anion &) y;
        alltraintargV("&",i)  = (const Vector<double> &) y;
    }

    else
    {
        retVector<double> tmpva;

        alltraintarg   = y;
        alltraintargR  = (double) y;
        alltraintargA  = (const d_anion &) y;
        alltraintargV  = (const Vector<double> &) y;
    }

    return 0;
}

int ML_Base::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    if ( isBasisUserUU && yn.size() )
    {
        int i;

        isBasisUserUU = 0;

        for ( i = 0 ; i < yn.size() ; ++i )
        {
            setBasisUU(i,yn(j(i)));
        }

        isBasisUserUU = 1;
    }

    if ( isBasisUserVV && yn.size() )
    {
        int i;

        isBasisUserVV = 0;

        for ( i = 0 ; i < yn.size() ; ++i )
        {
            setBasisVV(i,yn(j(i)));
        }

        isBasisUserVV = 1;
    }

    retVector<gentype> tmpva;

    alltraintarg("&",j,tmpva)  = yn;

    for ( int jj = 0 ; jj < j.size() ; ++jj )
    {
        alltraintargR("&",j(jj))  = (double) yn(jj);
        alltraintargA("&",j(jj))  = (const d_anion &) yn(jj);
        alltraintargV("&",j(jj))  = (const Vector<double> &) yn(jj);
    }

    return 0;
}

int ML_Base::sety(const Vector<gentype> &yn)
{
    NiceAssert( yn.size() == ML_Base::N() );

    if ( isBasisUserUU && N() )
    {
        int i;

        isBasisUserUU = 0;

        for ( i = 0 ; i < N() ; ++i )
        {
            setBasisUU(i,yn(i));
        }

        isBasisUserUU = 1;
    }

    if ( isBasisUserVV && N() )
    {
        int i;

        isBasisUserVV = 0;

        for ( i = 0 ; i < N() ; ++i )
        {
            setBasisVV(i,yn(i));
        }

        isBasisUserVV = 1;
    }

    alltraintarg  = yn;

    for ( int jj = 0 ; jj < yn.size() ; ++jj )
    {
        alltraintargR("&",jj)  = (double) yn(jj);
        alltraintargA("&",jj)  = (const d_anion &) yn(jj);
        alltraintargV("&",jj)  = (const Vector<double> &) yn(jj);
    }

    return 0;
}

int ML_Base::sety(int i, double z)
{
    const gentype y(z);

    return sety(i,y);
}

int ML_Base::sety(const Vector<int> &i, const Vector<double> &z)
{
    Vector<gentype> y(z.size());

    //if ( y.size() )
    {
        for ( int j = 0 ; j < y.size() ; ++j )
        {
            y("&",j) = z(j);
        }
    }

    return sety(i,y);
}

int ML_Base::sety(const Vector<double> &z)
{
    Vector<gentype> y(z.size());

    //if ( y.size() )
    {
        for ( int j = 0 ; j < y.size() ; ++j )
        {
            y("&",j) = z(j);
        }
    }

    return sety(y);
}

int ML_Base::sety(int i, const Vector<double> &z)
{
    gentype y(z);

    return sety(i,y);
}

int ML_Base::sety(const Vector<int> &i, const Vector<Vector<double> > &z)
{
    Vector<gentype> y(z.size());

    //if ( y.size() )
    {
        for ( int j = 0 ; j < y.size() ; ++j )
        {
            y("&",j) = z(j);
        }
    }

    return sety(i,y);
}

int ML_Base::sety(const Vector<Vector<double> > &z)
{
    Vector<gentype> y(z.size());

    //if ( y.size() )
    {
        for ( int j = 0 ; j < y.size() ; ++j )
        {
            y("&",j) = z(j);
        }
    }

    return sety(y);
}

int ML_Base::sety(int i, const d_anion &z)
{
    gentype y(z);

    return sety(i,y);
}

int ML_Base::sety(const Vector<int> &i, const Vector<d_anion> &z)
{
    Vector<gentype> y(z.size());

    //if ( y.size() )
    {
        for ( int j = 0 ; j < y.size() ; ++j )
        {
            y("&",j) = z(j);
        }
    }

    return sety(i,y);
}

int ML_Base::sety(const Vector<d_anion> &z)
{
    Vector<gentype> y(z.size());

    //if ( y.size() )
    {
        for ( int j = 0 ; j < y.size() ; ++j )
        {
            y("&",j) = z(j);
        }
    }

    return sety(y);
}


int ML_Base::setzerotol(double zt)
{
    NiceAssert( zt >= 0 );

    globalzerotol = zt;

    return 0;
}

int ML_Base::setCweight(int i, double nv)
{
    xCweight("&",i) = nv;

    return 0;
}

int ML_Base::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    retVector<double> tmpva;

    xCweight("&",i,tmpva) = nv;

    return 0;
}

int ML_Base::setCweight(const Vector<double> &nv)
{
    xCweight = nv;

    return 0;
}

int ML_Base::setCweightfuzz(int i, double nv)
{
    xCweightfuzz("&",i) = nv;

    return 0;
}

int ML_Base::setCweightfuzz(const Vector<int> &i, const Vector<double> &nv)
{
    retVector<double> tmpva;

    xCweightfuzz("&",i,tmpva) = nv;

    return 0;
}

int ML_Base::setCweightfuzz(const Vector<double> &nv)
{
    xCweightfuzz = nv;

    return 0;
}

int ML_Base::setsigmaweight(int i, double nv)
{
    return setCweight(i,1.0/nv);
}

int ML_Base::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    Vector<double> xnv(nv);

    //if ( xnv.size() )
    {
        for ( int j = 0 ; j < xnv.size() ; ++j )
        {
            xnv("&",j) = 1.0/nv(j);
        }
    }

    return setCweight(i,xnv);
}

int ML_Base::setsigmaweight(const Vector<double> &nv)
{
    Vector<double> xnv(nv);

    //if ( xnv.size() )
    {
        for ( int j = 0 ; j < xnv.size() ; ++j )
        {
            xnv("&",j) = 1.0/nv(j);
        }
    }

    return setCweight(xnv);
}

int ML_Base::setepsweight(int i, double nv)
{
    xepsweight("&",i) = nv;

    return 0;
}

int ML_Base::setepsweight(const Vector<int> &i, const Vector<double> &nv)
{
    retVector<double> tmpva;

    xepsweight("&",i,tmpva) = nv;

    return 0;
}

int ML_Base::setepsweight(const Vector<double> &nv)
{
    xepsweight = nv;

    return 0;
}

int ML_Base::scaleCweight(double s)
{
    NiceAssert( s >= 0 );

    xCweight *= s;

    return 0;
}

int ML_Base::scaleCweightfuzz(double s)
{
    NiceAssert( s >= 0 );

    xCweightfuzz *= s;

    return 0;
}

int ML_Base::scaleepsweight(double s)
{
    NiceAssert( s >= 0 );

    xepsweight *= s;

    return 0;
}

int ML_Base::setd(int i, int nd)
{
    xdzero -= xd(i) ? 0 : 1;
    xd("&",i) = nd;
    xdzero += xd(i) ? 0 : 1;

    return 0;
}

int ML_Base::setd(const Vector<int> &i, const Vector<int> &nd)
{
    retVector<int> tmpva;

    xd("&",i,tmpva) = nd;

    //if ( ML_Base::N() )
    {
        for ( int j = 0 ; j < ML_Base::N() ; ++j )
        {
            xdzero += xd(j) ? 0 : 1;
        }
    }

    return 0;
}

int ML_Base::setd(const Vector<int> &nd)
{
    xd = nd;

    //if ( ML_Base::N() )
    {
        for ( int j = 0 ; j < ML_Base::N() ; ++j )
        {
            xdzero += xd(j) ? 0 : 1;
        }
    }

    return 0;
}

const Vector<double> &ML_Base::sigmaweight(void) const
{
    (xsigmaweightonfly).useSlackAllocation();
    (xsigmaweightonfly) = xCweight;

    for ( int i = 0 ; i < (xsigmaweightonfly).size() ; ++i )
    {
        (xsigmaweightonfly)("&",i) = 1.0/((xsigmaweightonfly)(i));
    }

    return (xsigmaweightonfly);
}




















void ML_Base::stabProbTrainingVector(double &res, int i, int p, double pnrm, int rot, double mu, double B) const
{
    NiceAssert( p >= 0 );
    NiceAssert( pnrm >= 0 );
    NiceAssert( mu >= 0 );
    NiceAssert( B >= 0 );

    SparseVector<gentype> xaug(x(i));

    int k,l;

    res = 1;

    if ( ( p > 0 ) || rot )
    {
        int n = xspaceDim();

        if ( rot )
        {
            // Special case: can calculate in closed form.

            Vector<gentype> eigvals;
            Matrix<gentype> eigvects;
            Vector<gentype> fv1;

            gentype xmeanvec;
            gentype xvarmat;

            for ( k = 1 ; k <= p ; ++k )
            {
                // Setting f4(6) talls the model to evaluate the kth derivative

                xaug.f4("&",6) = k;

                // Calculate mean (vector) and variance (matrix) for kth derivative

                //gg(xmeanvec,xaug);
                var(xvarmat,xmeanvec,xaug);

                // Scale mean (vector) and variance (matrix) as required

                xmeanvec *= pow(B,k)/xnfact(k);
                xvarmat  *= pow(pow(B,k)/xnfact(k),2);

                // Do eigendecomposition of variance matrix

                ((const Matrix<gentype> &) xvarmat).eig(eigvals,eigvects,fv1);

                // Calculate rotated means

                Vector<gentype> &meanvals = (xmeanvec.dir_vector());
                meanvals *= eigvects; // eigvects has eigenvectors in columns

                // Calculate stability score

                int meandim = (int) pow(n,k);

                NiceAssert( meanvals.size() == meandim );

                for ( l = 0 ; l < meandim ; ++l )
                {
                    if ( testisvnan((double) eigvals(l)) || testisinf((double) eigvals(l)) )
                    {
                        eigvals("&",l) = 1.0;
                    }

                    if ( (double) eigvals(l) <= 1e-8 )
                    {
                        eigvals("&",l) = 1e-8;
                    }

                    OP_sqrt(eigvals("&",l));

                    double Phil = 0;
                    double Phir = 0;

                    //numbase_Phi(Phil,( ((double) meanvals(l)) + mu ) / ((double) eigvals(l)) );
                    //numbase_Phi(Phir,( ((double) meanvals(l)) - mu ) / ((double) eigvals(l)) );

                    Phil = normPhi(( ((double) meanvals(l)) + mu ) / ((double) eigvals(l)) );
                    Phir = normPhi(( ((double) meanvals(l)) - mu ) / ((double) eigvals(l)) );

                    res *= (Phil-Phir);
                }
            }
        }

        else
        {
            // General case: generate a bunch of samples and test

            // First calculate mean and variance

            gentype xmeanvec;
            gentype xvarmat;
            Matrix<gentype> xvarchol;

            for ( k = 1 ; k <= p ; ++k )
            {
                int meandim = (int) pow(n,k);

                // Setting f4(6) tells the model to evaluate the kth derivative

                xaug.f4("&",6) = k;

                // Calculate mean (vector) and variance (matrix) for kth derivative

                //gg(xmeanvec,xaug);
                var(xvarmat,xmeanvec,xaug);

                // Scale mean (vector) and variance (matrix) as required

                xmeanvec *= pow(B,k)/xnfact(k);
                xvarmat  *= pow(pow(B,k)/xnfact(k),2);

                Vector<double> xxmeanvec(meandim);
                Matrix<double> xxvarmat(meandim,meandim);

                int r,s;

                const Vector<gentype> &ghgh = (const Vector<gentype> &) xmeanvec;

                for ( r = 0 ; r < meandim ; ++r )
                {
                    xxmeanvec("&",r) = (double) ghgh(r);

                    const Matrix<gentype> &ghgh = (const Matrix<gentype> &) xvarmat;

                    for ( s = 0 ; s < meandim ; ++s )
                    {
                        xxvarmat("&",r,s) = (double) ghgh(r,s);
                    }
                }

                // Do Cholesky decomposition of variance matrix

                Matrix<double> xxvarchol(meandim,meandim);

                xxvarmat.naiveChol(xxvarchol,1);

                int totsamp = DEFAULT_BAYES_STABSTABREP;
                int goodsamp = 0;

                Vector<double> v(meandim);

                for ( l = 0 ; l < totsamp ; ++l )
                {
                    randnfill(v);
                    rightmult(xxvarchol,v);
                    v += xxmeanvec;

                    if ( absp(v,pnrm) <= mu )
                    {
                        ++goodsamp;
                    }
                }

                res *= ((double) goodsamp)/((double) totsamp);
            }
        }
    }

    return;
}





































std::ostream &ML_Base::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << " ML Base Class\n";
    repPrint(output,'>',dep) << " =============\n";
    repPrint(output,'>',dep) << " \n";
    repPrint(output,'>',dep) << " Base kernel:                      " << kernel           << "\n";
    repPrint(output,'>',dep) << " Base kernel bypass:               " << K2mat            << "\n";
    repPrint(output,'>',dep) << " Base prior mean type:             " << xmuprior         << "\n";
    repPrint(output,'>',dep) << " Base prior mean gentype:          " << xmuprior_gt      << "\n";
    repPrint(output,'>',dep) << " Base prior mean ML:               " << xmuprior_ml      << "\n";
    repPrint(output,'>',dep) << " Base training data:               " << allxdatagent     << "\n";
    repPrint(output,'>',dep) << " Base wild target:                 " << (ytargdata)      << "\n";
    repPrint(output,'>',dep) << " Base wild target real:            " << (ytargdataR)     << "\n";
    repPrint(output,'>',dep) << " Base wild target anion:           " << (ytargdataA)     << "\n";
    repPrint(output,'>',dep) << " Base wild target vector:          " << (ytargdataV)     << "\n";
    repPrint(output,'>',dep) << " Base wild target prior:           " << (ytargdatap)     << "\n";
    repPrint(output,'>',dep) << " Base wild target prior real:      " << (ytargdatapR)    << "\n";
    repPrint(output,'>',dep) << " Base wild target prior anion:     " << (ytargdatapA)    << "\n";
    repPrint(output,'>',dep) << " Base wild target prior vector:    " << (ytargdatapV)    << "\n";
    repPrint(output,'>',dep) << " Base training targets:            " << alltraintarg     << "\n";
    repPrint(output,'>',dep) << " Base training targets real:       " << alltraintargR    << "\n";
    repPrint(output,'>',dep) << " Base training targets anion:      " << alltraintargA    << "\n";
    repPrint(output,'>',dep) << " Base training targets vector:     " << alltraintargV    << "\n";
    repPrint(output,'>',dep) << " Base training targets prior:      " << alltraintargp    << "\n";
    repPrint(output,'>',dep) << " Base training targets prior real: " << alltraintargpR   << "\n";
    repPrint(output,'>',dep) << " Base training targets prir anion: " << alltraintargpA   << "\n";
    repPrint(output,'>',dep) << " Base training targets prior vect: " << alltraintargpV   << "\n";
    repPrint(output,'>',dep) << " Base training info:               " << traininfo        << "\n";
    repPrint(output,'>',dep) << " Base training tangles:            " << traintang        << "\n";
    repPrint(output,'>',dep) << " Base d:                           " << xd               << "\n";
    repPrint(output,'>',dep) << " Base dnz:                         " << xdzero           << "\n";
    repPrint(output,'>',dep) << " Base sigma:                       " << locsigma         << "\n";
    repPrint(output,'>',dep) << " Base learning rate:               " << loclr            << "\n";
    repPrint(output,'>',dep) << " Base learning rate:               " << loclrb           << "\n";
    repPrint(output,'>',dep) << " Base learning rate:               " << loclrc           << "\n";
    repPrint(output,'>',dep) << " Base learning rate:               " << loclrd           << "\n";
    repPrint(output,'>',dep) << " Base zero tol:                    " << globalzerotol    << "\n";
    repPrint(output,'>',dep) << " Base C weight:                    " << xCweight         << "\n";
    repPrint(output,'>',dep) << " Base C weight (fuzz):             " << xCweightfuzz     << "\n";
    repPrint(output,'>',dep) << " Base eps weight:                  " << xepsweight       << "\n";
    repPrint(output,'>',dep) << " Base \"alpha\" state:             " << xalphaState      << "\n";
    repPrint(output,'>',dep) << " Base training index key:          " << indexKey         << "\n";
    repPrint(output,'>',dep) << " Base training index count:        " << indexKeyCount    << "\n";
    repPrint(output,'>',dep) << " Base training type key:           " << typeKey          << "\n";
    repPrint(output,'>',dep) << " Base training type key breakdown: " << typeKeyBreak     << "\n";
    repPrint(output,'>',dep) << " Base index pruning:               " << isIndPrune       << "\n";
    repPrint(output,'>',dep) << " Base x assumed consistency:       " << xassumedconsist  << "\n";
    repPrint(output,'>',dep) << " Base x actual consistency:        " << xconsist         << "\n";
    repPrint(output,'>',dep) << " Base x data real:                 " << trainingDataReal << "\n";
    repPrint(output,'>',dep) << " Base x data assume real:          " << assumeReal       << "\n";
    repPrint(output,'>',dep) << " Base output kernel U:             " << UUoutkernel      << "\n";
    repPrint(output,'>',dep) << " Base RFF kernel:                  " << RFFkernel        << "\n";
    repPrint(output,'>',dep) << " Base local basis U:               " << isBasisUserUU    << "\n";
    repPrint(output,'>',dep) << " Base y basis U:                   " << locbasisUU       << "\n";
    repPrint(output,'>',dep) << " Base default projection U:        " << defbasisUU       << "\n";
    repPrint(output,'>',dep) << " Base local basis V:               " << isBasisUserVV    << "\n";
    repPrint(output,'>',dep) << " Base y basis V:                   " << locbasisVV       << "\n";
    repPrint(output,'>',dep) << " Base default projection V:        " << defbasisVV       << "\n";

    return output;
}

std::istream &ML_Base::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> kernel;
    input >> dummy; input >> K2mat;
    input >> dummy; input >> xmuprior;
    input >> dummy; input >> xmuprior_gt;
    input >> dummy; //input >> xmuprior_ml;
    input >> dummy; input >> allxdatagent;
    input >> dummy; input >> (ytargdata);
    input >> dummy; input >> (ytargdataR);
    input >> dummy; input >> (ytargdataA);
    input >> dummy; input >> (ytargdataV);
    input >> dummy; input >> (ytargdatap);
    input >> dummy; input >> (ytargdatapR);
    input >> dummy; input >> (ytargdatapA);
    input >> dummy; input >> (ytargdatapV);
    input >> dummy; input >> alltraintarg;
    input >> dummy; input >> alltraintargR;
    input >> dummy; input >> alltraintargA;
    input >> dummy; input >> alltraintargV;
    input >> dummy; input >> alltraintargp;
    input >> dummy; input >> alltraintargpR;
    input >> dummy; input >> alltraintargpA;
    input >> dummy; input >> alltraintargpV;
    input >> dummy; input >> traininfo;
    input >> dummy; input >> traintang;
    input >> dummy; input >> xd;
    input >> dummy; input >> xdzero;
    input >> dummy; input >> locsigma;
    input >> dummy; input >> loclr;
    input >> dummy; input >> loclrb;
    input >> dummy; input >> loclrc;
    input >> dummy; input >> loclrd;
    input >> dummy; input >> globalzerotol;
    input >> dummy; input >> xCweight;
    input >> dummy; input >> xCweightfuzz;
    input >> dummy; input >> xepsweight;
    input >> dummy; input >> xalphaState;
    input >> dummy; input >> indexKey;
    input >> dummy; input >> indexKeyCount;
    input >> dummy; input >> typeKey;
    input >> dummy; input >> typeKeyBreak;
    input >> dummy; input >> isIndPrune;
    input >> dummy; input >> xassumedconsist;
    input >> dummy; input >> xconsist;
    input >> dummy; input >> trainingDataReal;
    input >> dummy; input >> assumeReal;
    input >> dummy; input >> UUoutkernel;
    input >> dummy; input >> RFFkernel;
    input >> dummy; input >> isBasisUserUU;
    input >> dummy; input >> locbasisUU;
    input >> dummy; input >> defbasisUU;
    input >> dummy; input >> isBasisUserVV;
    input >> dummy; input >> locbasisVV;
    input >> dummy; input >> defbasisVV;

    fixpvects();

    incxvernum();
    incgvernum();

    calcSetAssumeReal();

    return input;
}

void ML_Base::setmemsize(int memsize)
{
   (void) memsize;

   return;
}

int ML_Base::prealloc(int expectedN)
{
    NiceAssert( ( expectedN == -1 ) || ( expectedN > 0 ) );

    if ( expectedN > LARGE_TRAIN_BOUNDARY )
    {
        disableAltContent();
    }

    xpreallocsize = expectedN;

    allxdatagent.prealloc(expectedN);
//FIXME    allxdatagentp.prealloc(expectedN);
    alltraintarg.prealloc(expectedN);
    alltraintargR.prealloc(expectedN);
    alltraintargA.prealloc(expectedN);
    alltraintargV.prealloc(expectedN);
    alltraintargp.prealloc(expectedN);
    alltraintargpR.prealloc(expectedN);
    alltraintargpA.prealloc(expectedN);
    alltraintargpV.prealloc(expectedN);
    traininfo.prealloc(expectedN);
    traintang.prealloc(expectedN);
//FIXME    traininfop.prealloc(expectedN);
    xd.prealloc(expectedN);
    xCweight.prealloc(expectedN);
    xCweightfuzz.prealloc(expectedN);
    xepsweight.prealloc(expectedN);
    xalphaState.prealloc(expectedN);
    //indexKey.prealloc(expectedN);
    //indexKeyCount.prealloc(expectedN);
    //typeKey.prealloc(expectedN);
    //typeKeyBreak.prealloc(expectedN);

    return 0;
}







































































// Global functions

int ML_Base::disable(int i)
{
    if ( i < 0 )
    {
        return 0;
    }
//    i = ( i >= 0 ) ? i : ((-i-1)%N());

    return setd(i,0);
}

int ML_Base::disable(const Vector<int> &i)
{
    retVector<int> tmpva;
    Vector<int> j(i);

    for ( int ii = 0 ; ii < j.size() ; ii++ )
    {
        if ( j(ii) < 0 )
        {
            j.remove(ii);
            --ii;
        }
//        j("&",ii) = ( j(ii) >= 0 ) ? j(ii) : ((-j(ii)-1)%N());
    }

    return setd(j,zerointvec(j.size(),tmpva));
}

int ML_Base::renormalise(void)
{
    int res = 0;

    if ( ML_Base::N() )
    {
        int i;
        double maxout = 0.0;
        gentype gres;
        double gtemp = 0;

        for ( i = 0 ; i < ML_Base::N() ; ++i )
        {
            ggTrainingVector(gres,i);

            if ( gres.isValNull() )
            {
                ;
            }

            else if ( gres.isCastableToRealWithoutLoss() )
            {
                gtemp = abs2((double) gres);
            }

            else if ( gres.isCastableToVectorWithoutLoss() )
            {
                gtemp = absinf((const Vector<gentype> &) gres);
            }

            else if ( gres.isCastableToAnionWithoutLoss() )
            {
                gtemp = absinf((const d_anion &) gres);
            }

            else
            {
                NiceThrow("Unrecognised output type in renormalisation");
            }

            maxout = ( gtemp > maxout ) ? gtemp : maxout;
        }

        if ( maxout > 1 )
        {
            scale(1/maxout);
            res = 1;
        }
    }

    return res;
}


int ML_Base::realign(void)
{
    int res = 0;

    if ( ML_Base::N() )
    {
        Vector<gentype> locy(y());

        int i;
        gentype hres;

        for ( i = 0 ; i < ML_Base::N() ; ++i )
        {
            hhTrainingVector(hres,i);
            locy("&",i) = hres;
        }

        res = sety(locy);
    }

    return res;
}

int ML_Base::autoen(void)
{
    NiceAssert( hOutType() != 'A' );

    int res = 0;

    if ( ML_Base::N() )
    {
        Vector<gentype> locy(y());
        int i,j;

        for ( i = 0 ; i < ML_Base::N() ; ++i )
        {
            gentype hres = y(i);

            switch ( hOutType() )
            {
                case 'R':
                {
                    {
                        Vector<gentype> xtemp;
                        xlateFromSparseTrainingVector(xtemp,i);

                        if ( xtemp.size() )
                        {
                            hres = (double) xtemp(0);
                        }

                        else
                        {
                            setrand(hres);
                            hres *= 2.0;
                            hres -= 1.0;
                        }
                    }

                    break;
                }

                case 'V':
                {
                    {
                        Vector<gentype> xtemp;
                        xlateFromSparseTrainingVector(xtemp,i);

                        if ( xtemp.size() )
                        {
                            for ( j = 0 ; j < ( xtemp.size() <= hres.dir_vector().size() ? xtemp.size() : hres.dir_vector().size() ) ; ++j )
                            {
                                hres.dir_vector()("&",j) = xtemp(j);
                            }
                        }

                        if ( hres.dir_vector().size() > xtemp.size() )
                        {
                            for ( j = xtemp.size() ; j < hres.dir_vector().size() ; ++j )
                            {
                                setrand(hres.dir_vector()("&",j));
                                hres.dir_vector()("&",j) *= 2.0;
                                hres.dir_vector()("&",j) -= 1.0;
                            }
                        }
                    }

                    break;
                }

                default:
                {
                    // Don't throw - this includes nullptr target
                    break;
                }
            }

            hhTrainingVector(hres,i);
            locy("&",i) = hres;
        }

        res = sety(locy);
    }

    return res;
}












int ML_Base::settspaceDim(int newdim)
{
    NiceAssert( ( ( N() == 0 ) && ( newdim >= -1 ) ) || ( newdim >= 0 ) );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ++ii )
        {
            alltraintarg("&",ii).dir_vector().resize(newdim);
            alltraintargA("&",ii).resize(newdim);
            alltraintargV("&",ii).resize(newdim);
        }
    }

    return 1;
}

int ML_Base::addtspaceFeat(int i)
{
    NiceAssert( ( i >= 0 ) && ( i <= tspaceDim() ) );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ++ii )
        {
            alltraintarg("&",ii).dir_vector().add(i);
            alltraintargA("&",ii) = (const d_anion &) alltraintarg(ii);
            alltraintargV("&",ii) = (const Vector<double> &) alltraintarg(ii);
        }
    }

    return 1;
}

int ML_Base::removetspaceFeat(int i)
{
    NiceAssert( ( i >= 0 ) && ( i < tspaceDim() ) );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ++ii )
        {
            alltraintarg("&",ii).dir_vector().remove(i);
            alltraintargA("&",ii) = (const d_anion &) alltraintarg(ii);
            alltraintargV("&",ii) = (const Vector<double> &) alltraintarg(ii);
        }
    }

    return 1;
}

int ML_Base::setorder(int neword)
{
    NiceAssert( neword >= 0 );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ++ii )
        {
            alltraintarg("&",ii).dir_anion().setorder(neword);
            alltraintargA("&",ii).setorder(neword);
            alltraintargV("&",ii).setorder(neword);
        }
    }

    return 1;
}































void ML_Base::fillCache(int Ns,int Ne)
{
    Ne = ( Ne >= 0 ) ? Ne : N()-1;

    if ( Ns <= Ne )
    {
        gentype dummy;
        int i,j;

        for ( i = Ns ; i <= Ne ; ++i )
        {
            for ( j = Ns ; j <= Ne ; ++j )
            {
                K2(dummy,i,j);
            }
        }
    }

    return;
}


int ML_Base::isKVarianceNZ(void) const
{
    return getKernel().isKVarianceNZ();
}

void ML_Base::fastg(double &res) const
{
    SparseVector<gentype> x;

    gg(res,x);

    return;
}

void ML_Base::fastg(double &res, 
                        int ia, 
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo) const
{
    (void) ia;
    (void) xainfo;

    gg(res,xa);

    return;
}

void ML_Base::fastg(double &res, 
                        int ia, int ib, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                        const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    (void) xainfo;
    (void) xbinfo;

    (void) ia;
    (void) ib;

    SparseVector<gentype> x(xa);

    //if ( xb.indsize() )
    {
        for ( int i = 0 ; i < xb.indsize() ; ++i )
        {
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void ML_Base::fastg(double &res, 
                        int ia, int ib, int ic, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;

    (void) ia;
    (void) ib;
    (void) ic;

    SparseVector<gentype> x(xa);

    //if ( xb.indsize() )
    {
        for ( int i = 0 ; i < xb.indsize() ; ++i )
        {
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    //if ( xc.indsize() )
    {
        for ( int i = 0 ; i < xc.indsize() ; ++i )
        {
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void ML_Base::fastg(double &res, 
                        int ia, int ib, int ic, int id,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;

    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;

    SparseVector<gentype> x(xa);

    //if ( xb.indsize() )
    {
        for ( int i = 0 ; i < xb.indsize() ; ++i )
        {
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    //if ( xc.indsize() )
    {
        for ( int i = 0 ; i < xc.indsize() ; ++i )
        {
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    //if ( xd.indsize() )
    {
        for ( int i = 0 ; i < xd.indsize() ; ++i )
        {
            x("&",xd.ind(i),3) = xd.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void ML_Base::fastg(double &res, 
                        Vector<int> &ia, 
                        Vector<const SparseVector<gentype> *> &xa,
                        Vector<const vecInfo *> &xainfo) const
{
    (void) xainfo;
    (void) ia;

    SparseVector<gentype> x;

    //if ( xa.size() )
    {
        int i,j;

        for ( j = 0 ; j < xa.size() ; ++j )
        {
            const SparseVector<gentype> &xb = (*(xa(j)));

            if ( xb.indsize() )
            {
                for ( i = 0 ; i < xb.indsize() ; ++i )
                {
                    x("&",xb.ind(i),j) = xb.direcref(i);
                }
            }
        }
    }

    gg(res,x);

    return;
}


void ML_Base::fastg(gentype &res) const
{
    SparseVector<gentype> x;

    gg(res,x);

    return;
}

void ML_Base::fastg(gentype &res, 
                        int ia, 
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo) const
{
    (void) ia;
    (void) xainfo;

    gg(res,xa);

    return;
}

void ML_Base::fastg(gentype &res, 
                        int ia, int ib, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                        const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    (void) xainfo;
    (void) xbinfo;

    (void) ia;
    (void) ib;

    SparseVector<gentype> x(xa);

    //if ( xb.indsize() )
    {
        for ( int i = 0 ; i < xb.indsize() ; ++i )
        {
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void ML_Base::fastg(gentype &res, 
                        int ia, int ib, int ic, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;

    (void) ia;
    (void) ib;
    (void) ic;

    SparseVector<gentype> x(xa);

    //if ( xb.indsize() )
    {
        for ( int i = 0 ; i < xb.indsize() ; ++i )
        {
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    //if ( xc.indsize() )
    {
        for ( int i = 0 ; i < xc.indsize() ; ++i )
        {
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void ML_Base::fastg(gentype &res, 
                        int ia, int ib, int ic, int id,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;

    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;

    SparseVector<gentype> x(xa);

    //if ( xb.indsize() )
    {
        for ( int i = 0 ; i < xb.indsize() ; ++i )
        {
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    //if ( xc.indsize() )
    {
        for ( int i = 0 ; i < xc.indsize() ; ++i )
        {
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    //if ( xd.indsize() )
    {
        for ( int i = 0 ; i < xd.indsize() ; ++i )
        {
            x("&",xd.ind(i),3) = xd.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void ML_Base::fastg(gentype &res, 
                        Vector<int> &ia, 
                        Vector<const SparseVector<gentype> *> &xa,
                        Vector<const vecInfo *> &xainfo) const
{
    (void) xainfo;
    (void) ia;

    SparseVector<gentype> x;

    //if ( xa.size() )
    {
        int i,j;

        for ( j = 0 ; j < xa.size() ; ++j )
        {
            const SparseVector<gentype> &xb = (*(xa(j)));

            if ( xb.indsize() )
            {
                for ( i = 0 ; i < xb.indsize() ; ++i )
                {
                    x("&",xb.ind(i),j) = xb.direcref(i);
                }
            }
        }
    }

    gg(res,x);

    return;
}

void ML_Base::K0xfer(gentype &res, int &minmaxind, int typeis,
                     const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K0(res,nullptr,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            res = 0.0;

            break;
        }

        default:
        {
            NiceThrow("K0xfer precursor type requested undefined at this level (only 800,806 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K1xfer(gentype &res, int &minmaxind, int typeis,
                     const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                     const SparseVector<gentype> &xa, 
                     const vecInfo &xainfo, 
                     int ia, 
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K1(res,ia,nullptr,&xa,&xainfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ia < 0 ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        {
            gentype ra;

            gg(ra,xa,&xainfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    oneProduct(res,ra.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;

            ggTrainingVector(ra,ia);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    oneProduct(res,ra.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            NiceThrow("K1xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                     const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                     const vecInfo &xainfo, const vecInfo &xbinfo,
                     int ia, int ib,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    dxyprod = 0.0;
    ddiffis = 0.0;
    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
//errstream() << "phantomxy 1: " << ia << "," << ib << "\t" << xa << "," << xb << "\n";
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K2(res,ia,ib,nullptr,&xa,&xb,&xainfo,&xbinfo,resmode);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    // K, dK/dxyprod, dK/dxnorm

                    K2(res,ia,ib,nullptr,&xa,&xb,&xainfo,&xbinfo,resmode);
                    dK(dxyprod,ddiffis,ia,ib,nullptr,&xa,&xb,&xainfo,&xbinfo,1); // deep derivative required

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        {
            gentype ra;
            gentype rb;

            gg(ra,xa,&xainfo);
            gg(rb,xb,&xbinfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    innerProduct(res,ra.cast_vector(),rb.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;
            gentype rb;

            ggTrainingVector(ra,ia);
            ggTrainingVector(rb,ib);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    innerProduct(res,ra.cast_vector(),rb.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 809:
        case 819:
        {
//errstream() << "phantomxy 1: " << ia << "," << ib << "\t" << xa << "," << xb << "\n";
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            switch ( resmode )
            {
                case 0:
                {
                    gentype dummy;

                    covTrainingVector(res,dummy,ia,ib);

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        default:
        {
            NiceThrow("K2xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K3xfer(gentype &res, int &minmaxind, int typeis,
                     const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                     int ia, int ib, int ic, 
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K3(res,ia,ib,ic,nullptr,&xa,&xb,&xc,&xainfo,&xbinfo,&xcinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        {
            gentype ra;
            gentype rb;
            gentype rc;

            gg(ra,xa,&xainfo);
            gg(rb,xb,&xbinfo);
            gg(rc,xc,&xcinfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    threeProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;
            gentype rb;
            gentype rc;

            ggTrainingVector(ra,ia);
            ggTrainingVector(rb,ib);
            ggTrainingVector(rc,ic);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    threeProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            NiceThrow("K3xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K4xfer(gentype &res, int &minmaxind, int typeis,
                     const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                     int ia, int ib, int ic, int id,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K4(res,ia,ib,ic,id,nullptr,&xa,&xb,&xc,&xd,&xainfo,&xbinfo,&xcinfo,&xdinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        {
            gentype ra;
            gentype rb;
            gentype rc;
            gentype rd;

            gg(ra,xa,&xainfo);
            gg(rb,xb,&xbinfo);
            gg(rc,xc,&xcinfo);
            gg(rd,xd,&xdinfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fourProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector(),rd.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;
            gentype rb;
            gentype rc;
            gentype rd;

            ggTrainingVector(ra,ia);
            ggTrainingVector(rb,ib);
            ggTrainingVector(rc,ic);
            ggTrainingVector(rd,id);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fourProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector(),rd.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            NiceThrow("K4xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::Kmxfer(gentype &res, int &minmaxind, int typeis,
                     const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                     Vector<const SparseVector<gentype> *> &x,
                     Vector<const vecInfo *> &xinfo,
                     Vector<int> &ii,
                     int xdim, int m, int densetype, int resmode, int mlid) const
{
    (void) mlid;
    (void) xdim;
    (void) minmaxind;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;
    (void) densetype;

//    if ( ( m == 0 ) || ( m == 2 ) || ( m == 4 ) )
//    {
//        kernPrecursor::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,ii,xdim,m,densetype,resmode,mlid);
//        return;
//    }

    NiceAssert( !densetype );

    res = 0.0;

    int iq;

    Vector<int> i(ii);

    for ( iq = 0 ; iq < m ; ++iq )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42;
    }

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            Vector<SparseVector<gentype> > *xx = nullptr;

            if ( !( i >= 0 ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ++ir )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    Km(m,res,i,nullptr,&x,&xinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( !( i >= 0 ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        {
            Vector<gentype> r(m);
            Vector<const Vector<gentype> *> rr(m);

            for ( iq = 0 ; iq < m ; ++iq )
            {
                gg(r("&",iq),*(x(iq)),xinfo(iq));
                rr("&",iq) = &(r(iq).cast_vector());
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    mProduct(res,rr);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            Vector<gentype> r(m);
            Vector<const Vector<gentype> *> rr(m);

            for ( iq = 0 ; iq < m ; ++iq )
            {
                ggTrainingVector(r("&",iq),i(iq));
                rr("&",iq) = &(r(iq).cast_vector());
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    mProduct(res,rr);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            NiceThrow("Kmxfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}



void ML_Base::K0xfer(double &res, int &minmaxind, int typeis,
                     double xyprod, double yxprod, double diffis,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    res = K0(nullptr,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            gentype gxyprod(xyprod);
            gentype gyxprod(yxprod);
            gentype gdiffis(diffis);

            K0xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K1xfer(double &res, int &minmaxind, int typeis,
                     double xyprod, double yxprod, double diffis,
                     const SparseVector<gentype> &xa, 
                     const vecInfo &xainfo, 
                     int ia, 
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    res = K1(ia,nullptr,&xa,&xainfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            gentype gxyprod(xyprod);
            gentype gyxprod(yxprod);
            gentype gdiffis(diffis);

            K1xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                     double xyprod, double yxprod, double diffis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                     const vecInfo &xainfo, const vecInfo &xbinfo,
                     int ia, int ib,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    dxyprod = 0.0;
    ddiffis = 0.0;
    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
//errstream() << "phantomxy 0: " << ia << "," << ib << "\n";
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    res = K2(ia,ib,nullptr,&xa,&xb,&xainfo,&xbinfo,resmode);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    // K, dK/dxyprod, dK/dxnorm

                    res = K2(ia,ib,nullptr,&xa,&xb,&xainfo,&xbinfo,resmode);
                    dK(dxyprod,ddiffis,ia,ib,nullptr,&xa,&xb,&xainfo,&xbinfo,1); // deep derivative required

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempdxyprod(dxyprod);
            gentype tempddiffis(ddiffis);

            gentype tempres;

            gentype gxyprod(xyprod);
            gentype gyxprod(yxprod);
            gentype gdiffis(diffis);

            K2xfer(tempdxyprod,tempddiffis,tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        case 809:
        case 819:
        {
//errstream() << "phantomxy 1: " << ia << "," << ib << "\t" << xa << "," << xb << "\n";
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            switch ( resmode )
            {
                case 0:
                {
                    gentype dummy;
                    gentype tempres;

                    covTrainingVector(tempres,dummy,ia,ib);

                    res = (double) tempres;

                    break;
                }

                default:
                {
                    NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        default:
        {
            kernPrecursor::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K3xfer(double &res, int &minmaxind, int typeis,
                     double xyprod, double yxprod, double diffis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                     int ia, int ib, int ic, 
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    res = K3(ia,ib,ic,nullptr,&xa,&xb,&xc,&xainfo,&xbinfo,&xcinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            gentype gxyprod(xyprod);
            gentype gyxprod(yxprod);
            gentype gdiffis(diffis);

            K3xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K4xfer(double &res, int &minmaxind, int typeis,
                     double xyprod, double yxprod, double diffis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                     int ia, int ib, int ic, int id,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    res = K4(ia,ib,ic,id,nullptr,&xa,&xb,&xc,&xd,&xainfo,&xbinfo,&xcinfo,&xdinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            gentype gxyprod(xyprod);
            gentype gyxprod(yxprod);
            gentype gdiffis(diffis);

            K4xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::Kmxfer(double &res, int &minmaxind, int typeis,
                     double xyprod, double yxprod, double diffis,
                     Vector<const SparseVector<gentype> *> &x,
                     Vector<const vecInfo *> &xinfo,
                     Vector<int> &ii,
                     int xdim, int m, int densetype, int resmode, int mlid) const
{
    (void) mlid;

//    if ( ( m == 0 ) || ( m == 2 ) || ( m == 4 ) )
//    {
//        kernPrecursor::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,ii,xdim,m,densetype,resmode,mlid);
//        return;
//    }

    NiceAssert( !densetype );

    res = 0.0;

    int iq;

    Vector<int> i(ii);

    for ( iq = 0 ; iq < m ; ++iq )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42;
    }

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            Vector<SparseVector<gentype> > *xx = nullptr;

            if ( !( i >= 0 ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ++ir )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    res = Km(m,i,nullptr,&x,&xinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( !( i >= 0 ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    NiceThrow("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            gentype gxyprod(xyprod);
            gentype gyxprod(yxprod);
            gentype gdiffis(diffis);

            Kmxfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

double ML_Base::getvalIfPresent_v(int numi, int numj, int &isgood) const
{
    (void) numi;
    (void) numj;

    isgood = 0;

    return 0;
}
















































//FIXME these should be variables, not constants
//#define NUMZOOMS 3
//#define NUMZOOMS 2
//#define ZOOMFACTOR 0.3
//#define PARA_REGUL 0.05

//#define DEF_MAGIC_EPS 1e-4
//#define DEF_MAGIC_EPS_ABS 0
//#define DEF_VOLUME_RELTOL 0.0
//#define DEF_SIGMA_RELTOL -1.0
//
//    volatile int killSwitch = 0;
//    double magic_eps_abs = DEF_MAGIC_EPS_ABS;
//    double volume_reltol = DEF_VOLUME_RELTOL;
//    double sigma_reltol = DEF_SIGMA_RELTOL;
//    int max_feval = 5000;
//    int max_iter = 1000;
//    double maxtime = 0;
//    double eps = DEF_MAGIC_EPS;
//    direct_algorithm algorithm = DIRECT_ORIGINAL;
//
//    double *x
//    double y
//    double *lb  0s
//    double *u   1s
//    void *fdata;
//
//    double feval(int dim, const double *x, int *, void *fdata);
//
//    int intres = direct_optimize(
//         fevel,fdata,
//         ddim,
//         lb,ub,
//         x,&y,
//         max_feval,max_iter,
//         maxtime,
//         eps,magic_eps_abs,
//         volume_reltol,sigma_reltol,
//         killSwitch,
//         DIRECT_UNKNOWN_FGLOBAL,0,
//         algorithm);








double ML_Base::tuneKernel(int method, double xwidth, int tuneK, int tuneP, const tkBounds *tuneBounds)
{
    int    numzooms   = !tuneBounds ? ( tuneK ? NUMZOOMS : 1 ) : ((*tuneBounds).numzooms);
    double zoomfactor = !tuneBounds ? ZOOMFACTOR               : ((*tuneBounds).zoomfactor);

    double xscale = 1;

    if ( !tuneK && !tuneP )
    {
        return 0;
    }

    ML_Base &model = *this;

    if ( tuneK )
    {
        // If the data doesn't range 0->1 (unit hypercube) then our assumptions are off
        // Here we calculate the max gap, which will be used to rescale the lengthscale
        // It's not perfect and it makes a bunch of assumptions (range same on all axis, data covers full range), but it's an ok approx I think?

        SparseVector<gentype> xtmpmax;
        SparseVector<gentype> xtmpmin;

        xmax(xtmpmax);
        xmin(xtmpmin);

        if ( xtmpmax.indsize() )
        {
            for ( int i = 0 ; i < xtmpmax.indsize() ; i++ )
            {
                if ( !xtmpmax.direref(i).isCastableToRealWithoutLoss() )
                {
                    xscale = 1;
                    break;
                }

                if ( !xtmpmin.direref(i).isCastableToRealWithoutLoss() )
                {
                    xscale = 1;
                    break;
                }

                double xgap = ((double) xtmpmax.direref(i)) - ((double) xtmpmin.direref(i));

                if ( xgap > xscale )
                {
                    xscale = xgap;
                }
            }
        }
    }

    if ( !model.N() || ( model.N() <= model.NNC(0) ) )
    {
        return 1;
    }

    MercerKernel &kernel = model.getKernel_unsafe();
    int kdim = kernel.size();

    int i,j;

    // Gather data on kernel

    int ddim = 0;
    int adim = 1;

    Vector<int> kind;    // which element in kernel dictionary
    Vector<int> kelm;    // which element in cRealConstants
    Vector<double> kmin; // range minimum
    Vector<double> kmax; // range maximum
    Vector<double> kvar; // added noise variance
    Vector<int> kstp;    // number of steps over range
    Vector<Vector<gentype> > constVecs(kdim);

    Vector<int> uu(kdim);

    uu = -1; // default to full dimension

    if ( kernel.numSplits() || kernel.numMulSplits() )
    {
        int uuu = 0;

        for ( i = 0 ; i < kdim ; ++i )
        {
            uu("&",i) = uuu;

            if ( kernel.isSplit(i) || kernel.isMulSplit(i) )
            {
                ++uuu;
            }
        }
    }

    // Preliminary adim scaling

    double trycount = 1;

tryagain:
    if ( kdim )
    {
        for ( i = 0 ; i < kdim ; ++i )
        {
            constVecs("&",i) = kernel.cRealConstants(i);

            if ( constVecs(i).size() )
            {
                for ( j = -1 ; j < constVecs(i).size() ; ++j )
                {
                         if ( ( kdim > 1 ) && ( j == -1 ) && !(kernel.cWeight(i).isNomConst) ) { ddim++; adim *= ( ( ((int) (15/trycount)) > 5 ) ? ((int) (15/trycount)) : 5 ); }
                    else if ( j == -1 ) { ; }
                    else if ( ( kernel.cType(i) == 5 ) && ( j == 1 ) && !(kernel.cRealConstants(i)(j).isNomConst) ) { ddim++; adim *= 6; }
                    else if ( ( kernel.cType(i) == 48 ) && ( j == 1 ) && !(kernel.cRealConstants(i)(j).isNomConst) ) { ddim++; adim *= ( ( ((int) (15/trycount)) > 5 ) ? ((int) (15/trycount)) : 5 ); }
                    else if ( ( kernel.cType(i) < 800 ) && ( kernel.cType(i) != 0 ) && ( kernel.cType(i) != 48 ) && ( j == 0 ) && !(kernel.cRealConstants(i)(j).isNomConst) ) { ddim++; adim *= ( ( ((int) (50/trycount)) > 5 ) ? ((int) (50/trycount)) : 5 ); }
                }
            }
        }
    }

    if ( tuneP & 1 ) { ddim++; adim *= ( ( ((int) (20/trycount)) > 5 ) ? ((int) (20/trycount)) : 5 ); }
    if ( tuneP & 2 ) { ddim++; adim *= ( ( ((int) (20/trycount)) > 5 ) ? ((int) (20/trycount)) : 5 ); }

    adim *= numzooms;

    if ( ( adim > MAXADIM ) && ( trycount == 1 ) )
    {
        // adim/trycount^ddim = MAXADIM
        // adim = MAXADIM.trycount^ddim
        // trycount^ddim = adim/MAXADIM
        // trycount = (adim/MAXADIM)^(1/ddim)

        trycount = std::pow(adim/((double) MAXADIM),1/((double) ddim));

errstream() << "tuneKernel: trycount = " << trycount << "\n";
        adim = 1;
        ddim = 0;
        goto tryagain;
    }

    // Now work out the (resized) grid

    adim = 1;
    ddim = 0;

    if ( kdim )
    {
        for ( i = 0 ; i < kdim ; ++i )
        {
            constVecs("&",i) = kernel.cRealConstants(i);

            if ( constVecs(i).size() )
            {
                // Element -1 is the weight of the kernel in the sum

                for ( j = -1 ; j < constVecs(i).size() ; ++j )
                {
                    double lb;
                    double ub;
                    int steps;
                    int addit = 0;

                    // Fixme: currently basically do lengthscale (r0) for "normal" kernels, need to extend
                    // NB: we only want to tune one "weight" per multiplicative kernel group

                    if ( ( kdim > 1 ) && ( j == -1 ) && !(kernel.cWeight(i).isNomConst) ) // && kernel.isAdjWeight(i) )
//                    if ( ( kdim > 1 ) && ( j == -1 ) && ( !i || ( ( kernel.isSplit(i-1) != 1 ) && ( kernel.isMulSplit(i-1) != 1 ) ) ) )
////                    if ( ( j == -1 ) && ( !i || ( ( kernel.isSplit(i-1) != 1 ) && ( kernel.isMulSplit(i-1) != 1 ) ) ) )
                    {
                        // This is weight (linear)

                        lb    = 0.1;
                        ub    = 3;
                        steps = ( ( ((int) (15/trycount)) > 5 ) ? ((int) (15/trycount)) : 5 ); //15/trycount; //10;
                        addit = 1;

                        // Bound bounding

                        lb = kernel.cWeightLB(i).isValNull() ? lb : ( (double) kernel.cWeightLB(i) );
                        ub = kernel.cWeightUB(i).isValNull() ? ub : ( (double) kernel.cWeightUB(i) );

                        lb = ( !tuneBounds || ( (((*tuneBounds).wlb)(i)) < lb ) ) ? lb : (((*tuneBounds).wlb)(i));
                        ub = ( !tuneBounds || ( (((*tuneBounds).wub)(i)) > ub ) ) ? ub : (((*tuneBounds).wub)(i));
                    }

                    else if ( j == -1 )
                    {
                        ;
                    }

                    else if ( ( kernel.cType(i) == 5 ) && ( j == 1 ) && !(kernel.cRealConstants(i)(j).isNomConst) ) //&& kernel.isAdjRealConstants(j,i) )
                    {
                        // This is norm order (linear)

                        lb    = 1;
                        ub    = 5;
                        steps = 6;
                        addit = 1;

                        lb = kernel.cRealConstantsLB(i)(j).isValNull() ? lb : ( (double) kernel.cRealConstantsLB(i)(j) );
                        ub = kernel.cRealConstantsUB(i)(j).isValNull() ? ub : ( (double) kernel.cRealConstantsUB(i)(j) );

                        lb = ( !tuneBounds || ( (((*tuneBounds).klb)(i)(j)) < lb ) ) ? lb : (((*tuneBounds).klb)(i)(j));
                        ub = ( !tuneBounds || ( (((*tuneBounds).kub)(i)(j)) > ub ) ) ? ub : (((*tuneBounds).kub)(i)(j));
                    }

                    else if ( ( kernel.cType(i) == 48 ) && ( j == 1 ) && !(kernel.cRealConstants(i)(j).isNomConst) ) //&& kernel.isAdjRealConstants(j,i) )
                    {
                        // This is inter-task relatedness (linear)

                        lb    = 0;
                        ub    = 1;
                        steps = ( ( ((int) (15/trycount)) > 5 ) ? ((int) (15/trycount)) : 5 ); //15;
                        addit = 1;

                        lb = kernel.cRealConstantsLB(i)(j).isValNull() ? lb : ( (double) kernel.cRealConstantsLB(i)(j) );
                        ub = kernel.cRealConstantsUB(i)(j).isValNull() ? ub : ( (double) kernel.cRealConstantsUB(i)(j) );

                        lb = ( !tuneBounds || ( (((*tuneBounds).klb)(i)(j)) < lb ) ) ? lb : (((*tuneBounds).klb)(i)(j));
                        ub = ( !tuneBounds || ( (((*tuneBounds).kub)(i)(j)) > ub ) ) ? ub : (((*tuneBounds).kub)(i)(j));
                    }

                    else if ( ( kernel.cType(i) < 800 ) && ( kernel.cType(i) != 0 ) && ( kernel.cType(i) != 48 ) && ( j == 0 ) && !(kernel.cRealConstants(i)(j).isNomConst) ) //&& kernel.isAdjRealConstants(j,i) )
                    {
                        // This is length-scale, always, with the single exception of kernels 0 and 48 where lengthscale is meaningless (log)

                        // distance between points scales as the square-root of their dimension
                        // also correct for the 1/N^dim

                        int xdim = xspaceDim(uu(i));

//errstream() << "phantomxyztune xdim(" << i << ") = " << xdim << "\n";
                        double lencorrect = std::sqrt((double) xdim)/std::pow( (double) N(),(double) xdim );

                        lencorrect = ( lencorrect < 0.05 ) ? 0.05 : lencorrect;
//errstream() << "phantomxyztune lencorrect(" << i << ") = " << lencorrect << "\n";

                        lb    = ( ( uu(i) == -1 ) ? xscale : 1.0 )*lencorrect*0.1*xwidth; //0.3*xwidth; // 0.01*xwidth; //1e-2*xwidth;
//errstream() << "phantomxyztune lb(" << i << ") = " << lb << "\n";
//FIXME: consider increasing ub
                        ub    = ( ( uu(i) == -1 ) ? xscale : 1.0 )*1.5*sqrt((double) xdim)*xwidth; //sqrt((double) xdim)*xwidth; // 3*xwidth; //15*xwidth;
//errstream() << "phantomxyztune ub(" << i << ") = " << ub << "\n";
                        steps = ( ( ((int) (50/trycount)) > 5 ) ? ((int) (50/trycount)) : 5 ); //50/trycount; //30; //20; // 15; //20;
                        addit = 1;

                        lb = kernel.cRealConstantsLB(i)(j).isValNull() ? lb : ( (double) kernel.cRealConstantsLB(i)(j) );
                        ub = kernel.cRealConstantsUB(i)(j).isValNull() ? ub : ( (double) kernel.cRealConstantsUB(i)(j) );

                        lb = ( !tuneBounds || ( (((*tuneBounds).klb)(i)(j)) < lb ) ) ? lb : (((*tuneBounds).klb)(i)(j));
                        ub = ( !tuneBounds || ( (((*tuneBounds).kub)(i)(j)) > ub ) ) ? ub : (((*tuneBounds).kub)(i)(j));
                    }

                    if ( addit )
                    {
                        kind.add(ddim); kind("&",ddim) = i;
                        kelm.add(ddim); kelm("&",ddim) = j;
                        kmin.add(ddim); kmin("&",ddim) = lb;
                        kmax.add(ddim); kmax("&",ddim) = ub;
                        kstp.add(ddim); kstp("&",ddim) = steps;

                        ++ddim;
                        adim *= steps;
                    }
                }
            }
        }
    }

    if ( tuneP & 1 )
    {
        // tune C

        double lb;
        double ub;
        int steps;

        lb = !tuneBounds ? DEFCMIN : ((*tuneBounds).Cmin);
        ub = !tuneBounds ? DEFCMAX : ((*tuneBounds).Cmax);
        steps = ( ( ((int) (20/trycount)) > 5 ) ? ((int) (20/trycount)) : 5 ); //20/trycount;

        kind.add(ddim); kind("&",ddim) = -2;
        kelm.add(ddim); kelm("&",ddim) = -2;
        kmin.add(ddim); kmin("&",ddim) = lb;
        kmax.add(ddim); kmax("&",ddim) = ub;
        kstp.add(ddim); kstp("&",ddim) = steps;

        ++ddim;
        adim *= steps;
    }

    if ( tuneP & 2 )
    {
        // tune eps

        double lb;
        double ub;
        int steps;

        lb = !tuneBounds ? DEFEPSMIN : ((*tuneBounds).epsmin);
        ub = !tuneBounds ? DEFEPSMAX : ((*tuneBounds).epsmax);
        steps = ( ( ((int) (20/trycount)) > 5 ) ? ((int) (20/trycount)) : 5 ); //20/trycount;

        kind.add(ddim); kind("&",ddim) = -3;
        kelm.add(ddim); kelm("&",ddim) = -3;
        kmin.add(ddim); kmin("&",ddim) = lb;
        kmax.add(ddim); kmax("&",ddim) = ub;
        kstp.add(ddim); kstp("&",ddim) = steps;

        ++ddim;
        adim *= steps;
    }

    double bestres = 1;

    if ( ddim )
    {
        Vector<int> pointspec(ddim);

        Vector<Vector<int> > stepgrid(adim);
        Vector<double> gridres(adim);

        Vector<double> weightval(kdim);

        double Cval = 1;
        double epsval = 1;

//        Vector<double> L1norm(adim); - tried regularization, it made it worse

        int bestind = -1;
        int dummy;

        for ( int zooms = 0 ; zooms < numzooms ; zooms++ )
        {
            if ( bestind == -1 )
            {
                // setup step grid

                stepgrid = pointspec;

                for ( i = 0 ; i < adim ; ++i )
                {
                    if ( !i )
                    {
                        stepgrid("&",i) = 0;
                    }

                    else
                    {
                        stepgrid("&",i) = stepgrid(i-1);

                        for ( j = 0 ; j < ddim ; ++j )
                        {
                            ++(stepgrid("&",i)("&",j));

                            if ( stepgrid(i)(j) < kstp(j) )
                            {
                                break;
                            }

                            else
                            {
                                stepgrid("&",i)("&",j) = 0;
                            }
                        }
                    }
                }
            }

            else
            {
                // Zoom grid around current optimal

                i = bestind;

                for ( j = 0 ; j < ddim ; ++j )
                {
                    double ublbdiff = zoomfactor*(kmax(j)-kmin(j)); // new width should be 0.3*old width
                    double midpoint = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));

                    double newkmin = midpoint-(ublbdiff/2);
                    double newkmax = midpoint+(ublbdiff/2);

                    kmin("&",j) = ( newkmin > kmin(j) ) ? newkmin : kmin(j);
                    kmax("&",j) = ( newkmax < kmax(j) ) ? newkmax : kmax(j);
                }
            }

            // Work out results on all of grid

            gridres = 0.0;

//            double gridresmax = 0.0;
//            double gridresmin = 0.0;

            for ( i = 0 ; i < adim ; ++i )
            {
//                L1norm("&",i) = 0.0;
                weightval = 1.0;

                for ( j = 0 ; j < ddim ; ++j )
                {
                    if ( kelm(j) >= 0 )
                    {
                        constVecs("&",kind(j))("&",kelm(j)) = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                    }

                    else if ( kelm(j) == -1 )
                    {
                        weightval("&",kind(j))              = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                    }

                    else if ( kelm(j) == -2 )
                    {
                        Cval                                = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                    }

                    else if ( kelm(j) == -3 )
                    {
                        epsval                              = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                    }

//                    L1norm("&",i) += (stepgrid(i)(j)/((double) kstp(j)-1))*(stepgrid(i)(j)/((double) kstp(j)-1));
                }

                for ( j = 0 ; j < kdim ; ++j )
                {
                    gentype wv(weightval(j));

                    kernel.setRealConstants(constVecs(j),j);
                    kernel.setWeight(wv,j);
                }

                model.resetKernel();

                if ( tuneP & 1 )
                {
                    setC(Cval);
                }

                if ( tuneP & 2 )
                {
                    seteps(epsval);
                }

                model.train(dummy);

                if ( method == 1 )
                {
                    gridres("&",i) = calcnegloglikelihood(model,1);
//gridres("&",i) = (((double) (N()-1))*calcnegloglikelihood(model,1)/((double) N())) + (calcRecall(model,0,1)/((double) N()));
                }

                else if ( method == 2 )
                {
                    gridres("&",i) = calcLOO(model,0,1);
                }

                else if ( method == 3 )
                {
                    gridres("&",i) = calcRecall(model,0,1);
                }

//                if ( !i )
//                {
//                    gridresmax = gridres(i);
//                    gridresmin = gridres(i);
//                }
//
//                else
//                {
//                    if ( gridres(i) > gridresmax )
//                    {
//                        gridresmax = gridres(i);
//                    }
//
//                    if ( gridres(i) < gridresmin )
//                    {
//                        gridresmin = gridres(i);
//                    }
//                }

//                for ( j = 0 ; j < ddim ; ++j )
//                {
//                    if ( islen(j) )
//                    {
//                        gridres("&",i) += (constVecs(kind(j))(kelm(j))-kmin(j))/((kmax(j)-kmin(j));
//                    }
//                }
//errstream() << "Tuning kernel: weight " << weightval << ", const " << constVecs << " = " << gridres(i) << "\n";
//errstream() << gridres(i) << "(" << i << "), ";
            }

            // Dynamic regularisation to prevent overfit, particularly on lengthscale
            // Note that regularisation scales to a fraction of the current result range

/*
            double resrange = std::abs(gridresmax-gridresmin);
            double lambda = -1;

            for ( i = 0 ; i < adim ; ++i )
            {
                if ( L1norm(i) > 0 )
                {
                    double loclambda = resrange/L1norm(i);

                    if ( ( lambda < 0 ) || ( loclambda < lambda ) )
                    {
                        lambda = loclambda;
                    }
                }
            }

            double regul_const = PARA_REGUL*lambda;

            for ( i = 0 ; i < adim ; ++i )
            {
                gridres("&",i) += regul_const*L1norm(i);
            }
*/

//errstream() << "\n";
            // Find best result index

            bestres = min(gridres,bestind);
        }

        // Set kernel params to best result

        i = bestind;
        {
            for ( j = 0 ; j < ddim ; ++j )
            {
                if ( kelm(j) >= 0 )
                {
                    constVecs("&",kind(j))("&",kelm(j)) = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                }

                else if ( kelm(j) == -1 )
                {
                    weightval("&",kind(j))              = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                }

                else if ( kelm(j) == -2 )
                {
                    Cval                                = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                }

                else if ( kelm(j) == -3 )
                {
                    epsval                              = kmin(j) + ((kmax(j)-kmin(j))*stepgrid(i)(j)/((double) kstp(j)-1));
                }
            }

//            weightval /= abs2(weightval);
//            weightval *= sqrt((double) kdim);

//errstream() << "LOO goodset params " << constVecs << "\n";
//errstream() << "LOO goodset weights " << weightval << "\n";
            for ( j = 0 ; j < kdim ; ++j )
            {
                gentype wv(weightval(j));

                kernel.setRealConstants(constVecs(j),j);
                kernel.setWeight(wv,j);
            }

//errstream() << "Test: " << model << "\n";
            model.resetKernel();

            if ( tuneP & 1 )
            {
                setC(Cval);
            }

            if ( tuneP & 2 )
            {
                seteps(epsval);
            }

            model.train(dummy);
        }
    }

    return bestres;
}












void ML_Base::calcprior(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainfo) const
{
    if ( xmuprior == 1 )
    {
        if ( !xainfo )
        {
            vecInfo xinfox;

            getKernel().getvecInfo(xinfox,xa);

            calcprior(res,xa,&xinfox);

            return;
        }

        const SparseVector<gentype> *xanear = nullptr; // rank left
        const SparseVector<gentype> *xafar  = nullptr; // rank right

        double arankL = 1; // rank left weight
        double arankR = 1; // rank right weight

        const SparseVector<gentype> *xafarfar    = nullptr; // grad left
        const SparseVector<gentype> *xafarfarfar = nullptr; // grad right

        int agradOrderL; // grad order left
        int agradOrderR; // grad order right

        const vecInfo *xanearinfo = nullptr; // rank left
        const vecInfo *xafarinfo  = nullptr; // rank right

        // unused

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        int ixa;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        int iia,xalr,xarr,xagr,iaokr,iaok,adiagr,iaplanr,iaplan,iaset,iadenseint,iadensederiv;

        int xagrR,agmuL,agmuR;

        double iadiagoffset = 0;
        int iavectset = 0;

        // Detangling step

        int ia = -1;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,&xa,xainfo,agradOrderL,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        NiceAssert( !(loctanga & 2048 ) );
        NiceAssert( !(loctanga & 2048 ) );

        (void) loctanga;

        res = 0.0_gent;

    // 128:  xfarfar refers to direction of gradient constraint, gradOrder > 0
    // 256:  gradOrder > 0, but xfarfar not set
    // 512:  xfarfarfar refers to direction of gradient constraint, gradOrderR > 0
    // 1024: gradOrderR > 0, but xfarfarfar not set


        if ( !( loctanga & (128+256) ) && xanear )
        {
            // Left rank without gradient

            gentype xargs(*xanear);
//outstream() << "gentype pycall xargs " << xargs << "\n";
            gentype locres = xmuprior_gt(xargs);

//outstream() << "gentype pycall locres " << locres << "\n";
            locres *= arankL;
            res += locres;
//outstream() << "gentype pycall res " << res << "\n";
        }

        else if ( xanear )
        {
            // Left rank with gradient

            NiceThrow("Need to implement prior gradients!");
        }

        if ( ( loctanga & 1 ) && !( loctanga & (512+1024) ) && xafar )
        {
            // Right rank without gradient

            gentype xargs(*xafar);
            gentype locres = xmuprior_gt(xargs);
//outstream() << "gentype pycall locres " << locres << " alt\n";

            locres *= arankR;
            res -= locres;
//outstream() << "gentype pycall res " << res << " alt\n";
        }

        if ( ( loctanga & 1 ) && xafar )
        {
            // Right rank with gradient

            NiceThrow("Need to implement prior gradients!");
        }
//outstream() << "gentype pycall res " << res << " later\n";
    }

    else if ( xmuprior == 2 )
    {
        xmuprior_ml->gg(res,xa,xainfo);
    }

    else
    {
        res = 0.0_gent;
    }

//outstream() << "gentype pycall res " << res << " final\n";
    return;
}

void ML_Base::calcallprior(void)
{
    static thread_local Vector<double> empvec;

    for ( int i = 0 ; i < ML_Base::N() ; i++ )
    {
        calcprior(alltraintargp("&",i),x(i));
//outstream() << "gentype pycall doneprior " << alltraintargp(i) << " final\n";
        alltraintargpR("&",i) = ( gOutType() == 'R' ) ? ( (double) alltraintargp(i) ) : 0.0;
        alltraintargpA("&",i) = ( gOutType() == 'A' ) ? ( (const d_anion &) alltraintargp(i) ) : 0_gent;
        alltraintargpV("&",i) = ( gOutType() == 'V' ) ? ( (const Vector<double> &) alltraintargp(i) ) : empvec;
    }
}










gentype &ML_Base::K0(gentype &res, 
                     const gentype **pxyprod, 
                     int resmode) const
{
    //return K0(res,zerogentype(),getKernel(),pxyprod,resmode);
    return K0(res,0.0_gent,getKernel(),pxyprod,resmode);
}

double ML_Base::K0(const gentype **pxyprod, 
                   int resmode) const
{
    return K0(0.0,getKernel(),pxyprod,resmode);
}

gentype &ML_Base::K0(gentype &res, 
                     const gentype &bias, const gentype **pxyprod, 
                     int resmode) const
{
    return K0(res,bias,getKernel(),pxyprod,resmode);
}

gentype &ML_Base::K0(gentype &res, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     int resmode) const
{
    //return K0(res,zerogentype(),altK,pxyprod,resmode);
    return K0(res,0.0_gent,altK,pxyprod,resmode);
}

Matrix<double> &ML_Base::K0(int spaceDim, Matrix<double> &res, 
                            const gentype **pxyprod, 
                            int resmode) const
{
    gentype tempres;

    K0(tempres,pxyprod,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K0(int order, d_anion &res, 
                     const gentype **pxyprod, 
                     int resmode) const
{
    gentype tempres;

    K0(tempres,pxyprod,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K1(gentype &res, 
                     int ia, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
//    return K1(res,ia,zerogentype(),getKernel(),pxyprod,xa,xainfo,resmode);
    return K1(res,ia,0.0_gent,getKernel(),pxyprod,xa,xainfo,resmode);
}

double ML_Base::K1(int ia, 
                   const gentype **pxyprod, 
                   const SparseVector<gentype> *xa, 
                   const vecInfo *xainfo, 
                   int resmode) const
{
    return K1(ia,0.0,getKernel(),pxyprod,xa,xainfo,resmode);
}

gentype &ML_Base::K1(gentype &res, 
                     int ia, 
                     const gentype &bias, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
    return K1(res,ia,bias,getKernel(),pxyprod,xa,xainfo,resmode);
}

gentype &ML_Base::K1(gentype &res, 
                     int ia, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
//    return K1(res,ia,zerogentype(),altK,pxyprod,xa,xainfo,resmode);
    return K1(res,ia,0.0_gent,altK,pxyprod,xa,xainfo,resmode);
}

Matrix<double> &ML_Base::K1(int spaceDim, Matrix<double> &res, 
                            int ia, 
                            const gentype **pxyprod, 
                            const SparseVector<gentype> *xa, 
                            const vecInfo *xainfo, 
                            int resmode) const
{
    gentype tempres;

    K1(tempres,ia,pxyprod,xa,xainfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K1(int order, d_anion &res, 
                     int ia, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
    gentype tempres;

    K1(tempres,ia,pxyprod,xa,xainfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K2(gentype &res,
                    int ib, int jb,
                    const gentype **pxyprod,
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy,
                    const vecInfo *xxinfo, const vecInfo *yyinfo,
                    int resmode) const
{
//    return K2(res,ib,jb,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    return K2(res,ib,jb,0.0_gent,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,resmode);
}

gentype &ML_Base::K2(gentype &res,
                    int ib, int jb,
                    const gentype &bias, const gentype **pxyprod,
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy,
                    const vecInfo *xxinfo, const vecInfo *yyinfo,
                    int resmode) const
{
    return K2(res,ib,jb,bias,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,resmode);
}

double ML_Base::K2(int ib, int jb,
                   const gentype **pxyprod,
                   const SparseVector<gentype> *xx, const SparseVector<gentype> *yy,
                   const vecInfo *xxinfo, const vecInfo *yyinfo,
                   int resmode) const
{
    return K2(ib,jb,0.0,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,resmode);
}

gentype &ML_Base::K2(gentype &res,
                    int ib, int jb,
                    const MercerKernel &altK, const gentype **pxyprod,
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy,
                    const vecInfo *xxinfo, const vecInfo *yyinfo,
                    int resmode) const
{
//    return K2(res,ib,jb,zerogentype(),altK,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    return K2(res,ib,jb,0.0_gent,altK,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
}

Matrix<double> &ML_Base::K2(int spaceDim, Matrix<double> &res,
                           int i, int j,
                           const gentype **pxyprod,
                           const SparseVector<gentype> *xx, const SparseVector<gentype> *yy,
                           const vecInfo *xxinfo, const vecInfo *yyinfo,
                           int resmode) const
{
    gentype tempres;

    K2(tempres,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K2(int order, d_anion &res,
                    int i, int j,
                    const gentype **pxyprod,
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy,
                    const vecInfo *xxinfo, const vecInfo *yyinfo,
                    int resmode) const
{
    gentype tempres;

    K2(tempres,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K2x2(gentype &res,
                       int i, int ia, int ib,
                       const SparseVector<gentype> *x, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
                       const vecInfo *xinfo, const vecInfo *xainfo, const vecInfo *xbinfo,
                       int resmode) const
{
//    return K2x2(res,i,ia,ib,zerogentype(),getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);
    return K2x2(res,i,ia,ib,0.0_gent,getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);
}

gentype &ML_Base::K2x2(gentype &res,
                       int i, int ia, int ib,
                       const gentype &bias,
                       const SparseVector<gentype> *x, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
                       const vecInfo *xinfo, const vecInfo *xainfo, const vecInfo *xbinfo,
                       int resmode) const
{
    return K2x2(res,i,ia,ib,bias,getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);
}

gentype &ML_Base::K2x2(gentype &res,
                       int i, int ia, int ib,
                       const MercerKernel &altK,
                       const SparseVector<gentype> *x, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
                       const vecInfo *xinfo, const vecInfo *xainfo, const vecInfo *xbinfo,
                       int resmode) const
{
//    return K2x2(res,i,ia,ib,zerogentype(),altK,x,xa,xb,xinfo,xainfo,xbinfo,resmode);
    return K2x2(res,i,ia,ib,0.0_gent,altK,x,xa,xb,xinfo,xainfo,xbinfo,resmode);
}

double ML_Base::K2x2(int i, int ia, int ib,
                      const SparseVector<gentype> *x, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
                      const vecInfo *xinfo, const vecInfo *xainfo, const vecInfo *xbinfo,
                      int resmode) const
{
    return K2x2(i,ia,ib,0,getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);
}

Matrix<double> &ML_Base::K2x2(int spaceDim, Matrix<double> &res,
                              int i, int ia, int ib,
                              const SparseVector<gentype> *x, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
                              const vecInfo *xinfo, const vecInfo *xainfo, const vecInfo *xbinfo,
                              int resmode) const
{
    gentype tempres;

//    K2x2(tempres,i,ia,ib,zerogentype(),getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);
    K2x2(tempres,i,ia,ib,0.0_gent,getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K2x2(int order, d_anion &res,
                       int i, int ia, int ib,
                       const SparseVector<gentype> *x, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
                       const vecInfo *xinfo, const vecInfo *xainfo, const vecInfo *xbinfo,
                       int resmode) const
{
    gentype tempres;

//    K2x2(tempres,i,ia,ib,zerogentype(),getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);
    K2x2(tempres,i,ia,ib,0.0_gent,getKernel(),x,xa,xb,xinfo,xainfo,xbinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K3(gentype &res, 
                     int ia, int ib, int ic, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
//    return K3(res,ia,ib,ic,zerogentype(),getKernel(),pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);
    return K3(res,ia,ib,ic,0.0_gent,getKernel(),pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);
}

double ML_Base::K3(int ia, int ib, int ic, 
                   const gentype **pxyprod, 
                   const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                   const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                   int resmode) const
{
    return K3(ia,ib,ic,0.0,getKernel(),pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);
}

gentype &ML_Base::K3(gentype &res, 
                     int ia, int ib, int ic, 
                     const gentype &bias, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
    return K3(res,ia,ib,ic,bias,getKernel(),pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);
}

gentype &ML_Base::K3(gentype &res, 
                     int ia, int ib, int ic, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
//    return K3(res,ia,ib,ic,zerogentype(),altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);
    return K3(res,ia,ib,ic,0.0_gent,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);
}

Matrix<double> &ML_Base::K3(int spaceDim, Matrix<double> &res, 
                            int ia, int ib, int ic, 
                            const gentype **pxyprod, 
                            const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                            const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                            int resmode) const
{
    gentype tempres;

    K3(tempres,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K3(int order, d_anion &res, 
                     int ia, int ib, int ic, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
    gentype tempres;

    K3(tempres,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K4(gentype &res, 
                     int ia, int ib, int ic, int id, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
//    return K4(res,ia,ib,ic,id,zerogentype(),getKernel(),pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);
    return K4(res,ia,ib,ic,id,0.0_gent,getKernel(),pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);
}

double ML_Base::K4(int ia, int ib, int ic, int id, 
                   const gentype **pxyprod, 
                   const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                   const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                   int resmode) const
{
    return K4(ia,ib,ic,id,0.0,getKernel(),pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);
}

gentype &ML_Base::K4(gentype &res, 
                     int ia, int ib, int ic, int id,
                     const gentype &bias, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
    return K4(res,ia,ib,ic,id,bias,getKernel(),pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);
}

gentype &ML_Base::K4(gentype &res, 
                     int ia, int ib, int ic, int id, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
//    return K4(res,ia,ib,ic,id,zerogentype(),altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);
    return K4(res,ia,ib,ic,id,0.0_gent,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);
}

Matrix<double> &ML_Base::K4(int spaceDim, Matrix<double> &res, 
                            int ia, int ib, int ic, int id, 
                            const gentype **pxyprod, 
                            const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                            const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                            int resmode) const
{
    gentype tempres;

    K4(tempres,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K4(int order, d_anion &res, 
                     int ia, int ib, int ic, int id, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
    gentype tempres;

    K4(tempres,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::Km(int m, gentype &res, 
                     Vector<int> &i, 
                     const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xxx, 
                     Vector<const vecInfo *> *xxxinfo, 
                     int resmode) const
{
//    return Km(m,res,i,zerogentype(),getKernel(),pxyprod,xxx,xxxinfo,resmode);
    return Km(m,res,i,0.0_gent,getKernel(),pxyprod,xxx,xxxinfo,resmode);
}

gentype &ML_Base::Km(int m, gentype &res, 
                     Vector<int> &i, 
                     const gentype &bias, const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xxx, 
                     Vector<const vecInfo *> *xxxinfo, 
                     int resmode) const
{
    return Km(m,res,i,bias,getKernel(),pxyprod,xxx,xxxinfo,resmode);
}

gentype &ML_Base::Km(int m, gentype &res, 
                     Vector<int> &i, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xxx, 
                     Vector<const vecInfo *> *xxxinfo, 
                     int resmode) const
{
//    return Km(m,res,i,zerogentype(),altK,pxyprod,xxx,xxxinfo,resmode);
    return Km(m,res,i,0.0_gent,altK,pxyprod,xxx,xxxinfo,resmode);
}

double ML_Base::Km(int m,
                    Vector<int> &i,
                    const gentype **pxyprod,
                    Vector<const SparseVector<gentype> *> *xxx,
                    Vector<const vecInfo *> *xxxinfo,
                    int resmode) const
{
    return Km(m,i,0.0,getKernel(),pxyprod,xxx,xxxinfo,resmode);
}

Matrix<double> &ML_Base::Km(int m, int spaceDim, Matrix<double> &res, 
                            Vector<int> &i, 
                            const gentype **pxyprod, 
                            Vector<const SparseVector<gentype> *> *xx, 
                            Vector<const vecInfo *> *xxinfo, 
                            int resmode) const
{
    gentype tempres;

    Km(m,tempres,i,pxyprod,xx,xxinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::Km(int m, int order, d_anion &res, 
                     Vector<int> &i, 
                     const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xx, 
                     Vector<const vecInfo *> *xxinfo, 
                     int resmode) const
{
    gentype tempres;

    Km(m,tempres,i,pxyprod,xx,xxinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

void ML_Base::dK(gentype &xygrad, gentype &xnormgrad, 
                 int ib, int jb, 
                 const gentype **pxyprod, 
                 const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                 const vecInfo *xxinfo, const vecInfo *yyinfo, 
                 int deepDeriv) const
{
//    dK(xygrad,xnormgrad,ib,jb,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv);
    dK(xygrad,xnormgrad,ib,jb,0.0_gent,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv);

    return;
}

void ML_Base::dK(double &xygrad, double &xnormgrad, 
                 int ib, int jb, 
                 const gentype **pxyprod, 
                 const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                 const vecInfo *xxinfo, const vecInfo *yyinfo, 
                 int deepDeriv) const
{
    dK(xygrad,xnormgrad,ib,jb,0.0,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv);

    return;
}

void ML_Base::dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, 
                     int ib, int jb, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
//    dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);
    dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,0.0_gent,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::dK2delx(double &xscaleres, double &yscaleres, int &minmaxind, 
                     int ib, int jb, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,0.0,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, 
                  int i, int j, 
                  const gentype **pxyprod, 
                  const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                  const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
//    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);
    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,0.0_gent,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, 
                  int i, int j, 
                  const gentype **pxyprod, 
                  const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                  const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,0.0,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
//    d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);
    d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,0.0_gent,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
//    d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);
    d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,0.0_gent,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,0.0,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,0.0,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                     const Vector<int> &q, 
                     int i, int j, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
//    dnK2del(sc,n,minmaxind,q,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);
    dnK2del(sc,n,minmaxind,q,i,j,0.0_gent,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}

void ML_Base::dnK2del(Vector<double> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                     const Vector<int> &q, 
                     int i, int j, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    dnK2del(sc,n,minmaxind,q,i,j,0.0,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo);

    return;
}






// get unique (kinda, cycling) pseudo non-training-vector index between -10 and -90

//int UPNTVI(int i, int off);
//int UPNTVI(int i, int off)
//{
////    static size_t ind = 0;
////
////    return -(((ind++)%81)+10);
//    return -((10*off)+i);
//}

// return 1 if index indicates vector in training set, 0 otherwise

int istrv(int i);
int istrv(int i)
{
    return ( i >= 0 ) || ( i <= -100 );
    //return ( i >= 0 ) || ( i <= -10 );
}

int istrv(const Vector<int> &i);
int istrv(const Vector<int> &i)
{
    //NB: this won't work exactly in the case where there
    //    is a mix of points >=0 and <=-100, but in
    //    general its good enough (the result of the above
    //    case is just slow operation, not incorrect operation).

    return ( i >= 0 ) || ( i <= -100 );
    //return ( i >= 0 ) || ( i <= -10 );
}






gentype &ML_Base::Keqn(gentype &res, int resmode) const
{
    return Keqn(res,getKernel(),resmode);
}

gentype &ML_Base::Keqn(gentype &res, const MercerKernel &altK, int resmode) const
{
    return altK.Keqn(res,resmode);
}






































double ML_Base::K0(double bias, const MercerKernel &altK, const gentype **pxyprod,
                   int resmode) const
{
//phantomx
    return altK.K0(bias,pxyprod,xspaceDim(),isXConsistent(),resmode,MLid(),assumeReal);
}

template <class T>
T &ML_Base::K0(T &res,
               const T &bias, const MercerKernel &altK, const gentype **pxyprod,
               int resmode) const
{
//phantomx
    altK.K0(res,bias,pxyprod,xspaceDim(),isXConsistent(),resmode,MLid(),assumeReal);

    return res;
}

double ML_Base::K1(int ia,
               double bias, const MercerKernel &basealtK, const gentype **pxyprod,
               const SparseVector<gentype> *xa,
               const vecInfo *xainfo,
               int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) ) ? getRFFKernel() : basealtK );

    double res = 0;

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        res = K1(ia,bias,basealtK,pxyprod,xa,&xinfox,resmode);
    }

    else if ( xa->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;

        if ( xa && isxymat(altK) && ( ia >= 0 ) && ( (*xa).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
        }

        res = altK.K1(*xa,*xainfo,bias,pxyprod,ia,xspaceDim(),isXConsistent() && istrv(ia),resmode,MLid(),x00,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear      = nullptr;
        const SparseVector<gentype> *xafar       = nullptr;
        const SparseVector<gentype> *xafarfar    = nullptr;
        const SparseVector<gentype> *xafarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xafarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        double arankL,arankR;
        int xagrR,agradOrderR,agmuL,agmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        int iavectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K1(sumind(i),bias,basealtK,nullptr,nullptr,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K1(sumind(i),bias,basealtK,nullptr,nullptr,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

//        NiceAssert( !iadenseint && !iadensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        res = altK.K1(*xai,*xainfoi,bias,nullptr,ia,xspaceDim(),isXConsistent() && istrv(ia),resmode,MLid(),nullptr,assumeReal);

        res += iadiagoffset;

        if ( xauntang ) { delete xauntang; }
        if ( xainfountang ) { delete xainfountang; }

        if ( iaokr )
        {
            Vector<int> iiokr(1);
            Vector<int> iiok(1);
            Vector<const gentype *> xxalt(1);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,1,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (double) UUres;
        }

        if ( iaplanr )
        {
            Vector<int> iiplanr(1);
            Vector<int> iiplan(1);
            Vector<const gentype *> xxalt(1);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,1,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (double) VVres;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

template <class T>
T &ML_Base::K1(T &res,
               int ia,
               const T &bias, const MercerKernel &basealtK, const gentype **pxyprod,
               const SparseVector<gentype> *xa,
               const vecInfo *xainfo,
               int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) ) ? getRFFKernel() : basealtK );

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        K1(res,ia,bias,basealtK,pxyprod,xa,&xinfox,resmode);
    }

    else if ( xa->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;

        if ( xa && isxymat(altK) && ( ia >= 0 ) && ( (*xa).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
        }

        altK.K1(res,*xa,*xainfo,bias,pxyprod,ia,xspaceDim(),isXConsistent() && istrv(ia),resmode,MLid(),x00,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear      = nullptr;
        const SparseVector<gentype> *xafar       = nullptr;
        const SparseVector<gentype> *xafarfar    = nullptr;
        const SparseVector<gentype> *xafarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xafarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        double arankL,arankR;
        int xagrR,agradOrderR,agmuL,agmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        int iavectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K1(res,sumind(i),bias,basealtK,nullptr,nullptr,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K1(restmp,sumind(i),bias,basealtK,nullptr,nullptr,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

//        NiceAssert( !iadenseint && !iadensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        altK.K1(res,*xai,*xainfoi,bias,nullptr,ia,xspaceDim(),isXConsistent() && istrv(ia),resmode,MLid(),nullptr,assumeReal);

        res += iadiagoffset;

        if ( xauntang ) { delete xauntang; }
        if ( xainfountang ) { delete xainfountang; }

        if ( iaokr )
        {
            Vector<int> iiokr(1);
            Vector<int> iiok(1);
            Vector<const gentype *> xxalt(1);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,1,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (T) UUres;
        }

        if ( iaplanr )
        {
            Vector<int> iiplanr(1);
            Vector<int> iiplan(1);
            Vector<const gentype *> xxalt(1);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,1,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (T) VVres;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

Vector<gentype> &ML_Base::phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa, const vecInfo *xainf) const
{
    if ( xa && !xainf )
    {
        vecInfo xinfoia;

        getKernel().getvecInfo(xinfoia,*xa);

        phi2(res,ia,xa,&xinfoia);
    }

    else if ( !xa )
    {
        const SparseVector<gentype> &xia = x(ia);
        const vecInfo &xinfoia = xinfo(ia);

        phi2(res,ia,&xia,&xinfoia);
    }

    else
    {
        getKernel().phi2(res,*xa,*xainf,ia,1,xspaceDim(),isXConsistent() && istrv(ia),assumeReal);
    }

    return res;
}

Vector<double> &ML_Base::phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa, const vecInfo *xainf) const
{
    if ( xa && !xainf )
    {
        vecInfo xinfoia;

        getKernel().getvecInfo(xinfoia,*xa);

        phi2(res,ia,xa,&xinfoia);
    }

    else if ( !xa )
    {
        const SparseVector<gentype> &xia = x(ia);
        const vecInfo &xinfoia = xinfo(ia);

        phi2(res,ia,&xia,&xinfoia);
    }

    else
    {
        getKernel().phi2(res,*xa,*xainf,ia,1,xspaceDim(),isXConsistent() && istrv(ia),assumeReal);
    }

    return res;
}

double ML_Base::K2x2(int ia, int ib, int ic,
                 double bias, const MercerKernel &basealtK,
                 const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc,
                 const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo,
                 int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) ) ? getRFFKernel() : basealtK );

    double res = 0;

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xc )
    {
        xc     = &x(ic);
        xcinfo = &xinfo(ic);
    }

    if ( !xainfo && !xbinfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K2x2(ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,&xinfob,&xinfoc,resmode);
    }

    else if ( !xbinfo && !xcinfo )
    {
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K2x2(ia,ib,ic,bias,basealtK,xa,xb,xc,xainfo,&xinfob,&xinfoc,resmode);
    }

    else if ( !xainfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K2x2(ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,xbinfo,&xinfoc,resmode);
    }

    else if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);

        res = K2x2(ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,&xinfob,xcinfo,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfoa;

        getKernel().getvecInfo(xinfoa,*xa);

        res = K2x2(ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,xbinfo,xcinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfob;

        getKernel().getvecInfo(xinfob,*xb);

        res = K2x2(ia,ib,ic,bias,basealtK,xa,xb,xc,xainfo,&xinfob,xcinfo,resmode);
    }

    else if ( !xcinfo )
    {
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoc,*xc);

        res = K2x2(ia,ib,ic,bias,basealtK,xa,xb,xc,xainfo,xbinfo,&xinfoc,resmode);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;
        const double *x20 = nullptr;
        //const double *x21 = nullptr;
        const double *x22 = nullptr;

        if ( xa && xb && xc && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) && ( (*xc).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
            x20 = &(getxymatelm(altK,ic,ia));
            //x21 = &(getxymatelm(altK,ic,ib));
            x22 = &(getxymatelm(altK,ic,ic));
        }

        double resb;

        res  = altK.K2(*xa,*xb,*xainfo,*xbinfo,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),x00,x10,x11,assumeReal);
        resb = altK.K2(*xa,*xc,*xainfo,*xcinfo,bias,nullptr,ia,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ic),resmode,MLid(),x00,x20,x22,assumeReal);

        res *= resb;
//        altK.K2x2(res,*xa,*xb,*xc,*xainfo,*xbinfo,*xcinfo,bias,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),x00,x10,x11,x20,x21,x22,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;
        const SparseVector<gentype> *xcnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;
        const SparseVector<gentype> *xcfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;
        const SparseVector<gentype> *xcfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;
        const SparseVector<gentype> *xcfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;
        const vecInfo *xcnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;
        const vecInfo *xcfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;
        int ixc,iic,xclr,xcrr,xcgr,icokr,icok,cdiagr,cgradOrder,icplanr,icplan,icset,icdenseint,icdensederiv;

        double arankL,arankR;
        double brankL,brankR;
        double crankL,crankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;
        int xcgrR,cgradOrderR,cgmuL,cgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        const gentype *ixctup = nullptr;
        const gentype *iictup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;
        SparseVector<gentype> *xcuntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;
        vecInfo *xcinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;
        double icdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;
        int icvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K2x2(sumind(i),ib,ic,bias,basealtK,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K2x2(sumind(i),ib,ic,bias,basealtK,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K2x2(ia,sumind(i),ic,bias,basealtK,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K2x2(ia,sumind(i),ic,bias,basealtK,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangc = detangle_x(xcuntang,xcinfountang,xcnear,xcfar,xcfarfar,xcfarfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,xclr,xcrr,xcgr,xcgrR,icokr,icok,crankL,crankR,cgmuL,cgmuR,ic,cdiagr,xc,xcinfo,cgradOrder,cgradOrderR,icplanr,icplan,icset,icdenseint,icdensederiv,sumind,sumweight,icdiagoffset,icvectset);

        if ( loctangc & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K2x2(ia,ib,sumind(i),bias,basealtK,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K2x2(ia,ib,sumind(i),bias,basealtK,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( ( iavectset == ibvectset ) && ( iavectset && icvectset ) ) ? 1 : 0;

//        NiceAssert( !iadenseint && !iadensederiv );
//        NiceAssert( !ibdenseint && !ibdensederiv );
//        NiceAssert( !icdenseint && !icdensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const SparseVector<gentype> *xci = xcuntang ? xcuntang : xc;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;
        const vecInfo *xcinfoi = xcinfountang ? xcinfountang : xcinfo;

        if ( issameset && !iadenseint && !iadensederiv )
        {
             double resb;

             res  = altK.K2(*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),nullptr,nullptr,nullptr,assumeReal);
             resb = altK.K2(*xai,*xci,*xainfoi,*xcinfoi,bias,nullptr,ia,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ic),resmode,MLid(),nullptr,nullptr,nullptr,assumeReal);

             res *= resb;
        }

        else if ( issameset )
        {
            res = altK.K2x2(*xai,*xbi,*xci,*xainfoi,*xbinfoi,*xcinfoi,bias,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumeReal);
        }

        else
        {
            res = 0.0;
        }

        if ( ( ia == ib ) && ( ia == ic ) )
        {
            res += iadiagoffset;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }
        if ( xcuntang ) { delete xcuntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }
        if ( xcinfountang ) { delete xcinfountang; }

        if ( iaokr || ibokr || icokr )
        {
            // Remember that this is a product of kernels!

            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            {
                gentype UUres;

                (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

                res *= (double) UUres;
            }

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = icokr;
            iiok("&",1)  = icok;
            xxalt("&",1) = (*xc).isf4indpresent(3) ? &((*xc).f4(3)) : &nullgentype();

            {
                gentype UUres;

                (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

                res *= (double) UUres;
            }
        }

        if ( iaplanr || ibplanr || icplanr )
        {
            // Remember that this is a product of kernels!

            Vector<int> iiplanr(2);
            Vector<int> iiplan(2);
            Vector<const gentype *> xxalt(2);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            gentype kvalorig(res);

            {
                gentype VVres;
                gentype kval(kvalorig);

                (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

                res = VVres;
            }

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = icplanr;
            iiplan("&",1)  = icplan;
            xxalt("&",1)   = (*xc).isf4indpresent(7) ? &((*xc).f4(7)) : &nullgentype();

            {
                gentype VVres;
                gentype kval(kvalorig);

                (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

                res = VVres;
            }
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}


template <class T>
T &ML_Base::K2x2(T &res,
                 int ia, int ib, int ic,
                 const T &bias, const MercerKernel &basealtK,
                 const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc,
                 const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo,
                 int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) ) ? getRFFKernel() : basealtK );

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xc )
    {
        xc     = &x(ic);
        xcinfo = &xinfo(ic);
    }

    if ( !xainfo && !xbinfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        K2x2(res,ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,&xinfob,&xinfoc,resmode);
    }

    else if ( !xbinfo && !xcinfo )
    {
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        K2x2(res,ia,ib,ic,bias,basealtK,xa,xb,xc,xainfo,&xinfob,&xinfoc,resmode);
    }

    else if ( !xainfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);

        K2x2(res,ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,xbinfo,&xinfoc,resmode);
    }

    else if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);

        K2x2(res,ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,&xinfob,xcinfo,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfoa;

        getKernel().getvecInfo(xinfoa,*xa);

        K2x2(res,ia,ib,ic,bias,basealtK,xa,xb,xc,&xinfoa,xbinfo,xcinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfob;

        getKernel().getvecInfo(xinfob,*xb);

        K2x2(res,ia,ib,ic,bias,basealtK,xa,xb,xc,xainfo,&xinfob,xcinfo,resmode);
    }

    else if ( !xcinfo )
    {
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoc,*xc);

        K2x2(res,ia,ib,ic,bias,basealtK,xa,xb,xc,xainfo,xbinfo,&xinfoc,resmode);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;
        const double *x20 = nullptr;
        //const double *x21 = nullptr;
        const double *x22 = nullptr;

        if ( xa && xb && xc && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) && ( (*xc).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
            x20 = &(getxymatelm(altK,ic,ia));
            //x21 = &(getxymatelm(altK,ic,ib));
            x22 = &(getxymatelm(altK,ic,ic));
        }

        T resb;

        altK.K2(res ,*xa,*xb,*xainfo,*xbinfo,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),x00,x10,x11,assumeReal);
        altK.K2(resb,*xa,*xc,*xainfo,*xcinfo,bias,nullptr,ia,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ic),resmode,MLid(),x00,x20,x22,assumeReal);

        res *= resb;
//        altK.K2x2(res,*xa,*xb,*xc,*xainfo,*xbinfo,*xcinfo,bias,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),x00,x10,x11,x20,x21,x22,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;
        const SparseVector<gentype> *xcnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;
        const SparseVector<gentype> *xcfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;
        const SparseVector<gentype> *xcfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;
        const SparseVector<gentype> *xcfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;
        const vecInfo *xcnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;
        const vecInfo *xcfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;
        int ixc,iic,xclr,xcrr,xcgr,icokr,icok,cdiagr,cgradOrder,icplanr,icplan,icset,icdenseint,icdensederiv;

        double arankL,arankR;
        double brankL,brankR;
        double crankL,crankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;
        int xcgrR,cgradOrderR,cgmuL,cgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        const gentype *ixctup = nullptr;
        const gentype *iictup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;
        SparseVector<gentype> *xcuntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;
        vecInfo *xcinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;
        double icdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;
        int icvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K2x2(res,sumind(i),ib,ic,bias,basealtK,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K2x2(restmp,sumind(i),ib,ic,bias,basealtK,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K2x2(res,ia,sumind(i),ic,bias,basealtK,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K2x2(restmp,ia,sumind(i),ic,bias,basealtK,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangc = detangle_x(xcuntang,xcinfountang,xcnear,xcfar,xcfarfar,xcfarfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,xclr,xcrr,xcgr,xcgrR,icokr,icok,crankL,crankR,cgmuL,cgmuR,ic,cdiagr,xc,xcinfo,cgradOrder,cgradOrderR,icplanr,icplan,icset,icdenseint,icdensederiv,sumind,sumweight,icdiagoffset,icvectset);

        if ( loctangc & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K2x2(res,ia,ib,sumind(i),bias,basealtK,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K2x2(restmp,ia,ib,sumind(i),bias,basealtK,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( ( iavectset == ibvectset ) && ( iavectset && icvectset ) ) ? 1 : 0;

//        NiceAssert( !iadenseint && !iadensederiv );
//        NiceAssert( !ibdenseint && !ibdensederiv );
//        NiceAssert( !icdenseint && !icdensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const SparseVector<gentype> *xci = xcuntang ? xcuntang : xc;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;
        const vecInfo *xcinfoi = xcinfountang ? xcinfountang : xcinfo;

        if ( issameset && !iadenseint && !iadensederiv )
        {
             T resb;

             altK.K2(res ,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),nullptr,nullptr,nullptr,assumeReal);
             altK.K2(resb,*xai,*xci,*xainfoi,*xcinfoi,bias,nullptr,ia,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ic),resmode,MLid(),nullptr,nullptr,nullptr,assumeReal);

             res *= resb;
        }

        else if ( issameset )
        {
            altK.K2x2(res,*xai,*xbi,*xci,*xainfoi,*xbinfoi,*xcinfoi,bias,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumeReal);
        }

        else
        {
            res = 0.0;
        }

        if ( ( ia == ib ) && ( ia == ic ) )
        {
            res += iadiagoffset;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }
        if ( xcuntang ) { delete xcuntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }
        if ( xcinfountang ) { delete xcinfountang; }

        if ( iaokr || ibokr || icokr )
        {
            // Remember that this is a product of kernels!

            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            {
                gentype UUres;

                (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

                res *= (T) UUres;
            }

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = icokr;
            iiok("&",1)  = icok;
            xxalt("&",1) = (*xc).isf4indpresent(3) ? &((*xc).f4(3)) : &nullgentype();

            {
                gentype UUres;

                (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

                res *= (T) UUres;
            }
        }

        if ( iaplanr || ibplanr || icplanr )
        {
            // Remember that this is a product of kernels!

            Vector<int> iiplanr(2);
            Vector<int> iiplan(2);
            Vector<const gentype *> xxalt(2);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            gentype kvalorig(res);

            {
                gentype VVres;
                gentype kval(kvalorig);

                (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

                res = (T) VVres;
            }

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = icplanr;
            iiplan("&",1)  = icplan;
            xxalt("&",1)   = (*xc).isf4indpresent(7) ? &((*xc).f4(7)) : &nullgentype();

            {
                gentype VVres;
                gentype kval(kvalorig);

                (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

                res = (T) VVres;
            }
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}


double ML_Base::K2(int ia, int ib,
              double bias, const MercerKernel &basealtK, const gentype **pxyprod,
              const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
              const vecInfo *xainfo, const vecInfo *xbinfo,
              int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

    double res = 0;

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

        res = K2(ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        res = K2(ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

        res = K2(ia,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy,resmode);
    }

    else if ( ( ia >= 0 ) && ( ib >= 0 ) && ( K2mat.numRows() ) && ( K2mat.numCols() ) )
    {
        res = K2mat(ia,ib);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        res = altK.K2(*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),x00,x10,x11,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K2(sumind(i),ib,bias,basealtK,nullptr,nullptr,xb,nullptr,xbinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K2(sumind(i),ib,bias,basealtK,nullptr,nullptr,xb,nullptr,xbinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K2(ia,sumind(i),bias,basealtK,nullptr,xa,nullptr,xainfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K2(ia,sumind(i),bias,basealtK,nullptr,xa,nullptr,xainfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        if ( issameset )
        {
            res = altK.K2(*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),nullptr,nullptr,nullptr,assumeReal);

if ( testisvnan(res) || testisinf(res) )
{
errstream() << "K2 evaluated null 1: " << res << "\n";
errstream() << "K2 evaluated null 2: " << *xa << "," << *xb << "\n";
errstream() << "K2 evaluated null 2a: " << *xai << "," << *xbi << "\n";
errstream() << "K2 evaluated null 2b: " << *xainfoi << "," << *xbinfoi << "\n";
errstream() << "K2 evaluated null 2c: " << bias << "\n";
errstream() << "K2 evaluated null 2d: " << ia << "," << ib << "\n";
errstream() << "K2 evaluated null 3: " << getKernel() << "\n";
}
        }

        else
        {
            res = 0.0;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }

        if ( ia == ib )
        {
            res += iadiagoffset;
        }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (double) UUres;
        }

        if ( iaplanr || ibplanr )
        {
            Vector<int> iiplanr(2);
            Vector<int> iiplan(2);
            Vector<const gentype *> xxalt(2);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (double) VVres;
        }
    }

if ( testisvnan(res) || testisinf(res) )
{
errstream() << "phantomxyggghhhqq 1: " << res << "\n";
errstream() << "phantomxyggghhhqq 2: " << *xa << "," << *xb << "\n";
errstream() << "phantomxyggghhhqq 3: " << getKernel() << "\n";
}

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

template <class T>
T &ML_Base::K2(T &res,
              int ia, int ib,
              const T &bias, const MercerKernel &basealtK, const gentype **pxyprod,
              const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,
              const vecInfo *xainfo, const vecInfo *xbinfo,
              int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

        K2(res,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        K2(res,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

        K2(res,ia,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy,resmode);
    }

    else if ( ( ia >= 0 ) && ( ib >= 0 ) && ( K2mat.numRows() ) && ( K2mat.numCols() ) )
    {
        res = (T) K2mat(ia,ib);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        altK.K2(res,*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),x00,x10,x11,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K2(res,sumind(i),ib,bias,basealtK,nullptr,nullptr,xb,nullptr,xbinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K2(restmp,sumind(i),ib,bias,basealtK,nullptr,nullptr,xb,nullptr,xbinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K2(res,ia,sumind(i),bias,basealtK,nullptr,xa,nullptr,xainfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K2(restmp,ia,sumind(i),bias,basealtK,nullptr,xa,nullptr,xainfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        if ( issameset )
        {
            altK.K2(res,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),resmode,MLid(),nullptr,nullptr,nullptr,assumeReal);

if ( testisvnan(res) || testisinf(res) )
{
errstream() << "K2 evaluated null 1: " << res << "\n";
errstream() << "K2 evaluated null 2: " << *xa << "," << *xb << "\n";
errstream() << "K2 evaluated null 2a: " << *xai << "," << *xbi << "\n";
errstream() << "K2 evaluated null 2b: " << *xainfoi << "," << *xbinfoi << "\n";
errstream() << "K2 evaluated null 2c: " << bias << "\n";
errstream() << "K2 evaluated null 2d: " << ia << "," << ib << "\n";
errstream() << "K2 evaluated null 3: " << getKernel() << "\n";
}
        }

        else
        {
            res = 0.0;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }

        if ( ia == ib )
        {
            res += iadiagoffset;
        }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (T) UUres;
        }

        if ( iaplanr || ibplanr )
        {
            Vector<int> iiplanr(2);
            Vector<int> iiplan(2);
            Vector<const gentype *> xxalt(2);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (T) VVres;
        }
    }

if ( testisvnan(res) || testisinf(res) )
{
errstream() << "phantomxyggghhhqq 1: " << res << "\n";
errstream() << "phantomxyggghhhqq 2: " << *xa << "," << *xb << "\n";
errstream() << "phantomxyggghhhqq 3: " << getKernel() << "\n";
}

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

double ML_Base::K3(int ia, int ib, int ic,
               double bias, const MercerKernel &basealtK, const gentype **pxyprod,
               const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc,
               const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo,
               int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) ) ? getRFFKernel() : basealtK );

    double res = 0;

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xc )
    {
        xc     = &x(ic);
        xcinfo = &xinfo(ic);
    }

    if ( !xainfo && !xbinfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K3(ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,&xinfob,&xinfoc,resmode);
    }

    else if ( !xbinfo && !xcinfo )
    {
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K3(ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,xainfo,&xinfob,&xinfoc,resmode);
    }

    else if ( !xainfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K3(ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,xbinfo,&xinfoc,resmode);
    }

    else if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);

        res = K3(ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,&xinfob,xcinfo,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfoa;

        getKernel().getvecInfo(xinfoa,*xa);

        res = K3(ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,xbinfo,xcinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfob;

        getKernel().getvecInfo(xinfob,*xb);

        res = K3(ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,xainfo,&xinfob,xcinfo,resmode);
    }

    else if ( !xcinfo )
    {
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoc,*xc);

        res = K3(ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,xainfo,xbinfo,&xinfoc,resmode);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;
        const double *x20 = nullptr;
        const double *x21 = nullptr;
        const double *x22 = nullptr;

        if ( xa && xb && xc && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) && ( (*xc).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
            x20 = &(getxymatelm(altK,ic,ia));
            x21 = &(getxymatelm(altK,ic,ib));
            x22 = &(getxymatelm(altK,ic,ic));
        }

        res = altK.K3(*xa,*xb,*xc,*xainfo,*xbinfo,*xcinfo,bias,pxyprod,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),x00,x10,x11,x20,x21,x22,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;
        const SparseVector<gentype> *xcnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;
        const SparseVector<gentype> *xcfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;
        const SparseVector<gentype> *xcfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;
        const SparseVector<gentype> *xcfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;
        const vecInfo *xcnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;
        const vecInfo *xcfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;
        int ixc,iic,xclr,xcrr,xcgr,icokr,icok,cdiagr,cgradOrder,icplanr,icplan,icset,icdenseint,icdensederiv;

        double arankL,arankR;
        double brankL,brankR;
        double crankL,crankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;
        int xcgrR,cgradOrderR,cgmuL,cgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        const gentype *ixctup = nullptr;
        const gentype *iictup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;
        SparseVector<gentype> *xcuntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;
        vecInfo *xcinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;
        double icdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;
        int icvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K3(sumind(i),ib,ic,bias,basealtK,nullptr,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K3(sumind(i),ib,ic,bias,basealtK,nullptr,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K3(ia,sumind(i),ic,bias,basealtK,nullptr,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K3(ia,sumind(i),ic,bias,basealtK,nullptr,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangc = detangle_x(xcuntang,xcinfountang,xcnear,xcfar,xcfarfar,xcfarfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,xclr,xcrr,xcgr,xcgrR,icokr,icok,crankL,crankR,cgmuL,cgmuR,ic,cdiagr,xc,xcinfo,cgradOrder,cgradOrderR,icplanr,icplan,icset,icdenseint,icdensederiv,sumind,sumweight,icdiagoffset,icvectset);

        if ( loctangc & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K3(ia,ib,sumind(i),bias,basealtK,nullptr,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K3(ia,ib,sumind(i),bias,basealtK,nullptr,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( ( iavectset == ibvectset ) && ( iavectset == icvectset ) ) ? 1 : 0;

//        NiceAssert( !iadenseint && !iadensederiv );
//        NiceAssert( !ibdenseint && !ibdensederiv );
//        NiceAssert( !icdenseint && !icdensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const SparseVector<gentype> *xci = xcuntang ? xcuntang : xc;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;
        const vecInfo *xcinfoi = xcinfountang ? xcinfountang : xcinfo;

        if ( issameset )
        {
            res = altK.K3(*xai,*xbi,*xci,*xainfoi,*xbinfoi,*xcinfoi,bias,nullptr,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumeReal);
        }

        else
        {
            res = 0.0;
        }

        if ( ( ia == ib ) && ( ia == ic ) )
        {
            res += iadiagoffset;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }
        if ( xcuntang ) { delete xcuntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }
        if ( xcinfountang ) { delete xcinfountang; }

        if ( iaokr || ibokr || icokr )
        {
            Vector<int> iiokr(3);
            Vector<int> iiok(3);
            Vector<const gentype *> xxalt(3);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            iiokr("&",2) = icokr;
            iiok("&",2)  = icok;
            xxalt("&",2) = (*xc).isf4indpresent(3) ? &((*xc).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,3,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (double) UUres;
        }

        if ( iaplanr || ibplanr || icplanr )
        {
            Vector<int> iiplanr(3);
            Vector<int> iiplan(3);
            Vector<const gentype *> xxalt(3);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            iiplanr("&",2) = icplanr;
            iiplan("&",2)  = icplan;
            xxalt("&",2)   = (*xc).isf4indpresent(7) ? &((*xc).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,3,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (double) VVres;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

template <class T>
T &ML_Base::K3(T &res, 
               int ia, int ib, int ic, 
               const T &bias, const MercerKernel &basealtK, const gentype **pxyprod, 
               const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
               const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
               int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) ) ? getRFFKernel() : basealtK );

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xc )
    {
        xc     = &x(ic);
        xcinfo = &xinfo(ic);
    }

    if ( !xainfo && !xbinfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        K3(res,ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,&xinfob,&xinfoc,resmode);
    }

    else if ( !xbinfo && !xcinfo )
    {
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        K3(res,ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,xainfo,&xinfob,&xinfoc,resmode);
    }

    else if ( !xainfo && !xcinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);

        K3(res,ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,xbinfo,&xinfoc,resmode);
    }

    else if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoa;
        vecInfo xinfob;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);

        K3(res,ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,&xinfob,xcinfo,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfoa;

        getKernel().getvecInfo(xinfoa,*xa);

        K3(res,ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,&xinfoa,xbinfo,xcinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfob;

        getKernel().getvecInfo(xinfob,*xb);

        K3(res,ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,xainfo,&xinfob,xcinfo,resmode);
    }

    else if ( !xcinfo )
    {
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoc,*xc);

        K3(res,ia,ib,ic,bias,basealtK,pxyprod,xa,xb,xc,xainfo,xbinfo,&xinfoc,resmode);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;
        const double *x20 = nullptr;
        const double *x21 = nullptr;
        const double *x22 = nullptr;

        if ( xa && xb && xc && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) && ( (*xc).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
            x20 = &(getxymatelm(altK,ic,ia));
            x21 = &(getxymatelm(altK,ic,ib));
            x22 = &(getxymatelm(altK,ic,ic));
        }

        altK.K3(res,*xa,*xb,*xc,*xainfo,*xbinfo,*xcinfo,bias,pxyprod,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),x00,x10,x11,x20,x21,x22,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;
        const SparseVector<gentype> *xcnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;
        const SparseVector<gentype> *xcfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;
        const SparseVector<gentype> *xcfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;
        const SparseVector<gentype> *xcfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;
        const vecInfo *xcnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;
        const vecInfo *xcfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;
        int ixc,iic,xclr,xcrr,xcgr,icokr,icok,cdiagr,cgradOrder,icplanr,icplan,icset,icdenseint,icdensederiv;

        double arankL,arankR;
        double brankL,brankR;
        double crankL,crankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;
        int xcgrR,cgradOrderR,cgmuL,cgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        const gentype *ixctup = nullptr;
        const gentype *iictup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;
        SparseVector<gentype> *xcuntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;
        vecInfo *xcinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;
        double icdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;
        int icvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K3(res,sumind(i),ib,ic,bias,basealtK,nullptr,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K3(restmp,sumind(i),ib,ic,bias,basealtK,nullptr,nullptr,xb,xc,nullptr,xbinfo,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K3(res,ia,sumind(i),ic,bias,basealtK,nullptr,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K3(restmp,ia,sumind(i),ic,bias,basealtK,nullptr,xa,nullptr,xc,xainfo,nullptr,xcinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangc = detangle_x(xcuntang,xcinfountang,xcnear,xcfar,xcfarfar,xcfarfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,xclr,xcrr,xcgr,xcgrR,icokr,icok,crankL,crankR,cgmuL,cgmuR,ic,cdiagr,xc,xcinfo,cgradOrder,cgradOrderR,icplanr,icplan,icset,icdenseint,icdensederiv,sumind,sumweight,icdiagoffset,icvectset);

        if ( loctangc & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K3(res,ia,ib,sumind(i),bias,basealtK,nullptr,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K3(restmp,ia,ib,sumind(i),bias,basealtK,nullptr,xa,xb,nullptr,xainfo,xbinfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( ( iavectset == ibvectset ) && ( iavectset == icvectset ) ) ? 1 : 0;

//        NiceAssert( !iadenseint && !iadensederiv );
//        NiceAssert( !ibdenseint && !ibdensederiv );
//        NiceAssert( !icdenseint && !icdensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const SparseVector<gentype> *xci = xcuntang ? xcuntang : xc;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;
        const vecInfo *xcinfoi = xcinfountang ? xcinfountang : xcinfo;

        if ( issameset )
        {
            altK.K3(res,*xai,*xbi,*xci,*xainfoi,*xbinfoi,*xcinfoi,bias,nullptr,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumeReal);
        }

        else
        {
            res = 0.0;
        }

        if ( ( ia == ib ) && ( ia == ic ) )
        {
            res += iadiagoffset;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }
        if ( xcuntang ) { delete xcuntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }
        if ( xcinfountang ) { delete xcinfountang; }

        if ( iaokr || ibokr || icokr )
        {
            Vector<int> iiokr(3);
            Vector<int> iiok(3);
            Vector<const gentype *> xxalt(3);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            iiokr("&",2) = icokr;
            iiok("&",2)  = icok;
            xxalt("&",2) = (*xc).isf4indpresent(3) ? &((*xc).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,3,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (T) UUres;
        }

        if ( iaplanr || ibplanr || icplanr )
        {
            Vector<int> iiplanr(3);
            Vector<int> iiplan(3);
            Vector<const gentype *> xxalt(3);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            iiplanr("&",2) = icplanr;
            iiplan("&",2)  = icplan;
            xxalt("&",2)   = (*xc).isf4indpresent(7) ? &((*xc).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,3,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (T) VVres;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

double ML_Base::K4(int ia, int ib, int ic, int id,
               double bias, const MercerKernel &basealtK, const gentype **pxyprod,
               const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd,
               const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo,
               int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) && RFFordata(id) ) ? getRFFKernel() : basealtK );

    double res = 0;

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xc )
    {
        xc     = &x(ic);
        xcinfo = &xinfo(ic);
    }

    if ( !xd )
    {
        xd     = &x(id);
        xdinfo = &xinfo(id);
    }

    if ( !xainfo && !xbinfo && !xcinfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,&xinfoc,&xinfod,resmode);
    }

    else if ( !xbinfo && !xcinfo && !xdinfo ) 
    {
        vecInfo xinfob;
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,&xinfoc,&xinfod,resmode);
    }

    else if ( !xainfo && !xcinfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,&xinfoc,&xinfod,resmode);
    }

    else if ( !xainfo && !xbinfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,xcinfo,&xinfod,resmode);
    }

    else if ( !xainfo && !xbinfo && !xcinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,&xinfoc,xdinfo,resmode);
    }

    else if ( !xainfo && !xbinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,xcinfo,xdinfo,resmode);
    }

    else if ( !xainfo && !xcinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,&xinfoc,xdinfo,resmode);
    }

    else if ( !xainfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,xcinfo,&xinfod,resmode);
    }

    else if ( !xbinfo && !xcinfo ) 
    {
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,&xinfoc,xdinfo,resmode);
    }

    else if ( !xbinfo && !xdinfo ) 
    {
        vecInfo xinfob;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,xcinfo,&xinfod,resmode);
    }

    else if ( !xcinfo && !xdinfo ) 
    {
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,&xinfoc,&xinfod,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfoa;

        getKernel().getvecInfo(xinfoa,*xa);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,xcinfo,xdinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfob;

        getKernel().getvecInfo(xinfob,*xb);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,xcinfo,xdinfo,resmode);
    }

    else if ( !xcinfo )
    {
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoc,*xc);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,&xinfoc,xdinfo,resmode);
    }

    else if ( !xdinfo )
    {
        vecInfo xinfod;

        getKernel().getvecInfo(xinfod,*xd);

        res = K4(ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,&xinfod,resmode);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() && xd->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;
        const double *x20 = nullptr;
        const double *x21 = nullptr;
        const double *x22 = nullptr;
        const double *x30 = nullptr;
        const double *x31 = nullptr;
        const double *x32 = nullptr;
        const double *x33 = nullptr;

        if ( isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) && ( id >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) && ( (*xc).nupsize() == 1 ) && ( (*xd).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
            x20 = &(getxymatelm(altK,ic,ia));
            x21 = &(getxymatelm(altK,ic,ib));
            x22 = &(getxymatelm(altK,ic,ic));
            x30 = &(getxymatelm(altK,id,ia));
            x31 = &(getxymatelm(altK,id,ib));
            x32 = &(getxymatelm(altK,id,ic));
            x33 = &(getxymatelm(altK,id,id));
        }

        res = altK.K4(*xa,*xb,*xc,*xd,*xainfo,*xbinfo,*xcinfo,*xdinfo,bias,pxyprod,ia,ib,ic,id,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic) && istrv(id),resmode,MLid(),x00,x10,x11,x20,x21,x22,x30,x31,x32,x33,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;
        const SparseVector<gentype> *xcnear = nullptr;
        const SparseVector<gentype> *xdnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;
        const SparseVector<gentype> *xcfar = nullptr;
        const SparseVector<gentype> *xdfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;
        const SparseVector<gentype> *xcfarfar = nullptr;
        const SparseVector<gentype> *xdfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;
        const SparseVector<gentype> *xcfarfarfar = nullptr;
        const SparseVector<gentype> *xdfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;
        const vecInfo *xcnearinfo = nullptr;
        const vecInfo *xdnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;
        const vecInfo *xcfarinfo = nullptr;
        const vecInfo *xdfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;
        int ixc,iic,xclr,xcrr,xcgr,icokr,icok,cdiagr,cgradOrder,icplanr,icplan,icset,icdenseint,icdensederiv;
        int ixd,iid,xdlr,xdrr,xdgr,idokr,idok,ddiagr,dgradOrder,idplanr,idplan,idset,iddenseint,iddensederiv;

        double arankL,arankR;
        double brankL,brankR;
        double crankL,crankR;
        double drankL,drankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;
        int xcgrR,cgradOrderR,cgmuL,cgmuR;
        int xdgrR,dgradOrderR,dgmuL,dgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        const gentype *ixctup = nullptr;
        const gentype *iictup = nullptr;

        const gentype *ixdtup = nullptr;
        const gentype *iidtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;
        SparseVector<gentype> *xcuntang = nullptr;
        SparseVector<gentype> *xduntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;
        vecInfo *xcinfountang = nullptr;
        vecInfo *xdinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;
        double icdiagoffset = 0;
        double iddiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;
        int icvectset = 0;
        int idvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K4(sumind(i),ib,ic,id,bias,basealtK,nullptr,nullptr,xb,xc,xd,nullptr,xbinfo,xcinfo,xdinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K4(sumind(i),ib,ic,id,bias,basealtK,nullptr,nullptr,xb,xc,xd,nullptr,xbinfo,xcinfo,xdinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K4(ia,sumind(i),ic,id,bias,basealtK,nullptr,xa,nullptr,xc,xd,xainfo,nullptr,xcinfo,xdinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K4(ia,sumind(i),ic,id,bias,basealtK,nullptr,xa,nullptr,xc,xd,xainfo,nullptr,xcinfo,xdinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangc = detangle_x(xcuntang,xcinfountang,xcnear,xcfar,xcfarfar,xcfarfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,xclr,xcrr,xcgr,xcgrR,icokr,icok,crankL,crankR,cgmuL,cgmuR,ic,cdiagr,xc,xcinfo,cgradOrder,cgradOrderR,icplanr,icplan,icset,icdenseint,icdensederiv,sumind,sumweight,icdiagoffset,icvectset);

        if ( loctangc & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K4(ia,ib,sumind(i),id,bias,basealtK,nullptr,xa,xb,nullptr,xd,xainfo,xbinfo,nullptr,xdinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K4(ia,ib,sumind(i),id,bias,basealtK,nullptr,xa,xb,nullptr,xd,xainfo,xbinfo,nullptr,xdinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangd = detangle_x(xduntang,xdinfountang,xdnear,xdfar,xdfarfar,xdfarfarfar,xdnearinfo,xdfarinfo,ixd,iid,ixdtup,iidtup,xdlr,xdrr,xdgr,xdgrR,idokr,idok,drankL,drankR,dgmuL,dgmuR,id,ddiagr,xd,xdinfo,dgradOrder,dgradOrderR,idplanr,idplan,idset,iddenseint,iddensederiv,sumind,sumweight,iddiagoffset,idvectset);

        if ( loctangd & 2048 )
        {
            double restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    res = K4(ia,ib,ic,sumind(i),bias,basealtK,nullptr,xa,xb,xc,nullptr,xainfo,xbinfo,xcinfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    restmp = K4(ia,ib,ic,sumind(i),bias,basealtK,nullptr,xa,xb,xc,nullptr,xainfo,xbinfo,xcinfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( ( iavectset == ibvectset ) && ( iavectset == icvectset ) && ( iavectset == idvectset ) ) ? 1 : 0;

//        NiceAssert( !iadenseint && !iadensederiv );
//        NiceAssert( !ibdenseint && !ibdensederiv );
//        NiceAssert( !icdenseint && !icdensederiv );
//        NiceAssert( !iddenseint && !iddensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const SparseVector<gentype> *xci = xcuntang ? xcuntang : xc;
        const SparseVector<gentype> *xdi = xduntang ? xduntang : xd;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;
        const vecInfo *xcinfoi = xcinfountang ? xcinfountang : xcinfo;
        const vecInfo *xdinfoi = xdinfountang ? xdinfountang : xdinfo;

        if ( issameset )
        {
            res = altK.K4(*xai,*xbi,*xci,*xdi,*xainfoi,*xbinfoi,*xcinfoi,*xdinfoi,bias,nullptr,ia,ib,ic,id,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic) && istrv(id),resmode,MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumeReal);
        }

        else
        {
            res = 0.0;
        }

        if ( ( ia == ib ) && ( ia == ic ) && ( ia == id ) )
        {
            res += iadiagoffset;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }
        if ( xcuntang ) { delete xcuntang; }
        if ( xduntang ) { delete xduntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }
        if ( xcinfountang ) { delete xcinfountang; }
        if ( xdinfountang ) { delete xdinfountang; }

        if ( iaokr || ibokr || icokr || idokr )
        {
            Vector<int> iiokr(4);
            Vector<int> iiok(4);
            Vector<const gentype *> xxalt(4);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            iiokr("&",2) = icokr;
            iiok("&",2)  = icok;
            xxalt("&",2) = (*xc).isf4indpresent(3) ? &((*xc).f4(3)) : &nullgentype();

            iiokr("&",3) = idokr;
            iiok("&",3)  = idok;
            xxalt("&",3) = (*xd).isf4indpresent(3) ? &((*xd).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,4,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (double) UUres;
        }

        if ( iaplanr || ibplanr || icplanr || idplanr )
        {
            Vector<int> iiplanr(4);
            Vector<int> iiplan(4);
            Vector<const gentype *> xxalt(4);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            iiplanr("&",2) = icplanr;
            iiplan("&",2)  = icplan;
            xxalt("&",2)   = (*xc).isf4indpresent(7) ? &((*xc).f4(7)) : &nullgentype();

            iiplanr("&",3) = idplanr;
            iiplan("&",3)  = idplan;
            xxalt("&",3)   = (*xd).isf4indpresent(7) ? &((*xd).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,4,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (double) VVres;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

template <class T>
T &ML_Base::K4(T &res, 
               int ia, int ib, int ic, int id, 
               const T &bias, const MercerKernel &basealtK, const gentype **pxyprod, 
               const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
               const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
               int resmode) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) && RFFordata(id) ) ? getRFFKernel() : basealtK );

//phantomx
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xc )
    {
        xc     = &x(ic);
        xcinfo = &xinfo(ic);
    }

    if ( !xd )
    {
        xd     = &x(id);
        xdinfo = &xinfo(id);
    }

    if ( !xainfo && !xbinfo && !xcinfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,&xinfoc,&xinfod,resmode);
    }

    else if ( !xbinfo && !xcinfo && !xdinfo ) 
    {
        vecInfo xinfob;
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,&xinfoc,&xinfod,resmode);
    }

    else if ( !xainfo && !xcinfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,&xinfoc,&xinfod,resmode);
    }

    else if ( !xainfo && !xbinfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,xcinfo,&xinfod,resmode);
    }

    else if ( !xainfo && !xbinfo && !xcinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,&xinfoc,xdinfo,resmode);
    }

    else if ( !xainfo && !xbinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfob;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfob,*xb);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,&xinfob,xcinfo,xdinfo,resmode);
    }

    else if ( !xainfo && !xcinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfoc,*xc);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,&xinfoc,xdinfo,resmode);
    }

    else if ( !xainfo && !xdinfo ) 
    {
        vecInfo xinfoa;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoa,*xa);
        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,xcinfo,&xinfod,resmode);
    }

    else if ( !xbinfo && !xcinfo ) 
    {
        vecInfo xinfob;
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfoc,*xc);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,&xinfoc,xdinfo,resmode);
    }

    else if ( !xbinfo && !xdinfo ) 
    {
        vecInfo xinfob;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfob,*xb);
        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,xcinfo,&xinfod,resmode);
    }

    else if ( !xcinfo && !xdinfo ) 
    {
        vecInfo xinfoc;
        vecInfo xinfod;

        getKernel().getvecInfo(xinfoc,*xc);
        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,&xinfoc,&xinfod,resmode);
    }

    else if ( !xainfo )
    {
        vecInfo xinfoa;

        getKernel().getvecInfo(xinfoa,*xa);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,xcinfo,xdinfo,resmode);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfob;

        getKernel().getvecInfo(xinfob,*xb);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,xcinfo,xdinfo,resmode);
    }

    else if ( !xcinfo )
    {
        vecInfo xinfoc;

        getKernel().getvecInfo(xinfoc,*xc);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,&xinfoc,xdinfo,resmode);
    }

    else if ( !xdinfo )
    {
        vecInfo xinfod;

        getKernel().getvecInfo(xinfod,*xd);

        K4(res,ia,ib,ic,id,bias,basealtK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,&xinfod,resmode);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() && xd->isnofaroffindpresent() )
    {
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;
        const double *x20 = nullptr;
        const double *x21 = nullptr;
        const double *x22 = nullptr;
        const double *x30 = nullptr;
        const double *x31 = nullptr;
        const double *x32 = nullptr;
        const double *x33 = nullptr;

        if ( isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) && ( id >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) && ( (*xc).nupsize() == 1 ) && ( (*xd).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
            x20 = &(getxymatelm(altK,ic,ia));
            x21 = &(getxymatelm(altK,ic,ib));
            x22 = &(getxymatelm(altK,ic,ic));
            x30 = &(getxymatelm(altK,id,ia));
            x31 = &(getxymatelm(altK,id,ib));
            x32 = &(getxymatelm(altK,id,ic));
            x33 = &(getxymatelm(altK,id,id));
        }

        altK.K4(res,*xa,*xb,*xc,*xd,*xainfo,*xbinfo,*xcinfo,*xdinfo,bias,pxyprod,ia,ib,ic,id,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic) && istrv(id),resmode,MLid(),x00,x10,x11,x20,x21,x22,x30,x31,x32,x33,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;
        const SparseVector<gentype> *xcnear = nullptr;
        const SparseVector<gentype> *xdnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;
        const SparseVector<gentype> *xcfar = nullptr;
        const SparseVector<gentype> *xdfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;
        const SparseVector<gentype> *xcfarfar = nullptr;
        const SparseVector<gentype> *xdfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;
        const SparseVector<gentype> *xcfarfarfar = nullptr;
        const SparseVector<gentype> *xdfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;
        const vecInfo *xcnearinfo = nullptr;
        const vecInfo *xdnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;
        const vecInfo *xcfarinfo = nullptr;
        const vecInfo *xdfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;
        int ixc,iic,xclr,xcrr,xcgr,icokr,icok,cdiagr,cgradOrder,icplanr,icplan,icset,icdenseint,icdensederiv;
        int ixd,iid,xdlr,xdrr,xdgr,idokr,idok,ddiagr,dgradOrder,idplanr,idplan,idset,iddenseint,iddensederiv;

        double arankL,arankR;
        double brankL,brankR;
        double crankL,crankR;
        double drankL,drankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;
        int xcgrR,cgradOrderR,cgmuL,cgmuR;
        int xdgrR,dgradOrderR,dgmuL,dgmuR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        const gentype *ixctup = nullptr;
        const gentype *iictup = nullptr;

        const gentype *ixdtup = nullptr;
        const gentype *iidtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;
        SparseVector<gentype> *xcuntang = nullptr;
        SparseVector<gentype> *xduntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;
        vecInfo *xcinfountang = nullptr;
        vecInfo *xdinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;
        double icdiagoffset = 0;
        double iddiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;
        int icvectset = 0;
        int idvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);

        if ( loctanga & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K4(res,sumind(i),ib,ic,id,bias,basealtK,nullptr,nullptr,xb,xc,xd,nullptr,xbinfo,xcinfo,xdinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K4(restmp,sumind(i),ib,ic,id,bias,basealtK,nullptr,nullptr,xb,xc,xd,nullptr,xbinfo,xcinfo,xdinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        if ( loctangb & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K4(res,ia,sumind(i),ic,id,bias,basealtK,nullptr,xa,nullptr,xc,xd,xainfo,nullptr,xcinfo,xdinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K4(restmp,ia,sumind(i),ic,id,bias,basealtK,nullptr,xa,nullptr,xc,xd,xainfo,nullptr,xcinfo,xdinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangc = detangle_x(xcuntang,xcinfountang,xcnear,xcfar,xcfarfar,xcfarfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,xclr,xcrr,xcgr,xcgrR,icokr,icok,crankL,crankR,cgmuL,cgmuR,ic,cdiagr,xc,xcinfo,cgradOrder,cgradOrderR,icplanr,icplan,icset,icdenseint,icdensederiv,sumind,sumweight,icdiagoffset,icvectset);

        if ( loctangc & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K4(res,ia,ib,sumind(i),id,bias,basealtK,nullptr,xa,xb,nullptr,xd,xainfo,xbinfo,nullptr,xdinfo,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K4(restmp,ia,ib,sumind(i),id,bias,basealtK,nullptr,xa,xb,nullptr,xd,xainfo,xbinfo,nullptr,xdinfo,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int loctangd = detangle_x(xduntang,xdinfountang,xdnear,xdfar,xdfarfar,xdfarfarfar,xdnearinfo,xdfarinfo,ixd,iid,ixdtup,iidtup,xdlr,xdrr,xdgr,xdgrR,idokr,idok,drankL,drankR,dgmuL,dgmuR,id,ddiagr,xd,xdinfo,dgradOrder,dgradOrderR,idplanr,idplan,idset,iddenseint,iddensederiv,sumind,sumweight,iddiagoffset,idvectset);

        if ( loctangd & 2048 )
        {
            T restmp;

            for ( int i = 0 ; i < sumind.size() ; i++ )
            {
                if ( !i )
                {
                    K4(res,ia,ib,ic,sumind(i),bias,basealtK,nullptr,xa,xb,xc,nullptr,xainfo,xbinfo,xcinfo,nullptr,resmode);
                    res *= sumweight(i);
                }

                else
                {
                    K4(restmp,ia,ib,ic,sumind(i),bias,basealtK,nullptr,xa,xb,xc,nullptr,xainfo,xbinfo,xcinfo,nullptr,resmode);
                    restmp *= sumweight(i);
                    res += restmp;
                }
            }

            return res;
        }

        int issameset = ( ( iavectset == ibvectset ) && ( iavectset == icvectset ) && ( iavectset == idvectset ) ) ? 1 : 0;

//        NiceAssert( !iadenseint && !iadensederiv );
//        NiceAssert( !ibdenseint && !ibdensederiv );
//        NiceAssert( !icdenseint && !icdensederiv );
//        NiceAssert( !iddenseint && !iddensederiv );

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const SparseVector<gentype> *xci = xcuntang ? xcuntang : xc;
        const SparseVector<gentype> *xdi = xduntang ? xduntang : xd;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;
        const vecInfo *xcinfoi = xcinfountang ? xcinfountang : xcinfo;
        const vecInfo *xdinfoi = xdinfountang ? xdinfountang : xdinfo;

        if ( issameset )
        {
            altK.K4(res,*xai,*xbi,*xci,*xdi,*xainfoi,*xbinfoi,*xcinfoi,*xdinfoi,bias,nullptr,ia,ib,ic,id,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic) && istrv(id),resmode,MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumeReal);
        }

        else
        {
            res = 0.0;
        }

        if ( ( ia == ib ) && ( ia == ic ) && ( ia == id ) )
        {
            res += iadiagoffset;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }
        if ( xcuntang ) { delete xcuntang; }
        if ( xduntang ) { delete xduntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }
        if ( xcinfountang ) { delete xcinfountang; }
        if ( xdinfountang ) { delete xdinfountang; }

        if ( iaokr || ibokr || icokr || idokr )
        {
            Vector<int> iiokr(4);
            Vector<int> iiok(4);
            Vector<const gentype *> xxalt(4);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            iiokr("&",2) = icokr;
            iiok("&",2)  = icok;
            xxalt("&",2) = (*xc).isf4indpresent(3) ? &((*xc).f4(3)) : &nullgentype();

            iiokr("&",3) = idokr;
            iiok("&",3)  = idok;
            xxalt("&",3) = (*xd).isf4indpresent(3) ? &((*xd).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,4,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (T) UUres;
        }

        if ( iaplanr || ibplanr || icplanr || idplanr )
        {
            Vector<int> iiplanr(4);
            Vector<int> iiplan(4);
            Vector<const gentype *> xxalt(4);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            iiplanr("&",2) = icplanr;
            iiplan("&",2)  = icplan;
            xxalt("&",2)   = (*xc).isf4indpresent(7) ? &((*xc).f4(7)) : &nullgentype();

            iiplanr("&",3) = idplanr;
            iiplan("&",3)  = idplan;
            xxalt("&",3)   = (*xd).isf4indpresent(7) ? &((*xd).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,4,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            res = (T) VVres;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}


double ML_Base::Km(int m,
               Vector<int> &i,
               double bias, const MercerKernel &basealtK, const gentype **pxyprod,
               Vector<const SparseVector<gentype> *> *xxx,
               Vector<const vecInfo *> *xxxinfo,
               int resmode) const
{
//phantomx
    NiceAssert( m >= 0 );
    NiceAssert( ( xxx && xxxinfo ) || ( !xxx && !xxxinfo ) );

    double res = 0;

    int z = 0;

    if ( !xxx )
    {
        MEMNEW(xxx,Vector<const SparseVector<gentype> *>(m));
        MEMNEW(xxxinfo,Vector<const vecInfo *>(m));

        *xxx = (const SparseVector<gentype> *) nullptr;
        *xxxinfo = (const vecInfo *) nullptr;

        res = Km(m,i,bias,basealtK,pxyprod,xxx,xxxinfo,resmode);

        MEMDEL(xxx);
        MEMDEL(xxxinfo);
    }

    if ( m == 0 )
    {
        res = K0(bias,basealtK,pxyprod,resmode);
    }

    else if ( m == 1 )
    {
        res = K1(i(z),bias,basealtK,pxyprod,(*xxx)(z),(*xxxinfo)(z),resmode);
    }

    else if ( m == 2 )
    {
        res = K2(i(z),i(1),bias,basealtK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxxinfo)(z),(*xxxinfo)(1),resmode);
    }

    else if ( m == 3 )
    {
        res = K3(i(z),i(1),i(2),bias,basealtK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxx)(2),(*xxxinfo)(z),(*xxxinfo)(1),(*xxxinfo)(2),resmode);
    }

    else if ( m == 4 )
    {
        res = K4(i(z),i(1),i(2),i(3),bias,basealtK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxx)(2),(*xxx)(3),(*xxxinfo)(z),(*xxxinfo)(1),(*xxxinfo)(2),(*xxxinfo)(3),resmode);
    }

    else
    {
        // Calculate if data-filling / pre-processing needed

        int datamissing = 0;
        int needpreproc = 0;
        int ii,jj;
        int z = 0;
        bool useRFFkernel = true;

        for ( ii = 0 ; ii < m ; ++ii )
        {
            if ( !RFFordata(i(ii)) )
            {
                useRFFkernel = false;
            }

            const SparseVector<gentype> *xxi = (*xxx)(ii);

            NiceAssert( ( xxi && (*xxxinfo)(ii) ) || ( !xxi && !(*xxxinfo)(ii) ) );

            if ( !xxi )
            {
                datamissing = 1;

                break;
            }

            else if ( !((*xxi).isnofaroffindpresent()) )
            {
                needpreproc = 1;

                break;
            }
        }

        const MercerKernel &altK = ( useRFFkernel ? getRFFKernel() : basealtK );

        if ( datamissing )
        {
            // Missing data - fill in and recurse

            Vector<const SparseVector<gentype> *> xx(*xxx);
            Vector<const vecInfo *> xxinfo(*xxxinfo);

            // Fill in any missing data vectors

            for ( ii = m-1 ; ii >= 0 ; --ii )
            {
                if ( !xx(ii) )
                {
                    xx("&",ii)     = &x(i(ii));
                    xxinfo("&",ii) = &xinfo(i(ii));
                }
            }

            res = Km(m,i,bias,basealtK,pxyprod,&xx,&xxinfo,resmode);
        }

        else if ( needpreproc )
        {
            // Overwrite data where required before call

            Vector<int> j(i);
            Vector<const SparseVector<gentype> *> xx(*xxx);
            Vector<const vecInfo *> xxinfo(*xxxinfo);

            Vector<SparseVector<gentype> *> xxi(m);
            Vector<vecInfo *> xxinfoi(m);

            xxi = (SparseVector<gentype> *) nullptr;
            xxinfoi = (vecInfo *) nullptr;

            Vector<int> iokr(m);
            Vector<int> iok(m);

            Vector<int> iplanr(m);
            Vector<int> iplan(m);

            iokr = z;
            iok  = z;

            iplanr = z;
            iplan  = z;

            double idiagoffset = 0;
            int issameset = 0;

            for ( ii = 0 ; ii < m ; ++ii )
            {
                if ( !((*(xx(ii))).isnofaroffindpresent()) )
                {
                    const SparseVector<gentype> *xanear      = nullptr;
                    const SparseVector<gentype> *xafar       = nullptr;
                    const SparseVector<gentype> *xafarfar    = nullptr;
                    const SparseVector<gentype> *xafarfarfar = nullptr;

                    const vecInfo *xanearinfo = nullptr;
                    const vecInfo *xafarinfo  = nullptr;

                    int ixa,iia,adiagr,aagradOrder,iiaset,iiadenseint,iiadensederiv,ilr,irr,igr;
                    double arankL,arankR;
                    int igrR,aagradOrderR,agmuL,agmuR;

                    const gentype *ixatup = nullptr;
                    const gentype *iiatup = nullptr;

                    Vector<int> sumind;
                    Vector<double> sumweight;

                    int ivectset = 0;

                    int loctang = detangle_x(xxi("&",ii),xxinfoi("&",ii),xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,ilr,irr,igr,igrR,iokr("&",ii),iok("&",ii),arankL,arankR,agmuL,agmuR,j(ii),adiagr,xx(ii),xxinfo(ii),aagradOrder,aagradOrderR,iplanr("&",ii),iplan("&",ii),iiaset,iiadenseint,iiadensederiv,sumind,sumweight,idiagoffset,ivectset);

                    if ( ii == 0 )
                    {
                        issameset = ivectset;
                    }

                    else if ( issameset != ivectset )
                    {
                        issameset = -1;
                    }

                    (void) loctang; NiceAssert( !( loctang & 2048 ) );

//                    NiceAssert( !iiadenseint && !iiadensederiv );

                    xx("&",ii) = xxi(ii) ? xxi(ii) : xx(ii);
                    xxinfo("&",ii) = xxinfoi(ii) ? xxinfoi(ii) : xxinfo(ii);
                }
            }

            issameset = ( issameset >= 0 ) ? 1 : 0;

            if ( issameset )
            {
                res = altK.Km(m,xx,xxinfo,bias,i,nullptr,xspaceDim(),isXConsistent() && istrv(i),resmode,MLid(),nullptr,assumeReal);
            }

            else
            {
                res = 0.0;
            }

            if ( i.size() && ( i == i(0) ) )
            {
                res += idiagoffset;
            }

            for ( ii = 0 ; ii < m ; ++ii )
            {
                if ( xxi("&",ii) ) { delete xxi("&",ii); }
                if ( xxinfoi("&",ii) ) { delete xxinfoi("&",ii); }
            }

            if ( sum(iokr) )
            {
                Vector<const gentype *> xxalt(m);

                for ( jj = 0 ; jj < m ; ++jj )
                {
                    xxalt("&",jj) = (*(xx(jj))).isf4indpresent(3) ? &((*(xx(jj))).f4(3)) : &nullgentype();
                }

                gentype UUres;

                (*UUcallback)(UUres,m,*this,iokr,iok,xxalt,defbasisUU);

                res *= (double) UUres;
            }

            if ( sum(iplanr) )
            {
                Vector<const gentype *> xxalt(m);

                for ( jj = 0 ; jj < m ; ++jj )
                {
                    xxalt("&",jj) = (*(xx(jj))).isf4indpresent(7) ? &((*(xx(jj))).f4(7)) : &nullgentype();
                }

                gentype VVres;
                gentype kval(res);

                (*VVcallback)(VVres,m,kval,*this,iplanr,iplan,xxalt,defbasisVV);

                res = (double) VVres;
            }
        }

        else
        {
            // This needs to be outside the next statement to ensure it remains on the stack until used!
            retMatrix<double> tmpma;

            const Matrix<double> *xyp = nullptr;

            if ( isxymat(altK) && ( i >= z ) )
            {
                xyp = &((getxymat(altK))(i,i,tmpma));
            }

            res = altK.Km(m,*xxx,*xxxinfo,bias,i,pxyprod,xspaceDim(),isXConsistent() && istrv(i),resmode,MLid(),xyp,assumeReal);
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

template <class T>
T &ML_Base::Km(int m, T &res,
               Vector<int> &i,
               const T &bias, const MercerKernel &basealtK, const gentype **pxyprod,
               Vector<const SparseVector<gentype> *> *xxx,
               Vector<const vecInfo *> *xxxinfo,
               int resmode) const
{
//phantomx
    NiceAssert( m >= 0 );
    NiceAssert( ( xxx && xxxinfo ) || ( !xxx && !xxxinfo ) );

    int z = 0;

    if ( !xxx )
    {
        MEMNEW(xxx,Vector<const SparseVector<gentype> *>(m));
        MEMNEW(xxxinfo,Vector<const vecInfo *>(m));

        *xxx = (const SparseVector<gentype> *) nullptr;
        *xxxinfo = (const vecInfo *) nullptr;

        Km(m,res,i,bias,basealtK,pxyprod,xxx,xxxinfo,resmode);

        MEMDEL(xxx);
        MEMDEL(xxxinfo);
    }

    if ( m == 0 )
    {
        K0(res,bias,basealtK,pxyprod,resmode);
    }

    else if ( m == 1 )
    {
        K1(res,i(z),bias,basealtK,pxyprod,(*xxx)(z),(*xxxinfo)(z),resmode);
    }

    else if ( m == 2 )
    {
        K2(res,i(z),i(1),bias,basealtK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxxinfo)(z),(*xxxinfo)(1),resmode);
    }

    else if ( m == 3 )
    {
        K3(res,i(z),i(1),i(2),bias,basealtK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxx)(2),(*xxxinfo)(z),(*xxxinfo)(1),(*xxxinfo)(2),resmode);
    }

    else if ( m == 4 )
    {
        K4(res,i(z),i(1),i(2),i(3),bias,basealtK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxx)(2),(*xxx)(3),(*xxxinfo)(z),(*xxxinfo)(1),(*xxxinfo)(2),(*xxxinfo)(3),resmode);
    }

    else
    {
        // Calculate if data-filling / pre-processing needed

        int datamissing = 0;
        int needpreproc = 0;
        int ii,jj;
        int z = 0;
        bool useRFFkernel = true;

        for ( ii = 0 ; ii < m ; ++ii )
        {
            if ( !RFFordata(i(ii)) )
            {
                useRFFkernel = false;
            }

            const SparseVector<gentype> *xxi = (*xxx)(ii);

            NiceAssert( ( xxi && (*xxxinfo)(ii) ) || ( !xxi && !(*xxxinfo)(ii) ) );

            if ( !xxi )
            {
                datamissing = 1;

                break;
            }

            else if ( !((*xxi).isnofaroffindpresent()) )
            {
                needpreproc = 1;

                break;
            }
        }

        const MercerKernel &altK = ( useRFFkernel ? getRFFKernel() : basealtK );

        if ( datamissing )
        {
            // Missing data - fill in and recurse

            Vector<const SparseVector<gentype> *> xx(*xxx);
            Vector<const vecInfo *> xxinfo(*xxxinfo);

            // Fill in any missing data vectors

            for ( ii = m-1 ; ii >= 0 ; --ii )
            {
                if ( !xx(ii) )
                {
                    xx("&",ii)     = &x(i(ii));
                    xxinfo("&",ii) = &xinfo(i(ii));
                }
            }

            Km(m,res,i,bias,basealtK,pxyprod,&xx,&xxinfo,resmode);
        }

        else if ( needpreproc )
        {
            // Overwrite data where required before call

            Vector<int> j(i);
            Vector<const SparseVector<gentype> *> xx(*xxx);
            Vector<const vecInfo *> xxinfo(*xxxinfo);

            Vector<SparseVector<gentype> *> xxi(m);
            Vector<vecInfo *> xxinfoi(m);

            xxi = (SparseVector<gentype> *) nullptr;
            xxinfoi = (vecInfo *) nullptr;

            Vector<int> iokr(m);
            Vector<int> iok(m);

            Vector<int> iplanr(m);
            Vector<int> iplan(m);

            iokr = z;
            iok  = z;

            iplanr = z;
            iplan  = z;

            double idiagoffset = 0;
            int issameset = 0;

            for ( ii = 0 ; ii < m ; ++ii )
            {
                if ( !((*(xx(ii))).isnofaroffindpresent()) )
                {
                    const SparseVector<gentype> *xanear      = nullptr;
                    const SparseVector<gentype> *xafar       = nullptr;
                    const SparseVector<gentype> *xafarfar    = nullptr;
                    const SparseVector<gentype> *xafarfarfar = nullptr;

                    const vecInfo *xanearinfo = nullptr;
                    const vecInfo *xafarinfo  = nullptr;

                    int ixa,iia,adiagr,aagradOrder,iiaset,iiadenseint,iiadensederiv,ilr,irr,igr;
                    double arankL,arankR;
                    int igrR,aagradOrderR,agmuL,agmuR;

                    const gentype *ixatup = nullptr;
                    const gentype *iiatup = nullptr;

                    Vector<int> sumind;
                    Vector<double> sumweight;

                    int ivectset = 0;

                    int loctang = detangle_x(xxi("&",ii),xxinfoi("&",ii),xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,ilr,irr,igr,igrR,iokr("&",ii),iok("&",ii),arankL,arankR,agmuL,agmuR,j(ii),adiagr,xx(ii),xxinfo(ii),aagradOrder,aagradOrderR,iplanr("&",ii),iplan("&",ii),iiaset,iiadenseint,iiadensederiv,sumind,sumweight,idiagoffset,ivectset);

                    if ( ii == 0 )
                    {
                        issameset = ivectset;
                    }

                    else if ( issameset != ivectset )
                    {
                        issameset = -1;
                    }

                    (void) loctang; NiceAssert( !( loctang & 2048 ) );

//                    NiceAssert( !iiadenseint && !iiadensederiv );

                    xx("&",ii) = xxi(ii) ? xxi(ii) : xx(ii);
                    xxinfo("&",ii) = xxinfoi(ii) ? xxinfoi(ii) : xxinfo(ii);
                }
            }

            issameset = ( issameset >= 0 ) ? 1 : 0;

            if ( issameset )
            {
                res = altK.Km(m,xx,xxinfo,bias,i,nullptr,xspaceDim(),isXConsistent() && istrv(i),resmode,MLid(),nullptr,assumeReal);
            }

            else
            {
                res = 0.0;
            }

            if ( i.size() && ( i == i(0) ) )
            {
                res += idiagoffset;
            }

            for ( ii = 0 ; ii < m ; ++ii )
            {
                if ( xxi("&",ii) ) { delete xxi("&",ii); }
                if ( xxinfoi("&",ii) ) { delete xxinfoi("&",ii); }
            }

            if ( sum(iokr) )
            {
                Vector<const gentype *> xxalt(m);

                for ( jj = 0 ; jj < m ; ++jj )
                {
                    xxalt("&",jj) = (*(xx(jj))).isf4indpresent(3) ? &((*(xx(jj))).f4(3)) : &nullgentype();
                }

                gentype UUres;

                (*UUcallback)(UUres,m,*this,iokr,iok,xxalt,defbasisUU);

                res *= (T) UUres;
            }

            if ( sum(iplanr) )
            {
                Vector<const gentype *> xxalt(m);

                for ( jj = 0 ; jj < m ; ++jj )
                {
                    xxalt("&",jj) = (*(xx(jj))).isf4indpresent(7) ? &((*(xx(jj))).f4(7)) : &nullgentype();
                }

                gentype VVres;
                gentype kval(res);

                (*VVcallback)(VVres,m,kval,*this,iplanr,iplan,xxalt,defbasisVV);

                res = (T) VVres;
            }
        }

        else
        {
            // This needs to be outside the next statement to ensure it remains on the stack until used!
            retMatrix<double> tmpma;

            const Matrix<double> *xyp = nullptr;

            if ( isxymat(altK) && ( i >= z ) )
            {
                xyp = &((getxymat(altK))(i,i,tmpma));
            }

            res = altK.Km(m,*xxx,*xxxinfo,bias,i,pxyprod,xspaceDim(),isXConsistent() && istrv(i),resmode,MLid(),xyp,assumeReal);
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

double ML_Base::KK0ip(double bias, const gentype **pxyprod) const
{
//phantomx
    // Shortcut for speed

    return getKernel().K0ip(bias,pxyprod,0,0,MLid(),assumeReal);
}

double ML_Base::KK1ip(int ib, double bias, const gentype **pxyprod, const SparseVector<gentype> *xx, const vecInfo *xxinfo) const
{
    const MercerKernel &altK = ( ( RFFordata(ib) ) ? getRFFKernel() : getKernel() );

//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );

    // Shortcut for speed

    if ( !xx )
    {
        const SparseVector<gentype> &xib = x(ib);
        const vecInfo &xinfoi = xinfo(ib);

        return KK1ip(ib,bias,pxyprod,&xib,&xinfoi);
    }

        const SparseVector<gentype> &xxx  = (*xx).n();
        const SparseVector<gentype> &xxxx = xxx.nup(0);

        return altK.K1ip(xxxx,*xxinfo,bias,pxyprod,ib,0,0,MLid(),assumeReal);
}

double ML_Base::KK2ip(int ib, int jb, double bias, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    const MercerKernel &altK = ( ( RFFordata(ib) && RFFordata(jb) ) ? getRFFKernel() : getKernel() );

//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = x(ib);
        const SparseVector<gentype> &xjb = x(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        return KK2ip(ib,jb,bias,pxyprod,&xib,&xjb,&xinfoi,&xinfoj);
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = x(jb);
        const vecInfo &xinfoj = xinfo(jb);

        return KK2ip(ib,jb,bias,pxyprod,xx,&xjb,xxinfo,&xinfoj);
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = x(ib);
        const vecInfo &xinfoi = xinfo(ib);

        return KK2ip(ib,jb,bias,pxyprod,&xib,yy,&xinfoi,yyinfo);
    }

        const SparseVector<gentype> &xxx  = (*xx).n();
        const SparseVector<gentype> &xxxx = xxx.nup(0);

        const SparseVector<gentype> &yyy  = (*yy).n();
        const SparseVector<gentype> &yyyy = yyy.nup(0);

        return altK.K2ip(xxxx,yyyy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,0,0,MLid(),assumeReal);
}

double ML_Base::KK3ip(int ia, int ib, int ic, double bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo) const
{
//phantomx
    NiceAssert( ( xa && xainfo ) || ( !xa && ( !xainfo || ( !xb && xc ) ) ) );
    NiceAssert( ( xb && xbinfo ) || ( !xb && ( !xbinfo || ( !xa && xc ) ) ) );
    NiceAssert( ( xc && xcinfo ) || ( !xc && !xcinfo ) );

    // Always simplify

         if ( xb && !xa ) { return KK3ip(ib,ia,ic,bias,pxyprod,xb,xa,xc,xbinfo,xainfo,xcinfo); }
    else if ( xc && !xa ) { return KK3ip(ic,ib,ia,bias,pxyprod,xc,xb,xa,xcinfo,xbinfo,xainfo); }
    else if ( xc && !xb ) { return KK3ip(ia,ic,ib,bias,pxyprod,xa,xc,xb,xainfo,xcinfo,xbinfo); }

    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) ) ? getRFFKernel() : getKernel() );

    // Shortcut for speed

    if ( xa && xb && xc )
    {
        const SparseVector<gentype> &xxa  = (*xa).n();
        const SparseVector<gentype> &xxxa = xxa.nup(0);

        const SparseVector<gentype> &xxb  = (*xb).n();
        const SparseVector<gentype> &xxxb = xxb.nup(0);

        const SparseVector<gentype> &xxc  = (*xc).n();
        const SparseVector<gentype> &xxxc = xxc.nup(0);

        return altK.K3ip(xxxa,xxxb,xxxc,*xainfo,*xbinfo,*xcinfo,bias,pxyprod,ia,ib,ic,0,0,MLid(),assumeReal);
    }

    else if ( xa && xb )
    {
        const SparseVector<gentype> &xcb = x(ic);
        const vecInfo &xinfoc = xinfo(ic);

        return KK3ip(ia,ib,ic,bias,pxyprod,xa,xb,&xcb,xainfo,xbinfo,&xinfoc);
    }

    else if ( xa && xa->isnofaroffindpresent() )
    {
        const SparseVector<gentype> &xbb = x(ib);
        const SparseVector<gentype> &xcb = x(ic);

        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);

        return KK3ip(ia,ib,ic,bias,pxyprod,xa,&xbb,&xcb,xainfo,&xinfob,&xinfoc);
    }

        const SparseVector<gentype> &xab = x(ia);
        const SparseVector<gentype> &xbb = x(ib);
        const SparseVector<gentype> &xcb = x(ic);

        const vecInfo &xinfoa = xinfo(ia);
        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);

        return KK3ip(ia,ib,ic,bias,pxyprod,&xab,&xbb,&xcb,&xinfoa,&xinfob,&xinfoc);
}

double ML_Base::KK4ip(int ia, int ib, int ic, int id, double bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo) const
{
//phantomx
    NiceAssert( ( xa && xainfo ) || ( !xa && ( !xainfo || ( !xb && xc && xd ) ) ) );
    NiceAssert( ( xb && xbinfo ) || ( !xb && ( !xbinfo || ( !xa && xc && xd ) ) ) );
    NiceAssert( ( xc && xcinfo ) || ( !xc && !xcinfo ) );
    NiceAssert( ( xd && xdinfo ) || ( !xd && !xdinfo ) );

    // Always simplify

         if ( xb && !xa ) { return KK4ip(ib,ia,ic,id,bias,pxyprod,xb,xa,xc,xd,xbinfo,xainfo,xcinfo,xdinfo); }
    else if ( xc && !xa ) { return KK4ip(ic,ib,ia,id,bias,pxyprod,xc,xb,xa,xd,xcinfo,xbinfo,xainfo,xdinfo); }
    else if ( xc && !xb ) { return KK4ip(ia,ic,ib,id,bias,pxyprod,xa,xc,xb,xd,xainfo,xcinfo,xbinfo,xdinfo); }
    else if ( xd && !xa ) { return KK4ip(id,ib,ic,ia,bias,pxyprod,xd,xb,xc,xa,xdinfo,xbinfo,xcinfo,xainfo); }
    else if ( xd && !xb ) { return KK4ip(ia,id,ic,ib,bias,pxyprod,xa,xd,xc,xb,xainfo,xdinfo,xcinfo,xbinfo); }
    else if ( xd && !xc ) { return KK4ip(ia,ib,id,ic,bias,pxyprod,xa,xb,xd,xc,xainfo,xbinfo,xdinfo,xcinfo); }

    // Shortcut for speed

    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) && RFFordata(ic) && RFFordata(id) ) ? getRFFKernel() : getKernel() );

    if ( xa && xb && xc && xd )
    {
        const SparseVector<gentype> &xxa  = (*xa).n();
        const SparseVector<gentype> &xxxa = xxa.nup(0);

        const SparseVector<gentype> &xxb  = (*xb).n();
        const SparseVector<gentype> &xxxb = xxb.nup(0);

        const SparseVector<gentype> &xxc  = (*xc).n();
        const SparseVector<gentype> &xxxc = xxc.nup(0);

        const SparseVector<gentype> &xxd  = (*xd).n();
        const SparseVector<gentype> &xxxd = xxd.nup(0);

        return altK.K4ip(xxxa,xxxb,xxxc,xxxd,*xainfo,*xbinfo,*xcinfo,*xdinfo,bias,pxyprod,ia,ib,ic,id,0,0,MLid(),assumeReal);
    }

    else if ( xa && xb && xc )
    {
        const SparseVector<gentype> &xdb = x(id);

        const vecInfo &xinfod = xinfo(id);

        return KK4ip(ia,ib,ic,id,bias,pxyprod,xa,xb,xc,&xdb,xainfo,xbinfo,xcinfo,&xinfod);
    }

    else if ( xa && xb )
    {
        const SparseVector<gentype> &xcb = x(ic);
        const SparseVector<gentype> &xdb = x(id);

        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        return KK4ip(ia,ib,ic,id,bias,pxyprod,xa,xb,&xcb,&xdb,xainfo,xbinfo,&xinfoc,&xinfod);
    }

    else if ( xa && xa->isnofaroffindpresent() )
    {
        const SparseVector<gentype> &xbb = x(ib);
        const SparseVector<gentype> &xcb = x(ic);
        const SparseVector<gentype> &xdb = x(id);

        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        return KK4ip(ia,ib,ic,id,bias,pxyprod,xa,&xbb,&xcb,&xdb,xainfo,&xinfob,&xinfoc,&xinfod);
    }

        const SparseVector<gentype> &xab = x(ia);
        const SparseVector<gentype> &xbb = x(ib);
        const SparseVector<gentype> &xcb = x(ic);
        const SparseVector<gentype> &xdb = x(id);

        const vecInfo &xinfoa = xinfo(ia);
        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        return KK4ip(ia,ib,ic,id,bias,pxyprod,&xab,&xbb,&xcb,&xdb,&xinfoa,&xinfob,&xinfoc,&xinfod);
}

double ML_Base::KKmip(int m, Vector<int> &i, double bias, const gentype **pxyprod, Vector<const SparseVector<gentype> *> *xxx, Vector<const vecInfo *> *xxxinfo) const
{
//phantomx
    NiceAssert( m >= 0 );

    // Make sure all info are present

    int ii;

    Vector<const SparseVector<gentype> *> xx(*xxx);
    Vector<const vecInfo *> xxinfo(*xxxinfo);

    bool useRFFkernel = true;

    for ( ii = m-1 ; ii >= 0 ; --ii )
    {
        if ( !RFFordata(i(ii)) )
        {
            useRFFkernel = false;
        }

        if ( !xx(ii) )
        {
            // Fill in data

            xx("&",ii)     = &x(i(ii));
            xxinfo("&",ii) = &xinfo(i(ii));

            // Simplify data

            xx("&",ii) = &((*(xx(ii))).n());
            xx("&",ii) = &((*(xx(ii))).nup(0));
        }
    }

    const MercerKernel &altK = ( useRFFkernel ? getRFFKernel() : getKernel() );

    return altK.Kmip(m,xx,xxinfo,i,bias,pxyprod,xspaceDim(),isXConsistent() && istrv(i),MLid(),assumeReal);
}

template <class T>
void ML_Base::dK(T &xygrad, T &xnormgrad, 
                 int ia, int ib, 
                 const T &bias, const MercerKernel &basealtK, const gentype **pxyprod, 
                 const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, 
                 const vecInfo *xainfo, const vecInfo *xbinfo, 
                 int deepDeriv) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

//phantomx
//errstream() << "phantomxyggghhh 0\n";
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

        dK(xygrad,xnormgrad,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb,deepDeriv);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        dK(xygrad,xnormgrad,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo,deepDeriv);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

        dK(xygrad,xnormgrad,ia,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy,deepDeriv);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
//errstream() << "phantomxyggghhh 1\n";
        int dummyind = -1;

        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        altK.dK(xygrad,xnormgrad,dummyind,*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),x00,x10,x11,deepDeriv,assumeReal);
    }

    else
    {
        int dummyind = -1;

        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        SparseVector<gentype> *xbuntang = nullptr;
        vecInfo *xbinfountang = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);
        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        (void) loctanga; NiceAssert( !( loctanga & 2048 ) );
        (void) loctangb; NiceAssert( !( loctangb & 2048 ) );

        NiceAssert( !iadenseint && !iadensederiv );
        NiceAssert( !ibdenseint && !ibdensederiv );

        //int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        //NiceAssert( issameset );

        altK.dK(xygrad,xnormgrad,dummyind,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),nullptr,nullptr,nullptr,deepDeriv,assumeReal);

        if ( xai     ) { delete xai;     }
        if ( xainfoi ) { delete xainfoi; }

        if ( xbi     ) { delete xbi;     }
        if ( xbinfoi ) { delete xbinfoi; }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            xygrad    *= (T) UUres;
            xnormgrad *= (T) UUres;
        }

        NiceAssert( !iaplanr );
        NiceAssert( !ibplanr );
    }

    return;
}

template <class T>
void ML_Base::d2K(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, 
                  int ia, int ib, 
                  const T &bias, const MercerKernel &basealtK, const gentype **pxyprod,
                  const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, 
                  const vecInfo *xainfo, const vecInfo *xbinfo) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

//phantomx
//errstream() << "phantomxyggghhh 0\n";
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

        d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

        d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
//errstream() << "phantomxyggghhh 1\n";
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        altK.d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),x00,x10,x11,0,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        SparseVector<gentype> *xbuntang = nullptr;
        vecInfo *xbinfountang = nullptr;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);
        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        (void) loctanga; NiceAssert( !( loctanga & 2048 ) );
        (void) loctangb; NiceAssert( !( loctangb & 2048 ) );

        NiceAssert( !iadenseint && !iadensederiv );
        NiceAssert( !ibdenseint && !ibdensederiv );

        //int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        //NiceAssert( issameset );

        altK.d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),nullptr,nullptr,nullptr,0,assumeReal);

        if ( xai     ) { delete xai;     }
        if ( xainfoi ) { delete xainfoi; }

        if ( xbi     ) { delete xbi;     }
        if ( xbinfoi ) { delete xbinfoi; }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            xygrad         *= (T) UUres;
            xnormgrad      *= (T) UUres;
            xyxygrad       *= (T) UUres;
            xyxnormgrad    *= (T) UUres;
            xyynormgrad    *= (T) UUres;
            xnormxnormgrad *= (T) UUres;
            xnormynormgrad *= (T) UUres;
            ynormynormgrad *= (T) UUres;
        }

        NiceAssert( !iaplanr );
        NiceAssert( !ibplanr );
    }

    return;
}

template <class T>
void ML_Base::d2K2delxdelx(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, 
                          int ia, int ib, 
                          const T &bias, const MercerKernel &basealtK, const gentype **pxyprod, 
                          const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, 
                          const vecInfo *xainfo, const vecInfo *xbinfo) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

//phantomx
//errstream() << "phantomxyggghhh 0\n";
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

        d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

        d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
//errstream() << "phantomxyggghhh 1\n";
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        altK.d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),x00,x10,x11,0,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        SparseVector<gentype> *xbuntang = nullptr;
        vecInfo *xbinfountang = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);
        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        (void) loctanga; NiceAssert( !( loctanga & 2048 ) );
        (void) loctangb; NiceAssert( !( loctangb & 2048 ) );

        NiceAssert( !iadenseint && !iadensederiv );
        NiceAssert( !ibdenseint && !ibdensederiv );

        //int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        //NiceAssert( issameset );

        altK.d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),nullptr,nullptr,nullptr,0,assumeReal);

        if ( xai     ) { delete xai;     }
        if ( xainfoi ) { delete xainfoi; }

        if ( xbi     ) { delete xbi;     }
        if ( xbinfoi ) { delete xbinfoi; }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            xxscaleres *= (T) UUres;
            yyscaleres *= (T) UUres;
            xyscaleres *= (T) UUres;
            yxscaleres *= (T) UUres;
            constres   *= (T) UUres;
        }

        NiceAssert( !iaplanr );
        NiceAssert( !ibplanr );
    }

    return;
}


template <class T>
void ML_Base::d2K2delxdely(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, 
                          int ia, int ib, 
                          const T &bias, const MercerKernel &basealtK, const gentype **pxyprod, 
                          const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, 
                          const vecInfo *xainfo, const vecInfo *xbinfo) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

//phantomx
//errstream() << "phantomxyggghhh 0\n";
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

         d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

         d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

         d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ia,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
//errstream() << "phantomxyggghhh 1\n";
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        altK.d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),x00,x10,x11,0,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        SparseVector<gentype> *xbuntang = nullptr;
        vecInfo *xbinfountang = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);
        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        (void) loctanga; NiceAssert( !( loctanga & 2048 ) );
        (void) loctangb; NiceAssert( !( loctangb & 2048 ) );

        NiceAssert( !iadenseint && !iadensederiv );
        NiceAssert( !ibdenseint && !ibdensederiv );

        //int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        //NiceAssert( issameset );

        altK.d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),nullptr,nullptr,nullptr,0,assumeReal);

        if ( xai     ) { delete xai;     }
        if ( xainfoi ) { delete xainfoi; }

        if ( xbi     ) { delete xbi;     }
        if ( xbinfoi ) { delete xbinfoi; }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            xxscaleres *= (T) UUres;
            yyscaleres *= (T) UUres;
            xyscaleres *= (T) UUres;
            yxscaleres *= (T) UUres;
            constres   *= (T) UUres;
        }

        NiceAssert( !iaplanr );
        NiceAssert( !ibplanr );
    }

    return;
}

template <class T>
void ML_Base::dnK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                     const Vector<int> &q, 
                     int ia, int ib, 
                     const T &bias, const MercerKernel &basealtK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, 
                     const vecInfo *xainfo, const vecInfo *xbinfo) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

//phantomx
//errstream() << "phantomxyggghhh 0\n";
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

        dnK2del(sc,n,minmaxind,q,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        dnK2del(sc,n,minmaxind,q,ia,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

        dnK2del(sc,n,minmaxind,q,ia,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
//errstream() << "phantomxyggghhh 1\n";
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        altK.dnK2del(sc,n,minmaxind,q,*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),x00,x10,x11,0,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        SparseVector<gentype> *xbuntang = nullptr;
        vecInfo *xbinfountang = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);
        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        (void) loctanga; NiceAssert( !( loctanga & 2048 ) );
        (void) loctangb; NiceAssert( !( loctangb & 2048 ) );

        NiceAssert( !iadenseint && !iadensederiv );
        NiceAssert( !ibdenseint && !ibdensederiv );

        //int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        //NiceAssert( issameset );

        altK.dnK2del(sc,n,minmaxind,q,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),nullptr,nullptr,nullptr,0,assumeReal);

        if ( xai     ) { delete xai;     }
        if ( xainfoi ) { delete xainfoi; }

        if ( xbi     ) { delete xbi;     }
        if ( xbinfoi ) { delete xbinfoi; }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            sc *= (T) UUres;
        }

        NiceAssert( !iaplanr );
        NiceAssert( !ibplanr );
    }

    return;
}

template <class T>
void ML_Base::dK2delx(T &xscaleres, T &yscaleres, int &minmaxind, 
                     int ia, int ib, 
                     const T &bias, const MercerKernel &basealtK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, 
                     const vecInfo *xainfo, const vecInfo *xbinfo) const
{
    const MercerKernel &altK = ( ( RFFordata(ia) && RFFordata(ib) ) ? getRFFKernel() : basealtK );

//phantomx
//errstream() << "phantomxyggghhh 0\n";
    if ( !xa )
    {
        xa     = &x(ia);
        xainfo = &xinfo(ia);
    }

    if ( !xb )
    {
        xb     = &x(ib);
        xbinfo = &xinfo(ib);
    }

    if ( !xainfo && !xbinfo )
    {
        vecInfo xinfoxa;
        vecInfo xinfoxb;

        getKernel().getvecInfo(xinfoxa,*xa);
        getKernel().getvecInfo(xinfoxb,*xb);

        dK2delx(xscaleres,yscaleres,minmaxind,ib,ib,bias,basealtK,pxyprod,xa,xb,&xinfoxa,&xinfoxb);
    }

    else if ( !xainfo )
    {
        vecInfo xinfox;

        getKernel().getvecInfo(xinfox,*xa);

        dK2delx(xscaleres,yscaleres,minmaxind,ib,ib,bias,basealtK,pxyprod,xa,xb,&xinfox,xbinfo);
    }

    else if ( !xbinfo )
    {
        vecInfo xinfoy;

        getKernel().getvecInfo(xinfoy,*xb);

        dK2delx(xscaleres,yscaleres,minmaxind,ib,ib,bias,basealtK,pxyprod,xa,xb,xainfo,&xinfoy);
    }

    else if ( xa->isnofaroffindpresent() && xb->isnofaroffindpresent() )
    {
//errstream() << "phantomxyggghhh 1\n";
        const double *x00 = nullptr;
        const double *x10 = nullptr;
        const double *x11 = nullptr;

        if ( xa && xb && isxymat(altK) && ( ia >= 0 ) && ( ib >= 0 ) && ( (*xa).nupsize() == 1 ) && ( (*xb).nupsize() == 1 ) )
        {
            x00 = &(getxymatelm(altK,ia,ia));
            x10 = &(getxymatelm(altK,ib,ia));
            x11 = &(getxymatelm(altK,ib,ib));
        }

        altK.dK2delx(xscaleres,yscaleres,minmaxind,*xa,*xb,*xainfo,*xbinfo,bias,pxyprod,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),x00,x10,x11,assumeReal);
    }

    else
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        vecInfo *xainfountang = nullptr;

        SparseVector<gentype> *xbuntang = nullptr;
        vecInfo *xbinfountang = nullptr;

        int ixa,iia,xalr,xarr,xagr,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        int xagrR,agradOrderR,agmuL,agmuR;
        int xbgrR,bgradOrderR,bgmuL,bgmuR;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);
        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        (void) loctanga; NiceAssert( !( loctanga & 2048 ) );
        (void) loctangb; NiceAssert( !( loctangb & 2048 ) );

        NiceAssert( !iadenseint && !iadensederiv );
        NiceAssert( !ibdenseint && !ibdensederiv );

        //int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;

        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        //NiceAssert( issameset );

        altK.dK2delx(xscaleres,yscaleres,minmaxind,*xai,*xbi,*xainfoi,*xbinfoi,bias,nullptr,ia,ib,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib),MLid(),nullptr,nullptr,nullptr,assumeReal);

        if ( xai     ) { delete xai;     }
        if ( xainfoi ) { delete xainfoi; }

        if ( xbi     ) { delete xbi;     }
        if ( xbinfoi ) { delete xbinfoi; }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            xscaleres *= (T) UUres;
            yscaleres *= (T) UUres;
        }

        NiceAssert( !iaplanr );
        NiceAssert( !ibplanr );
    }

    return;
}

void ML_Base::densedKdx(double &res, int i, int j, double bias) const
{
    const MercerKernel &altK = ( ( RFFordata(i) && RFFordata(j) ) ? getRFFKernel() : getKernel() );

//phantomx
    //FIXME: implement gradients and rank

    altK.densedKdx(res,x(i),x(j),xinfo(i),xinfo(j),bias,i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),assumeReal);

    return;
}

void ML_Base::denseintK(double &res, int i, int j, double bias) const
{
    const MercerKernel &altK = ( ( RFFordata(i) && RFFordata(j) ) ? getRFFKernel() : getKernel() );

//phantomx
    //FIXME: implement gradients and rank

    altK.denseintK(res,x(i),x(j),xinfo(i),xinfo(j),bias,i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),assumeReal);

    return;
}

double ML_Base::distK(int i, int j) const
{
    const MercerKernel &altK = ( ( RFFordata(i) && RFFordata(j) ) ? getRFFKernel() : getKernel() );

//phantomx
    //FIXME: implement gradients and rank

    return altK.distK(x(i),x(j),xinfo(i),xinfo(j),i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),nullptr,0,0,assumeReal);
}

void ML_Base::ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const
{
    const MercerKernel &altK = ( ( RFFordata(i) && RFFordata(j) ) ? getRFFKernel() : getKernel() );

//phantomx
    //FIXME: implement gradients and rank

    altK.ddistKdx(xscaleres,yscaleres,minmaxind,x(i),x(j),xinfo(i),xinfo(j),i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),nullptr,0,0,assumeReal);

    return;
}

































const gentype &ML_Base::xelm(gentype &res, int i, int j) const
{
    const MercerKernel &altK = ( ( RFFordata(i) && RFFordata(j) ) ? getRFFKernel() : getKernel() );

//phantomx
    //FIXME: implement gradients and rank

    return altK.xelm(res,x(i),i,j);
}

int ML_Base::xindsize(int i) const
{
    const MercerKernel &altK = ( RFFordata(i) ? getRFFKernel() : getKernel() );

//phantomx
    //FIXME: implement gradients and rank

    return altK.xindsize(x(i),i);
}
























void ML_Base::xferx(const ML_Base &xsrc)
{
    incxvernum();
    incgvernum();

    NiceAssert( xsrc.N() == N() );

    allxdatagent = xsrc.allxdatagent;

    getKernel_unsafe().setIPdiffered(1);

    resetKernel(1,-1);

    return;
}

















































void ML_Base::dgTrainingVector(Vector<double> &res, double &resn, int i) const
{
    Vector<gentype> tres;
    gentype tresn;

    dgTrainingVector(tres,tresn,i);

    res.resize(tres.size());

    int j;

    for ( j = 0 ; j < tres.size() ; ++j )
    {
        res("&",j) = (double) tres(j);
    }

    resn = (double) tresn;

    return;
}

void ML_Base::dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const
{
    Vector<gentype> tres;
    gentype tresn;

    dgTrainingVector(tres,tresn,i);

    res.resize(tres.size());

    int j;

    for ( j = 0 ; j < tres.size() ; ++j )
    {
        res("&",j) = (const Vector<double> &) tres(j);
    }

    resn = (const Vector<double> &) tresn;

    return;
}

void ML_Base::dgTrainingVector(Vector<d_anion> &res, d_anion &resn, int i) const
{
    Vector<gentype> tres;
    gentype tresn;

    dgTrainingVector(tres,tresn,i);

    res.resize(tres.size());

    int j;

    for ( j = 0 ; j < tres.size() ; ++j )
    {
        res("&",j) = (const d_anion &) tres(j);
    }

    resn = (const d_anion &) tresn;

    return;
}


void ML_Base::dgTrainingVectorX(Vector<double> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;

    NiceAssert( i >= 0 );

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = 0.0;

    int j,k;

    for ( k = 0 ; k < x(i).nindsize() ; ++k )
    {
        resx("&",x(i).ind(k)) += (double) ((resn*(x(i)).direcref(k)));
    }

    //if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; ++j )
        {
            //if ( x(j).nindsize() )
            {
                for ( k = 0 ; k < x(j).nindsize() ; ++k )
                {
                    resx("&",x(j).ind(k)) += (double) ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

void ML_Base::dgTrainingVectorX(Vector<gentype> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;

    NiceAssert( i >= 0 );

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = 0.0_gent;

    int j,k;

    for ( k = 0 ; k < x(i).nindsize() ; ++k )
    {
        resx("&",x(i).ind(k)) += ((resn*(x(i)).direcref(k)));
    }

    //if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; ++j )
        {
            //if ( x(j).nindsize() )
            {
                for ( k = 0 ; k < x(j).nindsize() ; ++k )
                {
                    resx("&",x(j).ind(k)) += ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

void ML_Base::deTrainingVectorX(Vector<gentype> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;

    NiceAssert( i >= 0 );

    deTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = 0.0_gent;

    int j,k;

    for ( k = 0 ; k < x(i).nindsize() ; ++k )
    {
        resx("&",x(i).ind(k)) += ((resn*(x(i)).direcref(k)));
    }

    //if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; ++j )
        {
            //if ( x(j).nindsize() )
            {
                for ( k = 0 ; k < x(j).nindsize() ; ++k )
                {
                    resx("&",x(j).ind(k)) += ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

void ML_Base::dgTrainingVectorX(Vector<double> &resx, const Vector<int> &i) const
{
    Vector<gentype> res;

    NiceAssert( i >= 0 );

    dgTrainingVector(res,i);

    resx.resize(xspaceDim()) = 0.0;

    int j,k;

    //if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; ++j )
        {
            //if ( x(j).nindsize() )
            {
                for ( k = 0 ; k < x(j).nindsize() ; ++k )
                {
                    resx("&",x(j).ind(k)) += (double) ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

void ML_Base::dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const
{
    Vector<gentype> res;

    NiceAssert( i >= 0 );

    dgTrainingVector(res,i);

    resx.resize(xspaceDim()) = 0.0_gent;

    int j,k;

    //if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; ++j )
        {
            //if ( x(j).nindsize() )
            {
                for ( k = 0 ; k < x(j).nindsize() ; ++k )
                {
                    resx("&",x(j).ind(k)) += ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

void ML_Base::deTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const
{
    Vector<gentype> res;

    NiceAssert( i >= 0 );

    deTrainingVector(res,i);

    resx.resize(xspaceDim()) = 0.0_gent;

    int j,k;

    //if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; ++j )
        {
            //if ( x(j).nindsize() )
            {
                for ( k = 0 ; k < x(j).nindsize() ; ++k )
                {
                    resx("&",x(j).ind(k)) += ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

void ML_Base::dgTrainingVector(Vector<gentype> &res, const Vector<int> &i) const
{
    NiceAssert( i >= 0 );

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            Vector<gentype> qres;
            gentype qresn;

            if ( !j )
            {
                dgTrainingVector(res,qresn,i(j));

                res("&",i(j)) += qresn;
            }

            else
            {
                dgTrainingVector(qres,qresn,i(j));

                res           += qres;
                res("&",i(j)) += qresn;
            }
        }

        for ( j = 0 ; j < res.size() ; ++j )
        {
            res("&",i(j)) *= 1.0/((double) i.size());
        }
    }

    else
    {
        res.resize(ML_Base::N());

        res = 0.0_gent;
    }

    return;
}

void ML_Base::deTrainingVector(Vector<gentype> &res, const Vector<int> &i) const
{
    NiceAssert( i >= 0 );

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            Vector<gentype> qres;
            gentype qresn;

            if ( !j )
            {
                deTrainingVector(res,qresn,i(j));

                res("&",i(j)) += qresn;
            }

            else
            {
                deTrainingVector(qres,qresn,i(j));

                res           += qres;
                res("&",i(j)) += qresn;
            }
        }

        for ( j = 0 ; j < res.size() ; ++j )
        {
            res("&",i(j)) *= 1.0/((double) i.size());
        }
    }

    else
    {
        res.resize(ML_Base::N());

        res = 0.0_gent;
    }

    return;
}

void ML_Base::dgTrainingVector(Vector<double> &res, const Vector<int> &i) const
{
    NiceAssert( i >= 0 );

    Vector<gentype> qres;

    dgTrainingVector(qres,i);

    res.resize(qres.size());

    int j;

    for ( j = 0 ; j < qres.size() ; ++j )
    {
        res("&",j) = (double) qres(j);
    }

    return;
}

void ML_Base::dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const
{
    NiceAssert( i >= 0 );

    Vector<gentype> qres;

    dgTrainingVector(qres,i);

    res.resize(qres.size());

    int j;

    for ( j = 0 ; j < qres.size() ; ++j )
    {
        res("&",j) = qres(j).cast_vector_real();
    }

    return;
}

void ML_Base::dgTrainingVector(Vector<d_anion> &res, const Vector<int> &i) const
{
    NiceAssert( i >= 0 );

    Vector<gentype> qres;

    dgTrainingVector(qres,i);

    res.resize(qres.size());

    int j;

    for ( j = 0 ; j < qres.size() ; ++j )
    {
        res("&",j) = qres(j).cast_anion();
    }

    return;
}

int ML_Base::covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const
{
    NiceAssert( ( i.size() == 0 ) || ( i >= 0 ) );

    int m = i.size();
    int res = 0;

    resv.resize(m,m);

    gentype dummy;

    //if ( m )
    {
        int ii,jj;

        for ( ii = 0 ; ii < m ; ++ii )
        {
            for ( jj = 0 ; jj < m ; ++jj )
            {
                res |= covTrainingVector(resv("&",ii,jj),dummy,i(ii),i(jj));
            }
        }
    }

    return 0;
}

int ML_Base::covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &xx) const
{
    int m = xx.size();
    int res = 0;

    resv.resize(m,m);

    gentype dummy;

    //if ( m )
    {
        int ii,jj;

        for ( ii = 0 ; ii < m ; ++ii )
        {
            for ( jj = 0 ; jj < m ; ++jj )
            {
                res |= cov(resv("&",ii,jj),dummy,xx(ii),xx(jj));
            }
        }
    }

    return 0;
}

int ML_Base::noisevarTrainingVector(gentype &resv, gentype &resmu, int i, const SparseVector<gentype> &xvar, int u, gentype ***pxyprodi, gentype **pxyprodii) const
{
// NB: code is repeated in GPR_Generic

    int res = varTrainingVector(resv,resmu,i,pxyprodi,pxyprodii);

    SparseVector<gentype> ximod(x(i));

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

            res |= gg(gradiu,ximod);

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

/*
int ML_Base::noisevar(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xvar, int u, const vecInfo *xainf, gentype ***pxyprodx, gentype **pxyprodxx) const
{
// NB: code is repeated in GPR_Generic

    int res = var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx);

    SparseVector<gentype> ximod(xa);

    ximod.f4("&",6) = ((int) ximod.f4(6))+1; // gradient is in addition to what is already there!

    int ui = ( u == -1 ) ? 0 : ximod.nupsize();

    int umin = ( u == -1 ) ? 0 : u;
    int umax = ( u == -1 ) ? ui-1 : u;

    for ( u = umin ; u <= umax ; ++u )
    {
        if ( xvar.nupsize(u) && ximod.nupsize(u) )
        {
            // Gradient wrt xi

            ximod.f4("&",9) = u; // with respect to xu in [ x0 ~ x1 ~ ... ]

            gentype gradiu;

            res |= gg(gradiu,ximod);

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
*/

int ML_Base::noisecovTrainingVector(gentype &resv, gentype &resmu, int i, int j, const SparseVector<gentype> &xvar, int u, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
// NB: code is repeated in GPR_Generic

    int res = covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij);

    SparseVector<gentype> ximod(x(i));
    SparseVector<gentype> xjmod(x(j));

    ximod.f4("&",6) = ((int) ximod.f4(6))+1; // gradient is in addition to what is already there!
    xjmod.f4("&",6) = ((int) xjmod.f4(6))+1;

    int ui = ( u == -1 ) ? 1 : ximod.nupsize();
    int uj = ( u == -1 ) ? 1 : xjmod.nupsize();

    int uij = ( ui > uj ) ? ui : uj;

    int umin = ( u == -1 ) ? 0 : ( ( u == -2 ) ? 1 : u );
    int umax = ( u < 0 ) ? uij-1 : u;

    for ( u = umin ; u <= umax ; ++u )
    {
        if ( xvar.nupsize(u) && ximod.nupsize(u) && xjmod.nupsize(u) )
        {
            // Gradient wrt xi

            ximod.f4("&",9) = u; // with respect to xu in [ x0 ~ x1 ~ ... ]
            xjmod.f4("&",9) = u;

            gentype gradiu;
            gentype gradju;

            res |= gg(gradiu,ximod);
            res |= gg(gradju,xjmod);

            // Assuming non-sparse data here!

            retVector<gentype> tmpvi;
            retVector<gentype> tmpvj;
            retVector<gentype> tmpva;

            const Vector<gentype> &xvaru = xvar.nup(u)(tmpva);
            const Vector<gentype> &gradiuv = (gradiu.cast_vector())(0,1,xvaru.size()-1,tmpvi);
            const Vector<gentype> &gradjuv = (gradju.cast_vector())(0,1,xvaru.size()-1,tmpvj);

            gentype tmp;

            resv += threeProduct(tmp,gradiuv,xvaru,gradjuv);
        }
    }

    return res;
}

/*
int ML_Base::noisecov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xvar, int u, const vecInfo *xainf, const vecInfo *xbinf, gentype ***pxyprodx, gentype ***pxyprody, gentype **pxyprodxy) const
{
// NB: code is repeated in GPR_Generic

    int res = cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodxy);

    SparseVector<gentype> ximod(xa);
    SparseVector<gentype> xjmod(xb);

    ximod.f4("&",6) = ((int) ximod.f4(6))+1; // gradient is in addition to what is already there!
    xjmod.f4("&",6) = ((int) xjmod.f4(6))+1;

    int ui = ( u == -1 ) ? 0 : ximod.nupsize();
    int uj = ( u == -1 ) ? 0 : xjmod.nupsize();

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

            res |= gg(gradiu,ximod);
            res |= gg(gradju,xjmod);

            // Assuming non-sparse data here!

            retVector<gentype> tmpvi;
            retVector<gentype> tmpvj;
            retVector<gentype> tmpva;

            const Vector<gentype> &xvaru = xvar.nup(u)(tmpva);
            const Vector<gentype> &gradiuv = (gradiu.cast_vector())(0,1,xvaru.size()-1,tmpvi);
            const Vector<gentype> &gradjuv = (gradju.cast_vector())(0,1,xvaru.size()-1,tmpvj);

            gentype tmp;

            resv += threeProduct(tmp,gradiuv,xvaru,gradjuv);
        }
    }

    return res;
}
*/



















// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================

gentype &UUcallbacknon(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis)
{
    (void) m;
    (void) caller;
    (void) iokr;
    (void) iok;
    (void) xalt;
    (void) defbasis;

    res = 1.0;

    return res;
}

gentype &UUcallbackdef(gentype &res, int mm, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis)
{
    NiceAssert( iokr.size() == mm );
    NiceAssert( iok.size()  == mm );
    NiceAssert( xalt.size() == mm );

    int m = mm;
    int i;

    if ( ( sum(iokr) > 0 ) || ( defbasis >= 0 ) )
    {
        Vector<SparseVector<gentype> > x(m);
        Vector<vecInfo> xinfo(m);

        for ( i = m-1 ; i >= 0 ; --i )
        {
            if ( !iokr(i) && ( defbasis >= 0 ) )
            {
                iokr("&",i) = 1;
                iok("&",i)  = defbasis;
            }

            if ( iokr(i) )
            {
                if ( iok(i) > 0 ) { x("&",i)("&",0) = (caller.VbasisUU())(iok(i)); caller.getUUOutputKernel().getvecInfo(xinfo("&",i),x(i)); }
                else              { x("&",i)("&",0) = (*(xalt(i)));                caller.getUUOutputKernel().getvecInfo(xinfo("&",i),x(i)); }
            }

            else
            {
                iok.remove(i);
                iokr.remove(i);
                xalt.remove(i);
                x.remove(i);
                xinfo.remove(i);

                --m;
            }
        }

        if ( m == mm )
        {
            int z = 0;

            if ( m == 0 )
            {
                caller.getUUOutputKernel().K0(res,0_gent,nullptr,0,0,0,caller.MLid());
            }

            else if ( m == 1 )
            {
                caller.getUUOutputKernel().K1(res,x(z),xinfo(z),0_gent,nullptr,iok(z),0,0,0,caller.MLid(),nullptr);
            }

            else if ( m == 2 )
            {
                caller.getUUOutputKernel().K2(res,x(z),x(1),xinfo(z),xinfo(1),0_gent,nullptr,iok(z),iok(1),0,0,0,caller.MLid(),nullptr,nullptr,nullptr);
            }

            else if ( m == 3 )
            {
                caller.getUUOutputKernel().K3(res,x(z),x(1),x(2),xinfo(z),xinfo(1),xinfo(2),0_gent,nullptr,iok(z),iok(1),iok(2),0,0,0,caller.MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
            }

            else if ( m == 4 )
            {
                caller.getUUOutputKernel().K4(res,x(z),x(1),x(2),x(3),xinfo(z),xinfo(1),xinfo(2),xinfo(3),0_gent,nullptr,iok(z),iok(1),iok(2),iok(3),0,0,0,caller.MLid(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
            }

            else if ( m >= 5 )
            {
                Vector<const SparseVector<gentype> *> xx(m);
                Vector<const vecInfo *> xxinfo(m);
                Vector<int> ii(m);

                for ( i = 0 ; i < m ; ++i )
                {
                    xx("&",i) = &(x(i));
                    xxinfo("&",i) = &(xinfo(i));
                    ii("&",i) = iok(i);
                }

                caller.getUUOutputKernel().Km(m,res,xx,xxinfo,0_gent,ii,nullptr,0,0,0,caller.MLid(),nullptr);
            }
        }

        else
        {
            NiceAssert( caller.getUUOutputKernel().isSimpleLinearKernel() );

            if ( m == 0 )
            {
                res = 1.0;
            }

            else if ( m == 1 )
            {
                res = x(0)(0);
            }

            else if ( m == 2 )
            {
                res = x(0)(0);
                res = emul(res,x(1)(0));
            }

            else if ( m == 3 )
            {
                res = x(0)(0);
                res = emul(res,x(1)(0));
                res = emul(res,x(2)(0));
            }

            else if ( m == 4 )
            {
                res = x(0)(0);
                res = emul(res,x(1)(0));
                res = emul(res,x(2)(0));
                res = emul(res,x(3)(0));
            }

            else if ( m >= 5 )
            {
                res = x(0)(0);

                for ( i = 1 ; i < m ; ++i )
                {
                    res = emul(res,x(i)(0));
                }
            }
        }
    }

    else
    {
        res = 1.0; // This will never actually be reached.
    }

    return res;
}

const gentype &VVcallbacknon(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis)
{
    (void) m;
    (void) kval;
    (void) caller;
    (void) iokr;
    (void) iok;
    (void) xalt;
    (void) defbasis;

    res = kval;

    return kval;
}

const gentype &VVcallbackdef(gentype &res, int mm, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis)
{
    NiceAssert( iokr.size() == mm );
    NiceAssert( iok.size()  == mm );
    NiceAssert( xalt.size() == mm );

    int m = mm;
    int i;

    res = kval;

    if ( mm && ( ( sum(iokr) > 0 ) || ( defbasis >= 0 ) ) )
    {
        Vector<gentype> s(m);

//        bool evalit = true;

        for ( i = m-1 ; i >= 0 ; --i )
        {
            if ( !iokr(i) && ( defbasis >= 0 ) )
            {
                iokr("&",i) = 1;
                iok("&",i)  = defbasis;
            }

            if ( iokr(i) )
            {
                if ( iok(i) > 0 ) { s("&",i) = (caller.VbasisVV())(iok(i)); }
                else              { s("&",i) = (*(xalt(i))); }
            }

            else
            {
//NO                // For non-planar types, if at least one of the arguments
//NO                // does not have a VVbasis argument then we don't evaluate
//NO                // this at all.  The reasoning here is that VVbasis is
//NO                // most likely being used to simulate multitask learning
//NO                // in this case, and the random feature vectors don't have
//NO                // a v component
//
//                if ( !caller.isPlanarType() )
//                {
//                    evalit = false;
//                    break;
//                }

                iok.remove(i);
                iokr.remove(i);
                xalt.remove(i);
                s.remove(i);

                --m;
            }
        }

//        if ( m && evalit )
        //if ( m )
        {
            for ( i = 0 ; i < m ; ++i )
            {
                res *= s(i);
            }
        }
    }

    return res;
}






int ML_Base::addToBasisUU(int i, const gentype &o)
{
    NiceAssert( !isBasisUserUU );
    NiceAssert( i >= 0 );
    NiceAssert( i <= NbasisUU() );

    // Add to basis set

    locbasisUU.add(i);
    locbasisUU("&",i) = o;

    return 1;
}

int ML_Base::removeFromBasisUU(int i)
{
    NiceAssert( !isBasisUserUU );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisUU() );

    // Remove from basis set

    locbasisUU.remove(i);

    return 1;
}

int ML_Base::setBasisUU(int i, const gentype &o)
{
    NiceAssert( !isBasisUserUU );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisUU() );

    locbasisUU("&",i) = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisUU(const Vector<gentype> &o)
{
    NiceAssert( !isBasisUserUU );

    locbasisUU = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisUU(int n, int d)
{
    NiceAssert( n >= 0 );
    NiceAssert( d >= 0 );

    Vector<gentype> o(n);
    Vector<double> ubase(d);

    int i;

    //if ( n )
    {
        for ( i = 0 ; i < n ; ++i )
        {
            if ( d > 0 )
            {
                ubase = 0.0;

                while ( sum(ubase) == 0 )
                {
                    randufill(ubase);
                }

                ubase /= sum(ubase);
            }

            o("&",i) = ubase;
        }
    }

    return setBasisUU(o);
}

int ML_Base::setBasisYUU(void)
{
    int res = 0;

    if ( !isBasisUserUU )
    {
        setBasisUU(y());
        isBasisUserUU = 1;
        res = 1;
    }

    return res;
}

int ML_Base::setBasisUUU(void)
{
    int res = 0;

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        res = 1;
    }

    return res;
}







int ML_Base::addToBasisVV(int i, const gentype &o)
{
    NiceAssert( !isBasisUserVV );
    NiceAssert( i >= 0 );
    NiceAssert( i <= NbasisVV() );

    // Add to basis set

    locbasisVV.add(i);
    locbasisVV("&",i) = o;

    return 1;
}

int ML_Base::removeFromBasisVV(int i)
{
    NiceAssert( !isBasisUserVV );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisVV() );

    // Remove from basis set

    locbasisVV.remove(i);

    return 1;
}

int ML_Base::setBasisVV(int i, const gentype &o)
{
    NiceAssert( !isBasisUserVV );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisVV() );

    locbasisVV("&",i) = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisVV(const Vector<gentype> &o)
{
    NiceAssert( !isBasisUserVV );

    locbasisVV = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisVV(int n, int d)
{
    NiceAssert( n >= 0 );
    NiceAssert( d >= 0 );

    Vector<gentype> o(n);
    Vector<double> ubase(d);

    int i;

    //if ( n )
    {
        for ( i = 0 ; i < n ; ++i )
        {
            if ( d > 0 )
            {
                ubase = 0.0;

                while ( sum(ubase) == 0 )
                {
                    randufill(ubase);
                }

                ubase /= sum(ubase);
            }

            o("&",i) = ubase;
        }
    }

    return setBasisVV(o);
}

int ML_Base::setBasisYVV(void)
{
    int res = 0;

    if ( !isBasisUserVV )
    {
        setBasisVV(y());
        isBasisUserVV = 1;
        res = 1;
    }

    return res;
}

int ML_Base::setBasisUVV(void)
{
    int res = 0;

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        res = 1;
    }

    return res;
}
































int ML_Base::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int ML_Base::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    int res = 0;

    desc = "";

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    gentype dummy;

    if ( ( ind <= 499 ) && ( !ia && !ib ) )
    {
        SparseVector<gentype> res;

        switch ( ind )
        {
            case  0: { val = C();                            desc = "ML_Base::C"; break; }
            case  1: { val = eps();                          desc = "ML_Base::eps";                break; }
            case  2: { val = sigma();                        desc = "ML_Base::sigma";              break; }
            case  3: { val = betarank();                     desc = "ML_Base::betarank";           break; }
            case  4: { val = tspaceDim();                    desc = "ML_Base::tspaceDim";          break; }
            case  5: { val = order();                        desc = "ML_Base::order";              break; }
            case  6: { val = sparlvl();                      desc = "ML_Base::sparlvl";            break; }
            case  7: { val = xspaceDim();                    desc = "ML_Base::xspaceDim";          break; }
            case  8: { val = tspaceSparse();                 desc = "ML_Base::tspaceSparse";       break; }
            case  9: { val = xspaceSparse();                 desc = "ML_Base::xspaceSparse";       break; }
            case 10: { val = N();                            desc = "ML_Base::N";                  break; }
            case 11: { val = type();                         desc = "ML_Base::type";               break; }
            case 12: { val = subtype();                      desc = "ML_Base::subtype";            break; }
            case 13: { val = numClasses();                   desc = "ML_Base::numClasses";         break; }
            case 14: { val = isTrained();                    desc = "ML_Base::isTrained";          break; }
            case 15: { val = isMutable();                    desc = "ML_Base::isMutable";          break; }
            case 16: { val = isPool();                       desc = "ML_Base::isPool";             break; }
            case 17: { val = isUnderlyingScalar();           desc = "ML_Base::isUnderlyingScalar"; break; }
            case 18: { val = isUnderlyingVector();           desc = "ML_Base::isUnderlyingVector"; break; }
            case 19: { val = isUnderlyingAnions();           desc = "ML_Base::isUnderlyingAnions"; break; }
            case 20: { val = isClassifier();                 desc = "ML_Base::isClassifer";        break; }
            case 21: { val = isRegression();                 desc = "ML_Base::isRegression";       break; }
            case 22: { val = numInternalClasses();           desc = "ML_Base::numInternalClasses"; break; }
            case 23: { val = NbasisUU();                     desc = "ML_Base::NbasisUU";           break; }
            case 24: { val = basisTypeUU();                  desc = "ML_Base::basesTypeUU";        break; }
            case 25: { val = defProjUU();                    desc = "ML_Base::defProjUU";          break; }
            case 26: { val.force_string() = gOutType();      desc = "ML_Base::gOutType";           break; }
            case 27: { val.force_string() = hOutType();      desc = "ML_Base::hOutType";           break; }
            case 28: { val.force_string() = targType();      desc = "ML_Base::targType";           break; }
            case 29: { val = ClassLabels();                  desc = "ML_Base::ClassLabels";        break; }
            case 30: { val = zerotol();                      desc = "ML_Base::zerotol";            break; }
            case 31: { val = Opttol();                       desc = "ML_Base::Opttol";             break; }
            case 32: { val = maxtraintime();                 desc = "ML_Base::maxtraintime";       break; }
            case 33: { val = lrmvrank();                     desc = "ML_Base::lrmvrank";           break; }
            case 34: { val = ztmvrank();                     desc = "ML_Base::ztmvrank";           break; }
            case 35: { val = memsize();                      desc = "ML_Base::memsize";            break; }
            case 36: { val = maxitcnt();                     desc = "ML_Base::maxitcnt";           break; }
            case 37: { val = maxitermvrank();                desc = "ML_Base::maxitermvrank";      break; }
            case 38: { val = y();                            desc = "ML_Base::y";                  break; }
            case 39: { val = d();                            desc = "ML_Base::d";                  break; }
            case 40: { val = Cweight();                      desc = "ML_Base::Cweight";            break; }
            case 41: { val = epsweight();                    desc = "ML_Base::epsweight";          break; }
            case 42: { val = alphaState();                   desc = "ML_Base::alphaState";         break; }
            case 43: { val = Cweightfuzz();                  desc = "ML_Base::Cweightfuzz";        break; }
            case 44: { val = sigmaweight();                  desc = "ML_Base::sigmaweight";        break; }
            case 45: { val = VbasisUU();                     desc = "ML_Base::VbasisUU";           break; }
            case 46: { val = indKey();                       desc = "ML_Base::indKey";             break; }
            case 47: { val = indKeyCount();                  desc = "ML_Base::indKeyCount";        break; }
            case 48: { val = dattypeKey();                   desc = "ML_Base::dattypeKey";         break; }
            case 49: { val = dattypeKeyBreak();              desc = "ML_Base::dattypeKeybreak";    break; }
            case 50: { convertSparseToSet(val,xsum(res));    desc = "ML_Base::xsum";               break; }
            case 51: { convertSparseToSet(val,xmean(res));   desc = "ML_Base::xmean";              break; }
            case 52: { convertSparseToSet(val,xmeansq(res)); desc = "ML_Base::xmeansq";            break; }
            case 53: { convertSparseToSet(val,xsqsum(res));  desc = "ML_Base::xsqsum";             break; }
            case 54: { convertSparseToSet(val,xsqmean(res)); desc = "ML_Base::xsqmean";            break; }
            case 55: { convertSparseToSet(val,xmedian(res)); desc = "ML_Base::xmedian";            break; }
            case 56: { convertSparseToSet(val,xvar(res));    desc = "ML_Base::xvar";               break; }
            case 57: { convertSparseToSet(val,xstddev(res)); desc = "ML_Base::xstddev";            break; }
            case 59: { convertSparseToSet(val,xmax(res));    desc = "ML_Base::xmax";               break; }
            case 60: { convertSparseToSet(val,xmin(res));    desc = "ML_Base::xmin";               break; }
            case 61: { val = NbasisVV();                     desc = "ML_Base::NbasisVV";           break; }
            case 62: { val = basisTypeVV();                  desc = "ML_Base::basisTypeVV";        break; }
            case 63: { val = defProjVV();                    desc = "ML_Base::defProjVV";          break; }
            case 65: { val = VbasisVV();                     desc = "ML_Base::VbasisVV";           break; }
            case 66: { val = traintimeend();                 desc = "ML_Base::traintimeend";       break; }
            case 67: { getvecformc(val);                     desc = "ML_Base::getvecforma";        break; }

            case 100: { val = Cclass((int) xa);              desc = "ML_Base::Cclass";             break; }
            case 101: { val = epsclass((int) xa);            desc = "ML_Base::epsclass";           break; }
            case 102: { val = isenabled((int) xa);           desc = "ML_Base::isenabled";          break; }
            case 103: { val = NNC((int) xa);                 desc = "ML_Base::NNC";                break; }
            case 104: { val = isenabled((int) xa);           desc = "ML_Base::isenabled";          break; }
            case 105: { val = getInternalClass(xa);          desc = "ML_Base::getInternalClass";   break; }
            case 106: { convertSparseToSet(val,x((int) xa)); desc = "ML_Base::convertSparseToSet"; break; }

            case 200: { val = calcDist(xa,xb);  desc = "ML_Base::calcDist"; break; }

            case 300: { ggTrainingVector(val,(int) xa); desc = "ML_Base::ggTrainingVector";                                                     break; }
            case 301: { hhTrainingVector(val,(int) xa); desc = "ML_Base::hhTrainingVector";                                                     break; }
            case 302: { varTrainingVector(val,dummy,(int) xa); desc = "ML_Base::varTrainingVector";                                             break; }
            case 303: { Vector<double> res; dgTrainingVectorX(res,(int) xa); val = res;  desc = "ML_Base::dgTrainingVectorX";                   break; }
            case 304: { Vector<gentype> res; gentype resn; dgTrainingVector(res,resn,(int) xa); val = res; desc = "ML_Base::dgTrainingVector";  break; }
            case 305: { Vector<gentype> res; gentype resn; dgTrainingVector(res,resn,(int) xa); val = resn; desc = "ML_Base::dgTrainingVector"; break; }
            case 306: { double res; int z = 0; stabProbTrainingVector(res, (int) xa, (int) xb(z), (double) xb(1), (int) xb(2), (double) xb(3), (double) xb(4)); val = res;  desc = "ML_Base::stabProfTrainingVector"; break; }

            case 400: { covTrainingVector(val,dummy,(int) xa, (int) xb); desc = "ML_Base::covTrainingVector"; break; }
            case 401: { K2(val,(int) xa, (int) xb);                      desc = "ML_Base::K2";                break; }
            case 402: { val = K2ip((int) xa, (int) xb,0.0);              desc = "ML_Base::K2ip";              break; }
            case 403: { val = distK((int) xa, (int) xb);                 desc = "ML_Base::distK";             break; }

            case 499: { Keqn(val); desc = "ML_Base::Keqn"; break; }

            default:
            {
                val.force_null();
                break;
            }
        }
    }

    else if ( ind <= 499 )
    {
        val.force_null();
    }

    else if ( ( ind <= 599 ) && !ib )
    {
        SparseVector<gentype> xx;

        if ( convertSetToSparse(xx,xa,ia) )
        {
            res = 1;
        }

        else
        {
            gentype dummy;

            switch ( ind )
            {
                case 500: { gg(val,xx); desc = "ML_Base::gg"; break; }
                case 501: { hh(val,xx); desc = "ML_Base::hh"; break; }
                case 502: { var(val,dummy,xx); desc = "ML_Base::var"; break; }
                case 503: { Vector<double> res; dgX(res,xx); val = res;  desc = "ML_Base::dgX"; break; }
                case 504: { Vector<gentype> res; gentype resn; dg(res,resn,xx); val = res;  desc = "ML_Base::dg"; break; }
                case 505: { Vector<gentype> res; gentype resn; dg(res,resn,xx); val = resn; desc = "ML_Base::dg"; break; }
                case 506: { double res; int z = 0; stabProb(res,xx,(int) xb(z),(double) xb(1),(int) xb(2),(double) xb(3),(double) xb(4)); val = res;  desc = "ML_Base::stabProb"; break; }

                case 9033: { val = isKreal();                     desc = "SVM_Scalar::isKreal"; break; }
                case 9034: { val = isKunreal();                   desc = "SVM_Scalar::isKunreal"; break; }

                default:
                {
                    val.force_null();
                    break;
                }
            }
        }
    }

    else if ( ind <= 599 )
    {
        val.force_null();
    }

    else if ( ind <= 699 )
    {
        SparseVector<gentype> xx;
        SparseVector<gentype> yy;

        convertSetToSparse(xx,xa,ia);
        convertSetToSparse(yy,xb,ib);

        gentype dummy;

        switch ( ind )
        {
            case 600: { cov(val,dummy,xx,yy);               desc = "ML_Base::cov"; break; }
            case 601: { K2(val,xx,yy);                      desc = "ML_Base::K2"; break; }
            case 602: { val = K2ip(xx,yy,0,0);              desc = "ML_Base::K2ip"; break; }
            case 603: { val = distK(xx,yy);                 desc = "ML_Base::val"; break; }
            case 604: { noisevar(val,dummy,xx,yy);          desc = "ML_Base::noisevar"; break; }

            default:
            {
                val.force_null();
                break;
            }
        }
    }

    else if ( ind <= 799 )
    {
        NiceAssert( ia = 0 );
        NiceAssert( ib = 0 );

        int i,m = (int) xa;

        Vector<SparseVector<gentype> > xx(m);
        const Vector<gentype> &xxb = (const Vector<gentype> &) xb;

        NiceAssert( xxb.size() == m );

        for ( i = 0 ; i < m ; ++i )
        {
            convertSetToSparse(xx("&",i),xxb(i),0);
        }

        switch ( ind )
        {
            case 701: { Km(val,xx); desc = "ML_Base::Km"; break; }

            default:
            {
                val.force_null();
                break;
            }
        }
    }

    else if ( ind <= 899 )
    {
        res = getKernel().getparam(ind-800,val,xa,ia,xb,ib,desc);
    }

    else if ( ind <= 999 )
    {
        res = getUUOutputKernel().getparam(ind-900,val,xa,ia,xb,ib,desc);
    }

    else
    {
        val.force_null();
    }

    return res;
}

