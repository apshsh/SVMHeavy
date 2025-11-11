
//
// LS-SVM base class
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
#include "lsv_generic.hpp"

std::ostream &LSV_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training alpha:            " << dalpha       << "\n";
    repPrint(output,'>',dep) << "Base training bias:             " << dbias        << "\n";
    repPrint(output,'>',dep) << "Base training training target:  " << alltraintarg << "\n";

    SVM_Scalar::printstream(output,dep+1);

    return output;
}

std::istream &LSV_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    killfasts();

    input >> dummy; input >> dalpha;
    input >> dummy; input >> dbias;
    input >> dummy; input >> alltraintarg;

    SVM_Scalar::inputstream(input);

    return input;
}

LSV_Generic::LSV_Generic() : SVM_Scalar()
{
    fastweights = nullptr;
    fastxsums   = nullptr;
    fastdim = 0;

    fastweights_base = nullptr;
    fastxsums_base   = nullptr;
    fastdim_base = 0;

    SVM_Scalar::setQuadraticCost();
    SVM_Scalar::seteps(0.0);
    //SVM_Scalar::fudgeOn(); - this causes numerical instability if enabled!

    return;
}

LSV_Generic::LSV_Generic(const LSV_Generic &src) : SVM_Scalar()
{
    fastweights = nullptr;
    fastxsums   = nullptr;
    fastdim = 0;

    fastweights_base = nullptr;
    fastxsums_base   = nullptr;
    fastdim_base = 0;

    SVM_Scalar::setQuadraticCost();
    SVM_Scalar::seteps(0.0);
    //SVM_Scalar::fudgeOn(); - this causes numerical instability if enabled!

    assign(src,0);

    return;
}

LSV_Generic::LSV_Generic(const LSV_Generic &src, const ML_Base *srcx) : SVM_Scalar()
{
    setaltx(srcx);

    fastweights = nullptr;
    fastxsums   = nullptr;
    fastdim = 0;

    fastweights_base = nullptr;
    fastxsums_base   = nullptr;
    fastdim_base = 0;

    SVM_Scalar::setQuadraticCost();
    SVM_Scalar::seteps(0.0);
    //SVM_Scalar::fudgeOn(); - this causes numerical instability if enabled!

    assign(src,-1);

    return;
}

int LSV_Generic::prealloc(int expectedN)
{
    SVM_Scalar::prealloc(expectedN);
    dalpha.prealloc(expectedN);
    alltraintarg.prealloc(expectedN);

    return 0;
}

int LSV_Generic::preallocsize(void) const
{
    return SVM_Scalar::preallocsize();
}

double LSV_Generic::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0.0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int LSV_Generic::sety(int i, const gentype &y)
{
    int res = 0;

    alltraintarg("&",i) = y;

    if ( y.isCastableToRealWithoutLoss() )
    {
        res = SVM_Scalar::sety(i,y);
    }

    else
    {
        res = SVM_Scalar::sety(i,0.0_gent);
    }

    return res;
}

int LSV_Generic::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    retVector<gentype> tmpva;
    alltraintarg("&",i,tmpva) = y;

    Vector<gentype> yy(y);

    for ( int ii = 0 ; ii < i.size() ; ii++ )
    {
        if ( !yy(i(ii)).isCastableToRealWithoutLoss() )
        {
            yy("&",i(ii)).force_double() = 0.0;
        }
    }

    return SVM_Scalar::sety(i,yy);
}

int LSV_Generic::sety(const Vector<gentype> &y)
{
    alltraintarg = y;

    Vector<gentype> yy(y);

    for ( int i = 0 ; i < yy.size() ; i++ )
    {
        if ( !yy(i).isCastableToRealWithoutLoss() )
        {
            yy("&",i).force_double() = 0.0;
        }
    }

    return SVM_Scalar::sety(yy);
}

int LSV_Generic::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    killfasts();

    dalpha.add(i);
    makezero(dalpha("&",i));

    alltraintarg.add(i);
    alltraintarg("&",i) = y;

    if ( y.isCastableToRealWithoutLoss() )
    {
        res = SVM_Scalar::addTrainingVector(i,y,x,Cweigh,epsweigh,dval);
    }

    else
    {
        gentype zeroy = 0.0_gent;
        zeroy.isNomConst = y.isNomConst;

        res = SVM_Scalar::addTrainingVector(i,zeroy,x,Cweigh,epsweigh,dval);
    }

    return res;
}

int LSV_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    killfasts();

    dalpha.add(i);
    makezero(dalpha("&",i));

    alltraintarg.add(i);
    alltraintarg("&",i) = y;

    if ( y.isCastableToRealWithoutLoss() )
    {
        res = SVM_Scalar::qaddTrainingVector(i,y,x,Cweigh,epsweigh,dval);
    }

    else
    {
        gentype zeroy = 0.0_gent;
        zeroy.isNomConst = y.isNomConst;

        res = SVM_Scalar::qaddTrainingVector(i,zeroy,x,Cweigh,epsweigh,dval);
    }

    return res;
}

int LSV_Generic::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
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

int LSV_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
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

int LSV_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    gentype ydummy;

    qswap(y,alltraintarg("&",i));
    int res = SVM_Scalar::removeTrainingVector(i,ydummy,x);

    killfasts();

    dalpha.remove(i);
    alltraintarg.remove(i);

    return res;
}

int LSV_Generic::removeTrainingVector(int i, int num)
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


// NB: we allow any value of d, including inequalities

int LSV_Generic::setd(int i, int nd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = SVM_Scalar::setd(i,nd);

    if ( !nd )
    {
        killfasts();

        setzeropassive(dalpha("&",i));
    }

    return res;
}

int LSV_Generic::setd(const Vector<int> &i, const Vector<int> &nd)
{
    NiceAssert( i.size() == nd.size() );

    int res = SVM_Scalar::setd(i,nd);

    for ( int j = 0 ; j < i.size() ; ++j )
    {
        if ( !nd(j) )
        {
            killfasts();

            setzeropassive(dalpha("&",i(j)));
        }
    }

    return res;
}

int LSV_Generic::setd(const Vector<int> &nd)
{
    NiceAssert( N() == nd.size() );

    int res = SVM_Scalar::setd(nd);

    for ( int j = 0 ; j < N() ; ++j )
    {
        if ( !nd(j) )
        {
            killfasts();

            setzeropassive(dalpha("&",j));
        }
    }

    return res;
}

int LSV_Generic::scale(double a)
{
    SVM_Scalar::scale(a);
    dalpha.scale(a);
    dbias *= a;

    killfasts();

    return 1;
}

int LSV_Generic::reset(void)
{
    SVM_Scalar::reset();
    setzeropassive(dalpha);
    setzeropassive(dbias);

    killfasts();

    return 1;
}

int LSV_Generic::train(int &res, svmvolatile int &killSwitch)
{
    incgvernum();

    res = 0;

    int blah = killSwitch;
    (void) blah;

    SVM_Scalar::maxFreeAlphaBias();
    SVM_Scalar::isStateOpt = 1;

    return 0;
}

int LSV_Generic::setgamma(const Vector<gentype> &newW)
{
    NiceAssert( dalpha.size() == newW.size() );

    killfasts();

    dalpha = newW;
    SVM_Scalar::setAlpha(newW);

    return 1;
}

int LSV_Generic::setdelta(const gentype &newB)
{
    killfasts();

    dbias = newB;
    SVM_Scalar::setBias(newB);

    return 1;
}

int LSV_Generic::setVardelta(void)
{
    SVM_Scalar::isStateOpt = 0;
    return SVM_Scalar::setVarBias();
}

int LSV_Generic::setZerodelta(void)
{
    SVM_Scalar::isStateOpt = 0;

    killfasts();

    setzeropassive(dbias);

    return SVM_Scalar::setFixedBias(0.0);
}

void LSV_Generic::dgTrainingVectorX(Vector<double> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = 0.0;

    for ( int j = 0 ; j < ML_Base::N() ; ++j )
    {
        for ( int k = 0 ; k < x(j).nindsize() ; ++k )
        {
            resx("&",x(j).ind(k)) += (double) ((res(j)*(x(j)).direcref(k)));
        }
    }

    for ( int k = 0 ; k < x(i).nindsize() ; ++k )
    {
        resx("&",x(i).ind(k)) += (double) ((resn*(x(i)).direcref(k)));
    }

    return;
}

void LSV_Generic::dgTrainingVectorX(Vector<gentype> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = 0.0_gent;

    for ( int j = 0 ; j < ML_Base::N() ; ++j )
    {
        for ( int k = 0 ; k < x(j).nindsize() ; ++k )
        {
            resx("&",x(j).ind(k)) += ((res(j)*(x(j)).direcref(k)));
        }
    }

    for ( int k = 0 ; k < x(i).nindsize() ; ++k )
    {
        resx("&",x(i).ind(k)) += ((resn*(x(i)).direcref(k)));
    }

    return;
}

void LSV_Generic::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    res.resize(N());

    if ( gOutType() == 'V' )
    {
        gentype zerotemplate('V');

        zerotemplate.dir_vector().resize(tspaceDim());

        setzero(zerotemplate);

        res  = zerotemplate;
        resn = zerotemplate;

        if ( N() )
        {
            int j;
            int dummyind = -1;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < N() ; ++j )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);

                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_vector().scaleAdd((double) xscale,(const Vector<gentype> &) dalpha(j));
                        res("&",j).dir_vector().scaleAdd((double) yscale,(const Vector<gentype> &) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_vector().scaleAdd((double) xscale,(const Vector<gentype> &) dalpha(j));
                        res("&",j).dir_vector().scaleAdd((double) yscale,(const Vector<gentype> &) dalpha(j));
                    }
                }
            }
        }
    }

    else if ( gOutType() == 'A' )
    {
        gentype zerotemplate('A');

        zerotemplate.dir_anion().setorder(order());

        setzero(zerotemplate);

        res  = zerotemplate;
        resn = zerotemplate;

        if ( N() )
        {
            int j;
            int dummyind = -1;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < N() ; ++j )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);

                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_anion() += ((double) xscale)*((const d_anion &) dalpha(j));
                        res("&",j).dir_anion() += ((double) yscale)*((const d_anion &) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_anion() += ((double) xscale)*((const d_anion &) dalpha(j));
                        res("&",j).dir_anion() += ((double) yscale)*((const d_anion &) dalpha(j));
                    }
                }
            }
        }
    }

    else
    {
        //gentype zerotemplate(0.0);

        res  = 0.0_gent;
        resn = 0.0_gent;

        if ( N() )
        {
            int j;
            int dummyind = -1;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < N() ; ++j )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);

                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_double() += ((double) xscale)*((double) dalpha(j));
                        res("&",j).dir_double() += ((double) yscale)*((double) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_double() += ((double) xscale)*((double) dalpha(j));
                        res("&",j).dir_double() += ((double) yscale)*((double) dalpha(j));
                    }
                }
            }
        }
    }

    return;
}

int LSV_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int LSV_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    int res = 0;

    desc = "";

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 5000: { val = isVardelta();       desc = "LSV_Generic::isVardelta"; break; }
        case 5001: { val = isZerodelta();      desc = "LSV_Generic::isZerodelta"; break; }
        case 5002: { val = gamma();            desc = "LSV_Generic::gamma"; break; }
        case 5003: { val = delta();            desc = "LSV_Generic::delta"; break; }
        case 5005: { val = loglikelihood();    desc = "LSV_Generic::loglikelihood"; break; }
        case 9070: { val = lsvGp();            desc = "LSV_Generic::lsvGp"; break; }

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

