
//
// k-nearest-neighbour anionic regressor
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
#include "knn_anions.hpp"


std::ostream &KNN_Anions::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Anionic KNN\n\n";

    repPrint(output,'>',dep) << "Class labels:  " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts:  " << classcnt    << "\n";
    repPrint(output,'>',dep) << "Order:         " << dorder      << "\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Anions::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;
    input >> dummy; input >> dorder;

    KNN_Generic::inputstream(input);

    return input;
}

KNN_Anions::KNN_Anions() : KNN_Generic()
{
    setaltx(nullptr);

    classlabels.resize(1);
    classcnt.resize(2); // includes class 0 (other two don't)

    classlabels("&",0) = +2;
    classcnt = 0;

    dorder = 0;

    return;
}

KNN_Anions::KNN_Anions(const KNN_Anions &src) : KNN_Generic()
{
    setaltx(nullptr);

    assign(src,0);

    return;
}

KNN_Anions::KNN_Anions(const KNN_Anions &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,-1);

    return;
}

KNN_Anions::~KNN_Anions()
{
    return;
}

double KNN_Anions::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int KNN_Anions::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    (void) dval;

    NiceAssert( y.isCastableToAnionWithoutLoss() );

    ++(classcnt("&",1));

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh,dval);
    KNN_Generic::dd("&",i) = 2;

    if ( ML_Base::y(i).order() > dorder )
    {
        dorder = (ML_Base::y(i)).order();
    }

    return 1;
}

int KNN_Anions::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    (void) dval;

    NiceAssert( y.isCastableToAnionWithoutLoss() );

    ++(classcnt("&",1));

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh,dval);
    KNN_Generic::dd("&",i) = 2;

    if ( (ML_Base::y(i)).order() > dorder )
    {
        dorder = (ML_Base::y(i)).order();
    }

    return 1;
}

int KNN_Anions::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ++ii )
        {
            addTrainingVector(i+ii,y(ii),x(ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return 1;
}

int KNN_Anions::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ++ii )
        {
            qaddTrainingVector(i+ii,y(ii),x("&",ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return 1;
}

int KNN_Anions::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        --(classcnt("&",dd(i)/2));
    }

    KNN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int KNN_Anions::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToAnionWithoutLoss() );

    KNN_Generic::sety(i,yy);

    if ( (ML_Base::y(i)).order() > dorder )
    {
        dorder = (ML_Base::y(i)).order();
    }

    return 1;
}

int KNN_Anions::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    NiceAssert( i.size() == y.size() );

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ++ii )
        {
            sety(i(ii),y(ii));
        }
    }

    return 1;
}

int KNN_Anions::sety(const Vector<gentype> &y)
{
    NiceAssert( N() == y.size() );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ++ii )
        {
            sety(ii,y(ii));
        }
    }

    return 1;
}

int KNN_Anions::setd(int i, int xdd)
{
    NiceAssert( ( xdd == +2 ) || ( xdd == 0 ) );

    KNN_Generic::setd(i,xdd);

    return 1;
}

int KNN_Anions::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ++ii )
        {
            setd(i(ii),d(ii));
        }
    }

    return 1;
}

int KNN_Anions::setd(const Vector<int> &d)
{
    NiceAssert( N() == d.size() );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ++ii )
        {
            setd(ii,d(ii));
        }
    }

    return 1;
}

void KNN_Anions::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    if ( !res.isValAnion() ) { res.force_anion(); }
    setzero(res.dir_anion());

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

void KNN_Anions::hfn(d_anion &res, const Vector<d_anion> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    setzero(res);

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

int KNN_Anions::randomise(double sparsity)
{
    (void) sparsity;


    int prefpos = ( sparsity < 0 ) ? 1 : 0;
    sparsity = ( sparsity < 0 ) ? -sparsity : sparsity;



    int res = 0;
    int Nnotz = N()-NNC(0);

    if ( Nnotz )
    {
        res = 1;

        retVector<int> tmpva;

        Vector<int> canmod(cntintvec(N(),tmpva));

        int i,j,k;

        for ( i = N()-1 ; i >= 0 ; --i )
        {
            if ( !d()(i) )
            {
                canmod.remove(i);
            }
        }

        // Randomise

        double lbloc = prefpos ? 0.0 : -1.0;
        double ubloc = +1.0;

        Vector<gentype> yloc(y());

        for ( i = 0 ; i < canmod.size() ; ++i )
        {
            j = canmod(i);

            NiceAssert( d()(j) );

            d_anion &bmod = yloc("&",j).dir_anion();

            if ( bmod.size() )
            {
                for ( k = 0 ; k < bmod.size() ; ++k )
                {
                    double &amod = bmod("&",k);

                    setrand(amod);
                    amod = lbloc+((ubloc-lbloc)*amod);
                }
            }
        }

        ML_Base::sety(yloc);
    }

    return res;
}

int KNN_Anions::settspaceDim(int newdim)
{
    NiceAssert( ( ( N() == 0 ) && ( newdim >= -1 ) ) || ( newdim >= 0 ) );

    ML_Base::settspaceDim(newdim);

    dorder = 1<<newdim;

    return 1;
}

int KNN_Anions::addtspaceFeat(int i)
{
    NiceAssert( ( ( i >= 0 ) && ( i <= tspaceDim() ) ) || ( tspaceDim() == -1 ) );

    ML_Base::addtspaceFeat(i);

    dorder = 1<<tspaceDim();

    return 1;
}

int KNN_Anions::removetspaceFeat(int i)
{
    NiceAssert( ( i >= 0 ) && ( i < tspaceDim() ) );

    ML_Base::removetspaceFeat(i);

    dorder = 1<<tspaceDim();

    return 1;
}

int KNN_Anions::setorder(int neword)
{
    NiceAssert( neword >= 0 );

    ML_Base::setorder(neword);

    dorder = neword;

    return 1;
}

