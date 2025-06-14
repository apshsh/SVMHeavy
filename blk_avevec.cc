
//
// Average result block
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
#include "blk_avevec.hpp"


std::ostream &BLK_AveVec::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector Average BLK\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";
    repPrint(output,'>',dep) << "Dimension:    " << dim         << "\n\n";

    BLK_Generic::printstream(output,dep+1);

    return output;
}

std::istream &BLK_AveVec::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;
    input >> dummy; input >> dim;

    BLK_Generic::inputstream(input);

    return input;
}

BLK_AveVec::BLK_AveVec(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(nullptr);

    classlabels.resize(1);
    classcnt.resize(2); // includes class 0 (other two don't)

    classlabels("&",0) = +2;
    classcnt = 0;

    dim = -1;

    return;
}

double BLK_AveVec::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int BLK_AveVec::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,y,xx,Cweigh,epsweigh,dval);
}

int BLK_AveVec::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    NiceAssert( y.isCastableToVectorWithoutLoss() );
    NiceAssert( ( dim == -1 ) || ( y.size() == dim ) );

    if ( dim == -1 )
    {
        dim = y.size();
    }

    ++(classcnt("&",1));

    gentype yy(y);
    yy.morph_vector();

    BLK_Generic::qaddTrainingVector(i,yy,x,Cweigh,epsweigh,dval);

    return 1;
}

int BLK_AveVec::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size()        );
    NiceAssert( y.size() == Cweigh.size()   );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;
    int j;

    if ( y.size() )
    {
        for ( j = 0 ; j < y.size() ; ++j )
        {
            res |= addTrainingVector(i+j,y(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int BLK_AveVec::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size()        );
    NiceAssert( y.size() == Cweigh.size()   );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;
    int j;

    if ( y.size() )
    {
        for ( j = 0 ; j < y.size() ; ++j )
        {
            res |= qaddTrainingVector(i+j,y(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int BLK_AveVec::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        --(classcnt("&",d()(i)/2));
    }

    BLK_Generic::removeTrainingVector(i,y,x);

    return 1;
}

int BLK_AveVec::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToVectorWithoutLoss() );
    NiceAssert( ( dim == -1 ) || ( yy.size() == dim ) );

    gentype yyy(yy);

    yyy.morph_vector();

    if ( dim == -1 )
    {
        dim = yyy.size();
    }

    BLK_Generic::sety(i,yyy);

    return 1;
}

int BLK_AveVec::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int BLK_AveVec::sety(const Vector<gentype> &y)
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

int BLK_AveVec::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    BLK_Generic::setd(i,dd);

    return 1;
}

int BLK_AveVec::setd(const Vector<int> &i, const Vector<int> &d)
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

int BLK_AveVec::setd(const Vector<int> &d)
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






































int BLK_AveVec::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***) const
{
    NiceAssert( !retaltg );

    (void) retaltg;

    resg = 0.0;

    if ( xindsize(i) )
    {
        int j;
        gentype dummy;

        for ( j = 0 ; j < xindsize(i) ; ++j )
        {
            resg += xelm(dummy,i,j);
        }

        resg *= (1.0/xindsize(i));
    }

    if ( outfn().isValNull() )
    {
	resh = resg;
    }

    else
    {
        resh = outfn()(resg);
    }

    return 0;
}

void BLK_AveVec::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    res.resize(N());

    gentype zerotemplate(0.0);

    res  = zerotemplate;
    resn = zerotemplate;

    if ( i >= 0 )
    {
        res("&",i) = ( xindsize(i) ? 1.0/xindsize(i) : 1.0 );
    }

    else
    {
        resn = ( xindsize(i) ? 1.0/xindsize(i) : 1.0 );
    }

    return;
}

void BLK_AveVec::dgTrainingVector(Vector<gentype> &res, const Vector<int> &i) const
{
    res.resize(N());

    gentype zerotemplate(0.0);

    res = zerotemplate;

    int q;

    for ( q = 0 ; q < i.size() ; ++q )
    {
        res("&",i(q)) = ( xindsize(i(q)) ? 1.0/xindsize(i(q)) : 1.0 );
    }

    return;
}

