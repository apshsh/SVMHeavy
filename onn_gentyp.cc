
//
// 1 layer neural network gentyp regression
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
#include "onn_gentyp.hpp"


std::ostream &ONN_Gentyp::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Gentyp ONN\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";

    ONN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &ONN_Gentyp::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;

    ONN_Generic::inputstream(input);

    return input;
}

ONN_Gentyp::ONN_Gentyp() : ONN_Generic()
{
    setaltx(nullptr);

    classlabels.resize(3); // include -1 and +1, even though not allowed
    classcnt.resize(4); // includes class 0 (other two don't)

    classlabels("&",0) = -1;
    classlabels("&",1) = +1;
    classlabels("&",2) = +2;

    classcnt = 0;


    return;
}

ONN_Gentyp::ONN_Gentyp(const ONN_Gentyp &src) : ONN_Generic()
{
    setaltx(nullptr);

    assign(src,0);

    return;
}

ONN_Gentyp::ONN_Gentyp(const ONN_Gentyp &src, const ML_Base *xsrc) : ONN_Generic()
{
    setaltx(xsrc);

    assign(src,-1);

    return;
}

ONN_Gentyp::~ONN_Gentyp()
{
    return;
}

double ONN_Gentyp::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( ha.isValNull() || ha.isValInteger() || ha.isValReal() )
    {
        if ( db == +1 )
        {
            // treat as lower bound constraint ha >= hb

            if ( (double) ha < (double) hb )
            {
                res = ( (double) ha ) - ( (double) hb );
                res *= res;
            }
        }

        else if ( db == -1 )
        {
            // treat as upper bound constraint ha <= hb

            if ( (double) ha > (double) hb )
            {
                res = ( (double) ha ) - ( (double) hb );
                res *= res;
            }
        }

        else if ( db )
        {
            res = ( (double) ha ) - ( (double) hb );
            res *= res;
        }
    }

    else if ( ha.isValAnion() || ha.isValVector() || ha.isValMatrix() )
    {
        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        if ( db )
        {
            res = (double) norm2(ha-hb);
        }
    }

    else
    {
        // Sets, graphs and strings are comparable by binary multiplication

        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        if ( db )
        {
            res = ( ha == hb ) ? 0 : 1;
        }
    }

    return res;
}

int ONN_Gentyp::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> temp(x);

    return qaddTrainingVector(i,y,temp,Cweigh,epsweigh);
}

int ONN_Gentyp::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<SparseVector<gentype> > temp(x);

    return qaddTrainingVector(i,y,temp,Cweigh,epsweigh);
}

int ONN_Gentyp::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToRealWithoutLoss() );







    ++(classcnt("&",3));
    ONN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int ONN_Gentyp::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int res = 0;

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ++ii )
        {
            res |= qaddTrainingVector(i+ii,y(ii),x("&",ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return res;
}

int ONN_Gentyp::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        --(classcnt("&",d()(i)+1));
    }

    ONN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int ONN_Gentyp::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToRealWithoutLoss() );








    ONN_Generic::sety(i,yy);






    return 1;
}

int ONN_Gentyp::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int ONN_Gentyp::sety(const Vector<gentype> &y)
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

int ONN_Gentyp::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    ONN_Generic::setd(i,dd);








    return 1;
}

int ONN_Gentyp::setd(const Vector<int> &i, const Vector<int> &d)
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

int ONN_Gentyp::setd(const Vector<int> &d)
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
