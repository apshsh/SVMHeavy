
//
// Binary Classification GPR by EP
//
// Version: 7
// Date: 18/12/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_binary_rff.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


GPR_Binary_rff::GPR_Binary_rff() : GPR_Scalar_rff()
{
    setaltx(nullptr);

    return;
}

GPR_Binary_rff::GPR_Binary_rff(const GPR_Binary_rff &src) : GPR_Scalar_rff()
{
    setaltx(nullptr);

    assign(src,0);

    return;
}

GPR_Binary_rff::GPR_Binary_rff(const GPR_Binary_rff &src, const ML_Base *xsrc) : GPR_Scalar_rff()
{
    setaltx(xsrc);

    assign(src,-1);

    return;
}

GPR_Binary_rff::~GPR_Binary_rff()
{
    return;
}

double GPR_Binary_rff::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int GPR_Binary_rff::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; ++i )
        {
            res |= sety(j(i),yn(i));
        }
    }

    return res;
}

int GPR_Binary_rff::sety(const Vector<gentype> &yn)
{
    int res = 0;

    if ( yn.size() )
    {
        int i;

        for ( i = 0 ; i < yn.size() ; ++i )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int GPR_Binary_rff::sety(const Vector<int> &j, const Vector<double> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; ++i )
        {
            res |= sety(j(i),yn(i));
        }
    }

    return res;
}

int GPR_Binary_rff::sety(const Vector<double> &yn)
{
    int res = 0;

    if ( yn.size() )
    {
        int i;

        for ( i = 0 ; i < yn.size() ; ++i )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int GPR_Binary_rff::setd(const Vector<int> &j, const Vector<int> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; ++i )
        {
            res |= setd(j(i),yn(i));
        }
    }

    return res;
}

int GPR_Binary_rff::setd(const Vector<int> &yn)
{
    int res = 0;

    if ( yn.size() )
    {
        int i;

        for ( i = 0 ; i < yn.size() ; ++i )
        {
            res |= setd(i,yn(i));
        }
    }

    return res;
}

int GPR_Binary_rff::setd(int i, int xd)
{
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != d()(i) )
    {
        res = 1;

        bintraintarg("&",i) = xd;

        GPR_Scalar_rff::setd(i,xd);
    }

    return res;
}

int GPR_Binary_rff::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToIntegerWithoutLoss() && ( ( (int) y == -1 ) || ( (int) y == 0 ) || ( (int) y == +1 ) ) );

    bintraintarg.add(i);
    bintraintarg("&",i) = y;

    int res = 0;

    res |= GPR_Scalar_rff::addTrainingVector(i,0.0_gent,x,Cweigh,epsweigh);
    res |= GPR_Scalar_rff::setd(i,(int) y);

    return res;
}

int GPR_Binary_rff::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToIntegerWithoutLoss() && ( ( (int) y == -1 ) || ( (int) y == 0 ) || ( (int) y == +1 ) ) );

    bintraintarg.add(i);
    bintraintarg("&",i) = (int) y;

    int res = 0;

    res |= GPR_Scalar_rff::qaddTrainingVector(i,0.0_gent,x,Cweigh,epsweigh);
    res |= GPR_Scalar_rff::setd(i,(int) y);

    return res;
}

int GPR_Binary_rff::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    y = bintraintarg(i);
    bintraintarg.remove(i);

    gentype dummy;

    return GPR_Scalar_rff::removeTrainingVector(i,dummy,x);
}

int GPR_Binary_rff::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int tempresh = GPR_Scalar_rff::ghTrainingVector(resh,resg,i,retaltg,pxyprodi);
    double resgd = (double) resg;

    resh = 0;

    if ( resgd > 0 ) { resh = 1; }
    if ( resgd < 0 ) { resh = -1; }

    return tempresh;
}

int GPR_Binary_rff::gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg, const vecInfo *xinf, gentype ***pxyprodx) const
{
    int tempresh = GPR_Scalar_rff::gh(resh,resg,x,retaltg,xinf,pxyprodx);
    double resgd = (double) resg;

    resh = 0;

    if ( resgd > 0 ) { resh = 1; }
    if ( resgd < 0 ) { resh = -1; }

    return tempresh;
}

std::ostream &GPR_Binary_rff::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary GPR (RFF version)\n\n";

    repPrint(output,'>',dep) << "y: " << bintraintarg << "\n";
    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base GPR: ";
    GPR_Scalar_rff::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &GPR_Binary_rff::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> bintraintarg;
    input >> dummy;
    GPR_Scalar_rff::inputstream(input);

    return input;
}

int GPR_Binary_rff::prealloc(int expectedN)
{
    bintraintarg.prealloc(expectedN);
    GPR_Scalar_rff::prealloc(expectedN);

    return 0;
}

int GPR_Binary_rff::preallocsize(void) const
{
    return GPR_Scalar_rff::preallocsize();
}


