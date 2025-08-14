
//
// Scalar regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_kconst.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

int SVM_KConst::setAlphaR(const Vector<double> &newAlpha)
{
    ddalphaR = newAlpha;

    int i;

    for ( i = 0 ; i < N() ; ++i )
    {
        basesetalpha(i,newAlpha(i));

        if ( useKwe )
        {
            gentype temp(newAlpha(i)*newAlpha(i));

            getKernel_unsafe().setWeight(temp,i);
        }
    }

    if ( useKwe )
    {
        Vector<gentype> kernweight(newAlpha.size());
    }

    return 1;
}

int SVM_KConst::setBiasR(double newBias)
{
    ddbiasR = newBias;

    basesetbias(newBias);

    return 1;
}

int SVM_KConst::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int unusedvar = 0;
    int tempresh = 0;

    tempresh = gTrainingVector(resg.force_double(),unusedvar,i,retaltg,pxyprodi);
    resh.force_double() = (double) resg;

    return tempresh;
}

double SVM_KConst::eTrainingVector(int i) const
{
    int unusedvar = 0;
    double res = 0;

    gTrainingVector(res,unusedvar,i);

    return res;
}

int SVM_KConst::gTrainingVector(double &res, int &unusedvar, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !( retaltg & 2 ) );

    (void) retaltg;

    res = biasR();

    int j;
    double Kxj;

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            Kxj = K2(i,j,pxyprodi ? (const gentype **) pxyprodi[j] : nullptr);
            res += (alphaR()(j))*Kxj;
        }
    }

    return ( unusedvar = ( res > 0 ) ? +1 : -1 );
}

void SVM_KConst::fastg(double &res) const
{
    int j;
    double Kxj;

    res = biasR();

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            Kxj = K1(j,nullptr,&(x(j)),&(xinfo(j)));

            res += (alphaR()(j))*Kxj;
        }
    }

    return;
}


void SVM_KConst::fastg(double &res,
                       int ia,
                       const SparseVector<gentype> &xa,
                       const vecInfo &xainfo) const
{
    if ( ia < 0 ) { ia = setInnerWildpa(&xa,&xainfo); }

    int j;
    double Kxj;

    res = biasR();

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            Kxj = K2(ia,j,nullptr,&xa,&(x(j)),&xainfo,&(xinfo(j)));

            res += (alphaR()(j))*Kxj;
        }
    }

    if ( ia < 0 ) { resetInnerWildp(); }

    return;
}

void SVM_KConst::fastg(double &res, 
                       int ia, int ib,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                       const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    if ( ia < 0 ) { ia = setInnerWildpa(&xa,&xainfo); }
    if ( ib < 0 ) { ib = setInnerWildpb(&xb,&xbinfo); }

    int j;
    double Kxj;

    res = biasR();

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            Kxj = K3(ia,ib,j,nullptr,&xa,&xb,&(x(j)),&xainfo,&xbinfo,&(xinfo(j)));

            res += (alphaR()(j))*Kxj;
        }
    }

    if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

    return;
}

void SVM_KConst::fastg(gentype &res) const
{
    int j;
    double Kxj;

    res = biasR();

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            Kxj = K1(j,nullptr,&(x(j)),&(xinfo(j)));

            res += (alphaR()(j))*Kxj;
        }
    }

    return;
}

void SVM_KConst::fastg(gentype &res,
                       int ia,
                       const SparseVector<gentype> &xa,
                       const vecInfo &xainfo) const
{
    if ( ia < 0 ) { ia = setInnerWildpa(&xa,&xainfo); }

    int j;
    double Kxj;

    res = biasR();

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            Kxj = K2(ia,j,nullptr,&xa,&(x(j)),&xainfo,&(xinfo(j)));

            res += (alphaR()(j))*Kxj;
        }
    }

    if ( ia < 0 ) { resetInnerWildp(); }

    return;
}

void SVM_KConst::fastg(gentype &res, 
                       int ia, int ib,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                       const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    if ( ia < 0 ) { ia = setInnerWildpa(&xa,&xainfo); }
    if ( ib < 0 ) { ib = setInnerWildpb(&xb,&xbinfo); }

    int j;
    double Kxj;

    res = biasR();

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            Kxj = K3(ia,ib,j,nullptr,&xa,&xb,&(x(j)),&xainfo,&xbinfo,&(xinfo(j)));

            res += (alphaR()(j))*Kxj;
        }
    }

    if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

    return;
}

int SVM_KConst::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    int res = SVM_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh,dval);

    ddalphaR.add(i);
    ddalphaR("&",i) = 0;

    return res;
}

int SVM_KConst::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    int res = SVM_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh,dval);

    ddalphaR.add(i);
    ddalphaR("&",i) = 0;

    return res;
}

int SVM_KConst::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int res = SVM_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; ++j )
        {
            ddalphaR.add(i+j);
            ddalphaR("&",i+j) = 0;
        }
    }

    return res;
}

int SVM_KConst::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int res = SVM_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; ++j )
        {
            ddalphaR.add(i+j);
            ddalphaR("&",i+j) = 0;
        }
    }

    return res;
}

int SVM_KConst::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int res = SVM_Generic::removeTrainingVector(i,y,x);

    ddalphaR.remove(i);

    return res;
}




// Stream operators

std::ostream &SVM_KConst::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Kconst SVM\n\n";

    repPrint(output,'>',dep) << "Base training ddalphaR:         " << ddalphaR    << "\n";
    repPrint(output,'>',dep) << "Base training ddbiasR:          " << ddbiasR     << "\n";
    repPrint(output,'>',dep) << "Write alphaR to kernel weights: " << useKwe      << "\n";

    SVM_Generic::printstream(output,dep+1);

    return output;
}

std::istream &SVM_KConst::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> ddalphaR;
    input >> dummy; input >> ddbiasR;
    input >> dummy; input >> useKwe;

    SVM_Generic::inputstream(input);

    return input;
}


int SVM_KConst::randomise(double sparsity)
{
    NiceAssert( sparsity >= -1 );
    NiceAssert( sparsity <= 1 );

    int prefpos = ( sparsity < 0 ) ? 1 : 0;
    sparsity = ( sparsity < 0 ) ? -sparsity : sparsity;

    int res = 0;
    int Nnotz = (int) (N()*sparsity);

    if ( Nnotz )
    {
        res = 1;

        retVector<int> tmpva;
        Vector<int> canmod(cntintvec(N(),tmpva));

        int i,j;

        // Observe sparsity

        while ( canmod.size() > Nnotz )
        {
            //canmod.remove(svm_rand()%(canmod.size()));
            canmod.remove(rand()%(canmod.size()));
        }

        // Need to randomise canmod alphas, set rest to zero
        // (need to take care as meaning of zero differs depending on goutType)

        Vector<double> newalpha(N());

        // Set zero

        newalpha = 0.0;

        // Next randomise

        double lbloc = -1.0;
        double ubloc = +1.0;

        lbloc = ( prefpos && ( ubloc > 0 ) ) ? 0 : lbloc;

        for ( i = 0 ; i < canmod.size() ; ++i )
        {
            j = canmod(i);

            double &amod = newalpha("&",j);

            setrand(amod);

            amod = lbloc+((ubloc-lbloc)*amod);
        }

        // Lastly set alpha

        setAlphaR(newalpha);
        setBiasR(0);
    }

    return res;
}
