
//
// ML summing block
//
// g(x) = weighted_sum(gi(x))
// gv(x) = weighted_sum(gv(x)) + var(weights \odot gi(x))
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
#include "blk_conect.hpp"
#include "gpr_generic.hpp"


BLK_Conect::BLK_Conect(int isIndPrune) : BLK_Generic(isIndPrune)
{
    localygood = 0;

    locsampleMode = 0;
    locxsampType  = 3;
    locNsamp      = -1;
    locsampSplit  = 1;
    locsampType   = 0;
    locsampScale  = 1.0;
    locsampSlack  = 0.0;

    setaltx(nullptr);

    return;
}

std::ostream &BLK_Conect::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "ML Averaging block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Conect::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}























void BLK_Conect::fillCache(int Ns, int Ne)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRep(ii).fillCache(Ns,Ne);
    }

    return;
}

double BLK_Conect::tuneKernel(int method, double xwidth, int, int, const tkBounds *, paraDef *probbnd)
{
    StrucAssert(!probbnd);

    int ii;
    double res = 0;

//    for ( ii = ( ( getmlqmode() == 2 ) ? (numReps()-1) : 0 )  ; ii < numReps() ; ++ii )
    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        int tuneK = 0;

        if ( getmlqmode() != 2 )
        {
            tuneK = 1;
        }

        else if ( ii == numReps()-1 )
        {
            tuneK = 1;
        }

        res += getRep(ii).tuneKernel(method,xwidth,tuneK,0,nullptr);
    }

    return res;
}

double BLK_Conect::loglikelihood(void) const
{
    int ii;
    double res = 0;

    for ( ii = ( getmlqmode() ? (numReps()-1) : 0 )  ; ii < numReps() ; ++ii )
    {
        res += ((getRepConst(ii).loglikelihood())/( ( getmlqmode() == 2 ) ? 1 : numReps() ));
    }

    return res;
}

double BLK_Conect::maxinfogain(void) const
{
    int ii;
    double res = 0;

    for ( ii = ( getmlqmode() ? (numReps()-1) : 0 )  ; ii < numReps() ; ++ii )
    {
        res += ((getRepConst(ii).maxinfogain())/( ( getmlqmode() == 2 ) ? 1 : numReps() ));
    }

    return res;
}

double BLK_Conect::RKHSnorm(void) const
{
    int ii;
    double res = 0;

    for ( ii = ( getmlqmode() ? (numReps()-1) : 0 )  ; ii < numReps() ; ++ii )
    {
        res += ((getRepConst(ii).RKHSnorm())/( ( getmlqmode() == 2 ) ? 1 : numReps() ));
    }

    return res;
}

double BLK_Conect::RKHSabs(void) const
{
    return sqrt(RKHSnorm());
}

int BLK_Conect::prealloc(int expectedN)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRep(ii).prealloc(expectedN);
    }

    return 0;
}

int BLK_Conect::preallocsize(void) const
{
    return numReps() ? getRepConst(0).preallocsize() : 0;
}

void BLK_Conect::setmemsize(int memsize)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRep(ii).setmemsize(memsize);
    }

    return;
}

void BLK_Conect::fudgeOn(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRep(ii).fudgeOn();
    }

    return;
}

void BLK_Conect::fudgeOff(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRep(ii).fudgeOff();
    }

    return;
}

void BLK_Conect::assumeConsistentX(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRep(ii).assumeConsistentX();
    }

    return;
}

void BLK_Conect::assumeInconsistentX(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRep(ii).assumeInconsistentX();
    }

    return;
}

int BLK_Conect::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    int ii;
    int res = 0;

    localygood = 0;

    gentype yrem(y);
    gentype ytmp(y);

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).addTrainingVector(i,yrem,x,Cweigh,epsweigh,dval);

        if ( getmlqmode() )
        {
            getRepConst(ii).ggTrainingVector(ytmp,i);

            if ( getRepConst(ii).d()(i) == 2 )
            {
                yrem -= ytmp;
            }

            else if ( getRepConst(ii).d()(i) == -1 )
            {
                if ( ytmp >= yrem )
                {
                    yrem -= ytmp;
                }

                else
                {
                    yrem *= 0;
                }
            }

            else if ( getRepConst(ii).d()(i) == +1 )
            {
                if ( ytmp <= yrem )
                {
                    yrem -= ytmp;
                }

                else
                {
                    yrem *= 0;
                }
            }
        }
    }

    return res;
}

int BLK_Conect::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    int ii;
    int res = 0;

    localygood = 0;

    gentype yrem(y);
    gentype ytmp(y);

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).addTrainingVector(i,yrem,x,Cweigh,epsweigh,dval);

        if ( getmlqmode() )
        {
            getRepConst(ii).ggTrainingVector(ytmp,i);

            if ( getRepConst(ii).d()(i) == 2 )
            {
                yrem -= ytmp;
            }

            else if ( getRepConst(ii).d()(i) == -1 )
            {
                if ( ytmp >= yrem )
                {
                    yrem -= ytmp;
                }

                else
                {
                    yrem *= 0;
                }
            }

            else if ( getRepConst(ii).d()(i) == +1 )
            {
                if ( ytmp <= yrem )
                {
                    yrem -= ytmp;
                }

                else
                {
                    yrem *= 0;
                }
            }
        }
    }

    return res;
}

int BLK_Conect::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int ii;
    int res = 0;

    localygood = 0;

    Vector<gentype> yrem(y);
    Vector<gentype> ytmp(y);

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).addTrainingVector(i,yrem,x,Cweigh,epsweigh);

        if ( getmlqmode() )
        {
            for ( int j = 0 ; j < y.size() ; j++ )
            {
                getRepConst(ii).ggTrainingVector(ytmp("&",j),i+j);

                if ( getRepConst(ii).d()(i+j) == 2 )
                {
                    yrem("&",j) -= ytmp(j);
                }

                else if ( getRepConst(ii).d()(i+j) == -1 )
                {
                    if ( ytmp(j) >= yrem(j) )
                    {
                        yrem("&",j) -= ytmp(j);
                    }

                    else
                    {
                        yrem("&",j) *= 0;
                    }
                }

                else if ( getRepConst(ii).d()(i+j) == +1 )
                {
                    if ( ytmp(j) <= yrem(j) )
                    {
                        yrem("&",j) -= ytmp(j);
                    }

                    else
                    {
                        yrem("&",j) *= 0;
                    }
                }
            }
        }
    }

    return res;
}

int BLK_Conect::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int ii;
    int res = 0;

    localygood = 0;

    Vector<gentype> yrem(y);
    Vector<gentype> ytmp(y);

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).addTrainingVector(i,yrem,x,Cweigh,epsweigh);

        if ( getmlqmode() )
        {
            for ( int j = 0 ; j < y.size() ; j++ )
            {
                getRepConst(ii).ggTrainingVector(ytmp("&",j),i+j);

                if ( getRepConst(ii).d()(i+j) == 2 )
                {
                    yrem("&",j) -= ytmp(j);
                }

                else if ( getRepConst(ii).d()(i+j) == -1 )
                {
                    if ( ytmp(j) >= yrem(j) )
                    {
                        yrem("&",j) -= ytmp(j);
                    }

                    else
                    {
                        yrem("&",j) *= 0;
                    }
                }

                else if ( getRepConst(ii).d()(i+j) == +1 )
                {
                    if ( ytmp(j) <= yrem(j) )
                    {
                        yrem("&",j) -= ytmp(j);
                    }

                    else
                    {
                        yrem("&",j) *= 0;
                    }
                }
            }
        }
    }

    return res;
}

int BLK_Conect::removeTrainingVector(int i)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).removeTrainingVector(i);
    }

    return res;
}

int BLK_Conect::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).removeTrainingVector(i,y,x);
    }

    return res;
}

int BLK_Conect::removeTrainingVector(int i, int num)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).removeTrainingVector(i,num);
    }

    return res;
}

int BLK_Conect::setx(int i, const SparseVector<gentype> &x)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setx(i,x);
    }

    return res;
}

int BLK_Conect::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setx(i,x);
    }

    return res;
}

int BLK_Conect::setx(const Vector<SparseVector<gentype> > &x)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setx(x);
    }

    return res;
}

int BLK_Conect::sety(int i, const gentype &y)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,y);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,y);
    }

    return res;
}

int BLK_Conect::sety(const Vector<gentype> &y)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(y);
    }

    return res;
}

int BLK_Conect::sety(int i, double z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<double> &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<double> &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(z);
    }

    return res;
}

int BLK_Conect::sety(int i, const Vector<double> &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<Vector<double> > &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<Vector<double> > &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(z);
    }

    return res;
}

int BLK_Conect::sety(int i, const d_anion &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<d_anion> &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<d_anion> &z)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).sety(z);
    }

    return res;
}

int BLK_Conect::setd(int i, int nd)
{
    int is = i;
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    i = ( i >= 0 ) ? i : (-i-1);

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        if ( getmlqmode() && ( is < 0 ) && ( ii == numReps()-1 ) )
        {
            res |= getRep(ii).setd(i,0);
        }

        else
        {
            res |= getRep(ii).setd(i,nd);
        }
    }

    return res;
}

int BLK_Conect::setd(const Vector<int> &i, const Vector<int> &nd)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setd(i,nd);
    }

    return res;
}

int BLK_Conect::setd(const Vector<int> &nd)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setd(nd);
    }

    return res;
}

int BLK_Conect::setCweight(int i, double nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setCweight(i,nv);
    }

    return res;
}

int BLK_Conect::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setCweight(i,nv);
    }

    return res;
}

int BLK_Conect::setCweight(const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setCweight(nv);
    }

    return res;
}

int BLK_Conect::setCweightfuzz(int i, double nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setCweightfuzz(i,nv);
    }

    return res;
}

int BLK_Conect::setCweightfuzz(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setCweightfuzz(i,nv);
    }

    return res;
}

int BLK_Conect::setCweightfuzz(const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setCweightfuzz(nv);
    }

    return res;
}

int BLK_Conect::setsigmaweight(int i, double nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setsigmaweight(i,nv);
    }

    return res;
}

int BLK_Conect::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setsigmaweight(i,nv);
    }

    return res;
}

int BLK_Conect::setsigmaweight(const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setsigmaweight(nv);
    }

    return res;
}

int BLK_Conect::setepsweight(int i, double nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setepsweight(i,nv);
    }

    return res;
}

int BLK_Conect::setepsweight(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setepsweight(i,nv);;
    }

    return res;
}

int BLK_Conect::setepsweight(const Vector<double> &nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setepsweight(nv);;
    }

    return res;
}

int BLK_Conect::scaleCweight(double s)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).scaleCweight(s);
    }

    return res;
}

int BLK_Conect::scaleCweightfuzz(double s)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).scaleCweightfuzz(s);
    }

    return res;
}

int BLK_Conect::scalesigmaweight(double s)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).scalesigmaweight(s);
    }

    return res;
}

int BLK_Conect::scaleepsweight(double s)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).scaleepsweight(s);
    }

    return res;
}

int BLK_Conect::randomise(double sparsity)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).randomise(sparsity);
    }

    return res;
}

int BLK_Conect::autoen(void)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).autoen();
    }

    return res;
}

int BLK_Conect::renormalise(void)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).renormalise();
    }

    return res;
}

int BLK_Conect::realign(void)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).realign();
    }

    return res;
}

int BLK_Conect::setzerotol(double zt)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setzerotol(zt);
    }

    return res;
}

int BLK_Conect::setOpttol(double xopttol)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setOpttol(xopttol);
    }

    return res;
}

int BLK_Conect::setOpttolb(double xopttol)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setOpttolb(xopttol);
    }

    return res;
}

int BLK_Conect::setOpttolc(double xopttol)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setOpttolc(xopttol);
    }

    return res;
}

int BLK_Conect::setOpttold(double xopttol)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setOpttold(xopttol);
    }

    return res;
}

int BLK_Conect::setlr(double xlr)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setlr(xlr);
    }

    return res;
}

int BLK_Conect::setlrb(double xlr)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setlrb(xlr);
    }

    return res;
}

int BLK_Conect::setlrc(double xlr)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setlrc(xlr);
    }

    return res;
}

int BLK_Conect::setlrd(double xlr)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setlrd(xlr);
    }

    return res;
}

int BLK_Conect::setmaxitcnt(int xmaxitcnt)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setmaxitcnt(xmaxitcnt);
    }

    return res;
}

int BLK_Conect::setmaxtraintime(double xmaxtraintime)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setmaxtraintime(xmaxtraintime);
    }

    return res;
}

int BLK_Conect::settraintimeend(double xtraintimeend)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).settraintimeend(xtraintimeend);
    }

    return res;
}

int BLK_Conect::setmaxitermvrank(int nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setmaxitermvrank(nv);
    }

    return res;
}

int BLK_Conect::setlrmvrank(double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setlrmvrank(nv);
    }

    return res;
}

int BLK_Conect::setztmvrank(double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setztmvrank(nv);
    }

    return res;
}

int BLK_Conect::setbetarank(double nv)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setbetarank(nv);
    }

    return res;
}

int BLK_Conect::setC(double xC)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setC(xC);
    }

    return res;
}

int BLK_Conect::setsigma(double xC)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setsigma(xC);
    }

    return res;
}

int BLK_Conect::setsigma_cut(double xC)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setsigma_cut(xC);
    }

    return res;
}

int BLK_Conect::seteps(double xC)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).seteps(xC);
    }

    return res;
}

int BLK_Conect::setCclass(int d, double xC)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setCclass(d,xC);
    }

    return res;
}

int BLK_Conect::setsigmaclass(int d, double xC)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setsigmaclass(d,xC);
    }

    return res;
}

int BLK_Conect::setepsclass(int d, double xC)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setepsclass(d,xC);
    }

    return res;
}

int BLK_Conect::scale(double a)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).scale(a);
    }

    return res;
}

int BLK_Conect::reset(void)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).reset();
    }

    return res;
}

int BLK_Conect::restart(void)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).restart();
    }

    return res;
}

int BLK_Conect::home(void)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).home();
    }

    return res;
}

int BLK_Conect::settspaceDim(int newdim)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).settspaceDim(newdim);
    }

    return res;
}

int BLK_Conect::addtspaceFeat(int i)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).addtspaceFeat(i);
    }

    return res;
}

int BLK_Conect::removetspaceFeat(int i)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).removetspaceFeat(i);
    }

    return res;
}

int BLK_Conect::addxspaceFeat(int i)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).addxspaceFeat(i);
    }

    return res;
}

int BLK_Conect::removexspaceFeat(int i)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).removexspaceFeat(i);
    }

    return res;
}

int BLK_Conect::setsubtype(int i)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setsubtype(i);
    }

    return res;
}

int BLK_Conect::setorder(int neword)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).setorder(neword);
    }

    return res;
}

int BLK_Conect::addclass(int label, int epszero)
{
    int ii;
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRep(ii).addclass(label,epszero);
    }

    return res;
}

int BLK_Conect::train(int &res, svmvolatile int &killSwitch)
{
    res = 0;

    int resi = 0;

    Vector<gentype> yrem;
    Vector<gentype> ytmp;

    if ( getmlqmode() )
    {
        yrem = getRep(0).y();
        ytmp = getRep(0).y();
    }

    for ( int ii = 0 ; ( ii < numReps() ) && !killSwitch ; ++ii )
    {
        if ( getmlqmode() && ii )
        {
            resi |= getRep(ii).sety(yrem);
        }

        if ( getmlqmode() && !ii )
        {
            localy     = getRepConst(ii).y();
            localygood = 1;
        }

        int restemp = 0;
        resi |= getRep(ii).train(restemp,killSwitch);
        res += 100*restemp;

        if ( getmlqmode() && ( ii < numReps()-1 ) )
        {
            for ( int j = 0 ; j < yrem.size() ; j++ )
            {
                getRepConst(ii).ggTrainingVector(ytmp("&",j),j);

                if ( getRepConst(ii).d()(j) == 2 )
                {
                    yrem("&",j) -= ytmp(j);
                }

                else if ( getRepConst(ii).d()(j) == -1 )
                {
                    if ( ytmp(j) >= yrem(j) )
                    {
                        yrem("&",j) -= ytmp(j);
                    }

                    else
                    {
                        yrem("&",j) *= 0;
                    }
                }

                else if ( getRepConst(ii).d()(j) == +1 )
                {
                    if ( ytmp(j) <= yrem(j) )
                    {
                        yrem("&",j) -= ytmp(j);
                    }

                    else
                    {
                        yrem("&",j) *= 0;
                    }
                }
            }
        }
    }

    return resi;
}

int BLK_Conect::disable(int i)
{
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    int is = i;
    int j = ( i >= 0 ) ? i : ((-i-1)%(BLK_Conect::N()));

    for ( int ii = 0 ; ii < numReps() ; ++ii )
    {
        if ( !getmlqmode() || ( ii == numReps()-1 ) || ( is >= 0 ) )
        {
            res |= getRep(ii).disable(j);
        }
    }

    return res;
}

int BLK_Conect::disable(const Vector<int> &i)
{
    int res = 0;
    // NB: if alpha(i) != 0 then this will set isTrained = 0, so retrain and y fix
    // will be automatically updated.

    for ( int jj = 0 ; jj < i.size() ; ++jj )
    {
        int is = i(jj);
        int j = ( i(jj) >= 0 ) ? i(jj) : ((-i(jj)-1)%(BLK_Conect::N()));

        for ( int ii = 0 ; ii < numReps() ; ++ii )
        {
            if ( !getmlqmode() || ( ii == numReps()-1 ) || ( is >= 0 ) )
            {
                res |= getRep(ii).disable(j);
            }
        }
    }

    return res;
}


double BLK_Conect::sparlvl(void) const
{
    int ii;
    double res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res += (getRepConst(ii).sparlvl())/numReps();
    }

    return res;
}

const Vector<int> &BLK_Conect::d(void) const
{
    int jj;

    dscratch.resize(N());
    dscratch = 2;

    for ( jj = 0 ; jj < N() ; ++jj )
    {
        if ( numReps() )
        {
            dscratch("[]",jj) = (getRepConst(0).d())(jj); // d should be properly set at base level, not necessarily above that
        }

        else
        {
            dscratch("[]",jj) = (getRepConst().d())(jj);
        }
    }

    return dscratch;
}

const Vector<int> &BLK_Conect::alphaState(void) const
{
    int ii,jj;
    alphaStateScratch.resize(N());
    alphaStateScratch = 1;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        for ( jj = 0 ; jj < N() ; ++jj )
        {
            alphaStateScratch("[]",jj) |= (getRepConst(ii).alphaState())(jj);
        }
    }

    return alphaStateScratch;
}















const Vector<gentype> &BLK_Conect::y(void) const
{
    Vector<gentype> &res = localy;

    if ( localygood <= 0 )
    {
        int ii,jj;

        Vector<Vector<gentype> > &yall = localyparts;

        if ( !localygood )
        {
            yall.resize(numReps());

            NiceAssert( locxmax.size() == locxmin.size() );

            // Pre-calculate y vectors

            bool needgrid = false;

            for ( ii = 0 ; ii < numReps() ; ++ii )
            {
                if ( ( ( getRepConst(ii).type() >= 0   ) && ( getRepConst(ii).type() <=  99 ) ) ||
                     ( ( getRepConst(ii).type() >= 100 ) && ( getRepConst(ii).type() <= 199 ) ) ||
                     ( ( getRepConst(ii).type() >= 400 ) && ( getRepConst(ii).type() <= 499 ) ) ||
                     ( ( getRepConst(ii).type() >= 500 ) && ( getRepConst(ii).type() <= 599 ) ) ||
                     ( getRepConst(ii).type() == 201 ) ||
                     ( getRepConst(ii).type() == 202 ) ||
                     ( getRepConst(ii).type() == 203 ) ||
                     ( getRepConst(ii).type() == 205 ) ||
                     ( getRepConst(ii).type() == 206 ) ||
                     ( getRepConst(ii).type() == 207 ) )
                {
                    yall("&",ii) = getRepConst(ii).y();
                }

                else
                {
                    needgrid = true;
                }
            }

            if ( needgrid )
            {
                // Generate x grid only if required.

                Vector<SparseVector<gentype> > xgrid;
                static thread_local GPR_Generic sampler;
                sampler.genSampleGrid(xgrid,locxmin,locxmax,locNsamp,locsampSplit,locxsampType,locsampSlack);

                int totsamp = xgrid.size();

                for ( ii = 0 ; ii < numReps() ; ++ii )
                {
                    if ( !( ( ( getRepConst(ii).type() >= 0   ) && ( getRepConst(ii).type() <=  99 ) ) ||
                           ( ( getRepConst(ii).type() >= 100 ) && ( getRepConst(ii).type() <= 199 ) ) ||
                           ( ( getRepConst(ii).type() >= 400 ) && ( getRepConst(ii).type() <= 499 ) ) ||
                           ( ( getRepConst(ii).type() >= 500 ) && ( getRepConst(ii).type() <= 599 ) ) ||
                           ( getRepConst(ii).type() == 201 ) ||
                           ( getRepConst(ii).type() == 202 ) ||
                           ( getRepConst(ii).type() == 205 ) ||
                           ( getRepConst(ii).type() == 206 ) ) )
                    {
                        yall("&",ii).resize(totsamp);

                        for ( jj = 0 ; jj < totsamp ; ++jj )
                        {
                            getRepConst(ii).gg(yall("&",ii)("&",jj),xgrid(jj));
                        }
                    }
                }
            }
        }

        localygood = 1;

        for ( ii = 0 ; ii < numReps() ; ++ii )
        {
            if ( !ii )
            {
                res = yall(ii);
                res.scale(getRepWeight(ii));
            }

            else
            {
                res.scaleAdd(getRepWeight(ii),yall(ii));
            }
        }
    }

    return res;
}


int BLK_Conect::isVarDefined(void) const
{
    int res = 1;

    for ( int ii = ( ( getmlqmode() == 2 ) ? (numReps()-1) : 0 )  ; ii < numReps() ; ++ii )
    {
        if ( res == 1 )
        {
            if ( getRepConst(ii).isVarDefined() == 0 )
            {
                res = 0;
                break;
            }

            else if ( getRepConst(ii).isVarDefined() == 2 )
            {
                res = 2;
            }
        }

        else if ( res == 2 )
        {
            if ( getRepConst(ii).isVarDefined() == 0 )
            {
                res = 0;
                break;
            }
        }
    }

    return res;
}

int BLK_Conect::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !retaltg );

    Vector<gentype> vech(numReps());
    Vector<gentype> vecg(numReps());

    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRepConst(ii).ghTrainingVector(vech("&",ii),vecg("&",ii),i,retaltg,pxyprodi);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecg("&",ii) *= getRepWeight(ii);

        if ( !isClassifier() )
        {
            vech("&",ii) *= getRepWeight(ii);
        }
    }

    if ( isClassifier() )
    {
        SparseVector<gentype> qq(vech);

        res |= combit.gh(resh,resg,qq,retaltg);
    }

    else
    {
        sum(resg,vecg);
        sum(resh,vech);
    }

    return res;
}

void BLK_Conect::fastg(gentype &res) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii));
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,xa,xb,xainfo,xbinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii));
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,xa,xb,xainfo,xbinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}




int BLK_Conect::gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg, const vecInfo *xinf, gentype ***pxyprodi) const
{
//errstream() << "phantomxgh gh 0: " << numReps() << "\n";
    Vector<gentype> vech(numReps());
    Vector<gentype> vecg(numReps());

    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        res |= getRepConst(ii).gh(vech("&",ii),vecg("&",ii),x,retaltg,xinf,pxyprodi);
//errstream() << "phantomxgh gh(" << ii << ") 1: " << vech(ii) << "\n";
//errstream() << "phantomxgh gh(" << ii << ") 2: " << vecg(ii) << "\n";
//errstream() << "phantomxgh gh(" << ii << ") 2b: " << getRepConst(ii).type() << "\n";

        vecg("&",ii) *= getRepWeight(ii);
//errstream() << "phantomxgh gh(" << ii << ") 3: " << vecg(ii) << "\n";

        if ( !isClassifier() )
        {
            vech("&",ii) *= getRepWeight(ii);
//errstream() << "phantomxgh gh(" << ii << ") 4: " << vech(ii) << "\n";
        }
    }

    if ( isClassifier() )
    {
        SparseVector<gentype> qq(vech);

        res |= combit.gh(resh,resg,qq,retaltg);
//errstream() << "phantomxgh gh 5 WTF!!!\n";
    }

    else
    {
        sum(resg,vecg); // Not mean, this is mostly used by globalopt
//errstream() << "phantomxgh gh 6: " << resg << "\n";
        sum(resh,vech);
//errstream() << "phantomxgh gh 7: " << resh << "\n";
    }
//errstream() << "phantomxgh gh 8: " << res << "\n";

    return res;
}

void BLK_Conect::stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).stabProb(vecres("&",ii),x,p,pnrm,rot,mu,B);

        vecres("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res,vecres);
    }

    return;
}

void BLK_Conect::stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).stabProbTrainingVector(vecres("&",ii),i,p,pnrm,rot,mu,B);

        vecres("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res,vecres);
    }

    return;
}


void BLK_Conect::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    Vector<Vector<gentype> > vecres(numReps());
    Vector<gentype> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dgTrainingVector(vecres("&",ii),vecresn("&",ii),i);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dgTrainingVector(Vector<gentype> &res, const Vector<int> &i) const
{
    Vector<Vector<gentype> > vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dgTrainingVector(vecres("&",ii),i);

        vecres("&",ii).scale(getRepWeight(ii));
    }

    {
        sum(res,vecres);
    }

    return;
}

void BLK_Conect::dg(Vector<gentype> &res, gentype &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<gentype> > vecres(numReps());
    Vector<gentype> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dg(Vector<double> &res, double &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<double> > vecres(numReps());
    Vector<double> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<Vector<double> > > vecres(numReps());
    Vector<Vector<double> > vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<gentype> > vecresx(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dgX(vecresx("&",ii),x,xinf);

        vecresx("&",ii).scale(getRepWeight(ii));
    }

    {
        sum(resx,vecresx);
    }

    return;
}

void BLK_Conect::dgX(Vector<double> &resx, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<double> > vecresx(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dgX(vecresx("&",ii),x,xinf);

        vecresx("&",ii).scale(getRepWeight(ii));
    }

    {
        sum(resx,vecresx);
    }

    return;
}

void BLK_Conect::dg(Vector<d_anion> &res, d_anion &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<d_anion> > vecres(numReps());
    Vector<d_anion> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

int BLK_Conect::covTrainingVector(gentype &resv, gentype &resmu,int i, int j, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    Vector<gentype> vecv(numReps());
    Vector<gentype> vecg(numReps());
    //gentype dummy;

    (void) j;

    NiceAssert( i == j );

    int ii;
    int res = 0;
    gentype dummyh;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        if ( !getmlqmode() || ( ii == numReps()-1 ) )
        {
            //res |= getRepConst(ii).ghTrainingVector(dummy,vecg("&",ii),i,0,pxyprodi);
            res |= getRepConst(ii).covTrainingVector(vecv("&",ii),vecg("&",ii),i,i,pxyprodi,pxyprodj,pxyprodij);

            vecg("&",ii) *= getRepWeight(ii);
            vecv("&",ii) *= getRepWeight(ii,1)*getRepWeight(ii,1);
        }

        else
        {
            res |= getRepConst(ii).ghTrainingVector(dummyh,vecg("&",ii),i,0,pxyprodi);

            vecg("&",ii) *= getRepWeight(ii);
            vecv("&",ii)  = 0.0;
        }
    }

    gentype addterm;

    sum(resv,vecv);
    sum(resmu,vecg);

    if ( !getmlqmode() )
    {
        vari(addterm,vecg);
        addterm *= (double) numReps();
        resv += addterm;
    }

    return res;
}

int BLK_Conect::cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf, const vecInfo *xbinf, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    Vector<gentype> vecv(numReps());
    Vector<gentype> vecg(numReps());
    //gentype dummy;

    NiceAssert( xa == xb );

    int ii;
    int res = 0;
    gentype dummyh;

    for ( ii = 0 ; ii < numReps() ; ++ii )
    {
        if ( !getmlqmode() || ( ii == numReps()-1 ) )
        {
            //res = getRepConst(ii).gh(dummy,vecg("&",ii),xa,0,xainf,pxyprodi);
            res = getRepConst(ii).cov(vecv("&",ii),vecg("&",ii),xa,xb,xainf,xbinf,pxyprodi,pxyprodj,pxyprodij);

            vecg("&",ii) *= getRepWeight(ii);
            vecv("&",ii) *= getRepWeight(ii,1)*getRepWeight(ii,1);
        }

        else
        {
            res |= getRepConst(ii).gh(dummyh,vecg("&",ii),xa,0,xainf,pxyprodi);

            vecg("&",ii) *= getRepWeight(ii);
            vecv("&",ii)  = 0.0;
        }
    }

    gentype addterm;

    sum(resv,vecv);
    sum(resmu,vecg);

    if ( !getmlqmode() )
    {
        vari(addterm,vecg);
        addterm *= (double) numReps();
        resv += addterm;
    }

    return res;
}
