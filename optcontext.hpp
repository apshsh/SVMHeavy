
//
// Quadratic optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _optcontext_h
#define _optcontext_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "vector.hpp"
#include "matrix.hpp"
#include "chol.hpp"
#include "numbase.hpp"

// Background
// ==========
//
// Consider the quadratic programming problem:
//
// [ alpha ]' [ Gp   Gpn ] [ alpha ] + [ alpha ]' [ gp ] + | alpha' |' [ hp ]
// [ beta  ]  [ Gpn' Gn  ] [ beta  ]   [ beta  ]  [ gn ]   | beta   |  [ 0  ]
//
// where alpha \in \Re^{aN} and beta \in \Re^{bN}.  It is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gn is negative semi-definite hermitian
//
// where each variable is in one of the following states:
//
// - alpha[i] = lb[i]         => sgn(alpha[i]) = -1 (constrained)
// - lb[i] <= alpha[i] <= 0   => sgn(alpha[i]) = -1 (free)
// - alpha[i] = 0             => sgn(alpha[i]) = 0  (constrained)
// - 0 <= alpha[i] <= ub[i]   => sgn(alpha[i]) = +1 (free)
// - alpha[i] = ub[i]         => sgn(alpha[i]) = +1 (constrained)
//
// - beta[i] = 0  (constrained)
// - beta[i] free (free)
//
// The positive and negative states of alpha are differentiated to allow for
// the hp term, which presents a step discontinuity in the gradient at
// alpha[i] = 0.  Pivotting identifies these states: specifically, we have the
// following pivot vectors (integer vectors):
//
// - pAlphaLB  s.t. alpha[pAlphaLB] == lb[pAlphaLB]      (constrained)
// - pAlphaZ   s.t. alpha[pAlphaZ] == 0                  (constrained)
// - pAlphaUB  s.t. alpha[pAlphaUB] == ub[pAlphaUB]      (constrained)
// - pAlphaF   s.t. lb[pAlphaLF] <= alpha[pAlphaLF] <= 0 (free)
//               or 0 <= alpha[pAlphaUF] <= ub[pAlphaUF] (free)
//
// - pBetaC s.t. beta[pBetaC] = 0  (constrained)
// - pBetaF s.t. beta[pBetaF] free (free)
//
// where every 0 <= i <= aN-1 is located in precisely one of the pAlpha* pivot
// vectors, and likewise every 0 <= j <= bN-1 is located in precisely one of
// the pBeta* pivot vectors.  We also define
//
// - dalphaState is a vector recording the state of alpha:
//
//    dalphaState[i] = -2: alpha[i] == lb[i]         (dalphaState[pAlphaLB] == -2)
//    dalphaState[i] = -1: lb[i] <= alpha[i] <= 0    (dalphaState[pAlphaF]  == +-1)
//    dalphaState[i] =  0: alpha[i] == 0             (dalphaState[pAlphaZ]  == 0)
//    dalphaState[i] = +1: 0 <= alpha[i] <= ub[i]    (dalphaState[pAlphaF]  == +-1)
//    dalphaState[i] = +2: alpha[i] == ub[i]         (dalphaState[pAlphaLB] == +2)
//
// - dbetaState is a vector recording the state of beta:
//
//    dbetaState[i] = 0: beta[i] == 0           (dbetaState[pBetaC] = 0)
//    dbetaState[i] = 1: beta[i] unconstrained  (dbetaState[pBetaF] = 1)
//
// The task of this class is to keep track of all the relevant pivot vectors
// and also maintain a (part) cholesky factorisation of the active part of
// the hessian if required, where the active part of the hessian is:
//
// [ Gp[pAlphaF,pAlphaF]   Gpn[pAlphaF,pBetaF] ]
// [ Gpn[pAlphaF,pBetaF]'  Gn[pBetaF,pBetaF]   ]
//
//
// Cholesky Factorisation
// ======================
//
// If keepfact is set then the class also maintains the cholesky factorisation,
// as stated above.  This is done using the chol.h class.  All of the pAlphaF
// components are placed into this class as well as the first nfact pBetaF
// elements.  The precise order in which the elements are placed into the
// factorisation (the interleaving) is controlled by the D vector:
//
// - D is the same size as the factorisation, namely nfact+size(pAlphaF).
// - D[i] == +1 implies that a row/column from Gp[pAlphaF,pAlphaf] should
//   be inserted at row/column i of the factorisation
// - D[i] == -1 implies that a row/column from Gn[pBetaF,pBetaF] should
//   be inserted at row/column i of the factorisation
//
// We also maintain the factor vectors fAlphaF and fBetaF as follows:
//
// - fAlphaF[i] = the position of the Gp[pAlphaF,pAlphaF][i,i] in the
//   factorisation
// - fBetaF[i] = the position of the Gn[pBetaF,pBetaF][i,i] in the
//   factorisation, or -1 if it is not included in the factorisation.
//
// Where we note that:
//
// - D[fAlphaF[i]] = +1
// - D[fBetaF[i]]  = -1
// - fAlphaF[i] < fAlphaF[i+1]
// - fBetaF[i]  < fBetaF[i+1] for all i : fBetaF[i] != -1 and fBetaF[i+1] != -1
//
// General rules for the factorisation:
//
// - the first nfact elements of pBetaF are included in the non-singular part
//   of the factorisation and hence have fBetaF[i] != -1 (i < nfact).
// - the remainder of elements of pBetaF are not included in the factorisation
//   and hence have fBetaF[i] == -1 (i >= nfact).
// - the first pfact elements of pAlphaF are included in the non-singular part
//   of the factorisation.
// - the remainder of elements of pAlphaF are in the singular part of the
//   factorisation.
//
// The ordering of the elements in pBetaF is set to maximise the number of
// elements (nfact) there are in the factorisation.  Ditto the elements in
// pAlphaF.
//
// When elements have been added to or removed from the factorisation or
// elements have been added to pBetaF but not the factorisation then the
// algorithm to update the factorisation is:
//
// 1. If there are elements i such that fBetaF[i] != -1 is in the singular
//    part of the factorisation then remove them.
// 2. If the non-singular part of the factorisation is empty or consists
//    entirely of elements from Gn then:
//    a. add as many elements of Gp to non-singular part of factorisation as
//       possible.
//    b. add as many elements of Gn to non-singular part of factorisation as
//       possible.
//    c. goto a until no more progress can be made.
//
// Finally, betaFix is defined as follows:
//
// - betaFix[i] == 0 means that beta[i] may be included in the factorisation
//   if unconstrained
// - betaFix[i] == 1 means that beta[i] cannot be included in the factorisation
//   even if it is not constrained.
// - betaFix[i] == -1 means that the state is unknown.
//
// This variable is intended to speed things up for those betas corresponding
// to all zero columns in the Gpn matrix which therefore cannot be successfully
// included in the factorisation, so there is no need to waste time trying.
// betaFix[i] == 0 unless otherwise set.


// Stream operators

class optContext;

std::ostream &operator<<(std::ostream &output, const optContext &src );
std::istream &operator>>(std::istream &input,        optContext &dest);

// Swap function

inline void qswap(optContext &a, optContext &b);

#define CALCBETAFIX(iii) ( ( ( (GpnFColNorm)(iii) <= dzt ) && ( Gn((iii),(iii)) >= -dzt ) ) ? 1 : 0 )

class optContext
{
    friend std::ostream &operator<<(std::ostream &output, const optContext &src );
    friend std::istream &operator>>(std::istream &input,        optContext &dest);

    friend inline void qswap(optContext &a, optContext &b);

public:

    // Constructors and assignment operators
    //
    // Note that constructors assume that all alphas are constrained to zero
    // and that all betas are constrained.

    optContext(void) : dzt(DEFAULT_ZTOL), dkeepfact(0), daNLF(0), daNUF(0), betaFixUpdate(0), dnfact(0), dpfact(0) { ; }
    optContext(const optContext &src) { *this = src; }

    optContext &operator=(const optContext &src)
    {
        pAlphaLB = src.pAlphaLB;
        pAlphaZ  = src.pAlphaZ;
        pAlphaUB = src.pAlphaUB;
        pAlphaF  = src.pAlphaF;

        pBetaC = src.pBetaC;
        pBetaF = src.pBetaF;

        dalphaState = src.dalphaState;
        dbetaState  = src.dbetaState;

        dzt       = src.dzt;
        dkeepfact = src.dkeepfact;

        daNLF = src.daNLF;
        daNUF = src.daNUF;

        GpnFColNorm   = src.GpnFColNorm;
        betaFix       = src.betaFix;
        betaFixUpdate = src.betaFixUpdate;

        dnfact = src.dnfact;
        dpfact = src.dpfact;

        D           = src.D;
        freeVarChol = src.freeVarChol;

        fAlphaF = src.fAlphaF;
        fBetaF  = src.fBetaF;

        return *this;
    }

    // Reconstructors:
    //
    // refact: like the constructor, but keeps the existing alpha LB/LF/Z/UF/UB
    //         and beta F/C information.
    // reset:  using modAlphaxxtoZ and modBetaFtoC functions to put all alpha
    //         in Z and all beta constrained.
    //
    // keepfact = 0: do not maintain the cholesky factorisation.
    //            1: do maintain the cholesky factorisation.
    // zt = zero tolerance for cholesky factorisation.

    void refact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int xkeepfact = -1, double xzt = -1);
    void reset (const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn);

    // Control functions using unpivotted index:
    //
    // addAlpha: add alpha[i], presumed constrained to zero
    // addBeta:  add beta[i],  presumed constrained to zero
    //
    // removeAlpha: remove alpha[i], presumed constrained to zero
    // removeBeta:  remove beta[i],  presumed constrained to zero
    //
    // Notes:
    //
    // - addAlpha and addBeta return the new position of the variable in the
    //   relevant pivot vector (either pAlphaZ or pBetaC)
    // - removeAlpha and removeBeta return the old position of the variable in
    //   the relevant pivot vector (either pAlphaZ or pBetaC)

    int addAlpha(int i);
    int addBeta (int i);

    int removeAlpha(int i);
    int removeBeta (int i);

    // Find position in pivotted variables

    int findInAlphaLB(int i) const
    {
        int iP = -1;

        for ( iP = 0 ; iP < aNLB() ; ++iP )
        {
            if ( pivAlphaLB(iP) == i )
            {
                break;
            }
        }

        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNLB() );

        return iP;
    }

    int findInAlphaZ(int i) const
    {
        int iP = -1;

        for ( iP = 0 ; iP < aNZ() ; ++iP )
        {
            if ( pivAlphaZ(iP) == i )
            {
                break;
	    }
        }

        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNZ() );

        return iP;
    }

    int findInAlphaUB(int i) const
    {
        int iP = -1;

        for ( iP = 0 ; iP < aNUB() ; ++iP )
        {
            if ( pivAlphaUB(iP) == i )
            {
                break;
	    }
        }

        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNUB() );

        return iP;
    }

    int findInAlphaF(int i) const
    {
        int iP = -1;

        for ( iP = 0 ; iP < aNF() ; ++iP )
        {
            if ( pivAlphaF(iP) == i )
            {
                break;
	    }
        }

        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );

        return iP;
    }

    int findInBetaC(int i) const
    {
        int iP = -1;

        for ( iP = 0 ; iP < bNC() ; ++iP )
        {
            if ( pivBetaC(iP) == i )
            {
                break;
	    }
        }

        NiceAssert( iP >= 0 );
        NiceAssert( iP < bNC() );

        return iP;
    }

    int findInBetaF(int i) const
    {
        int iP = -1;

        for ( iP = 0 ; iP < bNF() ; ++iP )
        {
            if ( pivBetaF(iP) == i )
            {
                break;
            }
        }

        NiceAssert( iP >= 0 );
        NiceAssert( iP < bNF() );

        return iP;
    }

    // Variable state control using index to relevant pivot vector
    //
    // modAlpha*to#: remove variable pAlpha*[iP] and put it in pAlpha#
    // modBeta*to#:  remove variable pBeta*[iP]  and put it in pBeta#
    //
    // Notes:
    //
    // - Each function returns the new position in the relevant pivot vector
    //   (either pAlphaLB, pAlphaZ, pAlphaUB, pAlphaF, pBetaC, or pBetaF).
    // - pAlphaF may change if all elements of Gp and Gn are zero before or
    //   after the operation.
    // - Because the ordering of variables in pAlphaF and pBetaF may change
    //   we include arguments apos and bpos.  Basically if point apos in
    //   pAlphaF moves then the function will change apos to the position
    //   to which it has moved.  Ditto bpos and pBetaF.

    int modAlphaLBtoLF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNLB() );
        NiceAssert( alphaState(pivAlphaLB(iP)) == -2 );

        dalphaState.sv(pivAlphaLB(iP),-1);

        ++daNLF;

        int i = pivAlphaLB(iP);
        int insertPos = pAlphaF.size();

        pAlphaLB.remove(iP);
        pAlphaF.add(pAlphaF.size());
        pAlphaF.sv(pAlphaF.size()-1,i);

        if ( dkeepfact )
        {
	    insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaLBtoZ (int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNLB() );
        NiceAssert( alphaState(pivAlphaLB(iP)) == -2 );

        dalphaState.sv(pivAlphaLB(iP),0);

        int i = pivAlphaLB(iP);
        int insertPos = pAlphaZ.size();

        pAlphaLB.remove(iP);
        pAlphaZ.add(pAlphaZ.size());
        pAlphaZ.sv(pAlphaZ.size()-1,i);

        return insertPos;
    }

    int modAlphaLBtoUF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNLB() );
        NiceAssert( alphaState(pivAlphaLB(iP)) == -2 );

        dalphaState.sv(pivAlphaLB(iP),+1);

        ++daNUF;

        int i = pivAlphaLB(iP);
        int insertPos = pAlphaF.size();

        pAlphaLB.remove(iP);
        pAlphaF.add(pAlphaF.size());
        pAlphaF.sv(pAlphaF.size()-1,i);

        if ( dkeepfact )
        {
	    insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaLBtoUB(int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNLB() );
        NiceAssert( alphaState(pivAlphaLB(iP)) == -2 );

        dalphaState.sv(pivAlphaLB(iP),+2);

        int i = pivAlphaLB(iP);
        int insertPos = pAlphaUB.size();

        pAlphaLB.remove(iP);
        pAlphaUB.add(pAlphaUB.size());
        pAlphaUB.sv(pAlphaUB.size()-1,i);

        return insertPos;
    }

    int modAlphaLFtoLB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == -1 );

        dalphaState.sv(pivAlphaF(iP),-2);

        --daNLF;

        int i = pivAlphaF(iP);
        int insertPos = pAlphaLB.size();

        pAlphaF.remove(iP);
        pAlphaLB.add(pAlphaLB.size());
        pAlphaLB.sv(pAlphaLB.size()-1,i);

        if ( dkeepfact )
        {
	    shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaLFtoZ (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == -1 );

        dalphaState.sv(pivAlphaF(iP),0);

        --daNLF;

        int i = pivAlphaF(iP);
        int insertPos = pAlphaZ.size();

        pAlphaF.remove(iP);
        pAlphaZ.add(pAlphaZ.size());
        pAlphaZ.sv(pAlphaZ.size()-1,i);

        if ( dkeepfact )
        {
	    shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaLFtoUF(int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == -1 );

        --daNLF;
        ++daNUF;

        dalphaState.sv(pivAlphaF(iP),+1);

        return iP;
    }

    int modAlphaLFtoUB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == -1 );

        dalphaState.sv(pivAlphaF(iP),+2);

        --daNLF;

        int i = pivAlphaF(iP);
        int insertPos = pAlphaUB.size();

        pAlphaF.remove(iP);
        pAlphaUB.add(pAlphaUB.size());
        pAlphaUB.sv(pAlphaUB.size()-1,i);

        if ( dkeepfact )
        {
	    shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaZtoLB (int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNZ() );
        NiceAssert( alphaState(pivAlphaZ(iP)) == 0 );

        dalphaState.sv(pivAlphaZ(iP),-2);

        int i = pivAlphaZ(iP);
        int insertPos = pAlphaLB.size();

        pAlphaZ.remove(iP);
        pAlphaLB.add(pAlphaLB.size());
        pAlphaLB.sv(pAlphaLB.size()-1,i);

        return insertPos;
    }

    int modAlphaZtoLF (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNZ() );
        NiceAssert( alphaState(pivAlphaZ(iP)) == 0 );

        dalphaState.sv(pivAlphaZ(iP),-1);

        ++daNLF;

        int i = pivAlphaZ(iP);
        int insertPos = (pAlphaF.size());

        pAlphaZ.remove(iP);
        pAlphaF.add(pAlphaF.size());
        pAlphaF.sv(pAlphaF.size()-1,i);

        if ( dkeepfact )
        {
	    insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaZtoUF (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNZ() );
        NiceAssert( alphaState(pivAlphaZ(iP)) == 0 );

        dalphaState.sv(pivAlphaZ(iP),+1);

        ++daNUF;

        int i = pivAlphaZ(iP);
        int insertPos = (pAlphaF.size());

        pAlphaZ.remove(iP);
        pAlphaF.add(pAlphaF.size());
        pAlphaF.sv(pAlphaF.size()-1,i);

        if ( dkeepfact )
        {
	    insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaZtoUB (int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNZ() );
        NiceAssert( alphaState(pivAlphaZ(iP)) == 0 );

        dalphaState.sv(pivAlphaZ(iP),+2);

        int i = pivAlphaZ(iP);
        int insertPos = pAlphaUB.size();

        pAlphaZ.remove(iP);
        pAlphaUB.add(pAlphaUB.size());
        pAlphaUB.sv(pAlphaUB.size()-1,i);

        return insertPos;
    }

    int modAlphaUFtoLB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == +1 );

        dalphaState.sv(pivAlphaF(iP),-2);

        --daNUF;

        int i = pivAlphaF(iP);
        int insertPos = pAlphaLB.size();

        pAlphaF.remove(iP);
        pAlphaLB.add(pAlphaLB.size());
        pAlphaLB.sv(pAlphaLB.size()-1,i);

        if ( dkeepfact )
        {
	    shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaUFtoLF(int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == +1 );

        --daNUF;
        ++daNLF;

        dalphaState.sv(pivAlphaF(iP),-1);

        return iP;
    }

    int modAlphaUFtoZ (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == +1 );

        dalphaState.sv(pivAlphaF(iP),0);

        --daNUF;

        int i = pivAlphaF(iP);
        int insertPos = pAlphaZ.size();

        pAlphaF.remove(iP);
        pAlphaZ.add(pAlphaZ.size());
        pAlphaZ.sv(pAlphaZ.size()-1,i);

        if ( dkeepfact )
        {
	    shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaUFtoUB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNF() );
        NiceAssert( alphaState(pivAlphaF(iP)) == +1 );

        dalphaState.sv(pivAlphaF(iP),+2);

        --daNUF;

        int i = pivAlphaF(iP);
        int insertPos = pAlphaUB.size();

        pAlphaF.remove(iP);
        pAlphaUB.add(pAlphaUB.size());
        pAlphaUB.sv(pAlphaUB.size()-1,i);

        if ( dkeepfact )
        {
	    shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaUBtoLB(int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNUB() );
        NiceAssert( alphaState(pivAlphaUB(iP)) == +2 );

        dalphaState.sv(pivAlphaUB(iP),-2);

        int i = pivAlphaUB(iP);
        int insertPos = pAlphaLB.size();

        pAlphaUB.remove(iP);
        pAlphaLB.add(pAlphaLB.size());
        pAlphaLB.sv(pAlphaLB.size()-1,i);

        return insertPos;
    }

    int modAlphaUBtoLF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNUB() );
        NiceAssert( alphaState(pivAlphaUB(iP)) == +2 );

        dalphaState.sv(pivAlphaUB(iP),-1);

        ++daNLF;

        int i = pivAlphaUB(iP);
        int insertPos = (pAlphaF.size());

        pAlphaUB.remove(iP);
        pAlphaF.add(pAlphaF.size());
        pAlphaF.sv(pAlphaF.size()-1,i);

        if ( dkeepfact )
        {
	    insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modAlphaUBtoZ (int iP)
    {
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNUB() );
        NiceAssert( alphaState(pivAlphaUB(iP)) == +2 );

        dalphaState.sv(pivAlphaUB(iP),0);

        int i = pivAlphaUB(iP);
        int insertPos = pAlphaZ.size();

        pAlphaUB.remove(iP);
        pAlphaZ.add(pAlphaZ.size());
        pAlphaZ.sv(pAlphaZ.size()-1,i);

        return insertPos;
    }

    int modAlphaUBtoUF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < aNUB() );
        NiceAssert( alphaState(pivAlphaUB(iP)) == +2 );

        dalphaState.sv(pivAlphaUB(iP),+1);

        ++daNUF;

        int i = pivAlphaUB(iP);
        int insertPos = (pAlphaF.size());

        pAlphaUB.remove(iP);
        pAlphaF.add(pAlphaF.size());
        pAlphaF.sv(pAlphaF.size()-1,i);

        if ( dkeepfact )
        {
	    insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modBetaCtoF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < bNC() );
        NiceAssert( betaState(pivBetaC(iP)) == 0 );

        dbetaState.sv(pivBetaC(iP),1);

        int i = pivBetaC(iP);
        int insertPos = (pBetaF.size());

        pBetaC.remove(iP);
        pBetaF.add(pBetaF.size());
        pBetaF.sv(pBetaF.size()-1,i);

        if ( dkeepfact )
        {
	    insertPos = extendFactBeta(Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    int modBetaFtoC(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( iP >= 0 );
        NiceAssert( iP < bNF() );
        NiceAssert( betaState(pivBetaF(iP)) == 1 );

        dbetaState.sv(pivBetaF(iP),0);

        int i = pivBetaF(iP);
        int insertPos = pBetaC.size();

        pBetaF.remove(iP);
        pBetaC.add(pBetaC.size());
        pBetaC.sv(pBetaC.size()-1,i);

        if ( dkeepfact )
        {
	    shrinkFactBeta(iP,Gp,Gn,Gpn,apos,bpos);
        }

        return insertPos;
    }

    // Batch variable state control
    //
    // modAlphaFAlltoLowerBound: remove all variables from pAlphaLF and put in pAlphaLB
    //                           remove all variables from pAlphaUF and put in pAlphaZ
    // modAlphaFAlltoUpperBound: remove all variables from pAlphaLF and put in pAlphaZ
    //                           remove all variables from pAlphaUF and put in pAlphaUB

    void modAlphaFAlltoLowerBound(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );

        while ( aNF() )
        {
	    if ( alphaState(pivAlphaF(aNF()-1)) == -1 )
	    {
	        modAlphaLFtoLB(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	    }

            else
	    {
	        modAlphaUFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	    }
        }

        return;
    }

    void modAlphaFAlltoUpperBound(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );

        while ( aNF() )
        {
	    if ( alphaState(pivAlphaF(aNF()-1)) == -1 )
	    {
	        modAlphaLFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	    }

	    else
	    {
	        modAlphaUFtoUB(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	    }
        }

        return;
    }

    // Pivotting and constraint data

    const Vector<int> &pivAlphaLB(void) const { return pAlphaLB; }
    const Vector<int> &pivAlphaZ (void) const { return pAlphaZ;  }
    const Vector<int> &pivAlphaUB(void) const { return pAlphaUB; }
    const Vector<int> &pivAlphaF (void) const { return pAlphaF;  }

    const Vector<int> &pivBetaC(void) const { return pBetaC; }
    const Vector<int> &pivBetaF(void) const { return pBetaF; }

    const Vector<int> &alphaState(void) const { return dalphaState; }
    const Vector<int> &betaState(void)  const { return dbetaState;  }

    int pivAlphaLB(int iP) const { return pAlphaLB.v(iP); }
    int pivAlphaZ (int iP) const { return pAlphaZ.v(iP);  }
    int pivAlphaUB(int iP) const { return pAlphaUB.v(iP); }
    int pivAlphaF (int iP) const { return pAlphaF.v(iP);  }

    int pivBetaC(int iP) const { return pBetaC.v(iP); }
    int pivBetaF(int iP) const { return pBetaF.v(iP); }

    int alphaState(int i) const { return dalphaState.v(i); }
    int betaState (int i) const { return dbetaState.v(i);  }

    // Information functions

    int aNLB(void) const { return pAlphaLB.size();     }
    int aNLF(void) const { return daNLF;               }
    int aNZ (void) const { return pAlphaZ.size();      }
    int aNUF(void) const { return daNUF;               }
    int aNUB(void) const { return pAlphaUB.size();     }
    int aNF (void) const { return pAlphaF.size();      }
    int aNC (void) const { return aNLB()+aNZ()+aNUB(); }
    int aN  (void) const { return aNF()+aNC();         }

    int bNF(void) const { return pBetaF.size(); }
    int bNC(void) const { return pBetaC.size(); }
    int bN (void) const { return bNF()+bNC();   }

    int keepfact(void) const { return dkeepfact; }
    double zt(void)    const { return dzt;       }

    // Factorisation functions
    //
    // These basically reflect the functions of chol.h, but take care of all the
    // requisit matrix pivotting.  That is:
    //
    // - bp is replaced by bp(pAlphaF)
    // - bn is replaced by bn(pBetaF)(0,1,nfact-1)
    // - Gp is replaced by Gp(pAlphaF,pAlphaF)
    // - Gn is replaced by Gn(pBetaF,pBetaF)(0,1,nfact-1,0,1,nfact-1)
    // - Gpn is replaced by Gpn(pAlphaF,pBetaF)(0,1,pAlphaF.size()-1,0,1,nfact-1)
    //
    // When calling fact_minverse and near_invert the following pivots are
    // also used:
    //
    // - ap is replaced by ap("&",pAlphaF)("&",0,1,fact.npos()-fact.nbadpos()-1)
    // - an is replaced by an("&",pBetaF)("&",0,1,nfact-1)
    // - bp is replaced by bp("&",pAlphaF)("&",0,1,fact.npos()-fact.nbadpos()-1)
    // - bn is replaced by bn("&",pBetaF)("&",0,1,nfact-1)
    //
    // and moreover when returning from these:
    //
    // - ap("&",pAlphaF)("&",fact.npos()-fact.nbadpos(),1,pAlphaF.size()-1) = 0
    // - an("&",pBetaF)("&",nfact,1,pBetaF.size()-1) = 0
    //
    // Notes:
    //
    // - When calling rank-one, bn is assumed to be zero.
    // - Because the ordering of variables in pAlphaF and pBetaF may change
    //   we include arguments apos and bpos.  Basically if point apos in
    //   pAlphaF moves then the function will change apos to the position
    //   to which it has moved.  Ditto bpos and pBetaF.
    // - bnZero = 1 tells the code to assume that bn is zero, bnZero = 0
    //              tells it to assume that bn is nonzero.
    // - bpNZ == -2 tells the code to assume that all of bp is nonzero.
    //   bpNZ == -1 tells the code to assume that all of bp is zero
    //   bpNZ >= 0  tells the code to assume that bp(pAlphaF)(0,1,fact.npos()-fact.nbadpos()-1)
    //              is zero except for one element, namely bp(pAlphaF)(bpNZ).
    // - bnNZ == -2 tells the code to assume that all of bn is nonzero.
    //   bnNZ == -1 tells the code to assume that all of bn is zero
    //   bnNZ >= 0  tells the code to assume that bn(pBetaF)(0,1,nfact-1)
    //              is zero except for one element, namely bn(pBetaF)(bnNZ).
    // - rankone and diagmult can both be called even if there is no factorisation.
    // - fact_minverse returns pfact+nfact (ie the size of the inverted hessian).
    // - There is a special case here.  If Gp == Gn == 0 then no factorisation is
    //   possible, but nevertheless if Gpn is nonzero then it may be possible to
    //   invert part of all of the active hessian in the form:
    //
    //    inv([  0      Gpnx  ]) = [  0          inv(Gpnx)'  ]
    //        [  Gpnx'  0     ]    [  inv(Gpnx)  0           ]
    //
    //    where Gpnx here represents the largest part of Gpn that is (a) square and
    //    (b) invertible.  In this case pfact() and nfact() will both return the
    //    size of Gpnx and fact_nofact returns true.

    void fact_rankone   (const Vector<double> &bp, const Vector<double> &bn, double c, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( bp.size() == aN() );
        NiceAssert( bn.size() == bN() );
        NiceAssert( dkeepfact );

        // No point doing this before the update as we'll need to complete
        // restart afterwards.
        //
        //if ( dkeepfact && betaFixUpdate )
        //{
        //    fixbetaFix(Gn,Gpn);
        //}

        retVector<double> tmpva;
        retVector<double> tmpvc;

        retMatrix<double> tmpmGp;
        retMatrix<double> tmpmGn;
        retMatrix<double> tmpmGpn;

        freeVarChol.rankone(bp(pAlphaF,tmpva),bn(pBetaF,0,1,dnfact-1,tmpvc),c,Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1),0,0,0,dnfact);
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

        if ( dkeepfact )
        {
            betaFix = -1;
            betaFixUpdate = 1;
            fixbetaFix(Gn,Gpn);
        }

        int aposdummy = 0;
        int bposdummy = 0;

        fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

        return;
    }

    void fact_diagmult  (const Vector<double> &bp, const Vector<double> &bn,                                                                                          int &apos, int &bpos)
    {
        (void) apos;
        (void) bpos;

        NiceAssert( bp.size() == aN() );
        NiceAssert( bn.size() == bN() );
        NiceAssert( dkeepfact );

        retVector<double> tmpva;
        retVector<double> tmpvc;

        freeVarChol.diagmult(bp(pAlphaF,tmpva),bn(pBetaF,0,1,dnfact-1,tmpvc));

        // Note that multiplying elements symmetrically by +-1 on the off-
        // diagonals makes no difference to the factorisability of a matrix, so
        // there is no need to do anything here.

        return;
        //apos = bpos;
    }

    void fact_diagoffset(const Vector<double> &bp, const Vector<double> &bn,           const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
    {
        //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gp.numRows() == Gpn.numRows() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( bp.size() == aN() );
        NiceAssert( bn.size() == bN() );
        NiceAssert( dkeepfact );

        retVector<double> tmpva;
        retVector<double> tmpvc;

        retMatrix<double> tmpmGp;
        retMatrix<double> tmpmGn;
        retMatrix<double> tmpmGpn;

        freeVarChol.diagoffset(bp(pAlphaF,tmpva),bn(pBetaF,0,1,dnfact-1,tmpvc),Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1),0,0,0,dnfact);
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

        int aposdummy = 0;
        int bposdummy = 0;

        fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

        return;
    }

    template <class S> int fact_minverse   (Vector<S> &ap, Vector<S> &an, const Vector<S> &bp, const Vector<S> &bn,                           const Matrix<double> &Gn, const Matrix<double> &Gpn, int bpNZ = -2, int bnNZ = -2) const;
    template <class S> int fact_near_invert(Vector<S> &ap, Vector<S> &an,                                           const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn                              ) const;
    template <class S> int fact_minvdiagsq (Vector<S> &ar, Vector<S> &br) const;

    double fact_det   (void) const { double res = freeVarChol.det();    return res*res; }
    double fact_logdet(void) const { double res = freeVarChol.logdet(); return 2*res;   }

    // Factorisation information functions
    //
    // fact_nfact:  returns the number of Gn row/columns in the factorised (or
    //              invertible if Gp == Gn == 0) part
    // fact_pfact:  returns the number of Gp row/columns in the factorised (or
    //              invertible if Gp == Gn == 0) part
    // fact_nofact: returns true if Gp == Gn == 0 (nonzero size).

    int fact_nfact (const Matrix<double> &Gn, const Matrix<double> &Gpn) const
    {
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( dkeepfact );

        int retval = dnfact;

        if ( fact_nofact(Gn,Gpn) )
        {
            retval = fact_pfact(Gn,Gpn);
        }

        return retval;
    }

    int fact_pfact (const Matrix<double> &Gn, const Matrix<double> &Gpn) const
    {
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( dkeepfact );

        int retval = dpfact;

        if ( fact_nofact(Gn,Gpn) )
        {
            int maxsize = ( ( aNF() < bNF() ) ? aNF() : bNF() );
	    int nonsingsize = 1;

            retMatrix<double> tmpmGp;
            retMatrix<double> tmpmGn;
            retMatrix<double> tmpmGpn;

            while ( ( (Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,( ( nonsingsize < maxsize ) ? nonsingsize : maxsize )-1,0,1,( ( nonsingsize < maxsize ) ? nonsingsize : maxsize )-1)).det() > dzt ) && ( nonsingsize <= maxsize ) )
	    {
	        ++nonsingsize;
	    }

            retval = nonsingsize-1;
        }

        return retval;
    }

    int fact_nofact(const Matrix<double> &Gn, const Matrix<double> &Gpn) const
    {
        NiceAssert( dkeepfact );

        int i;

        if ( betaFixUpdate )
        {
            fixbetaFix(Gn,Gpn);
        }

        int bNFgood = 0;

	for ( i = 0 ; i < bNF() ; ++i )
	{
	    if ( !(betaFix.v(i)) )
	    {
		++bNFgood;
                break; // we have the answer, so get out, no need to waste time.
	    }
	}

        return ( !(freeVarChol.ngood()) && aNF() && bNFgood );
    }

    // Factorisation control functions
    //
    // fudgeOn:  turns on  fudging (diagonal offsetting) to ensure full factorisation
    // fudgeOff: turns off fudging (diagonal offsetting) to ensure full factorisation
    //
    // NB: fudging doesn't work very well - appears to be numerically unstable

    void fact_fudgeOn (const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void fact_fudgeOff(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);

    // Test factoriation
    //
    // fact_testFact reconstructs the factorisation in dest matrices and returns the
    // maximum absolute difference between an element of Gp, Gn and Gpn and the
    // reconstructed version.

    double fact_testFact(Matrix<double> &Gpdest, Matrix<double> &Gndest, Matrix<double> &Gpndest, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const;

private:

    // Pivotting variables

    Vector<int> pAlphaLB;
    Vector<int> pAlphaZ;
    Vector<int> pAlphaUB;
    Vector<int> pAlphaF;

    Vector<int> pBetaC;
    Vector<int> pBetaF;

    Vector<int> dalphaState;
    Vector<int> dbetaState;

    // Miscellaneous variables
    //
    // dzt       = zero tolerance
    // dkeepfact = true if factorisation kept
    //
    // daNLF = number of alphas in LF
    // daNUF = number of alphas in UF

    double dzt;
    int dkeepfact;

    int daNLF;
    int daNUF;

    // Factorisation related stuff
    // ===========================
    //
    // GpnFColNorm(i) = sum_jP Gpn(pivAlphaF(jP),i)*Gpn(pivAlphaF(jP),i) + sum_jP Gn(pBetaF(jP),i)*Gn(pBetaF(jP),i)
    //
    // which is updated incrementally.  If the norm is below the threshold
    // GPNCOLNORMMIN then the column is considered to be effectively zero
    // and hence we can set betaFix(i) in the context.  Otherwise betaFix(i)
    // is zero and the column may be included in the factorisation.
    //
    // betaFixUpdate = 0: betaFix and GpnFColNorm are both up-to-date
    //               = 1: those betaFix elements set to -1 need to be updated.

    mutable Vector<double> GpnFColNorm;
    mutable Vector<int> betaFix;
    mutable int betaFixUpdate;

    // State variables
    //
    // nfact: the total number of elements in pBetaF included in the
    //        factorisation.  Note that elements of pBetaF should not be
    //        included in the singular part of the factorisation.
    // pfact: the total number of elements in pAlphaF included in the
    //        non-singular part of the factorisation.

    int dnfact;
    int dpfact;
    //int factsize;

    // Diagonal +-1 vector for LDL' cholesky style factorisation

    Vector<double> D;

    // The Cholesky Factorisation, suitably modified.

    Chol<double> freeVarChol;

    // Position in factorisation
    //
    // fAlphaF(i) is the position of alpha(pAlphaF(i)) in the factorisation
    // fBetaF(i)  is the position of beta(pBetaF(i))   in the non-singular part of the factorisation (-1 otherwise)

    Vector<int> fAlphaF;
    Vector<int> fBetaF;

    // Factorisation upkeep functions
    //
    // extendFactAlpha: a new alpha has been added, namely the final element in pAlphaF.
    //                  Extend the factorisation if possible.  By default leave at end of
    //                  factorisation, unless Gp(pAlphaF,pAlphaF) = 0 and Gn(pBetaF,pBetaF) = 0
    //                  prior to the extension, in which case we add it to the start.  The
    //                  returned value is where the alpha ends up in pAlphaF, so either
    //                  pAlphaF.size()-1 or zero.
    // extendFactBeta:  a new beta has been added, namely the final element in pBetaF.
    //                  Extend the factorisation if possible.  The returned value is where
    //                  the beta ends up in pBetaF.
    // shrinkFactAlpha: alpha element i in position fAlphaF(ipos) in the factorisation has been
    //                  removed.  Fix the factorisation where possible.
    // shrinkFactBeta:  beta element i in position fBetaF(ipos) in the factorisation has been
    //                  removed.  Fix the factorisation where possible.
    //
    // fixfact:    attempt to maximise the size of the factorisation.  This is achieved by changing
    //             the interleaving of alpha and beta components and also by changing the pivots
    //             pAlphaF and pBetaF.  pAlphaF will only be changed if none freeVarChol.ngood() == 0
    //             and either numNZGpDiag > 0 or numNZGnDiag > 0, and then only by squareswapping
    //             to elements in pAlphaF.  pBetaF may be changed arbitrarily.
    //             The arguments apos and bpos are used as follows: if the element pAlphaF(apos) is
    //             swapped then apos will be changed to the new position, and likewise if pBetaF(bpos)
    //             is moved then bpos will be changed to the new position.
    // fixbetaFix: fix betaFix and GpnFColNorm

    int extendFactAlpha(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int extendFactBeta(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void shrinkFactAlpha(int i, int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void shrinkFactBeta(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);

    void fixfact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos, int &aposalt, int &bposalt)
    {
      //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
      NiceAssert( Gn.isSquare() );
      NiceAssert( Gp.numRows() == Gpn.numRows() );
      NiceAssert( Gn.numCols() == Gpn.numCols() );
      NiceAssert( Gpn.numRows() == aN() );
      NiceAssert( Gpn.numCols() == bN() );
      NiceAssert( dkeepfact );

      if ( dkeepfact && (betaFixUpdate) )
      {
        fixbetaFix(Gn,Gpn);
      }

      int iP,jP,fpos;

      // Clean up the factorisation by removing any betas in the singular
      // part of the factorisation.

      retMatrix<double> tmpmGp;
      retMatrix<double> tmpmGn;
      retMatrix<double> tmpmGpn;

      if ( fBetaF.size() )
      {
	while ( ( fpos = max(fBetaF,iP) ) >= freeVarChol.ngood() )
	{
	    if ( iP != (pBetaF.size())-1 )
	    {
		if ( bpos == iP )
		{
		    bpos = pBetaF.size()-1;
		}

		else if ( bpos == pBetaF.size()-1 )
		{
		    bpos = iP;
		}

		if ( bposalt == iP )
		{
		    bposalt = pBetaF.size()-1;
		}

		else if ( bposalt == pBetaF.size()-1 )
		{
		    bposalt = iP;
		}

		pBetaF.squareswap(iP,pBetaF.size()-1);
		fBetaF.squareswap(iP,pBetaF.size()-1);
	    }

	    //if ( fAlphaF.size() )
	    {
		for ( jP = 0 ; jP < fAlphaF.size() ; ++jP )
		{
		    if ( fAlphaF.v(jP) > fpos )
		    {
			--(fAlphaF("&",jP));
		    }
		}
	    }

	    fBetaF.sv(pBetaF.size()-1,-1);
	    --dnfact;

	    D.remove(fpos);
	    freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
	    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());
	}
      }

      NiceAssert( (freeVarChol.ngood()) >= dnfact );

      int outerdone = 0;
      int innerdone = 0;
      int newpos;
      int oldpos;

      retVector<int> tmpva;

      while ( !outerdone )
      {
	outerdone = 1;

	// add as many elements from pBetaF as possible

	innerdone = 0;

	while ( !innerdone && ( dnfact < pBetaF.size() ) )
	{
            innerdone = 1;

	    for ( iP = dnfact ; iP < pBetaF.size() ; ++iP )
	    {
		if ( !(betaFix.v(pivBetaF(iP))) )
		{
		    if ( iP > dnfact )
		    {
			if ( bpos == iP )
			{
			    bpos = dnfact;
			}

			else if ( bpos == dnfact )
			{
			    bpos = iP;
			}

			if ( bposalt == iP )
			{
			    bposalt = dnfact;
			}

			else if ( bposalt == dnfact )
			{
			    bposalt = iP;
			}

			pBetaF.squareswap(iP,dnfact);
			fBetaF.squareswap(iP,dnfact);
		    }

		    fBetaF.sv(dnfact,freeVarChol.ngood());
		    ++dnfact;

		    D.add(freeVarChol.ngood());
		    D.sv(freeVarChol.ngood(),-1);
		    freeVarChol.add(freeVarChol.ngood(),D.v(freeVarChol.ngood()),Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
		    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

		    if ( freeVarChol.ngood() > fBetaF.v(dnfact-1) )
		    {
			innerdone = 0;
                        outerdone = 0;

			//if ( fAlphaF.size() )
			{
			    for ( jP = 0 ; jP < fAlphaF.size() ; ++jP )
			    {
				if ( fAlphaF.v(jP) >= fBetaF.v(dnfact-1) )
				{
				    ++(fAlphaF("&",jP));
				}
			    }
			}

			break;
		    }

		    else
		    {
                        NiceAssert( freeVarChol.ngood() == fBetaF(dnfact-1) );

			--dnfact;
			fBetaF.sv(dnfact,-1);

			D.remove(freeVarChol.ngood());
			freeVarChol.remove(freeVarChol.ngood(),Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
			dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

			if ( iP > dnfact )
			{
			    if ( bpos == iP )
			    {
				bpos = dnfact;
			    }

			    else if ( bpos == dnfact )
			    {
				bpos = iP;
			    }

			    if ( bposalt == iP )
			    {
				bposalt = dnfact;
			    }

			    else if ( bposalt == dnfact )
			    {
				bposalt = iP;
			    }

			    pBetaF.squareswap(iP,dnfact);
			    fBetaF.squareswap(iP,dnfact);
			}
		    }
		}
	    }
	}

        // add as many elements from pAlphaF as possible

	innerdone = 0;

	while ( !innerdone && ( dpfact+1 < pAlphaF.size() ) )
	{
            innerdone = 1;

	    for ( iP = dpfact+1 ; iP < pAlphaF.size() ; ++iP )
	    {
                oldpos = fAlphaF.v(iP);
		newpos = freeVarChol.ngood();

		if ( apos == iP )
		{
                    apos = (pAlphaF.size())-1;
		}

		else if ( apos > iP )
		{
                    --apos;
		}

		if ( aposalt == iP )
		{
                    aposalt = (pAlphaF.size())-1;
		}

		else if ( aposalt > iP )
		{
                    --aposalt;
		}

		pAlphaF.blockswap(iP,(pAlphaF.size())-1);
		fAlphaF.blockswap(iP,(fAlphaF.size())-1);

		fAlphaF("&",iP,1,(fAlphaF.size())-2,tmpva) -= 1;
		fAlphaF.sv((fAlphaF.size())-1,(D.size())-1);

		D.remove(oldpos);
                freeVarChol.remove(oldpos,Gp(pAlphaF,pAlphaF,tmpmGp,0,1,(pAlphaF.size())-2,0,1,(pAlphaF.size())-2),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-2,0,1,dnfact-1));

		if ( apos == (pAlphaF.size())-1 )
		{
                    apos = dpfact;
		}

		else if ( apos >= dpfact )
		{
                    ++apos;
		}

		if ( aposalt == (pAlphaF.size())-1 )
		{
                    aposalt = dpfact;
		}

		else if ( aposalt >= dpfact )
		{
                    ++aposalt;
		}

		pAlphaF.blockswap((pAlphaF.size())-1,dpfact);
		fAlphaF.blockswap((fAlphaF.size())-1,dpfact);

		fAlphaF.sv(dpfact,newpos);
                fAlphaF("&",dpfact+1,1,(fAlphaF.size())-1,tmpva) += 1;

		D.add(newpos);
		D.sv(newpos,+1);
		freeVarChol.add(newpos,D.v(newpos),Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));

		if ( freeVarChol.ngood() > newpos )
		{
		    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

		    innerdone = 0;
		    outerdone = 0;

		    break;
		}

		else
		{
                    NiceAssert( freeVarChol.ngood() == newpos );

		    if ( apos == dpfact )
		    {
			apos = (pAlphaF.size())-1;
		    }

		    else if ( apos > dpfact )
		    {
			--apos;
		    }

		    if ( aposalt == dpfact )
		    {
			aposalt = (pAlphaF.size())-1;
		    }

		    else if ( aposalt > dpfact )
		    {
			--aposalt;
		    }

		    pAlphaF.blockswap(dpfact,(pAlphaF.size())-1);
		    fAlphaF.blockswap(dpfact,(fAlphaF.size())-1);

		    fAlphaF("&",dpfact,1,(fAlphaF.size())-2,tmpva) -= 1;
		    fAlphaF.sv((fAlphaF.size())-1,(D.size())-1);

		    D.remove(newpos);
		    freeVarChol.remove(newpos,Gp(pAlphaF,pAlphaF,tmpmGp,0,1,(pAlphaF.size())-2,0,1,(pAlphaF.size())-2),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-2,0,1,dnfact-1));

		    if ( apos == (pAlphaF.size())-1 )
		    {
			apos = iP;
		    }

		    else if ( apos >= iP )
		    {
			++apos;
		    }

		    if ( aposalt == (pAlphaF.size())-1 )
		    {
			aposalt = iP;
		    }

		    else if ( aposalt >= iP )
		    {
			++aposalt;
		    }

		    pAlphaF.blockswap((pAlphaF.size())-1,iP);
		    fAlphaF.blockswap((fAlphaF.size())-1,iP);

		    fAlphaF.sv(iP,oldpos);
		    fAlphaF("&",iP+1,1,(fAlphaF.size())-1,tmpva) += 1;

		    D.add(oldpos);
		    D.sv(oldpos,+1);
		    freeVarChol.add(oldpos,D.v(oldpos),Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));

                    NiceAssert( dpfact == (freeVarChol.npos())-(freeVarChol.nbadpos()) );
		}
	    }
	}
      }

      return;
    }

    void fixbetaFix(const Matrix<double> &Gn, const Matrix<double> &Gpn) const // not really const, changes mutable stuff only
    {
        NiceAssert( Gn.isSquare() );
        NiceAssert( Gn.numCols() == Gpn.numCols() );
        NiceAssert( Gpn.numRows() == aN() );
        NiceAssert( Gpn.numCols() == bN() );
        NiceAssert( dkeepfact );

        int i,jP;

        //if ( betaFixUpdate && bN() )
        if ( (betaFixUpdate) )
        {
            for ( i = 0 ; i < bN() ; ++i )
            {
                if ( betaFix.v(i) == -1 )
                {
                    (GpnFColNorm).sv(i,0.0);

                    //if ( aNF() )
                    {
                        for ( jP = 0 ; jP < aNF() ; ++jP )
                        {
                            (GpnFColNorm)("&",i) += (Gpn.v(pivAlphaF(jP),i)*Gpn.v(pivAlphaF(jP),i));
                        }
                    }

                    (betaFix).sv(i,CALCBETAFIX(i));
                }
            }
        }

        betaFixUpdate = 0;

        return;
    }
};

inline void qswap(optContext &a, optContext &b)
{
    qswap(a.pAlphaLB     ,b.pAlphaLB     );
    qswap(a.pAlphaZ      ,b.pAlphaZ      );
    qswap(a.pAlphaUB     ,b.pAlphaUB     );
    qswap(a.pAlphaF      ,b.pAlphaF      );
    qswap(a.pBetaC       ,b.pBetaC       );
    qswap(a.pBetaF       ,b.pBetaF       );
    qswap(a.dalphaState  ,b.dalphaState  );
    qswap(a.dbetaState   ,b.dbetaState   );
    qswap(a.dzt          ,b.dzt          );
    qswap(a.dkeepfact    ,b.dkeepfact    );
    qswap(a.daNLF        ,b.daNLF        );
    qswap(a.daNUF        ,b.daNUF        );
    qswap(a.GpnFColNorm  ,b.GpnFColNorm  );
    qswap(a.betaFix      ,b.betaFix      );
    qswap(a.betaFixUpdate,b.betaFixUpdate);
    qswap(a.dnfact       ,b.dnfact       );
    qswap(a.dpfact       ,b.dpfact       );
    //qswap(a.factsize     ,b.factsize     );
    qswap(a.D            ,b.D            );
    qswap(a.freeVarChol  ,b.freeVarChol  );
    qswap(a.fAlphaF      ,b.fAlphaF      );
    qswap(a.fBetaF       ,b.fBetaF       );
}

template <class S>
int optContext::fact_minverse(Vector<S> &ap, Vector<S> &an, const Vector<S> &bp, const Vector<S> &bn, const Matrix<double> &Gn, const Matrix<double> &Gpn, int bpNZ, int bnNZ) const
{
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ap.size() == aN() );
    NiceAssert( an.size() == bN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( dkeepfact );

    int retval = 0;

    int zp_start = 0;
    int zn_start = 0;
    int zp_end = 0;
    int zn_end = 0;

    if ( bpNZ == -2 )
    {
	zp_start = 0;
        zp_end   = 0;
    }

    else if ( bpNZ == -1 )
    {
        zp_start = 0;
        zp_end   = dpfact;
    }

    else if ( bpNZ < dpfact )
    {
        zp_start = bpNZ;
	zp_end   = dpfact-bpNZ-1;
    }

    else
    {
	zp_start = 0;
        zp_end   = dpfact;
    }

    if ( bnNZ == -2 )
    {
	zn_start = 0;
        zn_end   = 0;
    }

    else if ( bnNZ == -1 )
    {
        zn_start = 0;
        zn_end   = dnfact;
    }

    else if ( bnNZ < dnfact )
    {
        zn_start = bnNZ;
	zn_end   = dnfact-bnNZ-1;
    }

    else
    {
	zn_start = 0;
        zn_end   = dnfact;
    }

    if ( freeVarChol.ngood() )
    {
        retVector<S> tmpvb;
        retVector<S> tmpvd;
        retVector<S> tmpvf;
        retVector<S> tmpvh;

	freeVarChol.minverse(ap("&",pAlphaF,0,1,dpfact-1,tmpvb),an("&",pBetaF,0,1,dnfact-1,tmpvd),bp(pAlphaF,0,1,dpfact-1,tmpvf),bn(pBetaF,0,1,dnfact-1,tmpvh),zp_start,zp_end,zn_start,zn_end);

        ap("&",pAlphaF,dpfact,1,(pAlphaF.size())-1,tmpvb).zero();
        an("&",pBetaF,dnfact,1,(pBetaF.size())-1,tmpvb).zero();

        retval = dpfact+dnfact;
    }

    else if ( pAlphaF.size() && pBetaF.size() )
    {
	int nonsingsize = fact_pfact(Gn,Gpn);

	if ( nonsingsize )
	{
            retVector<S> tmpvb;
            retVector<S> tmpvd;

            retMatrix<double> tmpmGpn;

	    Matrix<double> Gpninv((Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,nonsingsize-1,0,1,nonsingsize-1)).inve());

            ap("&",pAlphaF,0,1,nonsingsize-1,tmpvb) = bn(pBetaF,0,1,nonsingsize-1,tmpvd);
            an("&",pBetaF,0,1,nonsingsize-1,tmpvb)  = bp(pAlphaF,0,1,nonsingsize-1,tmpvd);

            ap("&",pAlphaF,0,1,nonsingsize-1,tmpvb) *= Gpninv;
	    rightmult(Gpninv,an("&",pBetaF,0,1,nonsingsize-1,tmpvb));
	}

        retVector<S> tmpvb;

        ap("&",pAlphaF,nonsingsize,1,(pAlphaF.size())-1,tmpvb).zero();
        an("&",pBetaF,nonsingsize,1,(pBetaF.size())-1,tmpvb).zero();

	retval = 2*nonsingsize;
    }

    return retval;
}

template <class S> int optContext::fact_minvdiagsq(Vector<S> &ar, Vector<S> &br) const
{
    NiceAssert( ar.size() == aN() );
    NiceAssert( br.size() == bN() );
    NiceAssert( dkeepfact );

    retVector<S> tmpvb;
    retVector<S> tmpvd;

    freeVarChol.minvdiagsq(ar("&",pAlphaF,0,1,dpfact-1,tmpvb),br("&",pBetaF,0,1,dnfact-1,tmpvd));

    ar("&",pAlphaF,dpfact,1,(pAlphaF.size())-1,tmpvb).zero();
    br("&",pBetaF,dnfact,1,(pBetaF.size())-1,tmpvb).zero();

    return dpfact+dnfact;
}

template <> inline int optContext::fact_near_invert(Vector<double> &ap, Vector<double> &an, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const;
template <> inline int optContext::fact_near_invert(Vector<double> &ap, Vector<double> &an, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ap.size() == aN() );
    NiceAssert( an.size() == bN() );
    NiceAssert( dkeepfact );

    int retval = 0;

    if ( freeVarChol.ngood() )
    {
        retVector<double> tmpvb;
        retVector<double> tmpvd;

	freeVarChol.near_invert(ap("&",pAlphaF,0,1,dpfact-1,tmpvb),an("&",pBetaF,0,1,dnfact-1,tmpvd));

	ap("&",pAlphaF,dpfact,1,(pAlphaF.size())-1,tmpvb) = 0.0;
	an("&",pBetaF,dnfact,1,(pBetaF.size())-1,tmpvb)   = 0.0;

        retval = dpfact+dnfact;
    }

    else if ( pAlphaF.size() && pBetaF.size() )
    {
	int nonsingsize = fact_pfact(Gn,Gpn);

        NiceAssert( ( nonsingsize < aNF() ) || ( nonsingsize < bNF() ) );

	if ( nonsingsize )
	{
	    if ( nonsingsize < aNF() )
	    {
		int i;

		//if ( nonsingsize )
		{
		    for ( i = 0 ; i < nonsingsize ; ++i )
		    {
			ap.sv(pAlphaF(i),Gpn.v(pAlphaF.v(nonsingsize),pBetaF.v(i)));
			an.sv(pBetaF(i), Gp.v(pAlphaF.v(nonsingsize),pAlphaF.v(i)));
		    }
		}
	    }

	    else
	    {
		Vector<double> bp(nonsingsize);

		int iP;

		for ( iP = 0 ; iP < nonsingsize ; ++iP )
		{
		    bp.sv(iP,Gpn(pAlphaF.v(pAlphaF.v(iP)),pBetaF.v(nonsingsize)));
		}

                retVector<double> tmpvb;
                retVector<double> tmpvc;
                retVector<double> tmpvd;

                retMatrix<double> tmpmGn;

		ap("&",pAlphaF,0,1,nonsingsize-1,tmpvb) = Gn(pBetaF,pBetaF,tmpmGn)(nonsingsize,0,1,nonsingsize-1,tmpvc,tmpvd);
		an("&",pBetaF,0,1,nonsingsize-1,tmpvb)  = bp;
	    }

            retVector<double> tmpvb;

            retMatrix<double> tmpmGpn;

	    Matrix<double> Gpninv((Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,nonsingsize-1,0,1,nonsingsize-1)).inve());

            ap("&",pAlphaF,0,1,nonsingsize-1,tmpvb) *= Gpninv;
	    rightmult(Gpninv,an("&",pBetaF,0,1,nonsingsize-1,tmpvb));
	}

        retVector<double> tmpvb;

        ap("&",pAlphaF,nonsingsize,1,(pAlphaF.size())-1,tmpvb) = 0.0;
        an("&",pBetaF,nonsingsize,1,(pBetaF.size())-1,tmpvb)   = 0.0;

	retval = 2*nonsingsize;
    }

    return retval;
}

template <class S> int optContext::fact_near_invert(Vector<S> &ap, Vector<S> &an, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ap.size() == aN() );
    NiceAssert( an.size() == bN() );
    NiceAssert( dkeepfact );

    int retval = 0;

    if ( freeVarChol.ngood() )
    {
        retVector<S> tmpvb;
        retVector<S> tmpvd;

	freeVarChol.near_invert(ap("&",pAlphaF,0,1,dpfact-1,tmpvb),an("&",pBetaF,0,1,dnfact-1,tmpvd));

	ap("&",pAlphaF,dpfact,1,(pAlphaF.size())-1,tmpvb) = 0.0;
	an("&",pBetaF,dnfact,1,(pBetaF.size())-1,tmpvb)   = 0.0;

        retval = dpfact+dnfact;
    }

    else if ( pAlphaF.size() && pBetaF.size() )
    {
	int nonsingsize = fact_pfact(Gn,Gpn);

        NiceAssert( ( nonsingsize < aNF() ) || ( nonsingsize < bNF() ) );

	if ( nonsingsize )
	{
	    if ( nonsingsize < aNF() )
	    {
		int i;

		//if ( nonsingsize )
		{
		    for ( i = 0 ; i < nonsingsize ; ++i )
		    {
			ap("&",pAlphaF(i)) = Gpn(pAlphaF(nonsingsize),pBetaF(i));
			an("&",pBetaF(i))  = Gp(pAlphaF(nonsingsize),pAlphaF(i));
		    }
		}
	    }

	    else
	    {
		Vector<S> bp(nonsingsize);

		int iP;

		for ( iP = 0 ; iP < nonsingsize ; ++iP )
		{
		    bp("&",iP) = Gpn(pAlphaF(pAlphaF(iP)),pBetaF(nonsingsize));
		}

                retVector<S> tmpvb;
                retVector<S> tmpvc;
                retVector<S> tmpvd;

                retMatrix<double> tmpmGn;

		ap("&",pAlphaF,0,1,nonsingsize-1,tmpvb) = Gn(pBetaF,pBetaF,tmpmGn)(nonsingsize,0,1,nonsingsize-1,tmpvc,tmpvd);
		an("&",pBetaF,0,1,nonsingsize-1,tmpvb)  = bp;
	    }

            retVector<S> tmpvb;

            retMatrix<double> tmpmGpn;

	    Matrix<double> Gpninv((Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,nonsingsize-1,0,1,nonsingsize-1)).inve());

            ap("&",pAlphaF,0,1,nonsingsize-1,tmpvb) *= Gpninv;
	    rightmult(Gpninv,an("&",pBetaF,0,1,nonsingsize-1,tmpvb));
	}

        retVector<S> tmpvb;

        ap("&",pAlphaF,nonsingsize,1,(pAlphaF.size())-1,tmpvb) = 0.0;
        an("&",pBetaF,nonsingsize,1,(pBetaF.size())-1,tmpvb)   = 0.0;

	retval = 2*nonsingsize;
    }

    return retval;
}

#endif
