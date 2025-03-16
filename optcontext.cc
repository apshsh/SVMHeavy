
//
// Quadratic optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "optcontext.hpp"

void optContext::refact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int xkeepfact, double xzt)
{
    //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ( xzt > 0 ) || ( xzt == -1 ) );
    NiceAssert( ( xkeepfact == 0 ) || ( xkeepfact == 1 ) || ( xkeepfact == -1 ) );

    if ( dkeepfact && (betaFixUpdate) )
    {
        fixbetaFix(Gn,Gpn);
    }

    int oldaN = aN();
    int oldbN = bN();
    int oldkeepfact = dkeepfact;

    // Keep current values for keepfact and zt if new values set to -1 ("keep")

    if ( xkeepfact == -1 )
    {
        xkeepfact = dkeepfact;
    }

    if ( xzt < 0 )
    {
        xzt = dzt;
    }

    // Set new values for zt and keepfact

    dzt       = xzt;
    dkeepfact = xkeepfact;

    // Save current pivotting and state

    Vector<int> spAlphaLB(pAlphaLB);
    Vector<int> spAlphaUB(pAlphaUB);
    Vector<int> spAlphaF(pAlphaF);

    Vector<int> spBetaC(pBetaC);
    Vector<int> spBetaF(pBetaF);

    Vector<int> salphaState(dalphaState);

    // Empty pivotting and state.

    pAlphaLB.resize(0);
    pAlphaZ.resize(0);
    pAlphaUB.resize(0);
    pAlphaF.resize(0);

    pBetaC.resize(0);
    pBetaF.resize(0);

    dalphaState.resize(0);
    dbetaState.resize(0);

    // Reset counts of NLF and NUF

    daNLF = 0;
    daNUF = 0;

    // Reset Gpn column totals and beta fixing vector.

    if ( oldkeepfact || dkeepfact )
    {
	(GpnFColNorm).resize(0);
	(betaFix).resize(0);
    }

    // Reset factorisation counts.

    dnfact = 0;
    dpfact = 0;

    // Reset factorisation

    if ( oldkeepfact || dkeepfact )
    {
	D.resize(0);

        Chol<double> temp(xzt,freeVarChol.fudge());
	freeVarChol = temp;

	fAlphaF.resize(0);
	fBetaF.resize(0);
    }

    int i,iP;

    // Systematically re-add variables

    for ( i = 0 ; i < oldbN ; ++i )
    {
        addBeta(i);
    }

    for ( i = 0 ; i < oldaN ; ++i )
    {
        addAlpha(i);
    }

    // Restore pivotting and state

    for ( iP = 0 ; iP < spAlphaLB.size() ; ++iP )
    {
        modAlphaZtoLB(findInAlphaZ(spAlphaLB.v(iP)));
    }

    for ( iP = 0 ; iP < spAlphaUB.size() ; ++iP )
    {
        modAlphaZtoUB(findInAlphaZ(spAlphaUB.v(iP)));
    }

    for ( iP = 0 ; iP < spAlphaF.size() ; ++iP )
    {
        if ( salphaState.v(spAlphaF.v(iP)) < 0 )
        {
            int aposdummy = 0;
            int bposdummy = 0;

            modAlphaZtoLF(findInAlphaZ(spAlphaF.v(iP)),Gp,Gn,Gpn,aposdummy,bposdummy);
        }

        else
        {
            int aposdummy = 0;
            int bposdummy = 0;

            modAlphaZtoUF(findInAlphaZ(spAlphaF.v(iP)),Gp,Gn,Gpn,aposdummy,bposdummy);
        }
    }

    for ( iP = 0 ; iP < spBetaF.size() ; ++iP )
    {
        int aposdummy = 0;
        int bposdummy = 0;

        modBetaCtoF(findInBetaC(spBetaF.v(iP)),Gp,Gn,Gpn,aposdummy,bposdummy);
    }

    return;
}

void optContext::reset(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn)
{
    //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );

    int apos = 0;
    int bpos = 0;

    while ( aNLB() )
    {
	modAlphaLBtoZ(aNLB()-1); // take from end for computational efficiency
    }

    while ( aNUB() )
    {
	modAlphaUBtoZ(aNUB()-1);
    }

    while ( aNF() )
    {
	if ( alphaState(pivAlphaF(aNF()-1)) == -1 )
	{
	    modAlphaLFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}

	else
	{
	    modAlphaUFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}
    }

    while ( bNF() )
    {
	modBetaFtoC(bNF()-1,Gp,Gn,Gpn,apos,bpos);
    }

    return;
}

int optContext::addAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= aN() );

    int j;

    // Fix all pivots to reflect pending increase in number of alphas

    //if ( aNLB() )
    {
	for ( j = 0 ; j < aNLB() ; ++j )
	{
	    if ( pivAlphaLB(j) >= i )
	    {
                ++(pAlphaLB("&",j));
	    }
	}
    }

    //if ( aNZ() )
    {
	for ( j = 0 ; j < aNZ() ; ++j )
	{
	    if ( pivAlphaZ(j) >= i )
	    {
                ++(pAlphaZ("&",j));
	    }
	}
    }

    //if ( aNUB() )
    {
	for ( j = 0 ; j < aNUB() ; ++j )
	{
	    if ( pivAlphaUB(j) >= i )
	    {
                ++(pAlphaUB("&",j));
	    }
	}
    }

    //if ( aNF() )
    {
	for ( j = 0 ; j < aNF() ; ++j )
	{
	    if ( pivAlphaF(j) >= i )
	    {
                ++(pAlphaF("&",j));
	    }
	}
    }

    // Add to zero pivot

    pAlphaZ.add(pAlphaZ.size());
    pAlphaZ("&",pAlphaZ.size()-1) = i;
    dalphaState.add(i);
    dalphaState("&",i) = 0;

    return (pAlphaZ.size())-1;
}

int optContext::addBeta(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= bN() );

    int j;

    // Fix all pivots to reflect pending increase in number of betas

    //if ( bNC() )
    {
	for ( j = 0 ; j < bNC() ; ++j )
	{
	    if ( pivBetaC(j) >= i )
	    {
                ++(pBetaC("&",j));
	    }
	}
    }

    //if ( bNF() )
    {
	for ( j = 0 ; j < bNF() ; ++j )
	{
	    if ( pivBetaF(j) >= i )
	    {
                ++(pBetaF("&",j));
	    }
	}
    }

    // Add to constrained pivot

    pBetaC.add(pBetaC.size());
    pBetaC("&",pBetaC.size()-1) = i;
    dbetaState.add(i);
    dbetaState("&",i) = 0;

    // Update Gpn column norms and betafix

    if ( dkeepfact )
    {
	(betaFixUpdate) = 1;

	(GpnFColNorm).add(i);
	(GpnFColNorm)("&",i) = 0.0;
	(betaFix).add(i);
	(betaFix)("&",i) = -1;
    }

    return (pBetaC.size())-1;
}

int optContext::removeAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );

    int iP = -1;
    int j;

    // Find i in pivot vector

    //if ( aNZ() )
    {
	for ( j = 0 ; j < aNZ() ; ++j )
	{
	    if ( pivAlphaZ(j) == i )
	    {
		iP = j;
	    }

	    else if ( pivAlphaZ(j) > i )
	    {
                --(pAlphaZ("&",j));
	    }
	}
    }

    // Sanity check

    NiceAssert( iP >= 0 );

    // Remove i from pivot vector

    pAlphaZ.remove(iP);
    dalphaState.remove(i);

    // Fix all pivots to reflect decrease in number of alphas

    //if ( aNLB() )
    {
	for ( j = 0 ; j < aNLB() ; ++j )
	{
	    if ( pivAlphaLB(j) > i )
	    {
                --(pAlphaLB("&",j));
	    }
	}
    }

    //if ( aNF() )
    {
	for ( j = 0 ; j < aNF() ; ++j )
	{
	    if ( pivAlphaF(j) > i )
	    {
                --(pAlphaF("&",j));
	    }
	}
    }

    //if ( aNUB() )
    {
	for ( j = 0 ; j < aNUB() ; ++j )
	{
	    if ( pivAlphaUB(j) > i )
	    {
                --(pAlphaUB("&",j));
	    }
	}
    }

    return iP;
}

int optContext::removeBeta(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );

    int iP = -1;
    int j;

    // Find i in pivot vector

    //if ( bNC() )
    {
	for ( j = 0 ; j < bNC() ; ++j )
	{
	    if ( pivBetaC(j) == i )
	    {
		iP = j;
	    }

	    else if ( pivBetaC(j) > i )
	    {
                --(pBetaC("&",j));
	    }
	}
    }

    // Sanity check

    NiceAssert( iP >= 0 );

    // Remove i from pivot vector

    pBetaC.remove(iP);
    dbetaState.remove(i);

    // Fix all pivots to reflect decrease in number of betas

    //if ( bNF() )
    {
	for ( j = 0 ; j < bNF() ; ++j )
	{
	    if ( pivBetaF(j) > i )
	    {
                --(pBetaF("&",j));
	    }
	}
    }

    if ( dkeepfact )
    {
	(GpnFColNorm).remove(i);
	(betaFix).remove(i);
    }

    return iP;
}














double optContext::fact_testFact(Matrix<double> &Gpdest, Matrix<double> &Gndest, Matrix<double> &Gpndest, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    double res = 0;

    Gpdest = Gp;
    Gndest = Gn;
    Gpndest = Gpn;

    Vector<double> Ddest(D);

    retMatrix<double> tmpmGp;
    retMatrix<double> tmpmGn;
    retMatrix<double> tmpmGpn;

    freeVarChol.testFact(Gpdest("&",pAlphaF,pAlphaF,tmpmGp),Gndest("&",pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpndest("&",pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1),Ddest);

    //if ( aN() )
    {
	for ( int i = 0 ; i < aN() ; ++i )
	{
	    for ( int j = 0 ; j < aN() ; ++j )
	    {
		if ( abs2(Gp.v(i,j)-Gpdest.v(i,j)) > res )
		{
                    res = abs2(Gp.v(i,j)-Gpdest.v(i,j));
		}
	    }
	}
    }

    //if ( bN() )
    {
	for ( int i = 0 ; i < bN() ; ++i )
	{
	    for ( int j = 0 ; j < bN() ; ++j )
	    {
		if ( abs2(Gn.v(i,j)-Gndest.v(i,j)) > res )
		{
                    res = abs2(Gn.v(i,j)-Gndest.v(i,j));
		}
	    }
	}
    }

    //if ( aN() && bN() )
    if ( bN() )
    {
	for ( int i = 0 ; i < aN() ; ++i )
	{
	    for ( int j = 0 ; j < bN() ; ++j )
	    {
		if ( abs2(Gpn.v(i,j)-Gpndest.v(i,j)) > res )
		{
                    res = abs2(Gpn.v(i,j)-Gpndest.v(i,j));
		}
	    }
	}
    }

    return res;
}

void optContext::fact_fudgeOn(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    // No point doing this before the update as we'll need to complete
    // restart afterwards.
    //
    //if ( dkeepfact && betaFixUpdate )
    //{
    //    fixbetaFix(Gn,Gpn);
    //}

    freeVarChol.fudgeOn();
    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

    if ( dkeepfact )
    {
        (betaFix) = -1;
        (betaFixUpdate) = 1;
        fixbetaFix(Gn,Gpn);
    }

    int aposdummy = 0;
    int bposdummy = 0;

    fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

    return;
}

void optContext::fact_fudgeOff(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    //comment out to enable threading in errortest NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    // No point doing this before the update as we'll need to complete
    // restart afterwards.
    //
    //if ( dkeepfact && betaFixUpdate )
    //{
    //    fixbetaFix(Gn,Gpn);
    //}

    freeVarChol.fudgeOff();
    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

    if ( dkeepfact )
    {
        (betaFix) = -1;
        (betaFixUpdate) = 1;
        fixbetaFix(Gn,Gpn);
    }

    int aposdummy = 0;
    int bposdummy = 0;

    fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

    return;
}

int optContext::extendFactAlpha(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
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

    int retval = (pAlphaF.size())-1;
    int fpos = D.size();
    int i = pivAlphaF(retval);
    int j;
    int addpos = 0;
    int betafixchange = 0;
    int betaunfixedextend = 0;
    int oldbetafix;

    fAlphaF.add(fAlphaF.size());
    fAlphaF("&",fAlphaF.size()-1) = fpos;

    // Update betaFix and set betafixchange if any change occurs.
    // betaFix[i] = 0 if the norm of elements in column i exceeds
    // a given threshold (and can therefore be included, potentially,
    // in the factorisation), zero otherwise.

    //if ( betaFix.size() )
    {
	for ( j = 0 ; j < (betaFix).size() ; ++j )
	{
	    (GpnFColNorm)("&",j) += (Gpn.v(i,j)*Gpn.v(i,j));

            oldbetafix = betaFix.v(j);

	    (betaFix)("&",j) = CALCBETAFIX(j);

	    if ( oldbetafix != betaFix.v(j) )
	    {
                betafixchange = 1;
	    }
	}
    }

    // Run through those columns of Gpn and row/columns of Gn not
    // currently included in the factorisation.  If any have betaFix[j]
    // non-zero then set betaunfixedextend = 1 to indicate that we
    // should try to add these to the factorisation.

    //if ( dnfact < pBetaF.size() )
    {
	for ( j = dnfact ; j < pBetaF.size() ; ++j )
	{
	    if ( !(betaFix.v(j)) )
	    {
                betaunfixedextend = 1;
	    }
	}
    }

    // Set flag if new diagonal in Hessian is non-zero to within zerotol

    if ( Gp.v(retval,retval) > dzt )
    {
        addpos = 1;
    }

    // If factoristion non-singular,
    // or if factorisation is completely singular and the new diagonal on
    //   Gp is non-zero,
    // or if any betas that had corresponded to zero columns in Gpn now
    //   correspond to non-zero columns,
    // or there are any parts in the segment Gn not currently included
    //   in the factorisation that correspond to non-zero columns in Gpn
    // then add the row/column to factorisation and try to extend it.
    //
    // Otherwise we know that there is no way to extend the factorisation,
    // so don't even try, just add to the end and leave.

    retMatrix<double> tmpmGp;
    retMatrix<double> tmpmGn;
    retMatrix<double> tmpmGpn;

    if ( !(freeVarChol.nbad()) || ( !(freeVarChol.ngood()) && addpos ) || betafixchange || betaunfixedextend )
    {
	D.add(fpos);
	D("&",fpos) = +1;
	freeVarChol.add(fpos,D.v(fpos),Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

        int bposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,retval,bposdummy);
    }

    else
    {
	D.add(fpos);
	D("&",fpos) = +1;
	freeVarChol.add(fpos,D.v(fpos),Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());
    }

    return retval;
}

void optContext::shrinkFactAlpha(int i, int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
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

    int j,jP;
    int fpos = fAlphaF.v(iP);

    fAlphaF.remove(iP);

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

    //if ( fBetaF.size() )
    {
	for ( jP = 0 ; jP < fBetaF.size() ; ++jP )
	{
	    if ( fBetaF.v(jP) > fpos )
	    {
                --(fBetaF("&",jP));
	    }
	}
    }

    //if ( betaFix.size() )
    {
	for ( j = 0 ; j < (betaFix).size() ; ++j )
	{
	    (GpnFColNorm)("&",j) -= (Gpn.v(i,j)*Gpn.v(i,j));

	    (betaFix)("&",j) = CALCBETAFIX(j);
	}
    }

    retMatrix<double> tmpmGp;
    retMatrix<double> tmpmGn;
    retMatrix<double> tmpmGpn;

    if ( fpos > freeVarChol.ngood() )
    {
	// Removing this alpha will have no affect on the singular/nonsingular block structure of the factorisation.

	D.remove(fpos);
        freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());
    }

    else
    {
	D.remove(fpos);
        freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

	int aposdummy = 0;
	int bposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);
    }

    return;
}

int optContext::extendFactBeta(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
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

    int retval = (pBetaF.size())-1;
    int i = pivBetaF(retval);
    int jP;

    (GpnFColNorm)("&",i) = 0.0;

    //if ( pAlphaF.size() )
    {
	for ( jP = 0 ; jP < pAlphaF.size() ; ++jP )
	{
	    (GpnFColNorm)("&",i) += (Gpn(pivAlphaF(jP),i)*Gpn(pivAlphaF(jP),i));
	}
    }

    (betaFix)("&",i) = CALCBETAFIX(i);

    fBetaF.add(fBetaF.size());
    fBetaF("&",fBetaF.size()-1) = -1;

    if ( !(betaFix.v(i)) )
    {
	int aposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,retval);
    }

    return retval;
}

void optContext::shrinkFactBeta(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
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

    int jP;
    int fpos = fBetaF.v(iP);

    fBetaF.remove(iP);

    if ( fpos >= 0 )
    {
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

	//if ( fBetaF.size() )
	{
	    for ( jP = 0 ; jP < fBetaF.size() ; ++jP )
	    {
		if ( fBetaF.v(jP) > fpos )
		{
		    --(fBetaF("&",jP));
		}
	    }
	}

        --dnfact;

        retMatrix<double> tmpmGp;
        retMatrix<double> tmpmGn;
        retMatrix<double> tmpmGpn;

	D.remove(fpos);
	freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpmGp),Gn(pBetaF,pBetaF,tmpmGn,0,1,dnfact-1,0,1,dnfact-1),Gpn(pAlphaF,pBetaF,tmpmGpn,0,1,(pAlphaF.size())-1,0,1,dnfact-1));
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

	int aposdummy = 0;
	int bposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);
    }

    return;
}









// Stream operators

std::ostream &operator<<(std::ostream &output, const optContext &src)
{
    output << "LB  pivot:          " << src.pAlphaLB      << "\n";
    output << "Z   pivot:          " << src.pAlphaZ       << "\n";
    output << "UB  pivot:          " << src.pAlphaUB      << "\n";
    output << "F   pivot:          " << src.pAlphaF       << "\n";
    output << "nC  pivot:          " << src.pBetaC        << "\n";
    output << "nF  pivot:          " << src.pBetaF        << "\n";
    output << "Alpha state:        " << src.dalphaState   << "\n";
    output << "Beta state:         " << src.dbetaState    << "\n";
    output << "Zero tolerance:     " << src.dzt           << "\n";
    output << "Keep factorisation: " << src.dkeepfact     << "\n";
    output << "aNLF:               " << src.daNLF         << "\n";
    output << "aNUF:               " << src.daNUF         << "\n";
    output << "Gpn column sums:    " << (src.GpnFColNorm)   << "\n";
    output << "Beta fixing:        " << (src.betaFix)       << "\n";
    output << "Beta fixing state:  " << (src.betaFixUpdate) << "\n";
    output << "nfact:              " << src.dnfact        << "\n";
    output << "pfact:              " << src.dpfact        << "\n";
    output << "D inter:            " << src.D             << "\n";
    output << "F position:         " << src.fAlphaF       << "\n";
    output << "nF position:        " << src.fBetaF        << "\n";
    output << "Factorisation:      " << src.freeVarChol   << "\n";

    return output;
}

std::istream &operator>>(std::istream &input, optContext &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.pAlphaLB;
    input >> dummy; input >> dest.pAlphaZ;
    input >> dummy; input >> dest.pAlphaUB;
    input >> dummy; input >> dest.pAlphaF;
    input >> dummy; input >> dest.pBetaC;
    input >> dummy; input >> dest.pBetaF;
    input >> dummy; input >> dest.dalphaState;
    input >> dummy; input >> dest.dbetaState;
    input >> dummy; input >> dest.dzt;
    input >> dummy; input >> dest.dkeepfact;
    input >> dummy; input >> dest.daNLF;
    input >> dummy; input >> dest.daNUF;
    input >> dummy; input >> (dest.GpnFColNorm);
    input >> dummy; input >> (dest.betaFix);
    input >> dummy; input >> (dest.betaFixUpdate);
    input >> dummy; input >> dest.dnfact;
    input >> dummy; input >> dest.dpfact;
    input >> dummy; input >> dest.D;
    input >> dummy; input >> dest.fAlphaF;
    input >> dummy; input >> dest.fBetaF;
    input >> dummy; input >> dest.freeVarChol;

    return input;
}
