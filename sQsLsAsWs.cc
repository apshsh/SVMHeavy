
//
// Sparse quadratic solver - large scale, active set, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "sQsLsAsWs.hpp"
#include "kcache.hpp"
#include "smatrix.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

//#define DEBUGTERSE

#define DEBUGCATCHEND {                               \
int _aerr,_berr; double _ferr,_gerr;                  \
errstream() << "\tmax factorisation error = " << (_ferr = x.fact_testFactInt(Gp,Gn,Gpn) ) << ", max gradient error = " << ( _gerr = x.testGradInt(_aerr,_berr,Gp,Gn,Gpn,gp,gn,hp) ) << "\t"; \
errstream() << _aerr << "," << _berr << " ";          \
if ( sqrt(_ferr*_ferr)+sqrt(_gerr*_gerr) > 0.1 ) {    \
errstream() << "*+*+*+"; } errstream() << "\n";       \
}

#ifdef DEBUGDEEP
#define DEBUGCATCH { int _aerr,_berr; double _ferr,_gerr; errstream() << "\tmax factorisation error = " << (_ferr = x.fact_testFactInt(Gp,Gn,Gpn) ) << ", max gradient error = " << ( _gerr = x.testGradInt(_aerr,_berr,Gp,Gn,Gpn,gp,gn,hp) ) << "\t"; errstream() << _aerr << "," << _berr << " "; if ( sqrt(_ferr*_ferr)+sqrt(_gerr*_gerr) > 0.1 ) { errstream() << "*+*+*+"; } errstream() << "\n"; }
#endif

#ifndef DEBUGDEEP
#ifdef DEBUGBRIEF
#define DEBUGCATCH { int _aerr,_berr; double _gerr; errstream() << "\tmax gradient error = " << ( _gerr = x.testGradInt(_aerr,_berr,Gp,Gn,Gpn,gp,gn,hp) ) << "\t"; errstream() << _aerr << "," << _berr << " "; if ( sqrt(_gerr*_gerr) > 0.1 ) { errstream() << "*+*+*+"; } errstream() << "\n"; }
//#define DEBUGCATCH { int __ttemp = x.testGradInt(Gp,Gn,Gpn,gp,gn,hp); errstream() << "\tmax gradient error = " << __ttemp << "\n"; if ( __ttemp > 0.1 ) { errstream() << "x is " << x << "\n"; exit(1); } }
#endif

#ifndef DEBUGBRIEF
#define DEBUGCATCH errstream() << "\n";
#endif
#endif

//////#define FEEDBACK_CYCLE 50
///#define FEEDBACK_CYCLE 64
//#define FEEDBACK_CYCLE 128
//#define FEEDBACK_CYCLE 256
#define FEEDBACK_CYCLE 512
#define MAJOR_FEEDBACK_CYCLE 10000



// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
//
// NB: all changes to solve_quadratic_program must be reflected in solve_quadratic_program_hpzero
//
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************









// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************

int int_solve_quadratic_program(svmvolatile int &killSwitch,
                                optState<double,double> &x,
                                const Matrix<double> &Gp, const Matrix<double> &Gn, Matrix<double> &Gpn,
                                const Vector<double> &gp, const Vector<double> &gn, Vector<double>  &hp, const Vector<double> &lb, const Vector<double> &ub,
                                stopCond sc, //int maxitcntint, double xmtrtime, double xmtrtimeend,
                                int GpnRowTwoSigned, const Vector<double> &GpnRowTwoMag,
                                double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg,
                                double stepscalefactor, int chistart, int linbreak = 0);
int int_solve_quadratic_program_hpzero(svmvolatile int &killSwitch,
                                optState<double,double> &x,
                                const Matrix<double> &Gp, const Matrix<double> &Gn, Matrix<double> &Gpn,
                                const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &lb, const Vector<double> &ub,
                                stopCond sc, //int maxitcntint, double xmtrtime, double xmtrtimeend,
                                double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg,
                                double stepscalefactor, int chistart, int linbreak = 0);

int fullOptStateActive::solve(svmvolatile int &killSwitch)
{
//    NiceAssert( !fixHigherOrderTerms );

    int res = 0;
    int chival = ( chistart == -1 ) ? gp.size() : chistart;

//    double stepscale = stepscalefactor;

    if ( hpzero )
    {
        //res = int_solve_quadratic_program_hpzero(killSwitch,x,Gp,Gn,Gpn,gp,gn,lb,ub,maxitcnt,maxruntime,runtimeend,nullptr,nullptr,1.0,chival,linbreak);
        res = int_solve_quadratic_program_hpzero(killSwitch,x,Gp,Gn,Gpn,gp,gn,lb,ub,sc,nullptr,nullptr,1.0,chival,linbreak);
    }

    else
    {
        //res = int_solve_quadratic_program(killSwitch,x,Gp,Gn,Gpn,gp,gn,hp,lb,ub,maxitcnt,maxruntime,runtimeend,GpnRowTwoSigned,GpnRowTwoMag,nullptr,nullptr,1.0,chival,linbreak);
        res = int_solve_quadratic_program(killSwitch,x,Gp,Gn,Gpn,gp,gn,hp,lb,ub,sc,GpnRowTwoSigned,GpnRowTwoMag,nullptr,nullptr,1.0,chival,linbreak);
    }

    return res;
}


// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************
// **********************************************************************************************

int int_solve_quadratic_program(svmvolatile int &killSwitch,
                                optState<double,double> &x,
                                const Matrix<double> &Gp, const Matrix<double> &Gn, Matrix<double> &Gpn,
                                const Vector<double> &gp, const Vector<double> &gn, Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub,
                                stopCond sc, //int maxitcntint, double xmtrtime, double xmtrtimeend,
                                int GpnRowTwoSigned, const Vector<double> &GpnRowTwoMag,
                                double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg,
                                double stepscalefactor, int chistart, int linbreak)
{
    nullPrint(errstream(),"#");
    int iP,res = 0;
    int kickstart = 0;

    double *uservars[] = { &(sc.maxitcnt), &(sc.maxruntime), &(sc.runtimeend), &stepscalefactor, nullptr };
    const char *varnames[] = { "maxitcnt", "maxruntime", "runtimeend", "stepscale", nullptr };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Absolute training time end (relative, secs, -1 for na)", "Step scale used for higher-order terms", nullptr };

    retVector<double> tmpva;
    retVector<double> tmpvb;

    (void) fixHigherOrderTerms;
    (void) htArg;
//    if ( fixHigherOrderTerms )
//    {
//        fixHigherOrderTerms(x, htArg);
//    }

    if ( x.aN() )
    {
	int isopt = 0;

	Vector<double> stepAlpha(x.alpha());
	Vector<double> stepBeta (x.beta ());

        int iset;
	int csts = 0; // 0 means not in combined step sequence, 1 otherwise
	Vector<double> combStepAlpha(x.alpha());
	Vector<double> combStepBeta (x.beta ());
	Vector<double> startAlphaGrad(x.alpha());
        Vector<int> FnF;
        Vector<int> startPivAlphaF;

	double scale = gp.v(isopt); // warning removal
	double gradmag;
        double setval;

	if ( x.bN() )
	{
	    gradmag = gn.v(isopt); // warning removal
	}

	int asize;
	int bsize;
	int steptype;
	int alphaFIndex;
	int alphaCIndex;
	int betaFIndex;
	int betaCIndex;
	int stateChange;
	int scaletype;
	int stalledflag = 0;

        // ..................................................................
        // ..................................................................
        // ..................................................................
        #ifdef DEBUGOPT
        #ifndef DEBUGTERSE
        errstream() << "DEBUG: initial state - " << x << "\n";
        errstream() << "DEBUG: initial Gp - " << Gp << "\n";
        errstream() << "DEBUG: initial Gpn - " << Gpn << "\n";
        errstream() << "DEBUG: initial Gn - " << Gn << "\n";
        errstream() << "DEBUG: initial gp - " << gp << "\n";
        errstream() << "DEBUG: initial gn - " << gn << "\n";
        errstream() << "DEBUG: initial hp - " << hp << "\n";
        errstream() << "DEBUG: initial lb - " << lb << "\n";
        errstream() << "DEBUG: initial ub - " << ub << "\n";
        errstream() << "DEBUG: initial GpnRowTwo (" << GpnRowTwoSigned << ") - " << GpnRowTwoMag << "\n";
        errstream() << "DEBUG: free all unrestricted betas - ";
	DEBUGCATCH;
        #endif
        #endif
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        time_used start_time = TIMECALL;
        time_used curr_time = start_time;
        size_t itcnt = 0;
        int timeout = 0;
        int bailout = 0;
        int stopout = 0;

        // Obscure note: in c++, if maxitcnt is a double then !maxitcnt is
        // true if maxitcnt == 0, false otherwise.  This is defined in the
        // standard, and the reason the following while statement will work.

        while ( !killSwitch && sc.stillrun(itcnt,isopt,timeout,bailout,stopout) )
	{
            // Refresh gradients (ie recalculate them if cumulative gradient
            // is too high) and calculate step

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "DEBUG: Calculate step - ";
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            x.refreshGrad(Gp,Gn,Gpn,gp,gn,hp);
	    steptype = x.fact_calcStep(stepAlpha,stepBeta,asize,bsize,Gp,Gp,Gn,Gpn,gp,gn,hp,lb,ub);

//            if ( fixHigherOrderTerms )
//            {
//                 stepAlpha("&",x.pivAlphaF()).scale(stepscalefactor);
//                 stepBeta("&",x.pivBetaF()).scale(stepscalefactor);
//            }

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            retVector<double> tmpva;
            retMatrix<double> tmpma;
            errstream() << "asize = " << asize << ", bsize = " << bsize << ", steptype = " << steptype << " - ";
            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
            errstream() << "\n";
            //DEBUGCATCH;
            #ifdef DEBUGDEEP
            errstream() << "\n";
            //errstream() << "stepAlpha: " << stepAlpha << "\n";
            //errstream() << "stepBeta: "  << stepBeta  << "\n";
            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF (),tmpva) << "\n";
            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF (),tmpva) << "\n";
            errstream() << "Gp:  " << Gp (x.pivAlphaF(),x.pivAlphaF(),tmpma) << "\n";
            errstream() << "Gpn: " << Gpn(x.pivAlphaF(),x.pivBetaF (),tmpma) << "\n";
            errstream() << "Gn:  " << Gn (x.pivBetaF (),x.pivBetaF (),tmpma) << "\n";
            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
            errstream() << "===============================================================\n";
            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF (),tmpva) << "\n";
            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF (),tmpva) << "\n";
            errstream() << "Gp:  " << Gp (x.pivAlphaF(),x.pivAlphaF(),tmpma) << "\n";
            errstream() << "Gpn: " << Gpn(x.pivAlphaF(),x.pivBetaF (),tmpma) << "\n";
            errstream() << "Gn:  " << Gn (x.pivBetaF (),x.pivBetaF (),tmpma) << "\n";
            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
            errstream() << "state: " << x << "\n";
            #endif
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            // Scale step to enforce feasibility

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "DEBUG: Scale Step - ";
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	    scaletype = x.scaleFStep(scale,alphaFIndex,betaFIndex,betaCIndex,stateChange,asize,bsize,bailout,stepAlpha("&",x.pivAlphaF(),tmpva),stepBeta("&",x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,hp,lb,ub);

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "scaletype = " << scaletype << ", scale = " << scale << " - ";
            errstream() << "\n";
            //DEBUGCATCH;
            #ifdef DEBUGDEEP
            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF (),tmpva) << "\n";
            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF (),tmpva) << "\n";
            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
            #endif
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	    if ( ( scale > 0 ) && ( asize || bsize ) )
	    {
		stalledflag = ( stalledflag > 0 ) ? ( stalledflag-1 ) : 0;

                // Step is non-zero, so take it

                if ( !steptype && !scaletype )
		{
//		    if ( fixHigherOrderTerms )
//		    {
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "DEBUG: Take Unscaled non-quadratic Newton Step - ";
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//			x.stepFGeneral(asize,bsize,stepAlpha(x.pivAlphaF()),stepBeta(x.pivBetaF()),Gp,Gn,Gpn,gp,gn,hp,0,0);
//                        fixHigherOrderTerms(x,htArg);
//
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "\n";
//                        //DEBUGCATCH;
//                        #ifdef DEBUGDEEP
//                        errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
//                        errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
//                        errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
//                        errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
//                        errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
//                        errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//		    }
//
//		    else
		    {
			if ( csts )
			{
                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take Unscaled Newton Step at sequence end - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    x.stepFNewtonFull(bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,hp,0);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "\n";
                            //DEBUGCATCH;
                            #ifdef DEBUGDEEP
                            retVector<double> tmpva;
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF (),tmpva) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF (),tmpva) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
                            #endif
                            #endif
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: add to combined step\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    combStepAlpha("&",x.pivAlphaF(),tmpva) += stepAlpha(x.pivAlphaF(),tmpvb);
			    combStepBeta("&",x.pivBetaF(),tmpva)   += stepBeta(x.pivBetaF(),tmpvb);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: update non-free gradients - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            #ifndef DEBUGTERSE
			    DEBUGCATCH;
                            #endif
                            #ifdef DEBUGDEEP
                            retVector<double> tmpva;
                            errstream() << "FAlpha:          " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "FBeta:           " << (x.beta())(x.pivBetaF (),tmpva) << "\n";
                            errstream() << "stepFAlpha:      " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "stepFBeta:       " << stepBeta (x.pivBetaF (),tmpva) << "\n";
                            errstream() << "alphaFGrad:      " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "betaFGrad:       " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
                            errstream() << "cstepFAlpha:     " << combStepAlpha(startPivAlphaF,tmpva) << "\n";
                            errstream() << "cstepFBeta:      " << combStepBeta(x.pivBetaF(),tmpva) << "\n";
                            errstream() << "startAlphaGrad:  " << startAlphaGrad << "\n";
                            errstream() << "FnF:             " << FnF << "\n";
                            errstream() << "startPivAlphaF:  " << startPivAlphaF << "\n";
                            #endif
                            errstream() << "DEBUG: clear FnF cache\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    FnF.resize(0);

			    csts = 0;
			}

			else
			{
                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take Unscaled Newton Step unsequenced - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    x.stepFNewtonFull(bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,hp);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            #ifndef DEBUGTERSE
			    DEBUGCATCH;
                            #endif
                            #ifdef DEBUGDEEP
                            retVector<double> tmpva;
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF (),tmpva) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF (),tmpva) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
                            #endif
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			}
		    }
		}

		else if ( !steptype )
		{
//		    if ( fixHigherOrderTerms )
//		    {
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "DEBUG: Take Scaled non-quadratic Newton Step - ";
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//			x.stepFGeneral(asize,bsize,stepAlpha(x.pivAlphaF()),stepBeta(x.pivBetaF()),Gp,Gn,Gpn,gp,gn,hp,0,0);
//                        fixHigherOrderTerms(x,htArg);
//
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "\n";
//                        //DEBUGCATCH;
//                        #ifdef DEBUGDEEP
//                        errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
//                        errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
//                        errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
//                        errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
//                        errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
//                        errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//		    }
//
//		    else
		    {
			if ( csts )
			{
                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take Scaled Newton Step in series - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    x.stepFNewton(scale,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,hp,0);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "\n";
                            //DEBUGCATCH;
                            #ifdef DEBUGDEEP
                            retVector<double> tmpva;
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF (),tmpva) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF (),tmpva) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
                            #endif
                            #endif
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: add to combined step\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    combStepAlpha("&",x.pivAlphaF(),tmpva) += stepAlpha(x.pivAlphaF(),tmpvb);
			    combStepBeta("&",x.pivBetaF(),tmpva)   += stepBeta(x.pivBetaF(),tmpvb);
			}

			else
			{
			    startAlphaGrad.set(x.pivAlphaF(),(x.alphaGrad())(x.pivAlphaF(),tmpvb));

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take first scaled Newton Step in series - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    x.stepFNewton(scale,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,hp,0);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "\n";
                            //DEBUGCATCH;
                            #ifdef DEBUGDEEP
                            retVector<double> tmpva;
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF (),tmpva) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF (),tmpva) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF(),tmpva) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),tmpva) << "\n";
                            #endif
                            #endif
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: set combined step variables\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    combStepAlpha.set(x.pivAlphaF(),stepAlpha(x.pivAlphaF(),tmpvb));
			    combStepBeta.set(x.pivBetaF(),stepBeta(x.pivBetaF(),tmpvb));

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: clear FnF cache\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    FnF.resize(0);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: record start pivots and state\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    startPivAlphaF  = x.pivAlphaF();

			    csts = 1;
			}

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Finished scaled newton step\n";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }
		}

		else
		{
//		    if ( fixHigherOrderTerms )
//		    {
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "DEBUG: Take Scaled Linear Step - " << stepAlpha(x.pivAlphaF()) << " - " << asize << "," << bsize << "\n";
//                        #ifndef DEBUGTERSE
//			DEBUGCATCH;
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//			x.stepFGeneral(asize,bsize,stepAlpha(x.pivAlphaF()),stepBeta(x.pivBetaF()),Gp,Gn,Gpn,gp,gn,hp,0,0);
//                        fixHigherOrderTerms(x,htArg);
//
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        #ifndef DEBUGTERSE
//			DEBUGCATCH;
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//		    }
//
//		    else
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Take Scaled Linear Step - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			if ( csts )
			{
			    x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			    csts = 0;
			}

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        #ifndef DEBUGTERSE
			DEBUGCATCH;
                        #endif
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( steptype == 1 )
                        {
                            if ( linbreak )
                            {
                                bailout = 50;
                            }

                            errstream("&");
                            x.stepFLinear(asize,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,hp);
                        }

                        else
                        {
                            if ( linbreak )
                            {
                                bailout = 50;
                            }

                            errstream("&$$");
                            x.stepFGeneral(asize,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,hp);
                        }

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        #ifndef DEBUGTERSE
			DEBUGCATCH;
                        #endif
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }
		}
	    }

	    else
	    {
                // Step is zero, so deal with that

                // ..................................................................
                // ..................................................................
                // ..................................................................
                #ifdef DEBUGOPT
                errstream() << "DEBUG: set stalled flag to " << (stalledflag+2) << " - ";
                #endif
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		if ( csts )
		{
		    x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

		    csts = 0;
		}

		if ( asize || bsize )
		{
		    stalledflag += 2;
		}

                // ..................................................................
                // ..................................................................
                // ..................................................................
                #ifdef DEBUGOPT
                errstream() << "\n";
                //DEBUGCATCH;
                #endif
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	    }

	    if ( scaletype )
	    {
		// If step scaled: constrain variable or free beta

		if ( alphaFIndex >= 0 )
		{
		    if ( stateChange == -2 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha LF-LB (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
			}

			x.modAlphaLFtoLB(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else if ( stateChange == -1 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha LF-Z (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
                            startAlphaGrad("&",x.pivAlphaF(alphaFIndex)) += hp.v(x.pivAlphaF(alphaFIndex));
			}

                        iset = x.pivAlphaF(alphaFIndex);

			x.modAlphaLFtoZ(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                        if ( iset >= chistart )
                        {
                            x.changeAlphaRestrict(iset,3,Gp,Gp,Gn,Gpn,gp,gn,hp);
                        }

			if ( GpnRowTwoSigned )
			{
			    setval = 0.0;

			    x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
			    Gpn("&",iset,Gpn.numCols()-1) = setval;
			}

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else if ( stateChange == +1 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha UF-Z (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
                            startAlphaGrad("&",x.pivAlphaF(alphaFIndex)) -= hp.v(x.pivAlphaF(alphaFIndex));

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: FnF extended: " << FnF << "\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			}

                        iset = x.pivAlphaF(alphaFIndex);

			x.modAlphaUFtoZ(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                        if ( iset >= chistart )
                        {
                            x.changeAlphaRestrict(iset,3,Gp,Gp,Gn,Gpn,gp,gn,hp);
                        }

			if ( GpnRowTwoSigned )
			{
			    setval = 0.0;

			    x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
			    Gpn("&",iset,Gpn.numCols()-1) = setval;
			}

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n" << x << "\n";
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else
		    {
                        NiceAssert( stateChange == +2 );

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha UF-UB (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
			}

			x.modAlphaUFtoUB(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }
		}

		else if ( betaFIndex >= 0 )
		{
                    NiceAssert( ( stateChange == -1 ) || ( stateChange == +1 ) );

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "DEBUG: Beta F-C (" << betaFIndex << " -> " << x.pivBetaF(betaFIndex) << ") - ";
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		    if ( csts )
		    {
			x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			csts = 0;
		    }

		    x.modBetaFtoC(betaFIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "\n";
                    //DEBUGCATCH;
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		}

		else
		{
                    NiceAssert( betaCIndex >= 0 );
                    NiceAssert( ( stateChange == -1 ) || ( stateChange == +1 ) );

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "DEBUG: Beta C-F (" << betaCIndex << " -> " << x.pivBetaC(betaCIndex) << ") - ";
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		    if ( csts )
		    {
			x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			csts = 0;
		    }

		    x.modBetaCtoF(betaCIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "\n";
                    //DEBUGCATCH;
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		}
	    }

	    else
	    {
//		if ( fixHigherOrderTerms )
//		{
//		    if ( x.aNF() )
//		    {
//			isopt = ( maxabs(x.alphaGrad()(x.pivAlphaF()),iP) > x.opttol() ) ? 0 : 1;
//		    }
//
//		    else
//		    {
//			isopt = 1;
//		    }
//		}
//
//		else
		{
                    if ( ( steptype == 1 ) || !(x.aNF()) )
                    {
                        isopt = 1;
                    }

                    else
                    {
			isopt = ( maxabs(x.alphaGrad()(x.pivAlphaF(),tmpva),iP) > x.opttol() ) ? 0 : 1;
                    }
		}

		if ( isopt )
		{
		    // If step unscaled: test optimality

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "DEBUG: Test optimality - ";
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		    if ( csts )
		    {
			x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			csts = 0;
		    }

		    if ( GpnRowTwoSigned )
		    {
			hp.set(x.pivAlphaZ(),GpnRowTwoMag(x.pivAlphaZ(),tmpvb));
			hp("&",x.pivAlphaZ(),tmpva) *= x.beta(Gpn.numCols()-1);
		    }

		    isopt = x.maxGradNonOpt(alphaCIndex,betaCIndex,stateChange,gradmag,Gp,Gn,Gpn,gp,gn,hp);

		    if ( GpnRowTwoSigned )
		    {
			hp("&",x.pivAlphaZ(),tmpva) = 0.0;
		    }

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
		    if ( !isopt )
		    {
                        errstream() << "max gradient error is " << gradmag << " for " << alphaCIndex << ", " << betaCIndex << ", " << stateChange << " - ";
		    }
                    errstream() << "\n";
                    //DEBUGCATCH;
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		    if ( !isopt && ( steptype != 2 ) )
		    {
			if ( ( alphaCIndex == -1 ) && ( betaCIndex == -1 ) )
			{
			    // Non-feasible, unable to proceed.
                            // Trigger external presolver

                            res = -1;
			    goto getout;
			}

			else if ( ( stalledflag >= 3 ) && !(x.aNC()) )
			{
			    // Stalled with no constrained variables and not optimal?
			    // Not actually possible.

			    isopt = 1;
			}

			else if ( stalledflag >= 3 )
			{
			    // Stalled for some reason.  Start randomly freeing variables
			    // to break the impasse

			    int conNLB = x.aNLB();
                            int conNZ  = x.aNZ();
			    int conNUB = x.aNUB();

                            NiceAssert( conNLB+conNZ+conNUB );

			tryagain:
			    //int randfreechoice = svm_rand()%(conNLB+conNZ+conNUB);
			    int randfreechoice = rand()%(conNLB+conNZ+conNUB);

			    if ( randfreechoice < conNLB )
			    {
				if ( Gp.v(x.pivAlphaLB(randfreechoice),x.pivAlphaLB(randfreechoice)) < x.zerotol() )
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "DEBUG: impasse breaker LB -> LF " << randfreechoice << " - ";
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				x.modAlphaLBtoLF(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "\n";
                                //DEBUGCATCH;
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			    }

			    else if ( randfreechoice < conNLB+conNZ )
			    {
				randfreechoice -= conNLB;

				if ( Gp.v(x.pivAlphaZ(randfreechoice),x.pivAlphaZ(randfreechoice)) < x.zerotol() )
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}

				if ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 2 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> LF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    if ( GpnRowTwoSigned )
				    {
					iset = x.pivAlphaZ(randfreechoice);

					setval = -GpnRowTwoMag.v(iset);

					x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
					Gpn("&",iset,Gpn.numCols()-1) = setval;
				    }

				    x.modAlphaZtoLF(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 1 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> UF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    if ( GpnRowTwoSigned )
				    {
					iset = x.pivAlphaZ(randfreechoice);

					setval = GpnRowTwoMag.v(iset);

					x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
					Gpn("&",iset,Gpn.numCols()-1) = setval;
				    }

				    x.modAlphaZtoUF(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				//else if ( ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 0 ) && ( svm_rand()%2 ) )
				else if ( ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 0 ) && ( rand()%2 ) )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> LF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    if ( GpnRowTwoSigned )
				    {
					iset = x.pivAlphaZ(randfreechoice);

					setval = -GpnRowTwoMag.v(iset);

					x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
					Gpn("&",iset,Gpn.numCols()-1) = setval;
				    }

				    x.modAlphaZtoLF(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 0 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> UF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    if ( GpnRowTwoSigned )
				    {
					iset = x.pivAlphaZ(randfreechoice);

					setval = GpnRowTwoMag.v(iset);

					x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
					Gpn("&",iset,Gpn.numCols()-1) = setval;
				    }

				    x.modAlphaZtoUF(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}
			    }

			    else
			    {
				randfreechoice -= (conNLB+conNZ);

				if ( Gp.v(x.pivAlphaUB(randfreechoice),x.pivAlphaUB(randfreechoice)) < x.zerotol() )
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "DEBUG: impasse breaker UB -> UF " << randfreechoice << " - ";
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				x.modAlphaUBtoUF(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "\n";
                                //DEBUGCATCH;
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			    }
			}

			else
			{
			    // If non-optimal: free variables

			    if ( alphaCIndex >= 0 )
			    {
				if ( stateChange == -2 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha LB-LF (" << alphaCIndex << " -> " << x.pivAlphaLB(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    x.modAlphaLBtoLF(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( stateChange == -1 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha Z-LF (" << alphaCIndex << " -> " << x.pivAlphaZ(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    if ( GpnRowTwoSigned )
				    {
					iset = x.pivAlphaZ(alphaCIndex);

					setval = -GpnRowTwoMag.v(iset);

					x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
					Gpn("&",iset,Gpn.numCols()-1) = setval;
				    }

				    x.modAlphaZtoLF(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( stateChange == +1 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha Z-UF (" << alphaCIndex << " -> " << x.pivAlphaZ(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    if ( GpnRowTwoSigned )
				    {
					iset = x.pivAlphaZ(alphaCIndex);

					setval = GpnRowTwoMag.v(iset);

					x.refactGpnElm(Gp,Gn,Gpn,setval,gp,gn,hp,iset,Gpn.numCols()-1);
					Gpn("&",iset,Gpn.numCols()-1) = setval;
				    }

				    x.modAlphaZtoUF(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else
				{
                                    NiceAssert( stateChange == +2 );

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha UB-UF (" << alphaCIndex << " -> " << x.pivAlphaUB(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    x.modAlphaUBtoUF(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}
			    }

			    else
			    {
                                NiceAssert( betaCIndex >= 0 );

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "DEBUG: Beta C to F (" << betaCIndex << " -> " << x.pivBetaC(betaCIndex) << ") - ";
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				x.modBetaCtoF(betaCIndex,Gp,Gp,Gn,Gpn,gp,gn,hp);

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "\n";
                                //DEBUGCATCH;
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			    }
			}
		    }
		}
	    }

            sc.testopt(itcnt,isopt,timeout,bailout,stopout,kickstart,FEEDBACK_CYCLE,MAJOR_FEEDBACK_CYCLE,start_time,curr_time,"quadratic (sQsLsAsWs) optimisation",uservars,varnames,vardescr);
	}

        // ..................................................................
        // ..................................................................
        // ..................................................................
        #ifdef DEBUGOPT
        errstream() << "DEBUG: Optimisation complete\n";
        DEBUGCATCHEND
        errstream() << "DEBUG: final state - " << x << "\n";
        #endif
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



















	//if ( isopt )
	//{
	//    int i;
        //
	    // Constrain alphas at bounds but left hanging
	    //
	    // BUGFIX: this is a bad idea!  In the presolver, the conditions here != the conditions
            // Alistair Shilton 2014 (c) wrote this code
            // in optstate, and it introduces some serious numerical problems!
            //
	    //if ( x.aNF() )
	    //{
	    //    for ( i = x.aNF()-1 ; i >= 0 ; --i )
	    //    {
	    //        if ( x.alphaRestrict(x.pivAlphaF()(i)) == 0 )
	    //        {
	    //    	if ( x.alphaState()(x.pivAlphaF()(i)) == -1 )
	    //    	{
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) >= -(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaLFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    //    	    }
            //
	    //    	    else if ( x.alpha()(x.pivAlphaF()(i)) <= lb.v(i)+(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaLFtoLB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
	    //    	    }
	    //    	}
            //
	    //    	else if ( x.alphaState()(x.pivAlphaF()(i)) == +1 )
	    //    	{
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) <= (x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaUFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    //    	    }
            //
	    //    	    else if ( x.alpha()(x.pivAlphaF()(i)) >= ub.v(i)-(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaUFtoUB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
	    //    	    }
	    //    	}
	    //        }
            //
	    //        else if ( x.alphaRestrict(x.pivAlphaF()(i)) == 1 )
	    //        {
	    //    	if ( x.alphaState()(x.pivAlphaF()(i)) == +1 )
	    //    	{
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) <= (x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaUFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    //    	    }
            //
	    //    	    else if ( x.alpha()(x.pivAlphaF()(i)) >= ub.v(i)-(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaUFtoUB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
	    //    	    }
	    //    	}
	    //        }
            //
	    //        else if ( x.alphaRestrict(x.pivAlphaF()(i)) == 2 )
	    //        {
	    //    	if ( x.alphaState()(x.pivAlphaF()(i)) == -1 )
	    //    	{
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) >= -(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaLFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    //    	    }
            //
	    //    	    else if ( x.alpha()(x.pivAlphaF()(i)) <= lb.v(i)+(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaLFtoLB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
	    //    	    }
	    //    	}
	    //        }
	    //    }
	    //}
//#ifdef DEBUGOPT
//errstream() << x << "\n";
//#endif
	//}

	if ( !isopt )
	{
	    res = 1;
	}

        if ( bailout )
        {
            res = 10*bailout;
        }
    }

getout:
    return res;
}






#undef DEBUGCATCH
#undef DEBUGCATCHEND

#define DEBUGCATCHEND { int _aerr,_berr; double _ferr,_gerr; errstream() << "\tmax factorisation error = " << (_ferr = x.fact_testFactInt(Gp,Gn,Gpn) ) << ", max gradient error = " << ( _gerr = x.testGradInthpzero(_aerr,_berr,Gp,Gn,Gpn,gp,gn) ) << "\t"; errstream() << _aerr << "," << _berr << " "; if ( sqrt(_ferr*_ferr)+sqrt(_gerr*_gerr) > 0.1 ) { errstream() << "*+*+*+"; } errstream() << "\n"; }

#ifdef DEBUGDEEP
#define DEBUGCATCH { int _aerr,_berr; double _ferr,_gerr; errstream() << "\tmax factorisation error = " << (_ferr = x.fact_testFactInt(Gp,Gn,Gpn) ) << ", max gradient error = " << ( _gerr = x.testGradInthpzero(_aerr,_berr,Gp,Gn,Gpn,gp,gn) ) << "\t"; errstream() << _aerr << "," << _berr << " "; if ( sqrt(_ferr*_ferr)+sqrt(_gerr*_gerr) > 0.1 ) { errstream() << "*+*+*+"; } errstream() << "\n"; }
#endif

#ifndef DEBUGDEEP
#ifdef DEBUGBRIEF
#define DEBUGCATCH { int _aerr,_berr; double _gerr; errstream() << "\tmax gradient error = " << ( _gerr = x.testGradInthpzero(_aerr,_berr,Gp,Gn,Gpn,gp,gn) ) << "\t"; errstream() << _aerr << "," << _berr << " "; if ( sqrt(_gerr*_gerr) > 0.1 ) { errstream() << "*+*+*+"; } errstream() << "\n"; }
//#define DEBUGCATCH { int __ttemp = x.testGradInthpzero(Gp,Gn,Gpn,gp,gn); errstream() << "\tmax gradient error = " << __ttemp << "\n"; if ( __ttemp > 0.1 ) { errstream() << "x is " << x << "\n"; exit(1); } }
#endif

#ifndef DEBUGBRIEF
#define DEBUGCATCH errstream() << "\n";
#endif
#endif





int int_solve_quadratic_program_hpzero(svmvolatile int &killSwitch, optState<double,double> &x,
                                       const Matrix<double> &Gp, const Matrix<double> &Gn, Matrix<double> &Gpn,
                                       const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &lb, const Vector<double> &ub,
                                       stopCond sc, //int maxitcntint, double xmtrtime, double xmtrtimeend,
                                       double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg,
                                       double stepscalefactor, int chistart, int linbreak)
{
    int kickstart = 0;
    nullPrint(errstream(),"#");
    int iP,res = 0;

    double *uservars[] = { &(sc.maxitcnt), &(sc.maxruntime), &(sc.runtimeend), &stepscalefactor, nullptr };
    const char *varnames[] = { "maxitcnt", "maxruntime", "runtimeend", "stepscale", nullptr };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Absolute training time end (relative, secs, -1 for na)", "Step scale used for higher-order terms", nullptr };

    (void) fixHigherOrderTerms;
    (void) htArg;
//    if ( fixHigherOrderTerms )
//    {
//        fixHigherOrderTerms(x,htArg);
//    }

    if ( x.aN() )
    {
	//if ( x.aNZ() )
	{
	    // Free all alphas that are unrestricted and constrained at zero.  The sign doesn't
	    // matter as we ignore it for these variables during training and fix any
            // incorrect labelling at exit.

	    for ( int iP = (x.aNZ())-1 ; iP >= 0 ; --iP )
	    {
		if ( x.alphaRestrict(x.pivAlphaZ(iP)) == 0 )
		{
		    x.modAlphaZtoUFhpzero(iP,Gp,Gp,Gn,Gpn,gp,gn);
		}
	    }
	}
    }

    retVector<double> tmpva;
    retVector<double> tmpvb;

    if ( x.aN() )
    {
	int isopt = 0;

	Vector<double> stepAlpha(x.alpha());
	Vector<double> stepBeta (x.beta ());


        int iset;
	int csts = 0; // 0 means not in combined step sequence, 1 otherwise
	Vector<double> combStepAlpha(x.alpha());
	Vector<double> combStepBeta (x.beta ());
	Vector<double> startAlphaGrad(x.alpha());
        Vector<int> FnF;
        Vector<int> startPivAlphaF;

	double scale = gp.v(isopt); // warning removal
	double gradmag;


	if ( x.bN() )
	{
	    gradmag = gn.v(isopt); // warning removal
	}

	int asize;
	int bsize;
	int steptype;
	int alphaFIndex;
	int alphaCIndex;
	int betaFIndex;
	int betaCIndex;
	int stateChange;
	int scaletype;
	int stalledflag = 0;

        // ..................................................................
        // ..................................................................
        // ..................................................................
        #ifdef DEBUGOPT
        #ifndef DEBUGTERSE
        errstream() << "DEBUG: initial state - " << x << "\n";
        errstream() << "DEBUG: initial Gp - " << Gp << "\n";
        errstream() << "DEBUG: initial Gpn - " << Gpn << "\n";
        errstream() << "DEBUG: initial Gn - " << Gn << "\n";
        errstream() << "DEBUG: initial gp - " << gp << "\n";
        errstream() << "DEBUG: initial gn - " << gn << "\n";

        errstream() << "DEBUG: initial lb - " << lb << "\n";
        errstream() << "DEBUG: initial ub - " << ub << "\n";

//        errstream() << "DEBUG: maxitcnt - " << maxitcnt << "\n";
//        errstream() << "DEBUG: xmtrtime - " << xmtrtime << "\n";
//        errstream() << "DEBUG: xmtrtimeend - " << xmtrtimeend << "\n";
        errstream() << "DEBUG: isopt - " << isopt << "\n";
        errstream() << "DEBUG: free all unrestricted betas - ";
	DEBUGCATCH;
        #endif
        #endif
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        time_used start_time = TIMECALL;
        time_used curr_time = start_time;
        size_t itcnt = 0;
        int timeout = 0;
        int bailout = 0;
        int stopout = 0;

        while ( !killSwitch && sc.stillrun(itcnt,isopt,timeout,bailout,stopout) )
	{
	    // Calculate step

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "DEBUG: Calculate step - ";
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            x.refreshGradhpzero(Gp,Gn,Gpn,gp,gn);

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "- ";
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            steptype = x.fact_calcStephpzero(stepAlpha,stepBeta,asize,bsize,Gp,Gp,Gn,Gpn,gp,gn,lb,ub);

//	    if ( fixHigherOrderTerms )
//	    {
//		stepAlpha("&",x.pivAlphaF()).scale(stepscalefactor);
//                stepBeta("&",x.pivBetaF()).scale(stepscalefactor);
//	    }

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "asize = " << asize << ", bsize = " << bsize << ", steptype = " << steptype << " - ";
            retVector<double> ttmva; errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF (),ttmva) << "\n";
            errstream() << "\n";
            //DEBUGCATCH;
            #ifdef DEBUGDEEP
            errstream() << "\n";
            //errstream() << "stepAlpha: " << stepAlpha << "\n";
            //errstream() << "stepBeta: "  << stepBeta  << "\n";
            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
            errstream() << "Gp:  " << Gp (x.pivAlphaF(),x.pivAlphaF()) << "\n";
            errstream() << "Gpn: " << Gpn(x.pivAlphaF(),x.pivBetaF ()) << "\n";
            errstream() << "Gn:  " << Gn (x.pivBetaF (),x.pivBetaF ()) << "\n";
            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
            errstream() << "===============================================================\n";
            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
            errstream() << "Gp:  " << Gp (x.pivAlphaF(),x.pivAlphaF()) << "\n";
            errstream() << "Gpn: " << Gpn(x.pivAlphaF(),x.pivBetaF ()) << "\n";
            errstream() << "Gn:  " << Gn (x.pivBetaF (),x.pivBetaF ()) << "\n";
            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
            errstream() << "state: " << x << "\n";
            #endif
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	    // Scale step

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "DEBUG: Scale Step - ";
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            scaletype = x.scaleFStephpzero(scale,alphaFIndex,betaFIndex,betaCIndex,stateChange,asize,bsize,bailout,stepAlpha("&",x.pivAlphaF(),tmpva),stepBeta("&",x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,lb,ub);

            // ..................................................................
            // ..................................................................
            // ..................................................................
            #ifdef DEBUGOPT
            errstream() << "scaletype = " << scaletype << ", scale = " << scale << " - ";
            errstream() << "\n";
            //DEBUGCATCH;
            #ifdef DEBUGDEEP
            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
            #endif
            #endif
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	    if ( ( scale > 0 ) && ( asize || bsize ) )
	    {
		stalledflag = ( stalledflag > 0 ) ? ( stalledflag-1 ) : 0;

		// Take step if non-zero

		if ( !steptype && !scaletype )
		{
//		    if ( fixHigherOrderTerms )
//		    {
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "DEBUG: Take Unscaled non-quadratic Newton Step - ";
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//			x.stepFGeneralhpzero(asize,bsize,stepAlpha(x.pivAlphaF()),stepBeta(x.pivBetaF()),Gp,Gn,Gpn,gp,gn,0,0);
//                        fixHigherOrderTerms(x,htArg);
//
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "\n";
//                        //DEBUGCATCH;
//                        #ifdef DEBUGDEEP
//                        errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
//                        errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
//                        errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
//                        errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
//                        errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
//                        errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//		    }
//
//		    else
		    {
			if ( csts )
			{
                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take Unscaled Newton Step at sequence end - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            x.stepFNewtonFullhpzero(bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,0);
 
                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "\n";
                            //DEBUGCATCH;
                            #ifdef DEBUGDEEP
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
                            #endif
                            #endif
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: add to combined step\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            combStepAlpha("&",x.pivAlphaF(),tmpva) += stepAlpha(x.pivAlphaF(),tmpvb);
			    combStepBeta("&",x.pivBetaF(),tmpva)   += stepBeta(x.pivBetaF(),tmpvb);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: update non-free gradients - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            #ifndef DEBUGTERSE
			    DEBUGCATCH;
                            #endif
                            #ifdef DEBUGDEEP
                            errstream() << "FAlpha:          " << (x.alpha())(x.pivAlphaF()) << "\n";
                            errstream() << "FBeta:           " << (x.beta())(x.pivBetaF ()) << "\n";
                            errstream() << "stepFAlpha:      " << stepAlpha(x.pivAlphaF()) << "\n";
                            errstream() << "stepFBeta:       " << stepBeta (x.pivBetaF ()) << "\n";
                            errstream() << "alphaFGrad:      " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
                            errstream() << "betaFGrad:       " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
                            errstream() << "cstepFAlpha:     " << combStepAlpha(startPivAlphaF) << "\n";
                            errstream() << "cstepFBeta:      " << combStepBeta(x.pivBetaF()) << "\n";
                            errstream() << "startAlphaGrad:  " << startAlphaGrad << "\n";
                            errstream() << "FnF:             " << FnF << "\n";
                            errstream() << "startPivAlphaF:  " << startPivAlphaF << "\n";
                            #endif
                            errstream() << "DEBUG: clear FnF cache\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            FnF.resize(0);

			    csts = 0;
			}

			else
			{
                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take Unscaled Newton Step unsequenced - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            x.stepFNewtonFullhpzero(bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            #ifndef DEBUGTERSE
			    DEBUGCATCH;
                            #endif
                            #ifdef DEBUGDEEP
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
                            #endif
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			}
		    }
		}

		else if ( !steptype )
		{
//		    if ( fixHigherOrderTerms )
//		    {
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "DEBUG: Take Scaled non-quadratic Newton Step - ";
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//                        x.stepFGeneralhpzero(asize,bsize,stepAlpha(x.pivAlphaF()),stepBeta(x.pivBetaF()),Gp,Gn,Gpn,gp,gn,0,0);
//                        fixHigherOrderTerms(x,htArg);
//
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "\n";
//                        //DEBUGCATCH;
//                        #ifdef DEBUGDEEP
//                        errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
//                        errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
//                        errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
//                        errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
//                        errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
//                        errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//		    }
//
//		    else
		    {
			if ( csts )
			{
                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take Scaled Newton Step in series - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            x.stepFNewtonhpzero(scale,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,0);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "\n";
                            //DEBUGCATCH;
                            #ifdef DEBUGDEEP
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
                            #endif
                            #endif
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: add to combined step\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            combStepAlpha("&",x.pivAlphaF(),tmpva) += stepAlpha(x.pivAlphaF(),tmpvb);
			    combStepBeta("&",x.pivBetaF(),tmpva)   += stepBeta(x.pivBetaF(),tmpvb);
			}

			else
			{
			    startAlphaGrad.set(x.pivAlphaF(),(x.alphaGrad())(x.pivAlphaF(),tmpvb));

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: Take first scaled Newton Step in series - ";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            x.stepFNewtonhpzero(scale,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn,0);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "\n";
                            //DEBUGCATCH;
                            #ifdef DEBUGDEEP
                            errstream() << "FAlpha: " << (x.alpha())(x.pivAlphaF()) << "\n";
                            errstream() << "FBeta: "  << (x.beta())(x.pivBetaF ()) << "\n";
                            errstream() << "stepFAlpha: " << stepAlpha(x.pivAlphaF()) << "\n";
                            errstream() << "stepFBeta: "  << stepBeta (x.pivBetaF ()) << "\n";
                            errstream() << "alphaFGrad:  " << (x.alphaGrad())(x.pivAlphaF()) << "\n";
                            errstream() << "betaFGrad:   " << (x.betaGrad ())(x.pivBetaF ()) << "\n";
                            #endif
                            #endif
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: set combined step variables\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    combStepAlpha.set(x.pivAlphaF(),stepAlpha(x.pivAlphaF(),tmpvb));
			    combStepBeta.set(x.pivBetaF(),stepBeta(x.pivBetaF(),tmpvb));


                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: clear FnF cache\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			    FnF.resize(0);

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................

                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: record start pivots and state\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            startPivAlphaF  = x.pivAlphaF();

			    csts = 1;
			}

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Finished scaled newton step\n";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }
		}

		else
		{
//		    if ( fixHigherOrderTerms )
//		    {
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        errstream() << "DEBUG: Take Scaled Linear Step - " << stepAlpha(x.pivAlphaF()) << " - " << asize << "," << bsize << "\n";
//                        #ifndef DEBUGTERSE
//			DEBUGCATCH;
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//			x.stepFGeneralhpzero(asize,bsize,stepAlpha(x.pivAlphaF()),stepBeta(x.pivBetaF()),Gp,Gn,Gpn,gp,gn,0,0);
//                        fixHigherOrderTerms(x,htArg);
//
//                        // ..................................................................
//                        // ..................................................................
//                        // ..................................................................
//                        #ifdef DEBUGOPT
//                        #ifndef DEBUGTERSE
//			DEBUGCATCH;
//                        #endif
//                        #endif
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//		    }
//
//		    else
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Take Scaled Linear Step - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( csts )
			{
			    x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			    csts = 0;
			}

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        #ifndef DEBUGTERSE
			DEBUGCATCH;
                        #endif
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( steptype == 1 )
                        {
                            if ( linbreak )
                            {
                                bailout = 50;
                            }

                            errstream("&");
                            x.stepFLinearhpzero(asize,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn);
                        }

                        else
                        {
                            if ( linbreak )
                            {
                                bailout = 50;
                            }

                            errstream("&$$");
                            x.stepFGeneralhpzero(asize,bsize,stepAlpha(x.pivAlphaF(),tmpva),stepBeta(x.pivBetaF(),tmpvb),Gp,Gn,Gpn,gp,gn);
                        }

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        #ifndef DEBUGTERSE
			DEBUGCATCH;
                        #endif
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }
		}
	    }

	    else
	    {
                // ..................................................................
                // ..................................................................
                // ..................................................................
                #ifdef DEBUGOPT
                errstream() << "DEBUG: set stalled flag to " << (stalledflag+2) << " - ";
                #endif
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                if ( csts )
		{
		    x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

		    csts = 0;
		}

		if ( asize || bsize )
		{
		    stalledflag += 2;
		}

                // ..................................................................
                // ..................................................................
                // ..................................................................
                #ifdef DEBUGOPT
                errstream() << "done\n";
                //DEBUGCATCH;
                #endif
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	    }

	    if ( scaletype )
	    {
		// If step scaled: constrain variable or free beta

		if ( alphaFIndex >= 0 )
		{
		    if ( stateChange == -3 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha LF-UB (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
			}

			x.modAlphaLFtoUBhpzero(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,ub);

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else if ( stateChange == -2 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha LF-LB (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
			}

			x.modAlphaLFtoLBhpzero(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,lb);

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else if ( stateChange == -1 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha LF-Z (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));

			}

                        iset = x.pivAlphaF(alphaFIndex);

			x.modAlphaLFtoZhpzero(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn);

                        if ( iset >= chistart )
                        {
                            x.changeAlphaRestricthpzero(iset,3,Gp,Gp,Gn,Gpn,gp,gn);
                        }

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else if ( stateChange == +1 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha UF-Z (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));

                            // ..................................................................
                            // ..................................................................
                            // ..................................................................
                            #ifdef DEBUGOPT
                            errstream() << "DEBUG: FnF extended: " << FnF << "\n";
                            #endif
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			}

                        iset = x.pivAlphaF(alphaFIndex);

			x.modAlphaUFtoZhpzero(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn);

                        if ( iset >= chistart )
                        {
                            x.changeAlphaRestricthpzero(iset,3,Gp,Gp,Gn,Gpn,gp,gn);
                        }

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n" << x << "\n";
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else if ( stateChange == +2 )
		    {
                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha UF-UB (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
			}

			x.modAlphaUFtoUBhpzero(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,ub);

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "\n";
                        //DEBUGCATCH;
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }

		    else
		    {
                        NiceAssert( stateChange == +3 );

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        errstream() << "DEBUG: Alpha UF-LB (" << alphaFIndex << " -> " << x.pivAlphaF(alphaFIndex) << ") - ";
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                        if ( csts )
			{
			    FnF.add(FnF.size());
			    FnF.sv((FnF.size())-1,x.pivAlphaF(alphaFIndex));
			}

			x.modAlphaUFtoLBhpzero(alphaFIndex,Gp,Gp,Gn,Gpn,gp,gn,lb);

                        // ..................................................................
                        // ..................................................................
                        // ..................................................................
                        #ifdef DEBUGOPT
                        #ifndef DEBUGTERSE
			DEBUGCATCH;
                        #endif
                        #endif
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		    }
		}

		else if ( betaFIndex >= 0 )
		{
                    NiceAssert( ( stateChange == -1 ) || ( stateChange == +1 ) );

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "DEBUG: Beta F-C (" << betaFIndex << " -> " << x.pivBetaF(betaFIndex) << ") - ";
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                    if ( csts )
		    {
			x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			csts = 0;
		    }

		    x.modBetaFtoChpzero(betaFIndex,Gp,Gp,Gn,Gpn,gp,gn);

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "\n";
                    //DEBUGCATCH;
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		}

		else
		{
                    NiceAssert( betaCIndex >= 0 );
                    NiceAssert( ( stateChange == -1 ) || ( stateChange == +1 ) );

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "DEBUG: Beta C-F (" << betaCIndex << " -> " << x.pivBetaC(betaCIndex) << ") - ";
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                    if ( csts )
		    {
			x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			csts = 0;
		    }

		    x.modBetaCtoFhpzero(betaCIndex,Gp,Gp,Gn,Gpn,gp,gn);

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "\n";
                    //DEBUGCATCH;
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		}
	    }

	    else
	    {
//		if ( fixHigherOrderTerms )
//		{
//		    if ( x.aNF() )
//		    {
//			isopt = ( maxabs(x.alphaGrad()(x.pivAlphaF()),iP) > x.opttol() ) ? 0 : 1;
//		    }
//
//		    else
//		    {
//			isopt = 1;
//		    }
//		}
//
//		else
		{
                    if ( ( steptype == 1 ) || !(x.aNF()) )
                    {
                        isopt = 1;
                    }

                    else
                    {
			isopt = ( maxabs(x.alphaGrad()(x.pivAlphaF(),tmpva),iP) > x.opttol() ) ? 0 : 1;
                    }
		}

		if ( isopt )
		{
		    // If step unscaled: test optimality

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
                    errstream() << "DEBUG: Test optimality - ";
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                    if ( csts )
		    {
			x.updateGradOpt(combStepAlpha(startPivAlphaF,tmpva),combStepBeta(x.pivBetaF(),tmpvb),startAlphaGrad,FnF,startPivAlphaF,Gp,Gpn);

			csts = 0;
		    }

		    isopt = x.maxGradNonOpthpzero(alphaCIndex,betaCIndex,stateChange,gradmag,Gp,Gn,Gpn,gp,gn);

                    // ..................................................................
                    // ..................................................................
                    // ..................................................................
                    #ifdef DEBUGOPT
		    if ( !isopt )
		    {
                        errstream() << "max gradient error is " << gradmag << " for " << alphaCIndex << ", " << betaCIndex << ", " << stateChange << " - ";
		    }
                    errstream() << "\n";
                    //DEBUGCATCH;
                    #endif
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		    if ( !isopt && ( steptype != 2 ) )
		    {
			if ( ( alphaCIndex == -1 ) && ( betaCIndex == -1 ) )
			{
			    // Non-feasible, unable to proceed.
                            // Trigger external presolver

			    res = -1;
			    goto getout;
			}

			else if ( ( stalledflag >= 3 ) && !(x.aNC()) )
			{
			    // Stalled with no constrained variables and not optimal?
			    // Not actually possible.

			    isopt = 1;
			}

			else if ( stalledflag >= 3 )
			{
			    // Stalled for some reason.  Start randomly freeing variables
			    // to break the impasse

			    int conNLB = x.aNLB();
			    int conNZ  = x.aNZ();
			    int conNUB = x.aNUB();

                            NiceAssert( conNLB+conNZ+conNUB );

			tryagain:
			    //int randfreechoice = svm_rand()%(conNLB+conNZ+conNUB);
			    int randfreechoice = rand()%(conNLB+conNZ+conNUB);

			    if ( randfreechoice < conNLB )
			    {
				if ( Gp.v(x.pivAlphaLB(randfreechoice),x.pivAlphaLB(randfreechoice)) < x.zerotol() )
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "DEBUG: impasse breaker LB -> LF " << randfreechoice << " - ";
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                                x.modAlphaLBtoLFhpzero(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn);

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "\n";
                                //DEBUGCATCH;
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			    }

			    else if ( randfreechoice < conNLB+conNZ )
			    {
				randfreechoice -= conNLB;

				if ( Gp.v(x.pivAlphaZ(randfreechoice),x.pivAlphaZ(randfreechoice)) < x.zerotol() )
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}

				if ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 2 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> LF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    x.modAlphaZtoLFhpzero(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 1 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> UF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    x.modAlphaZtoUFhpzero(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 0 ) && ( rand()%2 ) )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> LF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    x.modAlphaZtoLFhpzero(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( x.alphaRestrict()(x.pivAlphaZ(randfreechoice)) == 0 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: impasse breaker Z -> UF " << randfreechoice << " - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    x.modAlphaZtoUFhpzero(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}
			    }

			    else
			    {
				randfreechoice -= (conNLB+conNZ);

				if ( Gp.v(x.pivAlphaUB(randfreechoice),x.pivAlphaUB(randfreechoice)) < x.zerotol() )
				{
                                    NiceAssert( conNLB+conNUB+conNZ > 1 );

				    goto tryagain;
				}

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "DEBUG: impasse breaker UB -> UF " << randfreechoice << " - ";
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                                x.modAlphaUBtoUFhpzero(randfreechoice,Gp,Gp,Gn,Gpn,gp,gn);

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "\n";
                                //DEBUGCATCH;
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			    }
			}

			else
			{
			    // If non-optimal: free variables

			    if ( alphaCIndex >= 0 )
			    {
				if ( stateChange == -2 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha LB-LF (" << alphaCIndex << " -> " << x.pivAlphaLB(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                                    x.modAlphaLBtoLFhpzero(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( stateChange == -1 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha Z-LF (" << alphaCIndex << " -> " << x.pivAlphaZ(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#ifdef DEBUGOPT
try {
#endif
				    x.modAlphaZtoLFhpzero(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn);
#ifdef DEBUGOPT
} catch (...)
{
errstream() << "DEBUG: bad error detected - " << x << "\n";
exit(1);
}
#endif

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else if ( stateChange == +1 )
				{
                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha Z-UF (" << alphaCIndex << " -> " << x.pivAlphaZ(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

				    x.modAlphaZtoUFhpzero(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

				else
				{
                                    NiceAssert( stateChange == +2 );

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "DEBUG: Alpha UB-UF (" << alphaCIndex << " -> " << x.pivAlphaUB(alphaCIndex) << ") - ";
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                                    x.modAlphaUBtoUFhpzero(alphaCIndex,Gp,Gp,Gn,Gpn,gp,gn);

                                    // ..................................................................
                                    // ..................................................................
                                    // ..................................................................
                                    #ifdef DEBUGOPT
                                    errstream() << "\n";
                                    //DEBUGCATCH;
                                    #endif
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}
			    }

			    else
			    {
                                NiceAssert( betaCIndex >= 0 );

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "DEBUG: Beta C to F (" << betaCIndex << " -> " << x.pivBetaC(betaCIndex) << ") - ";
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                                x.modBetaCtoFhpzero(betaCIndex,Gp,Gp,Gn,Gpn,gp,gn);

                                // ..................................................................
                                // ..................................................................
                                // ..................................................................
                                #ifdef DEBUGOPT
                                errstream() << "\n";
                                //DEBUGCATCH;
                                #endif
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			    }
			}
		    }
		}
	    }

            sc.testopt(itcnt,isopt,timeout,bailout,stopout,kickstart,FEEDBACK_CYCLE,MAJOR_FEEDBACK_CYCLE,start_time,curr_time,"quadratic (sQsLsAsWs) optimisation",uservars,varnames,vardescr);
	}

        // ..................................................................
        // ..................................................................
        // ..................................................................
        #ifdef DEBUGOPT
        errstream() << "DEBUG: Optimisation complete\n";
        DEBUGCATCHEND
        errstream() << "DEBUG: final state - " << x << "\n";
        #endif
        #ifdef DEBUGOPT
        errstream() << "DEBUG: Start mod to zero\n";
        #endif
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        //if ( x.aNF() )
        {
            for ( int iP = x.aNF()-1 ; iP >= 0 ; --iP )
            {
                if ( !(x.alphaRestrict(x.pivAlphaF(iP))) )
                {
                    if ( ( x.alphaState(x.pivAlphaF(iP)) == -1 ) && ( x.alpha(x.pivAlphaF(iP)) > 0 ) )
                    {
                        x.modAlphaLFtoUFhpzero(iP,Gp,Gn,Gpn,gp,gn);
                    }

		    else if ( ( x.alphaState(x.pivAlphaF(iP)) == +1 ) && ( x.alpha(x.pivAlphaF(iP)) < 0 ) )
		    {
			x.modAlphaUFtoLFhpzero(iP,Gp,Gn,Gpn,gp,gn);
		    }
		}
	    }
	}

        // ..................................................................
        // ..................................................................
        // ..................................................................
        #ifdef DEBUGOPT
        errstream() << "DEBUG: End mod to zero\n";
        #endif
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	//if ( isopt )
	//{
	//    int i;
        //
	    // Constrain alphas at bounds but left hanging
	    //
	    // BUGFIX: this is a bad idea!  In the presolver, the conditions here != the conditions
            // in optstate, and it introduces some serious numerical problems!
            //
	    //if ( x.aNF() )
	    //{
	    //    for ( i = x.aNF()-1 ; i >= 0 ; --i )
	    //    {
	    //        if ( x.alphaRestrict(x.pivAlphaF()(i)) == 0 )
	    //        {
	    //    	if ( x.alphaState()(x.pivAlphaF()(i)) == -1 )
	    //    	{
            //
            //
            //
            //
            //
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) <= lb.v(i)+(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaLFtoLBhpzero(i,Gp,Gp,Gn,Gpn,gp,gn,lb);
	    //    	    }
	    //    	}
            //
	    //    	else if ( x.alphaState()(x.pivAlphaF()(i)) == +1 )
	    //    	{
            //
            //
            //
            //
            //
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) >= ub.v(i)-(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaUFtoUBhpzero(i,Gp,Gp,Gn,Gpn,gp,gn,ub);
	    //    	    }
	    //    	}
	    //        }
            //
	    //        else if ( x.alphaRestrict(x.pivAlphaF()(i)) == 1 )
	    //        {
	    //    	if ( x.alphaState()(x.pivAlphaF()(i)) == +1 )
	    //    	{
            //
            //
            //
            //
            //
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) >= ub.v(i)-(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaUFtoUBhpzero(i,Gp,Gp,Gn,Gpn,gp,gn,ub);
	    //    	    }
	    //    	}
	    //        }
            //
	    //        else if ( x.alphaRestrict(x.pivAlphaF()(i)) == 2 )
	    //        {
	    //    	if ( x.alphaState()(x.pivAlphaF()(i)) == -1 )
	    //    	{
            //
            //
            //
            //
            //
	    //    	    if ( x.alpha()(x.pivAlphaF()(i)) <= lb.v(i)+(x.zerotol()) )
	    //    	    {
	    //    		x.modAlphaLFtoLBhpzero(i,Gp,Gp,Gn,Gpn,gp,gn,lb);
	    //    	    }
	    //    	}
	    //        }
	    //    }
	    //}
//#ifdef DEBUGOPT
//errstream() << x << "\n";
//#endif
	//}

	if ( !isopt )
	{
	    res = 1;
	}

        if ( bailout )
        {
            res = 10*bailout;
        }
    }

getout:
        // ..................................................................
        // ..................................................................
        // ..................................................................
        #ifdef DEBUGOPT
        errstream() << "DEBUG: Post-optimisation complete\n";
        DEBUGCATCHEND
        errstream() << "DEBUG: fiinal state - " << x << "\n";
        #endif
        #ifdef DEBUGOPT
        errstream() << "DEBUG: Return to caller!\n";
        #endif

    return res;
}





































