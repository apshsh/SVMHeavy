
//
// DIRect global optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "directopt.hpp"
#include <math.h>
#include <stddef.h>
#include <iostream>


int directOpt(int dim,
              Vector<gentype> &Xres,
              gentype         &fres,
              int             &ires,
              Vector<Vector<gentype>> &allxres,
              Vector<gentype>          &allfres,
              Vector<Vector<gentype>> &allcres,
              Vector<gentype>         &allsres,
              const Vector<gentype> &xmin,
              const Vector<gentype> &xmax,
              void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
              void *fnarg,
              const DIRectOptions &dopts,
              svmvolatile int &killSwitch,
              const ML_Base *gridsource);





double tst_obj(int n, const double *x, int *resflag, void *fnarginnerdr);
double tst_obj(int n, const double *x, int *resflag, void *fnarginnerdr)
{
    (void) n;

    void **fnarginner = (void **) fnarginnerdr;

    double (*fn)(int, const double *, void *) = (double (*)(int, const double *, void *)) fnarginner[0];
    void *fnarg = fnarginner[1];

    int            &killSwitch = *((int            *) (((void **) fnarginnerdr)[2]));
    double         &hardmin    = *((double         *) (((void **) fnarginnerdr)[3]));
    Vector<double> &xmin       = *((Vector<double> *) (((void **) fnarginnerdr)[4]));
    Vector<double> &xmax       = *((Vector<double> *) (((void **) fnarginnerdr)[5]));
    double         *xx         =  ((double         *) (((void **) fnarginnerdr)[6]));
    int            &dim        = *((int            *) (((void **) fnarginnerdr)[7]));
    int            &effdim     = *((int            *) (((void **) fnarginnerdr)[8]));
    double         &hardmax    = *((double         *) (((void **) fnarginnerdr)[9]));
    double         *xxres      =  ((double         *) (((void **) fnarginnerdr)[10]));
    double         &yres       = *((double         *) (((void **) fnarginnerdr)[11]));
    int            &ires       = *((int            *) (((void **) fnarginnerdr)[12]));
    int            &icnt       = *((int            *) (((void **) fnarginnerdr)[13]));
    double         *jitter     =  ((double         *) (((void **) fnarginnerdr)[14]));
    int            &feasrescnt = *((int            *) (((void **) fnarginnerdr)[15]));
    double         &yminfnd    = *((double         *) (((void **) fnarginnerdr)[16]));
    double         &ymaxfnd    = *((double         *) (((void **) fnarginnerdr)[17]));

    (void) effdim;

    NiceAssert( n == effdim );

    int i,ii;

    for ( i = 0, ii = 0 ; i < dim ; ++i )
    {
        if ( xmin(i) < xmax(i) )
        {
            if ( jitter[ii] == 0 )
            {
                xx[i] = x[ii];
            }

            else if ( xx[i] <= 0.5 )
            {
                xx[i] = ((x[ii]-0.5)*(1+jitter[ii])) + (0.5*(1+jitter[ii]));
//0:   -0.5*1.1 + 0.5*1.1 = 0
//0.5:            0.5*1.1 = 0.55
            }

            else
            {
                xx[i] = ((x[ii]-0.5)*(1-jitter[ii])) + (0.5*(1+jitter[ii]));

//0.5:            0.5*1.1 = 0.55
//1:    0.5*0.9 + 0.5*1.1 = 1
            }

            xx[i] = (xx[i]*(xmax(i)-xmin(i)))+xmin(i);

            if ( xx[i] >= xmax(i) ) { xx[i] = xmax(i); }
            if ( xx[i] <= xmin(i) ) { xx[i] = xmin(i); }

            ++ii;
        }

        else
        {
            xx[i] = xmin(i);
        }
    }

    double res = (*fn)(dim,xx,fnarg);

    // Set resflag. As per nlopt_direct.cc:
    //
    // resflag: 1 if infeasible
    //          -1 if bad setup??? (whatever that means)
    //          0 if feasible
    //
    // resflag is preset to 0, but we use NaN res to
    // indicate (passback) infeasibility. We translate
    // that here.

    if ( std::isnan(res) || ( res >= NANTRIGTEST ) )
    {
        res = HUGE_VAL;
        *resflag = 1;
    }

    else
    {
        ++feasrescnt;

        if ( ( ires == -1 ) || ( res < yres ) )
        {
            // Subtle point: if directopt bails (or detects optimization
            // error) it will return any old result. This bit of code saves
            // the best test result so far so we can return it instead.

            if ( ires == -1 )
            {
                yminfnd = res;
                ymaxfnd = res;
            }

            ires = icnt;
            yres = res;

            for ( i = 0 ; i < dim ; ++i ) { xxres[i] = xx[i]; }
        }

        if ( res < yminfnd ) { yminfnd = res; }
        if ( res > ymaxfnd ) { ymaxfnd = res; }

        if      ( res <= hardmin ) { /* Trigger early termination if hardmin reached */ killSwitch = 1; }
        else if ( res >= hardmax ) { /* Trigger early termination if hardmax reached */ killSwitch = 1; }
    }

    ++icnt;

    return res;
}



int directOpt(int dim,
              Vector<double> &Xres,
              gentype        &fres,
              const Vector<double> &xmin,
              const Vector<double> &xmax,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const DIRectOptions &dopts,
              svmvolatile int &killSwitch,
              bool jitterbound)
{
    int i,ii;

    // WARNING: THERE IS A SERIOUS BUG IN THE NL_OPT VERSION OF DIRECT USED HERE!
    //
    // When the range [l,u] is anything other than [0,1] there is a good chance that the
    // result x will lie outside of [l,u].  For example I was using [-0.5,0.5] and found
    // that *if* the error code was -4 (that is, NLOPT_ROUNDOFF_LIMITED, which is supposed
    // to be a non-serious bug probably giving a typically useful result) x = 0.83333 was
    // a common result!  I suspect that the range is normalised to [0,1] inside of DIRect,
    // but return code -4 bypasses some sort of range correction to give the erroneous
    // result.
    //
    // WORKAROUND: give DIRect bounds [0,1] and re-scale for callback to *fn and when
    // the result is returned.

    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );
    NiceAssert( xmax >= xmin );

    int effdim = 0;

    for ( i = 0 ; i < dim ; ++i ) { if ( xmin(i) < xmax(i) ) { ++effdim; } }

    long int maxits          = dopts.maxits;
    long int maxevals        = (dopts.maxevals)*dim;
    double eps               = dopts.eps;
    double maxtraintime      = dopts.maxtraintime;
    double traintimeoverride = dopts.traintimeoverride;
    double hardmin           = dopts.hardmin;
    double hardmax           = dopts.hardmax;

    Xres.resize(dim);

    double *x;
    double *l;
    double *u;
    double *jitter;
    double *xx;
    double magic_eps_abs = 1e-4; //0;
    double volume_reltol = 0.0;
    double sigma_reltol  = -1.0;

    MEMNEWARRAY(x,     double,effdim+1); // effdim might be 0
    MEMNEWARRAY(l,     double,effdim+1);
    MEMNEWARRAY(u,     double,effdim+1);
    MEMNEWARRAY(xx,    double,dim+1);    // dim might be 0
    MEMNEWARRAY(jitter,double,effdim+1);

    for ( ii = 0 ; ii < effdim ; ++ii )
    {
        l[ii] = 0;
        u[ii] = 1;

        if ( jitterbound ) { randnfill(jitter[ii],0.0,JITTERVAR); }
        else               { jitter[ii] = 0;                      }
    }

    int feasrescnt  = 0;

    double yminfnd = 0;
    double ymaxfnd = 0;

    void *fnarginner[18];

    fnarginner[0] = (void *)  fn;
    fnarginner[1] = (void *)  fnarg;
    fnarginner[2] = (void *) &killSwitch;
    fnarginner[3] = (void *) &hardmin;
    fnarginner[4] = (void *) &xmin;
    fnarginner[5] = (void *) &xmax;
    fnarginner[6] = (void *)  xx;
    fnarginner[7] = (void *) &dim;
    fnarginner[8] = (void *) &effdim;
    fnarginner[9] = (void *) &hardmax;

    double *xxresalt;
    double  yresalt = 0.0;
    int     iresalt = -1;
    int     itcnt   = 0;

    MEMNEWARRAY(xxresalt,double,dim+1);

    fnarginner[10] = (void *)  xxresalt;
    fnarginner[11] = (void *) &yresalt;
    fnarginner[12] = (void *) &iresalt;
    fnarginner[13] = (void *) &itcnt;
    fnarginner[14] = (void *)  jitter;
    fnarginner[15] = (void *) &feasrescnt;
    fnarginner[16] = (void *) &yminfnd;
    fnarginner[17] = (void *) &ymaxfnd;

    void *fnarginnerdr = (void *) fnarginner;

    //errstream() << "DIRect Optimisation Initiated:\n";

    // Note use of effdim here!

    errstream() << "DIRect Optimisation Started (" << dopts.getsimname() << ").\n";

    int intres = 0;

    if ( effdim == 0 )
    {
        // dimensionless just evaluates the objective (through tst_obj to make sure everything gets stored as required) and returns

        double xdummy;
        int dummyresflag = 0;
        fres.force_double() = tst_obj(effdim,&xdummy,&dummyresflag,fnarginnerdr);
    }

    else
    {
        intres = direct_optimize(tst_obj,fnarginnerdr,
                                 effdim,
                                 l,u,
                                 x,&(fres.force_double()),
                                 static_cast<int>(maxevals),static_cast<int>(maxits),
                                 ( traintimeoverride == 0 ) ? maxtraintime : traintimeoverride,
                                 eps,magic_eps_abs,
                                 volume_reltol,sigma_reltol,
                                 killSwitch,
                                 DIRECT_UNKNOWN_FGLOBAL,
                                 0,
                                 dopts.algorithm);
    }

    // Need to re-scale result (and grab from xaltres - that is,
    // the best result as detected by the callback, not what
    // direct might happen to return if it gets lost.

//errstream() << "done xmin = " << xmin << "\n";
//errstream() << "done xmax = " << xmax << "\n";
//errstream() << "done feasrescnt = " << feasrescnt << "\n";
//errstream() << "done yminfnd = " << yminfnd << "\n";
//errstream() << "done ymaxfnd = " << ymaxfnd << "\n";
    for ( i = 0 ; i < dim ; ++i )
    {
//errstream() << "xxresalt[" << i << "] = " << xxresalt[i] << "\n";
        if ( xmin(i) < xmax(i) ) { Xres("&",i) = feasrescnt ? xxresalt[i] : valvnan("No solution found"); }
        else                     { Xres("&",i) = xmin(i);                                                 }
    }
//errstream() << "done Xres = " << Xres << "\n";

    if ( feasrescnt ) { fres.force_double() = yresalt; }
    else              { fres.force_null();             }

    MEMDELARRAY(xxresalt); xxresalt = nullptr;
    MEMDELARRAY(jitter);   jitter   = nullptr;
    MEMDELARRAY(xx);       xx       = nullptr;
    MEMDELARRAY(u);        u        = nullptr;
    MEMDELARRAY(l);        l        = nullptr;
    MEMDELARRAY(x);        x        = nullptr;

    errstream() << "DIRect Optimisation Ended: " << yminfnd << "\t" << ymaxfnd << "\n";

    return feasrescnt ? ( ( ymaxfnd-yminfnd > 1e-6 ) ? intres : -300 ) : -200;
}


double fninnerd(int dim, const double *x, void *arg);
double fninnerd(int dim, const double *x, void *arg)
{
    Vector<gentype> &xx                                    = *((Vector<gentype> *)          (((void **) arg)[0]));
    void (*fn)(gentype &res, Vector<gentype> &, void *arg) = ( (void (*)(gentype &, Vector<gentype> &, void *arg))  (((void **) arg)[1]) );
    void *arginner                                         = ((void *)                      (((void **) arg)[2]));
    int &ires                                              = *((int *)                      (((void **) arg)[3]));
    Vector<Vector<gentype>> &allxres                       = *((Vector<Vector<gentype>> *)  (((void **) arg)[4]));
    Vector<gentype> &allfres                               = *((Vector<gentype> *)          (((void **) arg)[5]));
    Vector<Vector<gentype>> &allcres                       = *((Vector<Vector<gentype>> *)  (((void **) arg)[6]));
    gentype &fres                                          = *((gentype *)                  (((void **) arg)[7]));
    Vector<gentype> &xres                                  = *((Vector<gentype> *)          (((void **) arg)[8]));
    gentype &tempres                                       = *((gentype *)                  (((void **) arg)[9]));
    svmvolatile int &killSwitch                            = *((int *)                      (((void **) arg)[10]));
    int &feascnt                                           = *((int *)                      (((void **) arg)[11]));

    if ( dim ) { for ( int i = 0 ; i < dim ; ++i ) { xx("&",i) = x[i]; } }

    tempres.force_int() = 0;
    (*fn)(tempres,xx,arginner);
errstream() << "\n";

// Return value: 0 if simple observation of f(x), stopflag may be set
//               1 if simple observation of f(x) and c(x) that is feasible, stopflag may be set
//               2 non-trivial observation
//

    Vector<gentype> ycgt(0);

    int stopflags = 0;
    int intres = readres(tempres,ycgt,stopflags);

    StrucAssert( intres != 2 );

    killSwitch = stopflags;

//    if ( !intres && ( ( allfres.size() == 0 ) || ( tempres < fres ) ) )
    if ( !intres && ( !feascnt || ( tempres < fres ) ) )
    {
        ires = allfres.size();
        fres = tempres;
        xres = xx;
outstream() << "***";
    }
outstream() << "\n";

    if ( !intres )
    {
        feascnt++;
    }

    {
        allfres.append(allfres.size(),tempres);
        allcres.append(allcres.size(),ycgt);
        allxres.append(allxres.size(),xx);
    }

    return !intres ? ((double) tempres) : NANTRIGSET; // valvnan("null returned to indicate not a result");
}

int directOpt(int dim,
              Vector<gentype> &xres,
              gentype         &fres,
              int             &ires,
              Vector<Vector<gentype>> &allxres,
              Vector<gentype>         &allfres,
              Vector<Vector<gentype>> &allcres,
              Vector<gentype>         &allsres,
              const Vector<gentype> &xmin,
              const Vector<gentype> &xmax,
              void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
              void *fnarg,
              const DIRectOptions &dopts,
              svmvolatile int &killSwitch,
              const ML_Base *gridsource)
{
    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );

    allxres.resize(0);
    allfres.resize(0);
    allcres.resize(0);
    allsres.resize(0);

    Vector<double> locxres;

    Vector<double> locxmin(dim);
    Vector<double> locxmax(dim);

    int i;
    gentype locfres(toGentype());

    for ( i = 0 ; i < dim ; ++i )
    {
        locxmin("&",i) = (double) xmin(i);
        locxmax("&",i) = (double) xmax(i);
    }

    void *fnarginner[12];
    Vector<gentype> xx(dim);
    gentype tempres;
    int feascnt = 0;

    fnarginner[0]  = (void *) &xx;
    fnarginner[1]  = (void *) fn;
    fnarginner[2]  = (void *) fnarg;
    fnarginner[3]  = (void *) &ires;
    fnarginner[4]  = (void *) &allxres;
    fnarginner[5]  = (void *) &allfres;
    fnarginner[6]  = (void *) &allcres;
    fnarginner[7]  = (void *) &fres;
    fnarginner[8]  = (void *) &xres;
    fnarginner[9]  = (void *) &tempres;
    fnarginner[10] = (void *) &killSwitch;
    fnarginner[11] = (void *) &feascnt;

    int res = -200;

    if ( !gridsource )
    {
        res = directOpt(dim,locxres,locfres,locxmin,locxmax,fninnerd,(void *) fnarginner,dopts,killSwitch);
    }

    else
    {
        for ( int i = 0 ; i < (*gridsource).N() ; ++i )
        {
            gentype yyval = (*gridsource).y()(i);         // assume a null observation
            SparseVector<gentype> xx((*gridsource).x(i)); // assume x is defined

            StrucAssert( yyval.isValNull() );
            StrucAssert( (*gridsource).d()(i) == 2 );

            int effdim = 0;

            for ( int j = 0 ; j < dim ; ++j )
            {
                if ( xx(j).isValNull() ) { locxmin("&",j) = (double) xmin(j); locxmax("&",j) = (double) xmax(j); ++effdim; }
                else                     { locxmin("&",j) = (double) xx(j);   locxmax("&",j) = (double) xx(j);             }
            }

            if ( locfres.isValNull() )
            {
                res = directOpt(dim,locxres,locfres,locxmin,locxmax,fninnerd,(void *) fnarginner,dopts,killSwitch);
            }

            else
            {
                gentype limpfres(toGentype());
                Vector<double> limpxres;

                int limpres = directOpt(dim,limpxres,limpfres,locxmin,locxmax,fninnerd,(void *) fnarginner,dopts,killSwitch);

                if ( !limpfres.isValNull() && ( limpfres < locfres ) )
                {
                    locfres = limpfres;
                    locxres = limpxres;
                    res     = limpres;
                }
            }
        }
    }

    allsres.resize(allxres.size()) = toGentype();

    // for consistency with bayesopt
    fres.negate();
    allfres.negate();

    return res;
}







int DIRectOptions::optim(int dim,
                      Vector<gentype> &Xres,
                      gentype         &fres,
                      Vector<gentype> &cres,
                      gentype         &Fres,
                      gentype         &mres,
                      gentype         &sres,
                      int             &ires,
                      int             &mInd,
                      Vector<Vector<gentype>> &allxres,
                      Vector<gentype>         &allfres,
                      Vector<Vector<gentype>> &allcres,
                      Vector<gentype>         &allFres,
                      Vector<gentype>         &allmres,
                      Vector<gentype>         &allsres,
                      Vector<double>          &s_score,
                      Vector<int>             &is_feas,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
{
    (void) mInd;
    (void) sres;

    int res = directOpt(dim,Xres,fres,ires,allxres,allfres,allcres,allsres,
                        xmin,xmax,fn,fnarg,*this,killSwitch,gridsource);

    mres = fres;
    cres = allcres(ires);
    Fres = ires;

    allmres = allfres;
    allFres.resize(allfres.size()); for ( int i = 0 ; i < allfres.size() ; ++i ) { allFres("&",i) = i; }

    s_score.resize(allfres.size()) = 1.0;
//    is_feas.resize(allfres.size()) = 1;
    is_feas.resize(allcres.size()); for ( int i = 0 ; i < allcres.size() ; ++i ) { is_feas("&",i) = ( allcres(i) >= 0.0_gent ) ? 1 : 0; }

    return res;
}
