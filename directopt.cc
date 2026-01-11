
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
              Vector<Vector<gentype> > &allxres,
              Vector<gentype>          &allfres,
              Vector<Vector<gentype> > &allcres,
              Vector<gentype>          &allsres,
              const Vector<gentype> &xmin,
              const Vector<gentype> &xmax,
              void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
              void *fnarg,
              const DIRectOptions &dopts,
              svmvolatile int &killSwitch);





double tst_obj(int n, const double *x, int *resflag, void *fnarginnerdr);
double tst_obj(int n, const double *x, int *resflag, void *fnarginnerdr)
{
    (void) n;

    void **fnarginner = (void **) fnarginnerdr;

    double (*fn)(int, const double *, void *) = (double (*)(int, const double *, void *)) fnarginner[0];
    void *fnarg = fnarginner[1];

    int            &killSwitch = *((int            *) (((void **) fnarginnerdr)[2]));
    double         &hardmin    = *((double         *) (((void **) fnarginnerdr)[3]));
    Vector<double> &l          = *((Vector<double> *) (((void **) fnarginnerdr)[4]));
    Vector<double> &u          = *((Vector<double> *) (((void **) fnarginnerdr)[5]));
    double         *xx         =  ((double         *) (((void **) fnarginnerdr)[6]));
    int            &dim        = *((int            *) (((void **) fnarginnerdr)[7]));
    int            &effdim     = *((int            *) (((void **) fnarginnerdr)[8]));
    double         &hardmax    = *((double         *) (((void **) fnarginnerdr)[9]));
    double         *xres       =  ((double         *) (((void **) fnarginnerdr)[10]));
    double         &yres       = *((double         *) (((void **) fnarginnerdr)[11]));
    int            &ires       = *((int            *) (((void **) fnarginnerdr)[12]));
    int            &icnt       = *((int            *) (((void **) fnarginnerdr)[13]));
    double         *jitter     =  ((double         *) (((void **) fnarginnerdr)[14]));
    int            &feasrescnt = *((int            *) (((void **) fnarginnerdr)[15]));

    NiceAssert( n == effdim );

    int i;

    for ( i = 0 ; i < dim ; ++i )
    {
        if ( i < effdim )
        {
            xx[i] = (x[effdim-1-i]*(u(i)-l(i)))+l(i)+jitter[i];

            if ( xx[i] >= u(i) ) { xx[i] = u(i); }
            if ( xx[i] <= l(i) ) { xx[i] = l(i); }
        }

        else
        {
            xx[i] = l(i);
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

    if ( std::isnan(res) )
    {
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

            ires = icnt;
            yres = res;

            for ( i = 0 ; i < dim ; ++i )
            {
                xres[i] = x[i];
            }
        }

        if ( res <= hardmin )
        {
            // Trigger early termination if hardmin reached

            killSwitch = 1;
        }

        else if ( res >= hardmax )
        {
            // Trigger early termination if hardmax reached

            killSwitch = 1;
        }
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
    int i;

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

    for ( i = 0 ; i < dim ; ++i )
    {
        if ( xmin(i) < xmax(i) )
        {
            effdim = i+1;
        }
    }

    NiceAssert( effdim > 0 );

    long int maxits          = dopts.maxits;
    long int maxevals        = dopts.maxevals;
    double eps               = dopts.eps;
    double maxtraintime      = dopts.maxtraintime;
    double traintimeoverride = dopts.traintimeoverride;
    double hardmin           = dopts.hardmin;
    double hardmax           = dopts.hardmax;

    Xres.resize(dim);

    double *x = &Xres("&",0);
    double *l;
    double *u;
    double *jitter;
    double *xx;
    double magic_eps_abs = 1e-4; //0;
    double volume_reltol = 0.0;
    double sigma_reltol = -1.0;

    MEMNEWARRAY(l,     double,dim);
    MEMNEWARRAY(u,     double,dim);
    MEMNEWARRAY(xx,    double,dim);
    MEMNEWARRAY(jitter,double,dim);

    for ( i = 0 ; i < dim ; ++i )
    {
        l[i] = 0;
        u[i] = +1;

        jitter[i] = 0;

        if ( jitterbound )
        {
            randnfill(jitter[i],0.0,JITTERVAR);
        }
    }

    int feasrescnt = 0;

    void *fnarginner[16];

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

    double *xresalt;
    double  yresalt = 0.0;
    int     iresalt = -1;
    int     itcnt   = 0;

    MEMNEWARRAY(xresalt,double,dim);

    fnarginner[10] = (void *)  xresalt;
    fnarginner[11] = (void *) &yresalt;
    fnarginner[12] = (void *) &iresalt;
    fnarginner[13] = (void *) &itcnt;
    fnarginner[14] = (void *)  jitter;
    fnarginner[15] = (void *) &feasrescnt;

    void *fnarginnerdr = (void *) fnarginner;

    //errstream() << "DIRect Optimisation Initiated:\n";

    // Note use of effdim here!

    int intres = direct_optimize(tst_obj,fnarginnerdr,
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

    // Need to re-scale result (and grab from xaltres - that is,
    // the best result as detected by the callback, not what
    // direct might happen to return if it gets lost.

    for ( i = 0 ; i < dim ; ++i )
    {
        if ( i < effdim )
        {
//            x[i] = (x[i]*(xmax(i)-xmin(i)))+xmin(i);
            x[i] = feasrescnt ? ((xresalt[effdim-1-i]*(xmax(i)-xmin(i)))+xmin(i)+jitter[i]) : valvnan("No solution found");

            if ( x[i] >= xmax(i) ) { x[i] = xmax(i); }
            if ( x[i] <= xmin(i) ) { x[i] = xmin(i); }
        }

        else
        {
            x[i] = xmin(i);
        }
    }

    if ( feasrescnt )
    {
        fres.force_double() = yresalt;
    }

    else
    {
        fres.force_null();
    }

    MEMDELARRAY(l);  l  = nullptr;
    MEMDELARRAY(u);  u  = nullptr;
    MEMDELARRAY(xx); xx = nullptr;

    MEMDELARRAY(xresalt); xresalt = nullptr;
    MEMDELARRAY(jitter ); jitter  = nullptr;

    //errstream() << "DIRect Optimisation Ended\n";

    return feasrescnt ? intres : -200;
}


double fninnerd(int dim, const double *x, void *arg);
double fninnerd(int dim, const double *x, void *arg)
{
    Vector<gentype> &xx                                    = *((Vector<gentype> *)          (((void **) arg)[0]));
    void (*fn)(gentype &res, Vector<gentype> &, void *arg) = ( (void (*)(gentype &, Vector<gentype> &, void *arg))  (((void **) arg)[1]) );
    void *arginner                                         = ((void *)                      (((void **) arg)[2]));
    int &ires                                              = *((int *)                      (((void **) arg)[3]));
    Vector<Vector<gentype> > &allxres                      = *((Vector<Vector<gentype> > *) (((void **) arg)[4]));
    Vector<gentype> &allfres                               = *((Vector<gentype> *)          (((void **) arg)[5]));
    Vector<Vector<gentype> > &allcres                      = *((Vector<Vector<gentype> > *) (((void **) arg)[6]));
    gentype &fres                                          = *((gentype *)                  (((void **) arg)[7]));
    Vector<gentype> &xres                                  = *((Vector<gentype> *)          (((void **) arg)[8]));
    gentype &tempres                                       = *((gentype *)                  (((void **) arg)[9]));
    svmvolatile int &killSwitch                            = *((int *)                      (((void **) arg)[10]));

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; ++i )
        {
            xx("&",i) = x[i];
        }
    }

    tempres.force_int() = 0;
    (*fn)(tempres,xx,arginner);
errstream() << "\n";

// Return value: 0 if simple observation of f(x), stopflag may be set
//               1 if simple observation of f(x) and c(x) that is feasible, stopflag may be set
//               2 non-trivial observation
//

    int stopflags = 0;
    int intres = readres(tempres,stopflags);

    StrucAssert( intres != 2 );

    killSwitch = stopflags;

    if ( !intres && ( ( allfres.size() == 0 ) || ( tempres < fres ) ) )
    {
        ires = allfres.size();
        fres = tempres;
        xres = xx;
    }

    if ( !intres )
    {
        Vector<gentype> tempcgt;

        allfres.append(allfres.size(),tempres);
        allcres.append(allcres.size(),tempcgt);
        allxres.append(allxres.size(),xx);
    }

    return !intres ? ((double) tempres) : valvnan("null returned to indicate not a result");
}

int directOpt(int dim,
              Vector<gentype> &xres,
              gentype         &fres,
              int             &ires,
              Vector<Vector<gentype> > &allxres,
              Vector<gentype>          &allfres,
              Vector<Vector<gentype> > &allcres,
              Vector<gentype>          &allsres,
              const Vector<gentype> &xmin,
              const Vector<gentype> &xmax,
              void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
              void *fnarg,
              const DIRectOptions &dopts,
              svmvolatile int &killSwitch)
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
    gentype locfres(0.0);

    for ( i = 0 ; i < dim ; ++i )
    {
        locxmin("&",i) = (double) xmin(i);
        locxmax("&",i) = (double) xmax(i);
    }

    void *fnarginner[11];
    Vector<gentype> xx(dim);
    gentype tempres;

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

    int res = directOpt(dim,locxres,locfres,locxmin,locxmax,fninnerd,(void *) fnarginner,dopts,killSwitch);

    allsres.resize(allxres.size());
    gentype dummynull;
    dummynull.force_null();
    allsres = dummynull;

//    xres.resize(dim);
//
//    for ( i = 0 ; i < dim ; ++i )
//    {
//        xres("&",i) = locxres(i);
//    }

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
                      Vector<Vector<gentype> > &allxres,
                      Vector<gentype>          &allfres,
                      Vector<Vector<gentype> > &allcres,
                      Vector<gentype>          &allFres,
                      Vector<gentype>          &allmres,
                      Vector<gentype>          &allsres,
                      Vector<double>           &s_score,
                      Vector<int>              &is_feas,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
{
    (void) mInd;
    (void) sres;

    int res = directOpt(dim,Xres,fres,ires,allxres,allfres,allcres,allsres,
                        xmin,xmax,fn,fnarg,*this,killSwitch);

    allmres = allfres;

    mres = fres;
    cres.resize(0);
    Fres = ires;

    allFres.resize(allfres.size());

    for ( int i = 0 ; i < allFres.size() ; ++i )
    {
        allFres("&",i) = i;
    }

    s_score.resize(allfres.size());
    s_score = 1.0;

    is_feas.resize(allfres.size());
    is_feas = 1;

    return res;
}
