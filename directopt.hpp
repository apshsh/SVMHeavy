
//
// DIRect global optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.hpp"
#include "nlopt_direct.hpp"

#ifndef _directopt_h
#define _directopt_h

//
// Uses DIRect optimisation algorithm to find global minima of function
// fn(x): R^dim -> R, where xmin <= x <= xmax
//
//
// Return value:
//
// -1: invalid bounds u(i) <= l(i) for some i
// -2: maxfeval too big
// -3: init failed
// -4: error in creation of sample-points
// -5: error in funciton sampling
// -6: error occured when trying to add all hyperrectangles with same
//     size and function value at the center.  Either increase maxdiv or
//     use some modification of other (Jones = 1)
// 1:  maxfeval exceeded
// 2:  maxiter exceeded
// 3:  global found (where global minima was pre-specified somehow)
// 4:  voltol error: volume of hyperrectangle with minf at centre less than
//     some percent of the original hyper-rectangle
// 5:  measure of hyperrectangle with minf at center less than sigmaper
// 6:  maxtime exceeded
// -100: out of memory
// -101: invalid args
// -102: forced stop
//

// Required by Bayesopt

class DIRectOptions;

int directOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              const Vector<double> &xmin,
              const Vector<double> &xmax,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const DIRectOptions &dopts,
              svmvolatile int &killSwitch);


class DIRectOptions : public GlobalOptions
{
public:

    // maxits: maximum cube divisions (suggest 100-200)
    // maxevals: maximum number of function evaluations (suggest 500-1000)
    // eps: epsilon argument for DIRect.  Suggest 0, 1e-4, 1e-3 etc
    // algorithm: method used (either DIRECT_ORIGINAL or DIRECT_GABLONSKY).
    // traintimeoverride: 0 for standard, >0 to set train time separately for direct optimiser

    long int maxits;
    long int maxevals;
    double eps;
    direct_algorithm algorithm;
    double traintimeoverride;

    DIRectOptions() : GlobalOptions()
    {
        optname = "DIRect Optimisation";

        maxits            = 1000; //200;
        maxevals          = 5000; //1000;
        eps               = 1e-4;
        algorithm         = DIRECT_ORIGINAL; //DIRECT_GABLONSKY;
        traintimeoverride = 0;
    }

    DIRectOptions(const DIRectOptions &src) : GlobalOptions(src)
    {
        *this = src;
    }

    DIRectOptions &operator=(const DIRectOptions &src)
    {
        GlobalOptions::operator=(src);

        maxits            = src.maxits;
        maxevals          = src.maxevals;
        eps               = src.eps;
        algorithm         = src.algorithm;
        traintimeoverride = src.traintimeoverride;

        return *this;
    }

    // Reset function so that the next simulation can run

    virtual void reset(void) override
    {
        GlobalOptions::reset();

        return;
    }

    // Generate a copy of the relevant optimisation class.

//    virtual GlobalOptions *makeDup(void) const
//    {
//        DIRectOptions *newver;
//
//        MEMNEW(newver,DIRectOptions(*this));
//
//        return newver;
//    }

    // allsres: none

    virtual int optim(int dim,
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
                      svmvolatile int &killSwitch) override;

    virtual int optim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype         &fres,
                      Vector<gentype> &cres,
                      gentype         &Fres,
                      gentype         &mres,
                      gentype         &sres,
                      int             &ires,
                      int             &mInd,
                      Vector<Vector<Vector<gentype> > > &allxres,
                      Vector<Vector<Vector<gentype> > > &allXres,
                      Vector<Vector<gentype> >          &allfres,
                      Vector<Vector<Vector<gentype> > > &allcres,
                      Vector<Vector<gentype> >          &allFres,
                      Vector<Vector<gentype> >          &allmres,
                      Vector<Vector<gentype> >          &allsres,
                      Vector<Vector<double> >           &s_score,
                      Vector<Vector<int> >              &is_feas,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch,
                      size_t numReps,
                      gentype &meanfres, gentype &varfres,
                      gentype &meanFres, gentype &varFres,
                      gentype &meanires, gentype &varires,
                      gentype &meantres, gentype &vartres,
                      gentype &meanTres, gentype &varTres,
                      Vector<gentype> &meanallfres, Vector<gentype> &varallfres,
                      Vector<gentype> &meanallFres, Vector<gentype> &varallFres,
                      Vector<gentype> &meanallmres, Vector<gentype> &varallmres) override
    {
        return GlobalOptions::optim(dim,xres,Xres,fres,cres,Fres,mres,sres,ires,mInd,allxres,allXres,allfres,allcres,allFres,allmres,allsres,s_score,is_feas,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch,numReps,meanfres,varfres,meanFres,varFres,meanires,varires,meantres,vartres,meanTres,varTres,meanallfres,varallfres,meanallFres,varallFres,meanallmres,varallmres);
    }

    virtual int optdefed(void) override
    {
        return 2;
    }
};

#endif
