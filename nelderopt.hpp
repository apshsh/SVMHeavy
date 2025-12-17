
//
// Nelder-Mead optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.hpp"

#ifndef _nelderopt_h
#define _nelderopt_h

//
// Uses Nelder-Mead optimisation to minimise a function.
// Can be warm-started.
//
// Return value:
//
// -1: generic fail
// -2: invalid args
// -3: out of memory
// -4: roundoff limited
// -5: forced stop
// 1: success
// 2: stopval reached
// 3: ftol reached
// 4: xtol reached
// 5: maxeval reached
// 6: maxtime reached
//

class NelderOptions : public GlobalOptions
{
public:

     // minf_max: maximum f value (-HUGE_VAL)
     // ftol_rel: relative tolerance of function value (0)
     // ftol_abs: absolute tolerance of function value (0)
     // xtol_rel: relative tolerance of x value (0)
     // xtol_abs: absolute tolerance of x value (0)
     // maxeval: max number of f evaluations (1000)
     // method: 0 is subplex, 1 is original Nelder-Mead

     double minf_max;
     double ftol_rel;
     double ftol_abs;
     double xtol_rel;
     double xtol_abs;
     int maxeval;
     int method;

    NelderOptions() : GlobalOptions()
    {
        optname = "Nelder-Mead Optimisation";

        minf_max = -HUGE_VAL;
        ftol_rel = 0;
        ftol_abs = 0;
        xtol_rel = 0;
        xtol_abs = 0;
        maxeval  = 1000;
        method   = 0;
    }

    NelderOptions(const NelderOptions &src) : GlobalOptions(src)
    {
        *this = src;
    }

    NelderOptions &operator=(const NelderOptions &src)
    {
        GlobalOptions::operator=(src);

        minf_max = src.minf_max;
        ftol_rel = src.ftol_rel;
        ftol_abs = src.ftol_abs;
        xtol_rel = src.xtol_rel;
        xtol_abs = src.xtol_abs;
        maxeval  = src.maxeval;
        method   = src.method;

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
//        NelderOptions *newver;
//
//        MEMNEW(newver,NelderOptions(*this));
//
//        return newver;
//    }

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
