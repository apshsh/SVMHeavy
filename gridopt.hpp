
//
// Grid-based Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.hpp"

//
// Does a simple grid-search to minimise function
//

#ifndef _gridopt_h
#define _gridopt_h

class GridOptions : public GlobalOptions
{
public:

    // numZooms: number of "zoom and repeat" operations (find optimal on
    //           grid, then zoom in and repeat on smaller scale.
    // zoomFact: zoom factor in above operations (range reduced by this
    //           factor around optimal, with cutoff for grid edges).
    // numPts:   number of grid points for each "axis"

    // NB: we use the default assignment operator in the code, so if anything
    //     tricky gets added you'll need to define an assignment operator.

    int numZooms;
    double zoomFact;
    Vector<int> numPts;

    GridOptions() : GlobalOptions()
    {
        optname = "Grid Optimisation";

        numZooms = 0;
        zoomFact = 0.333333333333;
    }

    GridOptions(const GridOptions &src) : GlobalOptions(src)
    {
        *this = src;
    }

    GridOptions &operator=(const GridOptions &src)
    {
        GlobalOptions::operator=(src);

        numZooms = src.numZooms;
        zoomFact = src.zoomFact;
        numPts   = src.numPts;

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
//        GridOptions *newver;
//
//        MEMNEW(newver,GridOptions(*this));
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
        return 1;
    }
};

#endif
