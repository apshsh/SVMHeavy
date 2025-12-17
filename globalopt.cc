
//
// Global optimisation options base-class
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.hpp"
#include "hyper_base.hpp"
#include "randfun.hpp"
#include "plotml.hpp"

bool isisValNone(const gentype &val) { return val.isValNone(); }
bool isisValNone(const double &)     { return false;           }

#define NOMINAL_MAX 1

GlobalOptions::GlobalOptions()
{
        ispydirect = false;

        optname = "Global Optimisation";

        simname = "regret";
        simoutformat = 2;
        simfreq = 1;
        simFmin = 1;
        simFmax = 0;
        simRmin = 1;
        simRmax = 0;

        altallxres = nullptr;

        maxtraintime = 0;

        softmin = valninf(); // 0;
        softmax = valpinf(); // 1;
        hardmin = valninf();
        hardmax = valpinf();

        penterm = 0.0;

        isProjection  = 0;
        includeConst  = 0;
        whatConst     = 2.0;
        randReproject = 0;
        useScalarFn   = 1;
        xxSampType    = 3;
        xNsamp        = DEFAULT_SAMPLES_SAMPLE;
        xSampSplit    = 1;
        xSampType     = 0;
        fnDim         = 1;
        bernstart     = 1;

        priorrandDirtemplateFnGP.ssetMLTypeClean("gpr");   // GPR_Scalar

        MLregfn = nullptr;

        xwidth = 0;

        MLregind = DEFAULT_MLREGIND;

        randDirtemplate    = nullptr;
        randDirtemplateInd = -1;
        projOp             = nullptr;
        projOpRaw          = nullptr;
        projOpInd          = -1;

        randDirtemplate = nullptr;

        overfnitcnt = 0;

        ismodel_convertx_simple = false;

        needsReset = false;
}

GlobalOptions::~GlobalOptions()
{
    delstuff();
}

void GlobalOptions::delstuff(void)
{
    if ( randDirtemplate )
    {
        MEMDEL(randDirtemplate); randDirtemplate = nullptr;
    }

    if ( subDef.size() )
    {
        for ( int i = 0 ; i < subDef.size() ; ++i )
        {
            if ( subDef(i) )
            {
                MEMDEL(subDef("&",i)); subDef("&",i) = nullptr;
            }
        }

        subDef.resize(0);
    }

    if ( projOpRaw )
    {
        MEMDEL(projOpRaw); projOpRaw = nullptr;
    }

    projOp = nullptr;
}

GlobalOptions::GlobalOptions(const GlobalOptions &src)
{
    *this = src;
}

GlobalOptions &GlobalOptions::operator=(const GlobalOptions &src)
{
    delstuff();

    ispydirect = src.ispydirect;

    optname = src.optname;

        simname = src.simname;
        simoutformat = src.simoutformat;
        simfreq = src.simfreq;
        simFmin = src.simFmin;
        simFmax = src.simFmax;
        simRmin = src.simRmin;
        simRmax = src.simRmax;

    altallxres = src.altallxres;

    maxtraintime = src.maxtraintime;

    softmin = src.softmin;
    softmax = src.softmax;
    hardmin = src.hardmin;
    hardmax = src.hardmax;

    penterm = src.penterm;

    isProjection  = src.isProjection;
    includeConst  = src.includeConst;
    whatConst     = src.whatConst;
    randReproject = src.randReproject;
    useScalarFn   = src.useScalarFn;
    xxSampType    = src.xxSampType;
    xNsamp        = src.xNsamp;
    xSampSplit    = src.xSampSplit;
    xSampType     = src.xSampType;
    fnDim         = src.fnDim;
    bernstart     = src.bernstart;

    priorrandDirtemplateVec    = src.priorrandDirtemplateVec;
    priorrandDirtemplateFnGP   = src.priorrandDirtemplateFnGP;
    priorrandDirtemplateFnBern = src.priorrandDirtemplateFnBern;
    priorrandDirtemplateFnRKHS = src.priorrandDirtemplateFnRKHS;

    MLregfn  = src.MLregfn;

    xwidth = src.xwidth;

    stopearly  = src.stopearly;
    firsttest  = src.firsttest;
    spOverride = src.spOverride;
    berndim    = src.berndim;
    bestyet    = src.bestyet;

    MLregind = src.MLregind;

    MLreglist = src.MLreglist;
    MLregltyp = src.MLregltyp;

    xpweight         = src.xpweight;
    xpweightIsWeight = src.xpweightIsWeight;
    xbasis           = src.xbasis;
    xbasisprod       = src.xbasisprod;

    a = src.a;
    b = src.b;
    c = src.c;

    locdistMode = src.locdistMode;
    locvarsType = src.locvarsType;

    ismodel_convertx_simple = src.ismodel_convertx_simple;

    projOptemplate    = src.projOptemplate;
    projOp            = nullptr;
    projOpRaw         = nullptr;
    projOpInd         = src.projOpInd;

    randDirtemplate    = nullptr;
    randDirtemplateInd = src.randDirtemplateInd;
    //subDef             = nullptr;
    subDefInd          = src.subDefInd;

    addSubDim = src.addSubDim;

    locfnarg = src.locfnarg;

    overfnitcnt = src.overfnitcnt;

    needsReset = src.needsReset;

    return *this;
}

// Reset function so that the next simulation can run

void GlobalOptions::reset(void)
{
    delstuff();

    altallxres = nullptr;

    xwidth = 0;

    stopearly  = 0;
    firsttest  = 1;
    spOverride = 0;
    berndim    = 0;
    bestyet    = 0.0;

    // We leave the following unchanged so that we don't get
    // ML overwriting
    //
    // int MLregind;
    //
    // Vector<int> MLreglist;
    // Vector<int> MLregltyp;

    xpweight.resize(0);
    xpweightIsWeight = 0;
    xbasis.resize(0);
    xbasisprod.resize(0,0);

    a.resize(0);
    b.resize(0);
    c.resize(0);

    locdistMode.resize(0);
    locvarsType.resize(0);

    randDirtemplate    = nullptr;
    randDirtemplateInd = -1;
    projOp             = nullptr;
    projOpRaw          = nullptr;
    projOpInd          = -1;

    ismodel_convertx_simple = false;

    subDef = nullptr;
    subDef.resize(0);
    subDefInd.resize(0);

    addSubDim = 0;

    locfnarg = nullptr;

    overfnitcnt = 0;

    return;
}



int GlobalOptions::optim(int dim,
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
                      Vector<gentype> &meanallmres, Vector<gentype> &varallmres)
    {
        if ( needsReset ) { reset(); }

        int res = 0;
        int nr = static_cast<int>(numReps);

        {
            allxres.resize(nr);
            allXres.resize(nr);
            allfres.resize(nr);
            allcres.resize(nr);
            allFres.resize(nr);
            allmres.resize(nr);
            allsres.resize(nr);
            s_score.resize(nr);
            is_feas.resize(nr);

            Vector<gentype> vecfres(nr);
            Vector<gentype> vecFres(nr);
            Vector<gentype> vecires(nr);
            Vector<gentype> vectres(nr);
            Vector<gentype> vecTres(nr);

            int maxxlen = 0;

            for ( size_t j = 0 ; j < numReps ; ++j )
            {
                int jj = static_cast<int>(j);

                //GlobalOptions *locopt = makeDup();

                res += realOptim(dim,
                                 xres,Xres,fres,cres,Fres,mres,sres,ires,mInd,
                                 allxres("&",jj),allXres("&",jj),allfres("&",jj),allcres("&",jj),allFres("&",jj),allmres("&",jj),allsres("&",jj),s_score("&",jj),is_feas("&",jj),
                                 xmin,xmax,distMode,varsType,fn,fnarg,killSwitch);

                vecfres("&",jj) = fres;
                vecFres("&",jj) = Fres;
                vecires("&",jj) = ires;

                // Sort allmres(j) to be strictly decreasing!

                int startyet = 0;

                for ( int kk = 0 ; kk < allfres(jj).size() ; ++kk )
                {
                    if ( !startyet )
                    {
                        if ( is_feas(jj)(kk) && allmres(jj)(kk).isCastableToRealWithoutLoss() ) { startyet = 1;                            }
                        else                                                                    { allmres("&",jj)("&",kk) = nullgentype(); }
                    }

                    else
                    {
                        if ( is_feas(jj)(kk) && allmres(jj)(kk).isCastableToRealWithoutLoss() ) { allmres("&",jj)("&",kk) = ( allmres(jj)(kk) > allmres(jj)(kk-1) ) ? allmres(jj)(kk) : allmres(jj)(kk-1); }
                        else                                                                    { allmres("&",jj)("&",kk) = allmres(jj)(kk-1);                                                             }
                    }
                }

                maxxlen = ( allmres(jj).size() > maxxlen ) ? allmres(jj).size() : maxxlen;

                findfirstLT(vectres("&",jj),allmres(jj),softmin);
                findfirstLT(vecTres("&",jj),allmres(jj),hardmin);

                if ( j != numReps-1 ) { reset(); }
            }

            calcmeanvar(meanfres,varfres,vecfres);
            calcmeanvar(meanFres,varFres,vecFres);
            calcmeanvar(meanires,varires,vecires);
            calcmeanvar(meantres,vartres,vectres);
            calcmeanvar(meanTres,varTres,vecTres);

            Vector<int> ii(nr);
            bool isdone = false;

            Vector<Vector<gentype> > vecallfres(allfres);
            Vector<Vector<gentype> > vecallFres(allFres);
            Vector<Vector<gentype> > vecallmres(allmres);

            for ( size_t j = 0 ; j < numReps ; ++j )
            {
                int jj = static_cast<int>(j);

                for ( int kk = 0 ; kk < allmres(jj).size() ; ++kk )
                {
                    if ( vecallmres(jj)(kk).isValNone() )
                    {
                        vecallmres("&",jj)("&",kk) = -NOMINAL_MAX;
                    }
                }
            }

            for ( ii = 0 ; !isdone ; )
            {
                isdone = true;

                for ( size_t j = 0 ; j < numReps ; ++j )
                {
                    int jj = static_cast<int>(j);

                    if ( ii(jj) < vecallmres(jj).size() )
                    {
                        isdone = false;
                        break;
                    }
                }

                if ( !isdone )
                {
                    double Fmin = 0;
                    int jjmin = -1;

                    for ( size_t j = 0 ; j < numReps ; ++j )
                    {
                        int jj = static_cast<int>(j);

                        if ( ( ii(jj) < vecallmres(jj).size() ) && ( ( jjmin == -1 ) || ( ((double) vecallFres(jj)(ii(jj))) < Fmin ) ) )
                        {
                            Fmin = vecallFres(jj)(ii(jj));
                            jjmin = jj;
                        }
                    }

                    StrucAssert( jjmin != -1 );

                    for ( size_t j = 0 ; j < numReps ; ++j )
                    {
                        int jj = static_cast<int>(j);

                        if ( ( ii(jj) >= vecallFres(jj).size() ) || ( ((double) vecallFres(jj)(ii(jj))) != Fmin ) )
                        {
                            vecallfres("&",jj).add(ii(jj)); vecallfres("&",jj)("&",ii(jj)) = nullgentype();
                            vecallFres("&",jj).add(ii(jj)); vecallFres("&",jj)("&",ii(jj)) = Fmin;
                            vecallmres("&",jj).add(ii(jj)); vecallmres("&",jj)("&",ii(jj)) = ( simRmin < simRmax ) ? simRmax : -NOMINAL_MAX; //nullgentype();

                            if ( ii(jj) ) { vecallmres("&",jj)("&",ii(jj)) = vecallmres(jj)(ii(jj)-1); }
                        }
                    }
                }

                ii += 1;
            }

            calcmeanvar(meanallfres,varallfres,vecallfres);
            calcmeanvar(meanallFres,varallFres,vecallFres);
            calcmeanvar(meanallmres,varallmres,vecallmres);
        }

        {
            // Plot comparison of different grids if selected

            if ( simfreq )
            {
                // Find unused filename to prevent sequential overwrite

                std::string plotnamepdf = simname+"_0.pdf";

                int fcnt = 0;

                while ( fileExists(plotnamepdf) )
                {
                    ++fcnt;
                    plotnamepdf = simname+"_"+(std::to_string(fcnt))+".pdf";
                }

                std::string plotname = simname+"_"+(std::to_string(fcnt));

                // Translate data

                Vector<double> xdata(meanallFres.size());
                Vector<double> ydata(meanallmres.size());
                Vector<double> yvar(varallmres.size());
                Vector<double> ybaseline(meanallmres.size(),0.0); // unused but must be present
                Vector<Vector<double> > xpos(0); //allFres.size());
                Vector<Vector<double> > ypos(0); //allFres.size());
                Vector<Vector<double> > xneg(0); //allFres.size());
                Vector<Vector<double> > yneg(0); //allFres.size());
                Vector<Vector<double> > xequ(0); //allFres.size());
                Vector<Vector<double> > yequ(0); //allmres.size());

                for ( int i = meanallFres.size()-1 ; i >= 0 ; --i )
                {
                    if ( meanallmres(i).isValNone() ) { xdata.remove(i); ydata.remove(i); yvar.remove(i); ybaseline.remove(i);                      }
                    else                              { xdata("&",i) = meanallFres(i); ydata("&",i) = -meanallmres(i); yvar("&",i) = varallmres(i); }
                }

                //for ( int j = 0 ; j < allFres.size() ; ++j )
                //{
                //    xequ("&",j).resize(allFres(j).size());
                //    yequ("&",j).resize(allmres(j).size());
                //
                //    for ( int i = allFres(j).size()-1 ; i >= 0 ; --i )
                //    {
                //        if ( allmres(j)(i).isValNone() ) { xequ("&",j).remove(i); yequ("&",j).remove(i);                            }
                //        else                             { xequ("^",j)("&",i) = allFres(j)(i); yequ("&",j)("&",i) = -allmres(j)(i); }
                //    }
                //}

                // Plot

                int incvar = 1;
                int incdata = ( xequ.size() > 1 ) ? 1 : 0;

                plot2d(xdata,ydata,yvar,ybaseline,xpos,ypos,xneg,yneg,xequ,yequ,simFmin,simFmax,simRmin,simRmax,plotname,plotname+"_dat",simoutformat,incdata,incvar);
            }
        }

        needsReset = true;

        return res;
    }

int GlobalOptions::realOptim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype         &fres,
                      Vector<gentype> &cres,
                      gentype         &Fres,
                      gentype         &mres,
                      gentype         &sres,
                      int             &ires,
                      int             &mInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype>          &allfres,
                      Vector<Vector<gentype> > &allcres,
                      Vector<gentype>          &allFres,
                      Vector<gentype>          &allmres,
                      Vector<gentype>          &allsres,
                      Vector<double>           &s_score,
                      Vector<int>              &is_feas,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
    {
        locfnarg = fnarg;

        // dim: dimension of problem
        // xres: x at minimum
        // Xres: xres in format seen by specific method, without projection or distortion
        // fres: f(x) at minimum
        // ires: index of optimal solution in allres
        // allxres: vector of all vectors evaluated
        // allXres: vector of all vectors evaluated, in raw (unprojected/undistorted) format
        // allfres: vector of all f(x) evaluations
        // allcres: vector of all c(x) evaluations
        // xmin: lower bound on variables
        // xmax: upper bound on variables
        // distMode: 0 = linear distribution of points
        //           1 = logarithmic distribution of points
        //           2 = anti-logarithmic distribution of points
        //           3 = uniform random distribution of points (grid only)
        //           4 = inverse logistic distribution of points
        //           5 = REMBO
        // varsType: 0 = integer
        //           1 = double
        // fn: function being optimised
        // fnarg: argument passed to fn
        // killSwitch: set nonzero to force stop

        NiceAssert( dim >= 0 );
        NiceAssert( optdefed() );
        NiceAssert( distMode.size() == dim );
        NiceAssert( varsType.size() == dim );

        addSubDim = 0;

        // For schedules Bernstein projection, effdim is the (reduced) dimension for this "round"

        int effdim = dim;

        stopearly  = 0;
        firsttest  = 1;
        spOverride = 0;
        bestyet    = 0.0;

        // The following variables are used to normalise the axis to [0,1].
        // Their exact meaning depends on distMode.

        Vector<gentype> allfakexmin(dim);
        Vector<gentype> allfakexmax(dim);

        xwidth = calcabc(dim,allfakexmin,allfakexmax,a,b,c,xmin,xmax,distMode,varsType,ismodel_convertx_simple);

        Vector<gentype> fakexmin(allfakexmin);
        Vector<gentype> fakexmax(allfakexmax);

        locdistMode = distMode;
        locvarsType = varsType;

        // These need to be passed back

        // overfn setup

        Vector<gentype> xmod(dim);

        void *backoverfnargs[19]; // It is very important that overfnargs[15] is defined here!
        void **overfnargs = backoverfnargs+3;

        //SparseVector<SparseVector<gentype> > altaltxmod;

        overfnargs[0] = (void *)  fnarg;
        overfnargs[1] = (void *) &dim;
        overfnargs[2] = (void *) &xmod;
        overfnargs[3] = (void *)  this;
        overfnargs[4] = altallxres ? altallxres : ( (void *) &allxres );
        overfnargs[5] = (void *) &penterm;
        overfnargs[15] = nullptr; //(void *) &locMLnumbers; // if the optimiser gets recursed (which only happens in gridOpt) then this is required to ensure thet MLnumbers is defined and doesn't overlap!

        overfnargs[6] = (void *) fn;

        backoverfnargs[0] = nullptr; //(void *) &altaltxmod;
        backoverfnargs[1] = nullptr; //(void *) &usealtoptfn;
        backoverfnargs[2] = nullptr; //(void *) &altoptfn;

        altallxres = overfnargs[4];

        // Projection templates

        if ( isProjection == 0 )
        {
            // No vector/function projection, so do nothing

            ;
        }

        else if ( isProjection == 1 )
        {
            // Create random subspace - vector-valued subspace

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(priorrandDirtemplateVec.type());

            (*temprandDirtemplate).getML() = priorrandDirtemplateVec;
            randDirtemplateInd = regMLloc(temprandDirtemplate,fnarg,0);

            randDirtemplate = &((*temprandDirtemplate).getML());
        }

        else if ( isProjection == 2 )
        {
            // Create random subspace - function-valued subspace

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(priorrandDirtemplateFnGP.type());

            (*temprandDirtemplate).getML() = priorrandDirtemplateFnGP;
            randDirtemplateInd = regMLloc(temprandDirtemplate,fnarg,1);

            randDirtemplate = &((*temprandDirtemplate).getML());

            // Need to fill in x distributions.

            Vector<int> sampleInd(fnDim);
            Vector<gentype> sampleDist(fnDim);

            int j;

            for ( j = 0 ; j < fnDim ; ++j )
            {
                sampleInd("&",j)  = j;
                sampleDist("&",j) = "urand(x,y)";

                SparseVector<SparseVector<gentype> > xy;

                xy("&",0)("&",0) = 0.0;
                xy("&",0)("&",1) = 1.0;

                sampleDist("&",j).substitute(xy);
            }

            initModelDistr(sampleInd,sampleDist);
        }

        else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
        {
            // Create non-random subspace - function-valued subspace, Bernstein basis

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(priorrandDirtemplateFnBern.type());

            (*temprandDirtemplate).getML() = priorrandDirtemplateFnBern;
            randDirtemplateInd = regMLloc(temprandDirtemplate,fnarg,1);

            randDirtemplate = &((*temprandDirtemplate).getML());

            if ( isProjection == 4 )
            {
                effdim = bernstart;
            }
        }

        else if ( isProjection == 5 )
        {
//RKHSFIXME
            // Projection via RKHSVector, so do nothing

            ;
        }

        // Optimisation loop
        //
        // substatus: 0 = this is the first optimisation
        //            1 = not the first optimisation, but the "best" subspace wasn't changed by the most recent optimisation
        //            2 = not the first optimisation, "best" subspace given by subsequent x vector

        int res = 0;

        int ii,jj;
        int substatus = 0;

        // We can't afford to call prelim operations between optimisation and
        // making a subspace, we we pre-make it here, then after each optim

        // Trigger preliminary "setup" operations

        ii = 0;
        {
            gentype trigval;
            Vector<gentype> xdummy;
            SparseVector<gentype> altlocxres;

            trigval.force_int() = (-ii*10000)-1; // -1,-10001,-20001,-30001,...

            if ( !ispydirect )
            {
                (*fn)(trigval,xdummy,fnarg);
            }

            makeSubspace(dim,fnarg,substatus,altlocxres); // until we actually have a projection, at least

            trigval.force_int() = (-ii*10000)-2; // -2,-10002,-20002,-30002,...

            if ( !ispydirect )
            {
                (*fn)(trigval,xdummy,fnarg);
            }
        }

        int locrandReproject = randReproject;

        for ( ii = 0 ; ii <= locrandReproject ; ++ii )
        {
            outstream() << "---\n";

            Vector<gentype> locxres;
            Vector<gentype> locXres;
            gentype         locfres;
            Vector<gentype> loccres;
            gentype         locFres;
            gentype         locmres;
            gentype         locsres;
            int             locires;
            int             locmInd;

            SparseVector<gentype> altlocxres;
            SparseVector<gentype> altlocXres;

            Vector<Vector<gentype> > locallxres;
            Vector<Vector<gentype> > locallXres;
            Vector<gentype>          locallfres;
            Vector<Vector<gentype> > locallcres;
            Vector<gentype>          locallFres;
            Vector<gentype>          locallmres;
            Vector<gentype>          locallsres;
            Vector<double>           locs_score;
            Vector<int>              locis_feas;

            stopearly = 0;

            if ( isProjection == 4 )
            {
                spOverride = ii; // only want one initial random batch

                // Update bernstein schedule (note that if isProjection == 3 this intentionally does nothing, and that effdim is bounded by dim)

                if ( ii && ( effdim < dim ) )
                {
                    ++effdim;
                }

                // Fake-out unused dimensions for scheduled Bernstein optimisation

                if ( effdim )
                {
                    for ( jj = 0 ; jj < effdim ; ++jj )
                    {
                        fakexmin("&",jj) = allfakexmin(jj);
                        fakexmax("&",jj) = allfakexmax(jj);
                    }
                }

                if ( effdim < dim )
                {
                    for ( jj = effdim ; jj < dim ; ++jj )
                    {
                        fakexmin("&",jj) = allfakexmin(jj);
                        fakexmax("&",jj) = allfakexmin(jj);
                    }
                }
            }

            else
            {
                effdim = dim;
            }

            berndim = effdim;

            // Optimisation

            double locxmin = ( xmin.size() > 0 ) ? xmin(0) : 0;
            double locxmax = ( xmax.size() > 0 ) ? xmax(0) : 1;

            double locymin = ( xmin.size() > 1 ) ? xmin(1) : 0;
            double locymax = ( xmax.size() > 1 ) ? xmax(1) : 1;

            model_log(0,locxmin,locxmax,locymin,locymax); // model_log records these min/max for later use!

            res = optim(dim,locXres,locfres,loccres,locFres,locmres,locsres,locires,locmInd,
                          locallXres,locallfres,locallcres,locallFres,locallmres,locallsres,locs_score,locis_feas,
                          fakexmin,fakexmax,overfn,overfnargs,killSwitch);

            model_log(2);

            // Update subspace if/as required

            if ( substatus == 0 )
            {
                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres);
                model_convertx(altlocxres,( altlocXres = locXres ),0,0,0,1);

                // Just completed the first optimisation, so the solution must be the best so far by definition

                xres = locxres;
                Xres = locXres;
                fres = locfres;
                cres = loccres;
                Fres = locFres;
                mres = locmres;
                sres = locsres;
                ires = locires+(allfres.size());
                mInd = projOpInd; // locmInd

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allcres.append(allcres.size(),locallcres);
                allFres.append(allFres.size(),locallFres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);
                is_feas.append(is_feas.size(),locis_feas);

                substatus = 2;
            }

            else if ( locfres < fres )
            {
                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres);
                model_convertx(altlocxres,( altlocXres = locXres ),0,0,0,1);

                // Not first optimisation, but there is a new best solution

                xres = locxres;
                Xres = locXres;
                fres = locfres;
                cres = loccres;
                Fres = locFres;
                mres = locmres;
                sres = locsres;
                ires = locires+(allfres.size());
                mInd = projOpInd; // locMind

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allcres.append(allcres.size(),locallcres);
                allFres.append(allFres.size(),locallFres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);
                is_feas.append(is_feas.size(),locis_feas);

                substatus = 2;
            }

            else
            {
                // Set "x" to zero so that it just reverts to previous best

                locXres = 0.0_gent; //zerogentype();

                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres,1); // Note that useOrigin is set here, so locXres is not actually used!
                model_convertx(altlocxres,( altlocXres = locXres ),1,0,0,1); // Note that useOrigin is set here, so locXres is not actually used!

                // Not first optimisation, no improvement found (but zero set anyhow)

                //xres = locxres;
                //Xres = locXres;
                //fres = locfres;
                //ires = locires+(allfres.size());
                //mInd = projOpInd;

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allcres.append(allcres.size(),locallcres);
                allFres.append(allFres.size(),locallFres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);
                is_feas.append(is_feas.size(),locis_feas);

                substatus = 1;
            }

            // Increase iterations if stopearly set (bernstein scheduled stopearly)

            if ( stopearly )
            {
                NiceAssert( isProjection == 4 );

                ++locrandReproject;
            }

            // Construct random subspace if required (but not for scheduled Bernstein)

            if ( ii < locrandReproject )
            {
                if ( isProjection != 4 )
                {
                    makeSubspace(dim,fnarg,substatus,altlocxres);
                }

                // Trigger intermediate (hyperparameter tuning) operations

                gentype trigval;
                Vector<gentype> xdummy;

                trigval.force_int() = (-(ii+1)*10000)-2; // -2,-10002,-20002,-30002,...

                if ( !ispydirect )
                {
                    (*fn)(trigval,xdummy,fnarg);
                }
            }
        }

        // And we're done

        a.resize(0);
        b.resize(0);
        c.resize(0);

        return res;
    }

int GlobalOptions::makeSubspace(int dim, void *fnarg, int substatus, const SparseVector<gentype> &locxres)
    {
        // substatus zero on first call, 1 or 2 otherwise

        // Note we don't delete any projections blocks as they may be
        // required in the functional optimisation case where return
        // is of form fnB(projOpInd,500,x).  We do register them though
        // so they will be deleted on exit.

        if ( substatus )
        {
            // This function is used by smboopt to clear models
            // if the model is built on x rather than model_convertx(x)

            model_clear();
        }

        if ( isProjection == 0 )
        {
            // No projection, nothing to see here

            addSubDim = 0;

            projOpInd = -1;
        }

        else if ( ( isProjection == 1 ) || ( isProjection == 2 ) )
        {
            // Projection onto either random vectors or random functions.  Both of
            // these are distributions, so we can clone and finalise them to get a
            // random projection.

            // If not first run then we add a dimension (zero point), permanently
            // weighted to 1, that represents the current best solution.

            addSubDim = substatus ? 1 : 0;

            errstream() << "Constructing random subspace... (" << dim+addSubDim << ") ";

            subDef.resize(dim+addSubDim);
            subDefInd.resize(dim+addSubDim);

            int i,j;

            ML_Mutable *randDir;

            errstream() << "creating random direction prototypes... ";

            int rdim = addSubDim ? dim+1 : dim;

            for ( i = 0 ; i < dim ; ++i )
            {
                if ( ( i < dim-1 ) || !( !addSubDim && includeConst ) )
                {
                    MEMNEW(randDir,ML_Mutable);
                    (*randDir).setMLTypeClean((*randDirtemplate).type());

                    (*randDir).getML() = (*randDirtemplate);
                    subDefInd("&",i) = regMLloc(randDir,fnarg,2);

                    subDef("&",i) = &((*randDir).getML());

                    if ( substatus )
                    {
                        consultTheOracle(*randDir,dim,locxres,!i);
                    }
                }

                else
                {
                    errstream() << "include constant term (weighted)... ";

                    // First round, so final variable controls a *constant* offset

                    gentype outfnhere(whatConst);

                    MEMNEW(randDir,ML_Mutable);
                    (*randDir).setMLTypeClean(207);  // BLK_UsrFnb

                    (*randDir).setoutfn(outfnhere);
                    subDefInd("&",i) = regMLloc(randDir,fnarg,2);

                    subDef("&",i) = &((*randDir).getML());
                }
            }

            if ( addSubDim )
            {
                errstream() << "include previous best term (fixed zero point)... ";

                // Put previous best as "fixed" point in new subspace.  projOp
                // has been made the best result from the most recent sub-optimisation.

                randDir = projOpRaw;
                subDefInd("&",dim) = projOpInd;
                subDef("&",dim) = &((*randDir).getML());
            }

            else if ( projOpRaw )
            {
                MEMDEL(projOpRaw); projOpRaw = nullptr;
            }

            // Make projOp a projection onto this subspace

            MEMNEW(projOpRaw,ML_Mutable);
            (*projOpRaw).setMLTypeClean(projOptemplate.type());

            (*projOpRaw).getML() = projOptemplate;
            projOpInd = regMLloc(projOpRaw,fnarg,3);

            projOp = &(dynamic_cast<BLK_Conect &>((*projOpRaw).getBLK()));

            errstream() << "sampling....";

            Vector<gentype> subWeight(dim+addSubDim);

            gentype biffer;

            subWeight = ( biffer = 0.0 );

            if ( addSubDim )
            {
                subWeight("&",dim) = 1.0;
            }

            (*projOp).setmlqlist(subDef);
            (*projOp).setmlqweight(subWeight);

//FIXMEFIXME fnDim
            errstream() << "combining... ";

            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            (*projOp).setSampleMode(1,xmin,xmax,xNsamp,xSampSplit,xSampType,xxSampType,1.0); // Need this even when subDef has been sampled to set variables used to construct y in projOp, when required

            if ( useScalarFn && !includeConst && ( isProjection == 2 ) )
            {
                xbasis.resize(rdim);

                for ( i = 0 ; i < rdim ; ++i )
                {
                    xbasis("&",i) = (*subDef(i)).y();
                    xbasis("&",i).scale(sqrt(1/((double) (*subDef(i)).y().size())));
                }

                gentype tempxp;

                xbasisprod.resize(rdim,rdim);

                for ( i = 0 ; i < rdim ; ++i )
                {
                    for ( j = 0 ; j < rdim ; ++j )
                    {
                        innerProduct(xbasisprod("&",i,j),xbasis(i),xbasis(j));
                    }
                }

                model_update();
            }

            errstream() << "done with weight " << ((*projOp).mlqweight()) << "\n";
        }

        else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
        {
            // We set up the same dimension, even though we don't use some of it

            NiceAssert( !includeConst );

            addSubDim = 0;

            errstream() << "Constructing Bernstein basis... ";

            subDef.resize(dim);
            subDefInd.resize(dim);

            int i,j,k;
            int axisdim = (int) pow((double) dim+1,1/((double) fnDim))-1; // floor

            Vector<double> bberndim(fnDim);
            Vector<double> bernind(fnDim);

            bberndim = (double) axisdim;
            bernind = 0.0;

            ML_Mutable *randDir;

            errstream() << "setting parameters... ";

            for ( i = 0 ; i < dim ; ++i )
            {
                MEMNEW(randDir,ML_Mutable);
                (*randDir).setMLTypeClean((*randDirtemplate).type());

                (*randDir).getML() = (*randDirtemplate);
                subDefInd("&",i) = regMLloc(randDir,fnarg,8);

                subDef("&",i) = &((*randDir).getML());

                k = i;

                for ( j = 0 ; j < fnDim ; ++j )
                {
                    bernind("&",j) = k%(((int) bberndim(j))+1);
                    k /= ((int) bberndim(j))+1;
                }

                gentype gberndim(bberndim);
                gentype gbernind(bernind);

//FIXMEFIXME fnDim
                dynamic_cast<BLK_Bernst &>((*(subDef("&",i)))).setBernDegree(gberndim);
                dynamic_cast<BLK_Bernst &>((*(subDef("&",i)))).setBernIndex(gbernind);
            }

            errstream() << "combining... ";

            // Make projOp a projection onto this subspace

            MEMNEW(projOpRaw,ML_Mutable);
            (*projOpRaw).setMLTypeClean(projOptemplate.type());

            (*projOpRaw).getML() = projOptemplate;
            projOpInd = regMLloc(projOpRaw,fnarg,9);

            projOp = &(dynamic_cast<BLK_Conect &>((*projOpRaw).getBLK()));

            errstream() << "sampling....";

            Vector<gentype> subWeight(dim);

            gentype biffer;

            subWeight = ( biffer = 0.0 );

            (*projOp).setmlqlist(subDef);
            (*projOp).setmlqweight(subWeight);

//FIXMEFIXME fnDim
            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            (*projOp).setSampleMode(1,xmin,xmax,xNsamp,xSampSplit,xSampType,xxSampType,1.0); // Need this even when subDef has been sampled to set variables used to construct y in projOp, when required

            errstream() << "done with weight " << ((*projOp).mlqweight()) << "\n";
        }

        else if ( isProjection == 5 )
        {
//RKHSFIXME
            // Projection via RKHSVector, nothing to see here

            addSubDim = 0;

            priorrandDirtemplateFnRKHS.resizeN(dim);

            int i,j;

            for ( i = 0 ; i < dim ; ++i )
            {
                priorrandDirtemplateFnRKHS.x("&",i).zero();

                for ( j = 0 ; j < fnDim ; ++j )
                {
                    randufill(priorrandDirtemplateFnRKHS.x("&",i)("&",j));
                }
            }

            projOpInd = -1;
        }

        return projOpInd;
    }










void overfn(gentype &res, Vector<gentype> &x, void *arg)
{
    //SparseVector<SparseVector<gentype> > &altaltxmod = *((SparseVector<SparseVector<gentype> > *) ((void **) arg)[-3]);
    //int &usealtoptfn                                 = *((int *)                                  ((void **) arg)[-2]);
    //gentype &altoptfn                                = *((gentype *)                              ((void **) arg)[-1]);
    void *fnarg                                      = ((void *)                                  ((void **) arg)[0]);
    int &dim                                         = *((int *)                                  ((void **) arg)[1]);
    Vector<gentype> &xmod                            = *((Vector<gentype> *)                      ((void **) arg)[2]);
    GlobalOptions &gopts                             = *((GlobalOptions *)                        ((void **) arg)[3]);
    Vector<Vector<gentype> > &allxres                = *((Vector<Vector<gentype> > *)             ((void **) arg)[4]);
    gentype &penterm                                 = *((gentype *)                              ((void **) arg)[5]);

    void (*fn)(gentype &, Vector<gentype> &, void *) = ((void (*)(gentype &, Vector<gentype> &, void *)) ((void **) arg)[6]);

    suppressallstreamcout(); // Keep logging to files, but don't clutter the screen
    gopts.convertx(dim,xmod,x,0,1);
    unsuppressallstreamcout(); // Back to normal output

    bool actualtest = false;

    if ( res.isValNull() )
    {
        // Process request to remove the previous (non) observation
        allxres.remove(allxres.size()-1);
        return;
    }

    if ( xmod.size() )
    {
        actualtest = true;
//        gopts.model_log(1);
    }

//errstream() << "what is xmod " << xmod << "\n";
    {
        suppressallstreamcout();
//errstream() << "and we call " << xmod << "\n";

        if ( !gopts.ispydirect || actualtest )
        {
            (*fn)(res,xmod,fnarg);
        }

        unsuppressallstreamcout();
    }

    std::stringstream resbuffer;
    resbuffer << "Evaluation: " << res << "\t= f(";
    printoneline(resbuffer,xmod);
    resbuffer << ")";
    //std::stringstream resbuffer;
    //resbuffer << "f(g(";
    //printoneline(resbuffer,xmod);
    //resbuffer << ")) = " << res;
    ////resbuffer << "f(g(conv(";
    ////printoneline(resbuffer,x);
    ////resbuffer << "))) = " << res;
    ////errstream() << gopts.optname << ": global optimiser: f(g(.))" << xmod << ") = " << res << "\n";

    if ( actualtest && ( !(penterm.isCastableToRealWithoutLoss()) || ( ( (double) penterm ) != 0.0 ) ) )
    {
        gentype xmodvecform(xmod);

        res += penterm(xmodvecform);

        resbuffer << " -> " << res;
        //errstream() << gopts.optname << ": global optimiser: f(g(.)) + penterm(" << xmodvecform << ") = " << res << "\n";
    }

    // Clip result at softmin/max

    if ( actualtest && ( ( res.isValInteger() || res.isValReal() ) && ( (double) res > gopts.softmax ) ) )
    {
        errstream() << gopts.optname << ": result clipped at softmax: " << res << " -> " << gopts.softmax << "\n";

        res = gopts.softmax;

        resbuffer << " (clip)";
    }

    if ( actualtest && ( ( res.isValInteger() || res.isValReal() ) && ( (double) res < gopts.softmin ) ) )
    {
        errstream() << gopts.optname << ": result clipped at softmin: " << res << " -> " << gopts.softmin << "\n";

        res = gopts.softmin;

        resbuffer << " (clip)";
    }

    if ( actualtest )
    {
        int &counter = gopts.overfnitcnt;
//static thread_local int counter = 0;
////static int lineblank = 0;
counter++;
resbuffer << " (" << counter << ")                ";
//blankPrint(outstream(),lineblank);
//lineblank = nullPrint(outstream(),resbuffer.str());
outstream() << resbuffer.str() << "\n";
//        errstream() << resbuffer.str() << "\n";
    }

    // Save to allxres if this is an actual evaluation ( x.size() == 0 if intermediate operation )

    if ( x.size() )
    {
        allxres.append(allxres.size(),xmod);

        if ( ( gopts.isProjection == 4 ) && ( gopts.stopearly == 0 ) && ( gopts.berndim > 1 ) && ( gopts.firsttest || ( res < gopts.bestyet ) ) )
        {
            int i,j;
            int effdim = gopts.berndim;

            gopts.stopearly = 0;
            gopts.firsttest = 0;
            gopts.bestyet   = res;

            for ( i = 0 ; i < effdim-1 ; ++i )
            {
                for ( j = i+1 ; j < effdim ; ++j )
                {
                    if ( (double) abs2(x(i)-x(j)) > 0.95 )
                    {
                        gopts.stopearly = 1;
                        break;
                    }
                }
            }
        }
    }

    //errstream() << ".";

    return;
}

double calcabc(int dim,
               Vector<gentype> &fakexmin, Vector<gentype> &fakexmax,
               Vector<double> &a, Vector<double> &b, Vector<double> &c,
               const Vector<gentype> &xmin, const Vector<gentype> &xmax,
               const Vector<int> &distMode, const Vector<int> &varsType,
               bool &ismodel_convertx_simple)
{
    double xwidth = 1;

    ismodel_convertx_simple = dim ? true : false;

    a.resize(dim+1);
    b.resize(dim+1);
    c.resize(dim+1);

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; ++i )
        {
            double lb = (double) xmin(i);
            double ub = (double) xmax(i);

            int zerowidth = ( lb == ub ) ? 1 : 0;

            if ( zerowidth || distMode(i) || !varsType(i) )
            {
                // zero width or nonlinear scale or integral variable

                ismodel_convertx_simple = false;
            }

            xwidth = ( !i || ( (ub-lb) > xwidth ) ) ? (ub-lb) : xwidth;

            fakexmin("&",i) = 0.0;
            fakexmax("&",i) = zerowidth ? 0.0 : 1.0;

            if ( distMode(i) == 1 )
            {
                ismodel_convertx_simple = false;

                // Logarithmic grid
                //
                // v = a + e^(b+c.t)
                // 
                // c = log(10)
                // 
                // t = 0 => v = lb
                //       => a+e^b = lb
                //       => b = log(lb-a)
                // t = 1 => v = ub
                //       => a+e^(b+c) = ub
                //       => e^(b+c) = ub-a
                //       => b+c = log(ub-a)
                //       => b = log(ub-a)-c
                // 
                // => log(lb-a) = lob(ub-a)-c
                // => log((ub-a)/(lb-a)) = log(e^c)
                // => (ub-a)/(lb-a) = e^c
                // => ub - a = (e^c)lb - (e^c)a
                // => ((e^c)-1).a = (e^c)lb - ub
                // => a = ((e^c)lb - ub)/((e^c)-1)

                c("&",i) = log(10.0);
                a("&",i) = ( (exp(c(i))*lb) - ub ) / ( exp(c(i)) - 1.0 );
                b("&",i) = log( lb - a(i) );
            }

            else if ( distMode(i) == 2 )
            {
                ismodel_convertx_simple = false;

                // Anti-logarithmic grid (inverse of logarithmic grid)
                //
                // v = (1/c) log(t-a) - (b/c)
                //
                // c = log(10)
                // 
                // t = 0 => v = lb
                //       => 0 = a + e^(b+c.lb)
                //       => -a = e^(b+c.lb)
                //       => b+c.lb = log(-a)
                //       => b = log(-a) - c.lb
                // t = 0 => t = ub
                //       => 1 = a + e^(b+c.ub)
                //       => 1-a = e^(b+c.ub)
                //       => b+c.ub = log(1-a)
                //       => b = log(1-a) - c.ub
                //
                // => log(-a) - c.lb = log(1-a) - c.ub
                // => c.(ub-lb) = log((a-1)/a)
                // => (a-1)/a = exp(c.(ub-lb))
                // => 1 - 1/a = exp(c.(ub-lb))
                // => 1/a = 1 - exp(c.(ub-lb))
                // => a = 1/(1 - exp(c.(ub-lb)))

                c("&",i) = log(10);
                a("&",i) = 1/(1-exp(c(i)*(ub-lb)));
                b("&",i) = log(-a(i))-(c(i)*lb);
            }

            else if ( distMode(i) == 4 )
            {
                ismodel_convertx_simple = false;

                // Anti-logistic grid
                //
                // v = a - (1/b) log( 1/(0.5+(c*(t-0.5))) - 1 )
                //   = a - (1/b) log(0.5-(c*(t-0.5))) + (1/b) log(0.5+(c*(t-0.5)))
                //   = a - (1/b) log(1/(0.5+(c*(t-0.5)))) + (1/b) log(1/(0.5-(c*(t-0.5))))
                //
                // a = lb+((ub-lb)/2)
                // a = (ub+lb)/2
                //
                // m = small number < 1/2
                // B = big number
                // D = (ub-lb)/2
                //
                // t = 0 => v = a-B
                //       => a - (1/b) log( 1/(0.5*(1-c)) - 1 ) = a-B
                //       => a - (1/b) log( 2/(1-c) - 1 ) = a-B
                //       => a - (1/b) log( (1+c)/(1-c) ) = a-B
                //       => a - (1/b) log(1+c) + (1/b) log(1-c) = a-B
                //       => (1/b) log(1+c) - (1/b) log(1-c) = B
                //       => log(1+c) - log(1-c) = b.B
                // t = 1 => v >= a+B
                //       => a - (1/b) log( 1/(0.5*(1+c)) - 1 ) = a+B
                //       => a - (1/b) log( 2/(1+c) - 1 ) = a+B
                //       => a - (1/b) log( (1-c)/(1+c) ) = a+B
                //       => a - (1/b) log(1-c) + (1/b) log(1+c) = a+B
                //       => (1/b) log(1-c) - (1/b) log(1+c) = -B
                //       => (1/b) log(1+c) - (1/b) log(1-c) = B
                //       => log(1+c) - log(1-c) = b.B
                // t = m => v = lb
                //       => a - (1/b) log( 1/(0.5+(c*(m-0.5))) - 1 ) = lb
                //       => a - (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = lb
                //       => (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = a-lb
                //       => (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = (ub-lb)/2
                //       => (1/b) log( 2/(1-c+2cm) - 1 ) = (ub-lb)/2
                //       => (1/b) log( (2-(1-c+2cm))/(1-c+2cm) ) = (ub-lb)/2
                //       => (1/b) log( (1+c-2cm)/(1-c+2cm) ) = (ub-lb)/2
                //       => log(1+c-2cm) - log(1-c+2cm) = b.(ub-lb)/2
                // t = 1-m => v = ub
                //       => a - (1/b) log( 1/(0.5+(c*(1-m-0.5))) - 1 ) = lb
                //       => a - (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = ub
                //       => (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = a-ub
                //       => (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = -(ub-lb)/2
                //       => (1/b) log( 2/(1+c-2cm) - 1 ) = -(ub-lb)/2
                //       => (1/b) log( (2-(1+c-2cm))/(1+c-2cm) ) = -(ub-lb)/2
                //       => (1/b) log( (1-c+2cm)/(1+c-2cm) ) = -(ub-lb)/2
                //       => log(1-c+2cm) - log(1+c-2cm) = -b.(ub-lb)/2
                //       => log(1+c-2cm) - log(1-c+2cm) = b.(ub-lb)/2
                //
                // NB: - symmetry means we need only consider one of t=m,1-m 
                //       (note that the final two expressions are identical)
                //       where both imply that:
                //
                //       b = (log(1+c-2cm)-log(1-c+2cm))/D
                //       b = (log(1+(c*(1-2m)))-log(1-(c*(1-2m))))/D   (*)
                //
                //       where we have defined D = (ub-lb)/2
                //
                //     - likewise symmetry means we only need consider one
                //       of t=0,1 (see final expressions), which using (*) imply:
                //
                //       q = B.(log(1+(c*(1-2m)))-log(1-(c*(1-2m)))) - D.(log(1+c)-log(1-c)) = 0    (**)
                //
                //     - note the derivative of (**) wrt c
                //
                //       B.(1-2m).( 1/(1+(c*(1-2m))) + 1/(1-(c*(1-2m))) ) - D.( 1/(1+c) + 1/(1-c) ) = 0
                //
                //       Assuming B sufficiently large this will be increasing.

                double aval = (ub+lb)/2;
                double qval,bval,cval;

                double cmin = 1e-6;
                double cmax = 0.5;

                double m = NEARZEROVAL;
                double B = NEARINFVAL;
                double D = (ub-lb)/2;
                double ratmax = RATIOMAX;

                cmin = 0;
                cmax = 1;

                while ( 2*(cmax-cmin)/(cmax+cmin) > ratmax )
                {
                    cval  = (cmin+cmax)/2;
                    //bval  = 2*((1/(1+(cval*(1-(2*m)))))-(1/(1-(cval*(1-(2*m))))))/D;
                    bval  = 2*((1/(1-(cval*(1-(2*m)))))-(1/(1+(cval*(1-(2*m))))))/D;
                    qval  = 2*((1.0/(1-cval))-(1.0/(1+cval)))/bval;
                    if ( qval == B )
                    {
                        cmax = cval;
                        cmin = cval;
                    }

                    else if ( qval > B )
                    {
                        cmax = cval;
                    }

                    else
                    {
                        cmin = cval;
                    }
                }

                //cval = cmax;
                //bval = 2*((1/(1+(cval*(1-(2*m)))))-(1/(1-(cval*(1-(2*m))))))/D;
                bval = 2*((1/(1-(cval*(1-(2*m)))))-(1/(1+(cval*(1-(2*m))))))/D;

                a("&",i) = aval;
                b("&",i) = bval;
                c("&",i) = cval;
            }

            else
            {
                // Linear (potentially random) grid
                //
                // v = a + b.t
                //
                // t = 0 => v = lb
                //       => a = lb
                // t = 1 => v = ub
                //       => lb+b = ub
                //       => b = ub-lb

                a("&",i) = lb;
                b("&",i) = ub-a(i);
                c("&",i) = 0;

                if ( ( a(i) != 0 ) || ( b(i) != 1 ) )
                {
                    ismodel_convertx_simple = false;
                }
            }
        }
    }

    return xwidth;
}





int GlobalOptions::analyse(const Vector<gentype> &allmres,
                           const Vector<int> &is_feas,
                           Vector<double> &hypervol,
                           Vector<int> &parind,
                           int calchypervol) const
{
//errstream() << "DEBUG: allmres = " << allmres << "\n";
    int N = allmres.size();

    parind.resize(N);
    hypervol.resize(N);

    hypervol = 0.0;

    if ( N )
    {
        // Work out hypervolume sequence

        if ( calchypervol )
        {
            if ( allmres(0).isValReal() || allmres(0).isValNone() )
            {
                for ( int i = 0 ; i < N ; ++i )
                {
                    if ( !i || ( (double) allmres(i) < -(hypervol(i-1)) ) ) { hypervol("&",i) = -((double) allmres(i)); }
                    else                                                    { hypervol("&",i) = hypervol(i-1);          }
                }
            }

            else
            {
                int M = (allmres(0)).size();
                double **X;

                MEMNEWARRAY(X,double *,N);
                NiceAssert(X);

                for ( int i = 0 ; i < N ; ++i )
                {
                    MEMNEWARRAY(X[i],double,M);
                    NiceAssert( X[i] );

                    for ( int j = 0 ; j < M ; ++j ) { X[i][j] = -((double) ((allmres(i))(j))); }

                    hypervol("&",i) = h(X,i+1,M);
                }

                for ( int i = 0 ; i < N ; ++i ) { MEMDELARRAY(X[i]); X[i] = nullptr; }

                MEMDELARRAY(X); X = nullptr;
            }
        }

        // Work out Pareto set

        int m = allmres(0).size();

        retVector<int> tmpva;

        parind = cntintvec(N,tmpva);

        // First remove all points that are infeasible - ie !(c(x) >= 0). Note that c(x)<0 can't be use (what if c(x) is 0-dimensional inequalities evaluate as true)

        for ( int pos = parind.size()-1 ; pos >= 0 ; --pos )
        {
            if ( !is_feas(pos) ) { parind.remove(pos); }
        }

        for ( int pos = parind.size()-1 ; pos >= 0 ; --pos )
        {
            NiceAssert( allmres(pos).size() == m );

            // Test if allfres(parind(pos)) is dominated by any points
            // in parind != pos.  If it is then remove it from parind.

            int isdom = 0;

            for ( int i = 0 ; !isdom && ( i < parind.size() ) ; ++i )
            {
                if ( parind(i) != parind(pos) )
                {
                    // Test if allmres(i) dominates allmres(pos) - that is,
                    // if allfres(pos)(j) >= allfres(i)(j) for all j

                    isdom = 1;

                    for ( int j = 0 ; isdom && ( j < m ) ; ++j )
                    {
                        if ( allmres(parind(pos))(j) < allmres(parind(i))(j) ) { isdom = 0; }
                    }
                }
            }

            if ( isdom ) { parind.remove(pos); }
        }
    }

    return parind.size();
}
