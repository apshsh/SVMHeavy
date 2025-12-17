
//
// Grid-based Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gridopt.hpp"
#include "randfun.hpp"

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000
#define LOGZTOL 1e-8

int GridOptions::optim(int dim,
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

    Vector<int> llocdistMode(getlocdistMode());
    Vector<int> llocvarsType(getlocvarsType());

    Vector<int> altnumPts(numPts);

    if ( llocdistMode.size() == 0 )
    {
        llocdistMode.resize(dim);
        llocvarsType.resize(dim);

        llocdistMode = 0;
        llocvarsType = 1;
    }

    if ( altnumPts.size() == 0 )
    {
        altnumPts.resize(dim);

        altnumPts = DEFAULT_NUMOPTS;
    }

    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );
    NiceAssert( altnumPts.size() == dim );
    NiceAssert( llocdistMode.size() == dim );
    NiceAssert( llocvarsType.size() == dim );
    NiceAssert( altnumPts >= 0 );
    NiceAssert( llocdistMode >= 0 );
    NiceAssert( llocdistMode <= 4 );
    NiceAssert( llocvarsType >= 0 );
    NiceAssert( llocvarsType <= 1 );

    int res = 0;
    int i,j,k;

    double dres = 0;
    double s,t;

    Xres.resize(dim);
    fres = 0.0;
    ires = 0;

    allxres.resize(0);
    allfres.resize(0);
    allcres.resize(0);
    allmres.resize(0);
    allsres.resize(0);

    // Enumerate testpoints on all axis

    Vector<Vector<gentype> > gridmarks(dim);
    Vector<int> isAxisRandom(llocvarsType); // This will be zero for all exis except the random ones

    isAxisRandom = 0;

    for ( i = 0 ; i < dim ; ++i )
    {
        gridmarks("&",i).resize(altnumPts(i));

        isAxisRandom("&",i) = llocvarsType(i) ? ( ( llocdistMode(i) == 4 ) ? 1 : 2 ) : 0;

        t = ( altnumPts(i) == 1 ) ? 0.5 : 0;

        for ( j = 0 ; j < altnumPts(i) ; ++j,(t+=(1/((double) altnumPts(i)-1))) )
        {
            randufill(s);

            gridmarks("&",i)("&",j) = ( llocdistMode(i) == 3 ) ? s : t;
        }

        // For randomised we sort ascending to make inter-grid jumps
        // as small as possible

        if ( llocdistMode(i) == 3 )
        {
            int kmin = 0;

            for ( j = 0 ; j < gridmarks(i).size()-1 ; ++j )
            {
                kmin = j;

                for ( k = j+1 ; k < gridmarks(i).size() ; ++k )
                {
                    if ( gridmarks(i)(k) < gridmarks(i)(kmin) )
                    {
                        kmin = k;
                    }
                }

                gridmarks("&",i).squareswap(j,kmin);
            }
        }
    }

    // Grab centre vector.  This will act as a secondary selector if we
    // have multiple results that have the same target (assuming centrality
    // is good, which it generally is).

    Vector<gentype> xcentre(dim);

    for ( i = 0 ; i < dim ; ++i )
    {
        xcentre("&",i) = gridmarks(i)((gridmarks(i).size()-1)/2);
    }

    // Main loop.

    Vector<int> ii(dim);
    Vector<int> idir(dim);

    Vector<gentype> locxres(dim);
    gentype locfres;
    gentype locsres;
    double locdres;

    // Initialise loop counters.  Direction is kept to minimise jump between
    // loop rows - basically we only want to change one variable at a time,
    // so count goes from 0 to N-1, then N-1 to 0, and so on

    for ( i = 0 ; i < dim ; ++i )
    {
        ii("&",i) = 0;
        idir("&",i) = +1;
    }

    int isopt = 0;
    int isfirst = 1;

    j = 0;

    double loczoomFact = zoomFact;
    int timeout = 0;
    double *uservars[] = { &maxtraintime, &loczoomFact, nullptr };
    const char *varnames[] = { "traintime", "loczoomFact", nullptr };
    const char *vardescr[] = { "Maximum training time (seconds, 0 for unlimited)", "Zoom factor", nullptr };

    while ( !killSwitch && !isopt && !timeout )
    {
        // Find position

        for ( i = 0 ; i < dim ; ++i )
        {
            locxres("&",i) = gridmarks(i)(ii(i));

            if ( isAxisRandom(i) == 1 )
            {
                randufill(s);

                locxres("&",i) = s;
            }
        }

        // Evaluate function

        locfres.force_int() = 0;

        (*fn)(locfres,locxres,fnarg);
        errstream() << "\n";

        locsres.force_vector().resize(2);
        locsres("&",0) = nullgentype();

        if ( locfres.isValSet() )
        {
            // Variance is also here, so grab it

            gentype mures((locfres.all())(0));
            gentype varres((locfres.all())(1));

            locfres = mures;
            locsres("&",0) = varres;
        }

        // Find distance to centre

        locxres -= xcentre;
        locdres = norm2(locxres);
        locxres += xcentre;

        locsres("&",1) = locdres;

        // Record if required

        if ( 1 )
        {
            Vector<gentype> loccres;

            allxres.append(allxres.size(),locxres);
            allfres.append(allfres.size(),locfres);
            allcres.append(allcres.size(),loccres);
            allmres.append(allmres.size(),locfres);
            allsres.append(allsres.size(),locsres);
            s_score.append(s_score.size(),1.0);
        }

        // Test optimality

        if ( isfirst || ( locfres < fres ) || ( ( locfres == fres ) && ( locdres < dres ) ) )
        {
            isfirst = 0;

            Xres = locxres;
            fres = locfres;
            dres = locdres;
            ires = j;
        }

        // Terminate if hardmin/max found, increment otherwise

        if ( (double) locfres <= hardmin )
        {
            killSwitch = 1;
        }

        else if ( (double) locfres >= hardmax )
        {
            killSwitch = 1;
        }

        else
        {
            isopt = 1;

            for ( i = 0 ; i < dim ; ++i )
            {
                if ( idir(i) == +1 )
                {
                    if ( ii(i) == gridmarks(i).size()-1 )
                    {
                        idir("&",i) = -1;
                    }

                    else
                    {
                        ++(ii("&",i));
                        isopt = 0;
                        break;
                    }
                }

                else
                {
                    if ( ii(i) == 0 )
                    {
                        idir("&",i) = +1;
                    }

                    else
                    {
                        --(ii("&",i));
                        isopt = 0;
                        break;
                    }
                }
            }

            if ( !timeout )
            {
                timeout = kbquitdet("Grid optimisation",uservars,varnames,vardescr);
            }
        }

        ++j;
    }

    NiceAssert( !isfirst );

    // If requested zoom, recurse and incorporate new results

    if ( numZooms && !killSwitch )
    {
        Vector<gentype> subxres;
        gentype subfres(0);
        Vector<gentype> subcres;
        gentype subFres;
        gentype submres;
        gentype subsres;
        int subires = 0;
        int submInd = 0;
        Vector<Vector<Vector<gentype> > > suballxres;
        Vector<Vector<gentype> >          suballfres;
        Vector<Vector<Vector<gentype> > > suballcres;
        Vector<Vector<gentype> >          suballmres;
        Vector<Vector<gentype> >          suballFres;
        Vector<Vector<gentype> >          suballsres;
        Vector<Vector<double> >           subs_score;
        Vector<Vector<int> >              subis_feas;
        Vector<gentype> subxmin(xmin);
        Vector<gentype> subxmax(xmax);

        double width;
        double centre;

        // First up we need to zoom and 

        for ( i = 0 ; i < dim ; ++i )
        {
            // real zoom always

            width  = ( (double) xmax(i) ) - ( (double) xmin(i) );
            centre = (double) Xres(i);
            width *= loczoomFact;

            subxmin("&",i) = centre - width/2;
            subxmax("&",i) = centre + width/2;

            // Need to make sure we stay within the feasible region

            subxmin("&",i) = ( subxmin(i) > xmin(i) ) ? subxmin(i) : xmin(i);
            subxmax("&",i) = ( subxmax(i) < xmax(i) ) ? subxmax(i) : xmax(i);
        }

        // Recurse

        GridOptions locgopts = *this;
        --(locgopts.numZooms);

        Vector<int> &subdistMode(llocdistMode);
        Vector<int> &subvarsType(llocvarsType);

        subdistMode = 0; // So we don't get twice-applied whatever scales.
        subvarsType = 1; // So we only round to integer once.

        Vector<gentype> subxignore;                      // We only really care about "raw" here
        Vector<Vector<Vector<gentype> > > suballxignore; // We only really care about "raw" here

        gentype dummymeanfres, dummyvarfres;
        gentype dummymeanFres, dummyvarFres;
        gentype dummymeanires, dummyvarires;
        gentype dummymeantres, dummyvartres;
        gentype dummymeanTres, dummyvarTres;

        Vector<gentype> dummymeanallfres, dummyvarallfres;
        Vector<gentype> dummymeanallFres, dummyvarallFres;
        Vector<gentype> dummymeanallmres, dummyvarallmres;

        res |= locgopts.optim(dim,subxignore,subxres,subfres,subcres,subFres,submres,subsres,subires,submInd,
                              suballxignore,suballxres,suballfres,suballcres,suballFres,suballmres,suballsres,subs_score,subis_feas,
                              subxmin,subxmax,subdistMode,subvarsType,fn,fnarg,killSwitch,1,
                              dummymeanfres,dummyvarfres,dummymeanFres,dummyvarFres,dummymeanires,dummyvarires,dummymeantres,dummyvartres,dummymeanTres,dummyvarTres,dummymeanallfres,dummyvarallfres,dummymeanallFres,dummyvarallFres,dummymeanallmres,dummyvarallmres);

        // Incorporate results

        if ( subfres < fres )
        {
            Xres   = subxres;
            fres   = subfres;
            cres   = subcres;
            Fres   = subFres;
            mres   = submres;
            sres   = subsres;
            ires   = subires+allfres.size();
//            mInd   = submInd;
        }

        if ( 1 )
        {
            allxres.append(allxres.size(),suballxres(0));
            allfres.append(allfres.size(),suballfres(0));
            allcres.append(allcres.size(),suballcres(0));
            allFres.append(allFres.size(),suballFres(0));
            allmres.append(allmres.size(),suballmres(0));
            allsres.append(allsres.size(),suballsres(0));
            s_score.append(s_score.size(),subs_score(0));
            is_feas.append(is_feas.size(),subis_feas(0));
        }
    }

    mres = fres;
    cres.resize(0);
    Fres = ires;

    allFres.resize(allfres.size());

    for ( int i = 0 ; i < allFres.size() ; ++i )
    {
        allFres("&",i) = i;
    }

    is_feas.resize(allfres.size());
    is_feas = 1;

    return res;
}

