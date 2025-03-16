
#include "basefn.hpp"
#include "adam.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "randfun.hpp"

// Return: 0  on success
//         -1 if timeout/too many iterations
//         >0 error in objective evaluation
//
// set killSwitch at any time to halt optimisation
// useadam: 0 for gradient, 1 for ADAM

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 10000

#define NOPROGRESSLIMIT 1e-6
#define NOPROGRESSLIMITB 1e-6
#define NOPROGRESSEXIT 10000

#define BADSTEPSCALE 0.01

int ADAMopt(double &objres, Vector<double> &x, int (*calcObj)(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs),
            const char *usestring, svmvolatile int &killSwitch, double lr, void *objargs, ADAMscratch &scratchpad, int useadam,
            stopCond sc, //double Opttol, int maxitcnt, double maxtraintime, double traintimeend,
            double abeta1, double abeta2, double aeps, int xsgn, double schedconst, double normmax, double minv, int nsgn,
            double (*stepscale)(const Vector<double> &x, const Vector<double> &dx, void *), void *stepscalearg)
{
//errstream() << "phantomxyzabc: useadam = " << useadam << "\n";
//errstream() << "phantomxyzabc: maxitcnt = " << maxitcnt << "\n";
//errstream() << "phantomxyzabc: maxtraintime = " << maxtraintime << "\n";
//errstream() << "phantomxyzabc: abeta1 = " << abeta1 << "\n";
//errstream() << "phantomxyzabc: abeta2 = " << abeta2 << "\n";
//errstream() << "phantomxyzabc: aeps = " << aeps << "\n";
//errstream() << "phantomxyzabc: xsgn = " << xsgn << "\n";
//errstream() << "phantomxyzabc: schedconst = " << schedconst << "\n";
//errstream() << "phantomxyzabc: normmax = " << normmax << "\n";
//errstream() << "phantomxyzabc: minv = " << minv << "\n";
    NiceAssert( !x.infsize() );
//errstream() << "phantomx -1: " << absinf(x);

    if ( !x.size() )
    {
        return 0;
    }

    if ( !xsgn )
    {
        x = 0.0;

        return 0;
    }

    //Vector<double> xstart(x);

    int againcnt = 0;

goagain:
    int i;

    for ( i = 0 ; i < ( ( nsgn == -1 ) ? x.size() : nsgn ) ; ++i )
    {
        if ( ( xsgn == 1 ) && ( x(i) < ZSTOL ) )
        {
            x("&",i) = ZSTOL;
        }

        else if ( ( xsgn == -1 ) && ( x(i) > -ZSTOL ) )
        {
            x("&",i) = -ZSTOL;
        }

        else if ( ( xsgn == 3 ) && ( x(i) < minv ) )
        {
            x("&",i) = minv;
        }

        else if ( ( xsgn == 4 ) && ( x(i) < ZSTOL ) )
        {
            x("&",i) = ZSTOL;
        }

        else if ( ( xsgn == 4 ) && ( x(i) > 1/minv ) )
        {
            x("&",i) = 1/minv;
        }
    }

    if ( normmax > 0 )
    {
        double xmag = abs2(x);

        if ( xmag > normmax )
        {
            x *= normmax/xmag;
        }
    }

    else if ( normmax < 0 )
    {
        retVector<double> tmpvaa;

        double xmag = abs2(x(0,1,x.size()-2,tmpvaa));

        if ( xmag > -normmax )
        {
            x *= -normmax/xmag;
        }
    }
//errstream() << "phantomx 0: " << absinf(x);

    NiceAssert( lr > 0 );

    //Vector<double> gradx(x);
    //Vector<double> dx(x);
    //Vector<double> xbest(x);

    //Vector<double> am(x);
    //Vector<double> av(x);

    //Vector<double> amhat(x);
    //Vector<double> avhat(x);

    int xdim = x.size();

    scratchpad.resize(xdim);

    Vector<double> &gradx     = scratchpad.gradx;
    Vector<double> &gradgradx = scratchpad.gradgradx;
    Vector<double> &dx        = scratchpad.dx;
    Vector<double> &xbest     = scratchpad.xbest;

    Vector<double> &am = scratchpad.am;
    Vector<double> &av = scratchpad.av;

    Vector<double> &amhat = scratchpad.amhat;
    Vector<double> &avhat = scratchpad.avhat;

    Vector<double> &xold = scratchpad.xold;

    xbest = x;

    am = 0.0;
    av = 0.0;

    double vlr = lr;
    double modlr = lr;
    double objval = 0;
    double soltol = sc.tol;

    double *uservars[] = { &(sc.maxitcnt), &(sc.maxruntime), &(sc.runtimeend), &vlr, &soltol, &abeta1, &abeta2, &aeps, &schedconst, &normmax, nullptr };
    const char *varnames[] = { "maxitcnt", "maxruntime", "runtimeend", "vlr", "soltol", "abeta1", "abeta2", "aeps", "schedconst", "normmax", nullptr };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Absolute training time end (relative, secs, -1 for na)", "Step scale (vlr)", "Solution tolerance", "abeta1", "abeta2", "aeps", "schedconst", "normmax", nullptr };

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    size_t itcnt = 0;
    int timeout = 0;
    int bailout = 0;
    int stopout = 0;

    int kickstart = 0;
    bool firststep = true;
    int isopt = 0;
    double bestres = 1000;
    double vgradmin = 0;
    int badprogress = 0;
    double badstep = 0;
    //double uphillstep = 0;
    double vgradminmin = 0;
    double prevobjval = 0;
    double gradmag = 0;
    double oldgradmag = 0;
    int ignorestep = 0;


    // Obscure note: in c++, if maxitcnt is a double then !maxitcnt is
    // true if maxitcnt == 0, false otherwise.  This is defined in the
    // standard, and the reason the following while statement will work.

    //char dummybuff[1024];

    while ( !killSwitch && sc.stillrun(itcnt,isopt,timeout,bailout,stopout) )
    {
        // Calculate gradient

        int nostop = 0;

//errstream() << "phantomx 1: " << abs2(x) << "\n";
        if ( !itcnt ) { errstreamunlogged("...\b\b\b"); }
        ignorestep = calcObj(objval,x,gradx,gradgradx,killSwitch,&nostop,objargs);
        if ( !itcnt ) { errstreamunlogged("<<<\b\b\b"); }

        // Calculate step (unscaled)

        dx = gradx;
//errstream() << "phantomx 2: " << abs2(dx) << "\n";

        // Preliminary scaling (Newton-ish)

        dx /= absinf(gradgradx);
        //dx /= gradgradx; - experimentally, the inf-norm scaling is *massively* faster than the elementwise version (or no scaling at all)!

        // Best result?

        vgradmin = absinf(dx);

        //if ( firststep || ( vgradmin < bestres ) ) - note that the version below insists on gradient and objective superiority.  This is
        //important for certain corner cases, or where the objective is estimated (or simply set as a decreasing counter if no estimate is available).
        if ( !ignorestep && !nostop && ( firststep || ( ( objval < bestres ) && ( vgradmin < vgradminmin ) ) ) )
        {
            xbest   = x;
            //bestres = vgradmin;
            bestres = objval;

            vgradminmin = vgradmin;

            badstep /= 1.2;
            //badstep -= 0.02;
            badstep = ( badstep < 0 ) ? 0 : badstep;
        }

        else if ( !ignorestep && !nostop && ( prevobjval < objval ) )
        {
            ////++badstep;
            //badstep += ( xdim > 200 ) ? 10 : xdim/20;
            badstep += 10;
        }

/*        if ( !ignorestep && !nostop && ( objval > prevobjval ) )
        {
            ++uphillstep;

            if ( uphillstep > 10 )
            {
                vlr /= 1.5;
                uphillstep /= 2;
            }
        }

        else if ( !ignorestep && !nostop && ( prevobjval < objval ) )
        {
            --uphillstep;
            uphillstep = ( uphillstep > 0 ) ? uphillstep : 0.0;
        }
*/

        // Test optimality

        isopt = ( !ignorestep && !nostop && ( vgradmin <= soltol ) );

        if ( !ignorestep && !isopt )
        {
            // Take scaled negative gradient descent step

            double realstepsize = 0;

            if ( useadam )
            {
                //if ( ADAM )
                {
                    am.scale(abeta1);
                    am.scaleAdd(1-abeta1,dx);

                    av.scale(abeta2);
                    av.sqscaleAdd(1-abeta2,dx);

                    amhat = am;
                    amhat.scale(1/(1-pow(abeta1,itcnt+1)));

                    avhat = av;
                    avhat.scale(1/(1-pow(abeta2,itcnt+1)));
                }

/*
                else if ( ADAMAX )
                {
                    am.scale(abeta1);
                    am.scaleAdd(1-abeta1,dx);

                    amhat = am;
                    amhat.scale(1/(1-pow(abeta1,itcnt+1)));

                    for ( i = 0 ; i < xdim ; ++i )
                    {
                        avhat("&",i) = ( avhat(i) > av(i) ) ? avhat(i) : av(i);

                        avhat.scale(1/(1-pow(abeta2,itcnt+1)));
                    }
                }

                else if ( AMSGRAD )
                {
                    am.scale(abeta1);
                    am.scaleAdd(1-abeta1,dx);

                    av.scale(abeta2);
                    av.sqscaleAdd(1-abeta2,dx);

                    for ( i = 0 ; i < xdim ; ++i )
                    {
                        amhat("&",i) = am(i);
                        avhat("&",i) = ( avhat(i) > av(i) ) ? avhat(i) : av(i);

                        avhat.scale(1/(1-pow(abeta2,itcnt+1)));
                    }
                }
//errstream() << "phantomx 3: " << absinf(amhat);
//errstream() << "phantomx 4: " << absinf(avhat);
*/

                double maxrgrad = absinf(gradx);
                maxrgrad = ( maxrgrad < 0.5 ) ? 0.5 : maxrgrad;

                for ( i = 0 ; i < xdim ; ++i )
                {
                    xold("&",i) = x(i);

                    modlr = vlr/(pow(itcnt+1,schedconst)*sqrt(avhat(i))+aeps);
                    //modlr = vlr/((1+(badstep*BADSTEPSCALE))*(pow(itcnt+1,schedconst)*(2*maxrgrad)*sqrt(avhat(i)))+aeps);
                    //modlr = vlr/((pow(itcnt+1,schedconst)*(2*maxrgrad)*sqrt(avhat(i)))+aeps);

                    dx("&",i) = -modlr*amhat(i);
                }

                double dxscale = stepscale ? stepscale(x,dx,stepscalearg) : 1.0;

                for ( i = 0 ; i < xdim ; ++i )
                {
                    x("&",i) += dxscale*dx(i);
                }
//errstream() << "phantomx 5: " << absinf(x);

                for ( i = 0 ; i < ( ( nsgn == -1 ) ? xdim : nsgn ) ; ++i )
                {
                    if ( ( xsgn == 1 ) && ( x(i) < ZSTOL ) )
                    {
                        x("&",i) = ZSTOL;
                    }

                    else if ( ( xsgn == -1 ) && ( x(i) > -ZSTOL ) )
                    {
                        x("&",i) = -ZSTOL;
                    }

                    else if ( ( xsgn == 3 ) && ( x(i) < minv ) )
                    {
                        x("&",i) = minv;
                    }

                    else if ( ( xsgn == 4 ) && ( x(i) < ZSTOL ) )
                    {
                        x("&",i) = ZSTOL;
                    }

                    else if ( ( xsgn == 4 ) && ( x(i) > 1/minv ) )
                    {
                        x("&",i) = 1/minv;
                    }
                }

                if ( normmax > 0 )
                {
                    double xmag = abs2(x);

                    if ( xmag > normmax )
                    {
                        x *= normmax/xmag;
                    }
                }

                else if ( normmax < 0 )
                {
                    retVector<double> tmpvaa;

                    double xmag = abs2(x(0,1,x.size()-2,tmpvaa));

                    if ( xmag > -normmax )
                    {
                        x *= -normmax/xmag;
                    }
                }

                for ( i = 0 ; i < xdim ; ++i )
                {
                    realstepsize += (xold(i)-x(i))*(xold(i)-x(i));
                }
            }

            else
            {
                //x.scaleAdd(-vlr,dx);

                modlr = vlr/pow(itcnt+1,schedconst);
//errstream() << "phantomx adam modlr = " << vlr << "/pow(" << itcnt+1 << "^" << schedconst << ") = " << modlr << "\n";

                for ( i = 0 ; i < xdim ; ++i )
                {
                    xold("&",i) = x(i);

                    dx("&",i) = -modlr*dx(i);
                }

                double dxscale = stepscale ? stepscale(x,dx,stepscalearg) : 1.0;

                for ( i = 0 ; i < xdim ; ++i )
                {
                    x("&",i) += dxscale*dx(i);
                }

                for ( i = 0 ; i < ( ( nsgn == -1 ) ? xdim : nsgn ) ; ++i )
                {
                    if ( ( xsgn == 1 ) && ( x(i) < ZSTOL ) )
                    {
                        x("&",i) = ZSTOL;
                    }

                    else if ( ( xsgn == -1 ) && ( x(i) > -ZSTOL ) )
                    {
                        x("&",i) = -ZSTOL;
                    }

                    else if ( ( xsgn == 3 ) && ( x(i) < minv ) )
                    {
                        x("&",i) = minv;
                    }

                    else if ( ( xsgn == 4 ) && ( x(i) < ZSTOL ) )
                    {
                        x("&",i) = ZSTOL;
                    }

                    else if ( ( xsgn == 4 ) && ( x(i) > 1/minv ) )
                    {
                        x("&",i) = 1/minv;
                    }
                }

                if ( normmax > 0 )
                {
                    double xmag = abs2(x);

                    if ( xmag > normmax )
                    {
//errstream() << "phantomx adam xscale = " << normmax << "/" << xmag << " = " << normmax/xmag << "\n";
                        x *= normmax/xmag;
                    }
                }

                else if ( normmax < 0 )
                {
                    retVector<double> tmpvaa;

                    double xmag = abs2(x(0,1,x.size()-2,tmpvaa));

                    if ( xmag > -normmax )
                    {
//errstream() << "phantomx adam xscale (bias) = " << -normmax << "/" << xmag << " = " << -normmax/xmag << "\n";
                        x *= -normmax/xmag;
                    }
                }

                for ( i = 0 ; i < xdim ; ++i )
                {
                    realstepsize += (xold(i)-x(i))*(xold(i)-x(i));
                }
            }

            oldgradmag = gradmag;
            gradmag = absinf(gradx);

            ////if ( sqrt(realstepsize) < NOPROGRESSLIMIT )
            //if ( !firststep && ( sqrt(realstepsize) < NOPROGRESSLIMIT ) && ( gradmag-oldgradmag > NOPROGRESSLIMITB ) )
            if ( !firststep && ( fabs(prevobjval-objval) <= NOPROGRESSLIMIT ) && ( fabs(gradmag-oldgradmag) <= NOPROGRESSLIMITB ) )
            {
                ++badprogress;
            }

            else
            {
                badprogress = 0;
            }

            if ( badprogress > NOPROGRESSEXIT )
            {
                badprogress = 0;
                bailout = 1;

                errstream() << "~";

                ++againcnt;

                if ( againcnt < 5 )
                {
                    vlr /= 1.5;

                    //x = xstart;
                    //x = 1.0;

                    Vector<double> xdisturb(x);
                    randufill(xdisturb); // uniform random [0,1]
                    xdisturb *= 0.02;
                    x += xdisturb;

                    for ( i = 0 ; i < ( ( nsgn == -1 ) ? xdim : nsgn ) ; ++i )
                    {
                        if ( ( xsgn == 1 ) && ( x(i) < ZSTOL ) )
                        {
                            x("&",i) = ZSTOL;
                        }

                        else if ( ( xsgn == -1 ) && ( x(i) > -ZSTOL ) )
                        {
                            x("&",i) = -ZSTOL;
                        }

                        else if ( ( xsgn == 3 ) && ( x(i) < minv ) )
                        {
                            x("&",i) = minv;
                        }

                        else if ( ( xsgn == 4 ) && ( x(i) < ZSTOL ) )
                        {
                            x("&",i) = ZSTOL;
                        }

                        else if ( ( xsgn == 4 ) && ( x(i) > 1/minv ) )
                        {
                            x("&",i) = 1/minv;
                        }
                    }

                    goto goagain;
                }

                else
                {
                    errstream() << "\nBailout in ADAM due to progress flatline\n";

                    bailout = 1;
                }
            }
        }

        if ( !ignorestep )
        {
            prevobjval = objval;

            firststep = false;
        }

        sc.testopt(itcnt,isopt,timeout,bailout,stopout,kickstart,FEEDBACK_CYCLE,MAJOR_FEEDBACK_CYCLE,start_time,curr_time,usestring,uservars,varnames,vardescr);
    }

    // If we exited due to time/iteration out then grab best solution so far

    objres = objval;

//errstream() << "phantomxyz adam x = " << x << "\n";
    if ( !isopt )
    {
        objres = bestres;
        x      = xbest;
    }
//errstream() << "phantomxyz adam xbest = " << x << "\n";

    return isopt ? 0 : ( bailout ? bailout : -1 );
}
