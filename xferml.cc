
//
// Transfer learning setup
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "xferml.hpp"
#include "svm_scalar.hpp"
#include "svm_binary.hpp"
#include "svm_kconst.hpp"
#include "mlcommon.hpp"
#include "numbase.hpp"
#include "nlopt_neldermead.hpp"
#include "nlopt_slsqp.hpp"
#include "randfun.hpp"

#define FEEDBACK_CYCLE 10
#define MAJOR_FEEDBACK_CYCLE 50
#define NEWTSCALE 1.0
#define KWEIGHT_THRES 512

Vector<gentype> &randitall(Vector<gentype> &x, int randtype, double randvari, SparseVector<gentype> &altvar);
Vector<gentype> &randitall(Vector<gentype> &x, int randtype, double randvari, SparseVector<gentype> &altvar)
{
    int i;
    int d = x.size();

    for ( i = 0 ; i < d ; ++i )
    {
        if ( randtype == 0 )
        {
            randnfill(x("&",i).force_double());
        }

        else if ( randtype == 1 )
        {
            randufill(x("&",i));
        }

        else if ( randtype == 2 )
        {
            randrfill(x("&",i).force_double());
        }

        else
        {
            randrfill(x("&",i).force_double());
        }

        if ( randtype != 3 )
        {
            if ( randvari <= 0 )
            {
                x("&",i) *= sqrt((double) altvar(i));
            }

            else
            {
                x("&",i) *= sqrt(randvari);
            }
        }
    }

    return x;
}



// Gradient/hessian calculator (also returns raw result if calcres set NZ)

double evalfandgrad(double C, int &Ntot, Matrix<double> &hess, Vector<double> &grad, const SVM_Generic &core, const Vector<ML_Base *> &cases, int n,
                Vector<double> &tempi, double &rawavediag, const Vector<double> &caseweight, int calchess,
                Vector<double> &tmip, Matrix<double> &wpgcache, Vector<double> &dedK, int calcres, int calcgrad, int regtype, Vector<double> &allR);
double evalfandgrad(double C, int &Ntot, Matrix<double> &hess, Vector<double> &grad, const SVM_Generic &core, const Vector<ML_Base *> &cases, int n,
                Vector<double> &tempi, double &rawavediag, const Vector<double> &caseweight, int calchess,
                Vector<double> &tmip, Matrix<double> &wpgcache, Vector<double> &dedK, int calcres, int calcgrad, int regtype, Vector<double> &allR)
{
/*
 Kijpq = K(xi,xj,zp,zq)

 Primary goal:

 R = sum_i e(i)
 dR/dbeta_p = sum_i dedg(i) d/dbeta_p ( sum_j alpha_j sum_rs beta_r beta_s Kijrs )
            = sum_i dedg(i) ( sum_j alpha_j 2 sum_s beta_s Kijps )
 d2R/dbeta_p dbeta_q = sum_i d2edg2(i) ( sum_j alpha_j 2 sum_s beta_s Kijps ) ( sum_j alpha_j 2 sum_s beta_s Kijqs )
                     + sum_i dedg(i) ( sum_j alpha_j 2 Kijpq )

 R = sum_i e(i)
 dR/dbeta_p = sum_i dedg(i) ( sum_j alpha_j 2 sum_s beta_s Kijps )
 d2R/dbeta_p dbeta_q = sum_i tempii ( sum_j alpha_j 2 sum_s beta_s Kijps ) ( sum_j alpha_j 2 sum_s beta_s Kijqs )
                     + sum_i dedg(i) ( sum_j alpha_j 2 Kijpq )

 R = sum_i e(i)
 dR/dbeta_p = sum_i ( sum_j dedK_ij 2 sum_s beta_s Kijps )
 d2R/dbeta_p dbeta_q = sum_i tempii ( sum_j alpha_j 2 sum_s beta_s Kijps ) ( sum_j alpha_j 2 sum_s beta_s Kijqs )
                     + sum_i ( sum_j dedK_ij 2 Kijpq )

 R = sum_i e(i)
 dR/dbeta_p = sum_ijs 2 dedK_ij beta_s Kijps
 d2R/dbeta_p dbeta_q = sum_i tempii ( sum_j alpha_j 2 sum_s beta_s Kijps ) ( sum_j alpha_j 2 sum_s beta_s Kijqs )
                     + sum_ij 2 dedK_ij Kijpq

 R = sum_i e(i)
 dR/dbeta_p = sum_ijq 2 dedK_ij beta_q Kijpq
 d2R/dbeta_p dbeta_q = sum_i tempii tmip_ip tmip_iq
                     + sum_ij 2 dedK_ij Kijpq

 where: tempii = d2edg2(i)
        dedK_ij = dedg(i) alpha_j
        tmip_ip = sum_jq 2 alpha_j beta_q Kijpq




 Regularisation:

 Want inheritted kernel to approximately satisfy K(xi,xi) = 1

 Q = C/2 sum_i ( sum_st beta_s beta_t Kiist - 1 )^2
 dQ/dbeta_p = 2C sum_i ( sum_st beta_s beta_t Kiist - 1 ) ( sum_t beta_t Kiipt )
 d2Q/dbeta_p dbeta_q = 2C sum_i ( sum_st beta_s beta_t Kiist - 1 ) Kiipq + 4C sum_i ( sum_s beta_s Kiisq  ) ( sum_t beta_t Kiipt )

 Q = C/2 sum_i ( kdiag_i - 1 )^2
 dQ/dbeta_p = 2C sum_i ( kdiag_i - 1 ) tempi_p
 d2Q/dbeta_p dbeta_q = 2C sum_i ( kdiag_i - 1 ) Kiipq + 4C sum_i tempi_p tempi_q


 where: tempi_p = sum_q beta_q Kiipq
*/

//errstream() << "phantomxyz 0 wtf" << caseweight << "\n";
    double R = 0;

    allR = 0.0;

    int p,q,i,j,k;
    int M = cases.size();
    Ntot = 0;

    if ( calcgrad )
    {
        grad = 0.0;
    }

    if ( calchess )
    {
        hess = 0.0;
    }

    double tempK;
    double tempii;
    double wpg;

    rawavediag = 0.0;

    double Rstep;

    for ( k = 0 ; k < M ; ++k )
    {
        const ML_Base &kcase = *(cases(k));
        int N = kcase.N();

        Ntot += N;

        for ( i = 0 ; i < N ; ++i )
        {
            rawavediag += kcase.k2diag(i);

            if ( calcres )
            {
                // Primary goal: R = sum_i e(i)
                Rstep = ((kcase.eTrainingVector(i))*caseweight(k));

                R           += Rstep;
                allR("&",k) += Rstep;

                if ( regtype == 1 )
                {
                    // Regularisation: Q = C/2 sum_i ( kdiag_i - 1 )^2
                    Rstep = (C/2)*pow(kcase.k2diag(i)-1,2);

                    R           += Rstep;
                    allR("&",M) += Rstep;
                }
            }

            if ( ( calcgrad || calchess ) && kcase.alphaState()(i) )
            {
                tempi = 0.0;

                if ( calcgrad )
                {
                    kcase.dedKTrainingVector(dedK,i);
                    dedK *= caseweight(k);
                }

                if ( calchess )
                {
                    kcase.d2edg2TrainingVector(tempii,i);
                    tempii *= caseweight(k);

                    tmip = 0.0;
                }

                for ( j = 0 ; j < N ; ++j )
                {
                    for ( q = 0 ; q < n ; ++q )
                    {
                        for ( p = 0 ; p <= q ; ++p )
                        {
                            wpgcache("&",q,p) = tempK = core.K4(-42-i,-42-j,p,q,nullptr,&(kcase.x(i)),&(kcase.x(j)),&(core.x(p)),&(core.x(q)),&(kcase.xinfo(i)),&(kcase.xinfo(j)),&(core.xinfo(p)),&(core.xinfo(q)));
                        }
                    }

                    for ( q = 0 ; q < n ; ++q )
                    {
                        for ( p = 0 ; p < n ; ++p )
                        {
                            wpg = ( p <= q ) ? wpgcache(q,p) : wpgcache(p,q);

                            if ( i == j )
                            {
                                tempi("&",p) += (core.alphaR())(q)*wpg;
                            }

                            if ( calchess )
                            {
                                tmip("&",p) += 2*(kcase.alphaVal(j))*(core.alphaR())(q)*wpg;
                            }
                        }
                    }

                    for ( q = 0 ; q < n ; ++q )
                    {
                        for ( p = 0 ; p < n ; ++p )
                        {
                            wpg = ( p <= q ) ? wpgcache(q,p) : wpgcache(p,q);

                            if ( calcgrad )
                            {
                                // Primary goal: dR/dbeta_p = sum_ijq 2 dedK_ij beta_q Kijpq
                                grad("&",p) += 2*dedK(j)*(core.alphaR())(q)*wpg;
                            }

                            if ( calchess )
                            {
                                tmip("&",p) += 2*(kcase.alphaVal(j))*(core.alphaR())(q)*wpg;

                                // Primary goal: d2R/dbeta_p dbeta_q = ... + sum_ij 2 dedK_ij Kijpq
                                hess("&",p,q) += 2*dedK(j)*wpg;
                            }
                        }
                    }

                    if ( regtype == 1 )
                    {
                        if ( i == j )
                        {
                            for ( p = 0 ; p < n ; ++p )
                            {
                                if ( calcgrad )
                                {
                                    // Regularisation: dQ/dbeta_p = 2C sum_i ( kdiag_i - 1 ) tempi_p
                                    grad("&",p) += 2*C*(kcase.k2diag(i)-1)*tempi(p);
                                }

                                for ( q = 0 ; q < n ; ++q )
                                {
                                    wpg = ( p <= q ) ? wpgcache(q,p) : wpgcache(p,q);

                                    // Regularisation: d2Q/dbeta_p dbeta_q = 2C sum_i ( kdiag_i - 1 ) Kiipq + ...
                                    hess("&",p,q) += 2*C*(kcase.k2diag(i)-1)*wpg;
                                }
                            }
                        }
                    }
                }

                if ( calchess )
                {
                    // Primary goal: d2R/dbeta_p dbeta_q = sum_i tempii tmip_ip tmip_iq + ...
                    hess.rankone(tempii,tmip,tmip);

                    if ( regtype == 1 )
                    {
                        // Regularisation: d2Q/dbeta_p dbeta_q = ... + 4C sum_i tempi_p tempi_q
                        hess.rankone(4*C,tempi,tempi);
                    }
                }
            }
        }
    }

    if ( regtype == 2 )
    {
        Rstep = C*norm2(core.alphaR())/2.0;

        R           += Rstep;
        allR("&",M) += Rstep;

        if ( calcgrad )
        {
            grad.scaleAdd(C,core.alphaR());
        }

        if ( calchess )
        {
            hess.diagoffset(C);
        }
    }

    else if ( regtype == 3 )
    {
        Rstep = C*norm1(core.alphaR());

        R           += Rstep;
        allR("&",M) += Rstep;

        if ( calcgrad )
        {
            for ( p = 0 ; p < n ; ++p )
            {
                if ( (core.alphaR())(p) > 0 )
                {
                    grad("&",p) += C;
                }

                else if ( (core.alphaR())(p) < 0 )
                {
                    grad("&",p) -= C;
                }
            }
        }
    }

    else if ( regtype == 4 )
    {
        // R                   += 1/2 ( sum_i |beta_i| - 1 )^2
        // dR/dbeta_p          += ( sum_i |beta_i| - 1 ) * sgn(beta_p)
        // d2R/dbeta_p.dbeta_q += sgn(beta_p).sgn(beta_q)

        double temp = (abssum(core.alphaR())-1);

        Rstep = C*temp*temp/2;

        R           += Rstep;
        allR("&",M) += Rstep;

        if ( calcgrad )
        {
            for ( p = 0 ; p < n ; ++p )
            {
                if ( (core.alphaR())(p) > 0 )
                {
                    grad("&",p) += C*temp;
                }

                else if ( (core.alphaR())(p) < 0 )
                {
                    grad("&",p) -= C*temp;
                }
            }
        }

        if ( calchess )
        {
            for ( p = 0 ; p < n ; ++p )
            {
                for ( q = 0 ; q < n ; ++q )
                {
                    if ( (core.alphaR())(p)*(core.alphaR())(q) > 0 )
                    {
                        hess("&",p,q) += C;
                    }

                    else if ( (core.alphaR())(p)*(core.alphaR())(q) < 0 )
                    {
                        hess("&",p,q) -= C;
                    }
                }
            }
        }
    }

    else if ( regtype == 5 )
    {
        // R                   += 1/2 ( sum_i beta_i - 1 )^2
        // dR/dbeta_p          += ( sum_i beta_i - 1 )
        // d2R/dbeta_p.dbeta_q += 1

        double temp = (sum(core.alphaR())-1);

        Rstep = C*temp*temp/2;

        R           += Rstep;
        allR("&",M) += Rstep;

        if ( calcgrad )
        {
            grad += C*temp;
        }

        if ( calchess )
        {
            hess += C;
        }
    }

    rawavediag /= Ntot;

    if ( calchess )
    {
        for ( p = 0 ; p < n ; ++p )
        {
            hess("&",p,p) += ( hess(p,p) < 1e-6 ) ? 1e-6 : hess(p,p);
        }
    }

//errstream() << "phantomxyz 1 wtf" << R << "\n";
    return R;
}








int xferMLtrain0(svmvolatile int &killSwitch,
                SVM_Generic &core, Vector<ML_Base *> &cases,
                int n, int maxitcntint, double xmtrtime, double soltol, const Vector<double> &caseweight, int usenewton, double lr,
                int randtype, double C, int regtype, double randvari, int alphaRange, int useH01);
int xferMLtrain12(svmvolatile int &killSwitch,
                SVM_Generic &core, Vector<ML_Base *> &cases,
                int n, int maxitcntint, double xmtrtime, double soltol, const Vector<double> &caseweight, int usenewton, double lr,
                int randtype, int method, double C, int regtype, double randvari, int alphaRange, int useH01);
int xferMLtrainreflect(svmvolatile int &killSwitch,
                SVM_Scalar &core, Vector<ML_Base *> &cases,
                int maxitcntint, double xmtrtime, double soltol, const Vector<double> &caseweight, double C, int regtype);





int xferMLtrain(svmvolatile int &killSwitch,
                SVM_Generic &core, Vector<ML_Base *> &cases,
                int n, int maxitcntint, double xmtrtime, double soltol, const Vector<double> &caseweight, int usenewton, double lr,
                int randtype, int method, double C, int regtype, double randvari, int alphaRange, int useH01)
{
    int res = 0;

    NiceAssert( ( core.type() == 21 ) || ( core.type() == 0 ) );

    if ( core.type() == 0 )
    {
        res = xferMLtrainreflect(killSwitch,dynamic_cast<SVM_Scalar &>(core),cases,maxitcntint,xmtrtime,soltol,caseweight,C,regtype);
    }

    else if ( method == 0 )
    {
        res = xferMLtrain0(killSwitch,core,cases,n,maxitcntint,xmtrtime,soltol,caseweight,usenewton,lr,randtype,C,regtype,randvari,alphaRange,useH01);
    }

    else
    {
        res = xferMLtrain12(killSwitch,core,cases,n,maxitcntint,xmtrtime,soltol,caseweight,usenewton,lr,randtype,method,C,regtype,randvari,alphaRange,useH01);
    }

    return res;
}



void coreinit(int n, SVM_Generic &core, const Vector<ML_Base *> &cases, int d, int randtype, double randvari, int alphaRange, int useH01);
void coreinit(int n, SVM_Generic &core, const Vector<ML_Base *> &cases, int d, int randtype, double randvari, int alphaRange, int useH01)
{
    int i;

    errstream() << "Calculating variance (if not set a-priori)\n";

    SparseVector<gentype> a,xvar;

    if ( randvari <= 0 )
    {
        int Ntot = 0;
        int m = cases.size();

        for ( i = 0 ; i < m ; ++i )
        {
            const ML_Base &kcase = *(cases(i));
            int N = kcase.N();

            kcase.xvar(a.zero());
            a.scale((double) N);

            Ntot += N;
            xvar += a;
        }

        xvar.scale(1.0/((double) Ntot));
    }

    errstream() << "Setting core size\n";

    while ( core.NNC(2) > n )
    {
        i = 0;

        while ( core.d()(i) != 2 )
        {
            ++i;
        }

        core.removeTrainingVector(i);
    }

    Vector<gentype> randx(d);

    errstream() << "Fixed points removed\n";

    for ( i = core.N()-1 ; i >= 0 ; --i )
    {
        if ( core.d()(i) == 2 )
        {
            SparseVector<gentype> tempx;

            tempx = randitall(randx,randtype,randvari,xvar);

            if ( !i && useH01 )
            {
                tempx = 1.0_gent;
            }

            core.setx(i,tempx);
        }

        else
        {
            core.removeTrainingVector(i);
        }
    }

    errstream() << "Existing vectors randomised";

    gentype ytmp(0.0);

    while ( core.NNC(2) < n )
    {
        SparseVector<gentype> tempx;

        tempx = randitall(randx,randtype,randvari,xvar);

        if ( !(core.N()) && useH01 )
        {
            tempx = 1.0_gent;
        }

        core.qaddTrainingVector(core.N(),ytmp,tempx);
    }

    // Initialise core - randomise weights

    errstream() << "Initialisation... ";

    //core.setFixedBias(0.0); // so it doesn't think it's beta-nonoptimal
    //core.setQuadraticCost(); // alpha should be unlimited
    //core.randomise(1);

    if ( !alphaRange )
    {
        core.randomise(1);
    }

    else
    {
        core.randomise(-1);
    }

errstream() << "alpha = " << core.alphaR() << "\n";

    errstream() << "...Complete\n";
    //errstream() << "Initialised core: " << core << "\n";

    return;
}





int xferMLtrain0(svmvolatile int &killSwitch,
                SVM_Generic &core, Vector<ML_Base *> &cases,
                int n, int maxitcntint, double xmtrtime, double soltol, const Vector<double> &caseweight, int usenewton, double lr,
                int randtype, double C, int regtype, double randvari, int alphaRange, int useH01)
{
    NiceAssert( cases.size() == caseweight.size() );

    // Rounding

    int res = 0;
    int M = cases.size();
    int i,k;

    if ( M == 0 )
    {
        return 1;
    }

    int Nmax = 0;

    for ( i = 0 ; i < M ; ++i )
    {
        const ML_Base &kcase = *(cases(i));

        Nmax = ( Nmax < kcase.N() ) ? kcase.N() : Nmax;
    }

    errstream() << "Nmax = " << Nmax << "\n";

    // Initialise core - add random training vectors

    const ML_Base &zcase = *(cases(0));
    int d = zcase.xspaceDim(); // We assume that this is consistent!

    coreinit(n,core,cases,d,randtype,randvari,alphaRange,useH01);

    // Preliminary outer train

    errstream() << "Preliminary training (M = " << M << ")\n";

    for ( k = 0 ; k < M ; ++k )
    {
        ML_Base &kcase = *(cases(k));

        //kcase.reset();
        kcase.resetKernel();
        //kcase.train(res,killSwitch);
    }

    // Some data stores

    Vector<double> alphabest(n);

    Vector<double> alphaold(n);
    Vector<double> alphastep(n);
    Vector<double> alphanew(n);
    Vector<double> alphaGrad(n);

    Matrix<double> alphaHess(n,n);
    Matrix<double> invhess(n,n);

    Vector<double> tempi(n);
    Vector<double> tmip(n);
    Matrix<double> wpgcache(n,n);
    Vector<double> dedK(Nmax);

    Vector<double> allR(M+1);

    // Re-weight kernel to ensure weights are sane!  We want the average diagonal kernel to be ~eps

    double rawavediag;
    int Ntot;

    errstream() << "Preliminary gradient\n";

    //evalfandgrad(C,Ntot,alphaHess,alphaGrad,core,cases,n,tempi,rawavediag,caseweight,usenewton,tmip,wpgcache,dedK,0,1,regtype);
    evalfandgrad(C,Ntot,alphaHess,alphaGrad,core,cases,n,tempi,rawavediag,caseweight,0,tmip,wpgcache,dedK,0,1,regtype,allR);

    double kweight = Ntot/rawavediag;

    errstream() << "kweight " << kweight << "\n";

    //FIXME: assuming kernel is at most a linear sum of kernels

    //if ( regtype == 1 )
    {
        if ( kweight > KWEIGHT_THRES )
        {
            for ( i = 0 ; i < core.getKernel().size() ; ++i )
            {
                core.getKernel_unsafe().setWeight(kweight*(core.getKernel().cWeight(i)),i);
            }
        }
    }

    // Second preliminary outer train

    for ( k = 0 ; k < M ; ++k )
    {
        ML_Base &kcase = *(cases(k));

        //kcase.reset();
        kcase.resetKernel();
        kcase.train(res,killSwitch);
    }

    // Setup for main training loop

    double maxitcnt = maxitcntint;
    double *uservars[] = { &maxitcnt, &xmtrtime, &soltol, nullptr };
    const char *varnames[] = { "itercount", "traintime", "soltol", nullptr };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Solution tolerance", nullptr };

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    size_t itcnt = 0;
    int timeout = 0;
    int isopt = 0;
    int firstit = 1;
    double bestR = 0;
    double R = 0;

    while ( !killSwitch && !isopt && ( ( itcnt < (size_t) maxitcnt ) || !maxitcnt ) && !timeout )
    {
        // State recording

        alphaold = core.alphaR();

        // Gradient calculation (core)

        R = evalfandgrad(C,Ntot,alphaHess,alphaGrad,core,cases,n,tempi,rawavediag,caseweight,usenewton,tmip,wpgcache,dedK,1,1,regtype,allR);

        if ( firstit || ( R < bestR ) )
        {
            firstit   = 0;
            bestR     = R;
            alphabest = core.alphaR();
        }

        // Newton step (important or scaling gets messed up)

        //errstream() << "phantomx 0: alpha grad: " << alphaGrad << "\n";
        //errstream() << "phantomx 0: alpha hess: " << alphaHess << "\n";

        if ( usenewton )
        {
            alphaGrad *= alphaHess.inve(invhess);
        }

        else
        {
            alphaGrad *= lr;
        }
//errstream() << "phantomx 0: alpha hess inver: " << invhess << "\n";

        // Step calculation (core)

        alphastep =  alphaGrad;
        alphastep *= -NEWTSCALE;

        // New alpha calculation

        alphanew =  alphaold;
        alphanew += alphastep;

        //errstream() << "phantomx 0: alpha old: " << alphaold << "\n";
        //errstream() << "phantomx 0: alpha step: " << alphastep << "\n";
        //errstream() << "phantomx 0: alpha new: " << alphanew << "\n";

        // Core alpha update

        core.setAlphaR(alphanew);

        // Optimality guessing

        double stepsize = absinf(alphastep);

        errstream() << "Step " << itcnt << " size " << stepsize << " (Objective\t= " << R << "\t= ";

        int kk;

        for ( kk = 0 ; kk < M ; ++kk )
        {
            errstream() << allR(kk) << "\t+ ";
        }

        errstream() << allR(M) << " )\n";

        isopt = itcnt && ( stepsize <= soltol ) && ( abs2(rawavediag-1) < 0.1 ) ? 1 : 0;
//isopt = 0;

        // Outer train

        for ( k = 0 ; k < M ; ++k )
        {
            ML_Base &kcase = *(cases(k));

            //kcase.reset();
//errstream() << "phantomx 100 (before): " << kcase.Gp() << "\n";
            kcase.resetKernel();
//errstream() << "phantomx 101 (after): " << kcase.Gp() << "\n";
            kcase.train(res,killSwitch);
        }

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "|\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "/\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "-\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "\\\b";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=" << itcnt << "=  ";
        }

        if ( xmtrtime > 1 )
        {
            curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Transfer learning optimisation",uservars,varnames,vardescr);
        }
    }

    // Set minimiser and final training:
    // Set kernel definition

    errstream() << "Choosing best solution\n";
    errstream() << "Best objective = " << bestR << "\n";

    core.setAlphaR(alphabest);

    // Train instances that are using the kernel

    int dummyres = 0;

    for ( i = 0 ; i < M ; ++i )
    {
        ML_Base &kcase = *(cases(i));

        //kcase.reset();
        kcase.resetKernel();
        kcase.train(dummyres,killSwitch);
    }

//errstream() << "phantomx -10: " << core << "\n";
    return !isopt ? 1 : 0;
}








class xferpar
{
public:
    xferpar(int _n, int _qNmax, SVM_Generic &_core, Vector<ML_Base *> &_cases, const Vector<double> &_caseweight,
            const char *_stateDescr, double **_uservars, const char **_varnames, const char **_vardescr,
            svmvolatile int *_killSwitch, svmvolatile int *_timeout, double xC, int _regtype, int _alphaRange)
            : core(_core), cases(_cases), caseweight(_caseweight)
    {
        n          = _n;
        Nmax       = _qNmax;
        C          = xC;
        regtype    = _regtype;
        alphaRange = _alphaRange;

        itcnt = 0;

        stateDescr = _stateDescr;
        uservars   = _uservars;
        varnames   = _varnames;
        vardescr   = _vardescr;

        killSwitch = _killSwitch;
        timeout    = _timeout;

        alpha.resize(n);
        grad.resize(n);
        hess.resize(n,n);

        tempi.resize(n);
        tmip.resize(n);
        wpgcache.resize(n,n);
        dedK.resize(Nmax);

        allR.resize(cases.size()+1);
    }

    double evaluate(unsigned int _n, const double *alphatest, double *gradres, double **hessres)
    {
        (void) _n;

        NiceAssert( ( n >= 0 ) && ( ((unsigned int) n) == _n ) );

        int i,j;

        int calcgrad = gradres ? 1 : 0; // gradres nullptr if grad not used
        int calchess = hessres ? 1 : 0; // hessres nullptr if hess not used

        int M = cases.size();

        // Get alpha into vector

        for ( i = 0 ; i < n ; ++i )
        {
            if ( !alphaRange )
            {
                alpha("&",i) = (20*alphatest[i])-10; // fix range to (-10,10)
            }

            else
            {
                alpha("&",i) = 10*alphatest[i]; // alt range 0,10
            }
        }

        // Set kernel definition

//errstream() << "phantomxyz 0: " << alpha << "\n";
        core.setAlphaR(alpha);
//errstream() << "phantomxyz 1: " << core.alphaR() << "\n";
//errstream() << "phantomxyz 2: " << core << "\n";


        // Train instances that are using the kernel

        int dummyres = 0;

        for ( i = 0 ; i < M ; ++i )
        {
            ML_Base &kcase = *(cases(i));

            //kcase.reset();
            kcase.resetKernel();
            kcase.train(dummyres,*killSwitch);

            *timeout = ((int) kbquitdet("Transfer learning optimisation",uservars,varnames,vardescr)) | *killSwitch;
        }

        // Calculate objective (and gradient if required)

        int Ntot = 0;
        double rawavediag = 0;

        double res = evalfandgrad(C,Ntot,hess,grad,core,cases,n,tempi,rawavediag,caseweight,calchess,tmip,wpgcache,dedK,1,calcgrad,regtype,allR);

        errstream() << itcnt++ << ": objective\t= " << res << "\t= ";

        int kk;

        for ( kk = 0 ; kk < M ; ++kk )
        {
            errstream() << allR(kk) << "\t+ ";
        }

        errstream() << allR(M) << " )\n";

        //if ( calcgrad )
        //{
        //    errstream() << "Gradient: " << grad << "\n";
        //}

        //if ( calchess )
        //{
        //    errstream() << "Hessian: " << hess << "\n";
        //}

        // Save gradient and hessian if required

        if ( calcgrad )
        {
            for ( i = 0 ; i < n ; ++i )
            {
                gradres[i] = 2*grad(i); // 2 due to rescaling
            }
        }

        if ( calchess )
        {
            for ( i = 0 ; i < n ; ++i )
            {
                for ( j = 0 ; j <= i ; ++j )
                {
                    hessres[i][j] = 4*hess(i,j); // 4 due to rescaling
                    hessres[j][i] = hessres[i][j];
                }
            }
        }

        return res;
    }

    const char *stateDescr;
    double **uservars;
    const char **varnames;
    const char **vardescr;

    svmvolatile int *killSwitch; // external "stop running" flag
    svmvolatile int *timeout;    // derived "stop running" flag

    double C;    // regularisation constant
    int n;       // number of vectors in core
    int Nmax;    // size of largest dataset in cases
    int regtype; // regularisation type
    int alphaRange; // alpha range (see .h)

    Vector<double> alpha; // alpha scratch
    Vector<double> grad;  // gradient scratchpad
    Matrix<double> hess;  // hessian scratchpad

    SVM_Generic &core;                   // core defining the kernel
    Vector<ML_Base *> &cases;       // cases using the kernel
    const Vector<double> &caseweight;   // weights for cases using the kernel

    Vector<double> tempi;     // n dim scratchpad
    Vector<double> tmip;      // n dim scratchpad
    Matrix<double> wpgcache;  // n*n dim scratchpad
    Vector<double> dedK;      // Nmax dim scratchpad

    Vector<double> allR; // used to store breakdown of R

    int itcnt;
};


// nlopt interface function

double nlopt_func_xferml(unsigned n, const double *x,
                         double *gradient, /* nullptr if not needed */
                         void *func_data);
double nlopt_func_xferml(unsigned n, const double *x,
                         double *gradient, /* nullptr if not needed */
                         void *func_data)
{
    xferpar &evalclass(*((xferpar *) func_data));

    return evalclass.evaluate(n,x,gradient,nullptr);
}







int xferMLtrain12(svmvolatile int &killSwitch,
                SVM_Generic &core, Vector<ML_Base *> &cases,
                int n, int maxitcntint, double xmtrtime, double soltol, const Vector<double> &caseweight, int usenewton, double lr,
                int randtype, int method, double C, int regtype, double randvari, int alphaRange, int useH01)
{
    (void) lr;
    (void) usenewton;

    NiceAssert( cases.size() == caseweight.size() );

    // Rounding

    int res = 0;
    int M = cases.size();
    int i,k;

    if ( M == 0 )
    {
        return 1;
    }

    int Nmax = 0;

    for ( i = 0 ; i < M ; ++i )
    {
        const ML_Base &kcase = *(cases(i));

        Nmax = ( Nmax < kcase.N() ) ? kcase.N() : Nmax;
    }

    errstream() << "Nmax = " << Nmax << "\n";

    // We just *assume* that each case inherits from core with kernel 801 and
    // is "sensible" (m=2, nothing fancy going on).

    //int z = 0;

    // Initialise core - add random training vectors

    const ML_Base &zcase = *(cases(0));
    int d = zcase.xspaceDim(); // We assume that this is consistent!

    coreinit(n,core,cases,d,randtype,randvari,alphaRange,useH01);

    // Preliminary outer train

    errstream() << "Preliminary training (M = " << M << ")\n";

    for ( k = 0 ; k < M ; ++k )
    {
        ML_Base &kcase = *(cases(k));

        //kcase.reset();
        kcase.resetKernel();
        //kcase.train(res,killSwitch);
    }

    // Some data stores

    Vector<double> alphaold(n);
    Vector<double> alphastep(n);
    Vector<double> alphanew(n);
    Vector<double> alphaGrad(n);

    Matrix<double> alphaHess(n,n);
    Matrix<double> invhess(n,n);

    Vector<double> tempi(n);
    Vector<double> tmip(n);
    Matrix<double> wpgcache(n,n);
    Vector<double> dedK(Nmax);

    Vector<double> allR(M+1);

    // Re-weight kernel to ensure weights are sane!  We want the average diagonal kernel to be ~eps

    double rawavediag;
    int Ntot;

    errstream() << "Preliminary gradient\n";

    //evalfandgrad(C,Ntot,alphaHess,alphaGrad,core,cases,n,tempi,rawavediag,caseweight,usenewton,tmip,wpgcache,dedK,0,1,regtype);
    evalfandgrad(C,Ntot,alphaHess,alphaGrad,core,cases,n,tempi,rawavediag,caseweight,0,tmip,wpgcache,dedK,0,1,regtype,allR);

    double kweight = Ntot/rawavediag;

    errstream() << "kweight " << kweight << "\n";

    //if ( regtype == 1 )
    {
        for ( i = 0 ; i < core.getKernel().size() ; ++i )
        {
            core.getKernel_unsafe().setWeight(kweight*(core.getKernel().cWeight(i)),i);
        }
    }

    // Second preliminary outer train

    for ( k = 0 ; k < M ; ++k )
    {
        ML_Base &kcase = *(cases(k));

        //kcase.reset();
        kcase.resetKernel();
        kcase.train(res,killSwitch);
    }

    // set up func_data for nlopt

    double *uservars[] = { nullptr };
    const char *varnames[] = { nullptr };
    const char *vardescr[] = { nullptr };

    svmvolatile int timeout = 0;

    xferpar evaldata(n,Nmax,core,cases,caseweight,"Transfer Learning Optimisation",uservars,varnames,vardescr,&killSwitch,&timeout,C,regtype,alphaRange);

    // Algorithm information

    nlopt_stopping optcond;

    double *xtol_abs;
    double *lb;
    double *ub;
    double *alphaval;
    double *alphastp;

    double Rres = 0;

    MEMNEWARRAY(xtol_abs,double,n);

    MEMNEWARRAY(lb,double,n);
    MEMNEWARRAY(ub,double,n);

    MEMNEWARRAY(alphaval,double,n);
    MEMNEWARRAY(alphastp,double,n);

    optcond.n          = n;
    optcond.minf_max   = -HUGE_VAL;   //nopts.minf_max; - minimum f value (-HUGE_VAL)
    optcond.ftol_rel   = 0;           //nopts.ftol_rel; - relative tolerance of function value (0)
    optcond.ftol_abs   = 0;           //nopts.ftol_abs; - absolute tolerance of function value (0)
    optcond.xtol_rel   = 0;           //nopts.xtol_rel; - relative tolerance of x value (0)
    optcond.xtol_abs   = xtol_abs;    //xtol_abs;       - absolute tolerance of x value (0)
    optcond.nevals     = 0;           // 0
    optcond.maxeval    = maxitcntint; //nopts.maxeval;  - max number of f evaluations
    optcond.maxtime    = xmtrtime;    //nopts.maxtraintime;
    optcond.killSwitch = &timeout;

    for ( i = 0 ; i < n ; ++i )
    {
        xtol_abs[i] = soltol;

        lb[i] = 0; // this is re-scaled out in evaluation (required by unstated assumptions in nlopt)
        ub[i] = 1;

        if ( !alphaRange )
        {
            alphaval[i] = (((core.alphaR())(i))+10)/20; // note rescaling
            alphastp[i] = 0.25; // see nelderopt.cc
        }

        else
        {
            alphaval[i] = (core.alphaR())(i)/10; // note lack of rescaling
            alphastp[i] = 0.25; // see nelderopt.cc
        }
    }

    if ( method == 1 )
    {
        // Nelder-mead method

        errstream() << "Nelder-Mead Optimisation Initiated:\n";

        res = nldrmd_minimize(n,nlopt_func_xferml,(void *) &evaldata,lb,ub,alphaval,&Rres,alphastp,&optcond);

        errstream() << "Nelder-Mead Optimisation Ended (best res = " << Rres << ", return code = " << res << ").\n";
    }

    else if ( method == 2 )
    {
        // Subplex optimiser

        errstream() << "Subplex Optimisation Initiated:\n";

        res = sbplx_minimize(n,nlopt_func_xferml,(void *) &evaldata,lb,ub,alphaval,&Rres,alphastp,&optcond);

        errstream() << "Subplex Optimisation Ended (best res = " << Rres << ", return code = " << res << ").\n";
    }

    else
    {
        // SLSQP optimiser

        errstream() << "SLSQP Optimisation Initiated:\n";

        res = nlopt_slsqp(n,nlopt_func_xferml,(void *) &evaldata,lb,ub,alphaval,&Rres,&optcond);

        errstream() << "SLSQP Optimisation Ended (best res = " << Rres << ", return code = " << res << ").\n";
    }

//{
//double *uservars[] = { nullptr };
//const char *varnames[] = { nullptr };
//const char *vardescr[] = { nullptr };
//kbquitdet("phantomxyz 0",uservars,varnames,vardescr,1);
//}

    // Set minimiser and final training:

    for ( i = 0 ; i < n ; ++i )
    {
        if ( !alphaRange )
        {
            alphanew("&",i) = (20*alphaval[i])-10; // fix range to (-10,10)
        }

        else
        {
            alphanew("&",i) = 10*alphaval[i]; // fix range to (0,10)
        }
    }

    // Set kernel definition

    errstream() << "Choosing best solution\n";
    errstream() << "Best objective = " << Rres << "\n";

//errstream() << "phantomxyz 0z: " << alphanew << "\n";
    core.setAlphaR(alphanew);
//errstream() << "phantomxyz 1z: " << core.alphaR() << "\n";
//errstream() << "phantomxyz 2z: " << core << "\n";

    // Train instances that are using the kernel

    int dummyres = 0;

    for ( i = 0 ; i < M ; ++i )
    {
        ML_Base &kcase = *(cases(i));

        //kcase.reset();
        kcase.resetKernel();
        kcase.train(dummyres,killSwitch);
    }

//{
//double *uservars[] = { nullptr };
//const char *varnames[] = { nullptr };
//const char *vardescr[] = { nullptr };
//kbquitdet("phantomxyz 1",uservars,varnames,vardescr,1);
//}

    MEMDELARRAY(xtol_abs);

    MEMDELARRAY(lb);
    MEMDELARRAY(ub);

    MEMDELARRAY(alphaval);
    MEMDELARRAY(alphastp);

    return res;
}



int xferMLtrainreflect(svmvolatile int &killSwitch,
                SVM_Scalar &core, Vector<ML_Base *> &cases,
                int maxitcntint, double xmtrtime, double soltol, const Vector<double> &caseweight, double C, int regtype)
{
    NiceAssert( cases.size() == caseweight.size() );

    int k;
    int res = 0;
    int M = cases.size();

    if ( M == 0 )
    {
        return 0;
    }

    // Save core kernel
    // cases kernels are assumed to be naive 801 kernels

    MercerKernel corekernel(core.getKernel());

    // Empty core training examples:

    errstream() << "Emptying core...\n";

    core.removeTrainingVector(0,core.N());

    // Set C and regularisation type for core

    errstream() << "Setting regulariser in core...\n";

    NiceAssert( regtype );

    if ( regtype == 1 )
    {
        core.setC(C);
        core.set1NormCost();
    }

    else if ( regtype == 2 )
    {
        core.setC(C);
        core.setLinearCost();
    }

    else if ( regtype == 3 )
    {
        core.setC(C);
        core.setQuadraticCost();
    }

    // Construct offsider

    errstream() << "Constructing offcore and offcore kernel...\n";

    SVM_Generic &zcase = dynamic_cast<SVM_Generic &>(*(cases(0)));

    SVM_KConst offcore;
    MercerKernel offkernel(zcase.getKernel());

    offkernel.setAltCall(offcore.MLid());

    // Fill core training examples
    //
    // This *should* just work for SVM_Scalar, SVM_Binary and some other scalar SVMs, but not more complicated objects
    // Note how caseweight is incorporated here.

    errstream() << "Filling core and offcore...\n";

    for ( k = 0 ; k < M ; ++k )
    {
        NiceAssert( caseweight(k) >= 0 )

        SVM_Generic &kcase = dynamic_cast<SVM_Generic &>(*(cases(k)));

//        core.addTrainingVector(core.N(),kcase.zR(),kcase.x(),caseweight(k)*kcase.Cweight(),kcase.epsweight(),kcase.d());
//        offcore.addTrainingVector(offcore.N(),kcase.y(),kcase.x(),kcase.Cweight(),kcase.epsweight());

        int ik;

        for ( ik = 0 ; ik < kcase.N() ; ++ik )
        {
            core.addTrainingVector(core.N(),kcase.zR()(ik),kcase.x(ik),caseweight(k)*kcase.Cweight()(ik),kcase.epsweight()(ik),kcase.d()(ik));
            offcore.addTrainingVector(offcore.N(),kcase.y()(ik),kcase.x(ik),kcase.Cweight()(ik),kcase.epsweight()(ik));
        }
    }

    // Randomise the weights for the core (so that we don't start with 0 kernels, which may cause problems!)

    core.randomise(1);

    // Train using bi-objective, reflective optimisation

    double maxitcnt = maxitcntint;
    double *uservars[] = { &maxitcnt, &xmtrtime, &soltol, nullptr };
    const char *varnames[] = { "itercount", "traintime", "soltol", nullptr };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Solution tolerance", nullptr };

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    size_t itcnt = 0;
    int timeout = 0;
    int isopt = 0;
    int dummyres = 0;

    Vector<double> offalpha(core.N());

    // Preliminary Inner-loop training

    errstream() << "Pre-training...\n";

    for ( k = 0 ; ( k < M ) && !killSwitch ; ++k )
    {
        ML_Base &kcase = *(cases(k));

        kcase.resetKernel();
        res |= kcase.train(dummyres,killSwitch);
    }

    errstream() << "Entering main training loop...\n";

    while ( !killSwitch && !isopt && ( ( itcnt < (size_t) maxitcnt ) || !maxitcnt ) && !timeout )
    {
        // Set alpha in offcore

        int ibase = 0;

        errstream() << "Transferring alpha to offcore... ";

        for ( k = 0 ; ( k < M ) && !killSwitch ; ++k )
        {
            SVM_Generic &kcase = dynamic_cast<SVM_Generic &>(*(cases(k)));

            retVector<double> tmpva;
            offalpha("&",ibase,1,ibase+kcase.N()-1,tmpva) = kcase.alphaR();

            ibase += kcase.N();
        }

        errstream() << offalpha << "\n";

        // Outer-loop training

        errstream() << "Training outer loop... ";

        if ( !killSwitch )
        {
            offcore.setAlphaR(offalpha);

            core.setKernel(offkernel);
            res |= core.train(dummyres,killSwitch);
            core.setKernel(corekernel);
        }

        errstream() << core.alphaR() << "\n";

        // Inner-loop training

        errstream() << "Training inner loop...\n";

        for ( k = 0 ; ( k < M ) && !killSwitch ; ++k )
        {
            ML_Base &kcase = *(cases(k));

            kcase.resetKernel();
            res |= kcase.train(dummyres,killSwitch);
        }

        errstream() << "Book-keeping...\n";

        // Book-keeping

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "|\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "/\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "-\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "\\\b";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=" << itcnt << "=  ";
        }

        if ( xmtrtime > 1 )
        {
            curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Transfer learning optimisation",uservars,varnames,vardescr);
        }
    }

    return res;
}

























































/*

Version 1: Based on forming whole problem at once.  Does not work

double calcextGpcore(int ia, int ib, const SVM_Binary &core, const Vector<SVM_Generic *> &cases);
double calcextGpcore(int ia, int ib, const SVM_Binary &core, const Vector<SVM_Generic *> &cases)
{
    int i,j,k;
    int M = cases.size();
    double res = 0;
    double tempres = 0;

    for ( k = 0 ; k < M ; ++k )
    {
        const SVM_Generic &casek = *cases(k);

        int N = casek.N();

        for ( i = 0 ; i < N ; ++i )
        {
            for ( j = 0 ; j < N ; ++j )
            {
                if ( (casek.alphaState())(i) && (casek.alphaState())(j) )
                {
                    res += ((double) ((casek.alpha())(i)*(casek.alpha()(j))))*core.K4(tempres,ia,ib,-42-i,-42-j,nullptr,&(core.x(ia)),&(core.x(ib)),&(casek.x(i)),&(casek.x(j)),&(core.xinfo(ia)),&(core.xinfo(ib)),&(casek.xinfo(i)),&(casek.xinfo(j)));
                }
            }
        }
    }

    return res;
}

void calcextGpcore(Matrix<double> &extGp, Matrix<double> &extGpsigma, const SVM_Binary &core, const Vector<SVM_Generic *> &cases);
void calcextGpcore(Matrix<double> &extGp, Matrix<double> &extGpsigma, const SVM_Binary &core, const Vector<SVM_Generic *> &cases)
{
    NiceAssert( extGp.isSquare() );

    int n = extGp.numRows();

    int ia,ib;

    for ( ia = 0 ; ia < n ; ++ia )
    {
        extGp("&",ia,ia) = calcextGpcore(ia,ia,core,cases);
        extGpsigma("&",ia,ia) = 0.0;
    }

    for ( ia = 1 ; ia < n ; ++ia )
    {
        for ( ib = 0 ; ib < ia ; ++ib )
        {
            extGp("&",ia,ib) = calcextGpcore(ia,ib,core,cases);
            extGp("&",ib,ia) = extGp(ia,ib);
        }
    }

    for ( ia = 1 ; ia < n ; ++ia )
    {
        for ( ib = 0 ; ib < ia ; ++ib )
        {
            extGpsigma("&",ia,ib) = extGp(ia,ia)+extGp(ib,ib)-extGp(ia,ib)-extGp(ib,ia);
            extGpsigma("&",ib,ia) = extGpsigma(ia,ib);
        }
    }

    return;
}

int xferMLtrain(svmvolatile int &killSwitch, SVM_Binary &core, Vector<SVM_Generic *> &cases, int n, int maxitcntint, double xmtrtime, double soltol)
{
//core.setFixedBias(0.0);
core.setLinBiasForce(-1.0);

    // Rounding

    n = 2*(n/2);

    int npos = n; //n/2; // n
    int nneg = 0; //n/2; // 0

    int res = 0;
    int M = cases.size();

    if ( M == 0 )
    {
        return 1;
    }

    // We just *assume* that each case inherits from core with kernel 801 and
    // is "sensible" (m=2, nothing fancy going on).

    int z = 0;
    int i,k;

    // Initialise core - want n/2 positive samples and n/2 negative

    int d = (*(cases(z))).xspaceDim(); // We assume that this is consistent!

    while ( core.NNC(-1) > nneg )
    {
        i = 0;

        while ( core.d()(i) != -1 )
        {
            ++i;
        }

        core.removeTrainingVector(i);
    }

    while ( core.NNC(+1) > npos )
    {
        i = 0;

        while ( core.d()(i) != +1 )
        {
            ++i;
        }

        core.removeTrainingVector(i);
    }

    Vector<gentype> randx(d);

    for ( i = core.N()-1 ; i >= 0 ; --i )
    {
        if ( core.d()(i) )
        {
            SparseVector<gentype> tempx;

            tempx = randitall(randx);

            core.setx(i,tempx);
        }

        else
        {
            core.removeTrainingVector(i);
        }
    }

    while ( core.NNC(-1) < nneg )
    {
        SparseVector<gentype> tempx;

        tempx = randitall(randx);

        core.qaddTrainingVector(core.N(),-1,tempx);
    }

    while ( core.NNC(+1) < npos )
    {
        SparseVector<gentype> tempx;

        tempx = randitall(randx);

        core.qaddTrainingVector(core.N(),+1,tempx);
    }

    core.randomise(0);

    // Preliminary outer train

    for ( k = 0 ; k < M ; ++k )
    {
        (*(cases("&",k))).resetKernel();
        (*(cases("&",k))).train(res,killSwitch);
    }

    // Setup for main training loop

    Matrix<double> extGp(n,n);
    Matrix<double> extGpsigma(n,n);
    Matrix<double> extGpn(n,1);

    for ( i = 0 ; i < core.N() ; ++i )
    {
        extGpn("&",i,z) = (double) core.d()(i);
    }

    Vector<gentype> alphaold;

    double maxitcnt = maxitcntint;
    double *uservars[] = { &maxitcnt, &xmtrtime, &soltol, nullptr };
    const char *varnames[] = { "itercount", "traintime", "soltol", nullptr };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Solution tolerance", nullptr };

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    unsigned long long itcnt = 0;
    int timeout = 0;
    int isopt = 0;

    core.SVM_Scalar::setGpnExt(nullptr,&extGpn);

    while ( !killSwitch && !isopt && ( ( itcnt < (size_t) maxitcnt ) || !maxitcnt ) && !timeout )
    {
        // State recording

        alphaold = core.alpha();

        // Inner train

        calcextGpcore(extGp,extGpsigma,core,cases);
        core.SVM_Scalar::setGp(&extGp,&extGpsigma,&extGpsigma);
        core.train(res,killSwitch);
errstream() << "phantomxy 0: " << extGp << "\n";
errstream() << "phantomxy 1: " << extGpn << "\n";
errstream() << "phantomxy 2: " << core << "\n";
        core.SVM_Scalar::setGp(nullptr,nullptr);

        // Optimality guessing

        double stepsize = (norm2(alphaold-core.alpha()))/n;

        errstream() << "Step " << stepsize << "\n";

        isopt = itcnt && ( stepsize <= soltol ) ? 1 : 0;
isopt = 0;

        // Outer train

        for ( k = 0 ; k < M ; ++k )
        {
            (*(cases("&",k))).resetKernel();
//errstream() << "phantomx 0: " << (*(cases("&",k))).Gp() << "\n";
            (*(cases("&",k))).train(res,killSwitch);
        }

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "|\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "/\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "-\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "\\\b";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=" << itcnt << "=  ";
        }

        if ( xmtrtime > 1 )
        {
            curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Transfer learning optimisation",uservars,varnames,vardescr);
        }
    }

    core.SVM_Scalar::setGpnExt(&extGpn,nullptr);

    return !isopt ? 1 : 0;
}
*/

