//when model_convertx is called, only go through near part. far part should mimic near part (it's the other side of a rank constraint)

//FIXME: can we plot slides on request
// FIXME: have option for human feedback that does disable(-i-1) to disable
//        in the GP layer of blk_conect but not the prior model(s)
//        THAT IS: present the experiment.  Human gp say 'go ahead", "give
//        hard feedback" (ie some sort of observation for all levels of the
//        model) or "give soft feedback, which uses disable (-i-1) to only
//        change the prior.  If answer is "give feedback" then you don't do
//        the experiment, you go back and find another x as the model has
//        changed!

//
// Bayesian Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "bayesopt.hpp"
#include "ml_mutable.hpp"
#include "imp_expect.hpp"
#include "randfun.hpp"
#include "gentype.hpp"
#include <cmath>

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000

#define DISCOUNTRATE 1e-6

//#define DEFITERS(_effdim_) (10*_effdim_)
#define DEFITERS(_effdim_) (15*_effdim_)

int bayesOpt(int dim,
             Vector<gentype> &xres,
             gentype &fres,
             int &ires,
             Vector<Vector<gentype> > &allxres,
             Vector<gentype> &allfres,
             Vector<Vector<gentype> > &allcres,
             Vector<gentype> &allmres,
             Vector<gentype> &supres,
             Vector<double> &sscore,
             const Vector<gentype> &xmin,
             const Vector<gentype> &xmax,
             void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
             void *fnarg,
             BayesOptions &bopts,
             svmvolatile int &killSwitch);

// Notes:
//
// - Max/min decisions:
//   o Code was originally written to maximise function fn.
//   o All expressions for EI, PI etc were written with this in mind.
//   o To bring it in line with DIRect I decided to change this to min.
//   o Rather than re-write various negations have been introduced.
// - Returning beta:
//   o We need to return beta in supres.
//   o To do this we need (a) a variant of the direct call (a function
//     passed to the direct optimiser for evaluation) that just returns beta,
//     and (b) a means of passing this up to the next level.
//   o We do (a) using the variable "justreturnbeta"
//   o We do (b) by hiding beta at the end of the x vector which is passed
//     to fninner for evaluation of the actual function being minimised.
//   o We use a slightly ugly trick of morphing a sparse vector that is
//     actually not sparse (ie sparse in name only) into a double * by
//     dereferencing the first element in that vector.  It works because
//     reasons.





int BayesOptions::optim(int dim,
                      Vector<gentype> &xres,
                      gentype &fres,
                      int &ires,
                      Vector<Vector<gentype> > &allxres,
                      Vector<gentype> &allfres,
                      Vector<Vector<gentype> > &allcres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &supres,
                      Vector<double> &sscore,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
{
    return bayesOpt(dim,xres,fres,ires,allxres,allfres,allcres,allmres,supres,sscore,
                        xmin,xmax,fn,fnarg,*this,killSwitch);
}









static void calcsscore(Vector<double> &sscore, const BayesOptions &bopts, const Vector<int> &xdatind, int stabp, double stabpnrm, int stabrot, double stabmu, double stabB);
static void calcsscore(Vector<double> &sscore, const BayesOptions &bopts, const Vector<int> &xdatind, int stabp, double stabpnrm, int stabrot, double stabmu, double stabB)
{
    NiceAssert( sscore.size() == xdatind.size() );

    int j;

    for ( j = 0 ; j < sscore.size() ; ++j )
    {
        bopts.model_stabProbTrainingVector(sscore("&",j),xdatind(j),stabp,stabpnrm,stabrot,stabmu,stabB);
    }

    return;
}









class fninnerinnerArg
{
    public:

    fninnerinnerArg(BayesOptions &_bbopts,
                    SparseVector<gentype> &_x,
                    gentype &_muy,
                    gentype &_sigmay,
                    gentype &_ymax,
                    const int &_iters,
                    const int &_dim,
                    const double &_ztol,
                    const double &_delta,
                    const double &_zeta,
                    const double &_nu,
                    gentype &_ires,
                    const double &_a,
                    const double &_b,
                    const double &_r,
                    const double &_p,
                    const double &_modD,
                    const double &_qR,
                    const double &_qBB,
                    const double &_mig,
                    gentype &_betafn,
                    const int &_locmethod,
                    const int &_justreturnbeta,
                    Matrix<gentype> &_covarmatrix,
                    Vector<gentype> &_meanvector,
                    const int &_thisbatchsize,
                    Vector<SparseVector<gentype> > &_multivecin,
                    const ML_Base *_direcpre,
                    gentype &_xytemp,
                    const int &_thisbatchmethod,
                    SparseVector<gentype> &_xappend,
                    const int &_anyindirect,
                    const double &_softmax,
                    const int &_itcntmethod,
                    const int &_itinbatch,
                    const Vector<ML_Base *> &_penalty,
                    gentype &_locpen,
                    const vecInfo **_xinf,
                    const int &_qNbasemu,
                    const int &_qNbasesigma,
                    const int &_qNbasecgt,
                    const int &_gridi,
                    const int &_isgridopt,
                    const int &_iscontopt,
                    const bool &_isfullgrid,
                    const int &_isstable,
                    Vector<int> &_ysort,
                    const int &_stabp,
                    const double &_stabpnrm,
                    const int &_stabrot,
                    const double &_stabmu,
                    const double &_stabB,
                    Vector<double> &_sscore,
                    int &_firstevalinseq,
                    const double &_stabZeroPt,
                    const int &_unscentUse,
                    const int &_unscentK,
                    const Matrix<double> &_unscentSqrtSigma,
                    const int &_stabUseSig,
                    const double &_stabThresh) : bbopts(_bbopts),
                                                _q_x(&_x),
                                                muy(_muy),
                                                sigmay(_sigmay),
                                                ymax(_ymax),
                                                iters(_iters),
                                                dim(_dim),
                                                ztol(_ztol),
                                                _q_delta(_delta),
                                                _q_zeta(_zeta),
                                                _q_nu(_nu),
                                                ires(_ires),
                                                _q_a(_a),
                                                _q_b(_b),
                                                _q_r(_r),
                                                _q_p(_p),
                                                _q_modD(_modD),
                                                _q_R(_qR),
                                                _q_BB(_qBB),
                                                _q_mig(_mig),
                                                _q_betafn(&_betafn),
                                                _q_locmethod(_locmethod),
                                                justreturnbeta(_justreturnbeta),
                                                covarmatrix(_covarmatrix),
                                                meanvector(_meanvector),
                                                _q_thisbatchsize(_thisbatchsize),
                                                multivecin(_multivecin),
                                                direcpre(_direcpre),
                                                xytemp(_xytemp),
                                                _q_thisbatchmethod(_thisbatchmethod),
                                                xappend(_xappend),
                                                anyindirect(_anyindirect),
                                                softmax(_softmax),
                                                itcntmethod(_itcntmethod),
                                                itinbatch(_itinbatch),
                                                penalty(_penalty),
                                                locpen(_locpen),
                                                xinf(_xinf),
                                                Nbasemu(_qNbasemu),
                                                Nbasesigma(_qNbasesigma),
                                                Nbasecgt(_qNbasecgt),
                                                _q_gridi(_gridi),
                                                isgridopt(_isgridopt),
                                                iscontopt(_iscontopt),
                                                isfullgrid(_isfullgrid),
                                                isstable(_isstable),
                                                ysort(_ysort),
                                                stabp(_stabp),
                                                stabpnrm(_stabpnrm),
                                                stabrot(_stabrot),
                                                stabmu(_stabmu),
                                                stabB(_stabB),
                                                sscore(_sscore),
                                                firstevalinseq(_firstevalinseq),
                                                stabZeroPt(_stabZeroPt),
                                                unscentUse(_unscentUse),
                                                unscentK(_unscentK),
                                                unscentSqrtSigma(_unscentSqrtSigma),
                                                stabUseSig(_stabUseSig),
                                                stabThresh(_stabThresh),
                                                mode(0),
                                                storebeta(0.0),
                                                storeepspinf(false)
    {
        return;
    }

    fninnerinnerArg(const fninnerinnerArg &src) : bbopts(src.bbopts),
                                                  _q_x(src._q_x),
                                                  muy(src.muy),
                                                  sigmay(src.sigmay),
                                                  ymax(src.ymax),
                                                  iters(src.iters),
                                                  dim(src.dim),
                                                  ztol(src.ztol),
                                                  _q_delta(src._q_delta),
                                                  _q_zeta(src._q_zeta),
                                                  _q_nu(src._q_nu),
                                                  ires(src.ires),
                                                  _q_a(src._q_a),
                                                  _q_b(src._q_b),
                                                  _q_r(src._q_r),
                                                  _q_p(src._q_p),
                                                  _q_modD(src._q_modD),
                                                  _q_R(src._q_R),
                                                  _q_BB(src._q_BB),
                                                  _q_mig(src._q_mig),
                                                  _q_betafn(src._q_betafn),
                                                  _q_locmethod(src._q_locmethod),
                                                  justreturnbeta(src.justreturnbeta),
                                                  covarmatrix(src.covarmatrix),
                                                  meanvector(src.meanvector),
                                                  _q_thisbatchsize(src._q_thisbatchsize),
                                                  multivecin(src.multivecin),
                                                  direcpre(src.direcpre),
                                                  xytemp(src.xytemp),
                                                  _q_thisbatchmethod(src._q_thisbatchmethod),
                                                  xappend(src.xappend),
                                                  anyindirect(src.anyindirect),
                                                  softmax(src.softmax),
                                                  itcntmethod(src.itcntmethod),
                                                  itinbatch(src.itinbatch),
                                                  penalty(src.penalty),
                                                  locpen(src.locpen),
                                                  xinf(src.xinf),
                                                  Nbasemu(src.Nbasemu),
                                                  Nbasesigma(src.Nbasesigma),
                                                  Nbasecgt(src.Nbasecgt),
                                                  _q_gridi(src._q_gridi),
                                                  isgridopt(src.isgridopt),
                                                  iscontopt(src.iscontopt),
                                                  isfullgrid(src.isfullgrid),
                                                  isstable(src.isstable),
                                                  ysort(src.ysort),
                                                  stabp(src.stabp),
                                                  stabpnrm(src.stabpnrm),
                                                  stabrot(src.stabrot),
                                                  stabmu(src.stabmu),
                                                  stabB(src.stabB),
                                                  sscore(src.sscore),
                                                  firstevalinseq(src.firstevalinseq),
                                                  stabZeroPt(src.stabZeroPt),
                                                  unscentUse(src.unscentUse),
                                                  unscentK(src.unscentK),
                                                  unscentSqrtSigma(src.unscentSqrtSigma),
                                                  stabUseSig(src.stabUseSig),
                                                  stabThresh(src.stabThresh),
                                                  mode(0),
                                                  storebeta(0.0),
                                                  storeepspinf(false)
    {
        NiceThrow("Can't use copy constructer on fninnerinnerArg");
    }

    fninnerinnerArg &operator=(const fninnerinnerArg &)
    {
        NiceThrow("Can't copy fninnerinnerArg");
        return *this;
    }

    BayesOptions &bbopts;
    SparseVector<gentype> *_q_x;
    gentype &muy;
    gentype &sigmay;
    gentype &ymax;
    const int &iters;
    const int &dim;
    const double &ztol;
    double _q_delta;
    double _q_zeta;
    double _q_nu;
    gentype &ires;
    double _q_a;
    double _q_b;
    double _q_r;
    double _q_p;
    double _q_modD;
    double _q_R;
    double _q_BB;
    double _q_mig;
    gentype *_q_betafn;
    int _q_locmethod;
    const int &justreturnbeta;
    Matrix<gentype> &covarmatrix;
    Vector<gentype> &meanvector;
    int _q_thisbatchsize;
    Vector<SparseVector<gentype> > &multivecin;
    const ML_Base *direcpre;
    gentype &xytemp;
    int _q_thisbatchmethod;
    SparseVector<gentype> &xappend;
    const int &anyindirect;
    const double &softmax;
    const int &itcntmethod;
    const int &itinbatch;
    const Vector<ML_Base *> &penalty;
    gentype &locpen;
    const vecInfo **xinf;
    const int &Nbasemu;
    const int &Nbasesigma;
    const int &Nbasecgt;
    int _q_gridi;
    const int &isgridopt;
    const int &iscontopt;
    const bool &isfullgrid;
    const int &isstable;
    Vector<int> &ysort;
    const int &stabp;
    const double &stabpnrm;
    const int &stabrot;
    const double &stabmu;
    const double &stabB;
    Vector<double> &sscore;
    int &firstevalinseq;
    const double &stabZeroPt;
    const int &unscentUse;
    const int &unscentK;
    const Matrix<double> &unscentSqrtSigma;
    const int &stabUseSig;
    const double &stabThresh;
    int PIscale = bbopts.PIscale;

    int mode; // 0 for normal, set 1 when entering DirECT, which will calc beta and zero x precisely once then set to 2, set 2 for no beta calc/x zero
    double storebeta;
    bool storeepspinf;

    double fnfnapprox(int n, const double *xx)
    {
        // Outer loop of unscented optimisation

        double res = 0;

        res = fnfnapproxNoUnscent(n,xx);

        if ( unscentUse )
        {
            NiceAssert( n == unscentSqrtSigma.numRows() );
            NiceAssert( n == unscentSqrtSigma.numCols() );

            res *= ((double) unscentK)/((double) (n+unscentK));

            double *xxx;

            MEMNEWARRAY(xxx,double,n);

            int i,j;
            double temp;

            for ( i = 0 ; i < n ; ++i )
            {
                for ( j = 0 ; j < n ; ++j )
                {
                    xxx[j] = xx[j] + sqrt(n+unscentK)*unscentSqrtSigma(i,j);
                }

                temp = fnfnapproxNoUnscent(n,xxx);
                res += temp/((double) (2*(n+unscentK)));

                for ( j = 0 ; j < n ; ++j )
                {
                    xxx[j] = xx[j] - sqrt(n+unscentK)*unscentSqrtSigma(i,j);
                }

                temp = fnfnapproxNoUnscent(n,xxx);
                res += temp/((double) (2*(n+unscentK)));
            }

            MEMDELARRAY(xxx);
        }

        return res;
    }

  double fnfnapproxNoUnscent(int n, const double *xx)
  {
    double delta = _q_delta;
    double zeta  = _q_zeta;
    double nu    = _q_nu;
    double a     = _q_a;
    double b     = _q_b;
    double r     = _q_r;
    double p     = _q_p;
    double modD  = _q_modD;
    double R     = _q_R;
    double B     = _q_BB;
    double mig   = _q_mig;

    int locmethod       = _q_locmethod;
    int thisbatchsize   = _q_thisbatchsize;
    int thisbatchmethod = _q_thisbatchmethod;
    int gridi           = _q_gridi;

    SparseVector<gentype> &x = *_q_x;
    gentype &betafn          = *_q_betafn;

    // NB: n != dim in general (in fact n = dim*multivecin)

    (void) softmax;
    (void) unscentUse;
    (void) unscentK;
    (void) unscentSqrtSigma;

    int i,j;

    // =======================================================================
    // =======================================================================
    // First work out "beta"
    // =======================================================================
    // =======================================================================

    double beta = 0;

    bool epspinf = false;
    int betasgn  = ( locmethod >= 0 ) ? 1 : -1;
    int method   = betasgn*locmethod;

    // We start by working out beta. We do this for all cases, even IMP etc.
    //
    // mode = 0: calculate beta every single time
    // mode = 1: calculate beta, set mode = 2 later (same beta is required multiple times)
    // mode = 2: we have beta, use that

    if ( ( mode == 0 ) || ( mode == 1 ) )
    {
        double altiters = static_cast<double>(( itcntmethod == 2 ) ? iters-itinbatch : iters); //+1;
        double d        = (double) dim;
        double eps      = 0;
        double locnu    = 1; // nu (beta scale) only applied to Srinivas variants of GP-UCB.
        double dvalreal = d-(bbopts.getdimfid()); // don't include fidelity variables in this scaling: it's not really a "feature" in the usual sense

        switch ( method )
        {
            // TS-Kand: Thompson-sampling (Kandasamy version - actually Chowdhury version, On Kernelised Multiarm Bandits)
            // Human: ask a human (unreachable, kept as placeholder)
            // HE: human exploration agent, as per BO-Muse paper
            // gpUCB MultiFid: gpUCB Kandasamy multifidelity 2017

            //FIXME ell1 should be the L1 diameter of X computed by scaling each dimension by the inverse of the SE bandwitdh

            case 0:  /* Mean only           */ {             eps = 0;                                                                                        break; }
            case 1:  /* EI                  */ {             eps = 0;                                                                                        break; }
            case 2:  /* PI                  */ {             eps = 0;                                                                                        break; }
            case 12: /* TS-Kand             */ {             eps = 0;                                                                                        break; }
            case 16: /* TS-Kand             */ {             eps = 0;                                                                                        break; }
            case 9:  /* PE                  */ {             eps = valpinf();  epspinf = true;                                                               break; }
            case 10: /* PEc                 */ {             eps = valpinf();  epspinf = true;                                                               break; }
            case 18: /* Human               */ {             eps = valvnan();                                                                                break; }
            case 19: /* HE                  */ {             eps = 0.001;                                                                                    break; }
            case 4:  /* gpUCB finite        */ { locnu = nu; eps = 2*log(modD*pow(altiters,2)*(NUMBASE_PI*NUMBASE_PI/(6*delta)));                            break; }
            case 5:  /* gpUCB infinite      */ { locnu = nu; eps = (2*log(pow(altiters,2)*(2*NUMBASE_PI*NUMBASE_PI/(3*delta)))) + (2*thisbatchsize*dvalreal*log(((thisbatchsize==1)?1.0:2.0)*pow(altiters,2)*dvalreal*b*r*sqrt(log(4*thisbatchsize*dvalreal*a/delta)))); break; }
            case 7:  /* gpUCB p finite      */ { locnu = nu; eps = 2*log(modD*numbase_zeta(p)*pow(altiters,p)/delta);                                        break; }
            case 8:  /* gpUCB p infinite    */ { locnu = nu; eps = (2*log(4*numbase_zeta(p)*pow(altiters,p)/delta)) + (2*thisbatchsize*dvalreal*log(((thisbatchsize==1)?1.0:2.0)*pow(altiters,2)*dvalreal*b*r*sqrt(log(4*thisbatchsize*dvalreal*a/delta)))); break; }
            case 11: /* gpUCB user def      */ {             gentype teval(altiters); gentype deval((double) dvalreal); gentype deltaeval(delta); gentype modDeval(modD); gentype aeval(a); eps = (double) betafn(teval,deval,deltaeval,modDeval,aeval); break; }
            case 13: /* gpUCB inf RKHS Sri1 */ { locnu = nu; eps = (2*B) + (300*mig*pow(log(altiters/delta),3));                                             break; }
            case 3:  /* gpUCB basic         */ {             eps = 2*log(pow(altiters,(2+(dvalreal/2)))*(NUMBASE_PI*NUMBASE_PI/(3*delta)));                  break; }
            case 6:  /* gpUCB p basic       */ {             eps = 2*log(2*pow(sqrt(altiters),dvalreal)*numbase_zeta(p)*pow(altiters,p)/delta);              break; }
            case 14: /* gpUCB inf RKHS Cho7 */ {             eps = B + (R*sqrt(2*(mig+1+log(2/delta)))); eps *= eps;                                         break; }
            case 15: /* gpUCB inf RKHS Bog2 */ {             eps = B + (R*sqrt(2*(mig+(2*log(1/delta)))));eps *= eps;                                        break; }
            case 17: /* gpUCB MultiFid      */ {             double ell1 = dvalreal/(bbopts.model_lenscale(0)); eps = 0.5*dvalreal*log((2*ell1*altiters)+1); break; }
            case 20: /* gpUCB as in BO-Muse */ {             eps = sqrt(bbopts.model_sigma(0)*((2*log(1/delta))+1+mig))+B; eps = 7*eps*eps;                  break; }
            default: /* fallback trigger    */ {             eps = valvnan();                                                                                break; }
        }

        StrucAssert( !testisvnan(eps) ); // Human and fallback will trigger this, but if either reaches here then that's an error

        beta = locnu*eps;

        storebeta    = beta;
        storeepspinf = epspinf;
        x.zero();
    }

    else if ( mode == 2 )
    {
        beta    = storebeta;
        epspinf = storeepspinf;
    }

    if ( justreturnbeta == 1 )
    {
        return beta;
    }

    // =======================================================================
    // =======================================================================
    // Re-express function input as sparse vector of gentype (size n)
    // =======================================================================
    // =======================================================================

    //x.zero(); - now done in beta calc loop (skipped for most of direct inner loop - we
    //want to leave x alone as much as possible to avoid free/alloc spiral. Note we also
    //the use of sv below if possible (to avoid resetting altcontent))

    bool xsimple = ( !xappend.size() && !anyindirect ) ? true : false;

    //if ( !(bbopts.getdimfid()) || ( mode != 2 ) ) // The first point is required to setup nup points on x. The second form can be used thereafter
    if ( mode != 2 )
    {
        for ( i = 0 ; i < n-bbopts.getdimfid() ; ++i )
        {
            x.n("&",i,0).force_double() = xx[i]; // use accelerated sv function to preserve if possible!
        }

        for ( i = n-bbopts.getdimfid() ; i < n ; ++i )
        {
            x.n("&",i-(n-bbopts.getdimfid()),1).force_double() = xx[i]; // cant use accelerated form yet as nup may not have been made
        }

        // Update mode to stop double-handling of x

        if ( xsimple && ( mode == 1 ) )
        {
            x.makealtcontent();
        }

        if ( mode == 1 )
        {
            if ( xsimple && ( x.altcontent || x.altcontentsp ) )
            {
                mode = 2;
            }

            else
            {
                mode = 0;
            }
        }
    }

    else
    {
        for ( i = 0 ; i < n ; ++i )
        {
            x.svdirec(i,xx[i]); // use accelerated svdirec function to preserve
        }
    }

    // =======================================================================
    // =======================================================================
    // Add xappend vector to end of vector if required
    // =======================================================================
    // =======================================================================

    for ( i = n ; i < n+xappend.size() ; ++i )
    {
        x.set(i,xappend(i-n)); // use accelerated set function to preserve if possible!
    }

    // =======================================================================
    // =======================================================================
    // Pre-process DIRect input if required
    // =======================================================================
    // =======================================================================

    if ( anyindirect )
    {
        (*direcpre).gg(xytemp,x,*xinf);
        x = (const Vector<gentype> &) xytemp;

        n = xytemp.size(); // This is important. dim cannot be used as it means something else.
    }

    // =======================================================================
    // =======================================================================
    // Calculate output mean and variance
    // (Note that muy can be a vector (multi-objective optim) or
    //  a scalar (regular optimiser))
    // =======================================================================
    // =======================================================================

    NiceAssert( thisbatchsize >= 1 );

    if ( thisbatchsize == 1 )
    {
        if ( isgridopt && ( gridi >= 0 ) && ( isfullgrid || ( bbopts.model_d()(Nbasemu+gridi) == 2 ) ) )
        {
            muy    = bbopts.model_y()(Nbasemu+gridi);
            sigmay = bbopts.model_sigma(0); // Yes this is lazy. It's also fast
            OP_sqrt(sigmay);
        }

        else if ( isgridopt && !iscontopt && ( gridi >= 0 ) )
        {
            if ( ( ( method != 0 ) && ( method != 12 ) && ( method != 16 ) ) || bbopts.isimphere() || PIscale )
            {
                // Model requires sigma

                bbopts.model_muvarTrainingVector(sigmay,muy,Nbasesigma+gridi,Nbasemu+gridi);
                OP_sqrt(sigmay);
            }

            else
            {
                // Model does not require sigma, but it's trivial to calculate

                bbopts.model_muTrainingVector(muy,Nbasemu+gridi);
                sigmay = bbopts.model_sigma(0); // Yes this is lazy. It's also fast
            }
        }

        else
        {
            if ( ( ( method != 0 ) && ( method != 12 ) && ( method != 16 ) ) || bbopts.isimphere() || PIscale )
            {
                // Model requires sigma.

                bbopts.model_muvar(sigmay,muy,x,*xinf);
                OP_sqrt(sigmay); // works elementwise on vectors
            }

            else if ( ( method == 12 ) || ( method == 16 ) )
            {
                // We don't *use* var, but we do need to call var to ensure that the point is properly sampled!

                bbopts.model_muvar(sigmay,muy,x,*xinf);
                if ( sigmay.isValVector() ) { sigmay.dir_vector() = 0.0_gent; }
                else                        { sigmay = 0.0;                   }
            }

            else
            {
                // Model does not require sigma, no JIT sampling involved, so don't waste time calculating it.

                bbopts.model_mu(muy,x,*xinf);
                if ( muy.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; }
                else                     { sigmay.force_double() = 0.0;                }
            }
        }
    }

    else
    {
        NiceAssert( !bbopts.getdimfid() );

        // x(given above) = [ v0, v1, ..., vthisbatchsize-1 ], where xi has dimension
        // dim.  Need to split these up and calculate mean vector and covariance matrix,
        // then reform them to get the min mean vector and modified determinat of the
        // covariance matrix (the geometric mean of the determinants).

        meanvector.resize(thisbatchsize);
        covarmatrix.resize(thisbatchsize,thisbatchsize);
        multivecin.resize(thisbatchsize);

        for ( i = 0 ; i < thisbatchsize ; ++i )
        {
            multivecin("&",i).zero();

            for ( j = 0 ; j < dim ; ++j )
            {
                multivecin("&",i)("&",j) = x((i*dim)+j);
            }

            bbopts.model_mu(meanvector("&",i),multivecin(i),nullptr);
        }

        bbopts.model_covar(covarmatrix,multivecin);

        switch ( thisbatchmethod )
        {
            case 1:  { muy = mean(meanvector);  sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize)); break; }
            case 2:  { muy = min(meanvector,i); sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize)); break; }
            case 3:  { muy = max(meanvector,i); sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            case 4:  { muy = mean(meanvector);  sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            case 5:  { muy = min(meanvector,i); sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            case 17: { muy = max(meanvector,i); sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            default: { muy = max(meanvector,i); sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize)); break; }
        }
    }

    double yymax = ( ymax.isValVector() ) ? 0.0 : ((double) ymax);
    double yymaxcorrect = yymax;

    if ( !ymax.isValVector() && !muy.isValVector() )
    {
        muy += 1.0; // DESIGN DECISION: typically mu in [-1,0] here, but this is bad for the constrained case. We add 1 to "fix" the problem and make mu in [0,1].
        yymaxcorrect += 1.0; // NEED TO CORRECT THIS AS WELL FOR EI
    }

    // Sanity check here!

    if ( testisvnan(sigmay) )
    {
        errstream() << "sigma NaN in Bayesian Optimisation!\n";
        if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; }
        else                        { sigmay.force_double() = 0.0;                }
    }

    else if ( testisinf(sigmay) )
    {
        errstream() << "sigma inf in Bayesian Optimisation!\n";
        if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; }
        else                        { sigmay.force_double() = 0.0;                }
    }

    // =======================================================================
    // =======================================================================
    // Work out PI scale (on the objective, but not including cgt)
    // =======================================================================
    // =======================================================================

    double resscale = 1.0;

    if ( PIscale )
    {
        NiceAssert( !ymax.isValVector() && !muy.isValVector() );

        if      ( (double) sigmay > ztol      ) { resscale = normPhi( ( (double) muy ) - yymaxcorrect ) / ( (double) sigmay ); }
        else if ( (double) muy < yymaxcorrect ) { resscale = 0;                                                                }
        else                                    { resscale = 1;                                                                }
    }

    // =======================================================================
    // =======================================================================
    // Incorporate constraints into mean and variance if cgtmethod == 1
    // =======================================================================
    // =======================================================================

    if ( (bbopts.cgtapprox).size() && ( bbopts.cgtmethod == 1 ) )
    {
        NiceAssert( thisbatchsize == 1 );

        Vector<gentype> mucgt;
        Vector<gentype> varcgtsq;

        if ( isgridopt && !iscontopt && ( gridi >= 0 ) ) { bbopts.model_muvarTrainingVector_cgt(varcgtsq,mucgt,Nbasecgt+gridi); }
        else                                             { bbopts.model_muvar_cgt(varcgtsq,mucgt,x);                            }

        for ( i = 0 ; i < mucgt.size() ; ++i )
        {
            // mucgt >= cgtmargin
            // mucgt-cgtmargin >= 0

            mucgt("&",i) -= bbopts.cgtmargin; // ((bbopts.cgtmargin)*2*sqrt(varcgtsq(i)))
        }

        // Work out mean and variance of delta(c_i(x) >= 0) for all constraints c_i

        Vector<double> cgtmean((bbopts.cgtapprox).size()+1);
        Vector<double> cgtvar((bbopts.cgtapprox).size()+1);

        for ( i = 0 ; i < (bbopts.cgtapprox).size() ; i++ )
        {
            // Sanity check here!

            double ecgt   = (double) mucgt(i);
            double vcgtsq = (double) varcgtsq(i);

            if      ( testisvnan(vcgtsq) ) { errstream() << "varcgt NaN in Bayesian Optimisation!\n"; vcgtsq = 0.0; }
            else if ( testisinf(vcgtsq)  ) { errstream() << "varcgt inf in Bayesian Optimisation!\n"; vcgtsq = 0.0; }
            else if ( vcgtsq <= 0        ) { errstream() << "varcgt neg in Bayesian Optimisation!\n"; vcgtsq = 0.0; }

            // This is basically PI

            if ( vcgtsq > ztol )
            {
                double phihere = normPhi(ecgt/sqrt(vcgtsq));
                cgtmean("&",i) = phihere; // E[delta(c(x)>=0)]
                cgtvar("&",i)  = sqrt((cgtmean(i)*cgtmean(i)*(1-phihere)) + ((1-cgtmean(i))*(1-cgtmean(i))*phihere));  // var[delta(c(x)>=0)]
            }

            else if ( ecgt <= 0 )
            {
                cgtmean("&",i) = 0;
                cgtvar("&",i)  = 0;
            }

            else
            {
                cgtmean("&",i) = 1;
                cgtvar("&",i)  = 0;
            }
        }

        cgtmean("&",(bbopts.cgtapprox).size()) = (double) muy;
        cgtvar("&",(bbopts.cgtapprox).size())  = (double) sigmay;

        // See Goodman: On the Exact Variance of Products

        double meantot = 1; // product of means
        double vartot = 0;  // sum of all products of means and variances (one for each contributor), but not all means

        for ( i = 0 ; i < cgtmean.size() ; i++ )
        {
            meantot *= cgtmean(i);
        }

        for ( i = 1 ; i < (1<<(cgtmean.size())) ; ++i )
        {
            double varsub = 1;

            // Binary, eg size = 3:
            //
            // i = 001b, 010b, 011b, 100b, 101b, 110b, 111b

            int k = 0;

            for ( int j = 1 ; j < (1<<(cgtmean.size())) ; j *= 2, ++k )
            {
                // Binary, eg size = 3:
                //
                // j = 001b, 010b, 100b

                if ( ( i & j ) )
                {
                    // bit set, posterior variance

                    varsub *= cgtvar(k)*cgtvar(k);
                }

                else
                {
                    // bit not set, posterior mean squared

                    varsub *= cgtmean(k)*cgtmean(k);
                }
            }

            vartot += varsub;
        }

        muy    = meantot;      // corrected mean
        sigmay = sqrt(vartot); // corrected variance
    }

    // =======================================================================
    // =======================================================================
    // Calculate stability scores
    // =======================================================================
    // =======================================================================

    double stabscore = 1.0;

    if ( isstable )
    {
        // Calculate stability score on x

        if ( !( isgridopt && ( gridi >= 0 ) ) )
        {
            bbopts.model_stabProb(stabscore,x,stabp,stabpnrm,stabrot,stabmu,stabB);
        }

        else
        {
            NiceAssert( !isfullgrid );
            NiceAssert( gridi >= 0 );
            bbopts.model_stabProbTrainingVector(stabscore,Nbasemu+gridi,stabp,stabpnrm,stabrot,stabmu,stabB);
        }

        if ( stabUseSig )
        {
            //stabscore = ( stabscore >= stabThresh ) ? 1.0 : DISCOUNTRATE;
            stabscore = 1/(1+exp(-1000*(stabscore-stabThresh)));
            //stabscore = 1/(1+exp(-(stabscore-stabThresh)/(stabscore*(1-stabscore))));
        }

        if ( firstevalinseq && ( method == 1 ) )
        {
            firstevalinseq = 0;

            // ...then the stability score on x(j)...

            sscore.resize(ysort.size()+1); // This will eventually become the products
            sscore("&",ysort.size()) = 0.0;

            retVector<double> tmpva;

            calcsscore(sscore("&",0,1,ysort.size()-1,tmpva),bbopts,ysort,stabp,stabpnrm,stabrot,stabmu,stabB);

            // ...and finally convert stability scores to combined stability scores.

            sscore *= -1.0;
            sscore += 1.0;

            for ( j = sscore.size()-1 ; j >= 1 ; --j )
            {
                sscore("&",0,1,j-1,tmpva) *= sscore(j);
            }

            if ( stabUseSig )
            {
                for ( j = 0 ; j < sscore.size() ; ++j )
                {
                    //sscore("&",j) = ( sscore(j) >= stabThresh ) ? 1.0 : DISCOUNTRATE;
                    sscore("&",j) = 1/(1+exp(-1000*(sscore(j)-stabThresh)));
                    //sscore("&",j) = 1/(1+exp(-(sscore(j)-stabThresh)/(sscore(j)*(1-sscore(j)))));
                }
            }
        }
    }

    // =======================================================================
    // =======================================================================
    // Work out improvement measure
    // =======================================================================
    // =======================================================================

    double res = 0;

    if ( bbopts.isimphere() )
    {
        SparseVector<gentype> xmean;

        if ( muy.isValVector() )
        {
            const Vector<gentype> &ghgh = (const Vector<gentype> &) muy;

            for ( int j = 0 ; j < muy.size() ; ++j )
            {
                xmean("&",j) = ghgh(j);
            }
        }

        else
        {
            xmean("&",0) = muy;
        }

        gentype altresv;

        bbopts.modelimp_imp(ires,altresv,xmean,sigmay); // IMP may or may not update sigmay
        res = ((double) ires);

        muy.force_double() = ((double) res); // This is done so that passthrough will apply to IMP
        sigmay.force_double() = sqrt((double) altresv);

        if ( ymax.isValVector() && ( ( method == 1 ) || ( method == 2 ) ) )
        {
            // EI and PI require ymax, which is a function of the scalarisation itself!
            // FIXME: this is a bit of a hack really.

            gentype muymax(ymax);
            SparseVector<gentype> muymean;
            const Vector<gentype> &ghghgh = (const Vector<gentype> &) muymax;

            for ( int j = 0 ; j < muymax.size() ; ++j )
            {
                muymean("&",j) = ghghgh(j);
            }

            gentype dummy,dummyvar,muypr;
            dummyvar.force_vector(muymax.size()) = dummy;
            bbopts.modelimp_imp(muypr,dummy,muymean,dummyvar);

            yymax = ((double) muypr);
        }
    }

    // else - Do this part anyhow to allow for standard GP-UCB.  Method 0 does passthrough of IMP, other methods will change things around
    {
//FIXME - TS-MOO
        NiceAssert( !(muy.isValVector()) );
//errstream() << "phantomxyz mu = " << muy << "\n";
//errstream() << "phantomxyz sigmay = " << sigmay << "\n";
//errstream() << "phantomxyz yymaxcorrect = " << yymaxcorrect << "\n";
//errstream() << "phantomxyz gridi = " << gridi << "\n";

        switch ( method )
        {
            case 1:  // EI - BIG ASSUMPTION: sigmay > ztol
            {
                if ( isstable )
                {
                    res = 0.0;

                    double parta = 0;
                    double partb = 0;
                    double partadec = 0;
                    double partbdec = 0;
                    double scalea,scaleb;

                    for ( int k = 0 ; k <= ysort.size() ; ++k ) // Range 0 to N+1
                    {
                        double dmuy  = (double) muy;
                        double dsigy = (double) sigmay;

                        double ykdec = ( k == 0 ) ? stabZeroPt : ((double) (bbopts.model_y())(ysort(k-1)));
                        double yk    = ( k == ysort.size() ) ? 1e12 : ((double) (bbopts.model_y())(ysort(k))); // 1e12 is a placeholder for bugnum. Never actually used.

                        //parta = phifn((dmuy-ykdec)/dsigy) - ( ( k == ysort.size() ) ? 0.0 : phifn((dmuy-yk)/dsigy) );
                        //partb = Phifn((dmuy-ykdec)/dsigy) - ( ( k == ysort.size() ) ? 0.0 : Phifn((dmuy-yk)/dsigy) );

                        //numbase_phi(parta,(dmuy-yk)/dsigy);
                        //numbase_Phi(partb,(dmuy-yk)/dsigy);

                        parta = normphi((dmuy-yk-zeta)/dsigy);
                        partb = normPhi((dmuy-yk-zeta)/dsigy);

                        //numbase_phi(partadec,(dmuy-ykdec)/dsigy);
                        //numbase_Phi(partbdec,(dmuy-ykdec)/dsigy);

                        partadec = normphi((dmuy-ykdec)/dsigy);
                        partbdec = normPhi((dmuy-ykdec)/dsigy);

                        parta = partadec - ( ( k == ysort.size() ) ? 0.0 : parta );
                        partb = partbdec - ( ( k == ysort.size() ) ? 0.0 : partb );

                        scalea = sscore(k);
                        scaleb = sscore(k)*(dmuy-ykdec)/dsigy;

                        for ( i = 0 ; i < k-1 ; ++i )
                        {
                            double yidec = ( i == 0 ) ? stabZeroPt : ((double) (bbopts.model_y())(ysort(i-1)));
                            double yi    = ( i == ysort.size() ) ? 1e12 : ((double) (bbopts.model_y())(ysort(i))); // i == ysort.size() never attained.

                            scaleb += sscore(i)*(yi-yidec)/dsigy;
                        }

                        res += (parta*scalea)+(partb*scaleb);
                    }

                    res *= ((double) sigmay)*stabscore;
                }

                else
                {
                    if ( (double) sigmay > ztol )
                    {
                        double z = ( ( (double) muy ) - yymaxcorrect ) / ( (double) sigmay );

                        //double Phiz = Phifn(z);
                        //double phiz = phifn(z);

                        double Phiz = 0;
                        double phiz = 0;

                        //numbase_Phi(Phiz,z);
                        //numbase_phi(phiz,z);

                        Phiz = normPhi(z-zeta);
                        phiz = normphi(z-zeta);

                        res = ( ( ( (double) muy ) - yymaxcorrect - zeta ) * Phiz ) + ( ( (double) sigmay ) * phiz );
                    }

                    else
                    {
                        // if muy > ymax then z = +infty, so Phiz = +1, phiz = 0
                        // if muy < ymax then z = -infty, so Phiz = 0,  phiz = infty
                        // assume lim_{z->-infty} sigmay.phiz = 0

                        res = ( (double) muy > yymaxcorrect-zeta ) ? ( ( (double) muy ) - yymaxcorrect - zeta ) : 0.0;
                    }
                }

                break;
            }

            case 2:
            {
                // PI

                NiceAssert( !isstable );

                if      ( (double) sigmay > ztol      ) { res = normPhi(( ( (double) muy ) - yymaxcorrect ) / ( (double) sigmay )); }
                else if ( (double) muy > yymaxcorrect ) { res = 1;                                                                  }
                else                                    { res = 0;                                                                  }

                break;
            }

            default:
            {
                // gpUCB variants, VO and MO, default method

                double locsigmay = (double) sigmay;
                double locmuy = (double) muy;

                //FIXME: assuming binomial distribution here, may not be valid

                if ( isstable && !stabUseSig )
                {
                    // For independent vars: var(x.y) = var(x).var(y) + var(x).mu(y).mu(y) + var(y).mu(x).mu(x)

//                    double stabvar = stabscore*(1-stabscore);
//errstream() << stabvar << "\t" << stabscore << "\t" << locmuy << "\t" << locsigmay << "\t";

//THIS REALLY DOESNT WORK AT ALL!  MAKES VARIANCE LARGE EVERYWHERE, WHICH HINDERS EXPLORATION                    locsigmay = (locsigmay*stabvar) + (locsigmay*stabscore*stabscore) + (locmuy*locmuy*stabvar);
                    locmuy    = stabscore*locmuy;
//phantomx - new variance calc
locsigmay = stabscore*stabscore*locsigmay;
//errstream() << "(\t" << locmuy << "\t" << locsigmay << "\t)\t";
                }

                if ( epspinf )
                {
                    res = betasgn*locsigmay;
                }

                else
                {
                    res = locmuy + ( betasgn*sqrt(beta) * locsigmay );

                    if ( ( bbopts.minstdev > 0 ) && ( locsigmay < bbopts.minstdev ) )
                    {
                        res -= 100;//(1+(2*fabs(locmuy)));
                    }
                }

                if ( isstable && stabUseSig )
                {
                    res *= stabscore;
                }

                break;
            }
        }
    }

    // =======================================================================
    // =======================================================================
    // Add penalties
    // =======================================================================
    // =======================================================================

    if ( penalty.size() )
    {
        for ( i = 0 ; i < penalty.size() ; ++i )
        {
            (*(penalty(i))).gg(locpen,x,*xinf);
            res -= (double) locpen;
        }
    }

    // =======================================================================
    // =======================================================================
    // Scale by probability that constraints are met if cgtmethod == 0
    // (or add variances together for PEc (method 10)
    // =======================================================================
    // =======================================================================

    if ( (bbopts.cgtapprox).size() && ( bbopts.cgtmethod == 0 ) )
    {
        Vector<gentype> mucgt;
        Vector<gentype> varcgt;

        if ( isgridopt && !iscontopt && ( gridi >= 0 ) )
        {
            bbopts.model_muvarTrainingVector_cgt(varcgt,mucgt,Nbasecgt+gridi);
        }

        else
        {
            bbopts.model_muvar_cgt(varcgt,mucgt,x);
        }

        for ( i = 0 ; i < mucgt.size() ; ++i )
        {
            // mucgt >= cgtmargin
            // mucgt-cgtmargin >= 0

            mucgt("&",i) -= bbopts.cgtmargin; // ((bbopts.cgtmargin)*2*sqrt(varcgt(i)));
        }

//errstream() << "phantomxyz mucgt = " << mucgt << "\n";
//errstream() << "phantomxyz varcgt = " << varcgt << "\n";
//errstream() << "phantomxyz gridi = " << gridi << "\n";
        double probofvalid             = 1;
        double totalconstraintvariance = 0;

        for ( i = 0 ; i < (bbopts.cgtapprox).size() ; i++ )
        {
            double ecgt = (double) mucgt(i);
            double vcgt = (double) varcgt(i);

            // Sanity check here!

            if      ( testisvnan(vcgt) ) { errstream() << "varcgt NaN in Bayesian Optimisation!\n"; vcgt = 0.0; }
            else if ( testisinf(vcgt)  ) { errstream() << "varcgt inf in Bayesian Optimisation!\n"; vcgt = 0.0; }
            else if ( vcgt <= 0        ) { errstream() << "varcgt neg in Bayesian Optimisation!\n"; vcgt = 0.0; }

            // This is basically PI

            if ( vcgt > ztol )
            {
                totalconstraintvariance += sqrt(vcgt);
                probofvalid *= normPhi(ecgt/sqrt(vcgt));
            }

            else if ( ecgt < 0 )
            {
                probofvalid = 0;
            }
        }

        // Method 10 aims for exploration in constraints and objective

        if ( method == 10 ) { res += betasgn*totalconstraintvariance; }
        else                { res *= probofvalid;                     }
    }

    // =======================================================================
    // =======================================================================
    // Negate on return (DIRect minimises, Bayesian Optimisation maximises -
    // but keep in mind note about max/min changes).
    // =======================================================================
    // =======================================================================

    res = -res; // Set up for minimiser

    return res*resscale;
  }
};



// fnfnapprox: callback for DIRect optimiser.
// fnfnfnapprox: has mean (to be maxed) in res[0], variance in res[1]

double fnfnapprox         (int n, const double *xx, void *arg) { return (*((fninnerinnerArg *) arg)).fnfnapprox(n,xx);          }
double fnfnapproxNoUnscent(int n, const double *xx, void *arg) { return (*((fninnerinnerArg *) arg)).fnfnapproxNoUnscent(n,xx); }





















// Alternative "grid" minimiser: takes all x indexed by vector in some
// model, tests them, then returns the index (and evaluation) of the
// minimum.  This is a drop-in replacement for directOpt when optimising
// on a grid, except:
//
// - has additional ires argument to return index of gridi vector result
// - has additional gridires argument to return index of x vector result
// - has no xmin/xmax arguments as they are meaningless here
// - takes gridsource (source of grid data_ and gridind (which grid
//   elements are as yet untested) arguments.

int dogridOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              int &ires,
              int &gridires,
              const Vector<double> &xmin,
              const Vector<double> &xmax,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const BayesOptions &bopts,
              ML_Base &gridsource,
              Vector<int> &gridind,
              svmvolatile int &killSwitch,
              const DIRectOptions &dopts,
              double xmtrtime);


int dogridOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              int &ires,
              int &gridires,
              const Vector<double> &xmin,
              const Vector<double> &xmax,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const BayesOptions &bopts,
              ML_Base &gridsource,
              Vector<int> &gridind,
              svmvolatile int &killSwitch,
              const DIRectOptions &dopts,
              double xmtrtime)
{
    NiceAssert( dim > 0 );
    NiceAssert( gridind.size() > 0 );

    ires     = -1;
    gridires = -1;

    int locires     = -1;
    int locgridires = -1;

    (void) gridsource;

    int &gridi  = (*((fninnerinnerArg *) fnarg))._q_gridi;
    int Nbasemu = (*((fninnerinnerArg *) fnarg)).Nbasemu;

    double hardmin = bopts.hardmin;
    double hardmax = bopts.hardmax;

    double tempfres;

    xres.resize(dim);

    Vector<double> xxres(dim);

    double *x  = &xres("&",0);
    double *xx = &xxres("&",0);

    nullPrint(errstream(),"MLGrid Optimisation Initiated");

    int oldgridi = gridi;

    int timeout = 0;
    double *uservars[] = { &xmtrtime, nullptr };
    const char *varnames[] = { "traintime", nullptr };
    const char *vardescr[] = { "Maximum training time (seconds, 0 for unlimited)", nullptr };
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;

    for ( int i = 0 ; ( i < gridind.size() ) && !killSwitch && !timeout ; ++i )
    {
        gridi = gridind(i); // This will propagate through fnarg[41], which will then be passed into function
        // *xinf = &((gridsource.xinfo(gridi))); // This will propagate through fnarg[38], which will then be passed into function

        Vector<double> locxmin(dim);
        Vector<double> locxmax(dim);

        int numnulls = 0;

        for ( int j = 0 ; j < dim ; ++j )
        {
            if ( !bopts.model_x(Nbasemu+gridi).direcref(j).isValNull() )
            {
                // element is pre-defined by the grid

                xxres("&",j) = (double) bopts.model_x(Nbasemu+gridi).direcref(j); //(gridsource.x(gridi))(j);
                locxmin("&",j) = xx[j];
                locxmax("&",j) = xx[j];
            }

            else if ( xmin(j) == xmax(j) )
            {
                // element not pre-defined, but no range, so this is the only possible value

                xxres("&",j) = xmin(j);
                locxmin("&",j) = xmin(j);
                locxmax("&",j) = xmax(j);
            }

            else
            {
                // element not defined, so need to call inner optimizer to find value

                ++numnulls;
                locxmin("&",j) = xmin(j);
                locxmax("&",j) = xmax(j);
            }
        }

        if ( !numnulls )
        {
            // x is fully defined by the grid, so we need only evaluate the acquisition function

            tempfres = (*fn)(dim,xx,fnarg);
        }

        else
        {
            // x is not fully defined by the grid, so we need to invoke direct to optimise the continuous factors
            // Note that the bounds xmin,xmax are already clamped to the non-null grid components to prevent change

            gentype dummyres;

            int directres = directOpt(dim,xxres,dummyres,locxmin,locxmax,fnfnapprox,fnarg,dopts,killSwitch);
            errstream() << "direct(" << gridi << "," << directres << "):";

            tempfres = (double) dummyres;
        }

        if ( ( locires == -1 ) || ( tempfres < (double) fres ) )
        {
            for ( int j = 0 ; j < dim ; ++j )
            {
                x[j] = xx[j];
            }

            fres        = tempfres;
            locires     = i;
            locgridires = gridi;

            //xinfopt = *xinf;

            if ( ( tempfres <= hardmin ) || ( tempfres >= hardmax ) )
            {
                break;
            }
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
            timeout = kbquitdet("Bayesian optimisation",uservars,varnames,vardescr);
        }
    }

    ires     = locires;
    gridires = locgridires;
    gridi    = oldgridi;

    nullPrint(errstream(),"MLGrid Optimisation Ended");

    return 0;
}

















// ===========================================================================
// Evaluates [ mu sigma ] for multi-recommendation via multi-objective
// inner loop.  In multi-recommendation via multi-obj systems there is an
// inner loop that multi-objectively maximises [ mu sigma ], so this function
// evaluates and returns that.  Everything needs to be in the negative
// quadrant, so we need to negative sigma.
// ===========================================================================

static void multiObjectiveCombine(gentype &res, Vector<gentype> &x, void *arg);
static void multiObjectiveCombine(gentype &res, Vector<gentype> &x, void *arg)
{
    BayesOptions &bopts = *((BayesOptions *) ((void **) arg)[0]);

    res.force_vector();
    res.resize(2);

    SparseVector<gentype> xx(x);

    // mu approx is designed for maximisation, so we must negate this as it is used in a minimisation context
    bopts.model_muvar(res("&",1),res("&",0),xx,nullptr); // ditto sigma by definition

    res("&",1).negate();

    return;
}





static gentype &negateoutery(gentype &res);
static gentype &negateoutery(gentype &res)
{
    // Defaults

    if ( res.isValSet() )
    {
        if ( ( res.dir_set().size() >= 1 ) && !res.dir_set().all()(0).isValNull() )
        {
            res.dir_set().ncall()("&",0).negate();
        }
    }

    else if ( res.isValDict() )
    {
        if ( res.dir_dict().iskeypresent("y") && !(res.dir_dict())("y").isValNull() )
        {
            (res.dir_dict())("&","y").negate();
        }
    }

    else
    {
        res.negate();
    }

    return res;
}



// res - either the result, or a set:
//
// { res; addvar; ycgt; xxreplace; xsidechan; obstype }
//
// res:              restype:         replacement for res (result of f(x))
// addvar:           NULL or double:  additional variance in f(x)
// ycgt:             NULL or vector:  constraint vector [c1(x);c2(x);...]
// xxreplace:        NULL or vector:  replacement for x (the x where the measurement is actually taken)
// stopflages:       NULL or int:     bitwise flags.
//                                    1: force early termination
// xsidechan:        NULL or vector:  side-channel data [ xx1 ... xxn ]
// obstype:          NULL or int:     observations type: 0 (na), 1 (>=), -1 (<=), 2 (==).
// obstype_cgt:      NULL or vector:  observations type for ycgt
// addexp:           NULL or gentype: additional observation of the same form (recursively)
// fidcost:          NULL or double:  cost of evaluation for multi-fidelity (-1 for calculate in code, default)
//
// use of NULL: if a variable is NULL or absent then the corresponding vector is not set
// replacexx: this is set if xxreplace is set

static void readres(gentype &res,
                    double &addvar,
                    Vector<gentype> &ycgt,
                    SparseVector<gentype> &xxreplace,
                    int &replacexx,
                    gentype &addexp,
                    int &stopflags,
                    Vector<gentype> &xsidechan,
                    int &xobstype,
                    Vector<int> &xobstype_cgt,
                    double &fidcost);


static void readres(gentype &res,
                    double &addvar,
                    Vector<gentype> &ycgt,
                    SparseVector<gentype> &xxreplace,
                    int &replacexx,
                    gentype &addexp,
                    int &stopflags,
                    Vector<gentype> &xsidechan,
                    int &xobstype,
                    Vector<int> &xobstype_cgt,
                    double &fidcost)
{
    // Defaults

    //xobstype = 2;           // constraint type. this may be preset in gridsource
    //xobstype_cgt.resize(0); // constraint observation types. this is always preset outside, but we leave as an option to pass in
    xsidechan.resize(0);    // side-channel observation
    addexp.force_null();    // additional observations
    stopflags = 0;          // return flags
    xxreplace.resize(0);    // x replacement
    replacexx = 0;          // x replacement flag
    ycgt.resize(0);         // y observation type
    addvar = 0;             // additional variance
    fidcost = -1;           // use internally calculate fidelity cost

    // Overrides

    if ( res.isValSet() )
    {
        bool resSet = false;

        // Keep set return for backwards compatability

        const Set<gentype> &z = (const Set<gentype> &) res;

        if ( ( z.size() >= 3 ) && !z.all()(2).isValNull() ) { ycgt = z.all()(2).cast_vector();      }
        if ( ycgt.size() && !xobstype_cgt.size()          ) { xobstype_cgt.resize(ycgt.size()) = 2; }

        if ( ( z.size() >= 1  ) && !z.all()(0).isValNull() ) { res          =          z.all()(0);                     resSet = true; }
        if ( ( z.size() >= 2  ) && !z.all()(1).isValNull() ) { addvar       = (double) z.all()(1);                                    }
        if ( ( z.size() >= 4  ) && !z.all()(3).isValNull() ) { xxreplace    =          z.all()(3).cast_sparsevector(); replacexx = 1; }
        if ( ( z.size() >= 5  ) && !z.all()(4).isValNull() ) { addexp       =          z.all()(4);                                    }
        if ( ( z.size() >= 6  ) && !z.all()(5).isValNull() ) { stopflags    = (int)    z.all()(5);                                    }
        if ( ( z.size() >= 7  ) && !z.all()(6).isValNull() ) { xsidechan    =          z.all()(6).cast_vector();                      }
        if ( ( z.size() >= 8  ) && !z.all()(7).isValNull() ) { xobstype     = (int)    z.all()(7);                                    }
        if ( ( z.size() >= 9  ) && !z.all()(8).isValNull() ) { xobstype_cgt =          z.all()(8).cast_vector_int();                  }
        if ( ( z.size() >= 10 ) && !z.all()(9).isValNull() ) { fidcost      = (double) z.all()(9);                                    }

        if ( !resSet )
        {
            res.force_null();
        }
    }

    else if ( res.isValDict() )
    {
        bool resSet = false;

        // Proper way to do it

        const Dict<gentype,dictkey> &z = (const Dict<gentype,dictkey> &) res;

        if ( z.iskeypresent("c")      && !z("c").isValNull() ) { ycgt = z("c").cast_vector();          }
        if ( ycgt.size() && !xobstype_cgt.size()             ) { xobstype_cgt.resize(ycgt.size()) = 2; }

        if ( z.iskeypresent("y")       && !z("y").isValNull()       ) { res          =          z("y");                     resSet = true; }
        if ( z.iskeypresent("addvar")  && !z("addvar").isValNull()  ) { addvar       = (double) z("addvar");                               }
        if ( z.iskeypresent("x")       && !z("x").isValNull()       ) { xxreplace    =          z("x").cast_sparsevector(); replacexx = 1; }
        if ( z.iskeypresent("and")     && !z("and").isValNull()     ) { addexp       =          z("and");                                  }
        if ( z.iskeypresent("sf")      && !z("sf").isValNull()      ) { stopflags    = (int)    z("sf");                                   }
        if ( z.iskeypresent("xsc")     && !z("xsc").isValNull()     ) { xsidechan    =          z("xsc").cast_vector();                    }
        if ( z.iskeypresent("d")       && !z("d").isValNull()       ) { xobstype     = (int)    z("d");                                    }
        if ( z.iskeypresent("dc")      && !z("dc").isValNull()      ) { xobstype_cgt =          z("dc").cast_vector_int();                 }
        if ( z.iskeypresent("fidcost") && !z("fidcost").isValNull() ) { fidcost      = (double) z("fidcost");                              }

        if ( !resSet )
        {
            res.force_null();
        }
    }
}

// Given y, ycgt and observation types, translate nulls in y to non-observations
//
// Return is true only if all aspects (objective and constraints) are fully specified.

bool process_obs(gentype &y, Vector<gentype> &ycgt, int &xobstype, Vector<int> &xobstype_cgt, int numcgt)
{
    bool fullobs = true;

    if ( ycgt.size() && !xobstype_cgt.size() )
    {
        xobstype_cgt.resize(ycgt.size()) = 2;
    }

    int isnull = 0;
    int isnull_cgt = 0;

    if ( y.isValVector() )
    {
        const Vector<gentype> &yy = (const Vector<gentype> &) y;

        for ( int iy = 0 ; ( !isnull && ( iy < yy.size() ) ) ; ++iy )
        {
            if ( yy(iy).isValNull() )
            {
                isnull = 1;
            }
        }
    }

    else
    {
        isnull = y.isValNull();
    }

    for ( int iy = 0 ; ( !isnull_cgt && ( iy < ycgt.size() ) ) ; ++iy )
    {
        if ( ycgt(iy).isValNull() )
        {
            isnull_cgt = 1;
        }
    }

    if ( isnull )
    {
        xobstype = 0; // if y null then unconstrained observation is made
        y = 0.0_gent; // null in f(x) gets translates to 0
        fullobs = false;
    }

    for ( int iy = 0 ; iy < ycgt.size() ; ++iy )
    {
        if ( ycgt(iy).isValNull() )
        {
            xobstype_cgt("&",iy) = 0; // if c(x) null then unconstrained
            ycgt("&",iy) = 0.0_gent;  // null to 0
            fullobs = false;
        }
    }

    if ( numcgt && !ycgt.size() )
    {
        xobstype_cgt = 0; // ...and unconstrained by definition
        ycgt = 0.0_gent; // if c(x) undefined then this is null observation
        fullobs = false;
    }

    return fullobs;
}

bool addDataToModel(int &ii,
                    int &Npreadd,
                    int &Npreadd_mu,
                    int &Npreadd_sigma,
                    int &Npreadd_cgt,
                    int &stopnow,
                    BayesOptions &bopts,
                    int dim,
                    bool isfullgrid,
                    bool isgridsource,
                    gentype &fres,
                    int &ires,
                    const SparseVector<gentype> &xx,
                    double addvar,
                    gentype yyval,
                    Vector<gentype> ycgt,
                    const SparseVector<gentype> &xxreplace,
                    int replacexx,
                    const gentype &addexp,
                    int stopflags,
                    const Vector<gentype> &xsidechan,
                    int xobstype,
                    Vector<int> xobstype_cgt,
                    bool doaddstep,
                    double fidcost);

bool addDataToModel(int &ii,
                    int &Npreadd,
                    int &Npreadd_mu,
                    int &Npreadd_sigma,
                    int &Npreadd_cgt,
                    int &stopnow,
                    BayesOptions &bopts,
                    int dim,
                    bool isfullgrid,
                    bool isgridsource,
                    gentype &fres,
                    int &ires,
                    const SparseVector<gentype> &xx,
                    double addvar,
                    gentype yyval,
                    Vector<gentype> ycgt,
                    const SparseVector<gentype> &xxreplace,
                    int replacexx,
                    const gentype &addexp,
                    int stopflags,
                    const Vector<gentype> &xsidechan,
                    int xobstype,
                    Vector<int> xobstype_cgt,
                    bool doaddstep,
                    double fidcost)
{

    // First off we need to "read" the result

    gentype ytoadd = yyval.negate(); // by default, negative of model
    const SparseVector<gentype> &xtoadd = replacexx ? xxreplace : xx; // allow x to be redefined (design choice)

    stopnow |= ( stopflags & 1 );

    bool fullobs = process_obs(ytoadd,ycgt,xobstype,xobstype_cgt,bopts.numcgt);

    if ( !ycgt.size() && bopts.numcgt )
    {
        ycgt.resize(bopts.numcgt) = 0.0_gent; //nullgentype(); - don't want null here!
        xobstype_cgt.resize(bopts.numcgt) = 0;
    }

    bool fullcgt = true;

    for ( int aa = 0 ; ( fullcgt && bopts.numcgt ) && ( aa < bopts.numcgt ) ; ++aa )
    {
        if ( xobstype_cgt(aa) == 0 )
        {
            fullcgt = false;
        }
    }

    bool locxhasnulls = false;
    bool isfullfid = true;

    for ( int aa = 0 ; aa < dim ; ++aa )
    {
        if ( aa < dim-bopts.getdimfid() )
        {
            if ( xtoadd(aa).isValNull() )
            {
                locxhasnulls = true;
            }
        }

        else
        {
            if ( xtoadd.n(aa-bopts.getdimfid(),1).isValNull() )
            {
                locxhasnulls = true;
                isfullfid = false;
            }

            else if ( ((int) (xtoadd.n(aa-(dim-bopts.getdimfid()),1)*((double) bopts.numfids))) != bopts.numfids )
            {
                isfullfid = false;
            }
        }
    }

    if ( doaddstep )
    {
        if ( locxhasnulls )
        {
            StrucAssert( isgridsource );

            Vector<int> loczerovec(xobstype_cgt);

            loczerovec = 0;

            bopts.model_addTrainingVector_musigma(ytoadd,yyval,xtoadd,xsidechan,0); // we set d=0 because this is a placeholder
            bopts.model_addTrainingVector_cgt(ycgt,xtoadd,xsidechan,loczerovec);    // we set d=0 because this is a placeholder

            if ( addvar != 0 )
            {
                bopts.model_setsigmaweight_addvar(bopts.model_N_mu(),bopts.model_N_sigma(),addvar);
            }
        }

        else if ( isfullgrid && isgridsource )
        {
            NiceAssert( ( xobstype == 0 ) || ( xobstype == 2 ) );

            bopts.model_addTrainingVector_musigma(ytoadd,yyval,xtoadd,xsidechan,0); // we set d=0 to prevent training and hence time-wasting
            bopts.model_addTrainingVector_cgt(ycgt,xtoadd,xsidechan,xobstype_cgt);

            if ( addvar != 0 )
            {
                bopts.model_setsigmaweight_addvar(bopts.model_N_mu(),bopts.model_N_sigma(),addvar);
            }
        }

        else
        {
            if ( xobstype || isgridsource )
            {
                bopts.model_addTrainingVector_musigma(ytoadd,yyval,xtoadd,xsidechan,xobstype);

                if ( addvar != 0 )
                {
                    bopts.model_setsigmaweight_addvar(bopts.model_N_mu(),bopts.model_N_sigma(),addvar);
                }
            }

            if ( ( fullcgt && bopts.numcgt ) || isgridsource )
            {
                bopts.model_addTrainingVector_cgt(ycgt,xtoadd,xsidechan,xobstype_cgt);
            }
        }

        if ( xobstype )
        {
            ++Npreadd_mu;
            ++Npreadd_sigma;
        }

        if ( fullcgt && bopts.numcgt )
        {
            ++Npreadd_cgt;
        }

        if ( fullobs )
        {
            ++Npreadd;
        }
    }

    // Record if optimal solution (so far)

    if ( !locxhasnulls ) // ie is this a true observation (all x known, constraints observed (or no constraints))
    {
         if (    isfullfid                                     // full fidelity (or "real system") observation
              && ( xobstype == 2 )                             // equality observation
              && ( ( ires == -1 ) || ( ytoadd > fres ) )       // first observation (ires == -1), or better than previous best
              && ( xobstype_cgt >= 1 )                         // equality OR GREATER THAN feasibility constraint observation (note [] >= [] is always true)
              && ( ycgt >= 0.0_gent )                          // feasible observation (note [] >= [] is always true)
              && ( xtoadd.indsize() == xtoadd.nindsize() ) )   // and double check for "implicit" rank/gradient/special observations for gridsource!
        {
            errstream() << "New fres (a) = " << fres << "\n";
            fres = (double) ytoadd;
            ires = ii;
        }
    }

    if ( fullobs && ( ii >= 0 ) )
    {
        ++ii; // this is local!
    }

    if ( !addexp.isValNull() )
    {
        StrucAssert( !isgridsource );

        // Process additional observation pairs. Note that these do NOT count in terms of fullobs

        // setup and read structure

        const SparseVector<gentype> &qxx = xx; // default assume x is x given
        gentype qres = addexp;
        double qaddvar = 0;
        Vector<gentype> qycgt;
        SparseVector<gentype> qxxreplace;
        int qreplacexx = 0;
        gentype qaddexp;
        int qstopflags = 0;
        Vector<gentype> qxsidechan;
        int qxobstype = 2;
        Vector<int> qxobstype_cgt;

        readres(qres,qaddvar,qycgt,qxxreplace,qreplacexx,qaddexp,qstopflags,qxsidechan,qxobstype,qxobstype_cgt,fidcost);

        // add to model, disgarding return

        addDataToModel(ii,Npreadd,Npreadd_mu,Npreadd_sigma,Npreadd_cgt,stopnow,bopts,dim,false,false,fres,ires,
                       qxx,qaddvar,qres,qycgt,qxxreplace,qreplacexx,qaddexp,qstopflags,qxsidechan,qxobstype,qxobstype_cgt,
                       doaddstep,fidcost);
    }

    return fullobs;
}

void SparsenearToSparsenonnear(SparseVector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid);
void SparsenearToSparsenonnear(SparseVector<double > &dest, const SparseVector<gentype> &src, int dim, int dimfid);
void SparsenearToSparsenonnear(SparseVector<gentype> &dest, const SparseVector<double > &src, int dim, int dimfid);
void SparsenearToSparsenonnear(SparseVector<double > &dest, const SparseVector<double > &src, int dim, int dimfid);

void SparsenonnearToSparsenear(SparseVector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid);
void SparsenonnearToSparsenear(SparseVector<double > &dest, const SparseVector<gentype> &src, int dim, int dimfid);
void SparsenonnearToSparsenear(SparseVector<gentype> &dest, const SparseVector<double > &src, int dim, int dimfid);
void SparsenonnearToSparsenear(SparseVector<double > &dest, const SparseVector<double > &src, int dim, int dimfid);

void NonsparseToSparsenonnear(SparseVector<gentype> &dest, const Vector<gentype> &src, int dim, int dimfid);
void NonsparseToSparsenonnear(SparseVector<double > &dest, const Vector<gentype> &src, int dim, int dimfid);
void NonsparseToSparsenonnear(SparseVector<gentype> &dest, const Vector<double > &src, int dim, int dimfid);
void NonsparseToSparsenonnear(SparseVector<double > &dest, const Vector<double > &src, int dim, int dimfid);

void NonsparseToSparsenear(SparseVector<gentype> &dest, const Vector<gentype> &src, int dim, int dimfid);
void NonsparseToSparsenear(SparseVector<double > &dest, const Vector<gentype> &src, int dim, int dimfid);
void NonsparseToSparsenear(SparseVector<gentype> &dest, const Vector<double > &src, int dim, int dimfid);
void NonsparseToSparsenear(SparseVector<double > &dest, const Vector<double > &src, int dim, int dimfid);

void SparsenearToNonsparse(Vector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid);
void SparsenearToNonsparse(Vector<double > &dest, const SparseVector<gentype> &src, int dim, int dimfid);
void SparsenearToNonsparse(Vector<gentype> &dest, const SparseVector<double > &src, int dim, int dimfid);
void SparsenearToNonsparse(Vector<double > &dest, const SparseVector<double > &src, int dim, int dimfid);

void SparsenearToSparsenonnear(SparseVector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest("&",j) = src.n(j-dimfid,1); }
    }
}

void SparsenearToSparsenonnear(SparseVector<double> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j);            }
        else                  { dest("&",j) = (double) src.n(j-dimfid,1); }
    }
}

void SparsenearToSparsenonnear(SparseVector<gentype> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j).force_double() = src(j);            }
        else                  { dest("&",j).force_double() = src.n(j-dimfid,1); }
    }
}

void SparsenearToSparsenonnear(SparseVector<double> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest("&",j) = src.n(j-dimfid,1); }
    }
}


void SparsenearToNonsparse(Vector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest("&",j) = src.n(j-dimfid,1); }
    }
}

void SparsenearToNonsparse(Vector<double> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j);            }
        else                  { dest("&",j) = (double) src.n(j-dimfid,1); }
    }
}

void SparsenearToNonsparse(Vector<gentype> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j).force_double() = src(j);            }
        else                  { dest("&",j).force_double() = src.n(j-dimfid,1); }
    }
}

void SparsenearToNonsparse(Vector<double> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest("&",j) = src.n(j-dimfid,1); }
    }
}


void SparsenonnearToSparsenear(SparseVector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest.n("&",j-dimfid,1) = src(j); }
    }
}

void SparsenonnearToSparsenear(SparseVector<double> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j);            }
        else                  { dest.n("&",j-dimfid,1) = (double) src(j); }
    }
}

void SparsenonnearToSparsenear(SparseVector<gentype> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j).force_double() = src(j);            }
        else                  { dest.n("&",j-dimfid,1).force_double() = src(j); }
    }
}

void SparsenonnearToSparsenear(SparseVector<double> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest.n("&",j-dimfid,1) = src(j); }
    }
}


void NonsparseToSparsenear(SparseVector<gentype> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest.n("&",j-dimfid,1) = src(j); }
    }
}

void NonsparseToSparsenear(SparseVector<double> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j);            }
        else                  { dest.n("&",j-dimfid,1) = (double) src(j); }
    }
}

void NonsparseToSparsenear(SparseVector<gentype> &dest, const Vector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j).force_double() = src(j);            }
        else                  { dest.n("&",j-dimfid,1).force_double() = src(j); }
    }
}

void NonsparseToSparsenear(SparseVector<double> &dest, const Vector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);            }
        else                  { dest.n("&",j-dimfid,1) = src(j); }
    }
}


void NonsparseToSparsenonnear(SparseVector<gentype> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j); }
        else                  { dest("&",j) = src(j); }
    }
}

void NonsparseToSparsenonnear(SparseVector<double> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j); }
        else                  { dest("&",j) = (double) src(j); }
    }
}

void NonsparseToSparsenonnear(SparseVector<gentype> &dest, const Vector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j).force_double() = src(j); }
        else                  { dest("&",j).force_double() = src(j); }
    }
}

void NonsparseToSparsenonnear(SparseVector<double> &dest, const Vector<double> &src, int dim, int dimfid)
{
    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j); }
        else                  { dest("&",j) = src(j); }
    }
}

//FIXME: Q: if we observe an inequality in a grid, should that count as a "real" observation?







// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// Bayesian optimiser.
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================

#define CONT_TEST ( !stopnow && !stopearly && !killSwitch && !isopt && ( ( itcnt-skipcnt < maxitcnt ) || !maxitcnt ) && !timeout && ( !isgridopt || ( gridind.size() > recBatchSize ) ) && ( !usefidbudget || ( fidtotcost < bopts.fidbudget ) ) )

int bayesOpt(int dim,
             Vector<double> &xres,
             gentype &fres,
             const Vector<double> &qmin,
             const Vector<double> &qmax,
             void (*fn)(int n, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_cgt, bool onlyAdd, double fidcost),
             void *fnarg,
             BayesOptions &bopts,
             svmvolatile int &killSwitch,
             Vector<double> &sscore)
{
//    BayesOptions bopts(bbopts);

    int muapproxsize = 0;
    int i,j,k;
    int stopnow = 0;

    double addvar = 0;
    Vector<gentype> ycgt;
    SparseVector<gentype> xxreplace;
    int replacexx = 0;
    gentype addexp;
    int stopflags = 0;
    Vector<gentype> xsidechan;
    int xobstype = 2;
    Vector<int> xobstype_cgt;

    // =======================================================================
    // Work out levels of indirection in acquisition function.
    //
    // - isindirect: 1 if the acquisition function is a(p(x)), where x is the
    //   value given by DIRect, a is the usual acquisition function, and p is
    //   some pre-processing function given by direcpre.
    // - partindirect: 1 if the first recommendation in a batch is obtained
    //   by optimising a(x) and subsequent recommendations in the batch by
    //   optimising a(p(x))
    // - anyindirect: this is set on a per-iteration basis.  If it is 1 for a
    //   given iteration then that particular recommendation is generated by
    //   optimising a(p(x)), otherwise it is found by optimising a(x).
    // - direcdim: if isindirect or partindirect this is the dimension seen by
    //   the optimiser (DIRect) when optimising a(p(x)), otherwise it is the
    //   dimension seen when optimising a(x).  This is used on a per-iteration
    //   basis in conjunction with anyindirect to work out the dimension seen
    //   for that particular iteration
    // - isstable: zero if no stability constraints, otherwise the number
    //   of stability constraints.
    //
    // - direcpre: the p function being used (if any).
    //
    //
    //
    //
    //
    // =======================================================================

    errstream() << "Entering Bayesian Optimisation Module.\n";

    int isindirect   = bopts.direcpre ? 1 : 0;
    int partindirect = bopts.direcsubseqpre ? 1 : 0;
    int anyindirect  = 0;
    int direcdim     = ( isindirect || partindirect ) ? bopts.direcdim : dim;
    int isstable     = bopts.stabpmax;

    ML_Base *direcpre = (bopts.direcpre) ? (bopts.direcpre) : (bopts.direcsubseqpre);

    NiceAssert( dim > 0 );
    NiceAssert( direcdim > 0 );
    NiceAssert( qmin.size() == dim );
    NiceAssert( qmax.size() == dim );
    NiceAssert( qmax >= qmin );

    // =======================================================================
    // Put min/max bounds in gentype vector
    // =======================================================================
    //
    // effdim: dimension ignoring points at-end that are fixed (xmin(i) == xmax(i))

    errstream() << "Re-expressing bounds.\n";

    Vector<double> xmin(qmin);
    Vector<double> xmax(qmax);

    Vector<gentype> xminalt(dim);
    Vector<gentype> xmaxalt(dim);

    int effdim = 0;

    for ( i = 0 ; i < dim ; ++i )
    {
        xminalt("&",i) = xmin(i);
        xmaxalt("&",i) = xmax(i);

        if ( xmax(i) > xmin(i) )
        {
            effdim = i+1;
        }
    }

    effdim -= bopts.getdimfid(); // multi-fidelity terms don't get included in effdim

    NiceAssert( effdim > 0 );

    // =======================================================================
    // recBatchSize: this is the number of recommendations in each "batch".
    // Note that this only gets set if we have n explicit strategies, it is
    // set to one otherwise.  This is not the same thing as determinant method.
    //
    // numRecs: counter that keeps track of the number of recommendations so far.
    // newRecs: the number of recommendations in a given iteration of the algorithm.
    //
    // =======================================================================

    int recBatchSize = ( ( bopts.method == 11 ) && (bopts.betafn).isValVector() ) ? (bopts.betafn).size() : 1;

    int numRecs = 0;
    int newRecs = 0;

    NiceAssert( recBatchSize > 0 );
    NiceAssert( ( recBatchSize == 1 ) || ( effdim+(bopts.getdimfid()) == dim ) );

    // =======================================================================
    // Local references for function approximation and sigma approximation.
    // sigmaml separate is useful when we have multi-rec and need
    // "hallucinated" samples that need to update the sigma and not the mu.
    // =======================================================================

//    int sigmuseparate = bopts.sigmuseparate;

    // =======================================================================
    //
    // gridsource: non-null if we are restricted to a grid of x values that are
    // mapped by the given ML.
    //
    // =======================================================================

    errstream() << "Grid optimiser setup (if relevant).\n";

    retVector<int> tmpva;

    ML_Base *gridsource = bopts.gridsource;

    int gridi      = -1;
    int gridref    = -1;
    double gridy   = 0;
    int Nbasemu    = bopts.model_N_mu();
    int Nbasesigma = Nbasemu; // bopts.model_N_sigma(); - these are the same at this point (no hallucinations yet!)
    int Nbasecgt   = bopts.model_N_cgt();
    int Ngrid      = 0;
    int Npreadd    = 0;
    double fidcost = -1;

    const vecInfo *xinf = nullptr;
    gentype ggridy;

    Vector<int> gridall;
    Vector<int> gridind;
    Vector<int> griddone; // this will store indices that have been added
    Vector<int> grid_xobstype;
    Vector<int> griddone_xobstype;

    double softmin = -(bopts.softmax);
    double softmax = -(bopts.softmin);

    int ires = -1;
    fres.force_double() = softmin; // this is IMPORTANT. PIscale needs to start with a sensible value, and -1 is the (presumed) min if f(x) is [0,1], as the model
                                   // is on -f(x) in [-1,0]. We min f(x), so we max -f(x), so we need to start with fres being at the *bottom* of this range.
                                   // (it didn't matter much pre PIscale as it only affected the first run, and then a "real" fres took over), but now... what
                                   // happens if the first round doesn't fit the constraint, acquisiton function becomes nominally 0 (because PIscale just goes
                                   // to zero, DIRect just keeps recommending the first thing it tries, and we repeat ad-infinatum!).

    bool isfullgrid   = false; // full grid with unknown constraints and no multifidelity if true (so don't actually need a GP model for the objective, just the constraints)
    bool gridHasNulls = false; // true if grid has undefined x's that may trigger inner-inner continuous optimizer

    if ( gridsource )
    {
        // Work out if this is a full grid with unknown constraints

        if ( bopts.numcgt )
        {
            isfullgrid = true;
            int allfullobs = 1;

            // full-grid not defined if multifidelity

            if ( bopts.getdimfid() )
            {
                isfullgrid = false;
            }

            for ( i = 0 ; i < (*gridsource).N() ; ++i )
            {
                gentype yyval = ((*gridsource).y())(i); // y value as defined in the model (not processed yet)

                xobstype = ((*gridsource).d())(i); // default value as specified in gridsource
                xobstype_cgt.resize(0); //bopts.numcgt) = 2; // default value is equality on constraints

                {
                    double addvar = 0;
                    Vector<gentype> ycgt;
                    SparseVector<gentype> xxreplace;
                    int replacexx = 0;
                    gentype addexp;
                    int stopflags = 0;
                    Vector<gentype> xsidechan;
                    int xobstype = 2;
                    Vector<int> xobstype_cgt;

                    readres(yyval,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost);

                    StrucAssert( addexp.isValNull() );

                    bool fullobs = process_obs(yyval,ycgt,xobstype,xobstype_cgt,bopts.numcgt);
                    const SparseVector<gentype> &xtoadd = replacexx ? xxreplace : (*gridsource).x(i);

                    if ( !xobstype ) { isfullgrid = false; }
                    if ( !fullobs  ) { allfullobs = 0;     }

                    for ( int aa = 0 ; aa < dim ; ++aa )
                    {
                        if ( aa < dim-bopts.getdimfid() )
                        {
                            if ( xtoadd(aa).isValNull() )
                            {
                                gridHasNulls = true; // nulls in fidelity terms are nulls
                            }
                        }

                        else
                        {
                            if ( xtoadd.n(aa-(dim-bopts.getdimfid()),1).isValNull() )
                            {
                                gridHasNulls = true; // nulls in fidelity terms are nulls
                            }
                        }
                    }

// Design decision: if there is a null in the vector but y is defined then this indicates that y is independent of nulled x values
//                   for ( int aa = 0 ; aa < effdim ; ++aa )
//                   {
//                        if ( xtoadd(aa).isValNull() )
//                        {
//                            isfullgrid = false;
//                        }
//                    }
                }
            }

            if ( allfullobs )
            {
                isfullgrid = false;
            }
        }

        // Pre-add vectors to mu and sigma approximators, then set d = 0 (so pre-add but not yet included in calculations)

        int ii = -3;

        for ( i = 0 ; i < (*gridsource).N() ; ++i )
        {
            gentype yyval = (*gridsource).y()(i); // y value as defined in the model (not processed yet)
            const SparseVector<gentype> &xx = (*gridsource).x(i); // x value as defined in the model (not processed yet)

            xobstype = (*gridsource).d()(i); // default value as specified in gridsource
            xobstype_cgt.resize(0); //bopts.numcgt) = 2; // default value is equality on constraints

            readres(yyval,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost);

            StrucAssert( addexp.isValNull() );

            int dNpreadd_mu    = 0; // dummy
            int dNpreadd_sigma = 0; // dummy
            int dNpreadd_cgt   = 0; // dummy

            bool fullobs = addDataToModel(ii,Npreadd,dNpreadd_mu,dNpreadd_sigma,dNpreadd_cgt,stopnow,bopts,dim,isfullgrid,true,fres,ires,
                                          xx,addvar,yyval,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,
                                          true,fidcost);

            if ( !fullobs )
            {
                gridall.add(gridall.size());
                gridall("&",gridall.size()-1) = i;

                gridind.add(gridind.size());
                gridind("&",gridind.size()-1) = i;

                grid_xobstype.add(grid_xobstype.size());
                grid_xobstype("&",grid_xobstype.size()-1) = xobstype;

//                grid_xobstype_cgt.add(grid_xobstype_cgt.size());
//                grid_xobstype_cgt("&",grid_xobstype_cgt.size()-1) = xobstype_cgt;

                ++Ngrid;
            }
        }
    }

    Nbasemu    += Npreadd;
    Nbasesigma += Npreadd;
    Nbasecgt   += Npreadd;

    stopflags = 0;

    // Possible optimiser modes:
    //
    // Pure grid:        isgridopt && !iscontopt
    // Pure continuous: !isgridopt &&  iscontopt
    // Mixed integer:    isgridopt &&  iscontopt

    int isgridopt = Ngrid ? 1 : 0;                          // is this optimisation on a grid. Note isfullgrid is false if Ngrid > 0
    int iscontopt = ( gridHasNulls || !isgridopt ) ? 1 : 0; // is this continuous optimization (with or without a grid), including the fidelity variable
    int ismultfid = bopts.getdimfid() ? 1 : 0;              // is multi-fidelity optimization

    NiceAssert( !isgridopt || !(bopts.isXconvertNonTrivial()) );

    // =======================================================================
    // Actual optimiser starts here
    // =======================================================================

    int dummy = 0;

    Vector<double> xa(dim);
    Vector<SparseVector<double> > xb(recBatchSize);   // Has fidelity variables in nup(0), and augmented data
    Vector<SparseVector<gentype> > xxb(recBatchSize); // Has fidelity variables in nup(1), and no augmented data
    gentype fnapproxout;
    gentype nothingmuch('N');
    SparseVector<gentype> xinb;
    gentype xytemp;

    // =======================================================================
    //
    // - softmax: we convert the minimisation problem to maximisation by
    //   negating, so this is the "soft" maximum.
    // - betaval: the beta for GP-UCB and related acquisition functions.  It
    //   is 1 for the initial random seeds to ensure that variance approxs
    //   make sense.
    // - mupred: stores mu predictions
    // - sigmapred: sigma predictions
    //
    // =======================================================================

    //double softmin = -(bopts.softmax);
    //double softmax = -(bopts.softmin);

    double betaval = 1.0; // for initial batch
    Vector<gentype> mupred(recBatchSize);
    Vector<gentype> sigmapred(recBatchSize);

    // =======================================================================
    // Various timers.
    // =======================================================================

    timediffunits bayesruntime;
    timediffunits mugptraintime;
    timediffunits sigmagptraintime;

    ZEROTIMEDIFF(bayesruntime);
    ZEROTIMEDIFF(mugptraintime);
    ZEROTIMEDIFF(sigmagptraintime);

    // =======================================================================
    // Ensure the model has at least startpoints seed points to begin.
    //
    // NB: we DO NOT want to add points to the model if it already has enough
    //     points in it!  This matters for nested methods like cBO where the
    //     "inner" model should provide one recommendation per "outer"
    //     iteration, and adding additional seeds will simply slow it down.
    //
    // Default: if startpoints == -1 then add dim+1 points.  Apparently this
    //          is standard practice.
    // =======================================================================

    bool usefidbudget = ismultfid && ( bopts.fidbudget > 0 ) && ( bopts.startpoints < 0 ) && ( bopts.totiters < 0 ); // multi-fidelity as per Kandasamy with budget and no over-rides
    int startpoints   = usefidbudget ? 0 : ( ( bopts.startpoints == -1 ) ? (effdim+1+(bopts.getdimfid())) : bopts.startpoints ); //( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints; - note use of effdim here!
    int &startseed    = bopts.startseed; // reference because we need to *persistently* update it!
    int &algseed      = bopts.algseed; // reference because we need to *persistently* update it!
    double modD       = ( bopts.modD == -1 ) ? ( isgridopt ? Ngrid : 10 ) : bopts.modD; // 10 is arbitrary here!
    int Nmodel        = Nbasemu; //bopts.model_N_mu();
    int Nsigma        = Nbasesigma; //bopts.model_N_sigma();
    double B          = 1; // set per iteration
    double mig        = 1; // set per iteration
    //double fidc       = 0.5; // see Kandasamy supplementary and the fidelity stuff
    double fidc       = 1; // see Kandasamy supplementary and the fidelity stuff
    int fidmaxcnt     = 0; // number of times that maximum fidelity has been queried
    double fidtotcost = 0; // amount of cost used in multi-fidelity as per Kandasamy
    bool isfullfid    = true; // set false for non-full-fidelity tests

    // Do no start points if scheduled Bernstein, not first iteration)

    if ( bopts.isspOverride() )
    {
        startpoints = 0;
    }





















    // =======================================================================
    // =======================================================================
    // Initial batch generation
    // =======================================================================
    // =======================================================================

    errstream() << "Testing initial batch.\n";

    int Nstart = Nmodel;

    if ( startpoints || usefidbudget )
    {
        k = 0;
        {
            if ( startseed == -2 )
            {
                srand((int) time(nullptr)); // seed with time
                double dodum = 0;
                randfill(dodum,'S',(int) time(nullptr));
            }

            int skipcnt = 0;

            for ( i = Nstart ; ( ( ( i-skipcnt < Nstart+startpoints ) || !startpoints ) && !killSwitch && ( !isgridopt || ( gridind.size() > recBatchSize ) ) && ( !usefidbudget || ( fidtotcost < (bopts.fidbudget)/10 ) ) ) ; ++i )
            {
                double varscale = 0;

                if ( startseed >= 0 )
                {
                    srand(startseed); // We seed the RNG to create predictable random numbers
                                      // We do it HERE so that random factors OUTSIDE of this
                                      // code that may get called don't get in the way and mess
                                      // up our predictable random sequence
                    double dodum = 0;
                    randfill(dodum,'S',startseed);

                    startseed += 12; // this is a predictable increment so that multiple repeat
                                     // experiments each have a different but predictable startpoint

                    startseed = ( startseed < 0 ) ? 42 : startseed; // Just in case (very unlikely that this will roll over unless we add a *lot* of points)
                }

                errstream() << "Setup batch: ";

                // ===========================================================
                // ===========================================================
                // Generate random point.  Each random point is generating by
                // sampling from U(0,1) and scaling to lie in U(xmin,xmax).
                // ===========================================================
                // ===========================================================

                gridi = -1;

                if ( isgridopt )
                {
                    gridref    = rand()%(gridind.size()); // stored for later when we *may* (or may not) remove from grid if the observation is "full"
                    gridi      = gridind(gridref);
                    gridy      = (double) bopts.model_y()(Nbasemu+gridi);
                    //xinf    = &(((*gridsource).xinfo(gridi)));
                }

                isfullfid = true;

                for ( j = 0 ; j < dim ; ++j )
                {
                    if ( j < dim-bopts.getdimfid() )
                    {
                        // Regular variables

                        //if ( nullxlist(j) )
                        if ( !isgridopt || bopts.model_x(Nbasemu+gridi)(j).isValNull() ) // short-circuit means model_x only used for isgridopt case
                        {
                            // Continuous approximation: values are chosen randomly from
                            // uniform distribution on (continuous) search space.

                            randufill(xxb("&",k)("&",j),xmin(j),xmax(j));
                        }

                        else
                        {
                            // Discrete approximation: values are chosen randomly from
                            // a finite grid.

                            xxb("&",k)("&",j) = bopts.model_x(Nbasemu+gridi)(j); // ((*gridsource).x(gridi))(j);
                        }
                    }

                    else
                    {
                        int qqq = 0;

                        // Fidelity variables

                        if ( !isgridopt || bopts.model_x(Nbasemu+gridi).n(j-(dim-bopts.getdimfid()),1).isValNull() ) // short-circuit means model_x only used for isgridopt case
                        {
                            qqq = 1+(rand()%(bopts.numfids));

                            xxb("&",k).n("&",j-(dim-bopts.getdimfid()),1).dir_double() = ((double) qqq)/((double) bopts.numfids);
                        }

                        else
                        {
                            // Discrete approximation: values are chosen randomly from
                            // a finite grid.

                            xxb("&",k).n("&",j-(dim-bopts.getdimfid()),1) = bopts.model_x(Nbasemu+gridi).n(j-(dim-bopts.getdimfid()),1); // ((*gridsource).x(gridi))(j);

                            qqq = (int) std::lround(((double) (xxb(k).n(j-(dim-bopts.getdimfid()),1)))*(bopts.numfids));
                        }

                        if ( qqq != bopts.numfids )
                        {
                            isfullfid = false;
                        }
                    }
                }

                // ===========================================================
                // ===========================================================
                // Translate data for experiment
                // ===========================================================
                // ===========================================================

                SparsenearToSparsenonnear(xb("&",k),xxb(k),dim,bopts.getdimfid());

                // ===========================================================
                // ===========================================================
                // Generate preliminary predictions for reference
                // ===========================================================
                // ===========================================================

                if ( isgridopt && ( isfullgrid || ( bopts.model_d()(Nbasemu+gridi) == 2 ) ) )
                {
                    mupred("&",k) = bopts.model_y()(Nbasemu+gridi);
                    sigmapred("&",k) = bopts.model_sigma(0);
                }

                else if ( isgridopt && !iscontopt )
                {
                    bopts.model_muvarTrainingVector(sigmapred("&",k),mupred("&",k),Nbasesigma+gridi,Nbasemu+gridi);
                }

                else
                {
                    bopts.model_muvar(sigmapred("&",k),mupred("&",k),xxb(k),xinf);
                }

                // ===========================================================
                // ===========================================================
                // Calculate supplementary data
                // ===========================================================
                // ===========================================================

                double rmupred    = mupred(k).isCastableToReal()    ? ( (double) mupred(k)    ) : 0.0;
                double rsigmapred = sigmapred(k).isCastableToReal() ? ( (double) sigmapred(k) ) : 0.0;
                double standev    = sqrt(betaval)*rsigmapred;

                xb("&",k)("&",dim  )  = 0; //newRecs-1; (make sure result is not sparse!)
                xb("&",k)("&",dim+1)  = betaval;
                xb("&",k)("&",dim+2)  = rmupred;
                xb("&",k)("&",dim+3)  = rsigmapred;
                xb("&",k)("&",dim+4)  = rmupred+standev;
                xb("&",k)("&",dim+5)  = rmupred-standev;
                xb("&",k)("&",dim+6)  = 2*standev;
                xb("&",k)("&",dim+7)  = softmax;
                xb("&",k)("&",dim+8)  = 0; // You need this to ensure vector is not sparse!
                xb("&",k)("&",dim+9)  = bayesruntime;
                xb("&",k)("&",dim+10) = mugptraintime;
                xb("&",k)("&",dim+11) = sigmagptraintime;
                xb("&",k)("&",dim+12) = gridi;
                xb("&",k)("&",dim+13) = gridy;
                xb("&",k)("&",dim+14) = ( ( bopts.B <= 0 ) && !bopts.ismoo ) ? bopts.model_RKHSnorm(0) : bopts.B; // Only makes sense for single-objective
                xb("&",k)("&",dim+15) = !bopts.ismoo ? bopts.model_maxinfogain(0) : 0.0;                          // Only makes sense for single-objective
                xb("&",k)("&",dim+16) = fidtotcost;

                // ===========================================================
                // ===========================================================
                // Run experiment:
                // ===========================================================
                // ===========================================================

                gentype gridresis = nullgentype();
                Vector<gentype> ycgtis(bopts.numcgt,nullgentype());

                if ( isgridopt && ( isfullgrid || ( bopts.model_d()(Nbasemu+gridi) == 2 ) ) )
                {
                    // Grid may have defined solutions already loaded.

                    gridresis = bopts.model_y()(Nbasemu+gridi);
                    gridresis.negate();
                }

                for ( int iy = 0 ; iy < bopts.numcgt ; ++iy )
                {
                    if ( isgridopt && ( bopts.model_d_cgt(iy)(Nbasemu+gridi) == 2 ) )
                    {
                        ycgtis("&",iy) = bopts.model_y_cgt(iy)(Nbasemu+gridi);
                    }
                }

                fidcost = -1;
                xobstype = 2;
                fnapproxout.force_int() = -1;
                (*fn)(dim,fnapproxout,&xb(k)(0),fnarg,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,gridresis,muapproxsize,ycgtis,false,fidcost);
                negateoutery(fnapproxout);

                bool fullobs    = false;
                bool addobs     = true; // remember that multiple results might be packed in fnapproxout
                bool firstinset = true;

                while ( addobs )
                {
                    if ( ( stopflags & 1 )               ) { stopnow = 1; }
                    if ( ( stopflags & 2 ) && firstinset ) { ++skipcnt;   }

                    // Work out if this is a full observation, and also zero out nulls

                    fullobs = process_obs(fnapproxout,ycgt,xobstype,xobstype_cgt,bopts.numcgt);

                    NiceAssert( !( isgridopt && !iscontopt ) || !replacexx );

                    if ( replacexx )
                    {
                        gridi = -1;

                        xxb("&",k) = xxreplace;

                        int numfids = bopts.numfids;
                        int dimfid  = bopts.getdimfid();

                        SparsenearToSparsenonnear(xb("&",k),xxb(k),dim,dimfid);

                        isfullfid = true;

                        if ( numfids )
                        {
                            // Retrieve fidelity

                            for ( int jij = dimfid-1 ; jij >= 0 ; jij-- )
                            {
                                if ( (int) std::lround(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids )
                                {
                                    isfullfid = false;
                                }
                            }
                        }
                    }

                    varscale = 0;

                    // DESIGN DECISIONS: all observations are counted in the budget!
                    {
                        SparseVector<SparseVector<gentype> > actfidel;

                        for ( int jij = 0 ; jij < bopts.getdimfid() ; jij++ )
                        {
                            actfidel("&",0)("&",jij) = xxb(k).n(jij,1);
                        }

                        fidtotcost += ( ( fidcost >= 0 ) ? fidcost : ( (double) bopts.fidpenalty(actfidel) ) );
                        varscale    = (double) bopts.fidvar(actfidel);
                        firstinset = false;
                    }

                    // ===========================================================
                    // ===========================================================
                    // Update models
                    // ===========================================================
                    // ===========================================================

                    errstream() << "a(" << mupred(k) << ";" << sigmapred(k) << ")";

                    if ( isgridopt && !iscontopt && ( gridi >= 0 ) )
                    {
                        // Remove the index from grid for full experiment, update model appropriately

                        fullobs = true;

                        if ( !isfullgrid )
                        {
                            if ( xobstype )
                            {
                                bopts.model_setyd(Nbasemu+gridi,Nbasesigma+gridi,xobstype,fnapproxout,varscale);

                                if ( addvar != 0 )
                                {
                                    bopts.model_setsigmaweight_addvar(Nbasemu+gridi,Nsigma,addvar);
                                }
                            }

                            else if ( !bopts.model_d()(Nbasemu+gridi) )
                            {
                                fullobs = false;
                            }
                        }

                        else
                        {
                            ; //grid_xobstype("&",gridi) = xobstype; // so we can update later
                        }

                        if ( ycgt.size() )
                        {
                            for ( int iy = 0 ; iy < ycgt.size() ; ++iy )
                            {
                                if ( !xobstype_cgt(iy) ) // ycgt(iy).isValNull() )
                                {
                                    if ( bopts.model_d_cgt(iy)(Nbasecgt+gridi) )
                                    {
                                        ycgt("&",iy) = bopts.model_y_cgt(iy)(Nbasecgt+gridi);
                                        xobstype_cgt("&",iy) = bopts.model_d_cgt(iy)(Nbasecgt+gridi);
                                    }

                                    else
                                    {
                                        fullobs = false;
                                    }
                                }
                            }

                            bopts.model_setyd_cgt(Nbasecgt+gridi,xobstype_cgt,ycgt,varscale);
                        }

                        if ( fullobs )
                        {
                            griddone.add(griddone.size());
                            griddone("&",griddone.size()-1) = gridind(gridref);
                            griddone_xobstype.add(griddone_xobstype.size());
                            griddone_xobstype("&",griddone_xobstype.size()-1) = grid_xobstype(gridref);
                            gridind.remove(gridref);
                            grid_xobstype.remove(gridref);
                        }
                    }

                    else if ( isgridopt && !iscontopt && ( gridi == -1 ) )
                    {
                        bopts.model_addTrainingVector_musigma(((fnapproxout.isValNull()) ? 0.0_gent : fnapproxout),mupred(k),xxb(k),xsidechan,xobstype,varscale);

                        if ( addvar != 0 )
                        {
                            bopts.model_setsigmaweight_addvar(Nmodel,Nsigma,addvar);
                        }

                        if ( ycgt.size() )
                        {
                            bopts.model_addTrainingVector_cgt(ycgt,xxb(k),xsidechan,xobstype_cgt,varscale);
                        }
                    }

                    else
                    {
                        if ( !isfullgrid || ( gridi == -1 ) || replacexx )
                        {
                            bopts.model_addTrainingVector_musigma(((fnapproxout.isValNull()) ? 0.0_gent : fnapproxout),mupred(k),xxb(k),xsidechan,xobstype,varscale);

                            if ( addvar != 0 )
                            {
                                bopts.model_setsigmaweight_addvar(Nmodel,Nsigma,addvar);
                            }
                        }

                        if ( ycgt.size() )
                        {
                            bopts.model_addTrainingVector_cgt(ycgt,xxb(k),xsidechan,xobstype_cgt,varscale);
                        }
                    }

                    // ===========================================================
                    // ===========================================================
                    // Update impulse model (if defined, for multiobjective opt)
                    // ===========================================================
                    // ===========================================================

                    if ( bopts.isimphere() )
                    {
                        if ( fnapproxout.isValVector() )
                        {
                            const Vector<gentype> &ghgh = (const Vector<gentype> &) fnapproxout;

                            for ( j = 0 ; j < fnapproxout.size() ; ++j )
                            {
                                xinb("&",j) = ghgh(j);
                            }
                        }

                        else
                        {
                            xinb("&",0) = fnapproxout;
                        }

                        xinb.negate();

                        bopts.modelimp_addTrainingVector(nothingmuch,xinb);
                    }

                    // ===========================================================
                    // ===========================================================
                    // Feedback
                    // ===========================================================
                    // ===========================================================

                    if ( bopts.isimphere() ) { errstream() << "(" << Nmodel << "," << bopts.modelimp_N() << ")."; }
                    else                     { errstream() << "(" << Nmodel << ").";                              }

                    // ===========================================================
                    // ===========================================================
                    // Fill in fres and ires based on result
                    // ===========================================================
                    // ===========================================================

//FIXME TS-MOO
                    if (    isfullfid
                         && ( xobstype == 2 )
                         && ( ( ires == -1 ) || ( fnapproxout > fres ) )
                         && ( xobstype_cgt >= 1 )
                         && ( ycgt >= 0.0_gent )
                         && ( xxb(k).indsize() == xxb(k).nindsize() ) )
                    {
                        errstream() << "New fres (d) = " << fres << "\n";
                        fres = (double) fnapproxout;
                        ires = ( isgridopt && !iscontopt && ( gridi >= 0 ) ) ? Nbasemu+gridi : (bopts.model_N_mu())-1;
                    }

                    // ===========================================================
                    // ===========================================================
                    // Counters
                    // ===========================================================
                    // ===========================================================

                    if ( xobstype || fullobs || ( isgridopt && iscontopt && isfullgrid ) )
                    {
                        ++Nmodel;
                        ++Nsigma;
                    }

                    else
                    {
                        --i; // We need Nstart non-null results here!
                    }

                    addobs = addexp.isValNull() ? false : true;

                    gridi = -1;

                    if ( addobs )
                    {
                        // recurse to next observation to process

                        fnapproxout = addexp;
                        addvar = 0;
                        ycgt.resize(0);
                        xxreplace.zero();
                        replacexx = 0;
                        addexp.force_null();
                        stopflags = 0;
                        xsidechan.resize(0);
                        xobstype = 2;
                        xobstype_cgt.resize(0);

                        readres(fnapproxout,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost);
                        negateoutery(fnapproxout);
                    }
                }

                errstream() << "\n";
            }
        }
    }

/*
    if ( startseed >= 0 )
    {
        startseed = oldstartseed + 19;
    }
*/






















    // ===========================================================
    // ===========================================================
    // Stability computations
    //
    // ysort is used to index bopts.model_y() from smallest to largest,
    // which is required for calculating stability scores.
    // ===========================================================
    // ===========================================================

    Vector<int> ysort;

    int stabpmax = bopts.stabpmax;
    int stabpmin = bopts.stabpmin;
    int stabrot  = 0; //bopts.stabrot;

    double stabpnrm   = 2; //bopts.stabpnrm;
    double stabA      = bopts.stabA;
    double stabB      = bopts.stabB;
    double stabF      = bopts.stabF;
    double stabbal    = bopts.stabbal;
    double stabZeroPt = bopts.stabZeroPt;

    int stabp = 0;

    double stabmumin = 0;
    double stabmumax = 0;
    double stabmu    = 0;
    double stablogD  = 0;
    double stabM     = 0;
    double stabLs    = 0;
    double stabDelta = 0;

    if ( isstable )
    {
        errstream() << "Setting stability parameters.\n";

        // FIXME: currently we are simply *assuming* an RBF kernel.  In general
        //        though we need to test what sort of kernel is actually being
        //        used and then set these constants accordingly, testing where
        //        required.

        stabLs    = 1.0/((bopts.model_lenscale(0))*(bopts.model_lenscale(0)));
        stabDelta = 0;

        double Lsca = 2*stabLs*stabB*stabB;

        NiceAssert( Lsca < 1 );

        for ( j = 0 ; j < dim ; ++j )
        {
            stabM += ((xmax(j)-xmin(j))*(xmax(j)-xmin(j)));
        }

        stabM = sqrt(stabM);
        //stabD = 0.816*NUMBASE_SQRTSQRTPI*exp(0.5*stabM*stabM*stabLs); - this overflows
        stablogD = log(0.816*NUMBASE_SQRTSQRTPI)+(0.5*stabM*stabM*stabLs);

        double lambres = 0.0;

        //numbase_lambertW(lambres, (2.0/NUMBASE_E)*(1.0/Lsca)*(log(NUMBASE_1ONSQRTSQRT2PI*((stabD*stabF)/(stabA-(stabDelta*stabF)))*(1.0/(1.0-sqrt(Lsca))))) ); - this overflows
        numbase_lambertW(lambres, (2.0/NUMBASE_E)*(1.0/Lsca)*(stablogD+log(NUMBASE_1ONSQRTSQRT2PI*(stabF/(stabA-(stabDelta*stabF)))*(1.0/(1.0-sqrt(Lsca))))) );

        stabp = (int) ceil((Lsca*exp(1+lambres))-1);

        //double Rbnd = (((stabD/sqrt(xnfact(stabp+1)))*(pow(Lsca,(stabp+1.0)/2.0)/(1.0-Lsca)))+stabDelta)*stabF; // Need to use un-slipped pmin here!
        // Need to do this manually to prevent an overflow

        /* aaand it still overflows
        double Rbnd = stabD;

        for ( j = 1 ; j <= stabp+1 ; ++j )
        {
            Rbnd /= sqrt(j);
        }
        */

        double Rbnd = stablogD;

        for ( j = 1 ; j <= stabp+1 ; ++j )
        {
            Rbnd -= (sqrt(j)/2.0);
        }

        Rbnd = exp(Rbnd);

        Rbnd *= pow(sqrt(Lsca),stabp+1)/(1.0-sqrt(Lsca));
        Rbnd += stabDelta;
        Rbnd *= stabF;

        stabmumin = stabA - Rbnd;
        stabmumax = stabA + Rbnd;

        stabmu = stabmumin + (stabbal*(stabmumax-stabmumin));

        errstream() << "Stable optimisation A     = " << stabA     << "\n";
        errstream() << "Stable optimisation B     = " << stabB     << "\n";
        errstream() << "Stable optimisation F     = " << stabF     << "\n";
        errstream() << "Stable optimisation M     = " << stabM     << "\n";
        errstream() << "Stable optimisation log(D)= " << stablogD  << "\n";
        errstream() << "Stable optimisation Ls    = " << stabLs    << "\n";
        errstream() << "Stable optimisation Delta = " << stabDelta << "\n";
        errstream() << "Stable optimisation Up    = " << Rbnd      << "\n";
        errstream() << "Stable optimisation pmin  = " << stabp     << "\n";
        errstream() << "Stable optimisation mu-   = " << stabmumin << "\n";
        errstream() << "Stable optimisation mu+   = " << stabmumax << "\n";
        errstream() << "Stable optimisation mu    = " << stabmu    << "\n";
        errstream() << "Stable optimisation p     = " << stabp     << "\n";

        stabp = ( stabp >= stabpmin ) ? stabp : stabpmin;
        stabp = ( stabp <= stabpmax ) ? stabp : stabpmax;

        errstream() << "Stable optimisation p (adjusted) = " << stabp << "\n";

        retVector<int> tmpva;

        Vector<int> rort(cntintvec(bopts.model_N_mu(),tmpva));

        int jj = 0;
        int firstone;
        int NC = bopts.model_N_mu()-bopts.model_NNCz_mu(0);

        //ysort.prealloc( NC+(( bopts.totiters == -1 ) ? 10*dim*( bopts.ismoo ? bopts.moodim : 1 ) : bopts.totiters)+10);
        //sscore.prealloc(NC+(( bopts.totiters == -1 ) ? 10*dim*( bopts.ismoo ? bopts.moodim : 1 ) : bopts.totiters)+10);
        ysort.prealloc( NC+(( bopts.totiters == -1 ) ? DEFITERS(dim) : bopts.totiters)+10);
        sscore.prealloc(NC+(( bopts.totiters == -1 ) ? DEFITERS(dim) : bopts.totiters)+10);

        while ( NC )
        {
            firstone = 1;

            for ( j = 0 ; j < rort.size() ; ++j )
            {
                if ( (bopts.model_d())(j) && ( firstone || ( (bopts.model_y())(rort(j)) < (bopts.model_y())(rort(jj)) ) ) )
                {
                    jj = j;
                    firstone = 0;
                }
            }

            ysort.add(ysort.size());
            ysort("&",ysort.size()-1) = rort(jj);
            rort.remove(jj);

            --NC;
        }
    }

    xinf = nullptr;
















    // ===================================================================
    // ===================================================================
    // Prelimin train the machine learning block(s)
    // ===================================================================
    // ===================================================================

    errstream() << "Model tuning.\n";

    bopts.model_train(dummy,killSwitch);
    bopts.modelimp_train(dummy,killSwitch);

    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================


    // =======================================================================
    // Proceed with Bayesian optimisation
    //
    // - itcnt: iteration count not including random seed.
    //
    // - altitcnt is used as t when calculating beta.  In plain vanilla bayesian
    //   this starts as N+1 (to make sure everything is well defined and noting
    //   the possible presence of previous iterations on the batch that affect the
    //   confidence interval, which is what beta_t measures) and iterates by 1 for
    //   each recommendation.  In batch things are a bit different:
    //
    //   - if itcntmethod = 0 then it starts as (N/B)+1 (where B is the batch size)
    //     and iterates by 1 for every batch.  This is in line with GP-UCB-PE, where
    //     t is the number of batches, and is default behaviour.
    //
    //   - if itcntmethod = 1 then it starts as N+1 and iterates by B for every
    //     batch.  This is in line with GP-BUCB, where t is the number of
    //     recommendations.
    //
    // - betavalmin: minimum beta value in the current batch.  This is necessary
    //   when calculating the variance adjustment factors eg for cBO.
    //
    // - ismultitargrec: 1 set 1 if using multi-objective mu/sigma optimisation
    //   to do multi-recommendation for this iteration.
    //
    // - currintrinbatchsize: see later.
    //
    // =======================================================================

    errstream() << "Training setup.\n";

    int skipcnt     = 0;
    int itcnt       = 0;
    int itcntmethod = bopts.itcntmethod;
    int altitcnt    = ( bopts.model_N_mu() / ( itcntmethod ? 1 : recBatchSize ) ) + 1;

    double betavalmin = 0.0;

    int ismultitargrec      = 0;
    int currintrinbatchsize = 1;

    gentype dummyres(0.0);
    gentype sigmax('R');
    gentype tempval;

    SparseVector<gentype> xappend;

    int justreturnbeta = 0;

    int thisbatchsize   = bopts.intrinbatch;
    int thisbatchmethod = bopts.intrinbatchmethod;

    Matrix<gentype> covarmatrix(thisbatchsize,thisbatchsize);
    Vector<gentype> meanvector(thisbatchsize);

    Vector<SparseVector<gentype> > multivecin(thisbatchsize);
    Vector<SparseVector<double> >  yb(recBatchSize);
    Vector<SparseVector<gentype> > yyb(recBatchSize);

    gentype locpen;

    double ztol  = bopts.ztol;
    double delta = bopts.delta;
    double zeta  = bopts.zeta;
    double nu    = bopts.nu;

    // Variables used in stable bayesian

    int firstevalinseq = 1;

    // =======================================================================
    // The following variable is used to pass variables through to fnfnapprox.
    // =======================================================================

    fninnerinnerArg fnarginner(bopts,
                               yyb("&",0),
                               fnapproxout,
                               sigmax,
                               fres,
                               ( itcntmethod != 3 ) ? altitcnt : itcnt,
                               dim,
                               ztol,
                               delta,
                               zeta,
                               nu,
                               tempval,
                               //yb("&",0),
                               (bopts.a),
                               (bopts.b),
                               (bopts.r),
                               (bopts.p),
                               modD,
                               (bopts.R),
                               B,
                               mig,
                               (bopts.betafn),
                               (bopts.method),
                               justreturnbeta,
                               covarmatrix,
                               meanvector,
                               thisbatchsize,
                               multivecin,
                               direcpre,
                               xytemp,
                               thisbatchmethod,
                               xappend,
                               anyindirect,
                               softmax,
                               itcntmethod,
                               k,
                               (bopts.penalty),
                               locpen,
                               &xinf,
                               Nbasemu,
                               Nbasesigma,
                               Nbasecgt,
                               gridi,
                               isgridopt,
                               iscontopt,
                               isfullgrid,
                               isstable,
                               ysort,
                               stabp,
                               stabpnrm,
                               stabrot,
                               stabmu,
                               stabB,
                               sscore,
                               firstevalinseq,
                               stabZeroPt,
                               (bopts.unscentUse),
                               (bopts.unscentK),
                               (bopts.unscentSqrtSigma),
                               (bopts.stabUseSig),
                               (bopts.stabThresh));

    void *fnarginnerdr = (void *) &fnarginner;

    // =======================================================================
    // Variables used to calculate termination conditions, timers etc
    //
    // maxitcnt: if totiters == -1 then this is set to 10dim, which is apparently
    //           the "standard" number of iterations.
    // =======================================================================

    int timeout     = 0;
    int isopt       = 0;
    double xmtrtime = bopts.maxtraintime;
    //double maxitcnt = ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*effdim*( bopts.ismoo ? bopts.moodim : 1 ) : bopts.totiters ); // note use of effdim, not dim ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*dim : bopts.totiters );
    double maxitcnt = ( ( bopts.totiters == -2 ) ||  usefidbudget ) ? 0 : ( ( bopts.totiters == -1 ) ? DEFITERS(effdim) : bopts.totiters ); // note use of effdim, not dim ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*dim : bopts.totiters );
    int dofreqstop  = ( ( bopts.totiters == -2 ) && !usefidbudget ) ? 1 : 0;

    double *uservars[]     = { &maxitcnt, &xmtrtime, &ztol, &delta, &zeta, &nu, nullptr };
    const char *varnames[] = { "itercount", "traintime", "ztol", "delta", "zeta", "nu", nullptr };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Zero tolerance", "delta", "zeta", "nu", nullptr };

    time_used start_time = TIMECALL;
    time_used curr_time  = start_time;






























    // =======================================================================
    // =======================================================================
    // Main optimisation loop
    // =======================================================================
    // =======================================================================

    errstream() << "Entering main optimisation loop.\n";
    if ( bopts.method != 11 ) { outstream() << "***(method " << bopts.method << ")\n"; }
    else                      { outstream() << "***(method " << bopts.method << "," << bopts.betafn << ")\n"; }

    if ( algseed >= 0 )
    {
        // See comments in startseed

        srand(algseed);
        double dodum = 0;
        randfill(dodum,'S',algseed);

        algseed += 12;
        algseed = ( algseed == -1 ) ? 42 : algseed;
    }

    else if ( algseed == -2 )
    {
        srand((int) time(nullptr)); // seed with time
        double dodum = 0;
        randfill(dodum,'S',(int) time(nullptr));
    }

    // Pre-log before optimisation

    bopts.model_log(1);

    bool stopearly = false;

    // Main loop starts here

    while ( CONT_TEST )
    {
        double varscale = 0;
        std::stringstream errbuffer;

        isfullfid = true;

        // ===================================================================
        // ===================================================================
        // Run intermediate string, if there is one.
        // ===================================================================
        // ===================================================================

        gentype dummyyres;
        Vector<gentype> dummyyres_cgt;

        fidcost = -1;
        xobstype = 2;
        dummyyres.force_int() = 0;
        (*fn)(-1,dummyyres,nullptr,fnarg,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,nullgentype(),muapproxsize,dummyyres_cgt,false,fidcost);

        // ===================================================================
        // ===================================================================
        // Generate experimental recommendations
        // ===================================================================
        // ===================================================================

        xappend.zero();

        numRecs = 0;
        betavalmin = 0;

        // Loop for multi-recommendation

        Vector<int> gridivec;
        Vector<int> gridrefvec;
        Vector<int> gridindtmp(gridind);

        for ( k = 0 ; k < recBatchSize ; ++k )
        {
            anyindirect = isindirect || ( partindirect && k );

            // ===============================================================
            // ===============================================================
            // Multi-recommendation: these change depending on which
            // recommendation we are processing
            // ===============================================================
            // ===============================================================

            fnarginner._q_x     = &(yyb("&",k));

            ismultitargrec      = 0;
            currintrinbatchsize = bopts.intrinbatch;

            int locmethod = bopts.method;

            if ( ( bopts.method == 11 ) && (bopts.betafn).isValVector() )
            {
                if ( ((bopts.betafn)(k)).isValVector() )
                {
                    // =======================================================
                    // =======================================================
                    // Explicit multi-recommendation via multiple strategies
                    // =======================================================
                    // =======================================================

                    NiceAssert( ((bopts.betafn)(k)).size() >= 1  );
                    NiceAssert( ((bopts.betafn)(k)).size() <= 12 );

                    gentype locbetafn;

                    if ( ((bopts.betafn)(k)).size() >= 3 )
                    {
                        gentype tmpbetafn((bopts.betafn)(k)(2));
                        locbetafn = tmpbetafn;
                    }

                    // Note use of cast_ here (without finalisation) just in case any of these are functions.

                    fnarginner._q_locmethod       = ((bopts.betafn)(k)(0)).cast_int(0);
                    fnarginner._q_p               = ( ( ((bopts.betafn)(k)).size() >= 2 ) && !(((bopts.betafn)(k)(1 )).isValVector()) ) ?  (((bopts.betafn)(k)(1 )).cast_double(0)) :  (bopts.p);
                    fnarginner._q_betafn          = ( ( ((bopts.betafn)(k)).size() >= 3 ) && !(((bopts.betafn)(k)(2 )).isValVector()) ) ? &locbetafn                                : &(bopts.betafn);
                    fnarginner._q_modD            = ( ( ((bopts.betafn)(k)).size() >= 4 ) && !(((bopts.betafn)(k)(3 )).isValVector()) ) ?  (((bopts.betafn)(k)(3 )).cast_double(0)) :  bopts.modD;
                    fnarginner._q_delta           = ( ( ((bopts.betafn)(k)).size() >= 5 ) && !(((bopts.betafn)(k)(4 )).isValVector()) ) ?  (((bopts.betafn)(k)(4 )).cast_double(0)) :  delta;
                    fnarginner._q_zeta            = 0.01; //( ( ((bopts.betafn)(k)).size() >= 5 ) && !(((bopts.betafn)(k)(4 )).isValVector()) ) ?  (((bopts.betafn)(k)(4 )).cast_double(0)) :  delta;
                    fnarginner._q_nu              = ( ( ((bopts.betafn)(k)).size() >= 6 ) && !(((bopts.betafn)(k)(5 )).isValVector()) ) ?  (((bopts.betafn)(k)(5 )).cast_double(0)) :  nu;
                    fnarginner._q_a               = ( ( ((bopts.betafn)(k)).size() >= 7 ) && !(((bopts.betafn)(k)(6 )).isValVector()) ) ?  (((bopts.betafn)(k)(6 )).cast_double(0)) :  (bopts.a);
                    fnarginner._q_b               = ( ( ((bopts.betafn)(k)).size() >= 8 ) && !(((bopts.betafn)(k)(7 )).isValVector()) ) ?  (((bopts.betafn)(k)(7 )).cast_double(0)) :  (bopts.b);
                    fnarginner._q_r               = ( ( ((bopts.betafn)(k)).size() >= 9 ) && !(((bopts.betafn)(k)(8 )).isValVector()) ) ?  (((bopts.betafn)(k)(8 )).cast_double(0)) :  (bopts.r);
                    fnarginner._q_thisbatchsize   = ( ( ((bopts.betafn)(k)).size() >= 10) && !(((bopts.betafn)(k)(9 )).isValVector()) ) ?  (((bopts.betafn)(k)(9 )).cast_int   (0)) :  (bopts.intrinbatch);
                    fnarginner._q_thisbatchmethod = ( ( ((bopts.betafn)(k)).size() >= 11) && !(((bopts.betafn)(k)(10)).isValVector()) ) ?  (((bopts.betafn)(k)(10)).cast_int   (0)) :  (bopts.intrinbatchmethod);
                    fnarginner._q_R               = ( ( ((bopts.betafn)(k)).size() >= 12) && !(((bopts.betafn)(k)(11)).isValVector()) ) ?  (((bopts.betafn)(k)(11)).cast_double(0)) :  (bopts.R);

                    if ( ( (bopts.betafn(k)).size() >= 10 ) && !(((bopts.betafn)(k)(9)).isValVector()) )
                    {
                        currintrinbatchsize = (int) ((bopts.betafn)(k)(9));
                    }

                    locmethod = fnarginner._q_locmethod;
                }

                else
                {
                    // =======================================================
                    // Multi-recommendation via multi-objective optimisation
                    // (beta function null)
                    // ...or...
                    // Single explicit strategy (beta function non-null)
                    // =======================================================

                    fnarginner._q_locmethod       = (bopts.method);
                    fnarginner._q_p               = (bopts.p);
                    fnarginner._q_betafn          = &(bopts.betafn);
                    fnarginner._q_modD            = bopts.modD;
                    fnarginner._q_delta           = delta;
                    fnarginner._q_zeta            = zeta;
                    fnarginner._q_nu              = nu;
                    fnarginner._q_a               = (bopts.a);
                    fnarginner._q_b               = (bopts.b);
                    fnarginner._q_r               = (bopts.r);
                    fnarginner._q_thisbatchsize   = (bopts.intrinbatch);
                    fnarginner._q_thisbatchmethod = (bopts.intrinbatchmethod);
                    fnarginner._q_R               = (bopts.R);

                    // If this gets set then we are using multi-objective optimisation on (mu,sigma) to construct relevant number of recommendations
                    ismultitargrec = ((bopts.betafn)(k)).isValNull() ? 1 : 0;
                }
            }

            // ===============================================================
            // ===============================================================
            // Find next experiment parameters using DIRect global optimiser
            // ===============================================================
            // ===============================================================

            {
                //std::stringstream resbuffer;
                errbuffer << "@(" << itcnt << ")";
                //widePrint(errstream(),resbuffer.str(),7);
            }

            time_used bayesbegintime = TIMECALL;

            if ( !ismultitargrec )
            {
                int jji,jjj;

                if ( dim*currintrinbatchsize > xa.size() )
                {
                    int oldibs = (xa.size())/dim;

                    xa.resize(dim*currintrinbatchsize);

                    xmax.resize(dim*currintrinbatchsize);
                    xmin.resize(dim*currintrinbatchsize);

                    for ( jji = oldibs ; jji < currintrinbatchsize ; ++jji )
                    {
                        for ( jjj = 0 ; jjj < dim ; ++jjj )
                        {
                            xmin("&",(jji*dim)+jjj) = xmin(jjj);
                            xmax("&",(jji*dim)+jjj) = xmax(jjj);
                        }
                    }
                }

                if ( direcdim > xa.size() )
                {
                    xa.resize(direcdim);
                }

                int n = anyindirect ? direcdim : dim*currintrinbatchsize;

                // Bounds are different if we are using direcpre

                const Vector<double> &direcmin = anyindirect ? bopts.direcmin : xmin;
                const Vector<double> &direcmax = anyindirect ? bopts.direcmax : xmax;

                double temphardmin = bopts.hardmin;
                double temphardmax = bopts.hardmax;

                bopts.hardmin = valninf();
                bopts.hardmax = valpinf();

                int dres = 0;

                firstevalinseq = 1;

                double B   = ( bopts.B <= 0 ) ? bopts.model_RKHSnorm(0) : bopts.B;
                double mig = bopts.model_maxinfogain(0);

                // Adjust bounds for multifidelity (recommendation is always done
                // for max fidelity, then the real fidelity is chosen afterwards

                if ( ( bopts.getdimfid() > 0 ) && ( locmethod != 18 ) )
                {
                    // variabls xa[n-1] is a fidelity variable, 1 is the max fidelity, so set it 1 and compress the range
                    // when choosing x as per Kandasamy.

                    for ( int jij = 0 ; jij < bopts.getdimfid() ; jij++ )
                    {
                        xmin("&",n-bopts.getdimfid()+jij) = 1;
                        xmax("&",n-bopts.getdimfid()+jij) = 1;

                        xa("&",n-bopts.getdimfid()+jij) = 1;
                    }
                }

                if ( ( locmethod == 12 ) || ( locmethod == 16 ) )
                {
                    double R         = bopts.R;
                    double delta     = bopts.delta;
                    double sampScale = ( locmethod == 12 ) ? ( B + (R*sqrt(2*(mig+1+log(2/delta)))) ) : 1.0;

                    bopts.model_sample(xmin,xmax,sampScale*sampScale);
                }

                gridi   = -1;
                gridref = -1;

                bool printrec = false;

                if ( !isgridopt && iscontopt )
                {
                    // ===============================================================
                    // ===============================================================
                    // Continuous search space, use DIRect to find minimum
                    // ===============================================================
                    // ===============================================================

                    NiceAssert( !anyindirect );

                    // but first over-ride goptssingleobj with relevant components from *this (assuming non-virtual assignment operators)
                    static_cast<GlobalOptions &>(bopts.goptssingleobj) = static_cast<GlobalOptions &>(bopts);

                    retVector<double> tmpva;
                    retVector<double> tmpvb;
                    retVector<double> tmpvc;

                    if ( locmethod == 18 )
                    {
                        // Ask a human

                        std::stringstream xpromptbuff;
                        xpromptbuff << "USER-GENERATED EXPERIMENTAL PARAMETERS REQUIRED\n";
                        promptstream(xpromptbuff.str());

                        for ( int jij = 0 ; jij < n ; jij++ )
                        {
                            std::stringstream promptbuff;

                            promptbuff << "Enter x[" << jij << "] in range [" << xmin(jij) << "," << xmax(jij) << "]";
                            promptbuff << ( ( jij >= n-bopts.getdimfid() ) ? " (fidelity variable)" : "" );
                            promptbuff << ": ";

                            prompttod(xa("&",jij),xmin(jij),xmax(jij),promptbuff.str());
                        }
                    }

                    else
                    {
                        // Use DIRect
                        fnarginner.mode = 1; // turn on single beta, stale x calculation fast mode
                        dres = directOpt(n,xa("&",0,1,n-1,tmpva),dummyres,direcmin(0,1,n-1,tmpvb),direcmax(0,1,n-1,tmpvc),
                                         fnfnapprox,fnarginnerdr,bopts.goptssingleobj,killSwitch);
                        fnarginner.mode = 0; // turn off single beta, stale x calculation fast mode
                    }

                    errstream() << "Unfiltered result x before fidelity selection: " << xa       << "\n";
                    errstream() << "Unfiltered result y before fidelity selection: " << dummyres << "\n";

                    printrec = true;
                }

                else
                {
                    // ===============================================================
                    // ===============================================================
                    // Optimization on a grid, use sweep of gridpoints
                    // (May be mixed-integer, in which case we also have an inner loop)
                    // ===============================================================
                    // ===============================================================

                    //isgridopt - for easier searching
                    // Discrete (grid) search space, use grid search.
                    //
                    // NB: - intrinsic batch can't work out x, so can't use it here.
                    //     - itorem: index in gridi of minimum, so we can remove it from grid.

                    NiceAssert( currintrinbatchsize == 1 );

                    int itorem   = -1;
                    int gridires = -1;

                    retVector<double> tmpva;
                    retVector<double> tmpvb;
                    retVector<double> tmpvc;

                    if ( locmethod == 18 )
                    {
                        // Ask a human

                        std::stringstream xpromptbuff;
                        xpromptbuff << "USER-GENERATED EXPERIMENTAL PARAMETERS REQUIRED\n";
                        promptstream(xpromptbuff.str());

                        for ( int jij = 0 ; jij < n ; jij++ )
                        {
                            std::stringstream promptbuff;

                            promptbuff << "Enter x[" << jij << "] in range [" << xmin(jij) << "," << xmax(jij) << "]";
                            promptbuff << ( ( jij >= n-bopts.getdimfid() ) ? " (fidelity variable)" : "" );
                            promptbuff << ": ";

                            prompttod(xa("&",jij),xmin(jij),xmax(jij),promptbuff.str());
                        }
                    }

                    else
                    {
                        // over-ride goptssingleobj with relevant components from *this (assuming non-virtual assignment operators)
                        static_cast<GlobalOptions &>(bopts.goptssingleobj) = static_cast<GlobalOptions &>(bopts);

                        fnarginner.mode = 1; // turn on single beta, stale x calculation fast mode
                        dres = dogridOpt(n,
                                         xa("&",0,1,n-1,tmpva),
                                         dummyres,
                                         itorem,
                                         gridires,
                                         direcmin(0,1,n-1,tmpvb),
                                         direcmax(0,1,n-1,tmpvc),
                                         fnfnapprox,
                                         fnarginnerdr,
                                         bopts,
                                         *gridsource,
                                         gridindtmp,
                                         killSwitch,
                                         bopts.goptssingleobj,
                                         xmtrtime);
                        fnarginner.mode = 0; // turn off single beta, stale x calculation fast mode
                    }

                    gridi = gridires;
                    //gridind.remove(itorem); - do this later if we get a full observation!
                    gridref = itorem;
                    gridy = (double) bopts.model_y()(Nbasemu+gridi); //((*gridsource).y())(gridi);

                    if ( !iscontopt )
                    {
                        gridindtmp.remove(itorem);
                    }

                    gridivec.add(gridivec.size());     gridivec("&",gridivec.size()-1)     = gridi;
                    gridrefvec.add(gridrefvec.size()); gridrefvec("&",gridrefvec.size()-1) = gridref;

                    errstream() << "Unfiltered result x before fidelity selection: " << xa       << "\n";
                    errstream() << "Unfiltered result y before fidelity selection: " << dummyres << "\n";

                    printrec = true;
                }

                if ( ( locmethod == 12 ) || ( locmethod == 16 ) )
                {
                    bopts.model_unsample();
                }

                // ===============================================================
                // ===============================================================
                // Fidelity evaluation section
                // ===============================================================
                // ===============================================================

                // Put fidelity bounds back how they should be

                if ( ( bopts.getdimfid() > 0 ) && ( locmethod != 18 ) )
                {
                    for ( int jij = 0 ; jij < bopts.getdimfid() ; jij++ )
                    {
                        xmin("&",n-bopts.getdimfid()+jij) = 0;
                        xmax("&",n-bopts.getdimfid()+jij) = 1;
                    }

                    {
                        int dimfid  = bopts.getdimfid();
                        int numfids = bopts.numfids;

                        NiceAssert( !anyindirect );

                        // variabls xa[n-1] is a fidelity variable, 1 is the max fidelity.

                        //errstream() << "phantomxyz fidelity model Gp " << dynamic_cast<const GPR_Generic &>(bopts.getmuapprox_sample(0)).gprGp() << "\n";
                        //errstream() << "phantomxyz fidelity model " << *(bopts.getmuapprox(0)) << "\n";

                        Vector<int> zindex(dimfid);
                        zindex = numfids; // default to highest fidelity if set empty

                        double lambdazmax = 0;
                        double lambdazmin = 0;

                        {
                            SparseVector<SparseVector<gentype> > actfidel;

                            for ( int jij = 0 ; jij < dimfid ; jij++ )
                            {
                                actfidel("&",0)("&",jij) = 1.0;
                            }

                            lambdazmax = (double) bopts.fidpenalty(actfidel); // lambda is an increasing function, so the max penalty corresponds to the highest fidelity
                            lambdazmin = lambdazmax; // remember we want to minimize
                        }

                        double fidkappa0 = bopts.model_kappa0(0); //bopts.fidkappa0;
                        justreturnbeta = 1;
                        double fidbeta = fnfnapprox(dim,&(xa("&",0)),fnarginnerdr); // Get beta value for optimiser
                        justreturnbeta = 0;
                        double fidq = 1.0/((double) (n+2)); // FIXME: just assume an SE kernel here, but in general this depends on the kernel

                        Vector<int> inumfid(dimfid);
                        bool isdone = false;

                        //errstream() << "fidelity fidkappa0 = " << fidkappa0 << "\n";

                        for ( int jij = 0 ; jij < dimfid ; jij++ )
                        {
                            xa("&",n-dimfid+jij) = 1.0/((double) numfids);
                        }

                        SparseVector<double> fidxglob;

                        NonsparseToSparsenear(fidxglob,xa,n,dimfid);

                        for ( inumfid = 1 ; !isdone ; )
                        {
                            // Work out fidelity variable {1/bopts.numfids,2/bopts.numfids,...,1}

                            //errstream() << "phantomxyz fidelity fidbeta = " << fidbeta << "\n";
                            //errstream() << "phantomxyz fidelity fidq = " << fidq << "\n";
                            //errstream() << "phantomxyz fidelity fidc = " << fidc << "\n";
                            //errstream() << "phantomxyz fidelity fidpenalty = " << bopts.fidpenalty << "\n";

                            for ( int jij = 0 ; jij < dimfid ; jij++ )
                            {
                                xa("&",n-dimfid+jij) = ((double) inumfid(jij))/((double) numfids);
                            }

                            SparseVector<double> fidxloc;

                            NonsparseToSparsenear(fidxloc,xa,n,dimfid);

                            // Work out penalty for this z

                            double lambdaz = 0;

                            {
                                SparseVector<SparseVector<gentype> > actfidel;

                                for ( int jij = 0 ; jij < dimfid ; jij++ )
                                {
                                    actfidel("&",0)("&",jij) = xa(n-dimfid+jij);
                                }

                                lambdaz = (double) bopts.fidpenalty(actfidel);
                            }

                            //errstream() << "phantomxyz fidelity lambdaz = " << lambdaz << "\n";
                            //errstream() << "phantomxyz fidelity lambdazmin = " << lambdazmin << "\n";

                            // Need to check we're in the set (7) in Kandasamy

                            // zeta(z)      = sqrt( 1 - (K([xa(0:n-2) z        ],[xa(0:n-2) 1])/K(xa(0:n-2),xa(0:n-2))^2) )
                            // ||zeta||_inf = sqrt( 1 - (K([xa(0:n-2) 1/numfids],[xa(0:n-2) 1])/K(xa(0:n-2),xa(0:n-2))^2) )
                            //
                            // Assumptions: - the kernel is formed as per the paper, so the denominator in the second part
                            //                effectively cancels out the "x" part of the kernel evaluation.
                            //              - the kernel is decreasing in ||x-x'||, so the inf-norm is as stated

                            //errstream() << "phantomxyz fidelity fidxglob = " << fidxglob << "\n";
                            //errstream() << "phantomxyz fidelity fidxloc  = " << fidxloc  << "\n";

                            double fidzetainf = bopts.inf_dist(fidxglob);
                            double fidzetaz   = bopts.inf_dist(fidxloc);

                            //errstream() << "phantomxyz fidelity fidzetainf = " << fidzetainf << "\n";
                            //errstream() << "phantomxyz fidelity fidzetaz   = " << fidzetaz   << "\n";

                            // Posterior standard-deviation calculation.  Remember to take the square-root.

                            gentype resvar;

                            SparseVector<gentype> xxfidxloc; xxfidxloc.castassign(fidxloc);

                            bopts.model_var(resvar,xxfidxloc);

                            double fidtau = sqrt( ( ((double) resvar) >= 0 ) ? ((double) resvar) : 0.0 );

                            //errstream() << "phantomxyz fidelity fidtau = " << fidtau << "\n";

                            // gamma(z) as per (7) in Kandasamy

                            double fidgammaz = fidc*sqrt(fidkappa0)*fidzetaz*(std::pow(lambdaz/lambdazmax,fidq/2.0));

                            //errstream() << "phantomxyz fidelity fidgammaz = " << fidgammaz << "\n";
                            //errstream() << "                    fidc = " << fidc << "\n";
                            //errstream() << "                    fidkappa0 = " << fidkappa0 << "\n";
                            //errstream() << "                    sqrt(fidkappa0) = " << sqrt(fidkappa0) << "\n";
                            //errstream() << "                    fidzetaz = " << fidzetaz << "\n";
                            //errstream() << "                    lambdaz = " << lambdaz << "\n";
                            //errstream() << "                    lambdazmax = " << lambdazmax << "\n";
                            //errstream() << "                    lambdaz/lambdazmax = " << (lambdaz/lambdazmax) << "\n";
                            //errstream() << "                    fidq/2.0 = " << (fidq/2.0) << "\n";
                            //errstream() << "                    fidq = " << fidq << "\n";
                            //errstream() << "                    (std::pow(lambdaz/lambdazmax,fidq/2.0)) = " << (std::pow(lambdaz/lambdazmax,fidq/2.0)) << "\n";

                            // First condition in (7) in Kandasamy
                            //
                            // 1: is the variance big enough?
                            // 2: is the information gap big enough?

                            errstream() << "Fidelity test 1: " << fidtau << " > " << fidgammaz << "(" << fidkappa0 << "," << fidc << ")";

                            if ( fidtau > fidgammaz )
                            {
                                errstream() << "\t" << "pass.\t";

                                // Second condition in (7) in Kandasamy

                                errstream() << "Test 2: " << fidzetaz << " > " << (fidzetainf/sqrt(fidbeta)) << "(" << fidzetainf << "/" << sqrt(fidbeta) << ")";

                                if ( fidzetaz > (fidzetainf/sqrt(fidbeta)) )
                                {
                                    errstream() << "\t" << "pass.\t";

                                    // Record if minimum

                                    errstream() << "Test 3: " << lambdaz << " < " << lambdazmin;

                                    if ( lambdaz < lambdazmin )
                                    {
                                        errstream() << "\t" << "pass.";

                                        zindex = inumfid;
                                        lambdazmin = lambdaz;
                                    }
                                }
                            }

                            errstream() << "\n";

                            for ( int jij = 0 ; jij < dimfid ; jij++ )
                            {
                                ++inumfid("&",jij);

                                if ( inumfid(jij) > numfids )
                                {
                                    inumfid("&",jij) = 1;

                                    if ( jij == dimfid-1 )
                                    {
                                        isdone = true;
                                    }
                                }

                                else
                                {
                                    break;
                                }
                            }
                        }

                        // Human fidelity override

                        if ( bopts.fidover == 1 )
                        {
                            std::stringstream xpromptbuff;
                            xpromptbuff << "USER-GENERATED FIDELITY PARAMETERS REQUIRED:\n";
                            xpromptbuff << "AI-generated recommendation: "; // << xa << "\n";
                            printoneline(xpromptbuff,xa);
                            xpromptbuff << "\n";
                            promptstream(xpromptbuff.str());

                            for ( int jij = 0 ; jij < dimfid ; jij++ )
                            {
                                std::stringstream promptbuff;

                                promptbuff << "Fidelity " << zindex(jij) << " recommended: enter desired fidelity in range " << 1 << " to " << numfids << ": ";

                                prompttoi(zindex("&",jij),1,numfids,promptbuff.str());
                            }
                        }

                        else if ( bopts.fidover == 2 )
                        {
                            for ( int jij = 0 ; jij < dimfid ; jij++ )
                            {
                                int xtmp = (rand()%(2*zindex(jij)))+1;
                                zindex("&",jij) = ( xtmp > zindex(jij) ) ? zindex(jij) : xtmp;
                            }
                        }

                        // Set the fidelity to the minimum

                        for ( int jij = 0 ; jij < dimfid ; jij++ )
                        {
                            xa("&",n-dimfid+jij) = ((double) zindex(jij))/((double) numfids);
                        }
                    }

                    if ( printrec && ( recBatchSize > 1 ) )
                    {
                        std::stringstream xpromptbuff;
                        xpromptbuff << "AI-GENERATED EXPERIMENTAL PARAMETERS: "; // << xa << "\n";
                        printoneline(xpromptbuff,xa);
                        xpromptbuff << "\n";
                        promptstream(xpromptbuff.str());
                    }
                }

                // ===============================================================
                // ===============================================================
                // Stuff - indirection, min/max redefinement etc
                // ===============================================================
                // ===============================================================

                bopts.hardmin = temphardmin;
                bopts.hardmax = temphardmax;

                {
                    //std::stringstream resbuffer;
                    errbuffer << dres;
                    //widePrint(errstream(),resbuffer.str(),4);
                }
                //errstream() << "Return code = " << dres << "\n\n";

                if ( anyindirect )
                {
                    // Direct result needs to be processed to get recommendation batch

                    SparseVector<gentype> tempx;

                    for ( i = 0 ; i < n ; ++i )
                    {
                        tempx("&",i) = xa(i); // gentype sparsevector
                    }

                    if ( xappend.size() && k )
                    {
                        for ( i = n ; i < n+xappend.size() ; ++i )
                        {
                            tempx("&",i) = xappend(i-n);
                        }
                    }

                    (*direcpre).gg(xytemp,tempx);

                    const Vector<gentype> &ghgh = (const Vector<gentype> &) xytemp;

                    for ( i = 0 ; i < dim*currintrinbatchsize ; ++i )
                    {
                        xa("&",i) = (double) ghgh(i);
                    }
                }

                if ( partindirect && !k )
                {
                    NiceAssert( !anyindirect );

                    // update xappend here

                    for ( i = 0 ; i < n ; ++i )
                    {
                        xappend("&",i) = xa(i);
                    }
                }

                //errstream() << dres << "...";

                newRecs = currintrinbatchsize;

                if ( numRecs+newRecs >= xb.size() )
                {
                    xb.resize(numRecs+newRecs);
                    xxb.resize(numRecs+newRecs);
                }

                retVector<double> tmpva;

                for ( jji = 0 ; jji < newRecs ; ++jji )
                {
                    xb("&",numRecs+jji).zero();
                    xb("&",numRecs+jji) = xa((jji*dim),1,(jji*dim)+dim-1,tmpva);
                }
            }

            else
            {
// TODO: need to be able to do grid-search here
                // ===========================================================
                // ===========================================================
                // grid-search is incompatible with this method!
                // ===========================================================
                // ===========================================================

                NiceAssert( currintrinbatchsize == 1 );
                NiceAssert( !anyindirect );
                NiceAssert( !isgridopt );

                // ===========================================================
                // Rather than maximising a simple acquisition function we are
                // multi-objectively maximising (mu,sigma) to give many
                // solutions (recommendations) in a single batch.
                //
                // To do this we basically need to recurse to an inner-loop
                // Bayesian optimiser, so we need to set up all the relevant
                // variables.
                // ===========================================================

                IMP_Expect locimpmeasu;
                BayesOptions locbopts(bopts);

                locimpmeasu.setehimethod(bopts.ehimethodmultiobj);

                locbopts.ismoo             = 1;
                locbopts.impmeasu          = &(static_cast<IMP_Generic &>(locimpmeasu));
                locbopts.method            = 1;
                locbopts.intrinbatch       = 1;
                locbopts.intrinbatchmethod = 0;
                locbopts.startpoints       = bopts.startpointsmultiobj;
                locbopts.totiters          = bopts.totitersmultiobj;

                locbopts.goptssingleobj = bopts.goptsmultiobj;

                Vector<gentype> dummyxres;
                gentype dummyfres;
                int dummyires;
                Vector<double> dummyhypervol;

                Vector<Vector<gentype> > locallxres;
                Vector<gentype> locallfres;
                Vector<Vector<gentype> > locallcres;
                Vector<gentype> locallmres;
                Vector<gentype> locsupres;
                Vector<double> locsscore;
                Vector<int> locparind;

                // ===========================================================
                // Call multi-objective bayesian optimiser
                // ===========================================================

                void *locfnarg[1];

                locfnarg[0] = (void *) &bopts;

                bayesOpt(dim,
                         dummyxres,
                         dummyfres,
                         dummyires,
                         locallxres,
                         locallfres,
                         locallcres,
                         locallmres,
                         locsupres,
                         locsscore,
                         xminalt,
                         xmaxalt,
                         &multiObjectiveCombine,
                         (void *) locfnarg,
                         locbopts,
                         killSwitch);

                // ===========================================================
                // Find pareto (recommendation) set and set numRecs
                // Pareto set will be indexed by locparind
                // ===========================================================

                newRecs = bopts.analyse(locallxres,locallmres,locallcres,dummyhypervol,locparind,1);

                NiceAssert( newRecs );

                // ===========================================================
                // Grow xb and xxb.  We grow enough to fit what has been + 
                // this + one recommendation for each future batch
                // ===========================================================

                if ( numRecs+newRecs >= xb.size() )
                {
                    xb.resize(numRecs+newRecs);
                    xxb.resize(numRecs+newRecs);
                }

                // ===========================================================
                // Transfer recommendations to xb
                // NB: xb is of type Vector<SparseVector<double> >
                //     locallxres(locparind) is of type Vector<SparseVector<gentype> >
                // ===========================================================

                int jji,jki;

                for ( jji = 0 ; jji < newRecs ; ++jji )
                {
                    xb("&",numRecs+jji).resize(locallxres(locparind(jji)).size());

                    for ( jki = 0 ; jki < locallxres(locparind(jji)).size() ; ++jki )
                    {
                        xb("&",numRecs+jji)("&",jki) = (double) locallxres(locparind(jji))(jki);
                    }
                }

                {
                    //std::stringstream resbuffer;
                    errbuffer << "M-rec";
                    //widePrint(errstream(),resbuffer.str(),7);
                }
            }

            // ===============================================================
            // ===============================================================
            // Recommended experiment(s) have been generated successfully
            // ===============================================================
            // ===============================================================

            time_used bayesendtime = TIMECALL;
            bayesruntime = TIMEDIFFSEC(bayesendtime,bayesbegintime);

            {
                //std::stringstream resbuffer;
                errbuffer << " (" << bayesruntime << " sec) ";
                //widePrint(errstream(),resbuffer.str(),15);
            }

            // ===============================================================
            // ===============================================================
            // Update models, record results etc
            // ===============================================================
            // ===============================================================

            justreturnbeta = 1; // This makes fnfnapprox return beta
            betaval        = fnfnapprox(dim,&(xa("&",0)),fnarginnerdr);
            betavalmin     = ( !k || ( betaval < betavalmin ) ) ? betaval : betavalmin;
            justreturnbeta = 0; // This puts things back to standard operation

            // If the iteration count is per batch size then do this

            altitcnt += itcntmethod ? 1 : 0;

            if ( newRecs )
            {
                int ij;

                // Find minimum beta value

                for ( ij = 0 ; ij < newRecs ; ++ij )
                {
//da hell? Why is there nothing here?  What is this loop meant to do?
                }
            }

            while ( newRecs )
            {
                xxb("&",numRecs).zero(); // if we don't do this then things like far points get copied over when resizing, don't want that

                // ===========================================================
                // ===========================================================
                // Data translation step
                // ===========================================================
                // ===========================================================

                SparsenonnearToSparsenear(xxb("&",numRecs),xb(numRecs),dim,bopts.getdimfid());

                isfullfid = true;

                if ( bopts.numfids )
                {
                    // Retrieve fidelity

                    for ( int jij = bopts.getdimfid()-1 ; jij >= 0 ; jij-- )
                    {
                        if ( (int) std::lround(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids )
                        {
                            isfullfid = false;
                        }
                    }
                }

                // ===========================================================
                // ===========================================================
                // Update sigma model if separate ("hallucinated" samples)
                // ===========================================================
                // ===========================================================

                if ( bopts.sigmuseparate )
                {
                    if ( isgridopt && isfullfid && ( isfullgrid || ( bopts.model_d()(Nbasemu+gridi) == 2 ) ) )
                    {
                        ;
                    }

                    else if ( isgridopt && !iscontopt )
                    {
                        bopts.model_setyd_sigma(Nbasesigma+gridi,2,fnapproxout,varscale);
                    }

                    else
                    {
                        bopts.model_addTrainingVector_sigmaifsep(fnapproxout,xxb(numRecs),varscale);
                    }

                    bopts.model_train_sigma(dummy,killSwitch);
                }

                // ===========================================================
                // ===========================================================
                // Record beta, Nrec etc
                // ===========================================================
                // ===========================================================

                if ( isgridopt && ( isfullgrid || ( bopts.model_d()(Nbasemu+gridi) == 2 ) ) )
                {
                    mupred("&",numRecs) = bopts.model_y()(Nbasemu+gridi);
                    sigmapred("&",numRecs) = bopts.model_sigma(0);
                }

                else if ( isgridopt && !iscontopt )
                {
                    bopts.model_muvarTrainingVector(sigmapred("&",numRecs),mupred("&",numRecs),Nbasesigma+gridi,Nbasemu+gridi);
                }

                else
                {
                    bopts.model_muvar(sigmapred("&",numRecs),mupred("&",numRecs),xxb(numRecs),xinf);
                }

                double rmupred    = mupred(numRecs).isCastableToReal()    ? ( (double) mupred(numRecs)    ) : 0.0;
                double rsigmapred = sigmapred(numRecs).isCastableToReal() ? ( (double) sigmapred(numRecs) ) : 0.0;
                double standev    = sqrt(betavalmin)*rsigmapred;

                xb("&",numRecs)("&",dim  )  = numRecs; //newRecs-1;
                xb("&",numRecs)("&",dim+1)  = betaval;
                xb("&",numRecs)("&",dim+2)  = rmupred;
                xb("&",numRecs)("&",dim+3)  = rsigmapred;
                xb("&",numRecs)("&",dim+4)  = rmupred+standev;
                xb("&",numRecs)("&",dim+5)  = rmupred-standev;
                xb("&",numRecs)("&",dim+6)  = 2*standev;
                xb("&",numRecs)("&",dim+7)  = softmax;
                xb("&",numRecs)("&",dim+8)  = 0; // You need this to ensure vector is not sparse!
                xb("&",numRecs)("&",dim+9)  = (double) bayesruntime;
                xb("&",numRecs)("&",dim+10) = (double) mugptraintime;
                xb("&",numRecs)("&",dim+11) = (double) sigmagptraintime;
                xb("&",numRecs)("&",dim+12) = gridi;
                xb("&",numRecs)("&",dim+13) = gridy;
                xb("&",numRecs)("&",dim+14) = ( ( bopts.B <= 0 ) && !bopts.ismoo ) ? bopts.model_RKHSnorm(0) : bopts.B;
                xb("&",numRecs)("&",dim+15) = !bopts.ismoo ? bopts.model_maxinfogain(0) : 0.0;
                xb("&",numRecs)("&",dim+16) = fidtotcost;

                ++numRecs;
                --newRecs;
            }
        }

        //int sigmamod = 0;

        bool doeval = true;

        for ( k = 0 ; k < numRecs ; ++k )
        {
            //errstream() << "e";

            // ===============================================================
            // ===============================================================
            // Get ready to run experiment
            // ===============================================================
            // ===============================================================

            if ( isgridopt )
            {
                gridi = gridivec(k);
            }

            else
            {
                gridi = -1;
            }

            doeval = true;
            int dval = 2;

            // ===============================================================
            // ===============================================================
            // Process User input if required
            // ===============================================================
            // ===============================================================

            if ( bopts.evaluse )
            {
                // Allow user to override and change the experiment, result etc if desired

                std::stringstream outbuff;
                outbuff << "USER INPUT REQUIRED:\n";
                outbuff << "Recommended experiment ";
                printoneline(outbuff,xxb(k));

                outbuff << "\n";
outeroptions:
                outbuff << "Options: p: Proceed with evaluation.\n";
                outbuff << "         q: Stop optimisation early.\n";
                outbuff << "         f: Give feedback.\n";

                std::string evaloption;

                promptstream(outbuff.str()) >> evaloption;

                if ( evaloption == "p" )
                {
                    ;
                }

                else if ( evaloption == "q" )
                {
                    stopearly  = true;
                    doeval     = false;
                }

                else if ( evaloption == "f" )
                {
                    outbuff.str("");
midoptions:
                    outbuff << "Options: y: Directly enter result for recommended experiment.\n";
                    outbuff << "         x: Change recommended experiment, then evaluate normally.\n";
                    outbuff << "         z: Change recommended experiment and directly enter result.\n";
                    outbuff << "         a: Advanced feedback for priors.\n";
                    outbuff << "         c: Cancel\n";

                    std::string evaloption;

                    promptstream(outbuff.str()) >> evaloption;

                    if ( ( evaloption == "x" ) || ( evaloption == "z" ) )
                    {
                        outbuff.str("");
                        outbuff << "Replacement Experiment: ";

                        promptstream(outbuff.str()) >> xxb("&",k);

                        SparsenearToSparsenonnear(xb("&",k),xxb(k),dim,bopts.getdimfid());

                        if ( evaloption == "z" )
                        {
                            goto option_y;
                        }
                    }

                    else if ( evaloption == "y" )
                    {
                        option_y:

                        std::string ystring;

                        outbuff.str("");
                        outbuff << "Result override: ";

                        promptstream(outbuff.str()) >> ystring;

//FIXME: use safe converseion here!
                        fnapproxout = ystring;
                        doeval = false;
                    }

                    else if ( evaloption == "c" )
                    {
                        outbuff.str("");
                        goto outeroptions;
                    }

                    else
                    {
                        outbuff.str("");
                        promptoutstream() << "ERROR: " << evaloption << " is not one of the options.";

                        outbuff.str("");
                        goto midoptions;
                    }
                }

                else if ( evaloption == "c" )
                {
                    ;
                }

                else
                {
                    outbuff.str("");
                    promptoutstream() << "ERROR: " << evaloption << " is not one of the options.";

                    outbuff.str("");
                    goto outeroptions;
                }
            }

            // ===============================================================
            // ===============================================================
            // User input processed - Run experiment
            // ===============================================================
            // ===============================================================

            bool fullobs = true;

            // Need preliminary isfullfid value (assuming x isn't substituted out)

            isfullfid = true;

            if ( bopts.numfids )
            {
                // Retrieve fidelity

                for ( int jij = 0 ; jij < bopts.getdimfid() ; jij++ )
                {
                    if ( (int) std::lround(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids )
                    {
                        isfullfid = false;
                    }
                }
            }

            if ( doeval )
            {
                gentype gridresis = nullgentype();
                Vector<gentype> ycgtis(bopts.numcgt,nullgentype());

                if ( isgridopt && ( isfullgrid || ( bopts.model_d()(Nbasemu+gridi) == 2 ) ) )
                {
                    // Grid may have defined solutions already loaded.

                    gridresis = bopts.model_y()(Nbasemu+gridi);
                    gridresis.negate();
                }

                for ( int iy = 0 ; iy < bopts.numcgt ; ++iy )
                {
                    if ( isgridopt && !iscontopt && ( bopts.model_d_cgt(iy)(Nbasemu+gridi) == 2 ) )
                    {
                        ycgtis("&",iy) = bopts.model_y_cgt(iy)(Nbasemu+gridi);
                    }
                }

                fidcost = -1;
                xobstype = 2;
                fnapproxout.force_int() = static_cast<int>(itcnt+1);
                (*fn)(dim,fnapproxout,&(xb(k)(0)),fnarg,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,gridresis,muapproxsize,ycgtis,false,fidcost);
                negateoutery(fnapproxout);
            }

            else
            {
                fnapproxout = addexp;
                addvar = 0;
                ycgt.resize(0);
                xxreplace.zero();
                replacexx = 0;
                addexp.force_null();
                stopflags = 0;
                xsidechan.resize(0);
                xobstype = dval; //2;
                xobstype_cgt.resize(0);

                readres(fnapproxout,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost);
                negateoutery(fnapproxout);
            }

            // Do this outside the loop for simplicity (only needed on first iteration of while)

            gridref = -1;

            if ( isgridopt && ( gridi >= 0 ) )
            {
                for ( int iy = 0 ; iy < gridind.size() ; ++iy )
                {
                    if ( gridind(iy) == gridi )
                    {
                        gridref = iy;
                        break;
                    }
                }
            }

            bool addobs = true; // remember that multiple results might be packed in fnapproxout
            bool firstinset = true;

            while ( addobs )
            {
                if ( ( stopflags & 1 )               ) { stopnow = 1; }
                if ( ( stopflags & 2 ) && firstinset ) { ++skipcnt;   }

                // Work out if this is a full observation, and also zero out nulls

                fullobs = process_obs(fnapproxout,ycgt,xobstype,xobstype_cgt,bopts.numcgt);

                NiceAssert( !( isgridopt && !iscontopt ) || !replacexx );

                if ( replacexx )
                {
                    gridi = -1;

                    xxb("&",k) = xxreplace;

                    int numfids = bopts.numfids;
                    int dimfid = bopts.getdimfid();

                    isfullfid = true;

                    SparsenearToSparsenonnear(xb("&",k),xxb(k),dim,dimfid);

                    isfullfid = true;

                    if ( numfids )
                    {
                        // Retrieve fidelity

                        for ( int jij = dimfid-1 ; jij >= 0 ; jij-- )
                        {
                            if ( (int) std::lround(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids )
                            {
                                isfullfid = false;
                            }
                        }
                    }
                }

                // Fidelity stuff goes here (we can't do it earlier in case the user over-rides the fidelity

                {
                    int numfids = bopts.numfids;
                    int dimfid = bopts.getdimfid();

                    Vector<int> zindex(dimfid);
                    zindex = numfids; // default to highest fidelity if set empty

                    // ===============================================================
                    // ===============================================================
                    // Update fidelity information
                    // ===============================================================
                    // ===============================================================

                    if ( numfids )
                    {
                        // Retrieve fidelity

                        for ( int jij = 0 ; jij < dimfid ; jij++ )
                        {
                            zindex("&",jij) = (int) std::lround(((double) (xxb(k).n(jij,1)))*numfids);
                        }

                        // Heuristic from Kandasamy supplementary C.1

                        if ( zindex == numfids )
                        {
                            ++fidmaxcnt;
                        }

                        // NB: really don't want to do this on the first iteration!
                        //if ( !((itcnt+1)%10) )
                        if ( !((itcnt+1)%20) )
                        {
                            //if ( fidmaxcnt > 7 )
                            if ( fidmaxcnt > 15 )
                            {
                                //fidc /= sqrt(2.0);
                                fidc /= 2;
                            }

                            //if ( fidmaxcnt < 3 )
                            if ( fidmaxcnt < 5 )
                            {
                                //fidc *= sqrt(2.0);
                                fidc *= 2;
                            }

                            fidc = ( fidc < 0.1 ) ? 0.1 : fidc;
                            fidc = ( fidc > 20 ) ? 20 : fidc;

                            fidmaxcnt = 0;
                            errstream() << "Fidelity: fidc = " << fidc << "\n";
                        }
                    }
                }

                varscale = 0;

                // DESIGN DECISION: all experiments count in fidbudget, even if not the primary experiment
                {
                    SparseVector<SparseVector<gentype> > actfidel;

                    for ( int jij = 0 ; jij < bopts.getdimfid() ; jij++ )
                    {
                        actfidel("&",0)("&",jij) = xxb(k).n(jij,1);
                    }

                    fidtotcost += ( ( fidcost >= 0 ) ? fidcost : ( (double) bopts.fidpenalty(actfidel) ) );
                    varscale    = (double) bopts.fidvar(actfidel);
                    firstinset = false;
                }

                // ===========================================================
                // ===========================================================
                // Update models
                // ===========================================================
                // ===========================================================

                errbuffer << "a(" << mupred(k) << ";" << sigmapred(k) << ")";

                int addpointpos = 0;

                if ( isgridopt && !iscontopt && ( gridi >= 0 ) )
                {
                    // Remove the index from grid for full experiment, update model appropriately

                    fullobs = true;
                    addpointpos = Nbasemu+gridi;

                    if ( !isfullgrid )
                    {
                        if ( xobstype )
                        {
                            bopts.model_setyd_mu(Nbasemu+gridi,xobstype,fnapproxout,varscale);

                            if ( addvar != 0 )
                            {
                                bopts.model_setsigmaweight_addvar(Nbasemu+gridi,Nbasesigma+gridi,addvar);
                            }
                        }

                        else if ( !bopts.model_d()(Nbasemu+gridi) )
                        {
                            fullobs = false;
                        }
                    }

                    else
                    {
                        ; //grid_xobstype("&",gridi) = xobstype; // so we can update later
                    }

                    if ( ycgt.size() )
                    {
                        for ( int iy = 0 ; iy < ycgt.size() ; ++iy )
                        {
                            if ( !xobstype_cgt(iy) ) //ycgt(iy).isValNull() )
                            {
                                if ( bopts.model_d_cgt(iy)(Nbasecgt+gridi) )
                                {
                                    ycgt("&",iy) = bopts.model_y_cgt(iy)(Nbasecgt+gridi);
                                    xobstype_cgt("&",iy) = bopts.model_d_cgt(iy)(Nbasecgt+gridi);
                                }

                                else
                                {
                                    fullobs = false;
                                }
                            }
                        }

                        bopts.model_setyd_cgt(Nbasecgt+gridi,xobstype_cgt,ycgt,varscale);
                    }

                    if ( fullobs )
                    {
                        griddone.add(griddone.size());
                        griddone("&",griddone.size()-1) = gridind(gridref);
                        griddone_xobstype.add(griddone_xobstype.size());
                        griddone_xobstype("&",griddone_xobstype.size()-1) = grid_xobstype(gridref);
                        gridind.remove(gridref);
                        grid_xobstype.remove(gridref);
                    }
                }

                else if ( isgridopt && !iscontopt && ( gridi == -1 ) )
                {
                    bopts.model_addTrainingVector_musigma(((fnapproxout.isValNull()) ? 0.0_gent : fnapproxout),mupred(k),xxb(k),xsidechan,xobstype,varscale);

                    if ( addvar != 0 )
                    {
                        bopts.model_setsigmaweight_addvar(Nmodel,Nsigma,addvar);
                    }

                    if ( ycgt.size() )
                    {
                        bopts.model_addTrainingVector_cgt(ycgt,xxb(k),xsidechan,xobstype_cgt,varscale);
                    }
                }

                else
                {
                    if ( !isfullgrid || ( gridi == -1 ) || replacexx )
                    {
                        bopts.model_addTrainingVector_mu_sigmaifsame(((fnapproxout.isValNull()) ? 0.0_gent : fnapproxout),mupred(k),xxb(k),xsidechan,xobstype,varscale);

                        if ( addvar != 0 )
                        {
                            bopts.model_setsigmaweight_addvar(bopts.model_N_mu()-1,bopts.model_N_sigma()-(numRecs-k),addvar);
                        }
                    }

                    if ( ycgt.size() )
                    {
                        bopts.model_addTrainingVector_cgt(ycgt,xxb(k),xsidechan,xobstype_cgt,varscale);
                    }
                }

                // ===========================================================
                // ===========================================================
                // Update impulse model (if defined, for multiobjective opt)
                // ===========================================================
                // ===========================================================

                if ( bopts.isimphere() )
                {
                    if ( fnapproxout.isValVector() )
                    {
                        const Vector<gentype> &ghgh = (const Vector<gentype> &) fnapproxout;

                        for ( j = 0 ; j < fnapproxout.size() ; ++j )
                        {
                            xinb("&",j) = ghgh(j);
                        }
                    }

                    else
                    {
                        xinb("&",0) = fnapproxout;
                    }

                    xinb.negate();

                    bopts.modelimp_addTrainingVector(nothingmuch,xinb);
                }

                // ===================================================================
                // ===================================================================
                // Fill in fres and ires based on result
                // ===================================================================
                // ===================================================================

//FIXME TS-MOO
                if (    isfullfid
                     && ( xobstype == 2 )
                     && ( ( ires == -1 ) || ( fnapproxout > fres ) )
                     && ( xobstype_cgt >= 1 )
                     && ( ycgt >= 0.0_gent )
                     && ( xxb(k).indsize() == xxb(k).nindsize() ) )
                {
                    errstream() << "New fres (e) = " << fres << "\n";
                    fres = fnapproxout;
                    ires = ( isgridopt && !iscontopt ) ? Nbasemu+gridi : (bopts.model_N_mu())-1;
                }

                // ===============================================================
                // ===============================================================
                // Sort into y list if stability constraints active
                // ===============================================================
                // ===============================================================

                if ( isstable )
                {
                    int jj;

                    if ( !isgridopt )
                    {
                        for ( jj = 0 ; jj < ysort.size() ; ++jj )
                        {
                            if ( ysort(jj) >= addpointpos )
                            {
                                ++(ysort("&",jj));
                            }
                        }
                    }

                    for ( jj = 0 ; jj < ysort.size() ; ++jj )
                    {
//FIXME TS-MOO
                        if ( (bopts.model_y())(addpointpos) < (bopts.model_y())(ysort(jj)) )
                        {
                            break;
                        }
                    }

                    ysort.add(jj);
                    ysort("&",jj) = addpointpos;
                }

                addobs = ( addexp.isValNull() ? false : true );

                gridi = -1;

                if ( addobs )
                {
                    // recurse to next observation to process

                    fnapproxout = addexp;
                    addvar = 0;
                    ycgt.resize(0);
                    xxreplace.zero();
                    replacexx = 0;
                    addexp.force_null();
                    stopflags = 0;
                    xsidechan.resize(0);
                    xobstype = 2;
                    xobstype_cgt.resize(0);

                    readres(fnapproxout,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost);
                    negateoutery(fnapproxout);
                }
            }
        }

        // ===================================================================
        // ===================================================================
        // Train model (unless this is the final iteration)
        // ===================================================================
        // ===================================================================

        time_used mugpbegintime = TIMECALL;

        //if ( CONT_TEST ) - actually train anyway, we may need to use this model!
        {
            bopts.model_train(dummy,killSwitch);
            bopts.modelimp_train(dummy,killSwitch);
        }

        time_used mugpendtime = TIMECALL;
        mugptraintime = TIMEDIFFSEC(mugpendtime,mugpbegintime);

        // ===================================================================
        // ===================================================================
        // Termination condition using model_err
        // ===================================================================
        // ===================================================================

        if ( doeval && dofreqstop )
        {
            // See "Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces", Kirschner et al

            double model_errpred = bopts.model_err(dim,qmin,qmax,killSwitch);

            suppressoutstreamcout();
            outstream() << "min_x err(x) = " << model_errpred << "\n";
            unsuppressoutstreamcout();

            if ( model_errpred < bopts.err )
            {
                isopt = 1;
            }
        }

        // ===================================================================
        // ===================================================================
        // Iterate and go again
        // ===================================================================
        // ===================================================================

++itcnt;
/*
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
*/

        //nullPrint(errstream(),"",-156);
        //resbuffer << "\t - " << fres; // << "\n";
        errstream() << errbuffer.str() << "\n";

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
            timeout = kbquitdet("Bayesian optimisation",uservars,varnames,vardescr);
        }

        altitcnt += itcntmethod ? 0 : 1;

        xinf = nullptr;

        // And log

        bopts.model_log(1);
//errstream() << "phantomxyz: exit !stopnow, stopnow = " << stopnow << "\n";
//errstream() << "phantomxyz: exit !stopearly, stopearly = " << stopearly << "\n";
//errstream() << "phantomxyz: exit !killSwitch, killSwitch = " << killSwitch << "\n";
//errstream() << "phantomxyz: exit !isopt, isopt = " << isopt << "\n";
//errstream() << "phantomxyz: exit !timeout, timout = " << timeout << "\n";
//errstream() << "phantomxyz: exit ( ( itcnt-skipcnt < maxitcnt ) || !maxitcnt ) = ( ( " << itcnt << "-" << skipcnt << " < " << maxitcnt << " ) || !" << maxitcnt << " )\n";
//errstream() << "phantomxyz: exit ( !isgridopt || ( gridind.size() > recBatchSize ) ) = ( !" << isgridopt << " || ( " << gridind.size() << " > " << recBatchSize << " ) )\n";
//errstream() << "phantomxyz: exit ( !usefidbudget || ( fidtotcost < bopts.fidbudget ) ) = ( !" << usefidbudget << " || ( " << fidtotcost << " < " << bopts.fidbudget << " ) )\n";
    }

    errstream() << "\n\n";

    xinf = nullptr;

    NiceAssert( ires >= 0 );

    // =======================================================================
    // Strip out unused pre-cached vectors
    // =======================================================================

//    if ( isgridopt && Ngrid )
//    {
//        while ( gridind.size() )
//        {
//            i = gridind(gridind.size()-1);
//            gridind.remove(gridind.size());
//
//            bopts.model_removeTrainingVector(Nbasemu+i,Nbasesigma+i);
//        }
//    }

    // =======================================================================
    // Final calculation of stability scores
    // =======================================================================

    sscore.resize(muapproxsize);

    if ( isstable )
    {
        sscore = 0.0;

//NB: we *absolutely do not* want to use ysort ordering here!
        retVector<int> tmpva;

        calcsscore(sscore,bopts,cntintvec(sscore.size(),tmpva),stabp,stabpnrm,stabrot,stabmu,stabB);
    }

    else
    {
        sscore = 1.0;
    }

    // =======================================================================
    // Record minimum
    // =======================================================================

    bopts.model_xcopy(xres,ires);
    // This is OK.  The only case where xres is not locally stored is gridopt,
    // and in this case we have asserted !isXconvertNonTrivial(), so x and
    // x convert are the same and backconvert will succeed.

    // =======================================================================
    // See note re max/min changes
    // =======================================================================

    setnegate(fres);

    // =======================================================================
    // Done
    // =======================================================================

    return 0;
}


















class fninnerArg
{
    public:

    fninnerArg(int dim,
               int nres,
               void (*_fn)(gentype &, Vector<gentype> &, void *arg),
               void *_arginner,
               int &_ires,
               Vector<Vector<gentype> > &_allxres,
               Vector<gentype> &_allfres,
               Vector<Vector<gentype> > &_allcres,
               Vector<gentype> &_allmres,
               gentype &_fres,
               Vector<gentype> &_xres,
               Vector<gentype> &_supres,
               volatile int &_killSwitch,
               double &_hardmin,
               double &_hardmax,
               double &_obsnoise,
               int _dimfid,
               int _numfids,
               Vector<double> &_scnoise,
               int _numcgt) : fn(_fn),
                                   arginner(_arginner),
                                   ires(_ires),
                                   allxres(_allxres),
                                   allfres(_allfres),
                                   allcres(_allcres),
                                   allmres(_allmres),
                                   fres(_fres),
                                   xres(_xres),
                                   supres(_supres),
                                   killSwitch(_killSwitch),
                                   hardmin(_hardmin),
                                   hardmax(_hardmax),
                                   obsnoise(_obsnoise),
                                   dimfid(_dimfid),
                                   numfids(_numfids),
                                   scnoise(_scnoise),
                                   numcgt(_numcgt)
    {
        xa.prealloc(dim+1);
        allxres.prealloc(nres);
        allfres.prealloc(nres);
        allcres.prealloc(nres);
        allmres.prealloc(nres);
        supres.prealloc(nres);

        xa.resize(dim);

        return;
    }

    fninnerArg(const fninnerArg &src) : fn(src.fn),
                                   arginner(src.arginner),
                                   ires(src.ires),
                                   allxres(src.allxres),
                                   allfres(src.allfres),
                                   allcres(src.allcres),
                                   allmres(src.allmres),
                                   fres(src.fres),
                                   xres(src.xres),
                                   supres(src.supres),
                                   killSwitch(src.killSwitch),
                                   hardmin(src.hardmin),
                                   hardmax(src.hardmax),
                                   obsnoise(src.obsnoise),
                                   dimfid(src.dimfid),
                                   numfids(src.numfids),
                                   scnoise(src.scnoise),
                                   numcgt(src.numcgt)
    {
        (void) src;
        NiceThrow("Can't duplicate fninnerArg");
        return;
    }

    fninnerArg &operator=(const fninnerArg &src)
    {
        (void) src;
        NiceThrow("Can't copy fninnerArg");
        return *this;
    }

    Vector<gentype> xa;
    SparseVector<gentype> xb;
    SparseVector<gentype> xxb;
    Vector<gentype> dummyxarg;
    void (*fn)(gentype &, Vector<gentype> &, void *arg);
    void *arginner;
    int &ires;
    Vector<Vector<gentype> > &allxres;
    Vector<gentype> &allfres;
    Vector<Vector<gentype> > &allcres;
    Vector<gentype> &allmres;
    gentype &fres;
    Vector<gentype> &xres;
    Vector<gentype> &supres;
    volatile int &killSwitch;
    double &hardmin;
    double &hardmax;
    double &obsnoise;
    int dimfid;
    int numfids;
    Vector<double> &scnoise;
    int numcgt;

    void operator()(int dim, gentype &res, const double *x, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_ycgt, bool onlyAdd, double fidcost)
    {
        // ===========================================================================
        // Inner loop evaluation function.  This is used as a buffer between the
        // actual Bayesian optimiser (above) and the outside-visible optimiser (below)
        // and saves things like timing, beta etc.
        // ===========================================================================

        if ( dim == -1 )
        {
            // This is just to trigger the intermediate command

            (*fn)(res,dummyxarg,arginner);

            return;
        }

        for ( int i = 0 ; i < dim ; ++i )
        {
            xa("&",i).force_double() = x[i];
        }

        int gridi = (int) x[dim+12];

        // gridi will be -1 if this is not grid optimisation.  Otherwise we need
        // to extend the size of xa to argnum+1 and load gridy into it.  The +1
        // will roll around to zero in mlinter, so the result will automatically
        // be loaded into result var which will be evaluated and passed back as
        // result!

        if ( !onlyAdd && ( gridi >= 0 ) )
        {
            xa.resize(dim+1);
            xa("[]",dim) = gridi;
        }

        // =======================================================================
        // Call function and record times
        // =======================================================================

        addvar = 0;

        time_used starttime = TIMECALL;

        if ( !onlyAdd                   ) { (*fn)(res,xa,arginner); }
        if ( onlyAdd || res.isValNull() ) { res = gridres;          }

        time_used endtime = TIMECALL;

        if ( !onlyAdd && ( gridi >= 0 ) )
        {
            xa.resize(dim);
        }

        xobstype_cgt.resize(0);

        readres(res,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost);

        NonsparseToSparsenear(xxb,xa,dim,dimfid);
        NonsparseToSparsenonnear(xb,xa,dim,dimfid);

        gentype altres(res);
        double altaddvar(addvar);
        Vector<gentype> altycgt(ycgt);
        SparseVector<gentype> altxxreplace(xxreplace);
        int altreplacexx(replacexx);
        gentype altaddexp(addexp);
        int altstopflags(stopflags);
        Vector<gentype> altxsidechan(xsidechan);
        int altxobstype(xobstype);
        Vector<int> altxobstype_cgt(xobstype_cgt);
        double altfidcost(fidcost);

        negateoutery(altres);
        gentype *restree = &res;

        if ( altycgt.size() == gridres_ycgt.size() )
        {
            for ( int iy = 0 ; iy < altycgt.size() ; ++iy )
            {
                if ( altycgt(iy).isValNull() && !gridres_ycgt(iy).isValNull() )
                {
                     altycgt("&",iy) = gridres_ycgt(iy);
                }
            }
        }


                bool addobs = true;

                while ( addobs )
                {
                    // Work out if this is a full observation, and also zero out nulls

                    bool fullobs = process_obs(altres,altycgt,altxobstype,altxobstype_cgt,numcgt);
                    bool isfullfid = true;

                    (void) fullobs;

                    if ( altreplacexx )
                    {
                        xxb = altxxreplace;

                        SparsenearToSparsenonnear(xb,xxb,dim,dimfid);
                        SparsenearToNonsparse(xa,xxb,dim,dimfid);
                    }

                    if ( numfids )
                    {
                        // Retrieve fidelity

                        for ( int jij = dimfid-1 ; jij >= 0 ; jij-- )
                        {
                            if ( (int) std::lround(((double) (xxb.n(jij,1)))*numfids) != numfids )
                            {
                                isfullfid = false;
                            }
                        }
                    }

                    // ===========================================================
                    // ===========================================================
                    // Fill in fres and ires based on result
                    // ===========================================================
                    // ===========================================================

                    if (    isfullfid
                         && ( altxobstype == 2 )
                         && ( ( ires == -1 ) || ( altres > fres ) )
                         && ( xobstype_cgt >= 1 )
                         && ( altycgt >= 0.0_gent )
                         && ( xxb.indsize() == xxb.nindsize() ) )
                    {
                        ires = allfres.size();
                        fres = altres;
                        xres = xa;
                    }

                    if ( (double) altres <= hardmin )
                    {
                        // Trigger early termination if hardmin reached

                        killSwitch = 1;
                    }

                    else if ( (double) altres >= hardmax )
                    {
                        // Trigger early termination if hardmax reached

                        killSwitch = 1;
                    }

                    {
                        muapproxsize++;

                        allxres.append(allxres.size(),xa);
                        allfres.append(allfres.size(),altres);
                        allcres.append(allcres.size(),altycgt);
                        allmres.append(allmres.size(),altres);
                        supres.add(supres.size());

                        double dstandev = x[dim+6];
                        double softmax  = x[dim+7];

                        double ucbdist = softmax - ( altres.isCastableToReal() ? ( (double) altres ) : 0.0 );
                        double sigbnd  = ( ucbdist < dstandev ) ? ucbdist : dstandev;

                        supres("&",supres.size()-1).force_vector().resize(20);

                        supres("&",supres.size()-1)("&",0)  = TIMEABSSEC(starttime);
                        supres("&",supres.size()-1)("&",1)  = TIMEABSSEC(endtime);
                        supres("&",supres.size()-1)("&",2)  = x[dim];    // numRecs
                        supres("&",supres.size()-1)("&",3)  = x[dim+1];  // beta
                        supres("&",supres.size()-1)("&",4)  = x[dim+2];  // mu
                        supres("&",supres.size()-1)("&",5)  = x[dim+3];  // sigma
                        supres("&",supres.size()-1)("&",6)  = x[dim+4];  // UCB
                        supres("&",supres.size()-1)("&",7)  = x[dim+5];  // LCB
                        supres("&",supres.size()-1)("&",8)  = dstandev;  // DVAR
                        supres("&",supres.size()-1)("&",9)  = ucbdist;   // UCBDIST
                        supres("&",supres.size()-1)("&",10) = sigbnd;    // SIGBND
                        supres("&",supres.size()-1)("&",11) = x[dim+9];  // DIRect runtime
                        supres("&",supres.size()-1)("&",12) = x[dim+10]; // mu model training time
                        supres("&",supres.size()-1)("&",13) = x[dim+11]; // sigma model training time
                        supres("&",supres.size()-1)("&",14) = TIMEDIFFSEC(endtime,starttime);
                        supres("&",supres.size()-1)("&",15) = x[dim+12]; // grid index (-1 if none)
                        supres("&",supres.size()-1)("&",16) = x[dim+13]; // known grid evaluation (0.0 if none)
                        supres("&",supres.size()-1)("&",17) = x[dim+14]; // B
                        supres("&",supres.size()-1)("&",18) = x[dim+15]; // max info gain
                        supres("&",supres.size()-1)("&",19) = x[dim+16]; // fidelity cost
                    }

                    addobs = ( altaddexp.isValNull() ? false : true );

                    // =======================================================================
                    // Add noise if required (we do this *after* saving the non-noisy result)
                    // =======================================================================

                    if ( obsnoise )
                    {
                        double tmprand;

                        if ( (*restree).isValSet() )
                        {
                            Set<gentype> &z = (*restree).dir_set();

                            if ( ( z.size() >= 7 ) && !z.all()(6).isValNull() )
                            {
                                Vector<gentype> &qxsidechan = z.ncall()("&",6).dir_vector();

                                if ( qxsidechan.size() )
                                {
                                    for ( int i = 0 ; i < qxsidechan.size() ; ++i )
                                    {
                                        if ( scnoise(i) )
                                        {
                                            qxsidechan("&",i) += randnfill(tmprand,0,scnoise(i));
                                        }
                                    }
                                }
                            }

                            if ( ( z.size() >= 1 ) && !z.all()(0).isValNull() )
                            {
                                gentype &qres = z.ncall()("&",0);

                                qres += randnfill(tmprand,0,obsnoise);
                            }
                        }

                        else if ( res.isValDict() )
                        {
                            Dict<gentype,dictkey> &z = (*restree).dir_dict();

                            if ( z.iskeypresent("xsc") && !z("xsc").isValNull() )
                            {
                                Vector<gentype> &qxsidechan = z("&","xsc").dir_vector();

                                if ( qxsidechan.size() )
                                {
                                    for ( int i = 0 ; i < qxsidechan.size() ; ++i )
                                    {
                                        if ( scnoise(i) )
                                        {
                                            qxsidechan("&",i) += randnfill(tmprand,0,scnoise(i));
                                        }
                                    }
                                }
                            }

                            if ( z.iskeypresent("y") && !z("&","y").isValNull() )
                            {
                                gentype &qres = z("&","y");

                                qres += randnfill(tmprand,0,obsnoise);
                            }
                        }

                        else if ( !(*restree).isValNull() )
                        {
                            (*restree) += randnfill(tmprand,0,obsnoise);
                        }
                    }

                    if ( addobs )
                    {
                        // recurse to next observation to process

                        if ( (*restree).isValSet() )
                        {
                            Set<gentype> &z = (*restree).dir_set();

                            if ( ( z.size() >= 5 ) && !z.all()(4).isValNull() )
                            {
                                restree = &z.ncall()("&",4);
                            }
                        }

                        else if ( res.isValDict() )
                        {
                            Dict<gentype,dictkey> &z = (*restree).dir_dict();

                            if ( z.iskeypresent("and") && !z("and").isValNull() )
                            {
                                restree = &z("&","and");
                            }
                        }

                        altres = altaddexp;
                        altaddvar = 0;
                        altycgt.resize(0);
                        altxxreplace.zero();
                        altreplacexx = 0;
                        altaddexp.force_null();
                        altstopflags = 0;
                        altxsidechan.resize(0);
                        altxobstype = 2;
                        altxobstype_cgt.resize(0);
                        altfidcost = -1;

                        readres(altres,altaddvar,altycgt,altxxreplace,altreplacexx,altaddexp,altstopflags,altxsidechan,altxobstype,altxobstype_cgt,altfidcost);
                        negateoutery(altres);
                    }
                }

        // Resize x (it is able to be changed, so this is important)

        xa.resize(dim);

        return;
    }
};



// ===========================================================================
// Inner loop evaluation function.  This is used as a buffer between the
// actual Bayesian optimiser (above) and the outside-visible optimiser (below)
// and saves things like timing, beta etc.
// ===========================================================================

static void fninner(int dim, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_cgt, bool onlyAdd, double fidcost);
static void fninner(int dim, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_cgt, bool onlyAdd, double fidcost)
{
    (*((fninnerArg *) arg))(dim,res,x,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,gridres,muapproxsize,gridres_cgt,onlyAdd,fidcost);
    return;
}




// ===========================================================================
// ===========================================================================
// ===========================================================================
// Outer loop bayesian optimiser.
// ===========================================================================
// ===========================================================================
// ===========================================================================

int bayesOpt(int dim,
              Vector<gentype> &xres,
              gentype &fres,
              int &ires,
              Vector<Vector<gentype> > &allxres,
              Vector<gentype> &allfres,
              Vector<Vector<gentype> > &allcres,
              Vector<gentype> &allmres,
              Vector<gentype> &supres,
              Vector<double> &sscore,
              const Vector<gentype> &xmin,
              const Vector<gentype> &xmax,
              void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
              void *fnarg,
              BayesOptions &bopts,
              svmvolatile int &killSwitch)
{
    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );

    double hardmin = bopts.hardmin;
    double hardmax = bopts.hardmax;

    allxres.resize(0);
    allfres.resize(0);
    allcres.resize(0);
    allmres.resize(0);
    supres.resize(0);
    sscore.resize(0);

    Vector<double> locxres;

    Vector<double> locxmin(dim);
    Vector<double> locxmax(dim);

    int dummy = 0;

    bopts.modelimp_train(dummy,killSwitch); // set xdim if required for random scalarisations

    int i,j,k;
    gentype locfres(0.0);

    for ( i = 0 ; i < dim ; ++i )
    {
        locxmin("&",i) = (double) xmin(i);
        locxmax("&",i) = (double) xmax(i);
    }

    ires = -1;

    double obsnoise;
    Vector<double> scnoise;
    int dimfid = bopts.getdimfid();

    obsnoise = bopts.getfnnoise();
    bopts.getfxnoise(scnoise);

//errstream() << "phantomxyzabcd makenoise = " << bopts.makenoise << "\n";
//errstream() << "phantomxyzabcd obsnoise = " << obsnoise << "\n";
//errstream() << "phantomxyzabcd scnoise = " << scnoise << "\n";
//    fninnerArg optargs(dim,
//                       ( ( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints ) + ( ( bopts.totiters == -1 ) ? 10*dim*( bopts.ismoo ? bopts.moodim : 1 ) : bopts.totiters ),
    fninnerArg optargs(dim,
                       ( ( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints ) + ( ( bopts.totiters == -1 ) ? DEFITERS(dim) : bopts.totiters ),
                       fn,
                       fnarg,
                       ires,
                       allxres,
                       allfres,
                       allcres,
                       allmres,
                       fres,
                       xres,
                       supres,
                       killSwitch,
                       hardmin,
                       hardmax,
                       obsnoise,
                       dimfid,
                       bopts.numfids,
                       scnoise,
                       bopts.numcgt);

    int res = bayesOpt(dim,locxres,locfres,locxmin,locxmax,fninner,(void *) &optargs,bopts,killSwitch,sscore);
    int isstable = bopts.stabpmax;

//errstream() << "allxres = " << allxres << "\n";
//errstream() << "allfres = " << allfres << "\n";
//errstream() << "allcres = " << allcres << "\n";
//errstream() << "allmres = " << allmres << "\n";
//errstream() << "ires = " << ires << "\n";
//errstream() << "fres = " << fres << "\n";
//errstream() << "xres = " << xres << "\n";
//errstream() << "supres = " << supres << "\n";

    killSwitch = 0; // Need to reset this trigger so that subsequent runs don't get hit (it is tripped by hardmin/hardmax)

    if ( bopts.unscentUse )
    {
        // Post-calculation for unscented optimisation

        int unscentK                           = bopts.unscentK;
        const Matrix<double> &unscentSqrtSigma = bopts.unscentSqrtSigma;

        NiceAssert( unscentSqrtSigma.numCols() == unscentSqrtSigma.numRows() );

        int N = allfres.size();
        int d = unscentSqrtSigma.numRows();

        if ( N )
        {
            ires = -1;

            gentype tempres;
            SparseVector<gentype> xxx;
            double modres = 0.0;
//            int dimfid = bopts.getdimfid();

            for ( k = 0 ; k < N ; ++k )
            {
                for ( j = 0 ; j < d ; ++j )
                {
                    xxx("&",j) = allxres(k)(j);
                }

                bopts.model_mu(tempres,xxx);
                modres = (((double) unscentK)/((double) (d+unscentK)))*((double) tempres);

                for ( i = 0 ; i < d ; ++i )
                {
                    for ( j = 0 ; j < d ; ++j )
                    {
                        xxx("&",j) = (allxres(k))(j) + sqrt(d+unscentK)*unscentSqrtSigma(i,j);
                    }

                    bopts.model_mu(tempres,xxx);
                    modres += ((double) tempres)/((double) (2*(d+unscentK)));

                    for ( j = 0 ; j < d ; ++j )
                    {
                        xxx("&",j) = (allxres(k))(j) - sqrt(d+unscentK)*unscentSqrtSigma(i,j);
                    }

                    bopts.model_mu(tempres,xxx);
                    modres += ((double) tempres)/((double) (2*(d+unscentK)));
                }

                allmres("&",k) = -modres; // Don't forget all that negation stuff

                bool ismaxfid = true;

//                if ( dimfid > 0 )
//                {
//                    for ( int jij = 0 ; ( ismaxfid && ( jij < dimfid ) ) ; jij++ )
//                    {
//                        if ( (allxres(k))(allxres(k).size()-dimfid+jij) != 1.0 )
//                        {
//                            ismaxfid = false;
//                        }
//                    }
//                }

                if ( ismaxfid && ( ( ires == -1 ) || ( allmres(k) < allmres(ires) ) ) )
                {
                    ires = k;
                }
            }

            NiceAssert( ires >= 0 );

            fres = allfres(ires);
            xres = allxres(ires);
        }
    }

    if ( isstable )
    {
//FIXME: at this point need to modify allmres to include sscore
        // Need to re-analyse results to find optimal result *that satisfies gradient constraints*

        int N = allfres.size();
//        int dimfid = bopts.getdimfid();

        if ( N )
        {
            ires = -1;

            for ( k = 0 ; k < N ; ++k )
            {
                if ( bopts.stabUseSig )
                {
//allmres("&",k) *= ( ( sscore(k) > bopts.stabThresh ) ? 1.0 : DISCOUNTRATE );
allmres("&",k) *= 1/(1+exp(-1000*(sscore(k)-bopts.stabThresh)));
//                    allmres("&",k) *= 1/(1+exp(-(sscore(k)-bopts.stabThresh)/(sscore(k)*(1-sscore(k)))));
                }

                else
                {
                    allmres("&",k) *= sscore(k); // Allows us to do unscented and stable together
                }

                bool ismaxfid = true;

//                if ( dimfid > 0 )
//                {
//                    for ( int jij = 0 ; ( ismaxfid && ( jij < dimfid ) ) ; jij++ )
//                    {
//                        if ( (allxres(k))(allxres(k).size()-dimfid+jij) != 1.0 )
//                        {
//                            ismaxfid = false;
//                        }
//                    }
//                }

                if ( ismaxfid && ( ( ires == -1 ) || ( allmres(k) < allmres(ires) ) ) )
                {
                    ires = k;
                }
            }

            NiceAssert( ires >= 0 );

            fres = allfres(ires);
            xres = allxres(ires);
        }
    }

//    xres.resize(dim);
//
//    for ( i = 0 ; i < dim ; ++i )
//    {
//        xres("&",i) = locxres(i);
//    }

    return res;
}

















