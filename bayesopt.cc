
/*
PIscale: this is basically just enforcing search in the level-set of results better than what we have so far whp. Consider formalizing this,
possibly by reversing it to look *outside of* the level set of results worse whp. Of course the danger here is that if the model is ill-conceived
then we might exclude the optimal solution, so take care! See also LCB and UCB - PIscale should remove those results below the LCB. And of course
PIscale only really makes sense when you have constraints and are searching in their level-set: otherwise it is enforced automatically.
*/

//when model_convertx is called, only go through near part. far part should mimic near part (it's the other side of a rank constraint)

//FIXME: if you hvae y specified in a gridfile, but x has nulls, you should do
//       a random spread of points for the nulls to induce equality for all x
// crazy idea: if true for 2 x values then you get a sum alpha1 x1 + alpha2 x2
// so a continuum should give a weighted integral.

#define ADDJITTER false

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
             BayesOptions &bopts,
             svmvolatile int &killSwitch);

// Notes:
//
// - Max/min decisions:
//   o Code was originally written to maximise function fn.
//   o All expressions for EI, PI etc were written with this in mind.
//   o To bring it in line with DIRect (and non-Bayesian optimization)
//     I decided to change this to min.
//   o Rather than re-write various negations have been introduced.
// - Returning beta:
//   o We need to return beta in allsres.
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
    return bayesOpt(dim,Xres,fres,cres,Fres,mres,sres,ires,mInd,allxres,allfres,allcres,allFres,allmres,allsres,s_score,is_feas,
                        xmin,xmax,fn,fnarg,*this,killSwitch);
}









void calcsscore(Vector<double> &s_score, const BayesOptions &bopts, const Vector<int> &xdatind, int stabp, double stabpnrm, int stabrot, double stabmu, double stabB);
void calcsscore(Vector<double> &s_score, const BayesOptions &bopts, const Vector<int> &xdatind, int stabp, double stabpnrm, int stabrot, double stabmu, double stabB)
{
    NiceAssert( s_score.size() == xdatind.size() );

    int j;

    for ( j = 0 ; j < s_score.size() ; ++j )
    {
        bopts.model_stabProbTrainingVector(s_score("&",j),xdatind(j),stabp,stabpnrm,stabrot,stabmu,stabB);
    }

    return;
}









class fninnerinnerArg
{
    public:

    fninnerinnerArg(BayesOptions &_bbopts,
                    SparseVector<gentype> &_x,
                    SparseVector<gentype> &_x_fullfid,
                    SparseVector<gentype> &_x_minfid,
                    gentype &_muy,
                    gentype &_sigmay,
                    gentype &_predsigmay,
                    gentype &_ymax,
                    const int &_iters,
                    const int &_dim,
                    const double &_ztol,
                    const double &_delta,
                    const double &_zeta,
                    const double &_defnu,
                    double &_nuscale,
                    gentype &_ires,
                    const double &_a,
                    const double &_b,
                    const double &_r,
                    const double &_p,
                    const double &_modD,
                    const double &_qR,
                    const double &_qBB,
                    const double &_mig,
                    const double &_rkhsnorm,
                    const double &_kappa0,
                    const double &_qRcgt,
                    const double &_qBBcgt,
                    const double &_migcgt,
                    const double &_rkhsnormcgt,
                    const double &_kappa0cgt,
                    gentype &_betafn,
                    gentype &_betafncgt,
                    const int &_locacq,
                    const int &_locacqcgt,
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
                    Vector<double> &_s_score,
                    int &_firstevalinseq,
                    const double &_stabZeroPt,
                    const int &_unscentUse,
                    const int &_unscentK,
                    const Matrix<double> &_unscentSqrtSigma,
                    const int &_stabUseSig,
                    const double &_stabThresh,
                    const int &_iires,
                    const Vector<double> &_fidpencost,
                    const Vector<double> &_fidpenvars,
                    const Vector<Vector<double>> &_fidpenvecs,
                    int _fidmode,
                    double _cgtVarScale) : bbopts(_bbopts),
                                                _q_x(&_x),
                                                _q_x_fullfid(&_x_fullfid),
                                                _q_x_minfid(&_x_minfid),
                                                muy(_muy),
                                                sigmay(_sigmay),
                                                predsigmay(_predsigmay),
                                                ymax(_ymax),
                                                iters(_iters),
                                                dim(_dim),
                                                ztol(_ztol),
                                                _q_delta(_delta),
                                                _q_zeta(_zeta),
                                                _q_defnu(_defnu),
                                                _q_nuscale(_nuscale),
                                                ires(_ires),
                                                _q_a(_a),
                                                _q_b(_b),
                                                _q_r(_r),
                                                _q_p(_p),
                                                _q_modD(_modD),
                                                _q_R(_qR),
                                                _q_BB(_qBB),
                                                _q_mig(_mig),
                                                _q_rkhsnorm(_rkhsnorm),
                                                _q_kappa0(_kappa0),
                                                _q_Rcgt(_qRcgt),
                                                _q_BBcgt(_qBBcgt),
                                                _q_migcgt(_migcgt),
                                                _q_rkhsnormcgt(_rkhsnormcgt),
                                                _q_kappa0cgt(_kappa0cgt),
                                                _q_betafn(&_betafn),
                                                _q_betafncgt(&_betafncgt),
                                                _q_locacq(_locacq),
                                                _q_locacqcgt(_locacqcgt),
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
                                                s_score(_s_score),
                                                firstevalinseq(_firstevalinseq),
                                                stabZeroPt(_stabZeroPt),
                                                unscentUse(_unscentUse),
                                                unscentK(_unscentK),
                                                unscentSqrtSigma(_unscentSqrtSigma),
                                                stabUseSig(_stabUseSig),
                                                stabThresh(_stabThresh),
                                                mode(0),
                                                storeeps(0.0),
                                                storenu(1.0),
                                                storealpha(1.0),
                                                storeepscgt(0.0),
                                                storenucgt(1.0),
                                                storealphacgt(1.0),
                                                iires(_iires),
                                                fidpencost(_fidpencost),
                                                fidpenvars(_fidpenvars),
                                                fidpenvecs(_fidpenvecs),
                                                fidmode(_fidmode),
                                                cgtVarScale(_cgtVarScale)
    {
        return;
    }

    fninnerinnerArg(const fninnerinnerArg &src) : bbopts(src.bbopts),
                                                  _q_x(src._q_x),
                                                  _q_x_fullfid(src._q_x_fullfid),
                                                  _q_x_minfid(src._q_x_minfid),
                                                  muy(src.muy),
                                                  sigmay(src.sigmay),
                                                  predsigmay(src.predsigmay),
                                                  ymax(src.ymax),
                                                  iters(src.iters),
                                                  dim(src.dim),
                                                  ztol(src.ztol),
                                                  _q_delta(src._q_delta),
                                                  _q_zeta(src._q_zeta),
                                                  _q_defnu(src._q_defnu),
                                                  _q_nuscale(src._q_nuscale),
                                                  ires(src.ires),
                                                  _q_a(src._q_a),
                                                  _q_b(src._q_b),
                                                  _q_r(src._q_r),
                                                  _q_p(src._q_p),
                                                  _q_modD(src._q_modD),
                                                  _q_R(src._q_R),
                                                  _q_BB(src._q_BB),
                                                  _q_mig(src._q_mig),
                                                  _q_rkhsnorm(src._q_rkhsnorm),
                                                  _q_kappa0(src._q_kappa0),
                                                  _q_Rcgt(src._q_Rcgt),
                                                  _q_BBcgt(src._q_BBcgt),
                                                  _q_migcgt(src._q_migcgt),
                                                  _q_rkhsnormcgt(src._q_rkhsnormcgt),
                                                  _q_kappa0cgt(src._q_kappa0cgt),
                                                  _q_betafn(src._q_betafn),
                                                  _q_betafncgt(src._q_betafncgt),
                                                  _q_locacq(src._q_locacq),
                                                  _q_locacqcgt(src._q_locacqcgt),
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
                                                  s_score(src.s_score),
                                                  firstevalinseq(src.firstevalinseq),
                                                  stabZeroPt(src.stabZeroPt),
                                                  unscentUse(src.unscentUse),
                                                  unscentK(src.unscentK),
                                                  unscentSqrtSigma(src.unscentSqrtSigma),
                                                  stabUseSig(src.stabUseSig),
                                                  stabThresh(src.stabThresh),
                                                  mode(0),
                                                  storeeps(0.0),
                                                  storenu(1.0),
                                                  storealpha(1.0),
                                                  storeepscgt(0.0),
                                                  storenucgt(1.0),
                                                  storealphacgt(1.0),
                                                  iires(src.iires),
                                                  fidpencost(src.fidpencost),
                                                  fidpenvars(src.fidpenvars),
                                                  fidpenvecs(src.fidpenvecs),
                                                  fidmode(src.fidmode),
                                                  cgtVarScale(src.cgtVarScale)
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
    SparseVector<gentype> *_q_x_fullfid;
    SparseVector<gentype> *_q_x_minfid;
    gentype &muy;
    gentype &sigmay;
    gentype &predsigmay;
    gentype &ymax;
    const int &iters;
    const int &dim;
    const double &ztol;
    double _q_delta;
    double _q_zeta;
    double _q_defnu;
    double &_q_nuscale; // we want to tune this as we go along!
    gentype &ires;
    double _q_a;
    double _q_b;
    double _q_r;
    double _q_p;
    double _q_modD;
    double _q_R;
    double _q_BB;
    double _q_mig;
    double _q_rkhsnorm;
    double _q_kappa0;
    double _q_Rcgt;
    double _q_BBcgt;
    double _q_migcgt;
    double _q_rkhsnormcgt;
    double _q_kappa0cgt;
    gentype *_q_betafn;
    gentype *_q_betafncgt;
    int _q_locacq;
    int _q_locacqcgt;
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
    Vector<double> &s_score;
    int &firstevalinseq;
    const double &stabZeroPt;
    const int &unscentUse;
    const int &unscentK;
    const Matrix<double> &unscentSqrtSigma;
    const int &stabUseSig;
    const double &stabThresh;
    int PIscale = bbopts.PIscale;

    int mode; // 0 for normal, set 1 when entering DirECT, which will calc beta and zero x precisely once then set to 2, set 2 for no beta calc/x zero
    double storeeps;
    double storenu;
    double storealpha;
    double storeepscgt;
    double storenucgt;
    double storealphacgt;

    const int &iires; // if -1 then we don't have a solution, so use alternative acquisition function to force exploration!

    const Vector<double> &fidpencost;
    const Vector<double> &fidpenvars;
    const Vector<Vector<double>> &fidpenvecs;

    int fidmode;

    double cgtVarScale;

    double fidpenalty;
    double maxfidpenalty;

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
            double temp;

            MEMNEWARRAY(xxx,double,n);

            for ( int i = 0 ; i < n ; ++i )
            {
                for ( int j = 0 ; j < n ; ++j ) { xxx[j] = xx[j] + sqrt(n+unscentK)*unscentSqrtSigma(i,j); }

                temp = fnfnapproxNoUnscent(n,xxx);
                res += temp/((double) (2*(n+unscentK)));

                for ( int j = 0 ; j < n ; ++j ) { xxx[j] = xx[j] - sqrt(n+unscentK)*unscentSqrtSigma(i,j); }

                temp = fnfnapproxNoUnscent(n,xxx);
                res += temp/((double) (2*(n+unscentK)));
            }

            MEMDELARRAY(xxx); xxx = nullptr;
        }

        return res;
    }

  double fnfnapproxNoUnscent(int n, const double *xx)
  {
    double delta    = _q_delta;
    double zeta     = _q_zeta;
    double defnu    = _q_defnu;
    double nuscale  = _q_nuscale; // this can change with user input!
    double a        = _q_a;
    double b        = _q_b;
    double r        = _q_r;
    double p        = _q_p;
    double modD     = _q_modD;
    double R        = _q_R;
    double B        = _q_BB;
    double mig      = _q_mig;
    double rkhsnorm = _q_rkhsnorm;
    double kappa0   = _q_kappa0;

    double Rcgt        = _q_Rcgt;
    double Bcgt        = _q_BBcgt;
    double migcgt      = _q_migcgt;
    double rkhsnormcgt = _q_rkhsnormcgt;
    double kappa0cgt   = _q_kappa0cgt;

    // Until we find a feasible solution we stick to variance-only maximization

    int locacq    = ( (bbopts.cgtapprox).size() && ( iires == -1 ) ) ? 10 : _q_locacq;
    int locacqcgt = ( (bbopts.cgtapprox).size() && ( iires == -1 ) ) ? 10 : _q_locacqcgt;

    int thisbatchsize   = _q_thisbatchsize;
    int thisbatchmethod = _q_thisbatchmethod;
    int gridi           = _q_gridi;

    int dimfid  = bbopts.getdimfid();
    int numfids = bbopts.numfids;

    SparseVector<gentype> &x         = (*_q_x);
    SparseVector<gentype> &x_fullfid = (*_q_x_fullfid);
    SparseVector<gentype> &x_minfid  = (*_q_x_minfid);

    gentype &betafn    = *_q_betafn;
    gentype &betafncgt = *_q_betafncgt;

    // NB: n != dim in general (in fact n = dim*multivecin)

    // =======================================================================
    // =======================================================================
    // First work out alpha (mean) and beta (variance) scale factors
    // =======================================================================
    // =======================================================================

    double alpha = 1; // scale for mean
    double beta  = 0; // scale for variance

    int betasgn  = ( locacq >= 0 ) ? 1 : -1;
    int acq      = betasgn*locacq;
    double eps   = 0;
    double nu    = 1;

    double alphacgt = 0; // scale for cgt mean
    double betacgt  = 0; // scale for cgt variance

    int betasgncgt  = ( locacqcgt >= 0 ) ? 1 : -1;
    int acqcgt      = betasgncgt*locacqcgt;
    double epscgt   = 0;
    double nucgt    = 1;

    // We start by working out beta. We do this for all cases, even IMP etc.
    //
    // mode = 0: calculate beta every single time
    // mode = 1: calculate beta, set mode = 2 later (same beta is required multiple times)
    // mode = 2: we have beta, use that

    if ( ( mode == 0 ) || ( mode == 1 ) )
    {
//FIXME: FIDELITY STUFF HERE - do we need a budget-based iteration count? equivalent number of iterations at max fidelity or something?
        double iter    = static_cast<double>(( itcntmethod == 2 ) ? iters-itinbatch : iters); //+1;
        double d       = (double) (dim-dimfid); // don't include fidelity variables in this scaling: it's not really a "feature" in the usual sense
        double Bsize   = (double) thisbatchsize;
        double ell1    = d/(bbopts.model_lenscale(0)); // this needs to be generalized. It's in BOCA, but the paper never says what it is, only that this is the SE case
        double ell1cgt = d/(bbopts.model_lenscale(0)); // this needs to be generalized. It's in BOCA, but the paper never says what it is, only that this is the SE case

        gentype betaargs;
        gentype betaargscgt;

        if ( acq == 11 )
        {
            Vector<gentype> &betaArg = betaargs.force_vector(18);

            betaArg("&",0)  = iter;
            betaArg("&",1)  = d;
            betaArg("&",2)  = delta;
            betaArg("&",3)  = modD;
            betaArg("&",4)  = a;
            betaArg("&",5)  = b;
            betaArg("&",6)  = r;
            betaArg("&",7)  = p;
            betaArg("&",8)  = Bsize;
            betaArg("&",9)  = R;
            betaArg("&",10) = B;
            betaArg("&",11) = mig;
            betaArg("&",12) = rkhsnorm;
            betaArg("&",13) = kappa0;
            betaArg("&",14) = bbopts.model_lenscale(0);
            betaArg("&",15) = bbopts.model_sigma(0);
            betaArg("&",16) = ell1;
            betaArg("&",17) = NUMBASE_PI;
        }

        if ( acqcgt == 11 )
        {
            Vector<gentype> &betaArg = betaargscgt.force_vector(18);

            betaArg("&",0)  = iter;
            betaArg("&",1)  = d;
            betaArg("&",2)  = delta;
            betaArg("&",3)  = modD;
            betaArg("&",4)  = a;
            betaArg("&",5)  = b;
            betaArg("&",6)  = r;
            betaArg("&",7)  = p;
            betaArg("&",8)  = Bsize;
            betaArg("&",9)  = Rcgt;
            betaArg("&",10) = Bcgt;
            betaArg("&",11) = migcgt;
            betaArg("&",12) = rkhsnormcgt;
            betaArg("&",13) = kappa0cgt;
            betaArg("&",14) = bbopts.modelcgt_lenscale(0);
            betaArg("&",15) = bbopts.model_sigma_cgt(0);
            betaArg("&",16) = ell1cgt;
            betaArg("&",17) = NUMBASE_PI;
        }

        switch ( acq )
        {
            // TS-Kand: Thompson-sampling (Kandasamy version - actually Chowdhury version, On Kernelised Multiarm Bandits)
            // Human: ask a human (unreachable, kept as placeholder)
            // HE: human exploration agent, as per BO-Muse paper
            // gpUCB MultiFid: gpUCB Kandasamy multifidelity 2017

            //FIXME ell1 should be the L1 diameter of X computed by scaling each dimension by the inverse of the SE bandwitdh

            case 1:   /* EI                  */ { alpha = 0; nu = 1;     eps = 0;         break; }
            case 2:   /* PI                  */ { alpha = 0; nu = 1;     eps = 0;         break; }

            case 23:  /* LSE Saddle, Bry1    */ { alpha = 0; nu = 1;     eps = 0;         break; }
            case 24:  /* LSE C2LSE, Ngo1     */ { alpha = 0; nu = 1;     eps = 0;         break; }

            case 22:  /* Zero                */ { alpha = 0; nu = 1;     eps = 0;         break; }
            case 0:   /* Mean only           */ { alpha = 1; nu = 1;     eps = 0;         break; }
            case 12:  /* TS-Kand             */ { alpha = 1; nu = 1;     eps = 0;         break; }
            case 16:  /* TS-Kand             */ { alpha = 1; nu = 1;     eps = 0;         break; }
            case 19:  /* HE                  */ { alpha = 1; nu = 1;     eps = 0.001;     break; }
            case 9:   /* PE                  */ { alpha = 0; nu = 1;     eps = 1;         break; }
            case 10:  /* PEc                 */ { alpha = 0; nu = 1;     eps = 1;         break; }
            case 4:   /* gpUCB finite        */ { alpha = 1; nu = defnu; eps = 2*log(modD*iter*iter*(NUMBASE_PI*NUMBASE_PI/(6*delta)));              break; }
            case 7:   /* gpUCB p finite      */ { alpha = 1; nu = defnu; eps = 2*log(modD*numbase_zeta(p)*pow(iter,p)/delta);                        break; }
            case 5:   /* gpUCB infinite      */ { alpha = 1; nu = defnu; eps = (2*log(iter*iter*(2*NUMBASE_PI*NUMBASE_PI/(3*delta)))) + (2*Bsize*d*log(((Bsize==1)?1.0:2.0)*iter*iter*d*b*r*sqrt(log(4*Bsize*d*a/delta)))); break; }
            case 8:   /* gpUCB p infinite    */ { alpha = 1; nu = defnu; eps = (2*log(4*numbase_zeta(p)*pow(iter,p)/delta))           + (2*Bsize*d*log(((Bsize==1)?1.0:2.0)*iter*iter*d*b*r*sqrt(log(4*Bsize*d*a/delta)))); break; }
            case 11:  /* gpUCB user def      */ { alpha = 1; nu = defnu; eps = (double) betafn(betaargs);                                            break; }
            case 13:  /* gpUCB inf RKHS Sri1 */ { alpha = 1; nu = defnu; eps = (2*B) + (300*mig*pow(log(iter/delta),3));                             break; }
            case 3:   /* gpUCB basic         */ { alpha = 1; nu = 1;     eps = 2*log(pow(iter,(2+(d/2)))*(NUMBASE_PI*NUMBASE_PI/(3*delta)));         break; }
            case 6:   /* gpUCB p basic       */ { alpha = 1; nu = 1;     eps = 2*log(2*pow(sqrt(iter),d)*numbase_zeta(p)*pow(iter,p)/delta);         break; }
            case 17:  /* gpUCB MultiFid      */ { alpha = 1; nu = 1;     eps = 0.5*d*log((2*ell1*iter)+1);                                           break; }
            case 14:  /* gpUCB inf RKHS Cho7 */ { alpha = 1; nu = 1;     eps = std::pow(B+(R*sqrt(2*(mig+(1+log(2/delta))))),2);                     break; }
            case 15:  /* gpUCB inf RKHS Bog2 */ { alpha = 1; nu = 1;     eps = std::pow(B+(R*sqrt(2*(mig+(2*log(1/delta))))),2);                     break; }
            case 20:  /* gpUCB as in BO-Muse */ { alpha = 1; nu = 1;     eps = std::pow(B+sqrt(bbopts.model_sigma(0)*((2*log(1/delta))+1+mig)),2)*7; break; }

            case 18:  /* Human               */ { alpha = 1; nu = 1;     eps = valvnan(); break; }
            case 21:  /* Random              */ { alpha = 1; nu = 1;     eps = valvnan(); break; }
            default:  /* fallback trigger    */ { alpha = 1; nu = 1;     eps = valvnan(); break; }
        }

        switch ( acqcgt )
        {
            // TS-Kand: Thompson-sampling (Kandasamy version - actually Chowdhury version, On Kernelised Multiarm Bandits)
            // Human: ask a human (unreachable, kept as placeholder)
            // HE: human exploration agent, as per BO-Muse paper
            // gpUCB MultiFid: gpUCB Kandasamy multifidelity 2017

            //FIXME ell1 should be the L1 diameter of X computed by scaling each dimension by the inverse of the SE bandwitdh

            case 1:   /* EI                  */ { alphacgt = 0; nucgt = 1;     epscgt = 0;         break; }
            case 2:   /* PI                  */ { alphacgt = 0; nucgt = 1;     epscgt = 0;         break; }

            case 23:  /* LSE Saddle, Bry1    */ { alphacgt = 0; nucgt = 1;     epscgt = 0;         break; }
            case 24:  /* LSE C2LSE, Ngo1     */ { alphacgt = 0; nucgt = 1;     epscgt = 0;         break; }

            case 22:  /* Zero                */ { alphacgt = 0; nucgt = 1;     epscgt = 0;         break; }
            case 0:   /* Mean only           */ { alphacgt = 1; nucgt = 1;     epscgt = 0;         break; }
            case 12:  /* TS-Kand             */ { alphacgt = 1; nucgt = 1;     epscgt = 0;         break; }
            case 16:  /* TS-Kand             */ { alphacgt = 1; nucgt = 1;     epscgt = 0;         break; }
            case 19:  /* HE                  */ { alphacgt = 1; nucgt = 1;     epscgt = 0.001;     break; }
            case 9:   /* PE                  */ { alphacgt = 0; nucgt = 1;     epscgt = 1;         break; }
            case 10:  /* PEc                 */ { alphacgt = 0; nucgt = 1;     epscgt = 1;         break; }
            case 4:   /* gpUCB finite        */ { alphacgt = 1; nucgt = defnu; epscgt = 2*log(modD*iter*iter*(NUMBASE_PI*NUMBASE_PI/(6*delta)));                    break; }
            case 7:   /* gpUCB p finite      */ { alphacgt = 1; nucgt = defnu; epscgt = 2*log(modD*numbase_zeta(p)*pow(iter,p)/delta);                              break; }
            case 5:   /* gpUCB infinite      */ { alphacgt = 1; nucgt = defnu; epscgt = (2*log(iter*iter*(2*NUMBASE_PI*NUMBASE_PI/(3*delta)))) + (2*Bsize*d*log(((Bsize==1)?1.0:2.0)*iter*iter*d*b*r*sqrt(log(4*Bsize*d*a/delta)))); break; }
            case 8:   /* gpUCB p infinite    */ { alphacgt = 1; nucgt = defnu; epscgt = (2*log(4*numbase_zeta(p)*pow(iter,p)/delta))           + (2*Bsize*d*log(((Bsize==1)?1.0:2.0)*iter*iter*d*b*r*sqrt(log(4*Bsize*d*a/delta)))); break; }
            case 11:  /* gpUCB user def      */ { alphacgt = 1; nucgt = defnu; epscgt = (double) betafncgt(betaargscgt);                                            break; }
            case 13:  /* gpUCB inf RKHS Sri1 */ { alphacgt = 1; nucgt = defnu; epscgt = (2*B) + (300*migcgt*pow(log(iter/delta),3));                                break; }
            case 3:   /* gpUCB basic         */ { alphacgt = 1; nucgt = 1;     epscgt = 2*log(pow(iter,(2+(d/2)))*(NUMBASE_PI*NUMBASE_PI/(3*delta)));               break; }
            case 6:   /* gpUCB p basic       */ { alphacgt = 1; nucgt = 1;     epscgt = 2*log(2*pow(sqrt(iter),d)*numbase_zeta(p)*pow(iter,p)/delta);               break; }
            case 17:  /* gpUCB MultiFid      */ { alphacgt = 1; nucgt = 1;     epscgt = 0.5*d*log((2*ell1cgt*iter)+1);                                              break; }
            case 14:  /* gpUCB inf RKHS Cho7 */ { alphacgt = 1; nucgt = 1;     epscgt = std::pow(Bcgt+(Rcgt*sqrt(2*(migcgt+(1+log(2/delta))))),2);                  break; }
            case 15:  /* gpUCB inf RKHS Bog2 */ { alphacgt = 1; nucgt = 1;     epscgt = std::pow(Bcgt+(Rcgt*sqrt(2*(migcgt+(2*log(1/delta))))),2);                  break; }
            case 20:  /* gpUCB as in BO-Muse */ { alphacgt = 1; nucgt = 1;     epscgt = std::pow(Bcgt+sqrt(bbopts.model_sigma(0)*((2*log(1/delta))+1+migcgt)),2)*7; break; }

            case 18:  /* Human               */ { alphacgt = 1; nucgt = 1;     epscgt = valvnan(); break; }
            case 21:  /* Random              */ { alphacgt = 1; nucgt = 1;     epscgt = valvnan(); break; }
            default:  /* fallback trigger    */ { alphacgt = 1; nucgt = 1;     epscgt = valvnan(); break; }
        }

        if ( testisinf(eps)    && !testisvnan(eps)    ) { alpha    = 0; eps    = 1; }
        if ( testisinf(epscgt) && !testisvnan(epscgt) ) { alphacgt = 0; epscgt = 1; }

        StrucAssert( justreturnbeta || !testisvnan(eps)    ); // Human and fallback will trigger this, but if either reaches here then that's an error
        StrucAssert( justreturnbeta || !testisvnan(epscgt) ); // Human and fallback will trigger this, but if either reaches here then that's an error

        storeeps   = eps;
        storenu    = nu;
        storealpha = alpha;

        storeepscgt   = epscgt;
        storenucgt    = nucgt;
        storealphacgt = alphacgt;

        x.zero();
        x_fullfid.zero();
        x_minfid.zero();
    }

    else if ( mode == 2 )
    {
        eps   = storeeps;
        nu    = storenu;
        alpha = storealpha;

        epscgt   = storeepscgt;
        nucgt    = storenucgt;
        alphacgt = storealphacgt;
    }

    beta    = nuscale*nu*eps;       // remember that nu can change!
    betacgt = nuscale*nucgt*epscgt; // remember that nu can change!

         if ( justreturnbeta == 1 ) { return beta;     }
    else if ( justreturnbeta == 2 ) { return betacgt;  }
    else if ( justreturnbeta == 3 ) { return alpha;    }
    else if ( justreturnbeta == 4 ) { return alphacgt; }
//errstream() << "phantomxyz testpoint 1 beta = " << beta << "\n";

    // =======================================================================
    // =======================================================================
    // Re-express function input as sparse vector of gentype (size n)
    // =======================================================================
    // =======================================================================

    //x.zero(); - now done in beta calc loop (skipped for most of direct inner loop - we
    //want to leave x alone as much as possible to avoid free/alloc spiral. Note we also
    //the use of sv below if possible (to avoid resetting altcontent))

    bool xsimple = ( !xappend.size() && !anyindirect ) ? true : false;

    // The first point is required to setup nup points on x. The second form can be used thereafter
    if ( mode != 2 )
    {
        for ( int i = 0 ; i < n-dimfid ; ++i ) { x.n("&",i,0).force_double()            = xx[i]; }
        for ( int i = n-dimfid ; i < n ; ++i ) { x.n("&",i-(n-dimfid),1).force_double() = xx[i]; }

        if ( xsimple && ( mode == 1 ) ) { x.makealtcontent(); }

        if ( dimfid && ( fidmode == 0 ) )
        {
            for ( int i = 0 ; i < n-dimfid ; ++i ) { x_fullfid.n("&",i,0).force_double()            = xx[i]; }
            for ( int i = n-dimfid ; i < n ; ++i ) { x_fullfid.n("&",i-(n-dimfid),1).force_double() = 1;     }

            if ( xsimple && ( mode == 1 ) ) { x_fullfid.makealtcontent(); }

            for ( int i = 0 ; i < n-dimfid ; ++i ) { x_minfid.n("&",i,0).force_double()            = xx[i];                  }
            for ( int i = n-dimfid ; i < n ; ++i ) { x_minfid.n("&",i-(n-dimfid),1).force_double() = 1.0/((double) numfids); }

            if ( xsimple && ( mode == 1 ) ) { x_fullfid.makealtcontent(); }
            if ( xsimple && ( mode == 1 ) ) { x_minfid.makealtcontent();  }
        }

        if ( mode == 1 ) { mode = ( xsimple && ( x.altcontent || x.altcontentsp ) && ( !dimfid || x_fullfid.altcontent || x_fullfid.altcontentsp ) && ( !dimfid || x_minfid.altcontent || x_minfid.altcontentsp ) ) ? 2 : 0; }
    }

    else
    {
        for ( int i = 0 ; i < n-dimfid ; ++i ) { x.svdirec(i,xx[i]); }
        for ( int i = n-dimfid ; i < n ; ++i ) { x.svdirec(i,xx[i]); }

        if ( dimfid && ( fidmode == 0 ) )
        {
            for ( int i = 0 ; i < n-dimfid ; ++i ) { x_fullfid.svdirec(i,xx[i]); }
            for ( int i = n-dimfid ; i < n ; ++i ) { x_fullfid.svdirec(i,1.0);   }

            for ( int i = 0 ; i < n-dimfid ; ++i ) { x_minfid.svdirec(i,xx[i]);                  }
            for ( int i = n-dimfid ; i < n ; ++i ) { x_minfid.svdirec(i,1.0/((double) numfids)); }
        }
    }
//errstream() << "phantomxyz testpoint 2 x = " << x << "\n";

    // =======================================================================
    // =======================================================================
    // Add xappend vector to end of vector if required
    // =======================================================================
    // =======================================================================

                                        for ( int i = n ; i < n+xappend.size() ; ++i ) { x.set(i,xappend(i-n));         }
    if ( dimfid && ( fidmode == 0 ) ) { for ( int i = n ; i < n+xappend.size() ; ++i ) { x_fullfid.set(i,xappend(i-n)); } }
    if ( dimfid && ( fidmode == 0 ) ) { for ( int i = n ; i < n+xappend.size() ; ++i ) { x_minfid.set(i,xappend(i-n));  } }

    // =======================================================================
    // =======================================================================
    // Pre-process DIRect input if required
    // =======================================================================
    // =======================================================================

    if ( anyindirect )
    {
        StrucAssert( !( dimfid && ( fidmode == 0 ) ) );

        (*direcpre).gg(xytemp,x);
        x = (const Vector<gentype> &) xytemp;

        n = x.size(); // This is important. dim cannot be used as it means something else.
    }

    // =======================================================================
    // =======================================================================
    // Calculate output mean and variance
    // (Note that muy can be a vector (multi-objective optim) or
    //  a scalar (regular optimiser))
    // =======================================================================
    // =======================================================================

//errstream() << "phantomxyz testpoint 3 x = " << x << "\n";
    NiceAssert( thisbatchsize >= 1 );

    predsigmay = bbopts.model_sigma(); // FIXME: we really should get the actual diagonal offset here! This is lazy, fast, and potentially wrong
    sigmay     = predsigmay;           // assume the current measurement is appropriately noisy by default. See FIXME caveat above

    double overscale = 1;
    bool needvar = false;
    bool answisy = false;

    if ( thisbatchsize == 1 )
    {
        // don't calculate sigmay unless we need to

        needvar = ( ( ( acq    != 0 ) && ( acq    != 12 ) && ( acq    != 16 ) ) || bbopts.isimphere() || PIscale ); // need the variance for calculating acquisition fn
        answisy = ( ( gridi >= 0 ) && ( isfullgrid || ( bbopts.model_d(Nbasemu+gridi) == 2 ) ) );                   // the point x,y is in the training set

//errstream() << "phantomxyz testpoint 4 needvar = " << needvar << "\n";
//errstream() << "phantomxyz testpoint 4 answisy = " << answisy << "\n";
//errstream() << "phantomxyz testpoint 4 dimfid = " << dimfid << "\n";
//errstream() << "phantomxyz testpoint 4 fidmode = " << fidmode << "\n";
        if ( dimfid && ( fidmode == 0 ) )
        {
            StrucAssert( predsigmay.size() == 1 );

            // We want posterior mean/variance for the full fidelity function
            // (that's what we're maximising), so even if this is a grid point
            // it's not a grid point we're interested in directly, so we need
            // to do the full calculation
            //
            // We also need *predicted* variance (if we choose to observe x)

            gentype predsigmay_max = predsigmay;

            bbopts.model_predmuvar(predsigmay,sigmay,muy,x_fullfid,x);
            bbopts.model_predmuvar(predsigmay_max,sigmay,muy,x_fullfid,x_minfid);

            beta *= ((double) predsigmay_max);
            sigmay /= predsigmay;
            sigmay += ((bbopts.model_sigma())/((double) predsigmay));
            //overscale = (double) predsigmay;

            sigmay += predsigmay;

//FIXME
/*
FIXME: I think the method suggested below will actually tend to choose noisy low-fidelity measurements. We need to cancel that out! We could fix this by scaling beta by
???
(var+sigma)/(predvar+sigma)
or
predvar / sigma
or
predvar - sigma

for all of the acquisition functions: we currently assume an exact measure. We need to assume a noisy measure.
to do this maybe we can just add the noise in measurement to the current posterior noise? In this case, the
predictive noise is predsigmay, not sigma()



I previously  had sigmay += predsigmay for non multi-fid as well, Justification was:
// transfer noise on measurement to noise on model (recall additivity of Gaussians)
// NB: most implementations don't do this, but mathematically they should. eg when
//     calculating EI or PI with a noisy measurement the measurement noise matters!
*/
        }

        else
        {
            if      ( isgridopt &&               ( gridi >= 0 ) && answisy ) { muy = bbopts.model_y(Nbasemu+gridi);                                         }
            else if ( isgridopt && !iscontopt && ( gridi >= 0 ) && needvar ) { bbopts.model_muvarTrainingVector(sigmay,muy,Nbasesigma+gridi,Nbasemu+gridi); }
            else if ( isgridopt && !iscontopt && ( gridi >= 0 )            ) { bbopts.model_muTrainingVector(muy,Nbasemu+gridi);                            }
            else if (                                              needvar ) { bbopts.model_muvar(sigmay,muy,x);                                            }
            else                                                             { bbopts.model_mu(muy,x);                                                      }
//errstream() << "phantomxyz testpoint 5 muy = " << muy << "\n";
//errstream() << "phantomxyz testpoint 5 sigmay = " << sigmay << "\n";

           //sigmay += predsigmay; - why was this here?
        }

        // Sanity check here!

        if      ( testisvnan(sigmay)      ) { errstream() << "sigma NaN in Bayesian Optimisation!\n"; if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; } else { sigmay.force_double() = 0.0; } }
        else if ( testisinf(sigmay)       ) { errstream() << "sigma inf in Bayesian Optimisation!\n"; if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; } else { sigmay.force_double() = 0.0; } }
        else if ( !( sigmay >= 0.0_gent ) ) { errstream() << "sigma neg in Bayesian Optimisation!\n"; if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; } else { sigmay.force_double() = 0.0; } }

        // Combine noise contributions

        OP_sqrt(sigmay);      // This works elementwise on vectors
//errstream() << "phantomxyz testpoint 6 muy = " << muy << "\n";
//errstream() << "phantomxyz testpoint 6 sigmay = " << sigmay << "\n";
    }

    else
    {
        StrucAssert( !dim );
        StrucAssert( !( dimfid && ( fidmode == 0 ) ) );

        // x(given above) = [ v0, v1, ..., vthisbatchsize-1 ], where xi has dimension
        // dim.  Need to split these up and calculate mean vector and covariance matrix,
        // then reform them to get the min mean vector and modified determinat of the
        // covariance matrix (the geometric mean of the determinants).

        meanvector.resize(thisbatchsize);
        covarmatrix.resize(thisbatchsize,thisbatchsize);
        multivecin.resize(thisbatchsize);

        for ( int i = 0 ; i < thisbatchsize ; ++i )
        {
            multivecin("&",i).zero();
            for ( int j = 0 ; j < dim ; ++j ) { multivecin("&",i)("&",j) = x((i*dim)+j); }
            bbopts.model_mu(meanvector("&",i),multivecin(i),nullptr);
        }

        bbopts.model_covar(covarmatrix,multivecin);

        switch ( thisbatchmethod )
        {
            case 1:  { muy = mean(meanvector); sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize)); break; }
            case 2:  { muy = min(meanvector);  sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize)); break; }
            case 3:  { muy = max(meanvector);  sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            case 4:  { muy = mean(meanvector); sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            case 5:  { muy = min(meanvector);  sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            case 17: { muy = max(meanvector);  sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace())); break; }
            default: { muy = max(meanvector);  sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize)); break; }
        }

        // Sanity check here!

        if      ( testisvnan(sigmay)      ) { errstream() << "sigma NaN in Bayesian Optimisation!\n"; if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; } else { sigmay.force_double() = 0.0; } }
        else if ( testisinf(sigmay)       ) { errstream() << "sigma inf in Bayesian Optimisation!\n"; if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; } else { sigmay.force_double() = 0.0; } }
        else if ( !( sigmay >= 0.0_gent ) ) { errstream() << "sigma neg in Bayesian Optimisation!\n"; if ( sigmay.isValVector() ) { sigmay.force_vector(muy.size()) = 0.0_gent; } else { sigmay.force_double() = 0.0; } }

        // Combine noise contributions

        OP_sqrt(sigmay);      // This works elementwise on vectors
    }

    double yymax        = ( ymax.isValVector() ) ? 0.0 : ((double) ymax);
    double yymaxcorrect = yymax;
    double hh           = -bbopts.h;
    double hhcgt        = bbopts.hcgt;

    // Correction factors (min/max and other misc)

    if ( !ymax.isValVector() && !muy.isValVector() )
    {
//Was += 1, but see ADDITIONAL
        muy          += 1.5; // DESIGN DECISION: typically mu in [-1,0] here, but this is bad for the constrained case. We add 1 to "fix" the problem and make mu in [0,1].
                             // ADDITIONAL: we observe that the posterior mean can actually range over something like [-1.5,0.5], so we add 1.5
        yymaxcorrect += 1.5; // NEED TO CORRECT THIS AS WELL FOR EI
        hh           += 1.5; // NEED TO CORRECT THIS AS WELL FOR LEVEL-SET

        // We observe that if muy is near zero then in the modified constrained
        // case this tends to "squash" the fidelity. So we correct for that.

//        if ( (bbopts.cgtapprox).size() && ( bbopts.cgtmethod == 1 ) && ( iires >= 0 ) )
//        {
//            muy          *= 0.9; muy          += 0.1;
//            yymaxcorrect *= 0.9; yymaxcorrect += 0.1;
//        }
    }

    int numcgt = bbopts.numcgt;
    gentype altsig(bbopts.model_sigma(0));

    if ( numcgt )
    {
        altsig = bbopts.model_sigma_cgt(0);
    }

    Vector<gentype> mucgt(numcgt,0.0_gent);
    Vector<gentype> varcgtsq(numcgt,altsig);
    Vector<gentype> varcgtsq_pred(numcgt,altsig);

    double totalconstraintvariance = 0;

    double probofvalid = 1;
    bool   objresfeas  = true;

    if ( numcgt )
    {
        bool inlevelset = true;

        NiceAssert( thisbatchsize == 1 );

        if ( dimfid && ( fidmode == 0 ) ) { bbopts.model_predmuvar_cgt(varcgtsq_pred,varcgtsq,mucgt,x_fullfid,x); varcgtsq += varcgtsq_pred; }
        else                              { bbopts.model_muvar_cgt(varcgtsq,mucgt,x);                                                        }

        varcgtsq.scale(cgtVarScale);

        for ( int i = 0 ; i < numcgt ; ++i )
        {
            if      ( testisvnan(varcgtsq(i))      ) { errstream() << "sigma NaN in Bayesian Optimisation!\n"; varcgtsq("&",i).force_double() = 0.0; }
            else if ( testisinf(varcgtsq(i))       ) { errstream() << "sigma inf in Bayesian Optimisation!\n"; varcgtsq("&",i).force_double() = 0.0; }
            else if ( !( varcgtsq(i) >= 0.0_gent ) ) { errstream() << "sigma neg in Bayesian Optimisation!\n"; varcgtsq("&",i).force_double() = 0.0; }

            double ecgt   = (double) mucgt(i);
            double vcgtsq = (double) varcgtsq(i);

            if ( ( bbopts.cgtmethod == 4 ) || ( bbopts.cgtmethod == 5 ) ) { vcgtsq = 0.01; }

            if      ( vcgtsq > ztol*ztol ) { probofvalid *= normPhi(ecgt/sqrt(vcgtsq)); }
            else if ( ecgt < 0           ) { probofvalid = 0;                           }

            if ( ecgt <= (bbopts.cgtcertain)*sqrt(vcgtsq) ) { inlevelset = false; }

            totalconstraintvariance += sqrt(vcgtsq);
        }

        // mucgt >= cgtmargin
        // mucgt-cgtmargin >= 0

        for ( int i = 0 ; i < mucgt.size() ; ++i ) { mucgt("&",i) -= bbopts.cgtmargin; } // ((bbopts.cgtmargin)*2*sqrt(varcgtsq(i)))

        if ( bbopts.cgtmethod == 6 ) { probofvalid = inlevelset ? 1.0 : 0.0; objresfeas = inlevelset; } // alternative "in or out of level set" test
    }

    // =======================================================================
    // =======================================================================
    // Work out PI scale (on the objective, but not including cgt)
    // =======================================================================
    // =======================================================================
//errstream() << "phantomxyz testpoint 7 muy = " << muy << "\n";
//errstream() << "phantomxyz testpoint 7 sigmay = " << sigmay << "\n";

    double resscale = 1.0;
    bool   resfeas  = true;

    if ( PIscale && ( iires >= 0 ) ) // don't do this if we don't have any solution yet, as that indicates to feasible results and hence completely flat CGT!
    {
        NiceAssert( !ymax.isValVector() && !muy.isValVector() );

        if      ( !answisy && ( (double) sigmay > ztol ) ) { double tempvalue = (((((double) muy)-yymaxcorrect)/((double) sigmay))*NUMBASE_SQRT1ON2); resscale = normPhi(tempvalue); }
        else if ( (double) muy <= yymaxcorrect           ) { resscale = 0; objresfeas = false; if ( PIscale == 2 ) { resfeas = false; } }
        else                                               { resscale = 1;                                                              }
    }

    // =======================================================================
    // =======================================================================
    // Incorporate constraints into mean and variance if cgtmethod == 1
    // =======================================================================
    // =======================================================================

    if ( numcgt && ( bbopts.cgtmethod == 1 ) && ( iires >= 0 ) ) // don't do this if we don't have any solution yet, as that indicates to feasible results and hence completely flat CGT!
    {
        // Work out mean and variance of delta(c_i(x) >= 0) for all constraints c_i

        Vector<double> cgtmean((bbopts.cgtapprox).size()+1);
        Vector<double> cgtvar ((bbopts.cgtapprox).size()+1);

        for ( int i = 0 ; i < (bbopts.cgtapprox).size() ; i++ )
        {
            // Sanity check here!

            double ecgt   = (double) mucgt(i);
            double vcgtsq = (double) varcgtsq(i);

            // This is basically PI
            // E[delta(c(x)>=0)]
            // var[delta(c(x)>=0)]

                //double phihere = normPhi(ecgt/sqrt(vcgtsq));
                //cgtmean("&",i) = phihere; // E[delta(c(x)>=0)]
                //cgtvar("&",i)  = sqrt((cgtmean(i)*cgtmean(i)*(1-phihere)) + ((1-cgtmean(i))*(1-cgtmean(i))*phihere));  // var[delta(c(x)>=0)]

            if ( vcgtsq > ztol*ztol ) { cgtmean("&",i) = normPhi(ecgt/sqrt(vcgtsq)); cgtvar("&",i) = sqrt((cgtmean(i)*cgtmean(i)*(1-cgtmean(i))) + ((1-cgtmean(i))*(1-cgtmean(i))*cgtmean(i))); }
            else if ( ecgt <= 0     ) { cgtmean("&",i) = 0;                          cgtvar("&",i) = 0; }
            else                      { cgtmean("&",i) = 1;                          cgtvar("&",i) = 0; }
        }

        cgtmean("&",(bbopts.cgtapprox).size()) = (double) muy;
        cgtvar("&",(bbopts.cgtapprox).size())  = (double) sigmay;

        // See Goodman: On the Exact Variance of Products

        double meantot = prod(cgtmean); // product of means
        double vartot = 0;              // sum of all products of means and variances (one for each contributor), but not all means

        for ( int i = 1 ; i < (1<<(cgtmean.size())) ; ++i )
        {
            double varsub = 1;

            // Binary, eg size = 3:
            // i = 001b, 010b, 011b, 100b, 101b, 110b, 111b

            int k = 0;

            for ( int jj = 1 ; jj < (1<<(cgtmean.size())) ; jj *= 2, ++k )
            {
                // Binary, eg size = 3:
                // jj = 001b, 010b, 100b

                if ( ( i & jj ) ) { varsub *= cgtvar(k) *cgtvar (k); } // bit set, posterior variance
                else              { varsub *= cgtmean(k)*cgtmean(k); } // bit not set, posterior mean squared
            }

            vartot += varsub;
        }

        muy    = meantot;      // corrected mean
        sigmay = sqrt(vartot); // corrected variance

        probofvalid = 1;
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

        if ( firstevalinseq && ( acq == 1 ) )
        {
            firstevalinseq = 0;

            // ...then the stability score on x(j)...

            s_score.resize(ysort.size()+1); // This will eventually become the products
            s_score("&",ysort.size()) = 0.0;

            retVector<double> tmpva;
            calcsscore(s_score("&",0,1,ysort.size()-1,tmpva),bbopts,ysort,stabp,stabpnrm,stabrot,stabmu,stabB);

            // ...and finally convert stability scores to combined stability scores.

            s_score *= -1.0;
            s_score += 1.0;

            for ( int j = s_score.size()-1 ; j >= 1 ; --j ) { s_score("&",0,1,j-1,tmpva) *= s_score(j); }

            if ( stabUseSig )
            {
                for ( int j = 0 ; j < s_score.size() ; ++j )
                {
                    //s_score("&",j) = ( s_score(j) >= stabThresh ) ? 1.0 : DISCOUNTRATE;
                    s_score("&",j) = 1/(1+exp(-1000*(s_score(j)-stabThresh)));
                    //s_score("&",j) = 1/(1+exp(-(s_score(j)-stabThresh)/(s_score(j)*(1-s_score(j)))));
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

        if ( muy.isValVector() ) { for ( int jj = 0 ; jj < muy.size() ; ++jj ) { xmean("&",jj) = ((const Vector<gentype> &) muy)(jj); } }
        else                     { xmean("&",0) = muy;                                                                                  }

        gentype altresv;

        bbopts.modelimp_imp(ires,altresv,xmean,sigmay); // IMP may or may not update sigmay
        res = ((double) ires);

        muy.force_double()    = ((double) res); // This is done so that passthrough will apply to IMP
        sigmay.force_double() = sqrt((double) altresv);

        if ( ymax.isValVector() && ( ( acq == 1 ) || ( acq == 2 ) ) )
        {
            // EI and PI require ymax, which is a function of the scalarisation itself!
            // FIXME: this is a bit of a hack really.

            gentype muymax(ymax);
            SparseVector<gentype> muymean;

            for ( int jj = 0 ; jj < muymax.size() ; ++jj ) { muymean("&",jj) = ((const Vector<gentype> &) muymax)(jj); }

            gentype dummy,dummyvar,muypr;
            dummyvar.force_vector(muymax.size()) = dummy;
            bbopts.modelimp_imp(muypr,dummy,muymean,dummyvar);

            yymax = ((double) muypr);
        }
    }

    double exploitFactor = 0;
    double exploreFactor = 0;

    double exploitFactorcgt = 0;
    double exploreFactorcgt = 0;

    double minexploitFactor = 0;
    double minexploreFactor = 0;

    double minexploitFactorcgt = 0;
    double minexploreFactorcgt = 0;

    double maxexploitFactor = 0;
    double maxexploreFactor = 0;

    double maxexploitFactorcgt = 0;
    double maxexploreFactorcgt = 0;

    bool objresfeascgt  = numcgt ? false : true; // start with false so we can set true if any *component* is true


    // else - Do this part anyhow to allow for standard GP-UCB.  acq 0 does passthrough of IMP, other methods will change things around
    {
//FIXME - TS-MOO
        NiceAssert( !(muy.isValVector()) );
//errstream() << "phantomxyz mu = " << muy << "\n";
//errstream() << "phantomxyz sigmay = " << sigmay << "\n";
//errstream() << "phantomxyz yymaxcorrect = " << yymaxcorrect << "\n";
//errstream() << "phantomxyz gridi = " << gridi << "\n";

        if ( isstable && ( ( acq == 1 ) || !( ( acq == 2 ) || ( acq == 23 ) || ( acq == 24 ) ) ) )
        {
            double locmuy     = (double) muy;
            double locsigmay  = (double) sigmay;

            // Stable EI or gpUCB variant

            switch ( acq )
            {
                case 1: // EI
                {
                    double parta = 0;
                    double partb = 0;
                    double partadec = 0;
                    double partbdec = 0;
                    double scalea,scaleb;

                    for ( int k = 0 ; k <= ysort.size() ; ++k ) // Range 0 to N+1
                    {
                        double dmuy  = (double) muy;
                        double dsigy = (double) sigmay;

                        double ykdec = ( k == 0            ) ? stabZeroPt : ((double) (bbopts.model_y(ysort(k-1))));
                        double yk    = ( k == ysort.size() ) ? 1e12       : ((double) (bbopts.model_y(ysort(k  )))); // 1e12 is a placeholder for bugnum. Never actually used.

                        parta = normphi((dmuy-yk-zeta)/dsigy);
                        partb = normPhi((dmuy-yk-zeta)/dsigy);

                        partadec = normphi((dmuy-ykdec)/dsigy);
                        partbdec = normPhi((dmuy-ykdec)/dsigy);

                        parta = partadec - ( ( k == ysort.size() ) ? 0.0 : parta );
                        partb = partbdec - ( ( k == ysort.size() ) ? 0.0 : partb );

                        scalea = s_score(k);
                        scaleb = s_score(k)*(dmuy-ykdec)/dsigy;

                        for ( int i = 0 ; i < k-1 ; ++i )
                        {
                            double yidec = ( i == 0            ) ? stabZeroPt : ((double) (bbopts.model_y(ysort(i-1))));
                            double yi    = ( i == ysort.size() ) ? 1e12       : ((double) (bbopts.model_y(ysort(i  )))); // i == ysort.size() never attained.

                            scaleb += s_score(i)*(yi-yidec)/dsigy;
                        }

                        exploitFactor += (parta*scalea)+(partb*scaleb);
                    }

                    exploitFactor *= ((double) sigmay)*stabscore;

                    break;
                }

                default: // gpUCB variant
                {
//                    if ( isstable && !stabUseSig )
//                    {
//                        // For independent vars: var(x.y) = var(x).var(y) + var(x).mu(y).mu(y) + var(y).mu(x).mu(x)
//
////                        double stabvar = stabscore*(1-stabscore);
////errstream() << stabvar << "\t" << stabscore << "\t" << locmuy << "\t" << locsigmay << "\t";
//
////THIS REALLY DOESNT WORK AT ALL!  MAKES VARIANCE LARGE EVERYWHERE, WHICH HINDERS EXPLORATION                    locsigmay = (locsigmay*stabvar) + (locsigmay*stabscore*stabscore) + (locmuy*locmuy*stabvar);
//                        locmuy    = stabscore*locmuy;
////phantomx - new variance calc
//locsigmay = stabscore*stabscore*locsigmay;
////errstream() << "(\t" << locmuy << "\t" << locsigmay << "\t)\t";
//                    }

                    if ( !stabUseSig ) { locmuy *= stabscore; locsigmay *= stabscore*stabscore; }

                    exploitFactor = alpha*locmuy;
                    exploreFactor = (betasgn*sqrt(beta)*locsigmay);

                    if ( stabUseSig ) { exploitFactor *= stabscore; exploreFactor *= stabscore; }

                    break;
                }
            }

            if ( ( bbopts.minstdev > 0 ) && ( locsigmay < bbopts.minstdev ) )
            {
                probofvalid = 0;
                objresfeas  = false;
            }

            exploitFactor *= probofvalid;
            exploreFactor *= probofvalid;

            StrucAssert( !numcgt );

            maxexploitFactor = 1;
            maxexploreFactor = 1;

            maxexploitFactorcgt = numcgt;
            maxexploreFactorcgt = numcgt;
        }

        else
        {
            StrucAssert( !isstable );

            double muR         = (double) muy;
            double sigmaR      = (double) sigmay;
            double yymaxR      = yymaxcorrect;
            double hhR         = hh;
            double alphaR      = alpha;
            double betaR       = beta;
            double betasgnR    = betasgn;
            double modelsigmaR = bbopts.model_sigma(0);
            double mumin       = 0; // based on the shifting done previously
            double mumax       = 2; // based on assumption that the objective lies (more or less) in [0,1], plus error factor, then shifted

            double exploitFactorR = 0;
            double exploreFactorR = 0;

            double minexploitFactorR = 0;
            double minexploreFactorR = 0;

            double maxexploitFactorR = 1;
            double maxexploreFactorR = 1;

            int acqR = acq;

            switch ( acqR )
            {
                case 1: // EI
                {
                    double Phizm = normPhi(((muR-yymaxR)/sigmaR)-zeta);
                    double phizm = normphi(((muR-yymaxR)/sigmaR)-zeta);

                    if ( sigmaR > ztol ) { exploitFactorR = ((muR-yymaxR-zeta)*Phizm) + (sigmaR*phizm);      exploreFactorR = 0; }
                    else                 { exploitFactorR = ( muR > yymaxR+zeta ) ? (muR-yymaxR-zeta) : 0.0; exploreFactorR = 0; }

                    minexploitFactorR = 0;
                    maxexploitFactorR = (mumax-mumin);

                    minexploreFactorR = 0;
                    maxexploreFactorR = 0;

                    break;
                }

                case 2: // PI
                {
                    double Phiz = normPhi((muR-yymaxR)/sigmaR);

                    if ( sigmaR > ztol ) { exploitFactorR = Phiz;                     exploreFactorR = 0; }
                    else                 { exploitFactorR = ( muR > yymaxR ) ? 1 : 0; exploreFactorR = 0; }

                    minexploitFactorR = 0;
                    maxexploitFactorR = 1;

                    minexploreFactorR = 0;
                    maxexploreFactorR = 0;

                    break;
                }

                case 23: // LSE Saddle, Bry1 - note additional factor to ensure positivity
                {
                    exploitFactorR = 0;
                    exploreFactorR = (1.96*sigmaR) - std::abs(muR-hhR) + ((mumax-mumin)+std::abs(hhR));

                    minexploitFactorR = 0;
                    maxexploitFactorR = 0;

                    minexploreFactorR = 0; //(1.96*modelsigmaR);
                    maxexploreFactorR = (1.96*(1+modelsigmaR)) + ((mumax-mumin)+std::abs(hhR));

                    break;
                }

                case 24: // LSE C2LSE, Ngo1 - rescaled and squished, but still has the same minima. Also bounds are a bit arbitrary
                {
                    exploitFactorR = 0;
                    exploreFactorR = log(1+(sigmaR/std::max(bbopts.lseeps,std::abs(muR-hhR))));

                    minexploitFactorR = 0;
                    maxexploitFactorR = 0;

                    minexploreFactorR = 0; //log(1+modelsigmaR/((mumax-mumin)+std::abs(hhR)));
                    maxexploreFactorR = minexploreFactorR+1; // This is arbitrary! The real max is very large, and we don't want to "squash" what matters.
                                                             // "real" max: log(1+((1+modelsigmaR)/lseeps))

                    break;
                }

                default: // gpUCB variants
                {
                    exploitFactorR = alphaR*(muR-mumin);
                    exploreFactorR = (betasgnR*sqrt(betaR)*sigmaR);

                    minexploitFactorR = 0;
                    maxexploitFactorR = alphaR*(mumax-mumin);

                    minexploreFactorR = 0; //(betasgnR*sqrt(betaR)*modelsigmaR);
                    maxexploreFactorR = (betasgnR*sqrt(betaR)*(1+modelsigmaR));

                    break;
                }
            }

            if ( ( bbopts.minstdev > 0 ) && ( sigmaR < bbopts.minstdev ) )
            {
                probofvalid = 0;
                objresfeas  = false;
            }

            exploitFactor += probofvalid*exploitFactorR;
            exploreFactor += probofvalid*exploreFactorR;

            minexploitFactor += minexploitFactorR;
            minexploreFactor += minexploreFactorR;

            maxexploitFactor += maxexploitFactorR;
            maxexploreFactor += maxexploreFactorR;
        }

        for ( int qq = 0 ; qq < numcgt ; ++qq )
        {
            double muR         = (double) mucgt(qq);
            double sigmaR      = sqrt((double) varcgtsq(qq));
            double yymaxR      = hhcgt; // 0
            double hhR         = hhcgt; // 0
            double alphaR      = alphacgt;
            double betaR       = betacgt;
            double betasgnR    = betasgncgt;
            double modelsigmaR = bbopts.model_sigma_cgt(qq);
            double mumin       = -15; // see next line
            double mumax       = 15;  // based on assumption that the objective lies (more or less) in [-10,10]

            double exploitFactorR = 0;
            double exploreFactorR = 0;

            double minexploitFactorR = 0;
            double minexploreFactorR = 0;

            double maxexploitFactorR = 0;
            double maxexploreFactorR = 0;

            int acqR = acqcgt;

            switch ( acqR )
            {
                case 1: // EI
                {
                    double Phizm = normPhi(((muR-yymaxR)/sigmaR)-zeta);
                    double phizm = normphi(((muR-yymaxR)/sigmaR)-zeta);

                    if ( sigmaR > ztol ) { exploitFactorR = ((muR-yymaxR-zeta)*Phizm) + (sigmaR*phizm);      exploreFactorR = 0; }
                    else                 { exploitFactorR = ( muR > yymaxR+zeta ) ? (muR-yymaxR-zeta) : 0.0; exploreFactorR = 0; }

                    minexploitFactorR = 0;
                    maxexploitFactorR = (mumax-mumin);

                    minexploreFactorR = 0;
                    maxexploreFactorR = 0;

                    break;
                }

                case 2: // PI
                {
                    double Phiz = normPhi((muR-yymaxR)/sigmaR);

                    if ( sigmaR > ztol ) { exploitFactorR = Phiz;                     exploreFactorR = 0; }
                    else                 { exploitFactorR = ( muR > yymaxR ) ? 1 : 0; exploreFactorR = 0; }

                    minexploitFactorR = 0;
                    maxexploitFactorR = 1;

                    minexploreFactorR = 0;
                    maxexploreFactorR = 0;

                    break;
                }

                case 23: // LSE Saddle, Bry1 - note additional factor to ensure positivity
                {
                    exploitFactorR = 0;
                    exploreFactorR = (1.96*sigmaR) - std::abs(muR-hhR) + ((mumax-mumin)+std::abs(hhR));

                    minexploitFactorR = 0;
                    maxexploitFactorR = 0;

                    minexploreFactorR = 0; //(1.96*modelsigmaR);
                    maxexploreFactorR = (1.96*(1+modelsigmaR)) + ((mumax-mumin)+std::abs(hhR));

                    break;
                }

                case 24: // LSE C2LSE, Ngo1 - rescaled and squished, but still has the same minima. Also bounds are a bit arbitrary
                {
                    exploitFactorR = 0;
                    exploreFactorR = log(1+(sigmaR/std::max(bbopts.lseeps,std::abs(muR-hhR))));

                    minexploitFactorR = 0;
                    maxexploitFactorR = 0;

                    minexploreFactorR = 0; //log(1+modelsigmaR/((mumax-mumin)+std::abs(hhR)));
                    maxexploreFactorR = 1; // This is arbitrary! The real max is very large, and we don't want to "squash" what matters.
                                           // "real" max: log(1+((1+modelsigmaR)/lseeps))

                    break;
                }

                default: // gpUCB variants
                {
                    exploitFactorR = alphaR*(muR-mumin);
                    exploreFactorR = (betasgnR*sqrt(betaR)*sigmaR);

                    minexploitFactorR = 0;
                    maxexploitFactorR = alphaR*(mumax-mumin);

                    minexploreFactorR = 0; //(betasgnR*sqrt(betaR)*modelsigmaR);
                    maxexploreFactorR = (betasgnR*sqrt(betaR)*(1+modelsigmaR));

                    break;
                }
            }

            double probofvalidcgt = 0;

            if ( !( ( bbopts.minstdev > 0 ) && ( sigmaR < bbopts.minstdev ) ) )
            {
                probofvalidcgt = 1;
                objresfeascgt  = true;
            } // Note the negated logic here!

            exploitFactorcgt += probofvalidcgt*exploitFactorR;
            exploreFactorcgt += probofvalidcgt*exploreFactorR;

            minexploitFactorcgt += minexploitFactorR;
            minexploreFactorcgt += minexploreFactorR;

            maxexploitFactorcgt += maxexploitFactorR;
            maxexploreFactorcgt += maxexploreFactorR;
        }
    }

    if ( PIscale != 2 )
    {
        exploitFactor *= resscale;
        exploreFactor *= resscale;
    }

    res = (exploitFactor+exploreFactor) + (exploitFactorcgt+exploreFactorcgt);

    double resmin = 0;
    double resmax = (maxexploitFactor+maxexploreFactor) + (maxexploitFactorcgt+maxexploreFactorcgt);
//errstream() << "phantomxyz testpoint 8 res = " << res << "\n";
//errstream() << "phantomxyz testpoint 8 muy = " << muy << "\n";
//errstream() << "phantomxyz testpoint 8 sigmay = " << sigmay << "\n";

    // =======================================================================
    // =======================================================================
    // Add constraint variance if required.
    // =======================================================================
    // =======================================================================

    if      ( acq == 10             ) { res += betasgn*     totalconstraintvariance; objresfeas = true; objresfeascgt = true; resfeas = true; }
    else if ( bbopts.cgtmethod == 2 ) { res += betasgn*beta*totalconstraintvariance; objresfeas = true; objresfeascgt = true; resfeas = true; }
    else if ( bbopts.cgtmethod == 3 ) { res += betasgn*     totalconstraintvariance; objresfeas = true; objresfeascgt = true; resfeas = true; }

    // =======================================================================
    // =======================================================================
    // Scale by the fidelity cost penalty.
    // =======================================================================
    // =======================================================================

    if ( dimfid && ( fidmode == 0 ) ) { res *= overscale*maxfidpenalty/fidpenalty; }

    // =======================================================================
    // =======================================================================
    // Negate on return (DIRect minimises, Bayesian Optimisation maximises -
    // but keep in mind note about max/min changes).
    // =======================================================================
    // =======================================================================

//    double baseoff = 1.5+(bbopts.numcgt); // To compensate for the fact that the variance contributions may be negative
//                                          // Note also that in EI the posterior mean may fall outside of [-1,1], so use 1.5 rather than 1 as a fudge
//    res = (baseoff-res)/(baseoff+1); // Set up for minimiser
    res = (resmax-res)/(resmax-resmin);

    // When only one of the acquisition (obj and cgt) contribute, we can generalise infeasibility

    if ( (           !objresfeas    ) && ( !numcgt || ( acqcgt == 22 ) ) ) { resfeas = false; }
    if ( ( numcgt && !objresfeascgt ) && (            ( acq    == 22 ) ) ) { resfeas = false; }

    if ( !objresfeas && !objresfeascgt ) { resfeas = false; }

    // Flag infeasibility for DIRect using NaN

    if      ( !resfeas     ) { res  = valvnan("infeasible inner loop result"); }
    else if ( PIscale == 2 ) { res *= resscale;                                }

//errstream() << "inner loop (" << objresfeas << "," << objresfeascgt << "," << acq << "," << acqcgt << "," << resfeas << "," << PIscale << "): ";
//printoneline(errstream(),x);
//errstream() << " -> " << res;
//errstream() << " =?= ( " << resscale << " * " << probofvalid << " * ( (" << alpha << "*" << muy << ") + " << " ( " << beta << " * " << sigmay << " ) )";
//errstream() << " = ( " << exploitFactor << " + " << exploreFactor << " ))";
//errstream() << " + ( " << exploitFactorcgt << " + " << exploreFactorcgt << " ))";
//errstream() << "\t ( " << minexploreFactorcgt << " to " << maxexploreFactorcgt << " )";
//errstream() << "\t" << mucgt;
//errstream() << "(maxy: " << yymaxcorrect <<  ")" << "\n";

    return res;
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
              Vector<int> &gridind,
              svmvolatile int &killSwitch,
              const DIRectOptions &dopts,
              double xmtrtime,
              bool isfullgrid);


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
              Vector<int> &gridind,
              svmvolatile int &killSwitch,
              const DIRectOptions &dopts,
              double xmtrtime,
              bool isfullgrid)
{
    NiceAssert( dim > 0 );
    NiceAssert( gridind.size() > 0 );

    ires     = -1;
    gridires = -1;

    int locires     = -1;
    int locgridires = -1;

    int dimfid  = bopts.getdimfid();
//    int numfids = bopts.numfids;

    int &gridi  = (*((fninnerinnerArg *) fnarg))._q_gridi;
    int Nbasemu = (*((fninnerinnerArg *) fnarg)).Nbasemu;

    const Vector<double>         &fidpencost = (*((fninnerinnerArg *) fnarg)).fidpencost;
    const Vector<Vector<double>> &fidpenvecs = (*((fninnerinnerArg *) fnarg)).fidpenvecs;

    double &fidpenalty    = (*((fninnerinnerArg *) fnarg)).fidpenalty;
    double &maxfidpenalty = (*((fninnerinnerArg *) fnarg)).maxfidpenalty;

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

        Vector<double> locxmin(dim);
        Vector<double> locxmax(dim);

        int numnulls    = 0;
        int numnullfids = 0;

        for ( int j = 0 ; j < dim ; ++j )
        {
            if ( !bopts.model_x(Nbasemu+gridi).direcref(j).isValNull() )
            {
                // element is pre-defined by the grid
                xxres("&",j)   = (double) bopts.model_x(Nbasemu+gridi).direcref(j);
                locxmin("&",j) = xx[j];
                locxmax("&",j) = xx[j];
            }

            else if ( xmin(j) == xmax(j) )
            {
                // element not pre-defined, but no range, so this is the only possible value
                xxres("&",j)   = xmin(j);
                locxmin("&",j) = xmin(j);
                locxmax("&",j) = xmax(j);
            }

            else
            {
                // element not defined, so need to call inner optimizer to find value
                locxmin("&",j) = xmin(j);
                locxmax("&",j) = xmax(j);

                if ( j <= dim-dimfid ) { ++numnulls;    }
                else                   { ++numnullfids; }
            }
        }

        // if we have the complete solution (even if there are nulls) then this will be true
        bool havesoln = ( ( (bopts.cgtapprox).size() == 0 ) && ( ( gridi >= 0 ) && ( isfullgrid || ( bopts.model_d(Nbasemu+gridi) == 2 ) ) ) );

        if ( fidpencost.size() )
        {
            maxfidpenalty = fidpencost(fidpencost.size()-1); // propogate through
            fidpenalty    = fidpencost(fidpencost.size()-1); // default propagate through

            for ( int k = 0 ; k < fidpencost.size() ; ++k )
            {
                bool isusable = true;

                for ( int j = dim-dimfid ; ( j < dim ) && isusable ; ++j )
                {
                    if      ( ( fidpenvecs(k)(j-(dim-dimfid)) != xxres(j) ) && ( locxmin(j) == locxmax(j) ) ) { isusable = false; }
                    else if ( ( xxres(j) < locxmin(j) ) || ( xxres(j) > locxmax(j) ) )                        { isusable = false; }
                }

                if ( isusable )
                {
                    fidpenalty = fidpencost(k); // propagate through

errstream() << "phantomxabc: maxfidpenalty " << maxfidpenalty << "\n";
errstream() << "phantomxabc: fidpenalty " << fidpenalty << "\n";
errstream() << "phantomxabc: locxmin " << locxmin << "\n";
errstream() << "phantomxabc: locxmax " << locxmax << "\n";
errstream() << "phantomxabc: numnulls " << numnulls << "\n";

                    if ( !numnulls )
                    {
                        // x is fully defined by the grid, so we need only evaluate the acquisition function
                        tempfres = (*fn)(dim,xx,fnarg);
                    }

                    else if ( havesoln && !numnullfids )
                    {
                        // x is fully defined by the grid, so we need only evaluate the acquisition function
                        tempfres = (*fn)(dim,xx,fnarg);

                        // Random null fill
                        for ( int j = 0 ; j < dim ; ++j )
                        {
                            if ( bopts.model_x(Nbasemu+gridi).direcref(j).isValNull() && ( xmin(j) != xmax(j) ) ) { randufill(xxres("&",j),xmin(j),xmax(j)); }
                        }
                    }

                    else
                    {
                        // x is not fully defined by the grid, so we need to invoke direct to optimise the continuous factors
                        // Note that the bounds xmin,xmax are already clamped to the non-null grid components to prevent change

                        gentype dummyres;

                        int directres = directOpt(dim,xxres,dummyres,locxmin,locxmax,fnfnapprox,fnarg,dopts,killSwitch,ADDJITTER);
                        errstream() << "direct(" << gridi << "," << directres << "):";

                        tempfres = !dummyres.isValNull() ? ((double) dummyres) : valvnan("no solution found");
                    }
errstream() << "phantomxabc: tempfres " << tempfres << "\n";

                    if ( !std::isnan(tempfres) && ( ( locires == -1 ) || ( tempfres < (double) fres ) ) )
                    {
                        for ( int j = 0 ; j < dim ; ++j ) { x[j] = xx[j]; }

                        fres        = tempfres;
                        locires     = i;
                        locgridires = gridi;
                    }
                }
            }
        }

        else
        {
            maxfidpenalty = 1.0; // default propogate through
            fidpenalty    = 1.0; // default propagate through

errstream() << "phantomxabc: maxfidpenalty " << maxfidpenalty << "\n";
errstream() << "phantomxabc: fidpenalty " << fidpenalty << "\n";
errstream() << "phantomxabc: locxmin " << locxmin << "\n";
errstream() << "phantomxabc: locxmax " << locxmax << "\n";
errstream() << "phantomxabc: numnulls " << numnulls << "\n";

            if ( !numnulls )
            {
                // x is fully defined by the grid, so we need only evaluate the acquisition function
                tempfres = (*fn)(dim,xx,fnarg);
            }

            else if ( havesoln && !numnullfids )
            {
                // x is fully defined by the grid, so we need only evaluate the acquisition function
                tempfres = (*fn)(dim,xx,fnarg);

                // Random null fill
                for ( int j = 0 ; j < dim ; ++j )
                {
                    if ( bopts.model_x(Nbasemu+gridi).direcref(j).isValNull() && ( xmin(j) != xmax(j) ) ) { randufill(xxres("&",j),xmin(j),xmax(j)); }
                }
            }

            else
            {
                // x is not fully defined by the grid, so we need to invoke direct to optimise the continuous factors
                // Note that the bounds xmin,xmax are already clamped to the non-null grid components to prevent change

                gentype dummyres;

                int directres = directOpt(dim,xxres,dummyres,locxmin,locxmax,fnfnapprox,fnarg,dopts,killSwitch,ADDJITTER);
                errstream() << "direct(" << gridi << "," << directres << "):";

                tempfres = !dummyres.isValNull() ? ((double) dummyres) : valvnan("no solution found");
            }
errstream() << "phantomxabc: tempfres " << tempfres << "\n";

            if ( !std::isnan(tempfres) && ( ( locires == -1 ) || ( tempfres < (double) fres ) ) )
            {
                for ( int j = 0 ; j < dim ; ++j ) { x[j] = xx[j]; }

                fres        = tempfres;
                locires     = i;
                locgridires = gridi;
            }
        }

        if ( !std::isnan(tempfres) && ( ( tempfres <= hardmin ) || ( tempfres >= hardmax ) ) )
        {
            break;
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

    return ( ires >= 0 ) ? 0 : -200;
}

















// ===========================================================================
// Evaluates [ mu sigma ] for multi-recommendation via multi-objective
// inner loop.  In multi-recommendation via multi-obj systems there is an
// inner loop that multi-objectively maximises [ mu sigma ], so this function
// evaluates and returns that.  Everything needs to be in the negative
// quadrant, so we need to negative sigma.
// ===========================================================================

void multiObjectiveCombine(gentype &res, Vector<gentype> &x, void *arg);
void multiObjectiveCombine(gentype &res, Vector<gentype> &x, void *arg)
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





gentype &negateoutery(gentype &res);
gentype &negateoutery(gentype &res)
{
    // Defaults

    if      ( res.isValSet()  ) { if ( ( res.dir_set().size() >= 1 ) && !res.dir_set().all()(0).isValNull()   ) { res.dir_set().ncall()("&",0).negate(); } }
    else if ( res.isValDict() ) { if ( res.dir_dict().iskeypresent("y") && !(res.dir_dict())("y").isValNull() ) { (res.dir_dict())("&","y").negate();    } }
    else                        { res.negate(); }

    return res;
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

    int isnull     = 0;
    int isnull_cgt = 0;

    if ( y.isValVector() )
    {
        for ( int iy = 0 ; ( !isnull && ( iy < ((const Vector<gentype> &) y).size() ) ) ; ++iy )
        {
            isnull = ((const Vector<gentype> &) y)(iy).isValNull() ? 1 : 0;
        }
    }

    else
    {
        isnull = y.isValNull();
    }

    for ( int iy = 0 ; ( !isnull_cgt && ( iy < ycgt.size() ) ) ; ++iy )
    {
        isnull_cgt = ycgt(iy).isValNull() ? 1 : 0;
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
        ycgt.resize(bopts.numcgt) = 0.0_gent; //toGentype(); - don't want null here!
        xobstype_cgt.resize(bopts.numcgt) = 0;
    }

    bool fullcgt = true;

    for ( int aa = 0 ; ( fullcgt && bopts.numcgt ) && ( aa < bopts.numcgt ) ; ++aa )
    {
        fullcgt = ( xobstype_cgt(aa) == 0 ) ? false : true;
    }

    bool locxhasnulls = false;
    bool isfullfid = true;

    int dimfid = bopts.getdimfid();

    for ( int aa = 0 ; aa < dim ; ++aa )
    {
        if ( aa < dim-dimfid )
        {
            locxhasnulls = ( xtoadd(aa).isValNull() ) ? true : false;
        }

        else
        {
            if ( xtoadd.n(aa-(dim-dimfid),1).isValNull() )
            {
                locxhasnulls = true;
                isfullfid = false;
            }

            else if ( ((int) (xtoadd.n(aa-(dim-dimfid),1)*((double) bopts.numfids))) != bopts.numfids )
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

        if ( xobstype                ) { ++Npreadd_mu; ++Npreadd_sigma; }
        if ( fullcgt && bopts.numcgt ) { ++Npreadd_cgt;                 }
        if ( fullobs                 ) { ++Npreadd;                     }
    }

    // Record if optimal solution (so far)

    if ( !locxhasnulls ) // ie is this a true observation (all x known, constraints observed (or no constraints))
    {
         if (    isfullfid                                     // full fidelity (or "real system") observation
              && ( ii >= 0 )
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

        double nuscaledummy = 1;

        readres(qres,qaddvar,qycgt,qxxreplace,qreplacexx,qaddexp,qstopflags,qxsidechan,qxobstype,qxobstype_cgt,fidcost,nuscaledummy);

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
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);                  }
        else                  { dest("&",j) = src.n(j-(dim-dimfid),1); }
    }
}

void SparsenearToSparsenonnear(SparseVector<double> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j);                  }
        else                  { dest("&",j) = (double) src.n(j-(dim-dimfid),1); }
    }
}

void SparsenearToSparsenonnear(SparseVector<gentype> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { if ( testisvnan(src(j))                  ) { dest("&",j) = toGentype(); } else { dest("&",j).force_double() = src(j);                  } }
        else                  { if ( testisvnan(src.n(j-(dim-dimfid),1)) ) { dest("&",j) = toGentype(); } else { dest("&",j).force_double() = src.n(j-(dim-dimfid),1); } }
    }
}

void SparsenearToSparsenonnear(SparseVector<double> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);                  }
        else                  { dest("&",j) = src.n(j-(dim-dimfid),1); }
    }
}


void SparsenearToNonsparse(Vector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    dest.resize(dim);

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);                  }
        else                  { dest("&",j) = src.n(j-(dim-dimfid),1); }
    }
}

void SparsenearToNonsparse(Vector<double> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    dest.resize(dim);

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j);                  }
        else                  { dest("&",j) = (double) src.n(j-(dim-dimfid),1); }
    }
}

void SparsenearToNonsparse(Vector<gentype> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    dest.resize(dim);

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { if ( testisvnan(src(j))                  ) { dest("&",j) = toGentype(); } else { dest("&",j).force_double() = src(j);                  } }
        else                  { if ( testisvnan(src.n(j-(dim-dimfid),1)) ) { dest("&",j) = toGentype(); } else { dest("&",j).force_double() = src.n(j-(dim-dimfid),1); } }
    }
}

void SparsenearToNonsparse(Vector<double> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    dest.resize(dim);

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j);                  }
        else                  { dest("&",j) = src.n(j-(dim-dimfid),1); }
    }
}


void SparsenonnearToSparsenear(SparseVector<gentype> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j)                  = src(j); }
        else                  { dest.n("&",j-(dim-dimfid),1) = src(j); }
    }
}

void SparsenonnearToSparsenear(SparseVector<double> &dest, const SparseVector<gentype> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j)                  = (double) src(j); }
        else                  { dest.n("&",j-(dim-dimfid),1) = (double) src(j); }
    }
}

void SparsenonnearToSparsenear(SparseVector<gentype> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { if ( testisvnan(src(j)) ) { dest("&",j)                  = toGentype(); } else { dest("&",j).force_double()                  = src(j); } }
        else                  { if ( testisvnan(src(j)) ) { dest.n("&",j-(dim-dimfid),1) = toGentype(); } else { dest.n("&",j-(dim-dimfid),1).force_double() = src(j); } }
    }
}

void SparsenonnearToSparsenear(SparseVector<double> &dest, const SparseVector<double> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j)                  = src(j); }
        else                  { dest.n("&",j-(dim-dimfid),1) = src(j); }
    }
}


void NonsparseToSparsenear(SparseVector<gentype> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j)                  = src(j); }
        else                  { dest.n("&",j-(dim-dimfid),1) = src(j); }
    }
}

void NonsparseToSparsenear(SparseVector<double> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j)                  = (double) src(j); }
        else                  { dest.n("&",j-(dim-dimfid),1) = (double) src(j); }
    }
}

void NonsparseToSparsenear(SparseVector<gentype> &dest, const Vector<double> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { if ( testisvnan(src(j)) ) { dest("&",j)                  = toGentype(); } else { dest("&",j).force_double()                  = src(j); } }
        else                  { if ( testisvnan(src(j)) ) { dest.n("&",j-(dim-dimfid),1) = toGentype(); } else { dest.n("&",j-(dim-dimfid),1).force_double() = src(j); } }
    }
}

void NonsparseToSparsenear(SparseVector<double> &dest, const Vector<double> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j)                  = src(j); }
        else                  { dest.n("&",j-(dim-dimfid),1) = src(j); }
    }
}


void NonsparseToSparsenonnear(SparseVector<gentype> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = src(j); }
        else                  { dest("&",j) = src(j); }
    }
}

void NonsparseToSparsenonnear(SparseVector<double> &dest, const Vector<gentype> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { dest("&",j) = (double) src(j); }
        else                  { dest("&",j) = (double) src(j); }
    }
}

void NonsparseToSparsenonnear(SparseVector<gentype> &dest, const Vector<double> &src, int dim, int dimfid)
{
    dest.zero();

    for ( int j = 0 ; j < dim ; ++j )
    {
        if ( j < dim-dimfid ) { if ( testisvnan(src(j)) ) { dest("&",j) = toGentype(); } else { dest("&",j).force_double() = src(j); } }
        else                  { if ( testisvnan(src(j)) ) { dest("&",j) = toGentype(); } else { dest("&",j).force_double() = src(j); } }
    }
}

void NonsparseToSparsenonnear(SparseVector<double> &dest, const Vector<double> &src, int dim, int dimfid)
{
    dest.zero();

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

#define CONT_TEST ( !stopnow    && \
                    !stopearly  && \
                    !killSwitch && \
                    !isopt      && \
                    !timeout    && \
                    ( !maxitcnt               || ( itcnt-skipcnt  < maxitcnt         ) ) && \
                    ( !isgridopt || iscontopt || ( gridind.size() > recBatchSize     ) ) && \
                    ( !usefidbudget           || ( fidtotcost     < bopts.fidbudget  ) )    )


int bayesOpt(int dim,
             Vector<double> &xres,
             gentype        &fres,
             const Vector<double> &qmin,
             const Vector<double> &qmax,
             void (*fn)(int n, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_cgt, bool onlyAdd, double fidcost, double &nuscale, double fidtotcost),
             void *fnarg,
             BayesOptions &bopts,
             svmvolatile int &killSwitch,
             Vector<double> &s_score)
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

    int dimfid  = bopts.getdimfid();
    int numfids = bopts.numfids;
    int fidmode = bopts.fidmode;

    double cgtVarScale = bopts.cgtVarScale;

    effdim -= dimfid; // multi-fidelity terms don't get included in effdim

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

    int recBatchSize = ( ( bopts.acq == 11 ) && (bopts.betafn).isValVector() ) ? (bopts.betafn).size() : 1;

    int numRecs = 0;
    int newRecs = 0;

    NiceAssert( recBatchSize > 0 );
    NiceAssert( ( recBatchSize == 1 ) || ( effdim+dimfid == dim ) );

    // =======================================================================
    // Local references for function approximation and sigma approximation.
    // sigmaml separate is useful when we have multi-rec and need
    // "hallucinated" samples that need to update the sigma and not the mu.
    // =======================================================================

//    int sigmuseparate = bopts.sigmuseparate;




















    // =======================================================================
    // Precalculate fidelity penalties
    //
    // There are dimfid fidelity axis, each with numfids+1 points along it
    // (including 0 in the set).
    // =======================================================================

    int fidpencostsize = ( bopts.numfids && dimfid ) ? ( (int) std::round(std::pow((double) (bopts.numfids),(double) dimfid)) ) : 0;

    Vector<double> fidpencost(fidpencostsize);
    Vector<double> fidpenvars(fidpencostsize);
    Vector<Vector<double>> fidpenvecs(fidpencostsize);

    if ( fidpencostsize )
    {
        Vector<int> fidind(dimfid);
        bool isdone = false;

        for ( fidind = 0 ; !isdone ; )
        {
            int fidindis = 0;

            SparseVector<SparseVector<gentype> > actfidel;
            Vector<double> fidpenvecloc(dimfid);

            for ( int jij = 0 ; jij < dimfid ; ++jij )
            {
                fidindis *= bopts.numfids;
                fidindis += fidind(jij);

                actfidel("&",0)("&",jij) = (((double) fidind(jij)+1)/((double) numfids));
                fidpenvecloc   ("&",jij) = (((double) fidind(jij)+1)/((double) numfids));
            }

            fidpencost("&",fidindis) = (double) bopts.fidpenalty(actfidel);
            fidpenvars("&",fidindis) = (double) bopts.fidvar(actfidel);
            fidpenvecs("&",fidindis) = fidpenvecloc;

            for ( int jij = 0 ; jij < dimfid ; ++jij )
            {
                ++fidind("&",jij);

                if ( fidind(jij) >= numfids )
                {
                    fidind("&",jij) = 0;

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
    }
















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

    bool isfullgrid   = false; // full grid with unknown constraints and no (BOCA) multifidelity if true (so don't actually need a GP model for the objective, just the constraints)
    bool gridHasNulls = false; // true if grid has undefined x's that may trigger inner-inner continuous optimizer

    if ( gridsource )
    {
        // Work out if this is a full grid with unknown constraints

        {
            isfullgrid = true;
            int allfullobs = 1;

            // full-grid not defined if multifidelity BOCA

            if ( dimfid && ( fidmode != 0 ) ) { isfullgrid = false; }

            for ( i = 0 ; i < (*gridsource).N() ; ++i )
            {
                gentype yyval = ((*gridsource).y())(i); // y value as defined in the model (not processed yet)

                xobstype = ((*gridsource).d())(i); // default value as specified in gridsource
                xobstype_cgt.resize(0); //bopts.numcgt) = 2; // default value is equality on constraints

                {
                    double xaddvar = 0;
                    Vector<gentype> xycgt;
                    SparseVector<gentype> xxxreplace;
                    int xxreplacexx = 0;
                    gentype xaddexp;
                    int xstopflags = 0;
                    Vector<gentype> xxsidechan;
                    int xxobstype = 2;
                    Vector<int> xxobstype_cgt;
                    double nuscaledummy = 1;

                    readres(yyval,xaddvar,xycgt,xxxreplace,xxreplacexx,xaddexp,xstopflags,xxsidechan,xxobstype,xxobstype_cgt,fidcost,nuscaledummy);

                    StrucAssert( xaddexp.isValNull() );

                    bool fullobs = process_obs(yyval,xycgt,xxobstype,xxobstype_cgt,bopts.numcgt);
                    const SparseVector<gentype> &xtoadd = xxreplacexx ? xxxreplace : (*gridsource).x(i);

//FIXME if d=2 and y!=none, and numcgt=0 or dcgt=2 and y!=none THEN fill in nulls with scatter of random points and train on that (this is a full observation)
                    if ( !xxobstype ) { isfullgrid = false; }
                    if ( !fullobs   ) { allfullobs = 0;     }

                    for ( int aa = 0 ; aa < dim ; ++aa )
                    {
                        if ( aa < dim-dimfid ) { if ( xtoadd(aa).isValNull()                  ) { gridHasNulls = true; } }
                        else                   { if ( xtoadd.n(aa-(dim-dimfid),1).isValNull() ) { gridHasNulls = true; } }
                    }
                }
            }

            if ( allfullobs ) { isfullgrid = false; }
        }

        // Pre-add vectors to mu and sigma approximators, then set d = 0 (so pre-add but not yet included in calculations)

        int ii = -3;
        int alti = 0;

        for ( i = 0 ; i < (*gridsource).N() ; ++i )
        {
            gentype yyval = (*gridsource).y()(i); // y value as defined in the model (not processed yet)
            SparseVector<gentype> xx((*gridsource).x(i)); // x value as defined in the model (not processed yet) - copy is deliberate!

            xobstype = (*gridsource).d()(i); // default value as specified in gridsource
            xobstype_cgt.resize(0); //bopts.numcgt) = 2; // default value is equality on constraints

            for ( int jj = 0 ; jj < ( ( dimfid && ( fidmode == 0 ) ) ? fidpenvecs.size() : 1 ) ; ++jj )
            {
                bool skipthis = false;

                if ( dimfid && ( fidmode == 0 ) )
                {
                    // fill-in fidelity variables

                    if ( jj ) { xx = (*gridsource).x(i); }

                    for ( int aa = dim-dimfid ; !skipthis && ( aa < dim ) ; ++aa )
                    {
                        if      ( xx.n(aa-(dim-dimfid),1).isValNull() ) { xx.n("&",aa-(dim-dimfid),1) = fidpenvecs(jj)(aa-(dim-dimfid)); }
                        else if ( ((double) xx.n(aa-(dim-dimfid),1)) != fidpenvecs(jj)(aa-(dim-dimfid)) ) { skipthis = true; }
                    }
                }

                if ( !skipthis )
                {
//FIXME if d=2 and y!=none, and numcgt=0 or dcgt=2 and y!=none THEN fill in nulls with scatter of random points and train on that (this is a full observation)
//FIXME also do for d=+-1, but these don't count as a full observation
                    double nuscaledummy = 1;

                    readres(yyval,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost,nuscaledummy);

                    StrucAssert( addexp.isValNull() ); // no multiple-observations hiding in single observations allowed here!
                    StrucAssert( !replacexx         ); // xx must be as given, no changes allowed here!

                    int dNpreadd_mu    = 0; // dummy
                    int dNpreadd_sigma = 0; // dummy
                    int dNpreadd_cgt   = 0; // dummy

                    bool fullobs = addDataToModel(ii,Npreadd,dNpreadd_mu,dNpreadd_sigma,dNpreadd_cgt,stopnow,bopts,dim,isfullgrid,true,fres,ires,
                                                  xx,addvar,yyval,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,
                                                  true,fidcost);

                    if ( !fullobs )
                    {
                        gridall.add(gridall.size());
                        gridall("&",gridall.size()-1) = alti;

                        gridind.add(gridind.size());
                        gridind("&",gridind.size()-1) = alti;

                        grid_xobstype.add(grid_xobstype.size());
                        grid_xobstype("&",grid_xobstype.size()-1) = xobstype;

//                        grid_xobstype_cgt.add(grid_xobstype_cgt.size());
//                        grid_xobstype_cgt("&",grid_xobstype_cgt.size()-1) = xobstype_cgt;

                        ++Ngrid;
                    }

                    ++alti;
                }
            }
        }
    }

    else if ( !gridsource && dimfid && ( fidmode == 0 ) )
    {
        isfullgrid   = false; // the x (non-fidelity) part is all nulls for all the (grid) of fidelities, and y/ycgt is unknown - hence not a full grid
        gridHasNulls = true;  // the x (non-fidelity) part is all nulls for all the (grid) of fidelities, and y/ycgt is unknown - hence the grid has nulls

        // Like above, but there is only one observation with all nulls

        int ii = -3;
        int alti = 0;
        {
            gentype yyval('N');
            SparseVector<gentype> xx;

            for ( int aa = 0 ; aa < dim ; ++aa )
            {
                if ( aa < dim-dimfid ) { xx("&",aa).force_null();                  }
                else                   { xx.n("&",aa-(dim-dimfid),1).force_null(); }
            }

            xobstype = 2;
            xobstype_cgt.resize(0);

            for ( int jj = 0 ; jj < fidpenvecs.size() ; ++jj )
            {
                for ( int aa = dim-dimfid ; aa < dim ; ++aa )
                {
                    xx.n("&",aa-(dim-dimfid),1) = fidpenvecs(jj)(aa-(dim-dimfid));
                }

                double nuscaledummy = 1;

                readres(yyval,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost,nuscaledummy);

                StrucAssert( addexp.isValNull() ); // no multiple-observations hiding in single observations allowed here!
                StrucAssert( !replacexx         ); // xx must be as given, no changes allowed here!

                int dNpreadd_mu    = 0; // dummy
                int dNpreadd_sigma = 0; // dummy
                int dNpreadd_cgt   = 0; // dummy

                bool fullobs = addDataToModel(ii,Npreadd,dNpreadd_mu,dNpreadd_sigma,dNpreadd_cgt,stopnow,bopts,dim,isfullgrid,true,fres,ires,
                                              xx,addvar,yyval,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,
                                              true,fidcost);

                if ( !fullobs )
                {
                    gridall.add(gridall.size());
                    gridall("&",gridall.size()-1) = alti;

                    gridind.add(gridind.size());
                    gridind("&",gridind.size()-1) = alti;

                    grid_xobstype.add(grid_xobstype.size());
                    grid_xobstype("&",grid_xobstype.size()-1) = xobstype;

//                    grid_xobstype_cgt.add(grid_xobstype_cgt.size());
//                    grid_xobstype_cgt("&",grid_xobstype_cgt.size()-1) = xobstype_cgt;

                    ++Ngrid;
                }

                ++alti;
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

    int isgridopt = Ngrid ? 1 : 0;                      // is this optimisation on a grid. Note isfullgrid is false if Ngrid > 0. We also assume that
                                                        // any grid observations at a fixed fidelity are by definition incomplete
    int iscontopt = ( gridHasNulls || !Ngrid ) ? 1 : 0; // is this continuous optimization (with or without a grid), including the fidelity variable
    int ismultfid = dimfid ? 1 : 0;                     // is multi-fidelity optimization

    NiceAssert( !isgridopt || !(bopts.isXconvertNonTrivial()) );














    // For the purely grid-based optimization we can get "stuck" due to poor choice of
    // hyperparameters. Find the min-distance on the grid and set this as a lower bound
    // on the lengthscale (simplistic case).

    if ( isgridopt && !iscontopt && bopts.model_lenscaleLB(0).isValNone() )
    {
        double mindist = 0;
        bool firstdist = true;

        for ( int jj = 0 ; jj < gridind.size()-1 ; ++jj )
        {
            for ( int kk = jj+1 ; kk < gridind.size() ; ++kk )
            {
                const SparseVector<gentype> &xjj = bopts.model_x(Nbasemu+gridind(jj));
                const SparseVector<gentype> &xkk = bopts.model_x(Nbasemu+gridind(kk));

                double ttmmpp = 0;
                double jjkkdist = sqrt(norm2(xjj)+norm2(xkk)-2*innerProductAssumeReal(ttmmpp,xjj,xkk));

                if ( ( ( jjkkdist > 0 ) && firstdist ) || ( ( jjkkdist > 0 ) && ( jjkkdist < mindist ) ) )
                {
                    mindist = jjkkdist;
                    firstdist = false;
                }
            }
        }

        if ( !firstdist )
        {
errstream() << "Setting min lengthscale from grid: " << (1.5*mindist) << "\n";
            bopts.model_getKernel_unsafe(0).setRealConstZeroLB(1.5*mindist);
            bopts.model_resetKernel(0);
        }
    }















    // =======================================================================
    // Actual optimiser starts here
    // =======================================================================

    int dummy = 0;

    Vector<double> xa(dim);
    Vector<SparseVector<double>  > xb (recBatchSize); // Has fidelity variables in nup(0), and augmented data
    Vector<SparseVector<gentype> > xxb(recBatchSize); // Has fidelity variables in nup(1), and no augmented data
    Vector<int> dresb(recBatchSize);
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

    bool usefidbudget = ismultfid && ( bopts.fidbudget > 0 ) && ( ( bopts.startpoints < 0 ) || ( bopts.totiters < 0 ) ); // multi-fidelity as per Kandasamy with budget and no over-rides
    int startpoints   = ( bopts.startpoints == -1 ) ? ( usefidbudget ? 0 : (effdim+1+dimfid) ) : bopts.startpoints; //( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints; - note use of effdim here!
    int &startseed    = bopts.startseed; // reference because we need to *persistently* update it!
    int &algseed      = bopts.algseed;   // reference because we need to *persistently* update it!
    double modD       = ( bopts.modD == -1 ) ? ( isgridopt ? Ngrid : 10 ) : bopts.modD; // 10 is arbitrary here!
    int Nmodel        = Nbasemu;    //bopts.model_N_mu();
    int Nsigma        = Nbasesigma; //bopts.model_N_sigma();
    double B          = 1; // set per iteration
    double mig        = 1; // set per iteration
    double rkhsnorm   = 1; // set per iteration
    double kappa0     = 1; // set per iteration
    double Bcgt          = 1; // set per iteration
    double migcgt        = 1; // set per iteration
    double rkhsnormcgt   = 1; // set per iteration
    double kappa0cgt     = 1; // set per iteration
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

            for ( i = Nstart ; ( ( ( i-skipcnt < Nstart+startpoints ) || !startpoints )
                                 && !killSwitch
                                 && ( !isgridopt || ( gridind.size() > recBatchSize ) )
                                 && ( !usefidbudget || startpoints || ( fidtotcost < (bopts.fidbudget)/10 ) ) ) ; ++i )
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
                    gridy      = (double) bopts.model_y(Nbasemu+gridi);
                    //xinf    = &(((*gridsource).xinfo(gridi)));
                }

                isfullfid = true;

                xxb("&",k).zero();

//errstream() << "phantomxyzxyz model_x = " << bopts.model_x(Nbasemu+gridi) << "\n";
                for ( j = 0 ; j < dim ; ++j )
                {
                    int qqq = 0;

                    if ( j < dim-dimfid )
                    {
                        bool randfillit = ( !isgridopt || bopts.model_x(Nbasemu+gridi)(j).isValNull() );

                        if ( randfillit ) { randufill(xxb("&",k)("&",j),xmin(j),xmax(j));        }
                        else              { xxb("&",k)("&",j) = bopts.model_x(Nbasemu+gridi)(j); }
                    }

                    else
                    {
                        bool randfillit = ( !isgridopt || bopts.model_x(Nbasemu+gridi).n(j-(dim-dimfid),1).isValNull() );

                        if ( randfillit ) { qqq = 1+(rand()%(bopts.numfids));
                                            xxb("&",k).n("&",j-(dim-dimfid),1).dir_double() = ((double) qqq)/((double) bopts.numfids); }
                        else              { xxb("&",k).n("&",j-(dim-dimfid),1) = bopts.model_x(Nbasemu+gridi).n(j-(dim-dimfid),1);
                                            qqq = (int) std::round(((double) (xxb(k).n(j-(dim-dimfid),1)))*(bopts.numfids)); }

                        if ( qqq != bopts.numfids ) { isfullfid = false; }
                    }
                }

                // ===========================================================
                // ===========================================================
                // Translate data for experiment
                // ===========================================================
                // ===========================================================

errstream() << "phantomxyzxyz xxb = " << xxb(k) << "\n";
                SparsenearToSparsenonnear(xb("&",k),xxb(k),dim,dimfid);
errstream() << "phantomxyzxyz xb = " << xb(k) << "\n";

                // ===========================================================
                // ===========================================================
                // Generate preliminary predictions for reference
                // ===========================================================
                // ===========================================================

                if ( isgridopt && ( isfullgrid || ( bopts.model_d(Nbasemu+gridi) == 2 ) ) )
                {
                    mupred("&",k) = bopts.model_y(Nbasemu+gridi);
                    sigmapred("&",k) = bopts.model_sigma(0);
                }

                else if ( isgridopt && !iscontopt ) { bopts.model_muvarTrainingVector(sigmapred("&",k),mupred("&",k),Nbasesigma+gridi,Nbasemu+gridi); }
                else                                { bopts.model_muvar(sigmapred("&",k),mupred("&",k),xxb(k),xinf);                                  }

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
                xb("&",k)("&",dim+14) = ( ( bopts.B <= 0 ) && !bopts.ismoo() ) ? bopts.model_RKHSnorm(0) : bopts.B; // Only makes sense for single-objective
                xb("&",k)("&",dim+15) = !bopts.ismoo() ? bopts.model_maxinfogain(0) : 0.0;                          // Only makes sense for single-objective
                xb("&",k)("&",dim+16) = fidtotcost;

                // ===========================================================
                // ===========================================================
                // Run experiment:
                // ===========================================================
                // ===========================================================

                gentype gridresis = toGentype();
                Vector<gentype> ycgtis(bopts.numcgt,toGentype());

                if ( isgridopt && ( isfullgrid || ( bopts.model_d(Nbasemu+gridi) == 2 ) ) )
                {
                    // Grid may have defined solutions already loaded.
                    gridresis = bopts.model_y(Nbasemu+gridi);
                    gridresis.negate();
                }

                for ( int iy = 0 ; iy < bopts.numcgt ; ++iy )
                {
                    if ( isgridopt && ( bopts.model_d_cgt(Nbasemu+gridi,iy) == 2 ) ) { ycgtis("&",iy) = bopts.model_y_cgt(Nbasemu+gridi,iy); }
                }

                double nuscaledummy;

                fidcost = -1;
                xobstype = 2;
                fnapproxout.force_int() = -1;
                (*fn)(dim,fnapproxout,&xb(k)(0),fnarg,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,gridresis,muapproxsize,ycgtis,false,fidcost,nuscaledummy,fidtotcost);
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
                        SparsenearToSparsenonnear(xb("&",k),xxb(k),dim,dimfid);
                        isfullfid = true;

                        for ( int jij = dimfid-1 ; isfullfid && ( jij >= 0 ) ; jij-- )
                        {
                            if ( (int) std::round(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids ) { isfullfid = false; }
                        }
                    }

                    varscale = 0;

                    // DESIGN DECISIONS: only the first observation counts in budget
                    if ( firstinset )
                    {
                        double locfidpenalty = 1;
                        double locvarscale   = 0;

                        if ( dimfid )
                        {
                            int fidindis = 0;

                            for ( int jij = 0 ; jij < dimfid ; ++jij )
                            {
                                fidindis *= bopts.numfids;
                                fidindis += ((int) std::round((((double) bopts.numfids)*((double) xxb(k).n(jij,1)))))-1;
                            }

                            NiceAssert( fidindis < fidpencost.size() );

                            locfidpenalty = fidpencost(fidindis);
                            locvarscale   = fidpenvars(fidindis);
                        }

                        fidtotcost += ( ( fidcost >= 0 ) ? fidcost : locfidpenalty );
                        varscale    = ( ( fidcost >= 0 ) ? 1.0     : locvarscale   );
                        firstinset = false;
//errstream() << "fidpenalty: " << bopts.fidpenalty << "\n";
//errstream() << "locfidpenalty: " << locfidpenalty << "\n";
//errstream() << "fidtotcost: " << fidtotcost << "\n";
//errstream() << "allocated budget: " << (bopts.fidbudget)/10 << "\n";
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
                            if ( xobstype && ( addvar != 0 ) ) { bopts.model_setsigmaweight_addvar(Nbasemu+gridi,Nsigma,addvar); }
                            if ( xobstype ) { bopts.model_setyd(Nbasemu+gridi,Nbasesigma+gridi,xobstype,fnapproxout,varscale); }
                            if ( !xobstype && !bopts.model_d(Nbasemu+gridi) ) { fullobs = false; }
                        }

                        else
                        {
                            ; //grid_xobstype("&",gridi) = xobstype; // so we can update later
                        }

                        if ( ycgt.size() )
                        {
                            for ( int iy = 0 ; iy < ycgt.size() ; ++iy )
                            {
                                if ( !xobstype_cgt(iy) && ( bopts.model_d_cgt(Nbasecgt+gridi,iy) ) )
                                {
                                    ycgt("&",iy) = bopts.model_y_cgt(Nbasecgt+gridi,iy);
                                    xobstype_cgt("&",iy) = bopts.model_d_cgt(Nbasecgt+gridi,iy);
                                }

                                if ( !xobstype_cgt(iy) && !bopts.model_d_cgt(Nbasecgt+gridi,iy) ) { fullobs = false; }
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

                        if ( addvar != 0 ) { bopts.model_setsigmaweight_addvar(Nmodel,Nsigma,addvar); }
                        if ( ycgt.size() ) { bopts.model_addTrainingVector_cgt(ycgt,xxb(k),xsidechan,xobstype_cgt,varscale); }
                    }

                    else
                    {
                        if ( !isfullgrid || ( gridi == -1 ) || replacexx )
                        {
                            bopts.model_addTrainingVector_musigma(((fnapproxout.isValNull()) ? 0.0_gent : fnapproxout),mupred(k),xxb(k),xsidechan,xobstype,varscale);

                            if ( addvar != 0 ) { bopts.model_setsigmaweight_addvar(Nmodel,Nsigma,addvar); }
                        }

                        if ( ycgt.size() ) { bopts.model_addTrainingVector_cgt(ycgt,xxb(k),xsidechan,xobstype_cgt,varscale); }
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

                    if ( xobstype || fullobs || ( isgridopt && iscontopt && isfullgrid ) ) { ++Nmodel; ++Nsigma; }
                    else { --i; } // We need Nstart non-null results here!

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

                        double nnuscaledummy = 1;

                        readres(fnapproxout,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost,nnuscaledummy);
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

        retVector<int> tmpvaa;

        Vector<int> rort(cntintvec(bopts.model_N_mu(),tmpvaa));

        int jj = 0;
        int firstone;
        int NC = bopts.model_N_mu()-bopts.model_NNCz_mu(0);

        //ysort.prealloc( NC+(( bopts.totiters == -1 ) ? 10*dim*( bopts.ismoo() ? bopts.moodim : 1 ) : bopts.totiters)+10);
        //s_score.prealloc(NC+(( bopts.totiters == -1 ) ? 10*dim*( bopts.ismoo() ? bopts.moodim : 1 ) : bopts.totiters)+10);
        ysort.prealloc( NC+(( bopts.totiters == -1 ) ? DEFITERS(dim) : bopts.totiters)+10);
        s_score.prealloc(NC+(( bopts.totiters == -1 ) ? DEFITERS(dim) : bopts.totiters)+10);

        while ( NC )
        {
            firstone = 1;

            for ( j = 0 ; j < rort.size() ; ++j )
            {
                if ( bopts.model_d(j) && ( firstone || ( (bopts.model_y(rort(j))) < (bopts.model_y(rort(jj))) ) ) )
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
    gentype predsigmax('R');
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
    Vector<SparseVector<gentype> > yyb_fullfid(recBatchSize);
    Vector<SparseVector<gentype> > yyb_minfid(recBatchSize);

    gentype locpen;

    double ztol    = bopts.ztol;
    double delta   = bopts.delta;
    double zeta    = bopts.zeta;
    double nu      = bopts.nu;
    double nuscale = 1;

    // Variables used in stable bayesian

    int firstevalinseq = 1;

    // =======================================================================
    // The following variable is used to pass variables through to fnfnapprox.
    // =======================================================================

    fninnerinnerArg fnarginner(bopts,
                               yyb("&",0),
                               yyb_fullfid("&",0),
                               yyb_minfid("&",0),
                               fnapproxout,
                               sigmax,
                               predsigmax,
                               fres,
                               ( itcntmethod != 3 ) ? altitcnt : itcnt,
                               dim,
                               ztol,
                               delta,
                               zeta,
                               nu,
                               nuscale,
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
                               rkhsnorm,
                               kappa0,
                               (bopts.R),
                               Bcgt,
                               migcgt,
                               rkhsnormcgt,
                               kappa0cgt,
                               (bopts.betafn),
                               (bopts.betafncgt),
                               (bopts.acq),
                               (bopts.acqcgt),
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
                               s_score,
                               firstevalinseq,
                               stabZeroPt,
                               (bopts.unscentUse),
                               (bopts.unscentK),
                               (bopts.unscentSqrtSigma),
                               (bopts.stabUseSig),
                               (bopts.stabThresh),
                               ires,
                               fidpencost,
                               fidpenvars,
                               fidpenvecs,
                               fidmode,
                               cgtVarScale);

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
    ////double maxitcnt = ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*effdim*( bopts.ismoo() ? bopts.moodim : 1 ) : bopts.totiters ); // note use of effdim, not dim ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*dim : bopts.totiters );
    //double maxitcnt = ( ( bopts.totiters == -2 ) ||  usefidbudget ) ? 0 : ( ( bopts.totiters == -1 ) ? DEFITERS(effdim) : bopts.totiters ); // note use of effdim, not dim ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*dim : bopts.totiters );
    double maxitcnt = ( ( bopts.totiters == -2 ) ||  usefidbudget ) ? ( ( bopts.totiters > 0 ) ? bopts.totiters : 0 ) : ( ( bopts.totiters == -1 ) ? DEFITERS(effdim) : bopts.totiters ); // note use of effdim, not dim ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*dim : bopts.totiters );
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
    if ( bopts.acq != 11 ) { outstream() << "***(acq " << bopts.acq << "," << bopts.acqcgt << ")\n"; }
    else                   { outstream() << "***(acq " << bopts.acq << "," << bopts.betafn << "," << bopts.betafncgt << ")\n"; }

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
        bool samplingUsed = false;

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
        (*fn)(-1,dummyyres,nullptr,fnarg,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,toGentype(),muapproxsize,dummyyres_cgt,false,fidcost,nuscale,fidtotcost);

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
            int dres = 0;

            anyindirect = isindirect || ( partindirect && k );

            // ===============================================================
            // ===============================================================
            // Multi-recommendation: these change depending on which
            // recommendation we are processing
            // ===============================================================
            // ===============================================================

            fnarginner._q_x         = &(yyb("&",k));
            fnarginner._q_x_fullfid = &(yyb_fullfid("&",k));
            fnarginner._q_x_minfid  = &(yyb_minfid("&",k));

            ismultitargrec      = 0;
            currintrinbatchsize = bopts.intrinbatch;

            int locacq    = bopts.acq;
            //int locacqcgt = bopts.acqcgt;

            if ( ( bopts.acq == 11 ) && (bopts.betafn).isValVector() )
            {
                if ( ((bopts.betafn)(k)).isValVector() )
                {
                    bool cgtisvec = bopts.numcgt && ( ( bopts.acqcgt == 11 ) && (bopts.betafncgt).isValVector() )
                                                 && ( ((bopts.betafncgt)(k)).isValVector() );

                    StrucAssert( !cgtisvec || ( bopts.betafn.size() == bopts.betafncgt.size() ) );

                    // =======================================================
                    // =======================================================
                    // Explicit multi-recommendation via multiple strategies
                    // =======================================================
                    // =======================================================

                    NiceAssert( ((bopts.betafn)(k)).size() >= 1  );
                    NiceAssert( ((bopts.betafn)(k)).size() <= 12 );

                    NiceAssert( ((bopts.betafncgt)(k)).size() >= 1  );
                    NiceAssert( ((bopts.betafncgt)(k)).size() <= 12 );

                    gentype locbetafn;
                    gentype locbetafncgt;

                    if (             ((bopts.betafn   )(k)).size() >= 3 ) { gentype tmpbetafn   ((bopts.betafn   )(k)(2)); locbetafn    = tmpbetafn;    }
                    if ( cgtisvec && ((bopts.betafncgt)(k)).size() >= 3 ) { gentype tmpbetafncgt((bopts.betafncgt)(k)(2)); locbetafncgt = tmpbetafncgt; }

                    // Note use of cast_ here (without finalisation) just in case any of these are functions.

                    int locacqk    = bopts.acq;
                    int locacqkcgt = bopts.acqcgt;

                                      locacqk    = ((bopts.betafn   )(k)(0)).cast_int(0);
                    if ( cgtisvec ) { locacqkcgt = ((bopts.betafncgt)(k)(0)).cast_int(0); }

                    fnarginner._q_locacq    = locacqk;
                    fnarginner._q_locacqcgt = locacqkcgt;

                    fnarginner._q_p               = ( ( ((bopts.betafn)(k)).size() >= 2  ) && !(((bopts.betafn)(k)(1 )).isValVector()) ) ?  (((bopts.betafn)(k)(1 )).cast_double(0)) :  (bopts.p);
                    fnarginner._q_betafn          = ( ( ((bopts.betafn)(k)).size() >= 3  ) && !(((bopts.betafn)(k)(2 )).isValVector()) ) ? &locbetafn                                : &(bopts.betafn);
                    fnarginner._q_modD            = ( ( ((bopts.betafn)(k)).size() >= 4  ) && !(((bopts.betafn)(k)(3 )).isValVector()) ) ?  (((bopts.betafn)(k)(3 )).cast_double(0)) :  bopts.modD;
                    fnarginner._q_delta           = ( ( ((bopts.betafn)(k)).size() >= 5  ) && !(((bopts.betafn)(k)(4 )).isValVector()) ) ?  (((bopts.betafn)(k)(4 )).cast_double(0)) :  delta;
                    fnarginner._q_zeta            = 0.01; //( ( ((bopts.betafn)(k)).size() >= 5  ) && !(((bopts.betafn)(k)(4 )).isValVector()) ) ?  (((bopts.betafn)(k)(4 )).cast_double(0)) :  delta;
                    fnarginner._q_defnu           = ( ( ((bopts.betafn)(k)).size() >= 6  ) && !(((bopts.betafn)(k)(5 )).isValVector()) ) ?  (((bopts.betafn)(k)(5 )).cast_double(0)) :  nu;
                    fnarginner._q_a               = ( ( ((bopts.betafn)(k)).size() >= 7  ) && !(((bopts.betafn)(k)(6 )).isValVector()) ) ?  (((bopts.betafn)(k)(6 )).cast_double(0)) :  (bopts.a);
                    fnarginner._q_b               = ( ( ((bopts.betafn)(k)).size() >= 8  ) && !(((bopts.betafn)(k)(7 )).isValVector()) ) ?  (((bopts.betafn)(k)(7 )).cast_double(0)) :  (bopts.b);
                    fnarginner._q_r               = ( ( ((bopts.betafn)(k)).size() >= 9  ) && !(((bopts.betafn)(k)(8 )).isValVector()) ) ?  (((bopts.betafn)(k)(8 )).cast_double(0)) :  (bopts.r);
                    fnarginner._q_thisbatchsize   = ( ( ((bopts.betafn)(k)).size() >= 10 ) && !(((bopts.betafn)(k)(9 )).isValVector()) ) ?  (((bopts.betafn)(k)(9 )).cast_int   (0)) :  (bopts.intrinbatch);
                    fnarginner._q_thisbatchmethod = ( ( ((bopts.betafn)(k)).size() >= 11 ) && !(((bopts.betafn)(k)(10)).isValVector()) ) ?  (((bopts.betafn)(k)(10)).cast_int   (0)) :  (bopts.intrinbatchmethod);
                    fnarginner._q_R               = ( ( ((bopts.betafn)(k)).size() >= 12 ) && !(((bopts.betafn)(k)(11)).isValVector()) ) ?  (((bopts.betafn)(k)(11)).cast_double(0)) :  (bopts.R);

                    if ( cgtisvec ) { fnarginner._q_betafncgt = ( ( ((bopts.betafncgt)(k)).size() >= 3  ) && !(((bopts.betafncgt)(k)(2 )).isValVector()) ) ? &locbetafncgt                                : &(bopts.betafncgt); }

                    if ( ( (bopts.betafn(k)).size() >= 10 ) && !(((bopts.betafn)(k)(9)).isValVector()) ) { currintrinbatchsize = (int) ((bopts.betafn)(k)(9)); }

                    locacq    = fnarginner._q_locacq;
                    //locacqcgt = fnarginner._q_locacqcgt;
                }

                else
                {
                    // =======================================================
                    // Multi-recommendation via multi-objective optimisation
                    // (beta function null)
                    // ...or...
                    // Single explicit strategy (beta function non-null)
                    // =======================================================

                    fnarginner._q_locacq          = (bopts.acq);
                    fnarginner._q_locacqcgt       = (bopts.acqcgt);
                    fnarginner._q_p               = (bopts.p);
                    fnarginner._q_betafn          = &(bopts.betafn);
                    fnarginner._q_modD            = bopts.modD;
                    fnarginner._q_delta           = delta;
                    fnarginner._q_zeta            = zeta;
                    fnarginner._q_defnu           = nu;
                    fnarginner._q_a               = (bopts.a);
                    fnarginner._q_b               = (bopts.b);
                    fnarginner._q_r               = (bopts.r);
                    fnarginner._q_thisbatchsize   = (bopts.intrinbatch);
                    fnarginner._q_thisbatchmethod = (bopts.intrinbatchmethod);
                    fnarginner._q_R               = (bopts.R);
                    fnarginner._q_betafncgt       = &(bopts.betafncgt);

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

                dres = 0;

                firstevalinseq = 1;

                B        = ( bopts.B <= 0 ) ? bopts.model_RKHSnorm(0) : bopts.B;
                mig      = bopts.model_maxinfogain(0);
                rkhsnorm = bopts.model_RKHSnorm(0);
                kappa0   = bopts.model_kappa0(0);

                if ( bopts.numcgt )
                {
                    Bcgt        = ( bopts.B <= 0 ) ? bopts.modelcgt_RKHSnorm(0) : bopts.B;
                    migcgt      = bopts.modelcgt_maxinfogain(0);
                    rkhsnormcgt = bopts.modelcgt_RKHSnorm(0);
                    kappa0cgt   = bopts.modelcgt_kappa0(0);
                }

                // Adjust bounds for multifidelity (with BOCA, recommendation is always done
                // for max fidelity, then the real fidelity is chosen afterwards

                if ( dimfid && ( fidmode != 0 ) )
                {
                    // variabls xa[n-1] is a fidelity variable, 1 is the max fidelity, so set it 1 and compress the range
                    // when choosing x as per Kandasamy.

                    for ( int jij = 0 ; jij < dimfid ; jij++ )
                    {
                        xmin("&",n-dimfid+jij) = 1;
                        xmax("&",n-dimfid+jij) = 1;

                        xa("&",n-dimfid+jij) = 1;
                    }
                }

                if ( ( locacq == 12 ) || ( locacq == 16 ) )
                {
                    double R         = bopts.R;
                    double ddelta    = bopts.delta;
                    double sampScale = ( locacq == 12 ) ? ( B + (R*sqrt(2*(mig+1+log(2/ddelta)))) ) : 1.0;

                    samplingUsed = true;

                    bopts.model_sample(xmin,xmax,sampScale*sampScale);
                }

                if ( ( bopts.cgtmethod == 4 ) || ( bopts.cgtmethod == 5 ) )
                {
                    double Rcgt         = bopts.R;
                    double ddeltacgt    = bopts.delta;
                    double sampScalecgt = ( bopts.cgtmethod == 5 ) ? ( Bcgt + (Rcgt*sqrt(2*(migcgt+1+log(2/ddeltacgt)))) ) : 1.0;

                    samplingUsed = true;

                    bopts.modelcgt_sample(xmin,xmax,sampScalecgt*sampScalecgt);
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

                    retVector<double> tmpvaa;
                    retVector<double> tmpvb;
                    retVector<double> tmpvc;

                    if ( locacq == 18 )
                    {
                        // Fill with nulls/nans to indicate "human expert to fill"

                        for ( int jij = 0 ; jij < n ; jij++ )
                        {
                            if      ( jij < n-dimfid ) { xa("&",jij) = valvnan(); }
                            else if ( fidmode != 0   ) { xa("&",jij) = 1;         }
                            else                       { xa("&",jij) = valvnan(); }
                        }
                    }

                    else if ( locacq == 21 )
                    {
                        // Random fill as per initial points

                        for ( int jij = 0 ; jij < n ; jij++ )
                        {
                            if      ( jij < n-dimfid ) { randufill(xa("&",jij),direcmin(jij),direcmax(jij));               }
                            else if ( fidmode != 0   ) { xa("&",jij) = 1;                                                  }
                            else                       { xa("&",jij) = ((double) (1+(rand()%numfids)))/((double) numfids); }
                        }
                    }

                    else
                    {
                        // Use DIRect to maximise the acquisition function

                        fnarginner.mode = 1; // turn on single beta, stale x calculation fast mode
                        dres = directOpt(n,
                                         xa("&",0,1,n-1,tmpvaa),
                                         dummyres,
                                         direcmin(0,1,n-1,tmpvb),
                                         direcmax(0,1,n-1,tmpvc),
                                         fnfnapprox,
                                         fnarginnerdr,
                                         bopts.goptssingleobj,
                                         killSwitch,
                                         ADDJITTER);
                        fnarginner.mode = 0; // turn off single beta, stale x calculation fast mode

                        errstream() << "#((" << dres << "," << dummyres << "))";
                    }

                    errstream() << "Unfiltered result x before fidelity selection: "; printoneline(errstream(),xa); errstream() << "\n";
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

                    retVector<double> tmpvaa;
                    retVector<double> tmpvb;
                    retVector<double> tmpvc;

                    if ( locacq == 18 )
                    {
                        // Fill with nulls/nans to indicate "human expert to fill"

                        for ( int jij = 0 ; jij < n ; jij++ )
                        {
                            if      ( jij < n-dimfid ) { xa("&",jij) = valvnan(); }
                            else if ( fidmode != 0   ) { xa("&",jij) = 1;         }
                            else                       { xa("&",jij) = valvnan(); }
                        }
                    }

                    else if ( locacq == 21 )
                    {
                        // Random fill as per initial points

                        itorem    = rand()%(gridind.size());
                        gridires  = gridind(gridref);

                        for ( int jij = 0 ; jij < n ; jij++ )
                        {
                            if      ( ( jij < n-dimfid ) && bopts.model_x(Nbasemu+gridires)(jij).isValNull() ) { randufill(xa("&",jij),direcmin(jij),direcmax(jij));               }
                            else if (   jij < n-dimfid                                                       ) { xa("&",jij) = (double) bopts.model_x(Nbasemu+gridires)(jij);      }
                            else if ( fidmode != 0                                                           ) { xa("&",jij) = 1;                                                  }
                            else                                                                               { xa("&",jij) = (double) bopts.model_x(Nbasemu+gridires)(jij);      }
                        }
                    }

                    else
                    {
                        // over-ride goptssingleobj with relevant components from *this (assuming non-virtual assignment operators)
                        static_cast<GlobalOptions &>(bopts.goptssingleobj) = static_cast<GlobalOptions &>(bopts);

                        fnarginner.mode = 1; // turn on single beta, stale x calculation fast mode
                        dres = dogridOpt(n,
                                         xa("&",0,1,n-1,tmpvaa),
                                         dummyres,
                                         itorem,
                                         gridires,
                                         direcmin(0,1,n-1,tmpvb),
                                         direcmax(0,1,n-1,tmpvc),
                                         fnfnapprox,
                                         fnarginnerdr,
                                         bopts,
                                         gridindtmp,
                                         killSwitch,
                                         bopts.goptssingleobj,
                                         xmtrtime,
                                         isfullgrid);
                        fnarginner.mode = 0; // turn off single beta, stale x calculation fast mode
                    }

                    gridi = gridires;
                    //gridind.remove(itorem); - do this later if we get a full observation!
                    gridref = itorem;
                    gridy = ( gridi >= 0 ) ? ((double) bopts.model_y(Nbasemu+gridi)) : valvnan(); //((*gridsource).y())(gridi);

                    if ( !iscontopt && ( dres != -200 ) )
                    {
                        gridindtmp.remove(itorem);
                    }

                    if ( dres != -200 )
                    {
                        gridivec.add(gridivec.size());     gridivec("&",gridivec.size()-1)     = gridi;
                        gridrefvec.add(gridrefvec.size()); gridrefvec("&",gridrefvec.size()-1) = gridref;
                    }

                    errstream() << "Unfiltered result x before fidelity selection: " << xa       << "\n";
                    errstream() << "Unfiltered result y before fidelity selection: " << dummyres << "\n";

                    printrec = true;
                }

                if ( ( locacq == 12 ) || ( locacq == 16 ) )
                {
                    bopts.model_unsample();
                }

                if ( ( bopts.cgtmethod == 4 ) || ( bopts.cgtmethod == 5 ) )
                {
                    bopts.modelcgt_unsample();
                }

                if ( dimfid && ( fidmode != 0 ) )
                {
                    for ( int jij = 0 ; jij < dimfid ; jij++ )
                    {
                        xmin("&",n-dimfid+jij) = 0;
                        xmax("&",n-dimfid+jij) = 1;
                    }
                }

                // ===============================================================
                // ===============================================================
                // Fidelity evaluation section
                // ===============================================================
                // ===============================================================

                // Put fidelity bounds back how they should be

                if ( ( dres != -200 ) && ( dimfid && ( fidmode != 0 ) ) && ( locacq != 18 ) )
                {
                    if ( fidmode == 21 )
                    {
                        int numfids = bopts.numfids;

                        NiceAssert( !anyindirect );

                        // variabls xa[n-1] is a fidelity variable, 1 is the max fidelity.

                        Vector<int> zindex(dimfid);
                        zindex = numfids; // default to highest fidelity if set empty

                        double lambdazmax = fidpencost(fidpencost.size()-1); // lambda is an increasing function, so the max penalty corresponds to the highest fidelity
                        double lambdazmin = lambdazmax;                      // remember we want to minimize

                        double fidkappa0 = bopts.model_kappa0(0); //bopts.fidkappa0;
                        double fidq = 1.0/((double) (n+2)); // FIXME: just assume an SE kernel here, but in general this depends on the kernel

                        justreturnbeta = 1;
                        double fidbeta = fnfnapprox(dim,&(xa("&",0)),fnarginnerdr); // Get beta value for optimiser
                        justreturnbeta = 0;

                        Vector<int> inumfid(dimfid);
                        bool isdone = false;

                        for ( int jij = 0 ; jij < dimfid ; jij++ ) { xa("&",n-dimfid+jij) = 1.0/((double) numfids); }

                        SparseVector<double> fidxglob;
                        NonsparseToSparsenear(fidxglob,xa,n,dimfid);

                        //errstream() << "fidelity fidkappa0 = " << fidkappa0 << "\n";
                        for ( inumfid = 1 ; !isdone ; )
                        {
                            // Work out fidelity variable {1/bopts.numfids,2/bopts.numfids,...,1}

                            //errstream() << "phantomxyz fidelity fidbeta = " << fidbeta << "\n";
                            //errstream() << "phantomxyz fidelity fidq = " << fidq << "\n";
                            //errstream() << "phantomxyz fidelity fidc = " << fidc << "\n";
                            //errstream() << "phantomxyz fidelity fidpenalty = " << bopts.fidpenalty << "\n";
                            for ( int jij = 0 ; jij < dimfid ; jij++ ) { xa("&",n-dimfid+jij) = ((double) inumfid(jij))/((double) numfids); }

                            SparseVector<double> fidxloc;
                            NonsparseToSparsenear(fidxloc,xa,n,dimfid);

                            // Work out penalty for this z

                            double lambdaz = 0;

                            {
                                int fidindis = 0;

                                for ( int jij = 0 ; jij < dimfid ; ++jij )
                                {
                                    fidindis *= bopts.numfids;
                                    fidindis += ((int) std::round(((((double) bopts.numfids)*((double) xa(n-dimfid+jij))))))-1;
                                }

                                lambdaz = fidpencost(fidindis);
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

                            bopts.inf_var(resvar,xxfidxloc);

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
                                if ( inumfid(jij) > numfids ) { inumfid("&",jij) = 1; if ( jij == dimfid-1 ) { isdone = true; } }
                                else { break; }
                            }
                        }

                        if ( bopts.fidover == 2 )
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
                            xa("&",n-dimfid+jij) = ( zindex(jij) == -1 ) ? valvnan() : (((double) zindex(jij))/((double) numfids));
                        }
                    }

                    else if ( ( fidmode >= 1 ) && ( fidmode <= 19 ) )
                    {
                        // Loop through all fidelities, work out which one gives the best information gain per unit budget

                        Vector<double> xalowfid(xa);
                        Vector<double> xautopia(xa);

                        for ( int jij = 0 ; jij < dimfid ; jij++ )
                        {
                            xautopia("&",n-dimfid+jij) = 1;
                        }

                        SparseVector<gentype> tmpxautopia;
                        NonsparseToSparsenear(tmpxautopia,xautopia,dim,dimfid);

                        double fidcst_max = fidpencost(fidpencostsize-1);
                        double target_max = 0;
                        gentype predsigmay,sigmay,muy;

                        Vector<gentype> predsigmay_cgt(bopts.numcgt);
                        Vector<gentype> sigmay_cgt(bopts.numcgt);
                        Vector<gentype> muy_cgt(bopts.numcgt);

                        for ( int kk = 0 ; kk < fidpencostsize ; kk++ )
                        {
                            for ( int jij = 0 ; jij < dimfid ; jij++ )
                            {
                                xalowfid("&",n-dimfid+jij) = fidpenvecs(kk)(jij);
                            }

                            SparseVector<gentype> tmpxalowfid;
                            NonsparseToSparsenear(tmpxalowfid,xalowfid,dim,dimfid);

//errstream() << "fidelity " << kk << ": sigmay = " << sigmay << "\n";
//errstream() << "fidelity " << kk << ": predsigmay = " << predsigmay << "\n";
//errstream() << "fidelity " << kk << ": sigma = " << bopts.model_sigma() << "\n";
                            double fidcst = fidpencost(kk);

                            double cost_rat  = fidcst_max/fidcst;
                            double cost_diff = fidcst_max-fidcst;

                            double fidtrm = std::log2(1+(fidcst_max/fidcst));

                            // sigmay = prior variance at xautopia
                            // predsigmay = posterior variance at xautopia if we make a measurement at xalowfid

                            bopts.model_predmuvar(predsigmay,sigmay,muy,tmpxautopia,tmpxalowfid);

                            double sigma_rat  = ((double) sigmay)/((double) predsigmay);
                            double sigma_diff = ((double) sigmay)-((double) predsigmay);

                            double fidmig = std::log2(1+((((double) sigmay)+bopts.model_sigma(0))/(((double) predsigmay)+bopts.model_sigma(0))));

                            if ( bopts.numcgt )
                            {
                                bopts.model_predmuvar_cgt(predsigmay_cgt,sigmay_cgt,muy_cgt,tmpxautopia,tmpxalowfid);

                                for ( int icgt = 0 ; icgt < bopts.numcgt ; ++icgt )
                                {
                                    sigma_diff += ((double) sigmay_cgt(icgt))-((double) predsigmay_cgt(icgt));

                                    fidmig += std::log2(1+((((double) sigmay_cgt(icgt))+bopts.model_sigma_cgt(icgt))/(((double) predsigmay_cgt(icgt))+bopts.model_sigma_cgt(icgt))));
                                }

                                sigma_rat = std::pow(2.0,fidmig)-1; // working back from the definitions
                            }

                            double target = 0;

                            if      ( fidmode == 1  ) { target = cost_rat  * sigma_rat;  }
                            else if ( fidmode == 2  ) { target = cost_diff * sigma_rat;  }
                            else if ( fidmode == 3  ) { target = fidtrm    * sigma_rat;  }
                            else if ( fidmode == 4  ) { target = cost_rat  * sigma_diff; } // ***
                            else if ( fidmode == 5  ) { target = cost_diff * sigma_diff; }
                            else if ( fidmode == 6  ) { target = fidtrm    * sigma_diff; }
                            else if ( fidmode == 7  ) { target = cost_rat  * fidmig;     }
                            else if ( fidmode == 8  ) { target = cost_diff * fidmig;     } // ***
                            else if ( fidmode == 9  ) { target = fidtrm    * fidmig;     } // ***
                            else if ( fidmode == 11 ) { target = cost_rat  + sigma_rat;  }
                            else if ( fidmode == 12 ) { target = cost_diff + sigma_rat;  }
                            else if ( fidmode == 13 ) { target = fidtrm    + sigma_rat;  }
                            else if ( fidmode == 14 ) { target = cost_rat  + sigma_diff; }
                            else if ( fidmode == 15 ) { target = cost_diff + sigma_diff; }
                            else if ( fidmode == 16 ) { target = fidtrm    + sigma_diff; }
                            else if ( fidmode == 17 ) { target = cost_rat  + fidmig;     }
                            else if ( fidmode == 18 ) { target = cost_diff + fidmig;     }
                            else if ( fidmode == 19 ) { target = fidtrm    + fidmig;     }

errstream() << "migpercost " << kk << ": target = (" << fidcst_max << "," << fidcst << " : " << sigmay << "," << predsigmay << ")";

                            // - fidmig will be >= 1 as measurements will never increase the variance, so sigmay >= predsigmay
                            // - fidmig will be 1 if the lower-fidelity measurement doesn't change the variance of the utopia point.
                            // - fidmig will be maximised as predsigmay -> 0 (that is, the posterior predicted variance at the utopia point goes to zero)
                            // - fidmig is relative to the current variance, which may be a problem
                            // - the term fidcst_max/fidcst = 1 if this is the max cost (utopia) measure, >1 otherwise
                            // - it measures the savings for using the lower-fidelity surrogate

                            if ( !kk || ( target > target_max ) )
                            {
                                target_max = target;
                                xa = xalowfid;
errstream() << "**";
                            }
errstream() << "\n";
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

                                                 for ( i = 0 ; i < n ; ++i                ) { tempx("&",i) = xa(i);        } // gentype sparsevector
                    if ( xappend.size() && k ) { for ( i = n ; i < n+xappend.size() ; ++i ) { tempx("&",i) = xappend(i-n); } }

                    (*direcpre).gg(xytemp,tempx);
                    const Vector<gentype> &ghgh = (const Vector<gentype> &) xytemp;

                    for ( i = 0 ; i < dim*currintrinbatchsize ; ++i ) { xa("&",i) = (double) ghgh(i); }
                }

                if ( partindirect && !k )
                {
                    NiceAssert( !anyindirect );
                    // update xappend here

                    for ( i = 0 ; i < n ; ++i ) { xappend("&",i) = xa(i); }
                }

                //errstream() << dres << "...";

                newRecs = ( dres == -200 ) ? 0 : currintrinbatchsize;

                if ( numRecs+newRecs >= xb.size() )
                {
                    xb.resize(numRecs+newRecs);
                    xxb.resize(numRecs+newRecs);
                    dresb.resize(numRecs+newRecs);
                }

                retVector<double> tmpvaaa;

                for ( jji = 0 ; jji < newRecs ; ++jji )
                {
                    xb("&",numRecs+jji).zero();
                    xb("&",numRecs+jji) = xa((jji*dim),1,(jji*dim)+dim-1,tmpvaaa);

                    if ( dres != -200 )
                    {
                        altitcnt += itcntmethod ? 1 : 0;
                    }

                    dresb("&",numRecs+jji) = dres;
                }
            }

            else
            {
                StrucAssert( !dimfid )

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

                locbopts.moodim            = recBatchSize;
                locbopts.impmeasu          = &(static_cast<IMP_Generic &>(locimpmeasu));
                locbopts.acq               = 1;
                locbopts.acqcgt            = 22;
                locbopts.intrinbatch       = 1;
                locbopts.intrinbatchmethod = 0;
                locbopts.startpoints       = bopts.startpointsmultiobj;
                locbopts.totiters          = bopts.totitersmultiobj;

                locbopts.goptssingleobj = bopts.goptsmultiobj;

                Vector<gentype> dummyxres;
                gentype dummyfres;
                Vector<gentype> dummycres;
                gentype dummyFres;
                gentype dummymres;
                gentype dummysres;
                int dummyires;
                int dummymInd;
                Vector<double> dummyhypervol;

                Vector<Vector<gentype> > locallxres;
                Vector<gentype> locallfres;
                Vector<Vector<gentype> > locallcres;
                Vector<gentype> locallFres;
                Vector<gentype> locallmres;
                Vector<gentype> locallsres;
                Vector<double> locs_score;
                Vector<int> locis_feas;
                Vector<int> locparind;

                // ===========================================================
                // Call multi-objective bayesian optimiser
                // ===========================================================

                void *locfnarg[1];

                locfnarg[0] = (void *) &bopts;

                bayesOpt(dim,
                         dummyxres,
                         dummyfres,
                         dummycres,
                         dummyFres,
                         dummymres,
                         dummysres,
                         dummyires,
                         dummymInd,
                         locallxres,
                         locallfres,
                         locallcres,
                         locallFres,
                         locallmres,
                         locallsres,
                         locs_score,
                         locis_feas,
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

                newRecs = bopts.analyse(locallmres,locis_feas,dummyhypervol,locparind,1);

                NiceAssert( newRecs );

                // ===========================================================
                // Grow xb and xxb.  We grow enough to fit what has been + 
                // this + one recommendation for each future batch
                // ===========================================================

                if ( numRecs+newRecs >= xb.size() )
                {
                    xb.resize(numRecs+newRecs);
                    xxb.resize(numRecs+newRecs);
                    dresb.resize(numRecs+newRecs);
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

                    if ( dres != -200 )
                    {
                        altitcnt += itcntmethod ? 1 : 0;
                    }

                    dresb("&",numRecs+jji) = dres;
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

                SparsenonnearToSparsenear(xxb("&",numRecs),xb(numRecs),dim,dimfid);

                isfullfid = true;

                if ( bopts.numfids )
                {
                    // Retrieve fidelity

                    for ( int jij = dimfid-1 ; jij >= 0 ; jij-- )
                    {
                        if ( (int) std::round(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids )
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

                if ( bopts.sigmuseparate && ( dres != -200 ) )
                {
                    if ( isgridopt && isfullfid && ( isfullgrid || ( bopts.model_d(Nbasemu+gridi) == 2 ) ) )
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

                if ( isgridopt && ( isfullgrid || ( bopts.model_d(Nbasemu+gridi) == 2 ) ) )
                {
                    mupred("&",numRecs)    = bopts.model_y(Nbasemu+gridi);
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
                xb("&",numRecs)("&",dim+14) = ( ( bopts.B <= 0 ) && !bopts.ismoo() ) ? bopts.model_RKHSnorm(0) : bopts.B;
                xb("&",numRecs)("&",dim+15) = !bopts.ismoo() ? bopts.model_maxinfogain(0) : 0.0;
                xb("&",numRecs)("&",dim+16) = fidtotcost;

                ++numRecs;
                --newRecs;
            }
        }

outstream() << numRecs << ": ";
        //int sigmamod = 0;
        bool doeval = true;
        bool haveAddedData = false;

        for ( k = 0 ; k < numRecs ; ++k )
        {
          if ( dresb(k) != -200 ) // Don't evaluate recommendations that don't actually recommend anything
          {
            //errstream() << "e";

            // ===============================================================
            // ===============================================================
            // Get ready to run experiment
            // ===============================================================
            // ===============================================================

            if ( isgridopt ) { gridi = gridivec(k); }
            else             { gridi = -1;          }

            doeval = true;
            int dval = 2;

            // ===============================================================
            // ===============================================================
            // Process User input if required
            // ===============================================================
            // ===============================================================

//REMOVED: YOU CAN DO THIS ON THE USER SIDE WITH COMPLEX RETURN TYPES

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

                for ( int jij = 0 ; jij < dimfid ; jij++ )
                {
                    if ( (int) std::round(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids )
                    {
                        isfullfid = false;
                    }
                }
            }

            if ( doeval )
            {
                gentype gridresis = toGentype();
                Vector<gentype> ycgtis(bopts.numcgt,toGentype());

                if ( isgridopt && ( isfullgrid || ( bopts.model_d(Nbasemu+gridi) == 2 ) ) )
                {
                    // Grid may have defined solutions already loaded.

                    gridresis = bopts.model_y(Nbasemu+gridi);
                    gridresis.negate();
                }

                for ( int iy = 0 ; iy < bopts.numcgt ; ++iy )
                {
                    if ( isgridopt && !iscontopt && ( bopts.model_d_cgt(Nbasemu+gridi,iy) == 2 ) )
                    {
                        ycgtis("&",iy) = bopts.model_y_cgt(Nbasemu+gridi,iy);
                    }
                }

                fidcost = -1;
                xobstype = 2;
                fnapproxout.force_int() = static_cast<int>(itcnt+1);
                (*fn)(dim,fnapproxout,&(xb(k)(0)),fnarg,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,gridresis,muapproxsize,ycgtis,false,fidcost,nuscale,fidtotcost);
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

                readres(fnapproxout,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost,nuscale);
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
                haveAddedData = true;

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

                    isfullfid = true;

                    SparsenearToSparsenonnear(xb("&",k),xxb(k),dim,dimfid);

                    isfullfid = true;

                    if ( numfids )
                    {
                        // Retrieve fidelity

                        for ( int jij = dimfid-1 ; jij >= 0 ; jij-- )
                        {
                            if ( (int) std::round(((double) (xxb(k).n(jij,1)))*(bopts.numfids)) != bopts.numfids )
                            {
                                isfullfid = false;
                            }
                        }
                    }
                }

                // Fidelity stuff goes here (we can't do it earlier in case the user over-rides the fidelity

                {
                    int numfids = bopts.numfids;

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
                            zindex("&",jij) = (int) std::round(((double) (xxb(k).n(jij,1)))*numfids);
                        }

                        if ( fidmode == 21 )
                        {
                            // Heuristic from Kandasamy supplementary C.1

                            if ( zindex == numfids ) { ++fidmaxcnt; }

                            // NB: really don't want to do this on the first iteration!
                            //if ( !((itcnt+1)%10) )
                            if ( !((itcnt+1)%20) )
                            {
                                //if ( fidmaxcnt > 7 ) { fidc /= sqrt(2.0); }
                                //if ( fidmaxcnt < 3 ) { fidc *= sqrt(2.0); }

                                if ( fidmaxcnt > 15 ) { fidc /= 2; }
                                if ( fidmaxcnt < 5  ) { fidc *= 2; }

                                fidc = ( fidc < 0.1 ) ? 0.1 : fidc;
                                fidc = ( fidc > 20  ) ? 20  : fidc;

                                fidmaxcnt = 0;
                                errstream() << "Fidelity: fidc = " << fidc << "\n";
                            }
                        }
                    }
                }

                varscale = 0;

                // DESIGN DECISIONS: only the first observation counts in budget
                if ( firstinset )
                {
//                    SparseVector<SparseVector<gentype> > actfidel;
//
//                    for ( int jij = 0 ; jij < dimfid ; jij++ )
//                    {
//                        actfidel("&",0)("&",jij) = xxb(k).n(jij,1);
//                    }
//
//                    double locfidpenalty = (double) bopts.fidpenalty(actfidel);
//                    double locvarscale   = (double) bopts.fidvar(actfidel);

                    double locfidpenalty = 1;
                    double locvarscale   = 0;

                    if ( dimfid )
                    {
                        int fidindis = 0;

                        for ( int jij = 0 ; jij < dimfid ; ++jij )
                        {
                            fidindis *= bopts.numfids;
                            fidindis += ((int) std::round(((((double) bopts.numfids)*((double) xxb(k).n(jij,1))))))-1;
                        }

                        locfidpenalty = fidpencost(fidindis);
                        locvarscale   = fidpenvars(fidindis);
                    }

                    fidtotcost += ( ( fidcost >= 0 ) ? fidcost : locfidpenalty );
                    varscale    = ( ( fidcost >= 0 ) ? fidcost : locvarscale   );
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

                        else if ( !bopts.model_d(Nbasemu+gridi) )
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
                                if ( bopts.model_d_cgt(Nbasecgt+gridi,iy) )
                                {
                                    ycgt("&",iy) = bopts.model_y_cgt(Nbasecgt+gridi,iy);
                                    xobstype_cgt("&",iy) = bopts.model_d_cgt(Nbasecgt+gridi,iy);
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
                        if ( (bopts.model_y(addpointpos)) < (bopts.model_y(ysort(jj))) )
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

                    double nuscaledummy = 1;

                    readres(fnapproxout,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost,nuscaledummy);
                    negateoutery(fnapproxout);
                }
            }

            altitcnt += itcntmethod ? 0 : 1;
            ++itcnt;
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

        if ( !samplingUsed && !haveAddedData )
        {
            isopt = 1;
        }


        // ===================================================================
        // ===================================================================
        // Iterate and go again
        // ===================================================================
        // ===================================================================

//++itcnt;
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

//    NiceAssert( ires >= 0 );

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

    s_score.resize(muapproxsize);

    if ( isstable )
    {
        s_score = 0.0;

//NB: we *absolutely do not* want to use ysort ordering here!
        retVector<int> tmpvaaa;

        calcsscore(s_score,bopts,cntintvec(s_score.size(),tmpvaaa),stabp,stabpnrm,stabrot,stabmu,stabB);
    }

    else
    {
        s_score = 1.0;
    }

    // =======================================================================
    // Record minimum
    // =======================================================================

    if ( ires == -1 )
    {
        fres.makeNull();
        xres.resize(0);
    }

    else
    {
        bopts.model_xcopy(xres,ires);
    }
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
               Vector<gentype> &_allFres,
               Vector<gentype> &_allmres,
               Vector<int> &_is_feas,
               gentype &_fres,
               Vector<gentype> &_cres,
               gentype &_Fres,
               gentype &_mres,
               gentype &_sres,
               Vector<gentype> &_xres,
               Vector<gentype> &_allsres,
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
                                   allFres(_allFres),
                                   allmres(_allmres),
                                   is_feas(_is_feas),
                                   fres(_fres),
                                   cres(_cres),
                                   Fres(_Fres),
                                   mres(_mres),
                                   sres(_sres),
                                   xres(_xres),
                                   allsres(_allsres),
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
        allFres.prealloc(nres);
        allmres.prealloc(nres);
        is_feas.prealloc(nres);
        allsres.prealloc(nres);

        xa.resize(dim);

        return;
    }

    fninnerArg(const fninnerArg &src) : fn(src.fn),
                                   arginner(src.arginner),
                                   ires(src.ires),
                                   allxres(src.allxres),
                                   allfres(src.allfres),
                                   allcres(src.allcres),
                                   allFres(src.allFres),
                                   allmres(src.allmres),
                                   is_feas(src.is_feas),
                                   fres(src.fres),
                                   cres(src.cres),
                                   Fres(src.Fres),
                                   mres(src.mres),
                                   sres(src.sres),
                                   xres(src.xres),
                                   allsres(src.allsres),
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
    Vector<gentype> &allFres;
    Vector<gentype> &allmres;
    Vector<int> &is_feas;
    gentype &fres;
    Vector<gentype> &cres;
    gentype &Fres;
    gentype &mres;
    gentype &sres;
    Vector<gentype> &xres;
    Vector<gentype> &allsres;
    volatile int &killSwitch;
    double &hardmin;
    double &hardmax;
    double &obsnoise;
    int dimfid;
    int numfids;
    Vector<double> &scnoise;
    int numcgt;

    void operator()(int dim, gentype &res, const double *x, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_ycgt, bool onlyAdd, double fidcost, double &nuscale, double fidtotcost)
    {
        // ===========================================================================
        // Inner loop evaluation function.  This is used as a buffer between the
        // actual Bayesian optimiser (above) and the outside-visible optimiser (below)
        // and saves things like timing, beta etc.
        // ===========================================================================

        if ( dim == -1 )
        {
            // This is just to trigger the intermediate command

//FIXME callback
#ifdef PYLOCAL
            gentype dummyyres;
            Dict<gentype,dictkey> & dummyaltres = dummyyres.force_dict();
            dummyaltres.zero();
            dummyaltres("[]","x")       = xres;
            dummyaltres("[]","f")       = -fres;
            dummyaltres("[]","y")       = fres;
            dummyaltres("[]","i")       = ires;
            dummyaltres("[]","allx")    = allxres;
            dummyaltres("[]","allf")    = -allfres;
            dummyaltres("[]","ally")    = allfres;
            dummyaltres("[]","allgt")   = allcres;
            dummyaltres("[]","allcost") = allFres;
            dummyaltres("[]","allfeas") = is_feas;

            (*fn)(dummyyres,dummyxarg,arginner);
#endif
#ifndef PYLOCAL
            (*fn)(res,dummyxarg,arginner);
#endif

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

        readres(res,addvar,ycgt,xxreplace,replacexx,addexp,stopflags,xsidechan,xobstype,xobstype_cgt,fidcost,nuscale);

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
                            if ( (int) std::round(((double) (xxb.n(jij,1)))*numfids) != numfids )
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

                    int x_is_feas = 0;

                    if (    isfullfid
                         && ( altxobstype == 2 )
                         && ( xobstype_cgt >= 1 )
                         && ( altycgt >= 0.0_gent )
                         && ( xxb.indsize() == xxb.nindsize() ) )
                    {
                        x_is_feas = 1;

                        if ( ( ires == -1 ) || ( altres > fres ) )
                        {
                            ires = allfres.size();
                            fres = altres;
                            cres = altycgt;
                            Fres = fidtotcost;
                            mres = altres;
                            //sres = ;
                            xres = xa;
                        }
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
                        allFres.append(allFres.size(),toGentype(fidtotcost));
                        allmres.append(allmres.size(),altres);
                        is_feas.append(is_feas.size(),x_is_feas);
                        allsres.append(allsres.size());

                        double dstandev = x[dim+6];
                        double softmax  = x[dim+7];

                        double ucbdist = softmax - ( altres.isCastableToReal() ? ( (double) altres ) : 0.0 );
                        double sigbnd  = ( ucbdist < dstandev ) ? ucbdist : dstandev;

                        allsres("&",allsres.size()-1).force_vector().resize(20);

                        allsres("&",allsres.size()-1)("&",0)  = TIMEABSSEC(starttime);
                        allsres("&",allsres.size()-1)("&",1)  = TIMEABSSEC(endtime);
                        allsres("&",allsres.size()-1)("&",2)  = x[dim];    // numRecs
                        allsres("&",allsres.size()-1)("&",3)  = x[dim+1];  // beta
                        allsres("&",allsres.size()-1)("&",4)  = x[dim+2];  // mu
                        allsres("&",allsres.size()-1)("&",5)  = x[dim+3];  // sigma
                        allsres("&",allsres.size()-1)("&",6)  = x[dim+4];  // UCB
                        allsres("&",allsres.size()-1)("&",7)  = x[dim+5];  // LCB
                        allsres("&",allsres.size()-1)("&",8)  = dstandev;  // DVAR
                        allsres("&",allsres.size()-1)("&",9)  = ucbdist;   // UCBDIST
                        allsres("&",allsres.size()-1)("&",10) = sigbnd;    // SIGBND
                        allsres("&",allsres.size()-1)("&",11) = x[dim+9];  // DIRect runtime
                        allsres("&",allsres.size()-1)("&",12) = x[dim+10]; // mu model training time
                        allsres("&",allsres.size()-1)("&",13) = x[dim+11]; // sigma model training time
                        allsres("&",allsres.size()-1)("&",14) = TIMEDIFFSEC(endtime,starttime);
                        allsres("&",allsres.size()-1)("&",15) = x[dim+12]; // grid index (-1 if none)
                        allsres("&",allsres.size()-1)("&",16) = x[dim+13]; // known grid evaluation (0.0 if none)
                        allsres("&",allsres.size()-1)("&",17) = x[dim+14]; // B
                        allsres("&",allsres.size()-1)("&",18) = x[dim+15]; // max info gain
                        allsres("&",allsres.size()-1)("&",19) = x[dim+16]; // fidelity cost
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

                        readres(altres,altaddvar,altycgt,altxxreplace,altreplacexx,altaddexp,altstopflags,altxsidechan,altxobstype,altxobstype_cgt,altfidcost,nuscale);
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

void fninner(int dim, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_cgt, bool onlyAdd, double fidcost, double &nuscale, double fidtotcost);
void fninner(int dim, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xxreplace, int &replacexx, gentype &addexp, int &stopflags, const gentype &gridres, int &muapproxsize, const Vector<gentype> &gridres_cgt, bool onlyAdd, double fidcost, double &nuscale, double fidtotcost)
{
    (*((fninnerArg *) arg))(dim,res,x,addvar,xsidechan,xobstype,xobstype_cgt,ycgt,xxreplace,replacexx,addexp,stopflags,gridres,muapproxsize,gridres_cgt,onlyAdd,fidcost,nuscale,fidtotcost);
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
              BayesOptions &bopts,
              svmvolatile int &killSwitch)
{
    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );

    (void) mInd;

    double hardmin = bopts.hardmin;
    double hardmax = bopts.hardmax;

    allxres.resize(0);
    allfres.resize(0);
    allcres.resize(0);
    allFres.resize(0);
    allmres.resize(0);
    allsres.resize(0);
    s_score.resize(0);
    is_feas.resize(0);

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
//                       ( ( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints ) + ( ( bopts.totiters == -1 ) ? 10*dim*( bopts.ismoo() ? bopts.moodim : 1 ) : bopts.totiters ),
    fninnerArg optargs(dim,
                       ( ( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints ) + ( ( bopts.totiters == -1 ) ? DEFITERS(dim) : bopts.totiters ),
                       fn,
                       fnarg,
                       ires,
                       allxres,
                       allfres,
                       allcres,
                       allFres,
                       allmres,
                       is_feas,
                       fres,
                       cres,
                       Fres,
                       mres,
                       sres,
                       xres,
                       allsres,
                       killSwitch,
                       hardmin,
                       hardmax,
                       obsnoise,
                       dimfid,
                       bopts.numfids,
                       scnoise,
                       bopts.numcgt);

    int res = bayesOpt(dim,locxres,locfres,locxmin,locxmax,fninner,(void *) &optargs,bopts,killSwitch,s_score);
    int isstable = bopts.stabpmax;

//errstream() << "allxres = " << allxres << "\n";
//errstream() << "allfres = " << allfres << "\n";
//errstream() << "allcres = " << allcres << "\n";
//errstream() << "allmres = " << allmres << "\n";
//errstream() << "ires = " << ires << "\n";
//errstream() << "fres = " << fres << "\n";
//errstream() << "xres = " << xres << "\n";
//errstream() << "allsres = " << allsres << "\n";

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
//FIXME: at this point need to modify allmres to include s_score
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
//allmres("&",k) *= ( ( s_score(k) > bopts.stabThresh ) ? 1.0 : DISCOUNTRATE );
allmres("&",k) *= 1/(1+exp(-1000*(s_score(k)-bopts.stabThresh)));
//                    allmres("&",k) *= 1/(1+exp(-(s_score(k)-bopts.stabThresh)/(s_score(k)*(1-s_score(k)))));
                }

                else
                {
                    allmres("&",k) *= s_score(k); // Allows us to do unscented and stable together
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

















/*
REMOVED FROM CODE

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

*/



/*
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
*/
