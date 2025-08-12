
//
// Bayesian Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "directopt.hpp"
#include "smboopt.hpp"
#include "imp_generic.hpp"
//#include "imp_nlsamp.hpp"
#include "plotml.hpp"

//
// Attempts to minimise target function using Bayesian optimisation.
//

#ifndef _bayesopt_h
#define _bayesopt_h

class BayesOptions;

int bayesOpt(int dim,
             Vector<double> &xres,
             gentype &fres,
             const Vector<double> &xmin,
             const Vector<double> &xmax,
             void (*fn)(int n, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, const Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xreplace, int &replacex),
             void *fnarg,
             BayesOptions &bopts,
             svmvolatile int &killSwitch,
             Vector<double> &sscore);

class BayesOptions : public SMBOOptions
{
    friend int bayesOpt(int dim,
             Vector<double> &xres,
             gentype &fres,
             const Vector<double> &xmin,
             const Vector<double> &xmax,
             void (*fn)(int n, gentype &res, const double *x, void *arg, double &addvar, Vector<gentype> &xsidechan, int &xobstype, const Vector<int> &xobstype_cgt, Vector<gentype> &ycgt, SparseVector<gentype> &xreplace, int &replacex),
             void *fnarg,
             BayesOptions &bopts,
             svmvolatile int &killSwitch,
             Vector<double> &sscore);

public:

    // WARNING: some documentation may be out of date.  Check code.
    //
    // method: 0  - MO (pure exploitation, mean only minimisation).
    //         1  - EI (expected improvement - default).
    //         2  - PI (probability of improvement).
    //         3  - GP-UCB as per Brochu (recommended GP-UCB).*
    //         4  - GP-UCB |D| finite as per Srinivas.
    //         5  - GP-UCB |D| infinite as per Srinivas.
    //         6  - GP-UCB p based on Brochu.
    //         7  - GP-UCB p |D| finite based on Srinivas.
    //         8  - GP-UCB p |D| infinite based on Srinivas.
    //         9  - PE (variance-only maximisation).
    //         10 - PEc (total variance-only, including constraints in variance maximisation).
    //         11 - GP-UCB with user-defined beta_t (see -gbv).
    //         12 - Thompson sampling.#
    //         13 - GP-UCB RKHS as per Srinivas.
    //         14 - GP-UCB RKHS as Chowdhury.#
    //         15 - GP-UCB RKHS as Bogunovic.~
    //         16 - Thompson sampling (unity scaling on variance).
    //         17 - GP-UCB as per Kandasamy (multifidelity 2017).
    //         18 - Human will be prompted to input x.
    //         19 - HE (human-level exploitation beta = 0.01).
    //         20 - GP-UCB as per BO-Muse (single AI).  Typically
    //              combined with human prompt
    //          * beta_n = 2.log((n^{2+dim/2}).(pi^2)/(3.delta))
    //          # Chowdhury, On Kernelised Multi-Arm Bandits, Alg 2
    //          ~ Bogunovic, Misspecified GP Bandit Optim., Lemma 1
    // evaluse: 0 = normal operation
    //          1 = user has option to change x and/or y
    // sigmuseparate: for multi-recommendation by default both sigma and by
    //         are approximated by the same ML.  Alternatively you can do
    //         them separately: mu is updated for each batch, and sigma
    //         independently for each point selected.
    //         0 = use the same ML.
    //         1 = use separate MLs.
    // startpoints: number of random (uniformly distributed) seeds used to
    //         initialise the problem.  Note that you can also put points
    //         into muapprox before calling this funciton if you have
    //         existing results or want to follow a particular pattern.
    // startseed: seed for RNG immediately prior to generating startpoints
    //         -1 if not used, -2 to seed with time (if >= 0 incremented whenever seeding happens so that
    //         multiple repeats have different (but predictable) sets of random numbers)
    //         Default 42.
    // algseed: seed for RNG immediately prior to running algorithm
    //         -1 if not used, -2 to seed with time (if >= 0 incremented whenever seeding happens so that
    //         multiple repeats have different (but predictable) sets of random numbers)
    //         Default -2.
    // totiters: total number of iterations in Bayesian optimisation
    //         set 0 for unlimited, -1 for 10d, -2 for err method (see err
    //         parameter).
    // intrinbatch: intrinsic batch size.  Default 1.  If > 0 then:
    //         mu(x) -> max(mu(x_0),mu(x_1),...,mu(x_{d-1}))
    //         sigma(x) -> det(covar(x_i,x_j))^(1/2intrinbatch), i,j = 1,2,...,d-1
    // intrinbatchmethod: 0 is standard, 1 means use mu(x) = min(...) instead
    // direcdim: if directpre != nullptr then this is the dimension of the
    //         input of the pre-processing function, which is the dimension
    //         of that DIRect sees.  Otherwise ignored.
    // itcntmethod: this controls how the iteration counter (t) used when
    //         calculating beta (for GP-UCB) is updated in batch mode.  If
    //         0 (default) then for each batch t -> t+1.  If 1 then for
    //         each batch t -> t+B, where B is the size of the batch (number
    //         of recommendations).
    // err: if maxitcnt == -2 then stopping uses this - see Kirschner et al,
    //         Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces
    // minstdev: if >0 then we add a penalty to the inner loop if the posterior
    //         variance is below minstdev
    //
    // cgtmethod: 0 - calculate probability of c(x)>=0, scale acquisition function by this (default)
    //            1 - build c(x) into mean/variance calculation before calculating acquisition function
    // cgtmargin: margin for cgt pass used in acquisiton function
    //
    // ztol:   zero tolerance (used when assessing sigma > 0, sigma = 0)
    // delta:  used by GP-UCB algorithm (0.1 by default)
    // nu:     used by GP-UCB algorithm (almost always 1)
    // modD:   used by GP-UCB {p} finite, size of search space set (-1 to infer from gridopt, if available - default).
    // a,b,r:  used by GP-UCB {p} infinite, see Srinivas theorem 2.
    // p:      used by GP-UCB p {in}finite, see Srinivas appendix.  Basically
    //         rather than set pi_t = (pi^2.t^2)/6 we3 set
    //         pi_t = zeta(p) t^p, which satisfies all the relevant
    //         requirements.  Srinivas considers the special case p = 2,
    //         where we note that zeta(2) = (pi^2)/6.
    // R:      scale factor used in TS (mode 12) (default 1)
    // B:      if <= 0 then not used, otherwise this is the bound on ||f||_K^2
    //         (default 1)
    // zeta:   the additional exploration heuristic for EI (default 0.01)
    //
    // impmeasu: improvement measure function.  If set will be used instead
    //           of EI/PI/whatever.
    // direcpre: rather than directly optimise the acquisition function
    //           if direcpre is set non-nullptr DIRect optimises a(p(x)), where
    //           a is the acquisition function and p is the function
    //           defined here.  The input of this must have dimension direcdim.
    // direcsubseqpre: rather than directly optimise the acquisition function
    //           if direcsubseqpre is set non-NULL DIRect optimisation
    //           a(p(x)) on all but the first recommendation in a block, where
    //           a is the acquisition function and p is the function
    //           defined here.  The input of this must have dimension direcdim.
    //
    // gridsource: usually optimisation is continuous. If you want to optimise
    //           on a finite set/grid then set this to point to the ML containing
    //           the valid x data. The index will be put in x[13]. There are a
    //           few modes:
    //           - null x1 x2 ... (no nulls in x)   - this is just a grid point, result unknown a-prior.
    //           - null x1 x2 ... (with nulls in x) - this is a partial grid point, the nulls will be optimised on a continuum.
    //           - y x1 x2 ...    (no nulls in x)   - this is a prior observation. It goes to build the objective model.
    //           - y x1 x2 ...    (with nulls in x) - this is a partial grid point where y is independent of the nulled xs.
    //           Constrained optimization: in this case y must include the constraint
    //           or it won't be considered a "full" observation. For example you can
    //           specify a grid in a constrained case containing only y: y will be
    //           treated as "known" and a model will be built for cgt
    //           NB: if there are nulls in any x then the method is selected *as
    //           if* the nulls were present in all x's.
    //           NULL by default.
    //
    // penalty: this is a vector of (positive valued) penalty functions.  When
    //           evaluating the acquisition function each of these will be
    //           evaluated and subtracted from the acquisition function.  These
    //           are used to enforce additional constraints on the ML.  Set them
    //           very positive in forbidden areas, near zero in the feasible
    //           region.
    //
    // Multi-fidelity BO:
    //
    // numfids: 0 = nothing
    //         >0 = final variable in x vector is the fidelity selector.  It
    //              has range {1/numfids,2/numfids,...,1} with numfids steps
    //              (if numfids = 1 then it is fixed to 1.  Method follows:
    //
    //              - Kandasamy, Bayesian Optimization with Continuous Approximations
    //
    //              Basically the first sample-selection step proceeds with this
    //              final variable fixed to 1, then a second step chooses this
    //              fidelity variable (keeping other variables fixed) to minimise
    //              some cost function.
    //              Data is cast as [ x~z ] for the model (x is data, z is fidelity),
    //              and the kernel will have multiplicative splitting, ie.
    //
    //                  -ks 2 -ki 1 -kS ...set x kernel... -ki 2 ...set z kernel...
    //
    // fidbudget:   budget for fidelity (-1 for unlimited).  If >1 then fidbudget/10
    //              is used for initialization and termination happens when budget is
    //              used up.
    // fidpenalty:  this function is the cost function for fidelity.  It's a function
    //              of z (the vector), larger means higher cost (lambda in the above paper).
    // dimfid:      if fidelity is multi-dimensional then set this >1.  1 by default
    // fidvar:      fidelity-based error-scale.  By default this is 0, but can be f(z)
    //              so that the noise variance becomes s+f(z)
    // fidover:     0: nothing
    //              1: human can override fidelity
    //              2: randomly choose fidelity less than recommended
    // FIXME: add fidvar to mlinter and test it
    //
    // Usage eg from Kandasamy with budget 100: ./svmheavyv7.exe -L res100
    // sigma value:            -gmd 0.5/14
    // GP-UCB method:          -gbH 3
    // fidelity gridsize:      -gbfid 10
    // budget cost:            -gbfp 0.1+\(6*x*x\)
    //                         -gbfb 100
    //                         -gbfn 2
    //                         -gmks 2
    //                         -gmki 0
    //                         -gmkS
    //                         -gmkt 3
    //                         -gmki 1
    //                         -gmkt 3
    //                         -gmki 0
    //                         -gb 3 \"-tM 1-\(1-\(0.1*\(1-v\)*exp\(-1/\(2*z\)\)\)\)*\(\(\(2300*y*y*y\)+\(1900*y*y\)+\(2092*y\)+60\)/\(\(100*y*y*y\)+\(500*y*y\)+\(4*y\)+20\)\)/14 -echo y -echo z -echo v\" \"-echo y -echo z -echo v\" fb 1 0 1 1 fb 2 0 1 13 fb 3 0 1 1
    //
    // Parameters controlling stable Bayesian optimisation
    //
    // stabpmax:    0 for no stability constraints, >= 1 for stability constraints 1:p, where p < pmax
    // stabpmin:    minimum value for p, if pmax >= 1
    // stabA:       A factor
    // stabB:       B factor
    // stabF:       F factor
    // stabbal:     [0,1]: 0 means use mu- (conservative), 1 means mu+ (optimistic), linear between
    // stabZeroPt:  zero point (lowbnd, chi)
    // stabDelrRep: repeats to calculate Delta_r
    // stabDelRep:  repeats to calculate Delta
    // stabUseSig:  set 1 to put stability scores through sigmoid function (default 1)
    // stabThresh:  threshold for stability (default 0.8)
    //
    // Unscented-Bayesian Optimisation
    //
    // @inproceedings{Nog1,
    //    author      = "Nogueira, Jos{\'e} and Martinez-Cantin, Ruben and Bernardina, Alexandre and Jamone, Lorenzo",
    //    title       = "Unscented Bayesian Optimization for Safe Robot Grasping",
    //    booktitle   = "Proceedings of the {IEEE/RSJ} International Conference on Intelligent Robots and Systems {IROS}",
    //    year        = "2016"}
    //
    // unscentUse:       0 normal, 1 use unscented transform
    // unscentK:         k value used in unscented optimisation (typically either 0 or -3)
    // unscentSqrtSigma: square root of sigma matrix used by unscented transform (noise on input x)
    //
    //
    //
    // See:
    //
    // @techreport{Bro2,
    //    author      = "Brochu, Eric and Cora, Vlad~M. and {de~Freitas}, Nando",
    //    title       = "A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Applications to Active User Modeling and Heirarchical Reinforcement Learning",
    //    institution = "{arXiv.org}",
    //    month       = "December",
    //    number      = "{arXiv:1012.2599}",
    //    type        = "eprint",
    //    year        = "2010"}
    //
    // gpUCB: in all cases sqrt(beta) = sqrt(nu.tau_t).
    //        tau_t depends on particular variant.
    //        d = dim(x).
    //
    // gpUCB basic:      tau_t = 2.log(t^{2+d/2}.pi^2/(3.delta))
    // gpUCB finite:     tau_t = 2.log(modD.t^2.pi^2/(6.delta))
    // gpUCB infinite:   tau_t = 2.log(t^2.2.pi^2/(3.delta)) + 4d.log(t^2.d.b.r.sqrt(log(4.d.a/delta)))
    // gpUCB p basic:    tau_t = 2.log(2.t^{d/2}.pi_t/delta)
    //                   pi_t = zeta(p).t^p
    // gpUCB p finite:   tau_t = 2.log(modD.pi_t/delta)
    //                   pi_t = zeta(p).t^p
    // gpUCB p infinite: tau_t = 2.log(4.pi_t/delta) + 4d.log(t^2.d.b.r.sqrt(log(4.d.a/delta)))
    //                   pi_t = zeta(p).t^p
    //
    // gpUCB p basic is inferred as follows.  Brochu states (without giving
    // working) that beta_t from Srinivas has the form of gpUCB basic.
    // Comparing this with gpUCB finite we see that they are equivalent if
    // we make the (unverified) assumption that |D_t| = 2.t^{d/2}.  This is
    // somewhat reminiscent of the proofs in Srinivas but does not match
    // exactly.  In any case, we note that gpUCB finite is just gpUCB p finite
    // with p = 2, so we generalise to get the expression above.
    //
    // Of course this is very speculative, but unfortunately Brochu's paper
    // entirely fails to report *where* their "bold" claim about tau_t (page
    // 16) is drawn from.  *Assuming* that Brochu has pulled this result
    // from somewhere sensible it seems reasonable to assume that the result
    // will follow.
    //
    //
    //
    // Multi-recommendation: choose method == 11 (gpUCB, user defined) and
    // betafn a vector of functions to select multi-recommendation.  Note
    // that in this case the number of start points added is s*n, where
    // s = startpoints and n = multi-recommendation batch size.
    //
    // To reference internal methods in multi-recommendation (that is, other
    // than method 11) replace the equation with a vector (where {} indicates
    // an optional element):
    //
    // [ method   ]
    // [ {p}      ]
    // [ {betafn} ]
    // [ {modD}   ]
    // [ {nu}     ]
    // [ {delta}  ]
    // [ {a}      ]
    // [ {b}      ]
    // [ {r}      ]
    // [ {R}      ]
    //
    // (if you want to change an element but not some before it use [] in
    // place of the elements you don't want to change).
    //
    // For multi-objective based multi-recommendation use betafn = null
    // (or [ null null ... ] for multiple rounds of multi-objective multi-rec)

    int method;
    int intrinbatch;
    int intrinbatchmethod;
    int evaluse;
    //int sigmuseparate;
    int startpoints;
    int startseed;         // this changes during optimisation setup to ensure no repeats unless specified
    int algseed;           // ...as does this
    int totiters;
    int itcntmethod;
    double err;
    double minstdev;
    int humanfreq;
    int cgtmethod;
    double cgtmargin;

    double ztol;
    double delta;
    double zeta;
    double nu;
    double modD;
    double a;
    double b;
    double r;
    double p;
    double R;
    double B;
    gentype betafn;

    IMP_Generic *impmeasu;
    ML_Base *direcpre;
    ML_Base *direcsubseqpre;
    int direcdim;
    Vector<double> direcmin;
    Vector<double> direcmax;
    ML_Base *gridsource;

    Vector<ML_Base *> penalty;

    int numfids;
    int dimfid;
    double fidbudget;
    gentype fidpenalty;
    gentype fidvar;
    int fidover;

    // Stable optimisation

    int stabpmax;
    int stabpmin;
    int stabUseSig;
    double stabA;
    double stabB;
    double stabF;
    double stabbal;
    double stabZeroPt;
    double stabDelrRep;
    double stabDelRep;
    double stabThresh;

    // Unscented optimisation

    int unscentUse;
    int unscentK;
    Matrix<double> unscentSqrtSigma;

    // DIRect options (note that global options in this are over-ridden by *this, so only DIRect parts matter)

    DIRectOptions extgoptssingleobj;

    // Multi-objective, multi-recommendation part

    DIRectOptions goptsmultiobj; // full over-ride
    int startpointsmultiobj;
    int totitersmultiobj;
    int ehimethodmultiobj;

    BayesOptions(IMP_Generic *impmeasux = nullptr, ML_Base *xdirecpre = nullptr, int xdirecdim = 0, ML_Base *xdirecsubseqpre = nullptr, ML_Base *xgridsource = nullptr) : SMBOOptions()
    {
        optname = "Bayesian Optimisation";

        method            = 1;
        intrinbatch       = 1;
        intrinbatchmethod = 0;
        evaluse           = 0;
        //sigmuseparate     = 0;
        startpoints       = -1; //5; //500; //10;
        startseed         = 42;
        algseed           = 69;
        totiters          = -1; //100; //200; //500;
        itcntmethod       = 0;
        err               = 1e-1;
        minstdev          = 0;
        humanfreq         = 0;
        cgtmethod         = 0;
        cgtmargin         = 0.1; //1;

        ztol   = DEFAULT_BAYES_ZTOL;
        delta  = DEFAULT_BAYES_DELTA;
        zeta   = 0; //0.01; use true EI as default, even if it's a bit slow
        nu     = DEFAULT_BAYES_NU;
        modD   = -1; // this is entirely arbitrary and must be set by the user
        a      = DEFAULT_BAYES_A; // a value
        b      = DEFAULT_BAYES_B; // another value
        r      = DEFAULT_BAYES_R; // This is basically the width of our search region in
                                  // any given dimension.  Usually you would want to
                                  // normalise to 0->1, so 1 is correct.
        p      = DEFAULT_BAYES_P;
        betafn = 0;
        R      = 1;
        //B      = 1; // small positive value or things get weird.
        B      = -1; // use actual norm

        numfids    = 0;
        dimfid     = 1;
        fidbudget  = -1;
        fidpenalty = 1;
        fidvar     = 0;
        fidover    = 0;

        impmeasu       = impmeasux;
        direcpre       = xdirecpre;
        direcsubseqpre = xdirecsubseqpre;
        direcdim       = xdirecdim;
        direcmin.resize(direcdim);
        direcmax.resize(direcdim);

        gridsource = xgridsource;

        startpointsmultiobj = startpoints;
        totitersmultiobj    = totiters;
        ehimethodmultiobj   = 0;

        stabpmax    = DEFAULT_BAYES_STABPMAX;
        stabpmin    = DEFAULT_BAYES_STABPMIN;
        stabUseSig  = DEFAULT_BAYES_STABUSESIG;
        stabA       = DEFAULT_BAYES_STABA;
        stabB       = DEFAULT_BAYES_STABB;
        stabF       = DEFAULT_BAYES_STABF;
        stabbal     = DEFAULT_BAYES_STABBAL;
        stabZeroPt  = DEFAULT_BAYES_STABZEROPT;
        stabDelrRep = DEFAULT_BAYES_STABDELRREP;
        stabDelRep  = DEFAULT_BAYES_STABDELREP;
        stabThresh  = DEFAULT_BAYES_STABTHRESH;

        unscentUse = 0;
        unscentK   = 0;
    }

    BayesOptions(const BayesOptions &src) : SMBOOptions(src)
    {
        *this = src;
    }

    BayesOptions &operator=(const BayesOptions &src)
    {
        SMBOOptions::operator=(src);

        method            = src.method;
        intrinbatch       = src.intrinbatch;
        intrinbatchmethod = src.intrinbatchmethod;
        evaluse           = src.evaluse;
        //sigmuseparate     = src.sigmuseparate;
        startpoints       = src.startpoints;
        startseed         = src.startseed;
        algseed           = src.algseed;
        totiters          = src.totiters;
        itcntmethod       = src.itcntmethod;
        err               = src.err;
        minstdev          = src.minstdev;
        numfids           = src.numfids;
        dimfid            = src.dimfid;
        fidbudget         = src.fidbudget;
        fidpenalty        = src.fidpenalty;
        fidvar            = src.fidvar;
        fidover           = src.fidover;
        humanfreq         = src.humanfreq;
        cgtmethod         = src.cgtmethod;
        cgtmargin         = src.cgtmargin;

        ztol   = src.ztol;
        delta  = src.delta;
        zeta   = src.zeta;
        nu     = src.nu;
        modD   = src.modD;
        a      = src.a;
        b      = src.b;
        r      = src.r;
        p      = src.p;
        betafn = src.betafn;
        R      = src.R;
        B      = src.B;

        impmeasu       = src.impmeasu;
        direcpre       = src.direcpre;
        direcsubseqpre = src.direcsubseqpre;
        direcdim       = src.direcdim;
        direcmin       = src.direcmin;
        direcmax       = src.direcmax;
        gridsource     = src.gridsource;

        penalty           = src.penalty;

        stabpmax    = src.stabpmax;
        stabpmin    = src.stabpmin;
        stabUseSig  = src.stabUseSig;
        stabA       = src.stabA;
        stabB       = src.stabB;
        stabF       = src.stabF;
        stabbal     = src.stabbal;
        stabZeroPt  = src.stabZeroPt;
        stabDelrRep = src.stabDelrRep;
        stabDelRep  = src.stabDelRep;
        stabThresh  = src.stabThresh;

        unscentUse       = src.unscentUse;
        unscentK         = src.unscentK;
        unscentSqrtSigma = src.unscentSqrtSigma;

        goptssingleobj    = src.goptssingleobj;
        extgoptssingleobj = src.extgoptssingleobj;

        goptsmultiobj       = src.goptsmultiobj;
        startpointsmultiobj = src.startpointsmultiobj;
        totitersmultiobj    = src.totitersmultiobj;
        ehimethodmultiobj   = src.ehimethodmultiobj;

        return *this;
    }

    // Reset function so that the next simulation can run

    virtual void reset(void) override
    {
        SMBOOptions::reset();

        goptssingleobj = extgoptssingleobj;

        return;
    }

    // Logging (plot IMP)

    virtual void model_log(int stage, double xmin = 0, double xmax = 1, double ymin = 0, double ymax = 1) override
    {
        std::string stagestr = ( stage == 0 ) ? "pre" : ( ( stage == 1 ) ? "mid" : "post" );

        if ( isimphere() && ( moodim == 2 ) && ( (*impmeasu).xspaceDim() == 2 ) )
        {
            if ( plotfreq && ( ( stage == 0 ) || ( stage == 2 ) || ( ( plotfreq > 0 ) && ( !(model_N_mu()%plotfreq) ) ) ) )
            {
                {
                    int plotoutformat = modeloutformat;
                    int plotsquare = 1;
                    int plotimp = 1;

                    SparseVector<gentype> plotxtemplate;

                    int xindex = 0;
                    int yindex = 1;

                    double xmin = 0; //FIXME: is this correct?
                    double xmax = 1; //FIXME: is this correct?
                    double ymin = 0; //FIXME: is this correct?
                    double ymax = 1; //FIXME: is this correct?
                    double omin = 1; //FIXME: is this correct?
                    double omax = 0; //FIXME: is this correct?

                    std::string fname(modelname);
                    std::string dname(modelname);
                    std::string mlname(modelname);

                    fname += "_imp_plot";
                    dname += "_imp_data";
                    mlname += "_imp_gpr";

                    std::stringstream ssi;

                    ssi << model_N_mu(); // /plotfreq;

                    fname += stagestr;
                    dname += stagestr;
                    mlname += stagestr;

                    fname += "_";
                    dname += "_";
                    mlname += "_";

                    fname += ssi.str();
                    dname += ssi.str();
                    mlname += ssi.str();

                    int incdata = 0;
                    int incvar  = 0;
                    int xusevar = 0;
                    int yusevar = 1;

                    gentype baseline(modelbaseline);

                    plotml(*impmeasu,xindex,yindex,xmin,xmax,ymin,ymax,omin,omax,fname,dname,plotoutformat,incdata,baseline,incvar,xusevar,yusevar,plotxtemplate,plotsquare,plotimp);

                    std::ofstream mldest(mlname,std::ofstream::out);
                    mldest << *impmeasu;
                    mldest.close();
//errstream() << "phantomxyz impmeasu = " << *impmeasu << "\n";
                }
            }
        }

        return SMBOOptions::model_log(stage,xmin,xmax,ymin,ymax);
    }

    // Generate a copy of the relevant optimisation class.

    virtual GlobalOptions *makeDup(void) const
    {
        BayesOptions *newver;

        MEMNEW(newver,BayesOptions(*this));

        return newver;
    }

    // supres: [ see .cc file ] for each evaluation of (*fn),
    //         where beta is the value used to find the point being
    //         evaluated (zero for initial startpoint block)

    virtual int optim(int dim,
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
                      svmvolatile int &killSwitch);

    virtual int optim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      int &mInd,
                      Vector<int> &muInd,
                      Vector<int> &augxInd,
                      Vector<int> &cgtInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
                      Vector<Vector<gentype> > &allcres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &allsres,
                      Vector<double>  &s_score,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch,
                      size_t numReps,
                      gentype &meanfres, gentype &varfres,
                      gentype &meanires, gentype &varires,
                      gentype &meantres, gentype &vartres,
                      gentype &meanTres, gentype &varTres,
                      Vector<gentype> &meanallfres, Vector<gentype> &varallfres,
                      Vector<gentype> &meanallmres, Vector<gentype> &varallmres)
    {
        int res = SMBOOptions::optim(dim,xres,Xres,fres,ires,mInd,muInd,augxInd,cgtInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,allfres,allcres,allmres,allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch,numReps,meanfres,varfres,meanires,varires,meantres,vartres,meanTres,varTres,meanallfres,varallfres,meanallmres,varallmres);

        return res;
    }

    // IMP use

    int isimphere(void) const { return impmeasu ? 1 : 0; }

    int modelimp_imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxvar) const
    {
        NiceAssert( impmeasu );

        return (*impmeasu).imp(resi,resv,xxmean,xxvar);
    }

    int modelimp_N(void) const
    {
        NiceAssert( impmeasu );

        return (*impmeasu).N();
    }

    int modelimp_addTrainingVector(const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1)
    {
        NiceAssert( impmeasu );

        int i = modelimp_N();

        return (*impmeasu).addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    int modelimp_train(int &res, svmvolatile int &killSwitch)
    {
        int ires = 0;

        if ( isimphere() )
        {
            (*impmeasu).setxdim(ismoo ? moodim : 1);

            ires = (*impmeasu).train(res,killSwitch);
        }

        return ires;
    }


    virtual int optdefed(void)
    {
        return 3;
    }

    int impmeasuNonLocal(void) const
    {
        return impmeasu != nullptr;
    }

    int direcpreDef(void) const
    {
        return direcpre != nullptr;
    }

    int direcsubseqpreDef(void) const
    {
        return direcsubseqpre != nullptr;
    }

    virtual int getdimfid(void) const override { return numfids ? dimfid : 0;  }

//private:
    // We actually work with this local version as it may need to be overwritten
    // (reset restores from ext... copy)

    DIRectOptions goptssingleobj;
};

#endif
