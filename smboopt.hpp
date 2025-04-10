
//
// Sequential model-based optimisation base class
//
// Date: 2/12/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//#define TURNOFFSHORTCUT

#include "ml_base.hpp"
#include "ml_mutable.hpp"
#include "gpr_scalar.hpp"
#include "gpr_scalar_rff.hpp"
#include "gpr_vector.hpp"
#include "globalopt.hpp"
#include "gridopt.hpp"
#include "directopt.hpp"
#include "nelderopt.hpp"
#include "errortest.hpp"
#include "addData.hpp"

#ifndef _smboopt_h
#define _smboopt_h

class SMBOOptions : public GlobalOptions
{
public:
    // Default models

    GPR_Scalar altfnapprox;            // Default model(s) for objective (objectives for MOO)
    GPR_Scalar altfxapprox;            // Default model(s) for augmented channel(s)
    GPR_Scalar_rff altfnapprox_rff;    // Default model for RFFs
    GPR_Scalar altfnapproxFNapprox;    // I have no idea what I included this for.  Some sort of kernel backup?

    // External models (moo == multi-objective), fx = model for q (see below), if used

    Vector<ML_Base *> extfnapprox;      // Use this to give alternative model(s) for objective(s).
    Vector<ML_Base *> extfxapprox;      // Use this to give alternative models for augmented channels.

    // sigmuseparate: for multi-recommendation by default both sigma and mu
    //         are approximated by the same ML.  Alternatively you can do
    //         them separately: mu is updated for each batch, and sigma
    //         independently for each point selected.
    //         0 = use the same ML.
    //         1 = use separate MLs.
    // ismoo:  set 1 for multi-objective optimisation
    // moodim: number of objectives
    // sampleNsamp: number of grid points in scalarisor (for TS, default -20)
    //
    // modeltype:  0 = model f(p(x)) using p(x)
    //             1 = model f(p(x)) using p(x), model_clear resets model
    //             2 = model f(p(x)) using x, model_clear resets model
    //             3 = model f(p(x)) using x
    // modelrff:   0 = normal
    //            >0 = use this many random features
    // oracleMode: 0 = oracle uses GP model derivative to find mean/variance
    //                 descent direction at current best solution, samples that
    //             1 = oracle uses GP model derivative to find mean descent
    //                 direction, uses that
    //             2 = fallback (purely random oracle)
    //             3 = use mode 0 for primary axis, mode 2 for the rest
    //             4 = use mode 1 for primary axis, mode 2 for the rest
    //
    // Thompson sampling:
    //
    // TSmode = 1: grid sample
    //          3: JIT sample (recommended)
    // TSNsamp: +N for take N samples in TSmode = 1
    //          -N for random samples in TSmode = 1
    //          0 for 10jd^2 random samples in TSmode = 1, random
    // TSsampType: see ml_base.h
    // TSxsampType: see ml_base.h
    // sigma_cut: sigma scale for variance sampling cutoff in TS
    //
    // Transfer learning:
    //
    // tranmeth: 0 - content of muModel treated as data from current model
    //           1 - transfer learning as per Joy1 (Shi21 Env-GP)
    //           2 - transfer learning as per Shi21, Diff-GP
    // alpha0:   starting alpha value
    // beta0:    starting beta value
    //
    // Kernel transfer learning:
    //
    // kernapprox: pointer to ML to copy kernel from
    // kxfnum:     kernel transfer type (-kt)
    // kxfnorm:    0 - no normalisation
    //             1 - yes
    //
    // Model tuning:
    //
    // tunemu:      set to tune muapprox at every step (default 1)
    // tunesigma:   set to tune sigmaapprox at every step if sigmuseparate (default 1)
    // tunesrcmod:  set to tune srcmodel at start (default 1)
    // tunediffmod: set to tune diffapprox at every step (default 1)
    // tuneaugxmod: set to tune augxapprox at every step (default 1)
    //
    // 0: don't tune
    // 1: tune for max-likelihood
    // 2: tune for leave-one-out (default)
    // 3: tune for recall
    //
    // xtemplate: "background" template for x data.  To construct data to be added
    //            to models we start with this template and over-write parts with x.
    //            For example if xtemplate = [ ~ xa ] then data x -> [ x ~ xa ].
    //
    // usemodelaugx: 0   for normal
    //               n>0 when modelling f, use [ x ~ q1(x) ~ q2(x) ~ ... ~ qn(x) ],
    //                   where qi(x) models an ancilliary function for i=1,2,...,n.
    //                   The output of the ancilliary functions is returned as the
    //                   third element of the set:
    //                      { f(x), var(x), [ q1(x), q2(x), ..., qn(x) ] }.
    //                   The GP of f (and others) have "training data" of the form
    //                   [ x ~ q1(x) ~ q2(x) ~ ... ~ qn(x) ], and the model of
    //                   qi(x) is used when BO optimises the acquisition function.
    // modelnaive: 0 = normal operation, noise from qi(x) is converted to noise
    //                 on output of model
    //             1 = suppress noise on qi(x) to give an overly optimistic model.
    // ennornaive: 0  = normal operation, augx stuff occurs as [ x ~ x' ~ ... ]
    //             >0 = change [ x ~ x' ~ x'' ~... ] to [ x ; n:x' ; 2n:x'' ... ]
    //                  when in models, where n = ennornaive
    //             NB: MUST BE ZERO IF USING MULTI-FIDELITY STUFF!
    //
    // Generated noise: makenoise = 0: no noise
    //                              1: add noise to samples
    //
    // modelname: model base-name.  If you plot the GPR periodically this is used
    // modeloutformat: model output format (0 terminal, 1 ps, 2 pdf)
    // plotfreq: visualise model (1-d only) every this many iterations if nz, plus
    //           on exit.
    //           -1 means log only on exit
    // modelbaseline: baseline (function) to plot, null() if not used

    std::string modelname;
    int modeloutformat;
    int plotfreq;
    std::string modelbaseline;

    int sigmuseparate;
    int ismoo;
    int moodim;
    int modeltype;
    int modelrff;
    int oracleMode;

    int TSmode;
    int TSNsamp;
    int TSsampType;
    int TSxsampType;
    double sigma_cut;

    int tranmeth;
    double alpha0;
    double beta0;

    ML_Base *kernapprox;
    int kxfnum;
    int kxfnorm;

    int tunemu;
    int tunesigma;
    int tunesrcmod;
    int tunediffmod;
    int tuneaugxmod;

    SparseVector<gentype> xtemplate;

    int usemodelaugx;
    int modelnaive;
    int makenoise;
    int ennornaive;

    virtual int getdimfid(void) const { return 0; } // This will be overridden in BayesOpt.  If >0 we need to converts [ x z ] to [ x~z ] in model_convertx, where z is getdimfid() dimensional

    // Constructors and assignment operators

    SMBOOptions();
    SMBOOptions(const SMBOOptions &src);
    SMBOOptions &operator=(const SMBOOptions &src);

    // Reset function so that the next simulation can run

    virtual void reset(void) override
    {
        GlobalOptions::reset();

        indpremu.resize(0);
        presigweightmu.resize(0);

        Nbasemu = 0;
        resdiff = 9;

        alpha = 0;
        beta  = 0;

        if ( srcmodel )
        {
            (*srcmodel).restart();
        }

        if ( diffmodel )
        {
            (*diffmodel).restart();
        }

        srcmodel  = nullptr;
        diffmodel = nullptr;

        srcmodelInd  = -1;
        diffmodelInd = -1;

        diffval  = 0;
        predval  = 0;
        storevar = 0;

        firsttrain = 1;

        xmodprod.resize(0,0);
        xshortcutenabled = 0;
        xsp.resize(0);
        xspp.resize(0);

        model_unsample();

        int i;

        for ( i = 0 ; i < muapprox.size() ; i++ )
        {
            if ( muapprox(i) )
            {
                (*(muapprox("&",i))).restart();
            }
        }

        if ( sigmaapprox )
        {
            (*sigmaapprox).restart();
        }

        for ( i = 0 ; i < augxapprox.size() ; ++i )
        {
            if ( augxapprox(i) )
            {
                (*(augxapprox("&",i))).restart();
            }
        }

        locires.resize(0);
        locxres.resize(0);
        locxresunconv.resize(0);
        locyres.resize(0);

        modelErrOptim = nullptr;
        ismodelErrLocal = 1;

        locdim = 0;

        xx.zero();
        xxvar.zero();

        for ( i = 0 ; i < fnapprox.size() ; ++i )
        {
            if ( fnapprox(i) )
            {
                (*(fnapprox("&",i))).restart();
            }
        }

        for ( i = 0 ; i < fxapprox.size() ; ++i )
        {
            if ( fxapprox(i) )
            {
                (*(fxapprox("&",i))).restart();
            }
        }

        //fnapprox.resize(0);
        //fxapprox.resize(0);

        //fnapproxInd.resize(0);
        //fxapproxInd.resize(0);

        return;
    }

    // Generate a copy of the relevant optimisation class.

    virtual GlobalOptions *makeDup(void) const;

    // virtual Destructor to get rid of annoying warnings

    virtual ~SMBOOptions()
    {
        killModelErrOptim();
    }

    // Oracles

    virtual void consultTheOracle(ML_Mutable &randDir, int dim, const SparseVector<gentype> &locxres, int isFirstAxis);

    // Optimisation functions etc fall back to GlobalOptions

    virtual int optim(int dim,
                      Vector<gentype> &rawxres,
                      gentype &fres,
                      int &ires,
                      Vector<Vector<gentype> > &allrawxres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allfresmod,
                      Vector<gentype> &supres,
                      Vector<double> &sscore,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
    {
        return GlobalOptions::optim(dim,rawxres,fres,ires,allrawxres,allfres,allfresmod,supres,sscore,xmin,xmax,fn,fnarg,killSwitch);
    }

    virtual int optim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      int &mInd,
                      Vector<int> &muInd,
                      Vector<int> &augxInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
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
        int res = GlobalOptions::optim(dim,xres,Xres,fres,ires,mInd,muInd,augxInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,allfres,allmres,allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch,numReps,meanfres,varfres,meanires,varires,meantres,vartres,meanTres,varTres,meanallfres,varallfres,meanallmres,varallmres);

        return res;
    }

    virtual int realOptim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &rawxres,
                      gentype &fres,
                      int &ires,
                      int &mres,
                      Vector<int> &muInd,
                      Vector<int> &augxInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allrawxres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allfresmod,
                      Vector<gentype> &supres,
                      Vector<double> &sscore,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch);

    // Local or global models?
    //
    // resvar is the variance of (part of) x, if this is esimated using the aug model
    // isvarnz is set if resvar is non-zero.  Otherwise it is made 0.

    template <class S>
    const SparseVector<gentype> &model_convertx(SparseVector<gentype> &res, const SparseVector<S> &x, int useOrigin = 0, int useShortcut = 0) const;
    template <class S>
    const SparseVector<gentype> &model_convertx(int &isvarnz, SparseVector<gentype> &resvar, SparseVector<gentype> &res, const SparseVector<S> &x, int useOrigin = 0, int useShortcut = 0, int debug = 0) const;
    template <class S>
    const Vector<SparseVector<gentype> > &model_convertx(Vector<SparseVector<gentype> > &res, const Vector<SparseVector<S> > &x) const
    {
        return GlobalOptions::model_convertx(res,x);
    }

    virtual void model_clear (void);
    virtual void model_update(void);

    double log_xmin; // temp used by model_log
    double log_xmax; // "

    double log_ymin; // temp used by model_log
    double log_ymax; // "

    virtual void model_log(int stage, double xmin = 0, double xmax = 1, double ymin = 0, double ymax = 1) override;

    // Sample model for Thompson sampling

    void model_sample  (const Vector<double> &xmin, const Vector<double> &xmax, double sampScale);
    void model_unsample(void);
    int  model_issample(void) const;

    // Model control and use functionality

    virtual int initModelDistr(const Vector<int> &sampleInd, const Vector<gentype> &sampleDist);

    // Observation noise levels in models (variance)

    double getfnnoise(void)
    {
        return makenoise ? (*sigmaapprox).sigma() : 0.0;
    }

    const Vector<double> &getfxnoise(Vector<double> &res)
    {
        res.resize(augxapprox.size());

        int i;

        for ( i = 0 ; i < fxapprox.size() ; ++i )
        {
            res("&",i) = makenoise ? (*(augxapprox(i))).sigma() : 0.0;
        }

        return res;
    }

    // Model use:
    //
    // There are potentially three models here:
    //
    // mu:    this models mu, and may also model sigma if there is no separate sigma model
    // sigma: if present, this models sigma
    // augx:  a vector of models that model qi(x) (this model is hidden)
    //
    // NB: - x() refers to real x, not the "fake" x seen by eg the Bayesian optimiser using a log-scale transform
    //     - N and NNC can differ between mu and sigma models
    //     - xsidechan is the "additional" vector [ q1(x), q2(x), ..., qn(x) ]
    //
    //
    // inf_dist: calculates (4) in Kamdasamy "Bayesian Optimisation with Continuous Approximations"
    //           zeta(z) = sqrt( 1 - (K([x z],[x 1])/K(x,x)) )
    //
    // Assumptions: - the kernel is formed as per the paper, so the denominator in the second part
    //                effectively cancels out the "x" part of the kernel evaluation.
    //              - the kernel is decreasing in ||x-x'||, so the inf-norm is as stated

    template <class S> int  model_mu   (                 gentype        &resmu, const SparseVector<S> &x, const vecInfo *xing = nullptr               ) const;
    template <class S> int  model_mu   (                 Vector<double> &resmu, const SparseVector<S> &x, const vecInfo *xing = nullptr               ) const;
    template <class S> int  model_muvar(gentype &resvar, gentype        &resmu, const SparseVector<S> &x, const vecInfo *xing = nullptr, int debug = 0) const;
    template <class S> int  model_var  (gentype &resvar,                        const SparseVector<S> &x, const vecInfo *xing = nullptr               ) const
    {
        gentype dummy;

        return model_muvar(resvar,dummy,x,xing);
    }

    template <class S> double inf_dist(const SparseVector<S> &xz) const;

    int model_muTrainingVector   (                 gentype        &resmu,           int imu) const;
    int model_muTrainingVector   (                 Vector<double> &resmu,           int imu) const;
    int model_muvarTrainingVector(gentype &resvar, gentype        &resmu, int ivar, int imu) const;
    int model_varTrainingVector  (gentype &resvar,                        int ivar         ) const;

    int model_N_mu   (int q = 0) const { return (*(muapprox(q))).N(); }
    int model_N_sigma(void)      const { return (*sigmaapprox).N();   }

    int model_NNCz_mu   (int q) const { return (*(muapprox(q))).NNC(0); }
    int model_NNCz_sigma(void)  const { return (*sigmaapprox).NNC(0);   }

    template <class S> int  model_covar(Matrix<gentype> &rescov, const Vector<SparseVector<S> > &x) const;
    template <class S> void model_stabProb(double &res, const SparseVector<S> &x, int p, double pnrm, int rot, double mu, double B) const;

    int  model_covarTrainingVector(Matrix<gentype> &rescov, const Vector<int> &i) const { return (*sigmaapprox).covarTrainingVector(rescov,i); }
    void model_stabProbTrainingVector(double &res, int i, int p, double pnrm, int rot, double mu, double B, int q = 0) const { (*(muapprox(q))).stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); }

    int model_addTrainingVector_musigma       (const gentype &y, const gentype &ypred, const SparseVector<gentype> &x,                                                                                                                                                                                               double varadd = 0              );
    int model_addTrainingVector_musigma       (const gentype &y, const gentype &ypred, const SparseVector<gentype> &x, const Vector<gentype> &xsidechan, const Vector<gentype> &xaddrank, const Vector<gentype> &xaddranksidechan, const Vector<gentype> &xaddgrad, const Vector<gentype> &xaddf4, int xobstype = 2, double varadd = 0              );
    int model_addTrainingVector_sigmaifsep    (const gentype &y,                       const SparseVector<gentype> &x,                                                                                                                                                                                               double varadd = 0              );
    int model_addTrainingVector_mu_sigmaifsame(const gentype &y, const gentype &ypred, const SparseVector<gentype> &x, const Vector<gentype> &xsidechan, const Vector<gentype> &xaddrank, const Vector<gentype> &xaddranksidechan, const Vector<gentype> &xaddgrad, const Vector<gentype> &xaddf4, int xobstype = 2, double varadd = 0, int dval = 2);

    double model_sigma(int q) const { return (*(muapprox(q))).sigma(); }

    const MercerKernel &model_getKernel(int q) const { return (*(muapprox(q))).getKernel(); }

    const SparseVector<gentype> &model_x(int i, int q = 0) const { return (*(muapprox(q))).x(i); }
    const Vector<gentype>       &model_y(int q = -1)       const { return ( q == -1 ) ? ( ismoo ? locyres : (*(muapprox(0))).y() ) : (*(muapprox(q))).y();  }
    const Vector<int>           &model_d(int q = 0)        const { return (*(muapprox(q))).d();  }

    int model_train      (int &res, svmvolatile int &killSwitch);
    int model_train_sigma(int &res, svmvolatile int &killSwitch);

    int model_setd                 (int imu, int isigma, int    nd);
    int model_setd_mu              (int imu, int                nd);
    int model_setd_sigma           (         int isigma, int    nd);
    int model_setsigmaweight_addvar(int imu, int isigma, double addvar);

    const Vector<double> &model_xcopy(Vector<double> &resx, int i) const;

    double model_negloglikelihood(int q)            const { return calcnegloglikelihood(*(muapprox(q))); }
    double model_maxinfogain     (int q)            const { return calcmaxinfogain     (*(muapprox(q))); }
    double model_RKHSnorm        (int q)            const { return calcRKHSnorm        (*(muapprox(q))); }
    double model_lenscale        (int q, int i = 0) const { return (double) ((*(getmuapprox(q))).getKernel().cRealConstants(i%((*(getmuapprox(q))).getKernel().size())))(0); }
    double model_kappa0          (int q)            const { return (double) (*(getmuapprox(q))).getKernel().effweight(q); }

    // Work out frequentist certainty - see "Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimentional Subspaces"

    double model_err(int dim, const Vector<double> &xmin, const Vector<double> &xmax, svmvolatile int &killSwitch);

    // some info

    int ismodelaug  (void) const { return usemodelaugx; }
    int ismodelnaive(void) const { return modelnaive;   }

    // Simple default-model adjustments

    int default_model_settspaceDim  (int                          nv) { return altfnapprox.settspaceDim(nv);   }
    int default_model_setsigma      (double                       nv) { return altfnapprox.setsigma(nv);       }
    int default_model_setvarApproxim(int                          nv) { return altfnapprox.setvarApproxim(nv); }
    int default_model_setkernelg    (const gentype               &nv);
    int default_model_setkernelgg   (const SparseVector<gentype> &nv);

    int default_modelaugx_settspaceDim  (int                          nv) { return altfxapprox.settspaceDim(nv); }
    int default_modelaugx_setsigma      (double                       nv) { return altfxapprox.setsigma(nv);     }
    int default_modelaugx_setvarApproxim(int                          nv) { return altfnapprox.setvarApproxim(nv); }
    int default_modelaugx_setkernelg    (const gentype               &nv);
    int default_modelaugx_setkernelgg   (const SparseVector<gentype> &nv);

//private:

    void addinxsidechan(SparseVector<gentype> &x, const Vector<gentype> &xsidechan, const Vector<gentype> &xaddrank, const Vector<gentype> &xaddranksidechan, const Vector<gentype> &xaddgrad, const Vector<gentype> &xaddf4);

    // This version adds to augx if enabled, augments x if required, then adds to mu - use this one by default!

    int modelmu_int_addTrainingVector   (const gentype         &y,                       const SparseVector<gentype> &x, const SparseVector<gentype> &xx, int xobstype = 2, double varadd = 0, int dval = 2);
    int modelsigma_int_addTrainingVector(const gentype         &y,                                                       const SparseVector<gentype> &xx, int xobstype = 2, double varadd = 0);
    int modeldiff_int_addTrainingVector (const gentype         &y, const gentype &ypred,                                 const SparseVector<gentype> &xx, int xobstype = 2, double varadd = 0, int dval = 2);
    int modelaugx_int_addTrainingVector (const Vector<gentype> &y,                       const SparseVector<gentype> &,                                                     double varadd = 0);

    int modelmu_int_train   (int &res, svmvolatile int &killSwitch);
    int modelsigma_int_train(int &res, svmvolatile int &killSwitch);
    int modeldiff_int_train (int &res, svmvolatile int &killSwitch);
    int modelaugx_int_train (int &res, svmvolatile int &killSwitch);

    template <class S> int modelaugx_int_mu   (                         Vector<gentype> &resmu, const SparseVector<S> &x) const;
    template <class S> int modelaugx_int_muvar(Vector<gentype> &resvar, Vector<gentype> &resmu, const SparseVector<S> &x) const;

    // Optimiser for model_err calculation
    //
    // JIT allocation, DIRect by default, set src to use alternative

    GlobalOptions &getModelErrOptim(GlobalOptions *src = nullptr) const;
    void killModelErrOptim(void) const;

    // Convert [ x ~ x' ~ ... ] to naive format that doesn't use upsize stuff, if ennornaive > 0

    const SparseVector<gentype> &convnearuptonaive(SparseVector<gentype> &res, const SparseVector<gentype> &x) const;

//private:
    // Variables for env-GP (Joy1,Shi21) and diff-GP (Shi21)
    //
    // indpremu: indices of pre-loaded samples (transfer learning) for env-GP
    // presigweightmu: sigma "stretch" for env-GP
    //
    // Nbasemu: number of base (transfer) variables for env-GP
    // resdiff: helper variable for env-GP
    //
    // alpha,beta: see Joy1, env-GP/diff-GP
    //
    // srcmodel:  for diff-GP, this is used to store the source and model data in "raw form"
    // diffmodel: for diff-GP, this is used to model the difference between source and target models
    //
    // srcmodelInd:  index for srcmodel
    // diffmodelInd: index for diffmodel
    //
    // diffval: used for diff-GP
    // predval: used for diff-GP
    //
    // firsttrain: set 1 by optim, then 0 again by train

    Vector<int> indpremu;
    Vector<double> presigweightmu;

    int Nbasemu;
    gentype resdiff;

    double alpha;
    double beta;

    ML_Mutable *srcmodel;
    ML_Mutable *diffmodel;

    int srcmodelInd;
    int diffmodelInd;

    gentype diffval;
    gentype predval;
    gentype storevar;

    int firsttrain;

    // Some data for short-cut version

    mutable Matrix<gentype> xmodprod; // xmodprod(i,j) is inner products between x(i) and xbasisj
    int xshortcutenabled;             // set 1 if we can do fast calculation of inner product using xmodprod and xbasisprod
    mutable Vector<gentype> xsp;      // vector used to record inner product of xi and suggested vector
    mutable Vector<gentype **> xspp;  // pointery stuff for some reason

    // Models in use

    Vector<ML_Base *> muapprox;
    ML_Base *sigmaapprox;
    Vector<ML_Base *> augxapprox;

    // Sampled mu approximation

    Vector<ML_Base *> muapprox_sample;

    const ML_Base *getmuapprox(int q) const { return ( muapprox_sample.size() && muapprox_sample(q) ) ? muapprox_sample(q) : muapprox(q); }

    // Local store for x vectors

    Vector<int> locires;
    Vector<SparseVector<gentype> > locxres;
    Vector<SparseVector<gentype> > locxresunconv;
    Vector<gentype> locyres;

    // Optimiser for model_err calculation
    //
    // JIT allocation, DIRect by default, set src to use alternative

    mutable GlobalOptions *modelErrOptim;
    mutable int ismodelErrLocal;

    // Other stuff for speed

    int locdim;

    mutable SparseVector<gentype> xxz;   // just use a global here rather than constant calls to constructors and destructors
    mutable SparseVector<gentype> xx1;   // just use a global here rather than constant calls to constructors and destructors
    mutable SparseVector<gentype> xx;    // just use a global here rather than constant calls to constructors and destructors
    mutable SparseVector<gentype> xxvar; // just use a global here rather than constant calls to constructors and destructors

    // Models (moo == multi-objective), fx = model for q (see below), if used

    Vector<ML_Base *> fnapprox;
    Vector<ML_Base *> fxapprox;

    Vector<int> fnapproxInd;
    Vector<int> fxapproxInd;
};


















template <class S>
const SparseVector<gentype> &SMBOOptions::model_convertx(SparseVector<gentype> &res, const SparseVector<S> &x, int useOrigin, int useShortcut) const
{
    if ( ( ( modeltype == 0 ) || ( modeltype == 1 ) ) &&
         ( !ismodelaug() ) &&
         ( ( x.nupsize() == 1 ) || ( ( x.nupsize() == 2 ) && getdimfid() ) ) &&
         ( x.f1indsize() == 0 ) &&
         ( x.f2indsize() == 0 ) &&
         ( x.f4indsize() == 0 )    )
    {
        // Direct conversion if possible
        return GlobalOptions::model_convertx(res,x,useOrigin,useShortcut);
    }

    if ( x.nindsize() )
    {
        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
            res.castassign(x.nup(0));
        }

        else
        {
            res = GlobalOptions::model_convertx(res,x.nup(0),useOrigin,useShortcut);
        }

        if ( x.nupsize() > 1 )
        {
            // aux data supplied, include it

            for ( int i = 1 ; i < x.nupsize() ; ++i )
            {
                int jdim = x.nupsize(i);

                for ( int j = 0 ; j < jdim ; ++j )
                {
                    res.n("&",j,i) = x.n(j,i);
                }
            }
        }

        else if ( ismodelaug() )
        {
            // aux data not supplied, need to fill in with predictions from model

            Vector<gentype> xaddmu;

            modelaugx_int_mu(xaddmu,res);

            for ( int i = 0 ; i < xaddmu.size() ; ++i )
            {
                if ( !(xaddmu(i).isValVector()) )
                {
                    res.n("&",0,i+1) = xaddmu(i);
                }

                else
                {
                    int jdim = xaddmu(i).size();

                    for ( int j = 0 ; j < jdim ; ++j )
                    {
                        res.n("&",j,i+1) = xaddmu(i)(j);
                    }
                }
            }
        }
    }

    if ( x.f1indsize() )
    {
        SparseVector<gentype> farpart;

        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
            farpart.castassign(x.f1up(0));
        }

        else
        {
            farpart = GlobalOptions::model_convertx(farpart,x.f1up(0),useOrigin,useShortcut);
        }

        for ( int i = 0 ; i < farpart.indsize() ; ++i )
        {
            res.f1("&",farpart.ind(i)) = farpart.direcref(i);
        }

        if ( x.f1upsize() > 1 )
        {
            // aux data supplied, include it

            for ( int i = 1 ; i < x.f1upsize() ; ++i )
            {
                int jdim = x.f1upsize(i);

                for ( int j = 0 ; j < jdim ; ++j )
                {
                    res.f1("&",j,i) = x.f1(j,i);
                }
            }
        }

        else if ( ismodelaug() )
        {
            // aux data not supplied, need to fill in with predictions from model

            Vector<gentype> xaddmu;

            modelaugx_int_mu(xaddmu,farpart);

            for ( int i = 0 ; i < xaddmu.size() ; ++i )
            {
                if ( !(xaddmu(i).isValVector()) )
                {
                    res.f1("&",0,i+1) = xaddmu(i);
                }

                else
                {
                    int jdim = xaddmu(i).size();

                    for ( int j = 0 ; j < jdim ; ++j )
                    {
                        res.f1("&",j,i+1) = xaddmu(i)(j);
                    }
                }
            }
        }
    }

    if ( x.f2indsize() )
    {
        SparseVector<gentype> farfarpart;

        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
            farfarpart.castassign(x.f2up(0));
        }

        else
        {
            farfarpart = GlobalOptions::model_convertx(farfarpart,x.f2up(0),useOrigin,useShortcut);
        }

        for ( int i = 0 ; i < farfarpart.indsize() ; ++i )
        {
            res.f2("&",farfarpart.ind(i)) = farfarpart.direcref(i);
        }
    }

    if ( x.f4indsize() )
    {
        SparseVector<gentype> f4part;

        f4part.castassign(x.f4up(0)); // don't want to convert the f4 part!

        for ( int i = 0 ; i < f4part.indsize() ; ++i )
        {
            res.f4("&",f4part.ind(i)) = f4part.direcref(i);
        }
    }

//    if ( getdimfid() )
//    {
//        int zref = res.nupsize(0)-getdimfid();
//
//        for ( int i = getdimfid()-1 ; i >= 0 ; i-- )
//        {
//            res.n("&",i,1) = res.n(zref+i,0);
//            res.zeroni(zref+i);
//        }
//    }

    return res;
}


template <class S>
const SparseVector<gentype> &SMBOOptions::model_convertx(int &isvarnz, SparseVector<gentype> &resvar, SparseVector<gentype> &res, const SparseVector<S> &x, int useOrigin, int useShortcut, int debugit) const
{
    isvarnz = 0;

    if ( debugit )
    {
        errstream() << "phantomxrr 0 x = " << x << "\n";
        errstream() << "phantomxrr 1 res = " << res << "\n";
        errstream() << "phantomxrr 2 resvar = " << resvar << "\n";
    }

    if ( x.nindsize() )
    {
        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
            res.castassign(x.nup(0));
        }

        else
        {
            res = GlobalOptions::model_convertx(res,x.nup(0),useOrigin,useShortcut);
        }

        if ( debugit )
        {
            errstream() << "phantomxrr 10 x = " << x << "\n";
            errstream() << "phantomxrr 11 res = " << res << "\n";
            errstream() << "phantomxrr 12 resvar = " << resvar << "\n";
        }

        if ( x.nupsize() > 1 )
        {
            // aux data supplied, so use that

            //resvar.n("&",0,0) = 0.0;

            for ( int i = 1 ; i < x.nupsize() ; ++i )
            {
                int jdim = x.nupsize(i);

                for ( int j = 0 ; j < jdim ; ++j )
                {
                    res.n("&",j,i)    = x.n(j,i);
                    resvar.n("&",j,i) = 0.0;
                }
            }

            if ( debugit )
            {
                errstream() << "phantomxrr 100 x = " << x << "\n";
                errstream() << "phantomxrr 101 res = " << res << "\n";
                errstream() << "phantomxrr 102 resvar = " << resvar << "\n";
            }
        }

        else if ( ismodelaug() )
        {
            // aux data not supplied, need to fill in with predictions from model
            // Note that we also need to calculate and store the *variance* of this estimate.

            Vector<gentype> xaddmu;
            Vector<gentype> xsidechan;

            modelaugx_int_muvar(xsidechan,xaddmu,res);

            if ( debugit )
            {
                errstream() << "phantomxrr 20 x = " << xsidechan << "\n";
                errstream() << "phantomxrr 21 res = " << xaddmu << "\n";
                errstream() << "phantomxrr 22 resvar = " << resvar << "\n";
            }

            //resvar.n("&",0,0) = 0.0;

            isvarnz = 1; // set flag to indicate that variance is non-zero

            for ( int i = 0 ; i < xaddmu.size() ; ++i )
            {
                if ( !(xaddmu(i).isValVector()) )
                {
                    res.n("&",0,i+1)    = xaddmu(i);
                    resvar.n("&",0,i+1) = xsidechan(i);
                }

                else
                {
                    int jdim = xaddmu(i).size();

                    for ( int j = 0 ; j < jdim ; ++j )
                    {
                        res.n("&",j,i+1)    = xaddmu(i)(j);
                        resvar.n("&",j,i+1) = xsidechan(i)(j);
                    }
                }
            }

            if ( debugit )
            {
                errstream() << "phantomxrr 30 x = " << x << "\n";
                errstream() << "phantomxrr 31 res = " << res << "\n";
                errstream() << "phantomxrr 32 resvar = " << resvar << "\n";
            }
        }

        if ( debugit )
        {
            errstream() << "phantomxrr 440 x = " << x << "\n";
            errstream() << "phantomxrr 441 res = " << res << "\n";
            errstream() << "phantomxrr 442 resvar = " << resvar << "\n";
        }
    }

    if ( x.f1indsize() )
    {
        SparseVector<gentype> farpart;

        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
            farpart.castassign(x.f1up(0));
        }

        else
        {
            farpart = GlobalOptions::model_convertx(farpart,x.f1up(0),useOrigin,useShortcut);
        }

        for ( int i = 0 ; i < farpart.indsize() ; ++i )
        {
            res.f1("&",farpart.ind(i)) = farpart.direcref(i);
        }

        if ( x.f1upsize() > 1 )
        {
            // aux data supplied, so use that

            //resvar.n("&",0,0) = 0.0;

            for ( int i = 1 ; i < x.f1upsize() ; ++i )
            {
                int jdim = x.f1upsize(i);

                for ( int j = 0 ; j < jdim ; ++j )
                {
                    res.f1("&",j,i)    = x.f1(j,i);
                    resvar.f1("&",j,i) = 0.0;
                }
            }

            if ( debugit )
            {
                errstream() << "phantomxrr 100 x = " << x << "\n";
                errstream() << "phantomxrr 101 res = " << res << "\n";
                errstream() << "phantomxrr 102 resvar = " << resvar << "\n";
            }
        }

        else if ( ismodelaug() )
        {
            // aux data not supplied, need to fill in with predictions from model
            // Note that we also need to calculate and store the *variance* of this estimate.

            Vector<gentype> xaddmu;
            Vector<gentype> xsidechan;

            modelaugx_int_muvar(xsidechan,xaddmu,farpart);

            if ( debugit )
            {
                errstream() << "phantomxrr 20 x = " << xsidechan << "\n";
                errstream() << "phantomxrr 21 res = " << xaddmu << "\n";
                errstream() << "phantomxrr 22 resvar = " << resvar << "\n";
            }

            //resvar.f1("&",0,0) = 0.0;

            isvarnz = 1; // set flag to indicate that variance is non-zero

            for ( int i = 0 ; i < xaddmu.size() ; ++i )
            {
                if ( !(xaddmu(i).isValVector()) )
                {
                    res.f1("&",0,i+1)    = xaddmu(i);
                    resvar.f1("&",0,i+1) = xsidechan(i);
                }

                else
                {
                    int jdim = xaddmu(i).size();

                    for ( int j = 0 ; j < jdim ; ++j )
                    {
                        res.f1("&",j,i+1)    = xaddmu(i)(j);
                        resvar.f1("&",j,i+1) = xsidechan(i)(j);
                    }
                }
            }

            if ( debugit )
            {
                errstream() << "phantomxrr 30 x = " << x << "\n";
                errstream() << "phantomxrr 31 res = " << res << "\n";
                errstream() << "phantomxrr 32 resvar = " << resvar << "\n";
            }
        }

        if ( debugit )
        {
            errstream() << "phantomxrr 440 x = " << x << "\n";
            errstream() << "phantomxrr 441 res = " << res << "\n";
            errstream() << "phantomxrr 442 resvar = " << resvar << "\n";
        }
    }

    if ( x.f2indsize() )
    {
        SparseVector<gentype> farfarpart;

        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
            farfarpart.castassign(x.f2up(0));
        }

        else
        {
            farfarpart = GlobalOptions::model_convertx(farfarpart,x.f2up(0),useOrigin,useShortcut);
        }

        for ( int i = 0 ; i < farfarpart.indsize() ; ++i )
        {
            res.f2("&",farfarpart.ind(i)) = farfarpart.direcref(i);
        }
    }

    if ( x.f4indsize() )
    {
        SparseVector<gentype> f4part;

        f4part.castassign(x.f4up(0));

        for ( int i = 0 ; i < f4part.indsize() ; ++i )
        {
            res.f4("&",f4part.ind(i)) = f4part.direcref(i);
        }
    }

//    if ( getdimfid() )
//    {
//        int zref = res.nupsize(0)-getdimfid();
//        int varzref = resvar.nupsize(0)-getdimfid();
//
//        for ( int i = getdimfid()-1 ; i >= 0 ; i-- )
//        {
//            res.n("&",i,1) = res.n(zref+i,0);
//            res.zeroni(zref+i);
//
//            resvar.n("&",i,1) = 0.0;
//            resvar.zeroni(varzref+i);
//        }
//    }

    return res;
}


template <class S>
double SMBOOptions::inf_dist(const SparseVector<S> &xz) const
{
    SparseVector<gentype> locxz; locxz.castassign(xz);
    SparseVector<gentype> locx1(locxz);

    for ( int i = 0 ; i < getdimfid() ; i++ )
    {
        setident(locx1.n("&",i,1));
    }

    //           zeta(z) = sqrt( 1 - (K([x~0],[x~1])/K([x~0],[x~0])) )

//errstream() << "phantomxyz inf_dist calc: xz " << xz << "\n";
//errstream() << "phantomxyz inf_dist calc: locxz " << locxz << "\n";
//errstream() << "phantomxyz inf_dist calc: locx1 " << locx1 << "\n";
    gentype Kxzx1;
    gentype Kxx;

    (*getmuapprox(0)).K2(Kxzx1,locxz,locx1); // kappa0.phi_z(||z-1||).phi_x(0) = kappa0.phi_z(||z-1||)
    (*getmuapprox(0)).K2(Kxx,  locxz,locxz); // kappa0.phi_z(0).phi_x(0) = kappa0
//errstream() << "phantomxyz inf_dist calc: Kxzx1 " << Kxzx1 << "\n";
//errstream() << "phantomxyz inf_dist calc: Kxx " << Kxx << "\n";

    double Kxzx1d = (double) Kxzx1;
    double Kxxd   = (double) Kxx;

    double res = 1-((Kxzx1d/Kxxd)*(Kxzx1d/Kxxd));
    res = ( res > 0 ) ? sqrt(res) : 0.0;
//errstream() << "phantomxyz inf_dist calc:res " << res << "\n";

    return res;
}

template <class S>
int SMBOOptions::model_mu(gentype &resg, const SparseVector<S> &x, const vecInfo *xing) const
{
        (void) xing;

        gentype ***pxyprodx = nullptr;
        vecInfo *xinf = nullptr;
        vecInfo xinfloc;
        gentype xxp;

        const SparseVector<gentype> *xxx = &xx;

        if ( !model_issample() && xshortcutenabled )
        {
            xxx = &model_convertx(xx,x,0,1);

            if ( !getxpweightIsWeight() )
            {
                goto bailout;
            }

            NiceAssert( getxpweight().size() == getxbasis().size() );

            int N = (*(muapprox(0))).N();

            xsp.resize(N);

            int i,j;

            while ( xmodprod.numRows() < N )
            {
                i = xmodprod.numRows();

                xmodprod.addRow(i);

                for ( j = 0 ; j < getxbasis().size() ; ++j )
                {
                    innerProduct(xmodprod("&",i,j),(*(muapprox(0))).x(i),getxbasis()(j));
                }
            }

            retVector<gentype> tmpva;
            retVector<gentype> tmpvb;

            for ( i = 0 ; i < N ; ++i )
            {
                innerProduct(xsp("&",i),xmodprod(i,tmpva,tmpvb),getxpweight());
            }

            while ( xspp.size() > N )
            {
                i = xspp.size()-1;

                MEMDELARRAY(xspp("&",i));
                xspp.remove(i);
            }

            while ( xspp.size() < N )
            {
                i = xspp.size();

                xspp.add(i);
                MEMNEWARRAY(xspp("&",i),gentype *,2);
            }

            for ( i = 0 ; i < N ; ++i )
            {
                xspp("&",i)[0] = &(xsp("&",i));
                xspp("&",i)[1] = nullptr;
            }

            pxyprodx = N ? &(xspp("&",0)) : nullptr;

            for ( i = 0 ; i < getxbasis().size() ; ++i )
            {
                for ( j = 0 ; j < getxbasis().size() ; ++j )
                {
                    xxp += getxpweight()(i)*getxpweight()(j)*getxbasisprod()(i,j);
                }
            }

            xinf = &((*(muapprox(0))).getKernel().getvecInfo(xinfloc,*xxx,&xxp)); // Can't calculate the inner-product of vectors that aren't actually formed!
        }

        else
        {
bailout:
            xx.zeronotnu(0);

            xxx = &model_convertx(xx,x);
        }

        (*xxx).makealtcontent();

        int i,ires = 0;

        {
            if ( !muapprox.size() )
            {
                resg.force_null();
            }

            else if ( muapprox.size() == 1 )
            {
                if ( ennornaive )
                {
                    SparseVector<gentype> tempx;

                    ires += (*getmuapprox(0)).gg(resg,convnearuptonaive(tempx,*xxx));
                }

                else
                {
                    ires += (*getmuapprox(0)).gg(resg,*xxx,xinf,pxyprodx);
                }
            }

            else
            {
                Vector<gentype> &resgvec = resg.force_vector(muapprox.size());

                for ( i = 0 ; i < muapprox.size() ; ++i )
                {
                    if ( ennornaive )
                    {
                        SparseVector<gentype> tempx;

                        ires += (*getmuapprox(i)).gg(resgvec("&",i),convnearuptonaive(tempx,*xxx));
                    }

                    else
                    {
                        ires += (*getmuapprox(i)).gg(resgvec("&",i),*xxx,xinf,pxyprodx);
                    }
                }
            }
        }

        return ires;
}

template <class S>
int SMBOOptions::model_mu(Vector<double> &resg, const SparseVector<S> &x, const vecInfo *xing) const
{
        (void) xing;

        gentype ***pxyprodx = nullptr;
        vecInfo *xinf = nullptr;
        vecInfo xinfloc;
        gentype xxp;
        const SparseVector<gentype> *xxx = &xx;

        if ( !model_issample() && xshortcutenabled )
        {
            xxx = &model_convertx(xx,x,0,1);

            if ( !getxpweightIsWeight() )
            {
                goto bailout;
            }

            NiceAssert( getxpweight().size() == getxbasis().size() );

            int N = (*(muapprox(0))).N();

            xsp.resize(N);

            int i,j;

            while ( xmodprod.numRows() < N )
            {
                i = xmodprod.numRows();

                xmodprod.addRow(i);

                for ( j = 0 ; j < getxbasis().size() ; ++j )
                {
                    innerProduct(xmodprod("&",i,j),(*(muapprox(0))).x(i),getxbasis()(j));
                }
            }

            retVector<gentype> tmpva;
            retVector<gentype> tmpvb;

            for ( i = 0 ; i < N ; ++i )
            {
                innerProduct(xsp("&",i),xmodprod(i,tmpva,tmpvb),getxpweight());
            }

            while ( xspp.size() > N )
            {
                i = xspp.size()-1;

                MEMDELARRAY(xspp("&",i));
                xspp.remove(i);
            }

            while ( xspp.size() < N )
            {
                i = xspp.size();

                xspp.add(i);
                MEMNEWARRAY(xspp("&",i),gentype *,2);
            }

            for ( i = 0 ; i < N ; ++i )
            {
                xspp("&",i)[0] = &(xsp("&",i));
                xspp("&",i)[1] = nullptr;
            }

            pxyprodx = N ? &(xspp("&",0)) : nullptr;

            for ( i = 0 ; i < getxbasis().size() ; ++i )
            {
                for ( j = 0 ; j < getxbasis().size() ; ++j )
                {
                    xxp += getxpweight()(i)*getxpweight()(j)*getxbasisprod()(i,j);
                }
            }

            xinf = &((*(muapprox(0))).getKernel().getvecInfo(xinfloc,*xxx,&xxp)); // Can't calculate the inner-product of vectors that aren't actually formed!
        }

        else
        {
bailout:
            xx.zeronotnu(0);

            xxx = &model_convertx(xx,x);
        }

        (*xxx).makealtcontent();

        int i,ires = 0;

        {
            if ( !muapprox.size() )
            {
                resg.resize(0);
            }

            else if ( muapprox.size() == 1 )
            {
                resg.resize(1);

                if ( ennornaive )
                {
                    SparseVector<gentype> tempx;

                    ires += (*getmuapprox(0)).gg(resg("&",0),convnearuptonaive(tempx,*xxx));
                }

                else
                {
                    ires += (*getmuapprox(0)).gg(resg("&",0),*xxx,0,xinf,pxyprodx);
                }
            }

            else
            {
                resg.resize(muapprox.size());

                for ( i = 0 ; i < muapprox.size() ; ++i )
                {
                    if ( ennornaive )
                    {
                        SparseVector<gentype> tempx;

                        ires += (*getmuapprox(i)).gg(resg,convnearuptonaive(tempx,*xxx));
                    }

                    else
                    {
                        ires += (*getmuapprox(i)).gg(resg,*xxx,0,xinf,pxyprodx);
                    }
                }
            }
        }

        return ires;
    }

template <class S>
int SMBOOptions::model_muvar(gentype &resv, gentype &resmu, const SparseVector<S> &x, const vecInfo *xing, int debugit) const
{
        (void) xing;

        int isvarnz = 0;

        if ( debugit )
        {
            errstream() << "phantomxqq 0\n";
        }

        gentype ***pxyprodx = nullptr;
        gentype **pxyprodxx = nullptr;
        vecInfo *xinf = nullptr;
        vecInfo xinfloc;
        gentype xxp;
        const SparseVector<gentype> *xxx = &xx;

        if ( !model_issample() && xshortcutenabled )
        {
            xxx = &model_convertx(xx,x,0,1);

            if ( !getxpweightIsWeight() )
            {
                goto bailout;
            }

            NiceAssert( getxpweight().size() == getxbasis().size() );

            int N = (*(muapprox(0))).N();

            xsp.resize(N);

            int i,j;

            while ( xmodprod.numRows() < N )
            {
                i = xmodprod.numRows();

                xmodprod.addRow(i);

                for ( j = 0 ; j < getxbasis().size() ; ++j )
                {
                    innerProduct(xmodprod("&",i,j),(*(muapprox(0))).x(i),getxbasis()(j));
                }
            }

            retVector<gentype> tmpva;
            retVector<gentype> tmpvb;

            for ( i = 0 ; i < N ; ++i )
            {
                innerProduct(xsp("&",i),xmodprod(i,tmpva,tmpvb),getxpweight());
            }

            while ( xspp.size() > N )
            {
                i = xspp.size()-1;

                MEMDELARRAY(xspp("&",i));
                xspp.remove(i);
            }

            while ( xspp.size() < N )
            {
                i = xspp.size();

                xspp.add(i);
                MEMNEWARRAY(xspp("&",i),gentype *,2);
            }

            for ( i = 0 ; i < N ; ++i )
            {
                xspp("&",i)[0] = &(xsp("&",i));
                xspp("&",i)[1] = nullptr;
            }

            pxyprodx = N ? &(xspp("&",0)) : nullptr;

            for ( i = 0 ; i < getxbasis().size() ; ++i )
            {
                for ( j = 0 ; j < getxbasis().size() ; ++j )
                {
                    xxp += getxpweight()(i)*getxpweight()(j)*getxbasisprod()(i,j);
                }
            }

            xinf = &((*(muapprox(0))).getKernel().getvecInfo(xinfloc,*xxx,&xxp)); // Can't calculate the inner-product of vectors that aren't actually formed!

            MEMNEWARRAY(pxyprodxx,gentype *,2);
            pxyprodxx[0] = &xxp;
            pxyprodxx[1] = nullptr;
        }

        else
        {
bailout:
            if ( debugit )
            {
                errstream() << "phantomxqq 14: " << x << "\n";
            }

            if ( ismodelaug() )
            {
                xx.zeronotnu(0);
                xxvar.zero();

                xxx = &model_convertx(isvarnz,xxvar,xx,x,0,0,debugit);

                if ( debugit )
                {
                    errstream() << "phantomxyzabc 0 isvarnz = " << isvarnz << "\n";
                    errstream() << "phantomxyzabc 0 xxvar = " << xxvar << "\n";
                    errstream() << "phantomxyzabc 0 xx = " << xx << "\n";
                    errstream() << "phantomxyzabc 0 x = " << x << "\n";
                }
            }

            else
            {
                xx.zeronotnu(0);

                xxx = &model_convertx(xx,x);
            }
        }

        int i,ires = 0;

        (*xxx).makealtcontent();

        if ( !sigmuseparate )
        {
            if ( isvarnz && !ismodelnaive() )
            {
                if ( debugit )
                {
                    errstream() << "phantomxqq 17 !isvarnz\n";

                    {
                        if ( !muapprox.size() )
                        {
                            resv.force_null();
                            resmu.force_null();
                        }

                        else if ( muapprox.size() == 1 )
                        {
                            if ( ennornaive )
                            {
                                SparseVector<gentype> tempx;

                                ires += (*getmuapprox(0)).var(resv,resmu,convnearuptonaive(tempx,*xxx));
                            }

                            else
                            {
                                ires += (*getmuapprox(0)).var(resv,resmu,*xxx,xinf,pxyprodx,pxyprodxx);
                            }
                        }

                        else
                        {
                            Vector<gentype> &resvvec = resv.force_vector(muapprox.size());
                            Vector<gentype> &resmuvec = resmu.force_vector(muapprox.size());

                            for ( i = 0 ; i < muapprox.size() ; ++i )
                            {
                                if ( ennornaive )
                                {
                                    SparseVector<gentype> tempx;

                                    ires += (*getmuapprox(i)).var(resvvec("&",i),resmuvec("&",i),convnearuptonaive(tempx,*xxx));
                                }

                                else
                                {
                                    ires += (*getmuapprox(i)).var(resvvec("&",i),resmuvec("&",i),*xxx,xinf,pxyprodx,pxyprodxx);
                                }
                            }
                        }
                    }

                    errstream() << "phantomxyzabc 1 naive resv = " << resv << "\n";
                    errstream() << "phantomxyzabc 2 naive resmu = " << resmu << "\n";
                }

                {
                    if ( !muapprox.size() )
                    {
                        resv.force_null();
                        resmu.force_null();
                    }

                    else if ( muapprox.size() == 1 )
                    {
                        if ( ennornaive )
                        {
                            SparseVector<gentype> tempx,tempv;

                            ires += (*getmuapprox(0)).noisevar(resv,resmu,convnearuptonaive(tempx,*xxx),convnearuptonaive(xxvar,tempv),-1);
                        }

                        else
                        {
                            ires += (*getmuapprox(0)).noisevar(resv,resmu,*xxx,xxvar,-2,xinf,pxyprodx,pxyprodxx);
                        }
                    }

                    else
                    {
                        Vector<gentype> &resvvec = resv.force_vector(muapprox.size());
                        Vector<gentype> &resmuvec = resmu.force_vector(muapprox.size());

                        for ( i = 0 ; i < muapprox.size() ; ++i )
                        {
                            if ( ennornaive )
                            {
                                SparseVector<gentype> tempx,tempv;

                                ires += (*getmuapprox(i)).noisevar(resvvec("&",i),resmuvec("&",i),convnearuptonaive(tempx,*xxx),convnearuptonaive(xxvar,tempv),-1);
                            }

                            else
                            {
                                ires += (*getmuapprox(i)).noisevar(resvvec("&",i),resmuvec("&",i),*xxx,xxvar,02,xinf,pxyprodx,pxyprodxx);
                            }
                        }
                    }
                }
            }

            else
            {
                if ( debugit )
                {
                    errstream() << "phantomxqq 17 isvarnz\n";
                }

                {
                    if ( !muapprox.size() )
                    {
                        resv.force_null();
                        resmu.force_null();
                    }

                    else if ( muapprox.size() == 1 )
                    {
                        if ( ennornaive )
                        {
                            SparseVector<gentype> tempx;

                            ires += (*getmuapprox(0)).var(resv,resmu,convnearuptonaive(tempx,*xxx));
                        }

                        else
                        {
                            ires += (*getmuapprox(0)).var(resv,resmu,*xxx,xinf,pxyprodx,pxyprodxx);
                        }
                    }

                    else
                    {
                        Vector<gentype> &resvvec = resv.force_vector(muapprox.size());
                        Vector<gentype> &resmuvec = resmu.force_vector(muapprox.size());

                        for ( i = 0 ; i < muapprox.size() ; ++i )
                        {
                            if ( ennornaive )
                            {
                                SparseVector<gentype> tempx;

                                ires += (*getmuapprox(i)).var(resvvec("&",i),resmuvec("&",i),convnearuptonaive(tempx,*xxx));
                            }

                            else
                            {
                                ires += (*getmuapprox(i)).var(resvvec("&",i),resmuvec("&",i),*xxx,xinf,pxyprodx,pxyprodxx);
                            }
                        }
                    }
                }
            }

            if ( debugit )
            {
                errstream() << "phantomxyzabc 1 resv = " << resv << "\n";
                errstream() << "phantomxyzabc 2 resmu = " << resmu << "\n";
                errstream() << "phantomxqq 18\n";
            }
        }

        else
        {
            if ( debugit )
            {
                errstream() << "phantomxqq 19\n";
            }

            gentype dummy;

            {
                if ( !muapprox.size() )
                {
                    resv.force_null();
                    resmu.force_null();
                }

                else if ( muapprox.size() == 1 )
                {
                    resv.zero();

                    if ( ennornaive )
                    {
                        SparseVector<gentype> tempx;

                        ires += (*getmuapprox(0)).gg(resmu,convnearuptonaive(tempx,*xxx));
                    }

                    else
                    {
                        ires += (*getmuapprox(0)).gg(resmu,*xxx,xinf,pxyprodx);
                    }
                }

                else
                {
                    resv.force_vector(muapprox.size()).zero();
                    Vector<gentype> &resmuvec = resmu.force_vector(muapprox.size());

                    for ( i = 0 ; i < muapprox.size() ; ++i )
                    {
                        if ( ennornaive )
                        {
                            SparseVector<gentype> tempx;

                            ires += (*getmuapprox(i)).gg(resmuvec("&",i),convnearuptonaive(tempx,*xxx));
                        }

                        else
                        {
                            ires += (*getmuapprox(i)).gg(resmuvec("&",i),*xxx,xinf,pxyprodx);
                        }
                    }
                }
            }


//isvarnz = 0;
            if ( isvarnz && !ismodelnaive() )
            {
                gentype resscalarv;

                if ( ennornaive )
                {
                    SparseVector<gentype> tempx,tempv;

                    ires |= (*sigmaapprox).noisevar(resscalarv,dummy,convnearuptonaive(tempx,*xxx),convnearuptonaive(tempv,xxvar),-1);
                }

                else
                {
                    ires |= (*sigmaapprox).noisevar(resscalarv,dummy,*xxx,xxvar,-2,xinf,pxyprodx,pxyprodxx);
                }

                if ( !muapprox.size() )
                {
                    resv.force_null();
                }

                else if ( muapprox.size() == 1 )
                {
                    resv = resscalarv;
                }

                else
                {
                    resv.force_vector(muapprox.size()) = resscalarv;
                }
            }

            else
            {
                gentype resscalarv;

                if ( ennornaive )
                {
                    SparseVector<gentype> tempx;

                    ires |= (*sigmaapprox).var(resscalarv,dummy,convnearuptonaive(tempx,*xxx));
                }

                else
                {
                    ires |= (*sigmaapprox).var(resscalarv,dummy,*xxx,xinf,pxyprodx,pxyprodxx);
                }

                if ( !muapprox.size() )
                {
                    resv.force_null();
                }

                else if ( muapprox.size() == 1 )
                {
                    resv = resscalarv;
                }

                else
                {
                    resv.force_vector(muapprox.size()) = resscalarv;
                }
            }
        }

        if ( pxyprodxx )
        {
            MEMDELARRAY(pxyprodxx);
        }

        return ires;
    }

template <class S>
int SMBOOptions::model_covar(Matrix<gentype> &resv, const Vector<SparseVector<S> > &x) const
{
    Vector<SparseVector<gentype> > xxx;

    model_convertx(xxx,x);

    return (*sigmaapprox).covar(resv,xxx);
}

template <class S>
void SMBOOptions::model_stabProb(double &res, const SparseVector<S> &x, int p, double pnrm, int rot, double mu, double B) const
{
    SparseVector<gentype> tempx;

    (*(muapprox(0))).stabProb(res,convnearuptonaive(tempx,model_convertx(xx,x)),p,pnrm,rot,mu,B);
}


template <class S>
int SMBOOptions::modelaugx_int_mu(Vector<gentype> &resmu, const SparseVector<S> &x) const
{
    int i,ires = 0;

    resmu.resize(augxapprox.size());

    for ( i = 0 ; i < augxapprox.size() ; ++i )
    {
        ires |= (*(augxapprox(i))).gg(resmu("&",i),x);
    }

    return ires;
}

template <class S>
int SMBOOptions::modelaugx_int_muvar(Vector<gentype> &resvar, Vector<gentype> &resmu, const SparseVector<S> &x) const
{
    int ires = 0;
    int i;

    resvar.resize(augxapprox.size());
    resmu.resize(augxapprox.size());

    for ( i = 0 ; i < augxapprox.size() ; ++i )
    {
        ires |= (*(augxapprox(i))).var(resvar("&",i),resmu("&",i),x);
    }

    return ires;
}











#endif



