//FIXME IMPLEMENT FIDELITY OVER-RIDE



//RKHSFIXME: mlinter: isProjection == 5
//           reProjections sets number of restarts per round


// see FIXMEFIX - see also smboopt.h

//FIXME: totiters == -2 means smart-stop.  In smboopt you need: model_err() = ( min_x f(x) + sigma(x) ) - ( min_x f(x) - sigma(x) ).  This can use directopt to calculate.
//       Then use model_err() <= eps and totiters == -2 then we exit at that point.
//FIXME: ALMOST DONE! in bayesopt.cc, have == -2 condition, just need to put target on this and exit when target reached!

//FIXME: option to tune weight on current best?

//
// Global optimisation options base-class
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ml_base.hpp"
#include "gpr_scalar.hpp"
#include "gpr_vector.hpp"
#include "ml_mutable.hpp"
#include "errortest.hpp"
#include "FNVector.hpp"
#include "mlcommon.hpp"
#include "blk_usrfnb.hpp"
#include "blk_bernst.hpp"
#include "blk_conect.hpp"
#include <cmath>

#ifndef _globalopt_h
#define _globalopt_h


bool isisValNone(const gentype &val);
bool isisValNone(const double  &val);


#define DEFAULT_NUMOPTS 10
#define DEFAULT_LOGZTOL 1e-8

// Near zero, near inf values for distMode 4 in global optimiser

#define NEARZEROVAL 0.1
#define NEARINFVAL  1e6
#define RATIOMAX    1e-15

// ML registration index

#define DEFAULT_MLREGIND 1024


void overfn(gentype &res, Vector<gentype> &x, void *arg);
void hhyperfn(gentype &res, Vector<gentype> &x, void *arg);

// returns maximum width of x

double calcabc(int dim,
               Vector<gentype> &fakexmin, Vector<gentype> &fakexmax,
               Vector<double> &a, Vector<double> &b, Vector<double> &c,
               const Vector<gentype> &xmin, const Vector<gentype> &xmax,
               const Vector<int> &distMode, const Vector<int> &varsType,
               bool &ismodel_convertx_simple);


class GridOptions;
class SMBOOptions;

class GlobalOptions
{
    //friend class SMBOOptions;

    friend void overfn(gentype &res, Vector<gentype> &x, void *arg);
    friend void hhyperfn(gentype &res, Vector<gentype> &x, void *arg);

public:

    bool ispydirect; // true if called directly from python (so don't muck around with intermediate callbacks etc)

    // Optname used for logging only

    std::string optname;

    // simname: simulation base-name.  If you plot regret this is used
    // simoutformat: model output format (0 terminal, 1 ps, 2 pdf)
    // simfreq: 0 don't plot regret
    //          1 plot regret
    // simFmin: on regret plot, start of x (iteration/budget) range (default 1, simFmin > simFmax means auto)
    // simFmax: on regret plot, end of x (iteration/budget) range (default 0)
    // simRmin: on regret plot, start of y (regret) range (default 1, simRmin > simRmax means auto)
    // simRmax: on regret plot, end of y (regret) range (default 0)

    std::string simname;      // "regret"
    int         simoutformat; // 2
    int         simfreq;      // 1
    double      simFmin;      // 1
    double      simFmax;      // 0
    double      simRmin;      // 1
    double      simRmax;      // 0

    // recursive allxres

    void *altallxres;

    // maxtraintime: maximum training time (sec).  0 for unlimited.  Note
    //           that this is not precise - for example in bayesian optim
    //           this applies to both the inner (DIRect) and outer loops,
    //           so actual upper bound could be >2x this number.
    //
    // softmin: min value of objective function, used as a reference point
    // softmax: max value of objective function, used as a reference point
    // hardmin: min value of objective function, terminate if f(x) <= hardmin.
    // hardmax: max value of objective function, terminate if f(x) >= hardmax.

    double maxtraintime; // default 0

    double softmin; // default -inf // 0
    double softmax; // default +inf // 1
    double hardmin; // default -inf
    double hardmax; // default +inf

    // Penalty term: penterm(p(x)) is added to the evaluation

    gentype penterm; // default 0.0

    // if isProjection set then x = projOp.g(x), where projOp is a weighted
    // combination of random directions specified by subDef, which are a vector
    // of dim versions of randDirtemplate, where the user must set the distribution
    // in randDir.  By default priorrandDirtemplateVec is used, which corresponds
    // to finite dimensional random directions, but you can change this to 
    // priorrandDirtemplateFnGP to specify function drawn from a GPR.
    //
    // isProjection  - 0 for normal (no fancy projection stuff)
    //                 1 for vector
    //                 2 for projection to function via draws from a GP
    //                 3 for projection to function via Bernstein polynomials
    //                 4 for projection to function via Bernstein polynomials with complexity schedule
    //                 5 for projection to function via RKHSVector
    // includeConst  - 0 for normal (no constant terms in projection)
    //                 1 means that the final term in the projection is a constant.
    // whatConst     - constant included under includeConst
    // randReproject - 0 for normal
    //                 1 for new random projection after each actual evaluation
    // useScalarFn   - 0 treat functional results as distributions
    //                 1 treat functional results as scalar functions (default).  Target must be 1-d
    // xxSampType    - see mlinter
    // xNsamp        - number of samples for approximate integration for functional approx
    // xSampSplit    - if > 1 then sample as m-kernel, where m = xSampSplit
    // xSampType     - 0 normal
    //                 1 positive (definite) by spectal cut
    //                 2 positive (definite) by spectal flip
    //                 3 negative (definite) by spectal cut
    //                 4 negative (definite) by spectal flip
    //                 5 symmetric
    // fnDim         - if we're trying to find an optimal function then this is the
    //                 dimension of the function.  Variables are x,y,z,... - that is, 
    //                 in order of increasing fnDim, f(), f(x), f(x,y), f(x,y,z) ...
    // bernstart     - for scheduled Bernstein projection (isProjection == 4), this is the
    //                 initial polynomial order.  The maximum is min(dim,bernStart+randReproject)

    int    isProjection;  // default 0
    int    includeConst;  // default 0
    double whatConst;     // default 2.0
    int    randReproject; // default 0
    int    useScalarFn;   // default 1
    int    xxSampType;    // default 3
    int    xNsamp;        // default DEFAULT_SAMPLES_SAMPLE
    int    xSampSplit;    // default 1
    int    xSampType;     // default 0
    int    fnDim;         // default 1
    int    bernstart;     // default 1

    BLK_UsrFnB priorrandDirtemplateVec;
    ML_Mutable priorrandDirtemplateFnGP;   // GPR_Scalar
    BLK_Bernst priorrandDirtemplateFnBern;
    RKHSVector priorrandDirtemplateFnRKHS;

    //BLK_UsrFnB priorrandDirtemplateVec;
    //GPR_Scalar priorrandDirtemplateFnGP;
    //BLK_Bernst priorrandDirtemplateFnBern;
    //RKHSVector priorrandDirtemplateFnRKHS;

    // ML registration stuff (for functional optimisation)
    //
    // MLregfn:   function to register ML.  ind is suggested, actual assigned
    //            index is returned.

    int (*MLregfn)(int ind, ML_Mutable *MLcase, void *fnarg);

    // Constructors and assignment operators

    GlobalOptions();
    GlobalOptions(const GlobalOptions &src);
    // It is important that this not be made virtual!
    GlobalOptions &operator=(const GlobalOptions &src);

    // Reset function so that the next simulation can run

    virtual void reset(void);

    // Generate a copy of the relevant optimisation class.

//    virtual GlobalOptions *makeDup(void) const;

    // virtual Destructor to get rid of annoying warnings

    virtual ~GlobalOptions();
    void delstuff(void);

    // Optimisation function stubs
    //
    // dim: problem dimension.
    // xres: x result.
    // Xres: x result (raw, unprocessed format).
    // fres: f(x) result.
    // cres: c(x) result
    // ires: index of result.
    // Fres; budget point of result.
    // mInd: for functional optimisation, this returns the (registered) index of the ML model found
    // allxres: all x results.
    // allfres: all f(x) results.
    // allcres: all c(x) results.
    // allmres: all f(x) results, modified (eg scaled by probability of
    //     feasibility etc, hypervolume etc - what you should be judging
    //     performance on).
    // allsres: suplementary results (timing etc - see specific method)
    // s_score: stability score of x (default 1)
    // xmin: lower bound on x.
    // xmax: upper bound on x.
    // distMode: method of forming each axis.
    //     0 = linear distribution
    //     1 = logarithmic distribution
    //     2 = anti-logarithmic distribution
    //     3 = random distribution (grid only)
    //     4 = inverse logistic distribution of points
    //     5 = REMBO
    // varsType: (grid only) variable types.
    //     0 = integer
    //     1 = real.
    // fn: callback for function being evaluated.
    // fnarg: arguments for fn.
    // killSwitch: usually zero, set 1 to force early exit.
    // numReps: number of repeats
    //
    // If numReps > 1 then the results are from the final run, and the
    // following variables reflect statistics
    //
    // meanfres, varfres: mean and variance of fres
    // meanires, varires: mean and variance of time to best result
    // meantres, vartres: mean and variance of time to softmin
    // meanTres, varTres: mean and variance of time to hardmin
    // meanallfres, varallfres: mean and variance of allfres
    // meanallmres, varallmres: mean and variance of allfres

//FIXME: add tres vector that is calculated based on either iteration (default) or budget (if used)
    virtual int optim(int dim,
                      Vector<gentype> &Xres,
                      gentype         &fres,
                      Vector<gentype> &cres,
                      gentype         &Fres,
                      gentype         &mres,
                      gentype         &sres,
                      int             &ires,
                      int             &mInd,
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
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &)
    {
        (void) dim;
        (void) Xres;
        (void) fres;
        (void) cres;
        (void) Fres;
        (void) mres;
        (void) sres;
        (void) ires;
        (void) mInd;
        (void) allXres;
        (void) allfres;
        (void) allcres;
        (void) allFres;
        (void) allmres;
        (void) allsres;
        (void) s_score;
        (void) is_feas;
        (void) xmin;
        (void) xmax;
        (void) fn;
        (void) fnarg;

        NiceThrow("You should not be here - it's just a stub!");

        return -42;
    }

    gentype &findfirstLT(gentype &res, const Vector<gentype> &src, double target) const
    {
        int k;
        int nores = 1;

        res = valpinf();

        for ( k = 0 ; k < src.size() ; ++k )
        {
            if ( !src(k).isValNone() && src(k).isCastableToRealWithoutLoss() && ( (double) src(k) <= target ) )
            {
                res = k+1;
                nores = 0;
                break;
            }
        }

        // Observation: BO can get stuck.  Even BO with built-in exploration (like GP-UCB) can
        //              get stuck.  It's obvious when you see it, but means that "time to target"
        //              invariably skews towards infinity.  This is uninformative.  Hence the
        //              following "inf -> bignum" hack

        if ( nores )
        {
            res = src.size();
        }

        return res;
    }

    template <class T>
    void calcmeanvar(T &meanres, T &varires, const Vector<T> &src)
    {
        T srcsum;
        T srcsqsum;

        sum(  srcsum,  src);
        sqsum(srcsqsum,src);

        int totReps = src.size();
        int numReps = 0;

        for ( int j = 0 ; j < totReps ; ++j )
        {
            if ( !isisValNone(src(j)) )
            {
                numReps++;
            }
        }

        if ( numReps > 1 )
        {
            meanres = srcsum;
            meanres /= ((double) numReps);

            varires =  srcsum;
            varires *= srcsum;
            varires /= ((double) numReps);
            varires -= srcsqsum;
            varires /= -((double) numReps);

            // Actually we want the variance of the sample mean here, so...
            varires /= ((double) numReps);
//            varires *= (((double) totReps)/((double) numReps));
        }

        else if ( numReps == 1 )
        {
            meanres = srcsum;
            varires = nullgentype();
        }

        else
        {
            meanres = nullgentype();
            varires = nullgentype();
        }
    }

    template <class T>
    void calcmeanvar(Vector<T> &meanres, Vector<T> &varires, const Vector<Vector<T> > &src)
    {
        meanres.resize(src(0).size());
        varires.resize(src(0).size());

        Vector<T> altsrc(src.size());

        for ( int i = 0 ; i < meanres.size() ; ++i )
        {
            for ( int j = 0 ; j < src.size() ; ++j )
            {
                altsrc("&",j) = src(j)(i);
            }

            calcmeanvar(meanres("&",i),varires("&",i),altsrc);
        }
    }

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
                      Vector<gentype> &meanallmres, Vector<gentype> &varallmres);

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
        Vector<Vector<Vector<gentype> > > vecallxres;
        Vector<Vector<Vector<gentype> > > vecallXres;
        Vector<Vector<gentype> >          vecallfres;
        Vector<Vector<Vector<gentype> > > vecallcres;
        Vector<Vector<gentype> >          vecallFres;
        Vector<Vector<gentype> >          vecallmres;
        Vector<Vector<gentype> >          vecallsres;
        Vector<Vector<double> >           vecs_score;
        Vector<Vector<int> >              vecis_feas;

        int retcode = optim(dim,
                      xres,Xres,fres,cres,Fres,mres,sres,ires,mInd,
                      vecallxres,vecallXres,vecallfres,vecallcres,vecallFres,vecallmres,vecallsres,vecs_score,vecis_feas,
                      xmin,xmax,distMode,varsType,fn,fnarg,killSwitch,numReps,
                      meanfres,varfres,meanFres,varFres,meanires,varires,meantres,vartres,meanTres,varTres,
                      meanallfres,varallfres,meanallFres,varallFres,meanallmres,varallmres);

        allxres = vecallxres(0);
        allXres = vecallXres(0);
        allfres = vecallfres(0);
        allcres = vecallcres(0);
        allFres = vecallFres(0);
        allmres = vecallmres(0);
        allsres = vecallsres(0);
        s_score = vecs_score(0);
        is_feas = vecis_feas(0);

        return retcode;
    }

    virtual int realOptim(int dim,
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
                      svmvolatile int &killSwitch);

    // Overload to set sample distributions in models (if any)

    virtual int initModelDistr(const Vector<int> &sampleInd, const Vector<gentype> &sampleDist) { (void) sampleInd; (void) sampleDist; return 0; }

    // Make random subspace

    int makeSubspace(int dim, void *fnarg, int substatus, const SparseVector<gentype> &locxres);

    // Results analysis.  In general the optim function will return one
    // result only, or none if this is a multi-objective bayesian optimisation
    // and/or there are not optima.  The following function goes through the
    // list of all results and returns the set of minima (or the Pareto set
    // if this is a vector optimisation problem).  Returns the number of
    // points in optimal set and sets optres index vector that contains the
    // indices of the Pareto set.
    //
    // hypervol: set to the hypervolume (or just best result so far).  This
    //           is actually based on -allfres in the scalar case, hypervolume
    //           dominated in the negative quadrant otherwise.
    // fdecreasing: min f up to iteration.

    int analyse(const Vector<gentype> &allmres,
                const Vector<int> &is_feas,
                Vector<double> &hypervol,
                Vector<int> &parind,
                int calchypervol) const; // = 1) const;

    // Internal function to differentiate between the base class and the
    // actual optimisation problem classes.  Will return zero for base
    // class, unique index for various global optimisers.

    virtual int optdefed(void)
    {
        return 0;
    }

    // Return 1 if x conversion is non-trivial

    int isXconvertNonTrivial(void) const
    {
        return isProjection || sum(locdistMode);
    }

    // model_clear: clear data from model if relevant.
    // model_update: update any pre-calculated inner-products with model vectors (only called if relevant)
    // stage: 0 pre-optimisation, 1 mid optimisation, 2 post optimisation

    virtual void model_clear(void) { }
    virtual void model_update(void) { }

    virtual void model_log(int stage, double xmin = 0, double xmax = 1, double ymin = 0, double ymax = 1) { (void) stage; (void) xmin; (void) xmax; (void) ymin; (void) ymax; return; }

    // Direction oracle: this one is a simple random selection oracle.
    // Specialise this to implement more specialised oracles (direction etc).
    // See eg smboopt
    //
    // isFirstAxis: the first axis may be treated differently than the rest in some cases

    virtual void consultTheOracle(ML_Mutable &randDir, int dim, const SparseVector<gentype> &locxres, int isFirstAxis)
    {
        (void) dim;
        (void) locxres;
        (void) isFirstAxis;

//FIXMEFIXME fnDim
        errstream() << "Oracle combining... ";

        Vector<gentype> xmin(fnDim);
        Vector<gentype> xmax(fnDim);

        gentype buffer;

        xmin = ( buffer = 0.0 );
        xmax = ( buffer = 1.0 );

        randDir.setSampleMode(1,xmin,xmax,xNsamp,xSampSplit,xSampType,xxSampType,1.0);
    }

    // Convert x to p(x) (we want to optimise f(p(x)))
    //
    // useShortcut: 0 usual - calculate res
    //              1 only calculate xpweight , not res (if possible: otherwise calculate res)
    //              2 calculate both xpweight and res (where possible)

    template <class S>
    const SparseVector<gentype> &model_convertx(SparseVector<gentype> &res, const SparseVector<S> &x, int useOrigin = 0, int useShortcut = 0, int givefeedback = 0, int convertanyhow = 0) const
    {
        if ( givefeedback )
        {
            errstream() << "Debug model_convertx\n";
        }

        if ( ismodel_convertx_simple && !useOrigin && !useShortcut && !isProjection && !convertanyhow )
        {
            return x;
        }

        xpweightIsWeight = 0;

        if ( (void *) &res == (void *) &x )
        {
            SparseVector<S> tempx(x);

            return model_convertx(res,tempx,useOrigin,useShortcut,givefeedback);
        }

        int dim = x.indsize();

        if ( givefeedback )
        {
            errstream() << "Debug model_convertx: " << dim << "\n";
        }

        res.indalign(x);

        if ( dim )
        {
            if ( a.size() )
            {
                int i,j,ii;

                for ( ii = 0 ; ii < dim ; ++ii )
                {
                    (res.direref(ii)).scalarfn_setisscalarfn(0);

                    i = x.ind(ii);

                    if ( locdistMode(i) == 1 )
                    {
                        // 1: v = a + e^(b+c.t)

                        //res.direref(ii) = a(i)+exp(b(i)+(c(i)*x.direcref(ii)));

                        res.direref(ii)  = c(i);
                        res.direref(ii) *= x.direcref(ii);
                        res.direref(ii) += b(i);
                        OP_exp(res.direref(ii));
                        res.direref(ii) += a(i);
                    }

                    else if ( locdistMode(i) == 2 )
                    {
                        // 2: v = (1/c) log(t-a) - (b/c)

                        //res.direref(ii) = (log(x.direcref(ii)-a(i))-b(i))/c(i);

                        res.direref(ii)  = x.direcref(ii);
                        res.direref(ii) -= a(i);
                        OP_log(res.direref(ii));
                        res.direref(ii) -= b(i);
                        res.direref(ii) /= c(i);
                    }

                    else if ( locdistMode(i) == 4 )
                    {
                        // 4: v = a - (1/b) log( 1/(0.5+(c*(t-0.5))) - 1 )
                        //      = a - (1/b) log(0.5-(c*(t-0.5))) + (1/b) log(0.5+(c*(t-0.5)))
                        //      = a - (1/b) log(1/(0.5+(c*(t-0.5)))) + (1/b) log(1/(0.5-(c*(t-0.5))))
                        //      = a + (1/b) log(1/(0.5-(c*(t-0.5)))) - (1/b) log(1/(0.5+(c*(t-0.5))))
                        //      = a + (1/b) log(1/tm) - (1/b) log(1/tp)

                        //res.direref(ii) = a(i)-(( log(1.0/(0.5+(c(i)*(x.direcref(ii)-0.5)))) - log(1.0/(0.5-(c(i)*(x.direcref(ii)-0.5)))) )/b(i));

                        gentype tm,tp;

                        tp  = x.direcref(ii);
                        tp -= 0.5;
                        tp *= c(i);

                        tm  = 0.5;
                        tm -= tp;

                        tp += 0.5;

                        tm.inverse();
                        OP_log(tm);

                        tp.inverse();
                        OP_log(tp);

                        res.direref(ii)  = tm;
                        res.direref(ii) -= tp;
                        res.direref(ii) /= b(i);
                        res.direref(ii) += a(i);
                    }

                    else
                    {
                        //res.direref(ii) = a(i)+(b(i)*x.direcref(ii));

                        res.direref(ii)  = b(i);
                        res.direref(ii) *= x.direcref(ii);
                        res.direref(ii) += a(i);
                    }

                    if ( locvarsType(i) == 0 )
                    {
                        j = (int) std::round((double) x.direcref(ii));

                        res.direref(ii).force_int() = j;
                    }

                    if ( useOrigin )
                    {
                        res.direref(ii).force_double() = 0.0;
                    }
                }
            }

            if ( ( isProjection == 4 ) && ( berndim < dim ) )
            {
                // Project up to equivalent (rather than messing with dim of bernstein polynomials

                int i,j;

                for ( i = berndim ; i < dim ; ++i )
                {
                    for ( j = i ; j >= 0 ; --j )
                    {
                        if ( j == 0 )
                        {
                            res.direref(j) = ( 1 - (((double) j)/((double) i)) )*res.direcref(j);
                        }

                        else if ( j < i )
                        {
                            res.direref(j) = (( (((double) j)/((double) i)) )*res.direcref(j-1)) + (( 1 - (((double) j)/((double) i)) )*res.direcref(j));
                        }

                        else
                        {
                            res.direref(j) = (( (((double) j)/((double) i)) )*res.direcref(j-1));
                        }
                    }
                }
            }

            if ( isProjection && ( isProjection <= 4 ) )
            {
                // This version is used to maintain models in smboopt.h, so we need to ensure that the result only includes the projection

                Vector<gentype> &pweight = xpweight;

                retVector<gentype> tmpva;
                retVector<gentype> tmpvb;

                pweight.resize(addSubDim ? dim+1 : dim);
                pweight("&",0,1,dim-1,tmpvb) = res(tmpva);

                if ( addSubDim )
                {
                    pweight("&",dim) = 1.0;
                }

                if ( givefeedback )
                {
                    errstream() << "ML weight = " << pweight << "\n";
                }

                (*projOp).setmlqweight(pweight);

                if ( useShortcut && useScalarFn && !includeConst && ( isProjection == 2 ) )
                {
                    xpweightIsWeight = 1;
                }

                if ( ( useShortcut != 1 ) || ( useShortcut && useScalarFn && !includeConst && ( isProjection == 2 ) ) )
                {
                    if ( isProjection == 1 )
                    {
                        SparseVector<gentype> xxmod;
                        SparseVector<gentype> xdummy;

                        (*projOp).gg(xxmod("&",(res.ind())(0)),xdummy);

                        res = xxmod;
                    }

                    else if ( isProjection == 2 )
                    {
                        if ( useScalarFn )
                        {
                            // Faster version: approximate function with vector evaluated on grid.

                            res = (*projOp).y(); // BLK_Conect does the work here.
                            res.scale(sqrt(1/((double) (*projOp).y().size()))); // To ensure the inner product approximates the true integral.  Note also, we use y().size() as N() won't work as you might expect for blk_bernst
                        }

                        else
                        {
                            // We need to duplicate projOp here so that each direction is kept

                            SparseVector<gentype> xxmod;

                            ML_Mutable *projOpNowRaw;
                            int projOpNowInd;

                            MEMNEW(projOpNowRaw,ML_Mutable);
                            (*projOpNowRaw).setMLTypeClean((*projOp).type());

                            (*projOpNowRaw).getML() = *projOp;
                            projOpNowInd = regMLloc(projOpNowRaw,locfnarg,4);

                            // Default (work anywhere) version

//FIXMEFIXME fnDim
                            //xxmod("&",(res.ind())(0)) = "fnB(var(1,0),500,x)"; // now gg(x) of ML var(1,0) (see instructvar.txt)
                            xxmod("&",(res.ind())(0)) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                            retVector<int> tmpvaa;

                            xxmod("&",(res.ind())(0)).scalarfn_setisscalarfn(useScalarFn);
                            xxmod("&",(res.ind())(0)).scalarfn_seti(zerointvec(fnDim,tmpvaa));
                            xxmod("&",(res.ind())(0)).scalarfn_setj(cntintvec(fnDim,tmpvaa));
                            xxmod("&",(res.ind())(0)).scalarfn_setnumpts(xNsamp);

                            Vector<gentype> varlist(fnDim);

                            int ii;

                            for ( ii = 0 ; ii < fnDim ; ++ii )
                            {
                                std::stringstream resbuffer;

                                resbuffer << "var(0," << ii << ")";
                                resbuffer >> varlist("&",ii);
                            }

                            SparseVector<SparseVector<gentype> > xy;
                            xy("&",1)("&",0) = projOpNowInd; // fill in var(1,0) with registered projOp index
                            xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                            xxmod("&",(res.ind())(0)).substitute(xy); // now gg(x,y,...) for ML projOp (that is, a function)

                            res = xxmod;
                        }
                    }

                    else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
                    {
                        if ( useScalarFn )
                        {
                            // Faster version: approximate function with vector evaluated on grid.

                            res = (*projOp).y(); // BLK_Conect does the work here.
                            res.scale(sqrt(1/((double) (*projOp).y().size()))); // To ensure the inner product approximates the true integral.  Note also, we use y().size() as N() won't work as you might expect for blk_bernst
                        }

                        else
                        {
                            // We need to duplicate projOp here so that each direction is kept

                            SparseVector<gentype> xxmod;

                            ML_Mutable *projOpNowRaw;
                            int projOpNowInd;

                            MEMNEW(projOpNowRaw,ML_Mutable);
                            (*projOpNowRaw).setMLTypeClean((*projOp).type());

                            (*projOpNowRaw).getML() = *projOp;
                            projOpNowInd = regMLloc(projOpNowRaw,locfnarg,4);

                            // Default (work anywhere) version

//FIXMEFIXME: fnDim
                            //xxmod("&",(res.ind())(0)) = "fnB(var(1,0),500,x)"; // now gg(x) of ML var(1,0) (see instructvar.txt)
                            xxmod("&",(res.ind())(0)) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                            retVector<int> tmpvaa;

                            xxmod("&",(res.ind())(0)).scalarfn_setisscalarfn(useScalarFn);
                            xxmod("&",(res.ind())(0)).scalarfn_seti(zerointvec(fnDim,tmpvaa));
                            xxmod("&",(res.ind())(0)).scalarfn_setj(cntintvec(fnDim,tmpvaa));
                            xxmod("&",(res.ind())(0)).scalarfn_setnumpts(xNsamp);

                            Vector<gentype> varlist(fnDim);

                            int ii;

                            for ( ii = 0 ; ii < fnDim ; ++ii )
                            {
                                std::stringstream resbuffer;

                                resbuffer << "var(0," << ii << ")";
                                resbuffer >> varlist("&",ii);
                            }

                            SparseVector<SparseVector<gentype> > xy;
                            xy("&",1)("&",0) = projOpNowInd; // fill in var(1,0) with registered projOp index
                            xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                            xxmod("&",(res.ind())(0)).substitute(xy); // now gg(x) for ML projOp (that is, a function)

                            res = xxmod;
                        }
                    }
                }
            }

            else if ( isProjection == 5 )
            {
//RKHSFIXME
                NiceAssert( res.indsize() == priorrandDirtemplateFnRKHS.N() );

                SparseVector<gentype> xxmod;

                retVector<gentype> tmpva;
                retVector<gentype> tmpvb;

                RKHSVector realres(priorrandDirtemplateFnRKHS);

                realres.a("&",tmpva) = res(tmpvb); // strip off sparseness (RHS), assign non-sparse version to alpha (weights) in RKHS
                xxmod("&",(res.ind())(0)) = realres; // set zeroth index of xxmod as RKHSVector using magic

                res = xxmod; // Assign result
            }
        }

        return res;
    }

    template <class S>
    const Vector<gentype> &convertx(int dim, Vector<gentype> &res, const Vector<S> &x, int useOrigin = 0, int givefeedback = 0) const
    {
        if ( (void *) &res == (void *) &x )
        {
            Vector<S> tempx(x);

            return convertx(dim,res,tempx);
        }

        res.resize(x.size());

        if ( x.size() )
        {
            if ( dim )
            {
                if ( a.size() )
                {
                    int i,j;

                    //for ( i = 0 ; ( i < dim ) && ( i < x.size() ) ; ++i )
                    for ( i = 0 ; i < x.size() ; ++i )
                    {
                        if ( i < dim )
                        {
                            res("&",i).scalarfn_setisscalarfn(0);

                            if ( locdistMode(i) == 1 )
                            {
                                // 1: v = a + e^(b+c.t)

                                //res("&",i) = a(i)+exp(b(i)+(c(i)*x(i)));

                                res("&",i)  = c(i);
                                res("&",i) *= x(i);
                                res("&",i) += b(i);
                                OP_exp(res("&",i));
                                res("&",i) += a(i);
                            }

                            else if ( locdistMode(i) == 2 )
                            {
                                // 2: v = (1/c) log(t-a) - (b/c)

                                //res("&",i) = (log(x(i)-a(i))-b(i))/c(i);

                                res("&",i)  = x(i);
                                res("&",i) -= a(i);
                                OP_log(res("&",i));
                                res("&",i) -= b(i);
                                res("&",i) /= c(i);
                            }

                            else if ( locdistMode(i) == 4 )
                            {
                                // 4: v = a - (1/b) log( 1/(0.5+(c*(t-0.5))) - 1 )
                                //      = a - (1/b) log(0.5-(c*(t-0.5))) + (1/b) log(0.5+(c*(t-0.5)))
                                //      = a - (1/b) log(1/(0.5+(c*(t-0.5)))) + (1/b) log(1/(0.5-(c*(t-0.5))))

                                //res("&",i) = a(i)-(( log(1.0/(0.5+(c(i)*(x(i)-0.5)))) - log(1.0/(0.5-(c(i)*(x(i)-0.5)))) )/b(i));

                                gentype tm,tp;

                                tp  = x(i);
                                tp -= 0.5;
                                tp *= c(i);

                                tm  = 0.5;
                                tm -= tp;

                                tp += 0.5;

                                tm.inverse();
                                OP_log(tm);

                                tp.inverse();
                                OP_log(tp);

                                res("&",i)  = tm;
                                res("&",i) -= tp;
                                res("&",i) /= b(i);
                                res("&",i) += a(i);
                            }

                            else
                            {
                                //res("&",i) = a(i)+(b(i)*x(i));

                                res("&",i)  = b(i);
                                res("&",i) *= x(i);
                                res("&",i) += a(i);
                            }

                            if ( locvarsType(i) == 0 )
                            {
                                j = (int) std::round((double) x(i));

                                res("&",i).force_int() = j;
                            }

                            if ( useOrigin )
                            {
                                res("&",i).force_double() = 0.0;
                            }
                        }

                        else if ( i == dim )
                        {
                            // for grid optimisation, this hold the expected (pre-processing) y value

                            res("&",i).scalarfn_setisscalarfn(0);
                            res("&",i) = x(i);
                        }
                    }
                }
            }

            if ( ( isProjection == 4 ) && ( berndim < dim ) )
            {
                // Project up to equivalent (rather than messing with dim of bernstein polynomials

                int i,j;

                for ( i = berndim ; i < dim ; ++i )
                {
                    for ( j = i ; j >= 0 ; --j )
                    {
                        if ( j == 0 )
                        {
                            res("&",j) = ( 1 - (((double) j)/((double) i)) )*res(j);
                        }

                        else if ( j < i )
                        {
                            res("&",j) = (( (((double) j)/((double) i)) )*res(j-1)) + (( 1 - (((double) j)/((double) i)) )*res(j));
                        }

                        else
                        {
                            res("&",j) = (( (((double) j)/((double) i)) )*res(j-1));
                        }
                    }
                }
            }

            if ( isProjection && ( isProjection <= 4 ) )
            {
                // This version is used to send data back to mlinter, so xxmod needs to 
                // be "filled out" to the same size as x *even though only the first bit is 
                // relevant*.  Note also that this is "after", so projOp is "fixed" from
                // here, meaning that we don't need to duplicate.

                Vector<gentype> &pweight = xpweight;

                retVector<gentype> tmpva;

                pweight.resize(addSubDim ? dim+1 : dim);
                pweight("&",0,1,dim-1,tmpva) = res;

                if ( addSubDim )
                {
                    pweight("&",dim) = 1.0;
                }

                if ( givefeedback )
                {
                    errstream() << "ML weight (2) = " << pweight << "\n";
                }

                (*projOp).setmlqweight(pweight);

                if ( isProjection == 1 )
                {
                    Vector<gentype> xxmod(res);
                    SparseVector<gentype> xdummy;

                    (*projOp).gg(xxmod("&",0),xdummy);

                    res = xxmod;
                }

                else if ( isProjection == 2 )
                {
                    Vector<gentype> xxmod(res);

//FIXMEFIXME fnDim
                    //xxmod("&",0) = "fnB(var(1,0),500,x)"; // now gg(x) of ML y (see instructvar.txt)
                    xxmod("&",0) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                    retVector<int> tmpvaa;

                    xxmod("&",0).scalarfn_setisscalarfn(useScalarFn);
                    xxmod("&",0).scalarfn_seti(zerointvec(fnDim,tmpvaa));
                    xxmod("&",0).scalarfn_setj(cntintvec(fnDim,tmpvaa));
                    xxmod("&",0).scalarfn_setnumpts(xNsamp);

                    Vector<gentype> varlist(fnDim);

                    int ii;

                    for ( ii = 0 ; ii < fnDim ; ++ii )
                    {
                        std::stringstream resbuffer;

                        resbuffer << "var(0," << ii << ")";
                        resbuffer >> varlist("&",ii);
                    }

                    SparseVector<SparseVector<gentype> > xy;
                    xy("&",1)("&",0) = projOpInd; // fill in var(1,0) with registered projOp index
                    xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                    xxmod("&",0).substitute(xy);

                    res = xxmod;
                }

                else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
                {
                    Vector<gentype> xxmod(res);

//FIXMEFIXME fnDim
                    //xxmod("&",0) = "fnB(var(1,0),500,x)"; // now gg(x) of ML y (see instructvar.txt)
                    xxmod("&",0) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                    retVector<int> tmpvaa;

                    xxmod("&",0).scalarfn_setisscalarfn(useScalarFn);
                    xxmod("&",0).scalarfn_seti(zerointvec(fnDim,tmpvaa));
                    xxmod("&",0).scalarfn_setj(cntintvec(fnDim,tmpvaa));
                    xxmod("&",0).scalarfn_setnumpts(xNsamp);

                    Vector<gentype> varlist(fnDim);

                    int ii;

                    for ( ii = 0 ; ii < fnDim ; ++ii )
                    {
                        std::stringstream resbuffer;

                        resbuffer << "var(0," << ii << ")";
                        resbuffer >> varlist("&",ii);
                    }

                    SparseVector<SparseVector<gentype> > xy;
                    xy("&",1)("&",0) = projOpInd; // fill in var(1,0) with registered projOp index
                    xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                    xxmod("&",0).substitute(xy);

                    res = xxmod;
                }
            }

            else if ( isProjection == 5 )
            {
//RKHSFIXME
                NiceAssert( res.size() == priorrandDirtemplateFnRKHS.N() );

                Vector<gentype> xxmod(res);

                retVector<gentype> tmpva;

                RKHSVector realres(priorrandDirtemplateFnRKHS);

                realres.a("&",tmpva) = res; // assign weights to alpha in RKHS
                xxmod("&",0) = realres; // set zeroth index of xxmod as RKHSVector using magic

                res = xxmod; // Assign result
            }
        }

        return res;
    }

    template <class S>
    const Vector<SparseVector<gentype> > &model_convertx(Vector<SparseVector<gentype> > &res, const Vector<SparseVector<S> > &x) const
    {
        res = x;

        if ( x.size() )
        {
            int i;

            for ( i = 0 ; i < x.size() ; ++i )
            {
                model_convertx(res("&",i),x(i),0,0,0,1);
            }
        }

        return res;
    }

    template <class S>
    const Vector<Vector<gentype> > &convertx(int dim, Vector<Vector<gentype> > &res, const Vector<Vector<S> > &x) const
    {
        res = x;

        if ( x.size() )
        {
            int i;

            for ( i = 0 ; i < x.size() ; ++i )
            {
                convertx(dim,res("&",i),x(i));
            }
        }

        return res;
    }

    // ML registration stuff (for functional optimisation)

    int regMLloc(ML_Mutable *MLcase, void *fnarg, int ltyp) const
    {
        int nres = -1;

        if ( MLregfn )
        {
            nres = ( MLregind = MLregfn(MLregind,MLcase,fnarg) );

            MLreglist.append(MLreglist.size(),nres);
            MLregltyp.append(MLregltyp.size(),ltyp);

            ++MLregind;
        }

        return nres;
    }

    // Needed by bayesopt

    int isspOverride(void) const { return spOverride; }

    // Needed by gridopt

    Vector<int> getlocdistMode(void) const { return locdistMode; }
    Vector<int> getlocvarsType(void) const { return locvarsType; }

    // Needed by smboopt

    double getxwidth(void) const { return xwidth; }

    const Vector<gentype> &getxpweight(void) const { return xpweight; }
    int getxpweightIsWeight(void)            const { return xpweightIsWeight; }

    const Vector<SparseVector<gentype> > &getxbasis(void) const { return xbasis; }
    const Matrix<gentype> &getxbasisprod(void)            const { return xbasisprod; }

private:
//FIXME - all variables that follow need to be reset by "reset" function
    // xwidth - set by optim to maximum width of x range

    double xwidth; // initially set 0

    // Used by scheduled Bernstein (isProjection == 4 )
    //
    // stopearly:  usually 0, set 1 during scheduled Bernstein if maxdiff in best x >= 0.95
    //             (keeps track of best result etc)
    // firsttest:  1 for first test
    // spOverride: set for all but the first run to make sure startpoints are only added for the first for scheduled Bernstein
    // bestyet:    best result yet

    int     stopearly;
    int     firsttest;
    int     spOverride;
    int     berndim;
    gentype bestyet;

    // ML registration stuff (for functional optimisation)
    //
    // MLregind:  index where ML is registered (unless full, in which case this
    //            gets incremented until an empty one is found)
    // MLregfn:   function to register ML.  ind is suggested, actual assigned
    //            index is returned.
    // MLreglist: list of all registered MLs (so they can be deleted)
    // MLregltyp: list of all registered ML types
    //            0  = randDirtemplate (copy of priorrandDirtemplateVec in constructor)
    //            1  = randDirtemplate (copy of priorrandDirtemplateFnGP in constructor)
    //            2  = subDef (in makeSubspace)
    //            3  = projOp (in makeSubspace)
    //            4  = projOpNow (in convertx)
    //            5  = muapprox (copy of altfnapprox in smboopt constructor)
    //            6  = muapprox (copy of altfnapproxmoo in smboopt constructor)
    //            7  = sigmaapprox (in smboopt constructor)
    //            8  = subDef (in makeSubspace)
    //            9  = projOp (in makeSubspace)
    //            10 = source model in diff-GP
    //            11 = difference model in diff-GP
    //            12 = augx (copy of altfxapprox in smboopt constructor)

    mutable int MLregind;

    mutable Vector<int> MLreglist;
    mutable Vector<int> MLregltyp;

    // Convert x to p(x) (we want to optimise f(p(x)))
    //
    // useShortcut: 0 usual - calculate res
    //              1 only calculate xpweight , not res (if possible: otherwise calculate res)
    //              2 calculate both xpweight and res (where possible)
    //
    // Relevant data: xpweight, xpweightIsWeight, xbasis, xbasisprod

    mutable Vector<gentype> xpweight; // We assume that this is single-threaded
    mutable int xpweightIsWeight; // if 1 then xpweight contains the actual weights
    Vector<SparseVector<gentype> > xbasis; // when xpweightIsWeight set this will be the basis vectors themselves
    Matrix<gentype> xbasisprod; // when xpweightIsWeight set this will be the inner products of the basis vectors (function) themselves

    // Convert vectors from optimisation space to "real" space.  The optimisers
    // work on (see) the "optimisation" space, which is then converted to "real"
    // space when evaluating f(x).

    Vector<double> a;
    Vector<double> b;
    Vector<double> c;

    Vector<int> locdistMode;
    Vector<int> locvarsType;

    bool ismodel_convertx_simple;

    // Private variables

    BLK_Conect projOptemplate;
    mutable BLK_Conect *projOp;
    ML_Mutable *projOpRaw;
    int projOpInd;

    ML_Base *randDirtemplate;
    int randDirtemplateInd;
    Vector<ML_Base *> subDef;
    Vector<int> subDefInd;

    int addSubDim; // additional (fixed) dimensions on subspace

    void *locfnarg;

    bool needsReset; // set after optimization run to indicate that reset() needs to be called next time around

public:
    int overfnitcnt;
};





#endif



