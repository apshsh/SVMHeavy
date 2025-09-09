
//
// SVMHeavyv7 abstracted interface
//
// Version: 7
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _mlinter_h
#define _mlinter_h

#include <iostream>
#include <fstream>
#include "basefn.hpp"
#include "ml_base.hpp"
#include "mlcommon.hpp"
#include "ml_mutable.hpp"
#include "gentype.hpp"
#include "ofiletype.hpp"
#include "vecstack.hpp"
#include "awarestream.hpp"
#include "gridopt.hpp"
#include "directopt.hpp"
#include "nelderopt.hpp"
#include "bayesopt.hpp"
#include "globalopt.hpp"



class SVMThreadContext;


// Core function
// =============
//
// runsvm: This is the core of svmheavy.  Given various arguments it interprets
//         commands given and runs operations on the given SVM (svmbase) as
//         requested.  This is abstracted so that it can be called recursively
//         (eg in gridsearch).
//
// svmContext - running data.
// commStack  - input (command) stream for this thread.
// globalvariables - data shared between all threads
//
// getsetExtVar: - get or set external (typically mex) variable.
//               - if num >= 0 then loads extvar num into res.  If extvar is
//                 a function handle then src acts as an argument (optional,
//                 not used if null, multiple arguments if set).
//               - if num == -1 then loads external variable named in res
//                 (res must be string) into res before returning.  In this
//                 case src gives preferred type if result interpretation is
//                 ambiguous (type of res will attempt to copy gentype of 
//                 src).
//               - if num == -2 then loads contents of src into external 
//                 variable named in res before returning.
//               - if num == -3 then evaluates fn(v) where fn is a matlab
//                 function named by res, v is the set of arguments (see
//                 num >= 0) and the result is stored in res.
//               - if num == -4 then returns 0 if this function actually
//                 does anything (eg if mex is present) or -1 if this is
//                 just a dummy function that does nothing.
//               - returns 0 on success, -1 on failure.
//
// Return value:
//
// 0:       commStack exhausted, ML still running
// -1:      -ZZZZ encoutered
// 1-99:    syntax error
// 101-199: file error
// 201-299: thread error
// 301-399: unknown throw

int runsvm(SVMThreadContext *svmContext,
           Stack<awarestream *> *commstack,
           svmvolatile SparseVector<SparseVector<gentype> > &globargvariables,
           int (*getsetExtVar)(gentype &res, const gentype &src, int num),
           SparseVector<SparseVector<int> > &returntag);

// Kill all threads, including main (0) thread if killmain is set.

void killallthreads(SVMThreadContext *svmContext);

// Direct access to MLs
//
// getMLref:      get reference to requested ML
// getMLrefconst: get constant reference to requested ML
// deleteMLs:     delete all MLs

const ML_Mutable    &getMLrefconst        (int MLInd        );
const GridOptions   &getgridrefconst      (int gridInd      );
const DIRectOptions &getDIRectrefconst    (int DIRectInd    );
const NelderOptions &getNelderMeadrefconst(int NelderMeadInd);
const BayesOptions  &getBayesianrefconst  (int BayesianInd  );

ML_Mutable    &getMLref        (int MLInd        );
GridOptions   &getgridref      (int gridInd      );
DIRectOptions &getDIRectref    (int DIRectInd    );
NelderOptions &getNelderMeadref(int NelderMeadInd);
BayesOptions  &getBayesianref  (int BayesianInd  );

void deleteMLs        (void);
void deletegrids      (void);
void deleteDIRects    (void);
void deleteNelderMeads(void);
void deleteBayess     (void);

// Get ML pseudo-indexes for elements in Bayesian optimization block

int MLIndForBayesian_muapprox   (int i, int k);
int MLIndForBayesian_cgtapprox  (int i, int k);
int MLIndForBayesian_augxapprox (int i, int k);
int MLIndForBayesian_sigmaapprox(int i);
int MLIndForBayesian_srcmodel   (int i);
int MLIndForBayesian_diffmodel  (int i);

int MLIndForBayesian_muapprox_prior   (int i);
int MLIndForBayesian_cgtapprox_prior  (int i);
int MLIndForBayesian_augxapprox_prior (int i);
int MLIndForBayesian_sigmaapprox_prior(int i);
int MLIndForBayesian_srcmodel_prior   (int i);
int MLIndForBayesian_diffmodel_prior  (int i);

int MLIndForgrid_randDirtemplate_prior      (int i);
int MLIndForDIRect_randDirtemplate_prior    (int i);
int MLIndForNelderMead_randDirtemplate_prior(int i);
int MLIndForBayesian_randDirtemplate_prior  (int i);





class SVMThreadContext;

inline SVMThreadContext *&setzero(SVMThreadContext *&x);
inline void qswap(SVMThreadContext *&a, SVMThreadContext *&b);
//inline ML_Mutable *&setzero(ML_Mutable *&x);

inline SVMThreadContext *&setident (SVMThreadContext *&a) { NiceThrow("SVMThreadContext setident not defined"); return a; }
inline SVMThreadContext *&setposate(SVMThreadContext *&a) { return a; }
inline SVMThreadContext *&setnegate(SVMThreadContext *&a) { NiceThrow("SVMThreadContext setnegate not defined"); return a; }
inline SVMThreadContext *&setconj  (SVMThreadContext *&a) { NiceThrow("SVMThreadContext setconj not defined"); return a; }
inline SVMThreadContext *&setrand  (SVMThreadContext *&a) { NiceThrow("SVMThreadContext setrand not defined"); return a; }
inline SVMThreadContext *&postProInnerProd(SVMThreadContext *&x);


class SVMThreadContext
{
public:
    SVMThreadContext(int xMLInd = 1, int xgridInd = 1, int xDIRectInd = 1, int xNelderMeadInd = 1, int xBayesianInd = 1)
    {
        verblevel        = 1;
        finalresult      = 0.0;
        MLInd            = xMLInd;
        gridInd          = xgridInd;
        DIRectInd        = xDIRectInd;
        NelderMeadInd    = xNelderMeadInd;
        BayesianInd      = xBayesianInd;
        biasdefault      = 0.0;
        filevariables.zero();
        xtemplate.zero();
        depthin          = 0;
        logfile          = "";
        binaryRelabel    = 0;
        singleDrop       = 0;
        updateargvars    = 1;
        killswitch       = 0;

        argvariables("&",1)("&",45) = MLInd;
    }

    SVMThreadContext(const SVMThreadContext &src)
    {
        *this = src;
    }

    SVMThreadContext &operator=(const SVMThreadContext &src)
    {
        verblevel        = src.verblevel;
        finalresult      = src.finalresult;
        MLInd            = src.MLInd;
        gridInd          = src.gridInd;
        DIRectInd        = src.DIRectInd;
        NelderMeadInd    = src.NelderMeadInd;
        BayesianInd      = src.BayesianInd;
        biasdefault      = src.biasdefault;
        argvariables     = src.argvariables;
        filevariables    = src.filevariables;
        xtemplate        = src.xtemplate;
        depthin          = 0;
        logfile          = "";
        binaryRelabel    = src.binaryRelabel;
        singleDrop       = src.singleDrop;
        updateargvars    = src.updateargvars;
        MLindstack       = src.MLindstack;
        killswitch       = 0;

        return *this;
    }

    // verblevel:     verbosity level when writing to logfile.
    //                0: minimal - only send feedback to errstream()
    //                1: normal - above, plus write details to logfiles
    // finalresult:   during the test phase of a given command block, the result
    //                of the last test run (as defined by the test ordering, not
    //                the order of commands given) is written to this variable.
    //                Generally speaking this is a measure of error, so lower
    //                results mean better performance, though the exact meaning
    //                depends on the SVM type and the resfilter (if any) used.  This
    //                argument is used by gridsearch.
    // MLInd:         decides which ML in svmbase is currently being operated on.
    // gridInd:       ditto for grid optimizers
    // DIRectInd:     ditto for DIRect optimizers
    // NelderMeadInd: ditto for Nelder-Mead optimizers
    // BayesianInd:   ditto for Bayesian optimizers
    // biasdefault:   default bias used by the SVM, for use when needed
    // argvariables:  this sparse matrix contains all relevant running variables
    //                for the SVM, which can be directly accessed in commands as
    //                described in the help section at the end.  This is included
    //                as an argument as it is kept and used during gridsearch
    //                recursion.
    // filevariables: for data processing it is possible to extract data from files
    //                at multiple stages in the training and testing process.
    //                Files opened for this purpose are stored in this vector.
    //                (the type ofiletype keeps track of which data entries from
    //                this file have been taken (used) and which are still available
    //                for future extraction).
    // xtemplate:     boilerplate (default) parts of all x vectors
    // depthin:       when recursing (for example) records depth.  1 for top layer
    // logfile:       name of logfile
    // binaryRelabel: see addData.h
    // singleDrop:    see addData.h
    // MLindstack:    used to keep indices when pushing/popping MLs
    //
    // killswitch:       set this 1 to tell thread to stop.
    // updateargvars:    set 1 if argvariables need to be updated, zero otherwise

    int verblevel;
    gentype finalresult;
    int MLInd;
    int gridInd;
    int DIRectInd;
    int NelderMeadInd;
    int BayesianInd;
    gentype biasdefault;
    SparseVector<SparseVector<gentype> > argvariables;
    SparseVector<ofiletype> filevariables;
    SparseVector<gentype> xtemplate;
    int depthin;
    std::string logfile;
    int binaryRelabel;
    int singleDrop;
    int updateargvars;
    Stack<int> MLindstack;
    svmvolatile int killswitch;
};

inline SVMThreadContext *&setzero(SVMThreadContext *&x)
{
    return ( x = nullptr ); // Must be deleted elsewhere
}

inline void qswap(SVMThreadContext *&a, SVMThreadContext *&b)
{
    SVMThreadContext *c;

    c = a; a = b; b = c;
}

//inline ML_Mutable *&setzero(ML_Mutable *&x)
//{
//    x = nullptr;
//
//    return x;
//}

#define BOILER_PTR(T) \
inline void qswap(T *&a, T *&b); \
inline void qswap(T *&a, T *&b) { T *x = a; a = b; b = x; } \
inline T *&setident (T *&a) { throw("something"); return a; } \
inline T *&setzero  (T *&a) { return a = nullptr;           } \
inline T *&setposate(T *&a) { return a;                     } \
inline T *&setnegate(T *&a) { throw("something"); return a; } \
inline T *&setconj  (T *&a) { throw("something"); return a; } \
inline T *&setrand  (T *&a) { throw("something"); return a; } \
inline T *&postProInnerProd(T *&a) { return a; }

BOILER_PTR(GridOptions);
BOILER_PTR(DIRectOptions);
BOILER_PTR(NelderOptions);
BOILER_PTR(BayesOptions);



#endif
