
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
           SparseVector<ML_Mutable *> &svmbase,
           Stack<awarestream *> *commstack,
           svmvolatile SparseVector<SparseVector<gentype> > &globargvariables,
           int (*getsetExtVar)(gentype &res, const gentype &src, int num),
           SparseVector<SparseVector<int> > &returntag);

// Kill all threads, including main (0) thread if killmain is set.

void killallthreads(SVMThreadContext *svmContext);

// Delete all MLs

void deleteMLs(SparseVector<ML_Mutable *> &svmbase);

// Direct access to MLs
//
// grabsvm:  make SVM svmInd exist
// regsvm:   register SVM whattoreg at first available index >= svmInd
//           returns index where registered
// getMLref: get reference to requested ML

void grabsvm(SparseVector<ML_Mutable *> &svmbase, int svmInd);
int regsvm(SparseVector<ML_Mutable *> &svmbase, int svmInd, ML_Mutable &whattoreg);
ML_Mutable &getMLref(SparseVector<ML_Mutable *> &svmbase, int svmInd);
const ML_Mutable &getMLrefconst(SparseVector<ML_Mutable *> &svmbase, int svmInd);




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
    SVMThreadContext(int xsvmInd = 1)
    {
        verblevel        = 1;
        finalresult      = 0.0;
        svmInd           = xsvmInd;
        biasdefault      = 0.0;
        filevariables.zero();
        xtemplate.zero();
        depthin          = 0;
        logfile          = "";
        binaryRelabel    = 0;
        singleDrop       = 0;
        updateargvars    = 1;
        killswitch       = 0;

        argvariables("&",1)("&",45) = svmInd;
    }

    SVMThreadContext(const SVMThreadContext &src)
    {
        *this = src;
    }

    SVMThreadContext &operator=(const SVMThreadContext &src)
    {
        verblevel        = src.verblevel;
        finalresult      = src.finalresult;
        svmInd           = src.svmInd;
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
    // svmInd:        decides which ML in svmbase is currently being operated on.
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
    int svmInd;
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

#endif
