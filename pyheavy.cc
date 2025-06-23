
//
// SVMHeavyv7 Python CLI-like Interface
//
// Version: 7
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Usage: svmpython('commands'), where commands are just like the regular CLI
//

#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

inline void qswap(py::object *&a, py::object *&b);
inline void qswap(py::object *&a, py::object *&b)
{
    py::object *x = a; a = b; b = x;
}

inline py::object *&setident (py::object *&a) { throw("something"); return a; }
inline py::object *&setzero  (py::object *&a) { return a; }
inline py::object *&setposate(py::object *&a) { return a; }
inline py::object *&setnegate(py::object *&a) { throw("something"); return a; }
inline py::object *&setconj  (py::object *&a) { throw("something"); return a; }
inline py::object *&setrand  (py::object *&a) { throw("something"); return a; }
inline py::object *&postProInnerProd(py::object *&a) { return a; }

inline void qswap(std::string *&a, std::string *&b);
inline void qswap(std::string *&a, std::string *&b)
{
    std::string *x = a; a = b; b = x;
}

#include "mlinter.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "vecfifo.hpp"

int glob_svmInd(int i = -1)
{
    static int iii = 0;

    if ( i != -1 )
    {
        iii = i;
    }

    return iii;
}

SparseVector<ML_Mutable *> &getMLmodels(void);


// unconvertable objects converted to nan

                   py::object convToPy(const int             &src);
                   py::object convToPy(const double          &src);
                   py::object convToPy(const d_anion         &src);
                   py::object convToPy(const std::string     &src);
template <class T> py::object convToPy(const Vector<T>       &src);
template <class T> py::object convToPy(const Set<T>          &src);
template <class T> py::object convToPy(const SparseVector<T> &src);
                   py::object convToPy(const gentype         &src);

// These return 1 if conversion fails

                   int convFromPy(int             &res, py::object src);
                   int convFromPy(double          &res, py::object src);
                   int convFromPy(d_anion         &res, py::object src); // - don't know how to do this conversion unfortunately, so it just set res = nan and returns 1 (error)
                   int convFromPy(std::string     &res, py::object src);
template <class T> int convFromPy(Vector<T>       &res, py::object src);
template <class T> int convFromPy(Set<T>          &res, py::object src);
template <class T> int convFromPy(SparseVector<T> &res, py::object src);
                   int convFromPy(gentype         &res, py::object src);

#define         QDO(modis,dofn,desc)        modis.def(#dofn,  &(mod_ ## dofn),  #dofn  "(i) - "         desc                      " for ML i (i=-1 for current ML).");
#define      QDOARG(modis,dofn,desc)        modis.def(#dofn,  &(mod_ ## dofn),  #dofn  "(i,p) - "       desc                      " for ML i (i=-1 for current ML).");
#define        QGET(modis,getfn,desc)       modis.def(#getfn, &(mod_ ## getfn), #getfn "(i) - get "     desc " parameter " #getfn " for ML i (i=-1 for current ML).");
#define        QSET(modis,getfn,setfn,desc) modis.def(#setfn, &(mod_ ## setfn), #setfn "(i,p) - set "   desc " parameter " #getfn " for ML i (i=-1 for current ML).");
#define     QGETSET(modis,getfn,setfn,desc) modis.def(#setfn, &(mod_ ## setfn), #setfn "(i,p) - set "   desc " parameter " #getfn " for ML i (i=-1 for current ML)."); \
                                            modis.def(#getfn, &(mod_ ## getfn), #getfn "(i) - get "     desc " parameter " #getfn " for ML i (i=-1 for current ML).");
#define     QGETCLA(modis,getfn,desc)       modis.def(#getfn, &(mod_ ## getfn), #getfn "(i,d) - get "   desc " parameter " #getfn " for ML i (i=-1 for current ML).");
#define     QSETCLA(modis,getfn,setfn,desc) modis.def(#setfn, &(mod_ ## setfn), #setfn "(i,d,p) - set " desc " parameter " #getfn " for ML i (i=-1 for current ML).");
#define  QGETSETCLA(modis,getfn,setfn,desc) modis.def(#setfn, &(mod_ ## setfn), #setfn "(i,d,p) - set " desc " parameter " #getfn " for ML i (i=-1 for current ML)."); \
                                            modis.def(#getfn, &(mod_ ## getfn), #getfn "(i,d) - get "   desc " parameter " #getfn " for ML i (i=-1 for current ML).");
#define    QGETSETD(modis,getfn,setfn,getname,setname,desc) modis.def(setname, &(mod_ ## setfn), setname "(i,t) - "   desc "\n\n" "for ML i (i=-1 for current ML)."); \
                                                            modis.def(getname, &(mod_ ## getfn), getname "(i) - get setting for "   setname "(i,...) "   "for ML i (i=-1 for current ML).");
#define QGETSETCLAD(modis,getfn,setfn,getname,setname,desc) modis.def(setname, &(mod_ ## setfn), setname "(i,d,t) - " desc "\n\n" "for ML i (i=-1 for current ML)."); \
                                                            modis.def(getname, &(mod_ ## getfn), getname "(i,d) - get setting for " setname "(i,d,...) " "for ML i (i=-1 for current ML).");

#define        DODEF(dofn)          py::object mod_ ## dofn(int i)                       { i = glob_svmInd(i); return convToPy(getMLref(getMLmodels(),i). dofn ());   }
#define     DOARGDEF(dofn,T)        int        mod_ ## dofn(int i,         py::object p) { i = glob_svmInd(i); T altp; convFromPy(altp,p); return getMLref(getMLmodels(),i). dofn (altp);   }
#define       GETDEF(getfn)         py::object mod_ ## getfn(int i)                      { i = glob_svmInd(i); return convToPy(getMLrefconst(getMLmodels(),i). getfn ());  }
#define    GETDEFCLA(getfn)         py::object mod_ ## getfn(int i, int d)               { i = glob_svmInd(i); return convToPy(getMLrefconst(getMLmodels(),i). getfn (d)); }
#define       SETDEF(setfn,T)       int        mod_ ## setfn(int i,        py::object p) { i = glob_svmInd(i); T altp; convFromPy(altp,p); return getMLref(getMLmodels(),i). setfn (altp);   }
#define    SETDEFCLA(setfn,T)       int        mod_ ## setfn(int i, int d, py::object p) { i = glob_svmInd(i); T altp; convFromPy(altp,p); return getMLref(getMLmodels(),i). setfn (d,altp); }
#define    GETSETDEF(getfn,setfn,T) py::object mod_ ## getfn(int i)                      { i = glob_svmInd(i); return convToPy(getMLrefconst(getMLmodels(),i). getfn ());  } \
                                    int        mod_ ## setfn(int i,        py::object p) { i = glob_svmInd(i); T altp; convFromPy(altp,p); return getMLref(getMLmodels(),i). setfn (altp);   }
#define GETSETDEFCLA(getfn,setfn,T) py::object mod_ ## getfn(int i, int d)               { i = glob_svmInd(i); return convToPy(getMLrefconst(getMLmodels(),i). getfn (d)); } \
                                    int        mod_ ## setfn(int i, int d, py::object p) { i = glob_svmInd(i); T altp; convFromPy(altp,p); return getMLref(getMLmodels(),i). setfn (d,altp); }


py::object pyogetsrc(int i);
gentype    gengetsrc(int i);

int pyosetsrc(int i, py::object src);
int gensetsrc(int i, gentype    src);

void pyosrcreset(void);
void gensrcreset(void);

void pyosrcreset(int i);
void gensrcreset(int i);

py::object genevalsrc(int i, py::object x); // evaluate gentype function with x and evaluate result
gentype    pyoevalsrc(int i, gentype    x); // evaluate py::object function with x and evaluate result

void svmheavya(void);                               // just get help screen
void svmheavyb(int permode);                        // set persistence mode
void svmheavyc(const std::string commstr);          // execute with string

void callintercalc(void);
void callsnakes(void);

py::object svmeval(std::string fn, py::object arg);

GETSETDEF(getMLType,ssetMLTypeClean,std::string);

GETSETDEF(getVmethod,setVmethod,std::string);
GETSETDEF(getCmethod,setCmethod,std::string);
GETSETDEF(getOmethod,setOmethod,std::string);
GETSETDEF(getAmethod,setAmethod,std::string);
GETSETDEF(getRmethod,setRmethod,std::string);
GETSETDEF(getTmethod,setTmethod,std::string);
GETSETDEF(getBmethod,setBmethod,std::string);
GETSETDEF(getMmethod,setMmethod,std::string);

GETSETDEF(prim,setprim,int);
GETSETDEF(prival,setprival,gentype);
int setpriml(int i, int j);

DODEF(train);
DODEF(reset);
DODEF(restart);
DOARGDEF(scale,double);
DOARGDEF(randomise,double);
DODEF(removeNonSupports);
DOARGDEF(trimTrainingSet,int);

GETSETDEF(C,        setC,        double);
GETSETDEF(sigma,    setsigma,    double);
GETSETDEF(sigma_cut,setsigma_cut,double);
GETSETDEF(eps,      seteps,      double);

GETSETDEF(LinBiasForce, setLinBiasForce, double);
GETSETDEF(QuadBiasForce,setQuadBiasForce,double);
GETSETDEF(nu,           setnu,           double);
GETSETDEF(nuQuad,       setnuQuad,       double);

GETSETDEF(k,  setk,  int);
GETSETDEF(ktp,setktp,int);

GETSETDEFCLA(Cclass,  setCclass,  double);
GETSETDEFCLA(epsclass,setepsclass,double);

GETSETDEFCLA(LinBiasForceclass, setLinBiasForceclass, double);
GETSETDEFCLA(QuadBiasForceclass,setQuadBiasForceclass,double);

GETSETDEF(tspaceDim,settspaceDim,int);
GETSETDEF(order,    setorder,    int);

GETSETDEF(m,      setm,      int   );
GETSETDEF(theta,  settheta,  double);
GETSETDEF(simnorm,setsimnorm,int   );

GETDEF(loglikelihood);
GETDEF(maxinfogain);
GETDEF(RKHSnorm);
GETDEF(RKHSabs);

GETDEF(N);
GETDEF(type);
GETDEF(subtype);
GETDEF(xspaceDim);
GETDEF(fspaceDim);
GETDEF(tspaceSparse);
GETDEF(xspaceSparse);
GETDEF(numClasses);
GETDEF(ClassLabels);
GETDEF(isTrained);
GETDEF(isSolGlob);
GETDEF(isUnderlyingScalar);
GETDEF(isUnderlyingVector);
GETDEF(isUnderlyingAnions);
GETDEF(isClassifier);
GETDEF(isRegression);
GETDEF(isPlanarType);

GETDEF(NZ);
GETDEF(NF);
GETDEF(NS);
GETDEF(NC);
GETDEF(NLB);
GETDEF(NLF);
GETDEF(NUF);
GETDEF(NUB);

GETDEFCLA(NNC);

GETDEF(x);
GETDEF(d);
GETDEF(y);
GETDEF(yp);
GETDEF(Cweight);
GETDEF(Cweightfuzz);
GETDEF(sigmaweight);
GETDEF(epsweight);
GETDEF(alphaState);
GETDEF(xtang);

GETDEF(kerndiag);

GETSETDEF(alpha,setAlpha,Vector<gentype>);
GETSETDEF(bias, setBias, gentype);

GETSETDEF(gamma,setgamma,Vector<gentype>);
GETSETDEF(delta,setdelta,gentype);

GETSETDEF(muWeight,setmuWeight,Vector<gentype>);
GETSETDEF(muBias,  setmuBias,  gentype);

int addTrainingVectorml    (int i, int j, py::object z, py::object x);
int maddTrainingVectorml   (int i, int j, py::object z, py::object x);
int detaddTrainingVectorml (int i, int j, py::object z, py::object x, py::object Cweigh, py::object epsweigh, py::object d);
int detmaddTrainingVectorml(int i, int j, py::object z, py::object x, py::object Cweigh, py::object epsweigh);

int removeTrainingVectorml(int i, int j, int num);

py::object muml (int i, py::object x);
py::object mugml(int i, py::object x);
py::object varml(int i, py::object x);
py::object covml(int i, py::object x, py::object y);

py::object mlalpha(int i);
py::object mlbias (int i);

int mlsetalpha(int i, py::object src);
int mlsetbias (int i, py::object src);



PYBIND11_MODULE(pyheavy, m) {
    m.doc() = "SVMHeavy Machine Learning Library.";
    m.def("help", &svmheavya, "Help screen.");
    m.def("mode", &svmheavyb, "Set persistence mode (default 1, persistent).");
    m.def("exec", &svmheavyc, "Run with string given (mimics CLI).");

    // ---------------------------

    auto m_int = m.def_submodule("internal", "Internal use.");
    m_int.def("pyogetsrc",  &pyogetsrc,  "Get python object from heap (used in passing/evaluating functions across the python/C++ boundary).");
    m_int.def("pyosetsrc",  &pyosetsrc,  "Set python object on heap (used in passing/evaluating functions across the python/C++ boundary).");
    m_int.def("genevalsrc", &genevalsrc, "Evaluate gentype on heap (used in passing/evaluating functions across the python/C++ boundary).");
    m_int.def("snakes", &callsnakes, "Snakes (test io, streams).");

    // ---------------------------

    auto m_maths = m.def_submodule("maths", "Mathematics related.");
    m_maths.def("eval", &svmeval, "eval(fn,x) - gentype mirror of pythons eval function fn (specified as a string with explicit x, eg sin(x)) using arguments given.");
    m_maths.def("calc", &callintercalc, "Calculator (explore functions are available in svmheavy maths).");

    // ---------------------------

    auto m_ml = m.def_submodule("ml", "Machine Learning Modules.");
    QGETSETD(m_ml,getMLType,ssetMLTypeClean,"type","settype", "set ML i type. Types are:\n"
                                             "\n"
                                             " r   - SVM: Scalar regression.\n"
                                             " s   - SVM: single class.\n"
                                             " c   - SVM: binary classification (default).\n"
                                             " m   - SVM: multiclass classification.\n"
                                             " r   - SVM: scalar regression.\n"
                                             " v   - SVM: vector regression.\n"
                                             " a   - SVM: anionic regression.\n"
                                             " u   - SVM: cyclic regression.\n"
                                             " g   - SVM: gentype regression (any target).\n"
                                             " p   - SVM: density estimation (1-norm base, kernel can be non-Mercer.\n"
                                             " t   - SVM: pareto frontier SVM.\n"
                                             " l   - SVM: binary scoring (zero bias by default).\n"
                                             " o   - SVM: scalar regression with scoring.\n"
                                             " q   - SVM: vector regression with scoring.\n"
                                             " i   - SVM: planar regression.\n"
                                             " h   - SVM: multi-expert ranking.\n"
                                             " j   - SVM: multi-expert binary classification.\n"
                                             " b   - SVM: similarity learning.\n"
                                             " d   - SVM: basic SVM for kernel inheritance (-x).\n"
                                             " R   - SVM: scalar regression using random FF (kernels 3,4,13,19).\n"
                                             " B   - SVM: binary classifier using random FF (kernels 3,4,13,19).\n"
                                             " lsr - LS-SVM: scalar regression.\n"
                                             " lsc - LS-SVM: binary classification.\n"
                                             " lsv - LS-SVM: vector regression.\n"
                                             " lsa - LS-SVM: anionic regression.\n"
                                             " lsg - LS-SVM: gentype regression.\n"
                                             " lsi - LS-SVM: planar regression.\n"
                                             " lso - LS-SVM: scalar regression with scoring.\n"
                                             " lsq - LS-SVM: vector regression with scoring.\n"
                                             " lsh - LS-SVM: multi-expert ranking.\n"
                                             " lsR - LS-SVM: scalar regression random FF (kernels 3,4,13,19).\n"
                                             " gpr - GPR: gaussian process scalar regression.\n"
                                             " gpc - GPR: gaussian process binary classification (unreliable).\n"
                                             " gpv - GPR: gaussian process vector regression.\n"
                                             " gpa - GPR: gaussian process anionic regression.\n"
                                             " gpg - GPR: gaussian process gentype regression.\n"
                                             " gpR - GPR: gaussian process scalar regression RFF (kernels 3,4,13,19).\n"
                                             " gpC - GPR: gaussian process binary classify RFF (kernels 3,4,13,19).\n"
                                             " knc - KNN: binary classification.\n"
                                             " knm - KNN: multiclass classification.\n"
                                             " knr - KNN: scalar regression.\n"
                                             " knv - KNN: vector regression.\n"
                                             " kna - KNN: anionic regression.\n"
                                             " kng - KNN: gentype regression.\n"
                                             " knp - KNN: density estimation.\n"
                                             " ei  - IMP: expected (hypervolume) improvement.\n"
                                             " svm - IMP: 1-norm 1-class modded SVM mono-surrogate.\n"
                                             " rls - IMP: Random linear scalarisation.\n"
                                             " rns - IMP: Random draw from a GP transformed into an increasing function on [0,1]^d.\n"
                                             " nop - BLK: NOP machine.\n"
                                             " mer - BLK: Mercer kernel inheritance block.\n"
                                             " con - BLK: consensus machine.\n"
                                             " fna - BLK: user function machine (elementwise).*\n"
                                             " fnb - BLK: user function machine (vectorwise).*\n"
                                             " mxa - BLK: mex function machine (elementwise).\n"
                                             " mxb - BLK: mex function machine (vectorwise).\n"
                                             " io  - BLK: user I/O machine.\n"
                                             " sys - BLK: system call machine.\n"
                                             " avr - BLK: scalar averaging machine.\n"
                                             " avv - BLK: vector averaging machine.\n"
                                             " ava - BLK: anionic averaging machine.\n"
                                             " fcb - BLK: function callback (do not use).\n"
                                             " ber - BLK: Bernstein basis polynomial.\n"
                                             " bat - BLK: Battery model.**\n"
                                             " ker - BLK: kernel specialisation.***\n"
                                             " mba - BLK: multi-block sum.");

    QGETSETD(m_ml,getMLType,ssetMLTypeClean,"type","settype", "ML type");

    QDO(m_ml,train,"train");
    QDO(m_ml,reset,"reset (undo training, alpha,bias = 0)");
    QDO(m_ml,restart,"restart (removing training data and reset)");
    QDOARG(m_ml,scale,"scale (alpha,bias) by factor given");
    QDOARG(m_ml,randomise,"randomise (alpha,bias)");

    QGETSET(m_ml,prim,setprim,"prior mean type (0 none (0), 1 mu(x) given directly, 2 mu_j(x) is posterior mean of model j)");
    QGETSET(m_ml,prival,setprival,"explicit SVM i prior mean (assuming prim(i) is 1)");
    m_ml.def("setpriml",&setpriml,"setpriml(i,j): set SVM i prior mean to posterior mean of ML j (i=-1 for current ML).");

    QGETSET(m_ml,tspaceDim,settspaceDim,"target space dimension");
    QGETSET(m_ml,order,    setorder,    "target space order");

    QGET(m_ml,N,                 "number of training vectors");
    QGET(m_ml,type,              "ML type number");
    QGET(m_ml,subtype,           "ML subtype number");
    QGET(m_ml,xspaceDim,         "input space dimension");
    QGET(m_ml,fspaceDim,         "function space dimension");
    QGET(m_ml,tspaceSparse,      "is target space sparse");
    QGET(m_ml,xspaceSparse,      "is input space sparse");
    QGET(m_ml,numClasses,        "number of classes");
    QGET(m_ml,ClassLabels,       "labels of classes");
    QGET(m_ml,isTrained,         "is ML trained");
    QGET(m_ml,isSolGlob,         "is solution global");
    QGET(m_ml,isUnderlyingScalar,"is underlying scalar type");
    QGET(m_ml,isUnderlyingVector,"is underlying vector type");
    QGET(m_ml,isUnderlyingAnions,"is underlying anionic type");
    QGET(m_ml,isClassifier,      "is ML a classifier");
    QGET(m_ml,isRegression,      "is ML a regressor");
    QGET(m_ml,isPlanarType,      "is ML a planar-type method");

    QGETCLA(m_ml,NNC,"number of active training vectors for class d");

    QGET(m_ml,x,          "class");
    QGET(m_ml,d,          "class");
    QGET(m_ml,y,          "target");
    QGET(m_ml,yp,         "prior");
    QGET(m_ml,Cweight,    "C weight");
    QGET(m_ml,Cweightfuzz,"C weight fuzzing");
    QGET(m_ml,sigmaweight,"sigma weight");
    QGET(m_ml,epsweight,  "eps weight");
    QGET(m_ml,alphaState, "alpha state");
    QGET(m_ml,xtang,      "class/vector specifics");
    m_ml.def("alpha",&mlalpha,"alpha(i): alpha for ML i");
    m_ml.def("bias", &mlbias, "bias(i): bias for ML i" );
    m_ml.def("setalpha",&mlsetalpha,"setalpha(i): set alpha for ML i");
    m_ml.def("setbias", &mlsetbias, "setbias(i): set bias for ML i" );


    m_ml.def("add",  &addTrainingVectorml,     "add(i,j,z,x): add training vector pair (z,x) to ML i at position j (i=-1 for current ML).");
    m_ml.def("addm", &maddTrainingVectorml,    "add(i,j,z,x): add multiple training vector pairs (z[k],x[k]) to ML i at position j (i=-1 for current ML).");
    m_ml.def("add",  &detaddTrainingVectorml,  "add(i,j,z,x,cw,ew,d): add training vector pair (z,x:cw,ew) (where cw and ew are the C and eps weight) in class d to ML i at position j (i=-1 for current ML).");
    m_ml.def("addm", &detmaddTrainingVectorml, "add(i,j,z,x,cw,ew,d): add multiple training vector pairs (z,x:cw,ew) (where cw and ew are the C and eps weights) to ML i at position j (i=-1 for current ML).");

    m_ml.def("remove", &removeTrainingVectorml, "remove(i,j,num): remove num training vector pairs from ML i at position j (i=-1 for current ML).");

    m_ml.def("mu",  &muml,  "mu(i,x): calculate posterior mean / output of ML i given input x, which is either a sparse vector or an integer (training vector number) (i=-1 for current ML).");
    m_ml.def("mug", &mugml, "mug(i,x): calculate underlying (ie continuous, even for a classification problem) posterior mean / output of ML i (i=-1 for current ML).");
    m_ml.def("var", &varml, "var(i,x): calculate posterior variance of ML i given input x, which is either a sparse vector or an integer (training vector number) (i=-1 for current ML).");
    m_ml.def("cov", &covml, "cov(i,x,y): calculate posterior covariance of ML i given inputs x,y, which are either sparse vectors or integers (training vector numbers) (i=-1 for current ML).");

    // ---------------------------

    auto m_ml_svm = m_ml.def_submodule("svm", "Support Vector Machines models.");
    QGETSETD(m_ml_svm,getMLType,ssetMLTypeClean,"type","settype", "set SVM i type. Types are:\n"
                                             "\n"
                                             " r - SVM: Scalar regression.\n"
                                             " s - SVM: single class.\n"
                                             " c - SVM: binary classification (default).\n"
                                             " m - SVM: multiclass classification.\n"
                                             " r - SVM: scalar regression.\n"
                                             " v - SVM: vector regression.\n"
                                             " a - SVM: anionic regression.\n"
                                             " u - SVM: cyclic regression.\n"
                                             " g - SVM: gentype regression (any target).\n"
                                             " p - SVM: density estimation (1-norm base, kernel can be non-Mercer.\n"
                                             " t - SVM: pareto frontier SVM.\n"
                                             " l - SVM: binary scoring (zero bias by default).\n"
                                             " o - SVM: scalar regression with scoring.\n"
                                             " q - SVM: vector regression with scoring.\n"
                                             " i - SVM: planar regression.\n"
                                             " h - SVM: multi-expert ranking.\n"
                                             " j - SVM: multi-expert binary classification.\n"
                                             " b - SVM: similarity learning.\n"
                                             " d - SVM: basic SVM for kernel inheritance (-x).\n"
                                             " R - SVM: scalar regression using random FF (kernels 3,4,13,19).\n"
                                             " B - SVM: binary classifier using random FF (kernels 3,4,13,19).");
    QGETSETD(m_ml_svm,getVmethod,setVmethod,"typeVR","settypeVR", "set SVM i vector-regression method.  Methods are:\n"
                                             "\n"
                                             " once - at-once regression.\n"
                                             " red  - reduction to binary regression (default).");
    QGETSETD(m_ml_svm,getCmethod,setCmethod,"typeMC","settypeMC", "set SVM i multi-class classification method.  Methods are:\n"
                                             "\n"
                                             " 1vsA   - 1 versus all (reduction to binary).\n"
                                             " 1vs1   - 1 versus 1 (reduction to binary).\n"
                                             " DAG    - directed acyclic graph (reduct to binary).\n"
                                             " MOC    - minimum output coding (reduct to binary).\n"
                                             " maxwin - max-wins SVM (at once).\n"
                                             " recdiv - recursive division SVM (at once, default).");
    QGETSETD(m_ml_svm,getOmethod,setOmethod,"typeOC","settypeOC", "set SVM i one-class method.  Methods are:\n"
                                             "\n"
                                             " sch - Scholkopt 1999 1-class SVM (default).\n"
                                             " tax - Tax and Duin 2004, Support Vector Data Description.");
    QGETSETD(m_ml_svm,getAmethod,setAmethod,"typeCM","settypeCM", "set SVM i classification method.  Methods are:\n"
                                             "\n"
                                             " svc - normal SVM classifier (default).\n"
                                             " svr - classify via regression.");
    QGETSETD(m_ml_svm,getRmethod,setRmethod,"typeER","settypeER", "set SVM i empirical risk type.  Methods are:\n"
                                             "\n"
                                             " l - linear (default).\n"
                                             " q - quadratic.\n"
                                             " o - linear, with 1-norm regularization on alpha (not feature space: use -m for that).\n"
                                             " g - generalised linear (iterative fuzzy).\n"
                                             " G - generalised quadratic (iterative fuzzy).");
    QGETSETD(m_ml_svm,getTmethod,setTmethod,"typeSM","settypeSM", "set SVM i tube shrinking method.  Methods are:\n"
                                             "\n"
                                             " f - fixed tube (default).\n"
                                             " s - tube shrinking.");
    QGETSETD(m_ml_svm,getBmethod,setBmethod,"typeBias","settypeBias", "set SVM i bias method.  Methods are:\n"
                                             "\n"
                                             " var - variable bias (default).\n"
                                             " fix - fixed bias (usually zero)."
                                             " pos - positive bias.\n"
                                             " neg - negative bias.\n");
    QGETSETD(m_ml_svm,getMmethod,setMmethod,"typeM","settypeM", "set SVM i monotonic method (sufficient, not necessary, and only for a few kernels in finite dimensions, assuming all training x >= 0).  Methods are:\n"
                                             "\n"
                                             " n - none (default).\n"
                                             " i - increasing.\n"
                                             " d - decreasing.");

    QGET(m_ml_svm,NZ, "number of training vectors with alpha = 0");
    QGET(m_ml_svm,NF, "number of training vectors with alpha unconstrained");
    QGET(m_ml_svm,NS, "number of training vectors with alpha != 0");
    QGET(m_ml_svm,NC, "number of training vectors with alpha constrained");
    QGET(m_ml_svm,NLB,"number of training vectors with alpha constrained at lower bound");
    QGET(m_ml_svm,NLF,"number of training vectors with alpha unconstrained between lower bound and zero");
    QGET(m_ml_svm,NUF,"number of training vectors with alpha unconstrained between zero and upper bound");
    QGET(m_ml_svm,NUB,"number of training vectors with alpha constrained at upper bound");

    QGET(m_ml_svm,kerndiag,"diagonals of kernel matrix")

    QGETSET(m_ml_svm,C,        setC,        "regularization trade-off (empirical risk weight)");
    QGETSET(m_ml_svm,sigma,    setsigma,    "regularization trade-off (regularization weight, lambda)");
    QGETSET(m_ml_svm,sigma_cut,setsigma_cut,"sigma scale for JIT sampling");
    QGETSET(m_ml_svm,eps,      seteps,      "epsilon-insensitivity width");
    QGETSET(m_ml_svm,m,        setm,        "margin-norm (default 2, or 2-kernel SVM)");
    QGETSET(m_ml_svm,theta,    settheta,    "theta (psd regularization) for similarity learning");
    QGETSET(m_ml_svm,simnorm,  setsimnorm,  "set normalized (1, default) or un-normalized (0) similarity learning");

    QGETSETCLA(m_ml_svm,Cclass,  setCclass,  "regularization trade-off (empirical risk weight) for class d");
    QGETSETCLA(m_ml_svm,epsclass,setepsclass,"epsilon-insensitivity width for class d");

    QGETSET(m_ml_svm,LinBiasForce, setLinBiasForce, "linear bias-forcing");
    QGETSET(m_ml_svm,QuadBiasForce,setQuadBiasForce,"quadratic bias-forcing");
    QGETSET(m_ml_svm,nu,           setnu,           "linear tube-shrinking constant");
    QGETSET(m_ml_svm,nuQuad,       setnuQuad,       "quadratic tube-shrinking constant");

    QGETSETCLA(m_ml_svm,LinBiasForceclass, setLinBiasForceclass, "linear bias-forcing for class d");
    QGETSETCLA(m_ml_svm,QuadBiasForceclass,setQuadBiasForceclass,"quadratic bias-forcing for class d");

    QGET(m_ml_svm,loglikelihood,"quasi-log-likelihood");
    QGET(m_ml_svm,maxinfogain,  "quasi-max-information-gain");
    QGET(m_ml_svm,RKHSnorm,     "RKHS norm ||f||_H^2");
    QGET(m_ml_svm,RKHSabs,      "RKHS norm ||f||_H");

    QGETSET(m_ml_svm,alpha,setAlpha,"alpha")
    QGETSET(m_ml_svm,bias, setBias, "bias")

    QDO(m_ml_svm,removeNonSupports,"remove all non-support vectors");
    QDOARG(m_ml_svm,trimTrainingSet,"trim training set to target size");

    // ---------------------------

    auto m_ml_lsv = m_ml.def_submodule("lsv", "Least-Squares Support Vector Machines models.");
    QGETSETD(m_ml_lsv,getMLType,ssetMLTypeClean,"type","settype", "set LSV i type. Types are:\n"
                                             "\n"
                                             " lsr - LS-SVM: scalar regression.\n"
                                             " lsc - LS-SVM: binary classification.\n"
                                             " lsv - LS-SVM: vector regression.\n"
                                             " lsa - LS-SVM: anionic regression.\n"
                                             " lsg - LS-SVM: gentype regression.\n"
                                             " lsi - LS-SVM: planar regression.\n"
                                             " lso - LS-SVM: scalar regression with scoring.\n"
                                             " lsq - LS-SVM: vector regression with scoring.\n"
                                             " lsh - LS-SVM: multi-expert ranking.\n"
                                             " lsR - LS-SVM: scalar regression random FF (kernels 3,4,13,19).");
    QGETSETD(m_ml_lsv,getBmethod,setBmethod,"typeBias","settypeBias", "set LSV i bias method.  Methods are:\n"
                                             "\n"
                                             " var - variable bias (default).\n"
                                             " fix - zero bias.");

    QGETSET(m_ml_lsv,C,        setC,        "regularization trade-off (empirical risk weight)");
    QGETSET(m_ml_lsv,sigma,    setsigma,    "regularization trade-off (regularization weight, lambda)");
    QGETSET(m_ml_lsv,sigma_cut,setsigma_cut,"sigma scale for JIT sampling");
    QGETSET(m_ml_lsv,eps,      seteps,      "epsilon-insensitivity width");

    QGETSETCLA(m_ml_lsv,Cclass,  setCclass,  "regularization trade-off (empirical risk weight) for class d");
    QGETSETCLA(m_ml_lsv,epsclass,setepsclass,"epsilon-insensitivity width for class d");

    QGET(m_ml_lsv,loglikelihood,"quasi-log-likelihood");
    QGET(m_ml_lsv,maxinfogain,  "quasi-max-information-gain");
    QGET(m_ml_lsv,RKHSnorm,     "RKHS norm ||f||_H^2");
    QGET(m_ml_lsv,RKHSabs,      "RKHS norm ||f||_H");

    QGETSET(m_ml_lsv,gamma,setgamma,"alpha")
    QGETSET(m_ml_lsv,delta,setdelta,"bias")

    // ---------------------------

    auto m_ml_gp = m_ml.def_submodule("gp", "Gaussian Process models.");
    QGETSETD(m_ml_gp,getMLType,ssetMLTypeClean,"type","settype", "set GP i type. Types are:\n"
                                             "\n"
                                             " gpr - GPR: gaussian process scalar regression.\n"
                                             " gpc - GPR: gaussian process binary classification (unreliable).\n"
                                             " gpv - GPR: gaussian process vector regression.\n"
                                             " gpa - GPR: gaussian process anionic regression.\n"
                                             " gpg - GPR: gaussian process gentype regression.\n"
                                             " gpR - GPR: gaussian process scalar regression RFF (kernels 3,4,13,19).\n"
                                             " gpC - GPR: gaussian process binary classify RFF (kernels 3,4,13,19).");
    QGETSETD(m_ml_gp,getBmethod,setBmethod,"typeBias","settypeBias", "set GP i bias method.  Methods are:\n"
                                             "\n"
                                             " var - variable bias.\n"
                                             " fix - zero bias (default).");

    QGETSET(m_ml_gp,sigma,    setsigma,    "measurement noise variance");
    QGETSET(m_ml_gp,sigma_cut,setsigma_cut,"measurement noise variance scale for JIT sampling");

    QGETSETCLA(m_ml_gp,C,setC,"measurement noise variance scale factor for class d (sigma -> sigma/Cd)");

    QGET(m_ml_gp,loglikelihood,"log-likelihood");
    QGET(m_ml_gp,maxinfogain,  "max-information-gain");
    QGET(m_ml_gp,RKHSnorm,     "RKHS norm ||f||_H^2");
    QGET(m_ml_gp,RKHSabs,      "RKHS norm ||f||_H");

    QGETSET(m_ml_gp,muWeight,setmuWeight,"alpha")
    QGETSET(m_ml_gp,muBias,  setmuBias,  "bias")

    // ---------------------------

    auto m_ml_knn = m_ml.def_submodule("knn", "Kernel nearest neighbours models.");
    QGETSETD(m_ml_knn,getMLType,ssetMLTypeClean,"type","settype", "set KNN i type. Types are:\n"
                                             "\n"
                                             " knc - KNN: binary classification.\n"
                                             " knm - KNN: multiclass classification.\n"
                                             " knr - KNN: scalar regression.\n"
                                             " knv - KNN: vector regression.\n"
                                             " kna - KNN: anionic regression.\n"
                                             " kng - KNN: gentype regression.\n"
                                             " knp - KNN: density estimation.");

    QGETCLA(m_ml_knn,NNC,"number of active training vectors for class d");

    QGETSET(m_ml_knn,k,  setk,  "number of neighbours");
    QGETSET(m_ml_knn,ktp,setktp,"weight function (see -K).");

    // ---------------------------

    auto m_ml_imp = m_ml.def_submodule("imp", "Impulse models.");
    QGETSETD(m_ml_imp,getMLType,ssetMLTypeClean,"type","settype", "set IMP i type. Types are:\n"
                                             "\n"
                                             " ei  - IMP: expected (hypervolume) improvement.\n"
                                             " svm - IMP: 1-norm 1-class modded SVM mono-surrogate.\n"
                                             " rls - IMP: Random linear scalarisation.\n"
                                             " rns - IMP: Random draw from a GP transformed into an increasing function on [0,1]^d.");

    // ---------------------------

    auto m_ml_blk = m_ml.def_submodule("blk", "Miscellaneous models.");
    QGETSETD(m_ml_blk,getMLType,ssetMLTypeClean,"type","settype", "set BLK i type. Types are:\n"
                                             "\n"
                                             " nop - BLK: NOP machine.\n"
                                             " mer - BLK: Mercer kernel inheritance block.\n"
                                             " con - BLK: consensus machine.\n"
                                             " fna - BLK: user function machine (elementwise).*\n"
                                             " fnb - BLK: user function machine (vectorwise).*\n"
                                             " mxa - BLK: mex function machine (elementwise).\n"
                                             " mxb - BLK: mex function machine (vectorwise).\n"
                                             " io  - BLK: user I/O machine.\n"
                                             " sys - BLK: system call machine.\n"
                                             " avr - BLK: scalar averaging machine.\n"
                                             " avv - BLK: vector averaging machine.\n"
                                             " ava - BLK: anionic averaging machine.\n"
                                             " fcb - BLK: function callback (do not use).\n"
                                             " ber - BLK: Bernstein basis polynomial.\n"
                                             " bat - BLK: Battery model.**\n"
                                             " ker - BLK: kernel specialisation.***\n"
                                             " mba - BLK: multi-block sum.");
}

void callintercalc(void)
{
    intercalc(outstream(),std::cin);
}

#define snakehigh 24
#define snakewide 80

py::object svmeval(std::string f, py::object x)
{
    gentype ff(f); // convert string to equation
    gentype xx; // this will hold x

    if ( convFromPy(xx,x) )
    {
        return convToPy(nan(""));
    }

    return convToPy(ff(xx)); // evaluate f(x) and convert back to python friendly form
}

void callsnakes(void) { snakes(snakewide,snakehigh,(20*snakehigh)/23,5,20,100000); }

void svmheavy(int method, int permode, const std::string commstr, int wml);

void svmheavya(void)                      { static std::string dummy; svmheavy(1,-1,dummy,-1);      }
void svmheavyb(int permode)               { static std::string dummy; svmheavy(2,permode,dummy,-1); }
void svmheavyc(const std::string commstr) {                           svmheavy(3,-1,commstr,-1);    }

int setpriml (int i, int j)
{
    i = glob_svmInd(i);

    return getMLref(getMLmodels(),i).setpriml(&(getMLrefconst(getMLmodels(),j).getMLconst()));
}

int addTrainingVectorml(int i, int j, py::object z, py::object x)
{
    i = glob_svmInd(i);

    gentype zz;
    SparseVector<gentype> xx;

    int errcode = 0;

    errcode |= convFromPy(zz,z);
    errcode |= convFromPy(xx,x);

    if ( errcode )
    {
        return 0;
    }

    return getMLref(getMLmodels(),i).addTrainingVector(j,zz,xx);
}

int maddTrainingVectorml(int i, int j, py::object z, py::object x)
{
    i = glob_svmInd(i);

    Vector<gentype> zz;
    Vector<SparseVector<gentype> > xx;

    int errcode = 0;

    errcode |= convFromPy(zz,z);
    errcode |= convFromPy(xx,x);

    if ( errcode || ( zz.size() != xx.size() ) )
    {
        return 0;
    }

    Vector<double> cw(zz.size());
    Vector<double> ew(zz.size());

    cw = 1.0;
    ew = 1.0;

    return getMLref(getMLmodels(),i).addTrainingVector(j,zz,xx,cw,ew);
}

int detaddTrainingVectorml (int i, int j, py::object z, py::object x, py::object Cweigh, py::object epsweigh, py::object d)
{
    i = glob_svmInd(i);

    gentype zz;
    SparseVector<gentype> xx;
    double cw;
    double ew;
    int dd;

    int errcode = 0;

    errcode |= convFromPy(zz,z);
    errcode |= convFromPy(xx,x);
    errcode |= convFromPy(cw,Cweigh);
    errcode |= convFromPy(ew,epsweigh);
    errcode |= convFromPy(dd,d);

    if ( errcode )
    {
        return 0;
    }

    return getMLref(getMLmodels(),i).addTrainingVector(j,zz,xx,cw,ew,dd);
}

int detmaddTrainingVectorml(int i, int j, py::object z, py::object x, py::object Cweigh, py::object epsweigh)
{
    i = glob_svmInd(i);

    Vector<gentype> zz;
    Vector<SparseVector<gentype> > xx;
    Vector<double> cw;
    Vector<double> ew;

    int errcode = 0;

    errcode |= convFromPy(zz,z);
    errcode |= convFromPy(xx,x);
    errcode |= convFromPy(cw,Cweigh);
    errcode |= convFromPy(ew,epsweigh);

    if ( errcode || ( zz.size() != xx.size() ) || ( zz.size() != cw.size() ) || ( zz.size() != ew.size() ) )
    {
        return 0;
    }

    return getMLref(getMLmodels(),i).addTrainingVector(j,zz,xx,cw,ew);
}

int removeTrainingVectorml(int i, int j, int num)
{
    i = glob_svmInd(i);

    return getMLref(getMLmodels(),i).removeTrainingVector(j,num);
}

py::object muml(int i, py::object x)
{
    i = glob_svmInd(i);

    gentype resh,resg;

    if ( py::isinstance<py::int_>(x) )
    {
        int j = py::cast<py::int_>(x);

        getMLref(getMLmodels(),i).ghTrainingVector(resh,resg,j);
    }

    else
    {
        SparseVector<gentype> xx;

        if ( convFromPy(xx,x) )
        {
            return py::none();
        }

        getMLref(getMLmodels(),i).gh(resh,resg,xx);
    }

    return convToPy(resh);
}

py::object mugml(int i, py::object x)
{
    i = glob_svmInd(i);

    gentype resh,resg;

    if ( py::isinstance<py::int_>(x) )
    {
        int j = py::cast<py::int_>(x);

        getMLref(getMLmodels(),i).ghTrainingVector(resh,resg,j);
    }

    else
    {
        SparseVector<gentype> xx;

        if ( convFromPy(xx,x) )
        {
            return py::none();
        }

        getMLref(getMLmodels(),i).gh(resh,resg,xx);
    }

    return convToPy(resg); // this is the difference from muml
}

py::object varml(int i, py::object x)
{
    i = glob_svmInd(i);

    gentype resv,resmu;

    if ( py::isinstance<py::int_>(x) )
    {
        int j = py::cast<py::int_>(x);

        getMLref(getMLmodels(),i).varTrainingVector(resv,resmu,j);
    }

    else
    {
        SparseVector<gentype> xx;

        if ( convFromPy(xx,x) )
        {
            return py::none();
        }

        getMLref(getMLmodels(),i).var(resv,resmu,xx);
    }

    return convToPy(resv);
}

py::object covml(int i, py::object x, py::object y)
{
    i = glob_svmInd(i);

    gentype resv,resmu;

    if ( py::isinstance<py::int_>(x) && py::isinstance<py::int_>(y) )
    {
        int j = py::cast<py::int_>(x);
        int k = py::cast<py::int_>(y);

        getMLref(getMLmodels(),i).covTrainingVector(resv,resmu,j,k);
    }

    else if ( py::isinstance<py::int_>(x) )
    {
        int j = py::cast<py::int_>(x);

        const SparseVector<gentype> &xx = getMLref(getMLmodels(),i).x(j);
        SparseVector<gentype> yy;

        if ( convFromPy(yy,y) )
        {
            return py::none();
        }

        getMLref(getMLmodels(),i).cov(resv,resmu,xx,yy);
    }

    else if ( py::isinstance<py::int_>(y) )
    {
        int k = py::cast<py::int_>(y);

        SparseVector<gentype> xx;
        const SparseVector<gentype> &yy = getMLref(getMLmodels(),i).x(k);

        if ( convFromPy(xx,x) )
        {
            return py::none();
        }

        getMLref(getMLmodels(),i).cov(resv,resmu,xx,yy);
    }

    else
    {
        SparseVector<gentype> xx;
        SparseVector<gentype> yy;

        if ( convFromPy(xx,x) || convFromPy(yy,y) )
        {
            return py::none();
        }

        getMLref(getMLmodels(),i).cov(resv,resmu,xx,yy);
    }

    return convToPy(resv);
}

py::object mlalpha(int i)
{
    i = glob_svmInd(i);

    Vector<gentype> res;

    int type = getMLref(getMLmodels(),i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { res = getMLref(getMLmodels(),i).alpha();    }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { res = getMLref(getMLmodels(),i).muWeight(); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { res = getMLref(getMLmodels(),i).gamma();    }

    return convToPy(res);
}

py::object mlbias(int i)
{
    i = glob_svmInd(i);

    gentype res('N');

    int type = getMLref(getMLmodels(),i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { res = getMLref(getMLmodels(),i).bias();   }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { res = getMLref(getMLmodels(),i).muBias(); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { res = getMLref(getMLmodels(),i).delta();  }

    return convToPy(res);
}

int mlsetalpha(int i, py::object src)
{
    int errcode = 0;

    i = glob_svmInd(i);

    Vector<gentype> altsrc;

    if ( ( errcode = convFromPy(altsrc,src) ) )
    {
        return errcode;
    }

    int type = getMLref(getMLmodels(),i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { getMLref(getMLmodels(),i).setAlpha(altsrc);    }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { getMLref(getMLmodels(),i).setmuWeight(altsrc); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { getMLref(getMLmodels(),i).setgamma(altsrc);    }

    else
    {
        return 1;
    }

    return 0;
}

int mlsetbias(int i, py::object src)
{
    int errcode = 0;

    i = glob_svmInd(i);

    gentype altsrc('N');

    if ( ( errcode = convFromPy(altsrc,src) ) )
    {
        return errcode;
    }

    int type = getMLref(getMLmodels(),i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { getMLref(getMLmodels(),i).setBias(altsrc);   }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { getMLref(getMLmodels(),i).setmuBias(altsrc); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { getMLref(getMLmodels(),i).setdelta(altsrc);  }

    else
    {
        return 1;
    }

    return 0;
}

















int isCallable(py::object src)
{
    static py::object builtins = py::module_::import("builtins");
    static py::object isinstance_function = builtins.attr("isinstance");
    static py::object callable_function = builtins.attr("callable");

    // if not bool let exception be thrown
    return py::cast<py::bool_>(callable_function(src));
}

int isComplex(py::object src)
{
    static py::object builtins = py::module_::import("builtins");
    static py::object isinstance_function = builtins.attr("isinstance");
    static py::object complex_class = builtins.attr("complex");

    // if not bool let exception be thrown
    return py::cast<py::bool_>(isinstance_function(src,complex_class));
}

py::object convToPy(const int &src)
{
    return py::cast(src);
}

py::object convToPy(const double &src)
{
    return py::cast(src);
}

py::object convToPy(const d_anion &src)
{
    if ( src.order() <= 1 )
    {
        return py::cast((std::complex<double>) src);
    }

    return py::cast(nan(""));
}

py::object convToPy(const std::string &src)
{
    return py::cast(src.c_str());
}

template <class T>
py::object convToPy(const Vector<T> &src)
{
    int vsize = src.size();

    py::list vres(vsize);

    for ( int i = 0 ; i < vsize ; ++i )
    {
        vres[i] = convToPy(src(i));
    }

    return vres;
}

template <class T>
py::object convToPy(const Set<T> &src)
{
    int vsize = src.size();

    py::tuple res(vsize);

    for ( int i = 0 ; i < vsize ; ++i )
    {
        res[i] = convToPy(src.all()(i));
    }

    return res;
}

template <class T>
py::object convToPy(const SparseVector<T> &src)
{
    gentype altsrc(src);

    return convToPy(altsrc);
}

py::object convToPy(const gentype &src)
{
    if ( src.isValNull() )
    {
        return py::none();
    }

    else if ( src.isValInteger() )
    {
        return convToPy((int) src);
    }

    else if ( src.isValReal() )
    {
        return convToPy((double) src);
    }

    else if ( src.isValAnion() )
    {
        return convToPy((const d_anion &) src);
    }

    else if ( src.isValString() )
    {
        return convToPy((const std::string &) src);
    }

    else if ( src.isValEqnDir() )
    {
        int i = -1;

        // Store function

        i = gensetsrc(i,src);

        // Construct python lambda expression that calls this function

        std::string fn;

        fn = "(lambda x : pyheavy.internal.genevalsrc(";
        fn += std::to_string(i);
        fn += ",x))";

        // Evaluated command to create function pointer

        static py::object builtins = py::module_::import("builtins");
        static py::object eval_function = builtins.attr("eval");

        return eval_function(fn);
    }

    else if ( src.isValVector() )
    {
        return convToPy((const Vector<gentype> &) src);
    }

    else if ( src.isValSet() )
    {
        return convToPy((const Set<gentype> &) src);
    }

    return py::cast(nan(""));
}










template <class T>
int convFromPy(T &res, py::handle src); // needed helper for raw python data

int convFromPy(int &res, py::object src)
{
    if ( py::isinstance<py::int_>(src) )
    {
        res = py::cast<py::int_>(src);
        return 0;
    }

    else if ( py::isinstance<py::float_>(src) )
    {
        double tmpres;
        tmpres = py::cast<py::float_>(src);
        res = (int) tmpres;
        return 0;
    }

    res = 0;
    return 1;
}

int convFromPy(double &res, py::object src)
{
    if ( py::isinstance<py::int_>(src) )
    {
        int tmpres = 0;
        tmpres = py::cast<py::int_>(src);
        res = tmpres;
        return 0;
    }

    else if ( py::isinstance<py::float_>(src) )
    {
        res = py::cast<py::float_>(src);
        return 0;
    }

    res = nan("");
    return 1;
}

int convFromPy(d_anion &res, py::object src)
{
    if ( isComplex(src) )
    {
        res = py::cast<std::complex<double> >(src);
        return 0;
    }

    res = nan("");
    return 1;
}

int convFromPy(std::string &res, py::object src)
{
    if ( py::isinstance<py::str>(src) )
    {
        res = py::cast<py::str>(src);
        return 0;
    }

    res = "";
    return 1;
}

template <class T>
int convFromPy(Vector<T> &res, py::object src)
{
    if ( py::isinstance<py::list>(src) )
    {
        py::list altsrc = py::cast<py::list>(src);

        res.resize((int) altsrc.size());

        int errcode = 0;
        int i = 0;

        for ( auto elm : altsrc )
        {
            errcode |= convFromPy(res("&",i),elm);
            ++i;
        }

        return errcode;
    }

    res.resize(0);
    return 1;
}

template <class T>
int convFromPy(Set<T> &res, py::object src)
{
    Set<gentype> temp;
    res = temp;

    if ( py::isinstance<py::tuple>(src) )
    {
        py::tuple altsrc = py::cast<py::tuple>(src);

        int errcode = 0;

        for ( auto elm : altsrc )
        {
            T tg;

            errcode |= convFromPy(tg,elm);
            res.add(tg); // will add to end
        }

        return errcode;
    }

    return 1;
}

template <class T>
int convFromPy(SparseVector<T> &res, py::object src)
{
    gentype altsrc;

    int errcode = convFromPy(altsrc,src); // sparsevector gets encoded into altres

    if ( !errcode )
    {
        if ( altsrc.isCastableToVectorWithoutLoss() )
        {
            res = (const SparseVector<gentype> &) altsrc;
        }

        else
        {
            errcode = 1;
        }
    }

    if ( errcode )
    {
        res.zero();
    }

    return errcode;
}

int convFromPy(gentype &res, py::object src)
{
    int errcode = 0;

    if ( src.is_none() )
    {
        res.force_null();
    }

    else if ( py::isinstance<py::int_>(src) )
    {
        errcode = convFromPy(res.force_int(),src);
    }

    else if ( py::isinstance<py::float_>(src) )
    {
        errcode = convFromPy(res.force_double(),src);
    }

    else if ( py::isinstance<py::str>(src) )
    {
        errcode = convFromPy(res.force_string(),src);
    }

    else if ( py::isinstance<py::list>(src) )
    {
        errcode = convFromPy(res.force_vector(),src);
    }

    else if ( py::isinstance<py::tuple>(src) )
    {
        errcode = convFromPy(res.force_set(),src);
    }

    else if ( isComplex(src) )
    {
        errcode = convFromPy(res.force_anion(),src);
    }

    else if ( isCallable(src) )
    {
        int i = -1;

        // Store function

        i = pyosetsrc(i,src);

        // Construct gentype expression that calls this function

        std::string fn;

        fn = "pycall(\"(pyheavy.internal.pyogetsrc(";
        fn += std::to_string(i);
        fn += "))\",x)";

        gentype altres(fn);

        res = altres;
    }

    else
    {
        errcode = 1;
    }

    return errcode;
}

template <class T>
int convFromPy(T &res, py::handle h)
{
    return convFromPy(res,py::reinterpret_borrow<py::object>(h));
}
























int cligetsetExtVar(gentype &res, const gentype &src, int num);
int cligetsetExtVar(gentype &res, const gentype &src, int num)
{
    (void) res;
    (void) src;
    (void) num;

    // Do nothing: this means you can throw logs out and not cause issues

    return -1;
}

#define LOGOUTTOFILE 1
#define LOGERRTOFILE 1

// Print state: 0 no output to stream (but file still done)
//              1 output to stream and file

int pyAllowPrintOut(int mod = -1);
int pyAllowPrintErr(int mod = -1);

void cliCharPrintOut(char c);
void cliCharPrintErr(char c);

void cliPrintToOutLog(char c, int mode = 0);
void cliPrintToErrLog(char c, int mode = 0);

// Get ML models and current index

//SparseVector<ML_Mutable *> &getMLmodels(void);





// Python function
//
// method: 1 - print help
//         2 - set persistence mode (permode)
//         3 - run commands (in ML wml if wml != -1)

void svmheavy(int method, int permode, const std::string commstr, int wml)
{
    static thread_local int hasbeeninit = 0;
    static thread_local int persistenceset = 0;
    static thread_local int persistencereq = 1;

    isMainThread(1);

    try
    {
        std::string commline;

        // Initialisation of static, overall state, set-once type, streamy stuff

        if ( !hasbeeninit )
        {
            // Register "time 0"

            TIMEABSSEC(TIMECALL);
        }

        // Set up streams (we divert outputs to logfiles)

        void(*xcliCharPrintErr)(char c) = cliCharPrintErr;
        static LoggingOstreamErr clicerr(xcliCharPrintErr);
        seterrstream(&clicerr);

        void(*xcliCharPrintOut)(char c) = cliCharPrintOut;
        static LoggingOstreamOut clicout(xcliCharPrintOut);
        setoutstream(&clicout);

        suppresserrstreamcout();
        pyAllowPrintErr(0);

        // Print help if no commands given, or too many
        //
        // Assumption: either no arguments or just one

        if ( 1 == method )
        {
            outstream() << "SVMheavy 7.0                                                                  \n";
            outstream() << "============                                                                  \n";
            outstream() << "                                                                              \n";
            outstream() << "Copyright: all rights reserved.                                               \n";
            outstream() << "Author: Alistair Shilton                                                      \n";
            outstream() << "                                                                              \n";
            outstream() << "Basic operation: pyheavy.exec(\"commands\")                                     \n";
            outstream() << "                                                                              \n";
            outstream() << "Example:                                                                      \n";
            outstream() << "                                                                              \n";
            outstream() << ">>> import pyheavy,pyheavypy,math                                             \n";
            outstream() << ">>> pyheavy.exec(\"-ECHO pycall(\\\"math.sin\\\",5/pi)\")                           \n";
            outstream() << "                                                                              \n";
            outstream() << "Translation rules:                                                            \n";
            outstream() << "                                                                              \n";
            outstream() << "- null (svmheavy) translates to None                                          \n";
            outstream() << "- int, double and string remain unchanged                                     \n";
            outstream() << "- complex (1st order anion, svmheavy) translates to complex.                  \n";
            outstream() << "- vector [ a b ... ] (svmheavy) translates to list [ a, b, ... ]              \n";
            outstream() << "- set { a b ... } (svmheavy) translates to tuple ( a, b, ... )                \n";
            outstream() << "                                                                              \n";
            outstream() << "Note that sets containing a single  element translate to a tuple of 1 and, due\n";
            outstream() << "to pythons rules for tuples, get downgraded to scalars.                       \n";
            outstream() << "                                                                              \n";
            outstream() << "Persistence: by default calls to  svmheavy are independent of each other. This\n";
            outstream() << "can be changed by turning on  persistence.  When persistence is on all MLs are\n";
            outstream() << "retained in memory between calls, allowing multiple operations on the ML.  For\n";
            outstream() << "example this  can be used  to test different  parameter settings, or  retain a\n";
            outstream() << "trained ML in memory for use.                                                 \n";
            outstream() << "                                                                              \n";
            outstream() << "Turn off persistence: pyheavy.mode(0)                                         \n";
            outstream() << "Turn on persistence:  pyheavy.mode(1)  (default)                              \n";
            outstream() << "                                                                              \n";
            outstream() << "Multiple MLs: multiple  MLs can  run simultaneously  (see -?? for  details, in\n";
            outstream() << "particular  the  -q...  commands) in  parallel.  To  simplify  operation  when\n";
            outstream() << "multiple MLs are  present an optional  second argument may  be used to specify\n";
            outstream() << "which ML is being addressed by the  command string.  The syntax for this is as\n";
            outstream() << "follows:                                                                      \n";
            outstream() << "                                                                              \n";
            outstream() << "pyheavy.execmod(\"commands\",mlnum)                                             \n";
            outstream() << "                                                                              \n";
            outstream() << "which runs  the command \"-qw mlnum -Zx\"  before executing the  commands given.\n";
            outstream() << "Use -1 to leave number unchanged.                                             \n";

            return;
        }

        if ( 2 == method )
        {
            if ( permode == 0 )
            {
                persistencereq = 0;
            }

            else if ( permode == 1 )
            {
                persistencereq = 0;
            }
        }

        if ( ( 3 == method ) && ( -1 != wml ) )
        {
            // Add prefix to command

            std::ostringstream oss;

            oss << wml;

            commline += "-qw ";
            commline += oss.str();
            commline += " -Zx ";
        }

        // If currently not persistent and persistence requested then turn on

        if ( !persistenceset && persistencereq )
        {
            outstream() << "Locking ML stack...\n";

            persistenceset = 1;
        }

        // Convert the command line arguments into a command string

        commline += commstr;

        // Add -Zx to the end of the command string to ensure that the output
        // stream used by -echo will remain available until the end.

        commline += " -Zx";
        outstream() << "Running command: " << commline << "\n";

        // Define global variable store

        static thread_local svmvolatile SparseVector<SparseVector<gentype> > globargvariables;

        // Construct command stack.  All commands must be in awarestream, which
        // is similar to a regular stream but can supply commands from a
        // variety of different sources: for example a string (as here), a stream
        // such as standard input, or various ports etc.  You can then open
        // further awarestreams, which are stored on the stack, with the uppermost
        // stream being the active stream from which current commands are sourced.

        Stack<awarestream *> *commstack;
        MEMNEW(commstack,Stack<awarestream *>);
        std::stringstream *commlinestring;
        MEMNEW(commlinestring,std::stringstream(commline));
        awarestream *commlinestringbox;
        MEMNEW(commlinestringbox,awarestream(commlinestring,1));
        commstack->push(commlinestringbox);

//        static thread_local int svmInd = 0;
        static thread_local SVMThreadContext *svmContext;
        SparseVector<ML_Mutable *> &svmbase = getMLmodels();
        MEMNEW(svmContext,SVMThreadContext(glob_svmInd()));
        errstream() << "{";

        // Now that everything has been set up so we can run the actual code.

        SparseVector<SparseVector<int> > returntag;

        runsvm(svmContext,svmbase,commstack,globargvariables,cligetsetExtVar,returntag);

        // Unlock the thread, signalling that the context can be deleted etc

        errstream() << "}";

        MEMDEL(commstack);

        // If currently persistent and persistence not requested then turn off

        if ( persistenceset && !persistencereq )
        {
            outstream() << "Unlocking ML stack...\n";

            persistenceset = 0;
        }

        // Delete everything if not persistent

        if ( !persistenceset )
        {
            outstream() << "Removing ML stack...\n";

            // Delete the thread SVM context and remove from vector.

            killallthreads(svmContext);

            deleteMLs(svmbase);

            cliPrintToOutLog('*',1);
            cliPrintToErrLog('*',1);

            hasbeeninit = 0;
        }

        else
        {
            hasbeeninit = 1;
        }
    }

    catch ( const char *errcode )
    {
        outstream() << "Unknown error: " << errcode << ".\n";
        return;
    }

    catch ( const std::string errcode )
    {
        outstream() << "Unknown error: " << errcode << ".\n";
        return;
    }

    isMainThread(0);

    return;
}












int pyAllowPrintOut(int mod)
{
    static int allowprint = 1;

    int tempstat = allowprint;

    if ( mod == 0 ) { allowprint = 0; }
    if ( mod == 1 ) { allowprint = 1; }

    return tempstat;
}

int pyAllowPrintErr(int mod)
{
    static int allowprint = 0;

    int tempstat = allowprint;

    if ( mod == 0 ) { allowprint = 0; }
    if ( mod == 1 ) { allowprint = 1; }

    return tempstat;
}



void cliCharPrintOut(char c)
{
    cliPrintToOutLog(c);

#ifndef HEADLESS
    if ( !LoggingOstreamOut::suppressStreamCout && pyAllowPrintOut() )
    {
        std::cout << c;
    }
#endif

    return;
}

void cliCharPrintErr(char c)
{
    cliPrintToErrLog(c);

#ifndef HEADLESS
    if ( !LoggingOstreamErr::suppressStreamCout && pyAllowPrintErr() )
    {
        std::cerr << c;
    }
#endif

    return;
}

void cliPrintToOutLog(char c, int mode)
{
    // mode = 0: print char
    //        1: close file for exit

    if ( LOGOUTTOFILE )
    {
        static std::ofstream *outlog = nullptr;

        if ( !mode && !outlog )
        {
            outlog = new std::ofstream;

            NiceAssert(outlog);

            std::string outfname("svmheavy.out.log");
            std::string outfnamebase("svmheavy.out.log");

            int fcnt = 0;

            while ( fileExists(outfname) )
            {
                ++fcnt;

                std::stringstream ss;

                ss << outfnamebase;
                ss << ".";
                ss << fcnt;

                outfname = ss.str();
            }

            (*outlog).open(outfname.c_str());
        }

        if ( mode )
        {
            if ( outlog )
            {
                (*outlog).close();
                delete outlog;
                outlog = nullptr;
            }
        }

        else if ( outlog && !LoggingOstreamOut::suppressStreamFile )
        {
            static int bstring = 0;

            if ( c != '\b' )
            {
                bstring = 0;

                (*outlog) << c;
                (*outlog).flush();
            }

            else if ( !bstring )
            {
                bstring = 1;

                (*outlog) << '\n';
                (*outlog).flush();
            }
        }
    }

    return;
}

void cliPrintToErrLog(char c, int mode)
{
    // mode = 0: print char
    //        1: close file for exit

    if ( LOGERRTOFILE )
    {
        static std::ofstream *errlog = nullptr;

        if ( !mode && !errlog )
        {
            errlog = new std::ofstream;

            NiceAssert(errlog);

            std::string errfname("svmheavy.err.log");
            std::string errfnamebase("svmheavy.err.log");

            int fcnt = 0;

            while ( fileExists(errfname) )
            {
                ++fcnt;

                std::stringstream ss;

                ss << errfnamebase;
                ss << ".";
                ss << fcnt;

                errfname = ss.str();
            }

            (*errlog).open(errfname.c_str());
        }

        if ( mode )
        {
            if ( errlog )
            {
                (*errlog).close();
                delete errlog;
                errlog = nullptr;
            }
        }

        else if ( errlog && !LoggingOstreamErr::suppressStreamFile )
        {
            static int bstring = 0;

            if ( c != '\b' )
            {
                bstring = 0;

                (*errlog) << c;
                (*errlog).flush();
            }

            else if ( !bstring )
            {
                bstring = 1;

                (*errlog) << '\n';
                (*errlog).flush();
            }
        }
    }

    return;
}










SparseVector<ML_Mutable *> &getMLmodels(void)
{
    static thread_local SparseVector<ML_Mutable *> svmbase;

    return svmbase;
}







// To prevent truncation, arguments sent to pycall are stored in a series of
// arrays. These functions maintain static arrays that the c++ code can store
// values in and python can then access.

template <class T>
const T &setgetsrc(int &i, int doset, T *val = nullptr)
{
    static thread_local SparseVector<T *> xval;
    static SparseVector<int> useind;

    if ( doset == 2 )
    {
        if ( i == -1 )
        {
            while ( xval.size() )
            {
                i = xval.ind(0);
                setgetsrc(i,doset,val);
            }

            i = 0;
        }

        else
        {
            MEMDEL(xval("&",i));
            xval.zero(i);
            useind.zero(i);
        }
    }

    else if ( doset )
    {
        if ( i == -1 )
        {
            i = 0;

            while ( useind.isindpresent(i) ) { ++i; } // This is shared between all stores, so indices are unique.
        }

        MEMNEW(xval("&",i),T);

        *(xval("&",i)) = *val;
        useind("&",i) = 1;
    }

    return *(xval(i));
}

py::object pyogetsrc(int i) { return setgetsrc<py::object>(i,0); }
gentype    gengetsrc(int i) { return setgetsrc<gentype>(i,0); }

int pyosetsrc(int i, py::object src) { setgetsrc(i,1,&src); return i; }
int gensetsrc(int i, gentype    src) { setgetsrc(i,1,&src); return 1; }

void pyosrcreset(void) { int i = -1; setgetsrc<py::object>(i,2); }
void gensrcreset(void) { int i = -1; setgetsrc<gentype>(i,2); }

void pyosrcreset(int i) { setgetsrc<py::object>(i,2); }
void gensrcreset(int i) { setgetsrc<gentype>(i,2); }

// evaluate gentype function with x and evaluate result

py::object genevalsrc(int i, py::object xx)
{
    // Grab function f from heap

    gentype f = gengetsrc(i);

    // Cleanup

    gensrcreset(i);

    // Convert xx to gentype

    gentype x;

    if ( convFromPy(x,xx) )
    {
        return convToPy(nan(""));
    }

    // Evaluate f(x)

    gentype res = f(x);

    // Convert and return

    return convToPy(res);
}

// evaluate py::object function with x and evaluate result

gentype pyoevalsrc(int i, gentype xx)
{
    // Convert xx to py::object

    py::object x = convToPy(xx);

    // Put x onto stack

    int j = -1;

    j = pyosetsrc(j,x);

    // Evaluate f(x) via call
    //
    // res = pyogetsrc(i)(pyogetsrc(j))

    gentype res;

    std::string evalfn;

    evalfn =  "pyogetsrc(";
    evalfn += std::to_string(i);
    evalfn += ")(pyogetsrc(";
    evalfn += std::to_string(j);
    evalfn += "))";

    // Evaluated run command

    static py::object builtins = py::module_::import("builtins");
    static py::object eval_function = builtins.attr("eval");

    py::object resultobj = eval_function(evalfn);

    // Cleanup

    gensrcreset(i);
    gensrcreset(j);

    // Retrieve results of operation

    if ( convFromPy(res,resultobj) )
    {
        res.force_double() = nan("");
    }

    // Return

    return res;
}

// drop-in replacement for pycall function in gentype.cc
// (the gentype version, which uses a system call, is disabled by the macro PYLOCAL)

void pycall(const std::string &fn, gentype &res, const gentype &x)
{
    std::string xstr;

    // Store arguments for python function and create reconstruction string

    Vector<int> indused;

    // Convert to string and store in transfer indices.

    py::object xx = convToPy(x);
    int i = pyosetsrc(-1,xx);
    xstr =  "pyheavy.internal.pyogetsrc(";
    xstr += std::to_string(i);
    xstr += ")";

    // Construct run command

    std::string evalfn;

    evalfn += fn;
    evalfn += "(";
    evalfn += xstr;
    evalfn += ")";

    // Evaluated run command

    static py::object builtins = py::module_::import("builtins");
    static py::object eval_function = builtins.attr("eval");

    py::object resultobj = eval_function(evalfn);

    // Retrieve results of operation

    convFromPy(res,resultobj);

    // Clear used indices

    pyosrcreset(i);

    return;
}

