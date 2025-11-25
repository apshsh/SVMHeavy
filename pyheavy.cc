
//
// SVMHeavyv7 Python CLI-like Interface
//
// Date: 01/07/2025
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//#ifndef DNDEBUG
//#define DEBUGPY
//#endif

#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <functional>

namespace py = pybind11;

inline void qswap(py::object *&a, py::object *&b);
inline void qswap(py::object *&a, py::object *&b)
{
    py::object *x = a; a = b; b = x;
}

inline py::object *&setident (py::object *&a) { throw("something"); return a; }
inline py::object *&setzero  (py::object *&a) { a = nullptr;        return a; }
inline py::object *&setposate(py::object *&a) {                     return a; }
inline py::object *&setnegate(py::object *&a) { throw("something"); return a; }
inline py::object *&setconj  (py::object *&a) { throw("something"); return a; }
inline py::object *&setrand  (py::object *&a) { throw("something"); return a; }
inline py::object *&postProInnerProd(py::object *&a) { return a; }

#include "mlinter.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "vecfifo.hpp"
#include "opttest.hpp"
#include "addData.hpp"
#include "errortest.hpp"
#include "gridopt.hpp"
#include "directopt.hpp"
#include "nelderopt.hpp"
#include "bayesopt.hpp"
#include "makemonot.hpp"

void dostartup(void);

int glob_MLInd        (int i = 0, int seti = 0);
int glob_gridInd      (int i = 0, int seti = 0);
int glob_DIRectInd    (int i = 0, int seti = 0);
int glob_NelderMeadInd(int i = 0, int seti = 0);
int glob_BayesianInd  (int i = 0, int seti = 0);

int glob_MLInd        (int i, int seti) { static thread_local int iii = 1; if ( i ) { if ( seti && i ) { iii = i; } } return iii; }
int glob_gridInd      (int i, int seti) { static thread_local int iii = 1; if ( i ) { if ( seti && i ) { iii = i; } } return iii; }
int glob_DIRectInd    (int i, int seti) { static thread_local int iii = 1; if ( i ) { if ( seti && i ) { iii = i; } } return iii; }
int glob_NelderMeadInd(int i, int seti) { static thread_local int iii = 1; if ( i ) { if ( seti && i ) { iii = i; } } return iii; }
int glob_BayesianInd  (int i, int seti) { static thread_local int iii = 1; if ( i ) { if ( seti && i ) { iii = i; } } return iii; }

py::object &pygetbuiltin(void) { static py::object builtins   = py::module_::import("builtins");   return builtins;   }
py::object &pyisinstance(void) { static py::object isinstance = pygetbuiltin().attr("isinstance"); return isinstance; }
py::object &pycallable  (void) { static py::object callable   = pygetbuiltin().attr("callable");   return callable;   }
py::object &pyeval      (void) { static py::object eval       = pygetbuiltin().attr("eval");       return eval;       }
py::object &pyvalueerror(void) { static py::object ValueError = pygetbuiltin().attr("ValueError"); return ValueError; }
py::object &pycomplex   (void) { static py::object complex    = pygetbuiltin().attr("complex");    return complex;    }

bool isValNone    (const py::object &src) { return src.is_none();                                                            }
bool isValInteger (const py::object &src) { return !isValNone(src) && py::isinstance<py::int_>(src);                         }
bool isValReal    (const py::object &src) { return !isValNone(src) && py::isinstance<py::float_>(src);                       }
bool isValComplex (const py::object &src) { return !isValNone(src) && py::cast<py::bool_>(pyisinstance()(&src,pycomplex())); }
bool isValList    (const py::object &src) { return !isValNone(src) && py::isinstance<py::list>(src);                         }
bool isValTuple   (const py::object &src) { return !isValNone(src) && py::isinstance<py::tuple>(src);                        }
bool isValDict    (const py::object &src) { return !isValNone(src) && py::isinstance<py::dict>(src);                         }
bool isValString  (const py::object &src) { return !isValNone(src) && py::isinstance<py::str>(src);                          }
bool isValCallable(const py::object &src) { return !isValNone(src) && py::cast<py::bool_>(pycallable()(src));                }

bool isValCastableToInteger(const py::object &src) { return isValNone(src) || isValInteger(src) || isValReal(src);                      }
bool isValCastableToReal   (const py::object &src) { return isValNone(src) || isValInteger(src) || isValReal(src);                      }
bool isValCastableToComplex(const py::object &src) { return isValNone(src) || isValInteger(src) || isValReal(src) || isValComplex(src); }

int                  toInt(const py::object &src) { return (int)      py::cast<py::int_>  (src); }
double               toDbl(const py::object &src) { return (double)   py::cast<py::float_>(src); }
std::complex<double> toCpl(const py::object &src) { return py::cast<std::complex<double> >(src); }

                   py::object convToPy(      int                   src);
                   py::object convToPy(      double                src);
                   py::object convToPy(      std::complex<double>  src);
                   py::object convToPy(const d_anion              &src);
                   py::object convToPy(const std::string          &src);
template <class T> py::object convToPy(const Vector<T>            &src);
template <>        py::object convToPy(const Vector<double>       &src);
template <class T> py::object convToPy(const Matrix<T>            &src);
template <class T> py::object convToPy(const Set<T>               &src);
template <class T> py::object convToPy(const Dict<T,dictkey>      &src);
template <class T> py::object convToPy(const SparseVector<T>      &src);
                   py::object convToPy(const gentype              &src);
template <class T> py::object convToPy(int size, const T          *src);

py::object makeError(const char *src) { return pyvalueerror()(src); }

// These return 1 if conversion fails

template <class T> int convFromPy(T                     &res, const py::handle &src);
                   int convFromPy(int                   &res, const py::object &src);
                   int convFromPy(double                &res, const py::object &src);
                   int convFromPy(std::complex<double>  &res, const py::object &src);
                   int convFromPy(d_anion               &res, const py::object &src);
                   int convFromPy(std::string           &res, const py::object &src);
template <class T> int convFromPy(Vector<T>             &res, const py::object &src);
template <class T> int convFromPy(Matrix<T>             &res, const py::object &src);
template <class T> int convFromPy(Set<T>                &res, const py::object &src);
template <class T> int convFromPy(Dict<T,dictkey>       &res, const py::object &src);
template <class T> int convFromPy(SparseVector<T>       &res, const py::object &src);
template <>        int convFromPy(SparseVector<gentype> &res, const py::object &src);
                   int convFromPy(gentype               &res, const py::object &src);

gentype convFromPy(const py::object &src);

// Helper macros for python module constructions

#define        QDO(modis,dofn,desc)        modis.def(#dofn,  &(mod_ ## dofn),  "Do "               desc);
#define     QDOARG(modis,dofn,desc,pname)  modis.def(#dofn,  &(mod_ ## dofn),  "Do "               desc, py::arg(pname));
#define       QGET(modis,getfn,desc)       modis.def(#getfn, &(mod_ ## getfn), "Get "              desc);
#define    QGETCLA(modis,getfn,desc)       modis.def(#getfn, &(mod_ ## getfn), "For class d, get " desc, py::arg("d"));
#define    QGETSET(modis,getfn,setfn,desc) modis.def(#setfn, &(mod_ ## setfn), "Set "              desc, py::arg(#getfn)); \
                                           modis.def(#getfn, &(mod_ ## getfn), "Get "              desc);
#define QGETSETCLA(modis,getfn,setfn,desc) modis.def(#setfn, &(mod_ ## setfn), "For class d, set " desc, py::arg("d"), py::arg(#getfn)); \
                                           modis.def(#getfn, &(mod_ ## getfn), "For class d, get " desc, py::arg("d"));

#define       QGETD(modis,getfn,getname,desc)               modis.def(getname, &(mod_ ## getfn), "Get "              desc);
#define    QGETSETD(modis,getfn,setfn,getname,setname,desc) modis.def(setname, &(mod_ ## setfn), "Do "               desc, py::arg(getname)); \
                                                            modis.def(getname, &(mod_ ## getfn), "Get "              desc);
#define QGETSETCLAD(modis,getfn,setfn,getname,setname,desc) modis.def(setname, &(mod_ ## setfn), "For class d, set " desc, py::arg("d"), py::arg(getname)); \
                                                            modis.def(getname, &(mod_ ## getfn), "For class d, get " desc, py::arg("d"));

#define QGETSETOPT(modis,varname,ty,sub,desc) modis ## _ ## ty ## _ ## sub.def("set" #varname, &(modoptset_ ## ty ## _ ## varname), "For " #ty          " optimiser, set " desc, py::arg(#varname)); \
                                              modis ## _ ## ty ## _ ## sub.def(      #varname, &(modoptget_ ## ty ## _ ## varname), "For " #ty          " optimiser, get " desc);
#define QGETSETOPTB(modis,varname,ty,desc   ) modis ## _ ##             ty.def("set" #varname, &(modoptset_ ## ty ## _ ## varname), "For " #ty          " optimiser, set " desc, py::arg(#varname)); \
                                              modis ## _ ##             ty.def(      #varname, &(modoptget_ ## ty ## _ ## varname), "For " #ty          " optimiser, get " desc);
#define QGETSETOPTALL(modis,varname,desc)     modis ## _grid              .def("set" #varname, &(modoptset_grid_       ## varname), "For " "grid"       " optimiser, set " desc, py::arg(#varname)); \
                                              modis ## _DIRect            .def("set" #varname, &(modoptset_DIRect_     ## varname), "For " "DIRect"     " optimiser, set " desc, py::arg(#varname)); \
                                              modis ## _NelderMead        .def("set" #varname, &(modoptset_NelderMead_ ## varname), "For " "NelderMead" " optimiser, set " desc, py::arg(#varname)); \
                                              modis ## _Bayesian          .def("set" #varname, &(modoptset_Bayesian_   ## varname), "For " "Bayesian"   " optimiser, set " desc, py::arg(#varname)); \
                                              modis ## _grid              .def(      #varname, &(modoptget_grid_       ## varname), "For " "grid"       " optimiser, get " desc); \
                                              modis ## _DIRect            .def(      #varname, &(modoptget_DIRect_     ## varname), "For " "DIRect"     " optimiser, get " desc); \
                                              modis ## _NelderMead        .def(      #varname, &(modoptget_NelderMead_ ## varname), "For " "NelderMead" " optimiser, get " desc); \
                                              modis ## _Bayesian          .def(      #varname, &(modoptget_Bayesian_   ## varname), "For " "Bayesian"   " optimiser, get " desc);

#define QGETSETKERD(modis,getfn,setfn,getname,setname,desc)  modis.def(setname, &(mod_k ## setfn), "Set kernel param "                  desc, py::arg(getname)); \
                                                             modis.def(getname, &(mod_k ## getfn), "Get kernel param "                  desc);                   \
                                                   (modis ##  _UU).def(setname, &(mod_e ## setfn), "Set output kernel param "           desc, py::arg(getname)); \
                                                   (modis ##  _UU).def(getname, &(mod_e ## getfn), "Get output kernel param "           desc);                   \
                                                   (modis ## _RFF).def(setname, &(mod_r ## setfn), "Set RFF kernel param "              desc, py::arg(getname)); \
                                                   (modis ## _RFF).def(getname, &(mod_r ## getfn), "Get RFF kernel param "              desc);
#define QGETSETKERQD(modis,getfn,setfn,getname,setname,desc) modis.def(setname, &(mod_k ## setfn), "Set kernel element q param "        desc, py::arg(getname), py::arg("q") = 0); \
                                                             modis.def(getname, &(mod_k ## getfn), "Get kernel element q param "        desc,                   py::arg("q") = 0); \
                                                   (modis ##  _UU).def(setname, &(mod_e ## setfn), "Set output kernel element q param " desc, py::arg(getname), py::arg("q") = 0); \
                                                   (modis ##  _UU).def(getname, &(mod_e ## getfn), "Set output kernel element q param " desc,                   py::arg("q") = 0); \
                                                   (modis ## _RFF).def(setname, &(mod_r ## setfn), "Set RFF kernel element q param "    desc, py::arg(getname), py::arg("q") = 0); \
                                                   (modis ## _RFF).def(getname, &(mod_r ## getfn), "Set RFF kernel element q param "    desc,                   py::arg("q") = 0);

#define QIMPA(modis,func,desc) modis.def( #func, &( gencalc_ ## func ), "Evaluate " desc " in gentype", py::arg("x"));
#define QIMPB(modis,func,desc) modis.def( #func, &( gencalc_ ## func ), "Evaluate " desc " in gentype", py::arg("x"),py::arg("y"));
#define QIMPC(modis,func,desc) modis.def( #func, &( gencalc_ ## func ), "Evaluate " desc " in gentype", py::arg("x"),py::arg("y"),py::arg("z"));



// Corresponding helper macros to auto-generate function definitions to be used by python module

#define        DODEF(dofn)          py::object mod_ ##  dofn(void)                { dostartup(); int i = glob_MLInd(0);                             return convToPy(getMLref(i). dofn ());        }
#define     DOARGDEF(dofn,T)        int        mod_ ##  dofn(       py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); return getMLref(i). dofn (altp);              }
#define       GETDEF(getfn)         py::object mod_ ## getfn(void)                { dostartup(); int i = glob_MLInd(0);                             return convToPy(getMLrefconst(i). getfn ());  }
#define    GETCLADEF(getfn)         py::object mod_ ## getfn(int d)               { dostartup(); int i = glob_MLInd(0);                             return convToPy(getMLrefconst(i). getfn (d)); }
#define    SETCLADEF(setfn,T)       int        mod_ ## setfn(int d, py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); return getMLref(i). setfn (d,altp);           }
#define    GETSETDEF(getfn,setfn,T) py::object mod_ ## getfn(void)                { dostartup(); int i = glob_MLInd(0);                             return convToPy(getMLrefconst(i). getfn ());  } \
                                    int        mod_ ## setfn(       py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); return getMLref(i). setfn (altp);             }
#define GETSETCLADEF(getfn,setfn,T) py::object mod_ ## getfn(int d)               { dostartup(); int i = glob_MLInd(0);                             return convToPy(getMLrefconst(i). getfn (d)); } \
                                    int        mod_ ## setfn(int d, py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); return getMLref(i). setfn (d,altp);           }

#define OPTGETSETDEF(varname,ty,T) py::object modoptget_ ## ty ## _ ## varname(void)         { dostartup(); int i = glob_ ## ty ## Ind(0); return convToPy((T) get ## ty ## refconst(i).varname); } \
                                   void       modoptset_ ## ty ## _ ## varname(py::object p) { dostartup(); int i = glob_ ## ty ## Ind(0); T altp; convFromPy(altp,p); get ## ty ## ref(i).varname = altp; }
#define OPTGETSETALLDEF(varname,T) py::object modoptget_grid_       ## varname(void)         { dostartup(); int i = glob_gridInd      (0); return convToPy(getgridrefconst      (i).varname); } \
                                   py::object modoptget_DIRect_     ## varname(void)         { dostartup(); int i = glob_DIRectInd    (0); return convToPy(getDIRectrefconst    (i).varname); } \
                                   py::object modoptget_NelderMead_ ## varname(void)         { dostartup(); int i = glob_NelderMeadInd(0); return convToPy(getNelderMeadrefconst(i).varname); } \
                                   py::object modoptget_Bayesian_   ## varname(void)         { dostartup(); int i = glob_BayesianInd  (0); return convToPy(getBayesianrefconst  (i).varname); } \
                                   void       modoptset_grid_       ## varname(py::object p) { dostartup(); int i = glob_gridInd      (0); T altp; convFromPy(altp,p); getgridref      (i).varname = altp; } \
                                   void       modoptset_DIRect_     ## varname(py::object p) { dostartup(); int i = glob_DIRectInd    (0); T altp; convFromPy(altp,p); getDIRectref    (i).varname = altp; } \
                                   void       modoptset_NelderMead_ ## varname(py::object p) { dostartup(); int i = glob_NelderMeadInd(0); T altp; convFromPy(altp,p); getNelderMeadref(i).varname = altp; } \
                                   void       modoptset_Bayesian_   ## varname(py::object p) { dostartup(); int i = glob_BayesianInd  (0); T altp; convFromPy(altp,p); getBayesianref  (i).varname = altp; }

#define GETSETKERAPDEF(getfn,setfn,T) py::object mod_k ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn ());  } \
                                      void       mod_k ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getKernel_unsafe(). setfn (altp); getMLref(i).resetKernel(0,-1,0); } \
                                      py::object mod_e ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn ());  } \
                                      void       mod_e ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp); getMLref(i).resetUUOutputKernel(0); } \
                                      py::object mod_r ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn ());  } \
                                      void       mod_r ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getRFFKernel_unsafe(). setfn (altp); getMLref(i).resetRFFKernel(0); }
#define GETSETKERBPDEF(getfn,setfn,T) py::object mod_k ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn ());  } \
                                      void       mod_k ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getKernel_unsafe(). setfn (altp); getMLref(i).resetKernel(0); } \
                                      py::object mod_e ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn ());  } \
                                      void       mod_e ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp); getMLref(i).resetUUOutputKernel(0); } \
                                      py::object mod_r ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn ());  } \
                                      void       mod_r ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getRFFKernel_unsafe(). setfn (altp); getMLref(i).resetRFFKernel(0); }
#define GETSETKERANDEF(getfn,setfn,T) py::object mod_k ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn ());  } \
                                      void       mod_k ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getKernel_unsafe(). setfn (altp); getMLref(i).resetKernel(0,-1,0); } \
                                      py::object mod_e ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn ());  } \
                                      void       mod_e ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp); getMLref(i).resetUUOutputKernel(0); } \
                                      py::object mod_r ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn ());  } \
                                      void       mod_r ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getRFFKernel_unsafe(). setfn (altp); getMLref(i).resetRFFKernel(0); }
#define GETSETKERBNDEF(getfn,setfn,T) py::object mod_k ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn ());  } \
                                      void       mod_k ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getKernel_unsafe(). setfn (altp); getMLref(i).resetKernel(0); } \
                                      py::object mod_e ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn ());  } \
                                      void       mod_e ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp); getMLref(i).resetUUOutputKernel(0); } \
                                      py::object mod_r ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn ());  } \
                                      void       mod_r ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getRFFKernel_unsafe(). setfn (altp); getMLref(i).resetRFFKernel(0); }
#define GETSETKERCNDEF(getfn,setfn,T) py::object mod_k ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn ());  } \
                                      void       mod_k ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getKernel_unsafe(). setfn (altp); } \
                                      py::object mod_e ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn ());  } \
                                      void       mod_e ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp); } \
                                      py::object mod_r ## getfn(void)         { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn ());  } \
                                      void       mod_r ## setfn(py::object p) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getRFFKernel_unsafe(). setfn (altp); }

#define GETSETKERAPQDEF(getfn,setfn,T) py::object mod_k ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn (q));  } \
                                       void       mod_k ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getKernel_unsafe(). setfn (altp,q); getMLref(i).resetKernel(0,-1,0); } \
                                       py::object mod_e ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn (q));  } \
                                       void       mod_e ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp,q); getMLref(i).resetUUOutputKernel(0); } \
                                       py::object mod_r ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn (q));  } \
                                       void       mod_r ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getRFFKernel_unsafe(). setfn (altp,q); getMLref(i).resetRFFKernel(0); }
#define GETSETKERBPQDEF(getfn,setfn,T) py::object mod_k ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn (q));  } \
                                       void       mod_k ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getKernel_unsafe(). setfn (altp,q); getMLref(i).resetKernel(0); } \
                                       py::object mod_e ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn (q));  } \
                                       void       mod_e ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp,q); getMLref(i).resetUUOutputKernel(0); } \
                                       py::object mod_r ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn (q));  } \
                                       void       mod_r ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).prepareKernel(); getMLref(i).getRFFKernel_unsafe(). setfn (altp,q); getMLref(i).resetRFFKernel(0); }
#define GETSETKERANQDEF(getfn,setfn,T) py::object mod_k ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn (q));  } \
                                       void       mod_k ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getKernel_unsafe(). setfn (altp,q); getMLref(i).resetKernel(0,-1,0); } \
                                       py::object mod_e ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn (q));  } \
                                       void       mod_e ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp,q); getMLref(i).resetUUOutputKernel(0); } \
                                       py::object mod_r ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn (q));  } \
                                       void       mod_r ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getRFFKernel_unsafe(). setfn (altp,q); getMLref(i).resetRFFKernel(0); }
#define GETSETKERBNQDEF(getfn,setfn,T) py::object mod_k ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn (q));  } \
                                       void       mod_k ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getKernel_unsafe(). setfn (altp,q); getMLref(i).resetKernel(0); } \
                                       py::object mod_e ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn (q));  } \
                                       void       mod_e ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp,q); getMLref(i).resetUUOutputKernel(0); } \
                                       py::object mod_r ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn (q));  } \
                                       void       mod_r ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getRFFKernel_unsafe(). setfn (altp,q); getMLref(i).resetRFFKernel(0); }
#define GETSETKERCNQDEF(getfn,setfn,T) py::object mod_k ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getKernel(). getfn (q));  } \
                                       void       mod_k ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getKernel_unsafe(). setfn (altp,q); } \
                                       py::object mod_e ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getUUOutputKernel(). getfn (q));  } \
                                       void       mod_e ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getUUOutputKernel_unsafe(). setfn (altp,q); } \
                                       py::object mod_r ## getfn(int q)               { dostartup(); int i = glob_MLInd(0); return convToPy(getMLrefconst(i).getRFFKernel(). getfn (q));  } \
                                       void       mod_r ## setfn(py::object p, int q) { dostartup(); int i = glob_MLInd(0); T altp; convFromPy(altp,p); getMLref(i).getRFFKernel_unsafe(). setfn (altp,q); }

#define MAKEVISA(func)                                                                      \
py::object gencalc_ ## func (py::object x)                                                  \
{                                                                                           \
    gentype xx;                                                                             \
    if ( convFromPy(xx,x) ) { return makeError("Couldn't convert argument 1 for " #func); } \
    return convToPy( func (xx));                                                            \
}

#define MAKEVISB(func)                                                                      \
py::object gencalc_ ## func (py::object x, py::object y)                                    \
{                                                                                           \
    gentype xx,yy;                                                                          \
    if ( convFromPy(xx,x) ) { return makeError("Couldn't convert argument 1 for " #func); } \
    if ( convFromPy(yy,y) ) { return makeError("Couldn't convert argument 2 for " #func); } \
    return convToPy( func (xx,yy));                                                         \
}

#define MAKEVISC(func)                                                                      \
py::object gencalc_ ## func (py::object x, py::object y, py::object z)                      \
{                                                                                           \
    gentype xx,yy,zz;                                                                       \
    if ( convFromPy(xx,x) ) { return makeError("Couldn't convert argument 1 for " #func); } \
    if ( convFromPy(yy,y) ) { return makeError("Couldn't convert argument 2 for " #func); } \
    if ( convFromPy(zz,z) ) { return makeError("Couldn't convert argument 3 for " #func); } \
    return convToPy( func (xx,yy,zz));                                                      \
}





// Actual functions (including auto-generation stubs) used in python module

py::object mloptgrid(      int i, int dim, int numreps, py::object objfn, py::object callback);
py::object mloptDIRect(    int i, int dim, int numreps, py::object objfn, py::object callback);
py::object mloptNelderMead(int i, int dim, int numreps, py::object objfn, py::object callback);
py::object mloptBayesian(  int i, int dim, int numreps, py::object objfn, py::object callback);

int selml(int i = 0);

int selmlmuapprox   (int i = 0, int k = 0);
int selmlcgtapprox  (int i = 0, int k = 0);
//int selmlaugxapprox (int i = 0, int k = 0);
int selmlsigmaapprox(int i = 0);
int selmldiffmodel  (int i = 0);
int selmlsrcmodel   (int i = 0);

int selmlmuapprox_prior   (int i = 0);
int selmlcgtapprox_prior  (int i = 0);
//int selmlaugxapprox_prior (int i = 0);
int selmlsigmaapprox_prior(int i = 0);
int selmldiffmodel_prior  (int i = 0);
int selmlsrcmodel_prior   (int i = 0);

void swapml  (int i = 0, int j = 0);
void copyml  (int i = 0, int j = 0);
void assignml(int i = 0, int j = 0);

int selgridopt      (int i, int rst);
int selDIRectopt    (int i, int rst);
int selNelderMeadopt(int i, int rst);
int selBayesianopt  (int i, int rst);

void svmheavya(void);                      // just get help screen
void svmheavyb(int permode);               // set persistence mode
void svmheavyc(const std::string commstr); // execute with string

#define snakehigh 24
#define snakewide 80

// Set, get and clear storage

py::object &pyogetsrc(int k);
int         pyosetsrc(int k, const py::object &src);
void        pyoclrsrc(void);

void logit        (py::object logstr); // print to errstream
void callintercalc(void);
void callsnakes   (int wide, int high);

py::object svmeval(py::object fn, py::object arg);

py::object svmtest(int i, std::string type);

int setpriml(int j = 0);

int makeMonot(int n, int t, py::object xb, py::object xlb, py::object xub, int d, py::object y, double Cweight, double epsweight, int j);

int  addTrainingVectorml(py::object x, py::object z, int j);
int faddTrainingVectorml(const std::string &fname, int ignoreStart, int imax, int reverse, int j);

int removeTrainingVectorml(int j, int num);

double mlcalcLOO   (void);
double mlcalcRecall(void);
double mlcalcCross (int m, int rndit = 0, int numreps = 1);

py::object muml     (py::object xa);
py::object mugml    (py::object xa, int fmt);
py::object varml    (py::object xa);
py::object covml    (py::object xa, py::object xb);
py::object predvarml(py::object xa, py::object pp, py::object sigw);
py::object predcovml(py::object xa, py::object xb, py::object pp, py::object sigw);

py::object mlalpha(void);
py::object mlbias (void);
py::object mlGp   (void);

int mlsetalpha(py::object src);
int mlsetbias (py::object src);

double mltuneKernel(int method, double xwidth = 1, int tuneK = 1, int tuneP = 0);

py::object mlK0(void);
py::object mlK1(py::object xa);
py::object mlK2(py::object xa, py::object xb);
py::object mlK3(py::object xa, py::object xb, py::object xc);
py::object mlK4(py::object xa, py::object xb, py::object xc, py::object xd);

void boSetgridsource      (int j);
void boSetkernapproxsource(int j);
void boSetimpmeas         (int j);

GETSETDEF(getMLType,ssetMLTypeClean,std::string)

GETSETDEF(getVmethod,setVmethod,std::string)
GETSETDEF(getCmethod,setCmethod,std::string)
GETSETDEF(getOmethod,setOmethod,std::string)
GETSETDEF(getAmethod,setAmethod,std::string)
GETSETDEF(getRmethod,setRmethod,std::string)
GETSETDEF(getTmethod,setTmethod,std::string)
GETSETDEF(getEmethod,setEmethod,std::string)
GETSETDEF(getBmethod,setBmethod,std::string)
GETSETDEF(getMmethod,setMmethod,std::string)

GETSETDEF(prim,  setprim,  int    )
GETSETDEF(prival,setprival,gentype)

DODEF(train            )
DODEF(reset            )
DODEF(restart          )
DODEF(removeNonSupports)

DOARGDEF(scale,          double)
DOARGDEF(randomise,      double)
DOARGDEF(trimTrainingSet,int   )

GETSETDEF(C,        setC,        double)
GETSETDEF(sigma,    setsigma,    double)
GETSETDEF(sigma_cut,setsigma_cut,double)
GETSETDEF(eps,      seteps,      double)

GETSETDEF(LinBiasForce, setLinBiasForce, double)
GETSETDEF(QuadBiasForce,setQuadBiasForce,double)
GETSETDEF(nu,           setnu,           double)
GETSETDEF(nuQuad,       setnuQuad,       double)

GETSETDEF(k,  setk,  int)
GETSETDEF(ktp,setktp,int)

GETSETCLADEF(Cclass,  setCclass,  double)
GETSETCLADEF(epsclass,setepsclass,double)

GETSETCLADEF(LinBiasForceclass, setLinBiasForceclass, double)
GETSETCLADEF(QuadBiasForceclass,setQuadBiasForceclass,double)

GETSETDEF(tspaceDim,settspaceDim,int)
GETSETDEF(order,    setorder,    int)

GETSETDEF(m,      setm,      int   )
GETSETDEF(theta,  settheta,  double)
GETSETDEF(simnorm,setsimnorm,int   )

GETDEF(loglikelihood)
GETDEF(maxinfogain  )
GETDEF(RKHSnorm     )
GETDEF(RKHSabs      )

GETDEF(N                 )
GETDEF(type              )
GETDEF(subtype           )
GETDEF(xspaceDim         )
GETDEF(fspaceDim         )
GETDEF(tspaceSparse      )
GETDEF(xspaceSparse      )
GETDEF(numClasses        )
GETDEF(ClassLabels       )
GETDEF(isTrained         )
GETDEF(isSolGlob         )
GETDEF(isUnderlyingScalar)
GETDEF(isUnderlyingVector)
GETDEF(isUnderlyingAnions)
GETDEF(isClassifier      )
GETDEF(isRegression      )
GETDEF(isPlanarType      )

GETDEF(NZ )
GETDEF(NF )
GETDEF(NS )
GETDEF(NC )
GETDEF(NLB)
GETDEF(NLF)
GETDEF(NUF)
GETDEF(NUB)

GETCLADEF(NNC)

GETDEF(x          )
GETDEF(d          )
GETDEF(y          )
GETDEF(yp         )
GETDEF(Cweight    )
GETDEF(Cweightfuzz)
GETDEF(sigmaweight)
GETDEF(epsweight  )
GETDEF(alphaState )
GETDEF(xtang      )

GETDEF(kerndiag)

GETSETKERAPDEF(isSymmSet, setSymmSet, int)
GETSETKERAPDEF(isFullNorm,setFullNorm,int)
GETSETKERAPDEF(isProd,    setProd,    int)
GETSETKERBPDEF(isAltDiff, setAltDiff, int)
GETSETKERAPDEF(rankType,  setrankType,int)

GETSETKERANDEF(denseZeroPoint,setdenseZeroPoint,double)

GETSETKERAPDEF(getlinGradOrd, setlinGradOrd, Vector<int>           )
GETSETKERAPDEF(getlinGradScal,setlinGradScal,Vector<Matrix<double>>)

GETSETKERAPDEF(getlinParity,    setlinParity,    Vector<int>    )
GETSETKERAPDEF(getlinParityOrig,setlinParityOrig,Vector<gentype>)

GETSETKERAPDEF(numSamples,        setnumSamples,        int            )
GETSETKERAPDEF(sampleDistribution,setSampleDistribution,Vector<gentype>)
GETSETKERAPDEF(sampleIndices,     setSampleIndices,     Vector<int>    )

GETSETKERAPDEF(getTypes,         setTypes,         Vector<int>              )
GETSETKERAPDEF(getHyper,         setHyper,         Vector<Vector<gentype>>  )
GETSETKERAPDEF(getIntConstants,  setIntConstantss, Vector<Vector<int>>      )
GETSETKERBNDEF(getRealOverwrites,setRealOverwrites,Vector<SparseVector<int>>)
GETSETKERBNDEF(getIntOverwrites, setIntOverwrites, Vector<SparseVector<int>>)
GETSETKERAPDEF(getIsNormalised,  setIsNormalised,  Vector<int>              )
GETSETKERAPDEF(getIsMagTerm,     setIsMagTerm,     Vector<int>              )
GETSETKERAPDEF(getIsNomConst,    setIsNomConst,    Vector<int>              )

GETSETKERAPDEF(getChained, setChained, Vector<int>)
GETSETKERAPDEF(getSplit,   setSplit,   Vector<int>)
GETSETKERAPDEF(getMulSplit,setMulSplit,Vector<int>)

GETSETKERAPDEF(getHyperLB,setHyperLB,Vector<Vector<gentype>>)
GETSETKERAPDEF(getHyperUB,setHyperUB,Vector<Vector<gentype>>)

GETSETKERAPDEF(getIntConstantsLB,setIntConstantssLB,Vector<Vector<int>>)
GETSETKERAPDEF(getIntConstantsUB,setIntConstantssUB,Vector<Vector<int>>)

GETSETKERAPQDEF(cWeight,     setWeight,      gentype)
GETSETKERAPQDEF(cType,       setType,        int    )
GETSETKERAPQDEF(isNormalised,setisNormalised,int    )
GETSETKERAPQDEF(isMagTerm,   setisMagTerm,   int    )
GETSETKERAPQDEF(isNomConst,  setisNomConst,  int    )

GETSETKERAPQDEF(cRealConstants,setRealConstants,Vector<gentype>  )
GETSETKERAPQDEF(cIntConstants, setIntConstants, Vector<int>      )
GETSETKERBNQDEF(cRealOverwrite,setRealOverwrite,SparseVector<int>)
GETSETKERBNQDEF(cIntOverwrite, setIntOverwrite, SparseVector<int>)

GETSETKERAPQDEF(getRealConstZero,setRealConstZero,double)
GETSETKERAPQDEF(getIntConstZero, setIntConstZero, int   )

GETSETKERAPQDEF(isChained, setisChained, int)
GETSETKERAPQDEF(isSplit,   setisSplit,   int)
GETSETKERAPQDEF(isMulSplit,setisMulSplit,int)

GETSETKERCNQDEF(cWeightLB,setWeightLB,gentype)

GETSETKERCNQDEF(cRealConstantsLB,setRealConstantsLB,Vector<gentype>)
GETSETKERCNQDEF(cIntConstantsLB, setIntConstantsLB, Vector<int>    )

GETSETKERCNQDEF(getRealConstZeroLB,setRealConstZeroLB,double)
GETSETKERCNQDEF(getIntConstZeroLB, setIntConstZeroLB, int   )

GETSETKERCNQDEF(cWeightUB,setWeightUB,gentype)

GETSETKERCNQDEF(cRealConstantsUB,setRealConstantsUB,Vector<gentype>)
GETSETKERCNQDEF(cIntConstantsUB, setIntConstantsUB, Vector<int>    )

GETSETKERCNQDEF(getRealConstZeroUB,setRealConstZeroUB,double)
GETSETKERCNQDEF(getIntConstZeroUB, setIntConstZeroUB, int   )

// Optimisation options

OPTGETSETALLDEF(optname,std::string)

OPTGETSETALLDEF(maxtraintime,double)

OPTGETSETALLDEF(softmin,double)
OPTGETSETALLDEF(softmax,double)
OPTGETSETALLDEF(hardmin,double)
OPTGETSETALLDEF(hardmax,double)

OPTGETSETDEF(numZooms,grid,int   )
OPTGETSETDEF(zoomFact,grid,double)

OPTGETSETDEF(numPts,grid,Vector<int>)

OPTGETSETDEF(maxits,   DIRect,int   )
OPTGETSETDEF(maxevals, DIRect,int   )
OPTGETSETDEF(eps,      DIRect,double)

OPTGETSETDEF(minf_max,NelderMead,double)
OPTGETSETDEF(ftol_rel,NelderMead,double)
OPTGETSETDEF(ftol_abs,NelderMead,double)
OPTGETSETDEF(xtol_rel,NelderMead,double)
OPTGETSETDEF(xtol_abs,NelderMead,double)
OPTGETSETDEF(maxeval, NelderMead,int   )
OPTGETSETDEF(method,  NelderMead,int   )

OPTGETSETDEF(acq,        Bayesian,int    )
OPTGETSETDEF(startpoints,Bayesian,int    )
OPTGETSETDEF(totiters,   Bayesian,int    )
OPTGETSETDEF(startseed,  Bayesian,int    )
OPTGETSETDEF(algseed,    Bayesian,int    )
OPTGETSETDEF(itcntmethod,Bayesian,int    )
OPTGETSETDEF(err,        Bayesian,double )
OPTGETSETDEF(minstdev,   Bayesian,double )
OPTGETSETDEF(moodim,     Bayesian,int    )
OPTGETSETDEF(sigmuseparate,Bayesian,int    )
OPTGETSETDEF(numcgt,     Bayesian,int    )
OPTGETSETDEF(cgtmethod,  Bayesian,int    )
OPTGETSETDEF(cgtmargin,  Bayesian,double )
OPTGETSETDEF(ztol,       Bayesian,double )
OPTGETSETDEF(delta,      Bayesian,double )
OPTGETSETDEF(zeta,       Bayesian,double )
OPTGETSETDEF(nu,         Bayesian,double )
OPTGETSETDEF(modD,       Bayesian,double )
OPTGETSETDEF(a,          Bayesian,double )
OPTGETSETDEF(b,          Bayesian,double )
OPTGETSETDEF(r,          Bayesian,double )
OPTGETSETDEF(p,          Bayesian,double )
OPTGETSETDEF(R,          Bayesian,double )
OPTGETSETDEF(B,          Bayesian,double )
OPTGETSETDEF(betafn,     Bayesian,gentype)
OPTGETSETDEF(numfids,    Bayesian,int    )
OPTGETSETDEF(fidover,    Bayesian,int    )
OPTGETSETDEF(fidmode,    Bayesian,int    )
OPTGETSETDEF(dimfid,     Bayesian,int    )
OPTGETSETDEF(fidbudget,  Bayesian,double )
OPTGETSETDEF(fidpenalty, Bayesian,gentype)
OPTGETSETDEF(fidvar,     Bayesian,gentype)
OPTGETSETDEF(PIscale,    Bayesian,int    )
OPTGETSETDEF(TSmode,     Bayesian,int    )
OPTGETSETDEF(TSNsamp,    Bayesian,int    )
OPTGETSETDEF(TSsampType, Bayesian,int    )
OPTGETSETDEF(TSxsampType,Bayesian,int    )
OPTGETSETDEF(sigma_cut,  Bayesian,double )
OPTGETSETDEF(tranmeth,   Bayesian,int    )
OPTGETSETDEF(alpha0,     Bayesian,double )
OPTGETSETDEF(beta0,      Bayesian,double )
OPTGETSETDEF(kxfnum,     Bayesian,int    )
OPTGETSETDEF(kxfnorm,    Bayesian,int    )

OPTGETSETDEF(intrinbatch,        Bayesian,int)
OPTGETSETDEF(intrinbatchmethod,  Bayesian,int)
OPTGETSETDEF(startpointsmultiobj,Bayesian,int)
OPTGETSETDEF(totitersmultiobj,   Bayesian,int)
OPTGETSETDEF(ehimethodmultiobj,  Bayesian,int)

OPTGETSETDEF(tunemu,          Bayesian,int        )
OPTGETSETDEF(tunecgt,         Bayesian,int        )
OPTGETSETDEF(tunesigma,       Bayesian,int        )
OPTGETSETDEF(tunesrcmod,      Bayesian,int        )
OPTGETSETDEF(tunediffmod,     Bayesian,int        )
//OPTGETSETDEF(tuneaugxmod,     Bayesian,int        )
OPTGETSETDEF(modelname,       Bayesian,std::string)
OPTGETSETDEF(modeloutformat,  Bayesian,int        )
OPTGETSETDEF(plotfreq,        Bayesian,int        )
OPTGETSETDEF(modelbaseline,   Bayesian,gentype    )

// Functions we might as well import from gentype

MAKEVISA(sqrt     )
MAKEVISA(Sqrt     )
MAKEVISA(cbrt     )
MAKEVISA(Cbrt     )
MAKEVISB(nthrt    )
MAKEVISB(Nthrt    )
MAKEVISA(exp      )
MAKEVISA(tenup    )
MAKEVISA(log      )
MAKEVISA(Log      )
MAKEVISA(log10    )
MAKEVISA(Log10    )
MAKEVISB(logb     )
MAKEVISB(Logb     )
MAKEVISB(logbl    )
MAKEVISB(Logbl    )
MAKEVISB(logbr    )
MAKEVISB(Logbr    )
MAKEVISA(sin      )
MAKEVISA(cos      )
MAKEVISA(tan      )
MAKEVISA(cosec    )
MAKEVISA(sec      )
MAKEVISA(cot      )
MAKEVISA(asin     )
MAKEVISA(Asin     )
MAKEVISA(acos     )
MAKEVISA(Acos     )
MAKEVISA(atan     )
MAKEVISA(acosec   )
MAKEVISA(Acosec   )
MAKEVISA(asec     )
MAKEVISA(Asec     )
MAKEVISA(acot     )
MAKEVISA(sinc     )
MAKEVISA(cosc     )
MAKEVISA(tanc     )
MAKEVISA(vers     )
MAKEVISA(covers   )
MAKEVISA(hav      )
MAKEVISA(excosec  )
MAKEVISA(exsec    )
MAKEVISA(avers    )
MAKEVISA(Avers    )
MAKEVISA(acovers  )
MAKEVISA(Acovers  )
MAKEVISA(ahav     )
MAKEVISA(Ahav     )
MAKEVISA(aexcosec )
MAKEVISA(Aexcosec )
MAKEVISA(aexsec   )
MAKEVISA(Aexsec   )
MAKEVISA(castrg   )
MAKEVISA(casctrg  )
MAKEVISA(acastrg  )
MAKEVISA(acasctrg )
MAKEVISA(Acastrg  )
MAKEVISA(Acasctrg )
MAKEVISA(sinh     )
MAKEVISA(cosh     )
MAKEVISA(tanh     )
MAKEVISA(cosech   )
MAKEVISA(sech     )
MAKEVISA(coth     )
MAKEVISA(asinh    )
MAKEVISA(acosh    )
MAKEVISA(Acosh    )
MAKEVISA(atanh    )
MAKEVISA(Atanh    )
MAKEVISA(acosech  )
MAKEVISA(asech    )
MAKEVISA(Asech    )
MAKEVISA(acoth    )
MAKEVISA(Acoth    )
MAKEVISA(sinhc    )
MAKEVISA(coshc    )
MAKEVISA(tanhc    )
MAKEVISA(versh    )
MAKEVISA(coversh  )
MAKEVISA(havh     )
MAKEVISA(excosech )
MAKEVISA(exsech   )
MAKEVISA(aversh   )
MAKEVISA(Aversh   )
MAKEVISA(acovrsh  )
MAKEVISA(ahavh    )
MAKEVISA(Ahavh    )
MAKEVISA(aexcosech)
MAKEVISA(aexsech  )
MAKEVISA(Aexsech  )
MAKEVISA(cashyp   )
MAKEVISA(caschyp  )
MAKEVISA(acashyp  )
MAKEVISA(acaschyp )
MAKEVISA(Acashyp  )
MAKEVISA(Acaschyp )
MAKEVISA(sigm     )
MAKEVISA(gd       )
MAKEVISA(asigm    )
MAKEVISA(Asigm    )
MAKEVISA(agd      )
MAKEVISA(Agd      )
MAKEVISB(bern     )
MAKEVISA(normDistr)
MAKEVISB(polyDistr)
MAKEVISB(PolyDistr)
MAKEVISA(gamma    )
MAKEVISA(lngamma  )
MAKEVISA(psi      )
MAKEVISA(zeta     )
MAKEVISA(lambertW )
MAKEVISA(lambertWx)
MAKEVISA(erf      )
MAKEVISA(erfc     )
MAKEVISA(dawson   )



// We need this because there are static py::objects, and things get messy at shutdown
// (I don't fully understand what this does, but it seems to work)

void shutdown_module() {
    removeallaltpycall(); // remove any remaining std::function references back to python in gentype objects
    pyoclrsrc(); // clear local store
    py::gil_scoped_acquire gil;
}



// Python module definition

//PYBIND11_MODULE(_pyheavy, m) {
PYBIND11_MODULE(pyheavy, m) {
    m.doc() = "SVMHeavy Machine Learning Library.";

    m.def("shutdown",&shutdown_module,"Safely shutdown the pyheavy module");

    // ---------------------------
    // ---------------------------
    // ---------------------------
    // ---------------------------

    auto m_cli = m.def_submodule("cli","Commandline Emulation.");

    m_cli.def("help",&svmheavya,"Display basic help-screen for CLI emulation."                 );
    m_cli.def("mode",&svmheavyb,"Set persistence mode (default 1, persistent).",py::arg("mode"));
    m_cli.def("exec",&svmheavyc,"Run with string given (mimic CLI).",           py::arg("str") );

    // ---------------------------
    // ---------------------------
    // ---------------------------
    // ---------------------------

    auto m_int = m.def_submodule("internal","Internal use.");

    m_int.def("pyogetsrc",&pyogetsrc,    "Get python object k from heap.",py::arg("k"));
    m_int.def("pyosetsrc",&pyosetsrc,    "Set python object k = val on heap. If val is a callable then this can be used  \n"
                                         "used in a gentype expression via pycall(k,x).",py::arg("k"),py::arg("val"));
    m_int.def("eval",     &svmeval,      "Evaluate function fn (gentype as string, or python function) provided (eg      \n"
                                         "\"sin(x)\" for the gentype sin function) with argument x. x can be: None, int, \n"
                                         "float, complex, list (mapped to vector), tuple (mapped to set) or dictionary.  \n"
                                         "To explore available functions you use the inbuilt calculator (calc function). \n"
                                         "See also pyheavy.maths.fn...",py::arg("fn"),py::arg("x"));
    m_int.def("snakes",   &callsnakes,   "Snakes (test io, streams).",py::arg("w")=snakewide,py::arg("h")=snakehigh);
    m_int.def("calc",     &callintercalc,"Calculator (explore functions available in gentype expressions).");
    m_int.def("log",      &logit,        "Print to log.",py::arg("str"));

    // ---------------------------
    // ---------------------------
    // ---------------------------
    // ---------------------------

    auto m_maths = m.def_submodule("maths","Mathematics related.");
    auto m_maths_fn = m_maths.def_submodule("fn","Functions imported from c++.");

    m_maths.def("test",&svmtest,"Return normalised (inputs and outputs in range [0,1]) test function (lambda)   \n"
                                "for evaluating function i. Available test functions are:                       \n"
                                "                                                                               \n"
                                " 1: Rastrigin function        (n-dimensional).                                 \n"
                                " 2: Ackley's function         (n-dimensional).                                 \n"
                                " 3: Sphere function           (n-dimensional).                                 \n"
                                " 4: Rosenbrock function      (>1 dimensional).                                 \n"
                                " 5: Beale's function          (2-dimensional).                                 \n"
                                " 6: Goldstein–Price function  (2-dimensional).                                 \n"
                                " 7: Booth's function          (2-dimensional).                                 \n"
                                " 8: Bukin function N.6        (2-dimensional).                                 \n"
                                " 9: Matyas function           (2-dimensional).                                 \n"
                                "10: Levi function N.13        (2-dimensional).                                 \n"
                                "11: Himmelblau's function     (2-dimensional).                                 \n"
                                "12: Three-hump camel function (2-dimensional).                                 \n"
                                "13: Easom function            (2-dimensional).                                 \n"
                                "14: Cross-in-tray function    (2-dimensional).                                 \n"
                                "15: Eggholder function        (2-dimensional).                                 \n"
                                "16: Holder table function     (2-dimensional).                                 \n"
                                "17: McCormick function        (2-dimensional).                                 \n"
                                "18: Schaffer function N. 2    (2-dimensional).                                 \n"
                                "19: Schaffer function N. 4    (2-dimensional).                                 \n"
                                "20: Styblinski–Tang function  (n-dimensional).                                 \n"
                                "21: Stability test function 1 (1-dimensional).                                 \n"
                                "22: Stability test function 2 (1-dimensional).                                 \n"
                                "23: Test function 3           (currently not working).                         \n"
                                "24: Drop-wave                 (n-dimensional).                                 \n"
                                "25: Gramancy and Lee          (1-dimensional).                                 \n"
                                "26: Langermann function       (2-dimensional).                                 \n"
                                "27: Griewank function         (n-dimensional).                                 \n"
                                "28: Levy function             (n-dimensional).                                 \n"
                                "29: Schwefel function         (n-dimensional).                                 \n"
                                "30: Shubert function          (2-dimensional).                                 \n"
                                "31: Bohachevsky Function 1    (2-dimensional).                                 \n"
                                "32: Bohachevsky Function 2    (2-dimensional).                                 \n"
                                "33: Bohachevsky Function 3    (2-dimensional).                                 \n"
                                "34: Perm function 0,D,1       (-dimensional).                                  \n"
                                "35: Currin test function      (2-dimensional + 1-fidelity).                    \n"
                                "                                                                               \n"
                                "The x argument later provided must have dimension stated. The optional second  \n"
                                "argument type can take two values: \"norm\" (default), which returns a function\n"
                                "on the range [0,1]^n with range [0,1]; and \"raw\", which returns the test     \n"
                                "function with it's conventional domain and range.                              ",
                                py::arg("i"),py::arg("type")="norm");
    m_maths.def("calc",&callintercalc,"Calculator (explore functions available in gentype expressions).");

    QIMPA(m_maths_fn,sqrt,     "square root"                         )
    QIMPA(m_maths_fn,cbrt,     "cube root"                           )
    QIMPB(m_maths_fn,nthrt,    "nth root"                            )
    QIMPA(m_maths_fn,exp,      "exponential"                         )
    QIMPA(m_maths_fn,tenup,    "ten to the power"                    )
    QIMPA(m_maths_fn,log,      "natural log"                         )
    QIMPA(m_maths_fn,log10,    "log base 10"                         )
    QIMPB(m_maths_fn,logb,     "log base b"                          )
    QIMPB(m_maths_fn,logbl,    "log base b (left-handed)"            )
    QIMPB(m_maths_fn,logbr,    "log base b (right-handed)"           )
    QIMPA(m_maths_fn,sin,      "sine"                                )
    QIMPA(m_maths_fn,cos,      "cosine"                              )
    QIMPA(m_maths_fn,tan,      "tangent"                             )
    QIMPA(m_maths_fn,cosec,    "cosecant"                            )
    QIMPA(m_maths_fn,sec,      "secant"                              )
    QIMPA(m_maths_fn,cot,      "cotangent"                           )
    QIMPA(m_maths_fn,asin,     "inverse sine"                        )
    QIMPA(m_maths_fn,acos,     "inverse cosine"                      )
    QIMPA(m_maths_fn,atan,     "inverse tangent"                     )
    QIMPA(m_maths_fn,acosec,   "inverse cosecant"                    )
    QIMPA(m_maths_fn,asec,     "inverse secant"                      )
    QIMPA(m_maths_fn,acot,     "inverse cotangent"                   )
    QIMPA(m_maths_fn,sinc,     "sinc"                                )
    QIMPA(m_maths_fn,cosc,     "cosinc"                              )
    QIMPA(m_maths_fn,tanc,     "cotanc"                              )
    QIMPA(m_maths_fn,vers,     "versed sine"                         )
    QIMPA(m_maths_fn,covers,   "coversed sine"                       )
    QIMPA(m_maths_fn,hav,      "half versed sine"                    )
    QIMPA(m_maths_fn,excosec,  "external cosecant"                   )
    QIMPA(m_maths_fn,exsec,    "external secant"                     )
    QIMPA(m_maths_fn,avers,    "inverse versed sine"                 )
    QIMPA(m_maths_fn,acovers,  "inverse coversed sine"               )
    QIMPA(m_maths_fn,ahav,     "inverse half versed sine"            )
    QIMPA(m_maths_fn,aexcosec, "inverse external cosecant"           )
    QIMPA(m_maths_fn,aexsec,   "inverse external secant"             )
    QIMPA(m_maths_fn,castrg,   "cas"                                 )
    QIMPA(m_maths_fn,casctrg,  "complementary cas"                   )
    QIMPA(m_maths_fn,acastrg,  "inverse cas"                         )
    QIMPA(m_maths_fn,acasctrg, "inverse complementary cas"           )
    QIMPA(m_maths_fn,sinh,     "hyperbolic sin"                      )
    QIMPA(m_maths_fn,cosh,     "hyperbolic cosine"                   )
    QIMPA(m_maths_fn,tanh,     "hyperbolic tangent"                  )
    QIMPA(m_maths_fn,cosech,   "hyperbolic cosecant"                 )
    QIMPA(m_maths_fn,sech,     "hyperbolic secant"                   )
    QIMPA(m_maths_fn,coth,     "hyperbolic cotangent"                )
    QIMPA(m_maths_fn,asinh,    "inverse hyperbolic sine"             )
    QIMPA(m_maths_fn,acosh,    "inverse hyperbolic cosine"           )
    QIMPA(m_maths_fn,atanh,    "inverse hyperbolic tangent"          )
    QIMPA(m_maths_fn,acosech,  "inverse hyperbolic cosecant"         )
    QIMPA(m_maths_fn,asech,    "inverse hyperbolic secant"           )
    QIMPA(m_maths_fn,acoth,    "inverse hyperbolic cotangent"        )
    QIMPA(m_maths_fn,sinhc,    "hyperbolic sinc"                     )
    QIMPA(m_maths_fn,coshc,    "hyperbolic cosc"                     )
    QIMPA(m_maths_fn,tanhc,    "hyperbolic tanc"                     )
    QIMPA(m_maths_fn,versh,    "hyperbolic versed sine"              )
    QIMPA(m_maths_fn,coversh,  "hyperbolic coversed sine"            )
    QIMPA(m_maths_fn,havh,     "hyperbolic half-versed sine"         )
    QIMPA(m_maths_fn,excosech, "hyperbolic external cosecant"        )
    QIMPA(m_maths_fn,exsech,   "hyperbolic external secant"          )
    QIMPA(m_maths_fn,aversh,   "hyperbolic half-versed secand"       )
    QIMPA(m_maths_fn,Aversh,   "inverse hyperbolic versed sine"      )
    QIMPA(m_maths_fn,acovrsh,  "inverse hyperbolic coversed sine"    )
    QIMPA(m_maths_fn,ahavh,    "inverse hyperbolic half-versed sine" )
    QIMPA(m_maths_fn,aexcosech,"inverse hyperbolic external cosecant")
    QIMPA(m_maths_fn,aexsech,  "inverse hyperbolic external secant"  )
    QIMPA(m_maths_fn,cashyp,   "hyperbolic cas"                      )
    QIMPA(m_maths_fn,caschyp,  "hyperbolic complementary cas"        )
    QIMPA(m_maths_fn,acashyp,  "inverse hyperbolic cas"              )
    QIMPA(m_maths_fn,acaschyp, "inverse hyperbolic complementary cas")
    QIMPA(m_maths_fn,sigm,     "sigmoid"                             )
    QIMPA(m_maths_fn,gd,       "Gudermanian"                         )
    QIMPA(m_maths_fn,asigm,    "inverse sigmoid"                     )
    QIMPA(m_maths_fn,agd,      "inverse Gudermanian"                 )
    QIMPB(m_maths_fn,bern,     "Bernstein polynomial of order x"     )
    QIMPA(m_maths_fn,normDistr,"normal distribution"                 )
    QIMPB(m_maths_fn,polyDistr,"polynomial distribution of order y"  )
    QIMPA(m_maths_fn,gamma,    "gamma"                               )
    QIMPA(m_maths_fn,lngamma,  "log gamma"                           )
    QIMPA(m_maths_fn,psi,      "digamma"                             )
    QIMPA(m_maths_fn,zeta,     "polygamma function of index x"       )
    QIMPA(m_maths_fn,lambertW, "Lambert W (main branch W0, W>-1)"    )
    QIMPA(m_maths_fn,lambertWx,"Lambert W (lower branch W1, W<-1)"   )
    QIMPA(m_maths_fn,erf,      "error"                               )
    QIMPA(m_maths_fn,erfc,     "complementary error"                 )
    QIMPA(m_maths_fn,dawson,   "Dawson"                              )

    QIMPA(m_maths_fn,Sqrt,     "square root (alternate cut)"                         )
    QIMPA(m_maths_fn,Cbrt,     "cube root (alternate cut)"                           )
    QIMPB(m_maths_fn,Nthrt,    "nth root (alternate cut)"                            )
    QIMPA(m_maths_fn,Log,      "natural log (alternate cut)"                         )
    QIMPA(m_maths_fn,Log10,    "log base 10 (alternate cut)"                         )
    QIMPB(m_maths_fn,Logb,     "log base b (alternate cut)"                          )
    QIMPB(m_maths_fn,Logbl,    "log base b (left-handed, alternate cut)"             )
    QIMPB(m_maths_fn,Logbr,    "log base b (right-handed, alternate cut)"            )
    QIMPA(m_maths_fn,Asin,     "inverse sine (alternate cut)"                        )
    QIMPA(m_maths_fn,Acos,     "inverse cosine (alternate cut)"                      )
    QIMPA(m_maths_fn,Acosec,   "inverse cosecant (alternate cut)"                    )
    QIMPA(m_maths_fn,Asec,     "inverse secant (alternate cut)"                      )
    QIMPA(m_maths_fn,Avers,    "inverse versed sine (alternate cut)"                 )
    QIMPA(m_maths_fn,Acovers,  "inverse coversed sine (alterate cut)"                )
    QIMPA(m_maths_fn,Ahav,     "inverse half versed sine (alternate cut)"            )
    QIMPA(m_maths_fn,Aexcosec, "inverse external cosecant (alternate cut)"           )
    QIMPA(m_maths_fn,Aexsec,   "inverse external secant (alternate cut)"             )
    QIMPA(m_maths_fn,Acasctrg, "inverse complementary cas (alternate cut)"           )
    QIMPA(m_maths_fn,Acosh,    "inverse hyperbolic cosine (alternate cut)"           )
    QIMPA(m_maths_fn,Atanh,    "inverse hyperbolic tangent (alternate cut)"          )
    QIMPA(m_maths_fn,Asech,    "inverse hyperbolic secant (alternate cut)"           )
    QIMPA(m_maths_fn,Acoth,    "inverse hyperbolic cotangent (alternate cut)"        )
    QIMPA(m_maths_fn,Ahavh,    "inverse hyperbolic half-versed sine (alternate cut)" )
    QIMPA(m_maths_fn,Aexsech,  "inverse hyperbolic external secant (alternate cut)"  )
    QIMPA(m_maths_fn,Acaschyp, "inverse hyperbolic complementary cas (alternate cut)")
    QIMPA(m_maths_fn,Acastrg,  "inverse cas (alternate cut)"                         )
    QIMPA(m_maths_fn,Acashyp,  "inverse hyperbolic cas (alternate cut)"              )
    QIMPA(m_maths_fn,Asigm,    "inverse sigmoid (alternate cut)"                     )
    QIMPA(m_maths_fn,Agd,      "inverse Gudermanian (alternate cut)"                 )
    QIMPB(m_maths_fn,PolyDistr,"polynomial distribution of order y (alternate cut)"  )












    // ---------------------------
    // ---------------------------
    // ---------------------------
    // ---------------------------

    auto m_ml = m.def_submodule("ml","Machine Learning Modules.");

    QGETSETD(m_ml,getMLType,ssetMLTypeClean,"type","settype", "ML type. Types are (strings):\n"
                                            "                                                                               \n"
                                            "  s - SVM: single class classifier.                                            \n"
                                            "  c - SVM: binary classification (default).                                    \n"
                                            "  m - SVM: multi-class classification.                                         \n"
                                            "  r - SVM: scalar regression.                                                  \n"
                                            "  v - SVM: vector regression.                                                  \n"
                                            "  a - SVM: anionic regression.                                                 \n"
                                            "  u - SVM: cyclic regression.                                                  \n"
                                            "  g - SVM: gentype regression (any target).                                    \n"
                                            "  p - SVM: density estimation (1-norm base, kernel can be non-Mercer.          \n"
                                            "  t - SVM: pareto frontier SVM.                                                \n"
                                            "  l - SVM: binary scoring (zero bias by default).                              \n"
                                            "  o - SVM: scalar regression with scoring.                                     \n"
                                            "  q - SVM: vector regression with scoring.                                     \n"
                                            "  i - SVM: planar regression.                                                  \n"
                                            "  h - SVM: multi-expert ranking.                                               \n"
                                            "  j - SVM: multi-expert binary classification.                                 \n"
                                            "  b - SVM: similarity learning.                                                \n"
                                            "  d - SVM: basic SVM for kernel inheritance (-x).                              \n"
                                            "  B - SVM: binary classifier using random FF (kernels 3,4,13,19).              \n"
                                            "  R - SVM: scalar regression using random FF (kernels 3,4,13,19).              \n"
                                            "lsc - LS-SVM: binary classification.                                           \n"
                                            "lsr - LS-SVM: scalar regression.                                               \n"
                                            "lsv - LS-SVM: vector regression.                                               \n"
                                            "lsa - LS-SVM: anionic regression.                                              \n"
                                            "lsg - LS-SVM: gentype regression.                                              \n"
                                            "lso - LS-SVM: scalar regression with scoring.                                  \n"
                                            "lsq - LS-SVM: vector regression with scoring.                                  \n"
                                            "lsi - LS-SVM: planar regression.                                               \n"
                                            "lsh - LS-SVM: multi-expert ranking.                                            \n"
                                            "lsR - LS-SVM: scalar regression random FF (kernels 3,4,13,19).                 \n"
                                            "gpc - GPR: gaussian process binary classification (unreliable).                \n"
                                            "gpr - GPR: gaussian process scalar regression.                                 \n"
                                            "gpv - GPR: gaussian process vector regression.                                 \n"
                                            "gpa - GPR: gaussian process anionic regression.                                \n"
                                            "gpg - GPR: gaussian process gentype regression.                                \n"
                                            "gpC - GPR: gaussian process binary classify RFF (kernels 3,4,13,19).           \n"
                                            "gpR - GPR: gaussian process scalar regression RFF (kernels 3,4,13,19).         \n"
                                            "knc - KNN: binary classification.                                              \n"
                                            "knm - KNN: multiclass classification.                                          \n"
                                            "knr - KNN: scalar regression.                                                  \n"
                                            "knv - KNN: vector regression.                                                  \n"
                                            "kna - KNN: anionic regression.                                                 \n"
                                            "kng - KNN: gentype regression.                                                 \n"
                                            "knp - KNN: density estimation.                                                 \n"
                                            "ei  - IMP: expected (hypervolume) improvement.                                 \n"
                                            "svm - IMP: 1-norm 1-class modded SVM mono-surrogate.                           \n"
                                            "rls - IMP: Random linear scalarisation.                                        \n"
                                            "rns - IMP: Random draw from a GP xformed into an increasing fn on [0,1]^d.     \n"
                                            "nop - BLK: NOP machine (holds data but does nothing, posterior = prior.        \n"
                                            "mer - BLK: Mercer kernel inheritance block.                                    \n"
                                            "con - BLK: consensus machine.                                                  \n"
                                            "fna - BLK: user function machine (elementwise).*                               \n"
                                            "fnb - BLK: user function machine (vectorwise).*                                \n"
                                            "mxa - BLK: mex function machine (elementwise).                                 \n"
                                            "mxb - BLK: mex function machine (vectorwise).                                  \n"
                                            "io  - BLK: user I/O machine.                                                   \n"
                                            "sys - BLK: system call machine.                                                \n"
                                            "avr - BLK: scalar averaging machine.                                           \n"
                                            "avv - BLK: vector averaging machine.                                           \n"
                                            "ava - BLK: anionic averaging machine.                                          \n"
                                            "fcb - BLK: function callback (do not use).                                     \n"
                                            "ber - BLK: Bernstein basis polynomial.                                         \n"
                                            "bat - BLK: Battery model.**                                                    \n"
                                            "ker - BLK: kernel specialisation.***                                           \n"
                                            "mba - BLK: multi-block sum.                                                    ");

    m_ml.def("sel",    &selml,   "Select ML i > 0. If i = 0, return current ML (default 1) without modification. \n"
                                 "You can have arbitrarily many models at any given time. Notes:                 \n"
                                 "                                                                               \n"
                                 "- if i is negative then the ML is a member of a BO. See e.g. selaltmu etc. for \n"
                                 "  more information.                                                            \n"
                                 "                                                                               \n"
                                 "See also: selmlmuapprox, selmlcgtapprox, selmlsigmaapprox, selmldiffmodel,     \n"
                                 "          selmlsrcmodel, selmlmuapprox_prior, selmlcgtapprox_prior,            \n"
                                 "          selmlsigmaapprox_prior, selmldiffmodel_prior, selmlsrcmodel_prior    ",
                                 py::arg("i")=0);
    m_ml.def("swap",   &swapml,  "Swap ML and ML j.",py::arg("i")=0,py::arg("j")=0);
    m_ml.def("copy",   &copyml,  "Let ML := ML j.",  py::arg("i")=0,py::arg("j")=0);
    m_ml.def("assign", &assignml,"Let ML j := ML.",  py::arg("i")=0,py::arg("j")=0);

    QDO(m_ml,train,  "training (if required)."                    );
    QDO(m_ml,reset,  "undo training (alpha,bias = 0, if defined).");
    QDO(m_ml,restart,"removing training data and reset."          );

    QDOARG(m_ml,scale,    "scale training variables (alpha,bias) by sf.","sf");
    QDOARG(m_ml,randomise,"randomise (alpha,bias) with sparsity 0<=s<=1","s" );

    QGETSETD(m_ml,prim,  setprim,  "pritype","setpritype","set prior mean type:\n"
                                                          "                                                                               \n"
                                                          "0 - no (0) prior mean (default).                                               \n"
                                                          "1 - primu(x) defined directly.                                                 \n"
                                                          "2 - prior is set to posterior mean of ML priML.                                ");
    QGETSETD(m_ml,prival,setprival,"primu",  "setprimu",  "explicit prior mean mu (assuming pritype=1)");
    m_ml.def("setpriML", &setpriml,"Set prior mean to posterior mean of ML j (assuming pritype=2).",py::arg("j"));

    QGETSETD(m_ml,tspaceDim,settspaceDim,"tspaceDim","settspaceDim", "target space dimension");
    QGETSETD(m_ml,order,    setorder,    "order",    "setorder",     "target space order"    );

    QGET(m_ml,N,                 "number of training vectors" );
    QGET(m_ml,type,              "ML type number"             );
    QGET(m_ml,subtype,           "ML subtype number"          );
    QGET(m_ml,xspaceDim,         "input space dimension"      );
    QGET(m_ml,fspaceDim,         "function space dimension"   );
    QGET(m_ml,tspaceSparse,      "is target space sparse?"    );
    QGET(m_ml,xspaceSparse,      "is input space sparse?"     );
    QGET(m_ml,numClasses,        "number of classes"          );
    QGET(m_ml,ClassLabels,       "class labels"               );
    QGET(m_ml,isTrained,         "is ML trained?"             );
    QGET(m_ml,isSolGlob,         "is solution global?"        );
    QGET(m_ml,isUnderlyingScalar,"is underlying scalar?"      );
    QGET(m_ml,isUnderlyingVector,"is underlying vector?"      );
    QGET(m_ml,isUnderlyingAnions,"is underlying anionic?"     );
    QGET(m_ml,isClassifier,      "is ML a classifier?"        );
    QGET(m_ml,isRegression,      "is ML a regressor?"         );
    QGET(m_ml,isPlanarType,      "is ML a planar-type method?");

    QGETCLA(m_ml,NNC,"number of active training vectors for class d");

    QGET(m_ml,x,          "training vectors"                         );
    QGET(m_ml,d,          "training classifications/constraint types");
    QGET(m_ml,y,          "training targets"                         );
    QGET(m_ml,yp,         "training target priors"                   );
    QGET(m_ml,Cweight,    "training target C weights"                );
    QGET(m_ml,Cweightfuzz,"training target C weight fuzzing"         );
    QGET(m_ml,sigmaweight,"training sigma weights"                   );
    QGET(m_ml,epsweight,  "training eps weights"                     );
    QGET(m_ml,alphaState, "training alpha states"                    );
    QGET(m_ml,xtang,      "training class/vector/type specifics"     );

    m_ml.def("alpha",&mlalpha,"Get training alpha (SVM,LSV,GP).");
    m_ml.def("bias", &mlbias, "Get training bias (SVM,LSV,GP)." );

    m_ml.def("setalpha",&mlsetalpha,"Set training alpha (SVM,LSV,GP).",py::arg("alpha"));
    m_ml.def("setbias", &mlsetbias, "Set training bias (SVM,LSV,GP).", py::arg("bias") );

    m_ml.def("Gp",&mlGp,"Get the Gp matrix (SVM,LSV,GP).");

    QGET(m_ml,loglikelihood,"log-likelihood (quasi for SVM,LSV, actual for GP)."      );
    QGET(m_ml,maxinfogain,  "max-information-gain (quasi for SVM,LSV, actual for GP).");
    QGET(m_ml,RKHSnorm,     "RKHS norm ||f||_H^2 (SVM,LSV,GP)."                       );
    QGET(m_ml,RKHSabs,      "RKHS norm ||f||_H (SVM,LSV,GP)."                         );

    m_ml.def("constrain",&makeMonot,"Constrain the posterior mean / trained ML using inducing-point method.         \n"
                                    "                                                                               \n"
                                    "This adds n training observations (inducing points) where the constraints are  \n"
                                    "strictly enforced, with the goal of global enforcement if the inducing points  \n"
                                    "are sufficiently dense. The grid type is controlled by parameter t:            \n"
                                    "                                                                               \n"
                                    "t = 0: uniform grid of inducing points.                                        \n"
                                    "t = 1: random inducing points drawn from uniform distribution.                 \n"
                                    "t = 2: inducing points drawn randomly from data in ML j.                       \n"
                                    "                                                                               \n"
                                    "The inducing points themselves are generated from xlb, xub and xb. First, a    \n"
                                    "vector x in the range xlb < x < xub is generated as defined by t, with the     \n"
                                    "dimension (and sparsity) following xlb and xub. Then xb is incorporated to give\n"
                                    "the full inducing point. For example:                                          \n"
                                    "                                                                               \n"
                                    "     xb  = ['::',1,0.5]                                                        \n"
                                    "     xlb = [0,0]                                                               \n"
                                    "     xub = [1,1]                                                               \n"
                                    "                                                                               \n"
                                    "will generate points of the form x = [ x1 x2 :: 1 0.5 ], where x1,x2 are in the\n"
                                    "range [0,1]. At each inducing point the form of constraint is controlled by the\n"
                                    "parameters d and y. y is the target, and d controls the type of constraint:    \n"
                                    "                                                                               \n"
                                    "d = 0:  no constraint enforced.                                                \n"
                                    "d = 2:  equality constraint g(x) = y enforced.                                 \n"
                                    "d = +1: lower bound constraint g(x) >= y enforced.                             \n"
                                    "d = +1: upper bound constraint g(x) <= y enforced.                             \n"
                                    "                                                                               \n"
                                    "So in our example if we let y=0 and d=+1 then the inducing points will enforce:\n"
                                    "                                                                               \n"
                                    "     g([x1 x2 :: 1 0.5]) = d/dx1 g(x) + 0.5.d/dx2 g(x) >= 0                    \n"
                                    "                                                                               \n"
                                    "which, for a sufficiently dense set of inducing points, will approximate a     \n"
                                    "global constraint on the gradient. You can also include C and epsilon weights. \n"
                                    "                                                                               \n"
                                    "Defaults are n=10^d, t=1, d=1.                                                 ",
                                    py::arg("n")=-1,py::arg("t")=1,py::arg("xb")=py::none(),py::arg("xlb")=py::none(),
                                    py::arg("xub")=py::none(),py::arg("d")=1,py::arg("y")=py::none(),
                                    py::arg("Cweight")=1.0,py::arg("epsweight")=1.0,py::arg("j")=0);

    m_ml.def("add", &addTrainingVectorml, "Add a single training vector pair [z,x] at position j (j=-1 to add at end),    \n"
                                          "where x is a list (vector). To add multiple training pairs use z = (z1,z2,...),\n"
                                          "x=(x1,x2,...). To set d, Cweight etc use z = {\"y\":y, \"d\":d, \"cw\":cw,     \n"
                                          "\"ew\":ew} (all keys are optional).                                            ",
                                          py::arg("x"),py::arg("z")=py::none(),py::arg("j")=-1);
    m_ml.def("addf",&faddTrainingVectorml,"Add up to imax (let imax=0 (default) for all) training vector pairs [z,x] from \n"
                                          "file fname at position j in the ML (j=-1 (default) to add, skipping ignoreStart\n"
                                          "(default 0) training pairs at start of file.                                   \n"
                                          "                                                                               \n"
                                          "By default it is assumed that the file is in target-at-start format, but you   \n"
                                          "can use target-at-end format by setting reverse=1 (reverse=0 by default). You  \n"
                                          "can include Cweight, epsweight and d as described in the CLI documentation.    ",
                                          py::arg("fname"),py::arg("ignoreStart")=0,py::arg("imax")=-1,py::arg("reverse")=0,
                                          py::arg("j")=-1);

    m_ml.def("remove",&removeTrainingVectorml,"Remove num training vectors at position j (j=-1 (default) to remove from end)",
                                              py::arg("j")=-1,py::arg("num")=1);

    m_ml.def("mu", &muml,         "Calculate the posterior mean (output) mu(x), where x is either a vector (list) \n"
                                  "or an integer indexing a training vector in the ML training set. To evaluate   \n"
                                  "multiple posterior means, let x be a tuple of vectors (lists).                 ",
                                  py::arg("x"));
    m_ml.def("g",  &mugml,        "Calculate the underlying (ie continuous) output g(x), where x is either a      \n"
                                  "vector (list) or an integer indexing a training vector in the ML training set. \n"
                                  "Set fmt=1 (fmt=0 by default) for alternate return format (eg a vector if the ML\n"
                                  "is (nominally) a vector type at base or a vector of probabilities for a GP     \n"
                                  "binary classifier). To evaluate multiple outpuys, let x be a tuple of vectors  \n"
                                  "(lists).                                                                       ",
                                  py::arg("x")=py::none(),py::arg("fmt")=0);
    m_ml.def("var",&varml,        "Calculate the posterior variance var(x), where x is either a vector (list) or  \n"
                                  "an integer indexing a training vector in the ML training set. To evaluated     \n"
                                  "multiple posterior variances, let x be a tuple of vectors (lists).             ",
                                  py::arg("x"));
    m_ml.def("cov",&covml,        "Calculate the posterior covariance cov(x,y), where each of x and y can be      \n"
                                  "either a vector (list) or an integer indexing a training vector in the ML      \n"
                                  "training set. To evaluated multiple posterior covariances, let x and/or y be a \n"
                                  "tuple of vectors (lists).                                                      ",
                                  py::arg("xa"),py::arg("xb"));
    m_ml.def("predvar",&predvarml,"Calculate the predictive posterior variance var(x) predicated on z being added \n"
                                  "to the training set (ie. if we observed g(z), what would the posterior variance\n"
                                  "var(x) be). Each of x and z can be either a vector (list) or an integer        \n"
                                  "indexing a training vector in the ML trainig set. To evaluates multiple        \n"
                                  "predictive variances, let x and/or z be a tuple of vectors (lists).            \n"
                                  "                                                                               \n"
                                  "Optional argument: sigw (default 1.0) controls the noise of the predicated     \n"
                                  "(assumed to be taken) observation g(z) ~ N(...,sigw*sigma()).                  ",
                                  py::arg("x"),py::arg("p"),py::arg("sigw")=1);
    m_ml.def("predcov",&predcovml,"Calculate the predictive posterior covariance cov(x,y) predicated on z being   \n"
                                  "added to the training set (ie. if we observed g(z), what would the posterior   \n"
                                  "covariance cov(x,y) be). Each of x,y and z can be either a vector (list) or an \n"
                                  "integer indexing a training vector in the ML trainig set. To evaluate multiple \n"
                                  "predictive covariances, let x,y and/or z be tuples of vectors (lists).         \n"
                                  "                                                                               \n"
                                  "Optional argument: sigw (default 1.0) controls the noise of the predicated     \n"
                                  "(assumed to be taken) observation g(z) ~ N(...,sigw*sigma()).                  ",
                                  py::arg("xa"),py::arg("xb"),py::arg("p"),py::arg("sigw")=1);

    m_ml.def("tuneKernel",&mltuneKernel,"Tune the kernel to minimise some metric, specified by method:                  \n"
                                        "                                                                               \n"
                                        "1 - negative log-likelihood (default)                                          \n"
                                        "2 - leave-one-out error                                                        \n"
                                        "3 - recall error                                                               \n"
                                        "                                                                               \n"
                                        "Optional arguments are:                                                        \n"
                                        "                                                                               \n"
                                        "xwidth - the maximum kernel lengthscale override (default 1)                   \n"
                                        "tuneK  - 0 don't tune kernel parameters, 1 tune kernel parameters (default)    \n"
                                        "tuneP  - 0 don't tune C (1/sigma) parameter (default), 1 tune C                ",
                                        py::arg("method")=2,py::arg("xwidth")=1,py::arg("tuneK")=1,py::arg("tuneP")=0);

    m_ml.def("K0",&mlK0,"Calculate K0()."                                                                   );
    m_ml.def("K1",&mlK1,"Calculate K1(xa).",         py::arg("xa")                                          );
    m_ml.def("K2",&mlK2,"Calculate K2(xa,xb).",      py::arg("xa"),py::arg("xb")                            );
    m_ml.def("K3",&mlK3,"Calculate K3(xa,xb,xc).",   py::arg("xa"),py::arg("xb"),py::arg("xc")              );
    m_ml.def("K4",&mlK4,"Calculate K4(xa,xb,xc,xd).",py::arg("xa"),py::arg("xb"),py::arg("xc"),py::arg("xd"));

    m_ml.def("calcLOO",   &mlcalcLOO,   "Calculate leave-one-out error.");
    m_ml.def("calcRecall",&mlcalcRecall,"Calculate recall error."       );
    m_ml.def("calcCross", &mlcalcCross, "Calculate n-fold validation error. If numreps>1 then does numreps repetitions, \n"
                                        "which are randomised in rndit=1.                                               ",
                                        py::arg("n"),py::arg("rndit")=0,py::arg("numreps")=1);

    // ---------------------------

    auto m_ml_svm = m_ml.def_submodule("svm","Support Vector Machines specific options.");

    QGETSETD(m_ml_svm,getMLType,ssetMLTypeClean,"type","settype", "SVM type. Types are:\n"
                                                                  "                                                                               \n"
                                                                  "  s - SVM: single class.                                                       \n"
                                                                  "  c - SVM: binary classification (default).                                    \n"
                                                                  "  m - SVM: multiclass classification.                                          \n"
                                                                  "  r - SVM: scalar regression.                                                  \n"
                                                                  "  v - SVM: vector regression.                                                  \n"
                                                                  "  a - SVM: anionic regression.                                                 \n"
                                                                  "  u - SVM: cyclic regression.                                                  \n"
                                                                  "  g - SVM: gentype regression (any target).                                    \n"
                                                                  "  p - SVM: density estimation (1-norm base, kernel can be non-Mercer.          \n"
                                                                  "  t - SVM: pareto frontier SVM.                                                \n"
                                                                  "  l - SVM: binary scoring (zero bias by default).                              \n"
                                                                  "  o - SVM: scalar regression with scoring.                                     \n"
                                                                  "  q - SVM: vector regression with scoring.                                     \n"
                                                                  "  i - SVM: planar regression.                                                  \n"
                                                                  "  h - SVM: multi-expert ranking.                                               \n"
                                                                  "  j - SVM: multi-expert binary classification.                                 \n"
                                                                  "  b - SVM: similarity learning.                                                \n"
                                                                  "  d - SVM: basic SVM for kernel inheritance (-x).                              \n"
                                                                  "  B - SVM: binary classifier using random FF (kernels 3,4,13,19).              \n"
                                                                  "  R - SVM: scalar regression using random FF (kernels 3,4,13,19).              ");
    QGETSETD(m_ml_svm,getVmethod,setVmethod,"typeVR","settypeVR", "SVM vector-regression method.  Methods are:\n"
                                                                  "                                                                               \n"
                                                                  "once - at-once regression.                                                     \n"
                                                                  "red  - reduction to binary regression (default).                               ");
    QGETSETD(m_ml_svm,getCmethod,setCmethod,"typeMC","settypeMC", "SVM multi-class classification method.  Methods are:\n"
                                                                  "                                                                               \n"
                                                                  "1vsA   - 1 versus all (reduction to binary).                                   \n"
                                                                  "1vs1   - 1 versus 1 (reduction to binary).                                     \n"
                                                                  "DAG    - directed acyclic graph (reduct to binary).                            \n"
                                                                  "MOC    - minimum output coding (reduct to binary).                             \n"
                                                                  "maxwin - max-wins SVM (at once).                                               \n"
                                                                  "recdiv - recursive division SVM (at once, default).                            ");
    QGETSETD(m_ml_svm,getOmethod,setOmethod,"typeOC","settypeOC", "SVM one-class method.  Methods are:\n"
                                                                  "                                                                               \n"
                                                                  "sch - Scholkopt 1999 1-class SVM (default).                                    \n"
                                                                  "tax - Tax and Duin 2004, Support Vector Data Description.                      ");
    QGETSETD(m_ml_svm,getAmethod,setAmethod,"typeCM","settypeCM", "SVM classification method.  Methods are:\n"
                                                                  "                                                                               \n"
                                                                  "svc - normal SVM classifier (default).                                         \n"
                                                                  "svr - classify via regression.                                                 ");
    QGETSETD(m_ml_svm,getRmethod,setRmethod,"typeER","settypeER", "SVM empirical risk type.  Methods are:\n"
                                                                  "                                                                               \n"
                                                                  "l - linear (default).                                                          \n"
                                                                  "q - quadratic.                                                                 \n"
                                                                  "o - linear, 1-norm regularization on alpha (not feature space: use -m for that)\n"
                                                                  "g - generalised linear (iterative fuzzy).                                      \n"
                                                                  "G - generalised quadratic (iterative fuzzy).                                   ");
    QGETSETD(m_ml_svm,getTmethod,setTmethod,"typeSM","settypeSM", "SVM tube method.  Methods are:\n"
                                                                  "                                                                               \n"
                                                                  "f - fixed tube (default).                                                      \n"
                                                                  "s - tube shrinking.                                                            ");
    QGETSETD(m_ml_svm,getBmethod,setBmethod,"typeBias","settypeBias","SVM bias method.  Methods are:\n"
                                                                     "                                                                               \n"
                                                                     "var - variable bias (default).                                                 \n"
                                                                     "fix - fixed bias (usually zero).                                               \n"
                                                                     "pos - positive bias.                                                           \n"
                                                                     "neg - negative bias.                                                           ");
    QGETSETD(m_ml_svm,getMmethod,setMmethod,"typeM","settypeM",   "SVM monotonic method (sufficient, not necessary, and only for a few\n"
                                                                  "kernels in finite dimensions, assuming all training x >= 0).  Methods are:     \n"
                                                                  "                                                                               \n"
                                                                  "n - none (default).                                                            \n"
                                                                  "i - increasing.                                                                \n"
                                                                  "d - decreasing.                                                                \n"
                                                                  "                                                                               \n"
                                                                  "Note: this method is a a bit inefficient - recommend not using.                ");

    QGET(m_ml_svm,NZ, "number training vectors with alpha = 0"                              );
    QGET(m_ml_svm,NF, "number training vectors with alpha unconstrained"                    );
    QGET(m_ml_svm,NS, "number training vectors with alpha != 0"                             );
    QGET(m_ml_svm,NC, "number training vectors with alpha constrained"                      );
    QGET(m_ml_svm,NLB,"number training vectors with alpha constrained at lower bound"       );
    QGET(m_ml_svm,NLF,"number training vectors with alpha free between lower bound and zero");
    QGET(m_ml_svm,NUF,"number training vectors with alpha free between zero and upper bound");
    QGET(m_ml_svm,NUB,"number training vectors with alpha constrained at upper bound"       );

    QGET(m_ml_svm,kerndiag,"diagonals of kernel matrix");

    QGETSET(m_ml_svm,C,        setC,        "regularization trade-off (empirical risk weight, C=1/lambda)"        );
    QGETSET(m_ml_svm,sigma,    setsigma,    "regularization trade-off (regularization weight, lambda=1/C)"        );
    QGETSET(m_ml_svm,sigma_cut,setsigma_cut,"sigma scale for JIT sampling"                                        );
    QGETSET(m_ml_svm,eps,      seteps,      "epsilon-insensitivity width"                                         );
    QGETSET(m_ml_svm,m,        setm,        "margin-norm (default 2, or 2-kernel SVM)"                            );
    QGETSET(m_ml_svm,theta,    settheta,    "theta (psd regularization) for similarity learning"                  );
    QGETSET(m_ml_svm,simnorm,  setsimnorm,  "normalized (1, default) or un-normalized (0) similarity learning"    );

    QGETSETCLA(m_ml_svm,Cclass,  setCclass,  "regularization trade-off scale (empirical risk weight) C for class d");
    QGETSETCLA(m_ml_svm,epsclass,setepsclass,"epsilon-insensitivity width eps scale for class d"                   );

    QGETSET(m_ml_svm,LinBiasForce, setLinBiasForce, "linear bias-forcing (binary, on/off)"   );
    QGETSET(m_ml_svm,QuadBiasForce,setQuadBiasForce,"quadratic bias-forcing (binary, on/off)");
    QGETSET(m_ml_svm,nu,           setnu,           "linear tube-shrinking constant"         );
    QGETSET(m_ml_svm,nuQuad,       setnuQuad,       "quadratic tube-shrinking constant"      );

    QGETSETCLA(m_ml_svm,LinBiasForceclass, setLinBiasForceclass, "linear bias-forcing LinBiasForce scale for class d"    );
    QGETSETCLA(m_ml_svm,QuadBiasForceclass,setQuadBiasForceclass,"quadratic bias-forcing QuadBiasForce scale for class d");

       QDO(m_ml_svm,removeNonSupports,"remove all non-support vectors alpha=0");
    QDOARG(m_ml_svm,trimTrainingSet,  "trim training set to target size N","N");

    // ---------------------------

    auto m_ml_lsv = m_ml.def_submodule("lsv","Least-Squares Support Vector Machines specific options.");

    QGETSETD(m_ml_lsv,getMLType,ssetMLTypeClean,"type","settype",    "LSV type. Types are:\n"
                                                                     "                                                                               \n"
                                                                     "lsc - LS-SVM: binary classification.                                           \n"
                                                                     "lsr - LS-SVM: scalar regression.                                               \n"
                                                                     "lsv - LS-SVM: vector regression.                                               \n"
                                                                     "lsa - LS-SVM: anionic regression.                                              \n"
                                                                     "lsg - LS-SVM: gentype regression.                                              \n"
                                                                     "lso - LS-SVM: scalar regression with scoring.                                  \n"
                                                                     "lsq - LS-SVM: vector regression with scoring.                                  \n"
                                                                     "lsi - LS-SVM: planar regression.                                               \n"
                                                                     "lsh - LS-SVM: multi-expert ranking.                                            \n"
                                                                     "lsR - LS-SVM: scalar regression random FF (kernels 3,4,13,19).                 ");
    QGETSETD(m_ml_lsv,getBmethod,setBmethod,"typeBias","settypeBias","LSV bias method.  Methods are:\n"
                                                                     "                                                                               \n"
                                                                     "var - variable bias (default).                                                 \n"
                                                                     "fix - zero bias.                                                               ");

    QGETSET(m_ml_lsv,C,        setC,        "regularization trade-off (empirical risk weight, C=1/lambda)");
    QGETSET(m_ml_lsv,sigma,    setsigma,    "regularization trade-off (regularization weight, lambda=1/C)");
    QGETSET(m_ml_lsv,sigma_cut,setsigma_cut,"sigma scale for JIT sampling"                                );
    QGETSET(m_ml_lsv,eps,      seteps,      "epsilon-insensitivity width"                                 );

    QGETSETCLA(m_ml_lsv,Cclass,  setCclass,  "regularization trade-off scale (empirical risk weight) C for class d");
    QGETSETCLA(m_ml_lsv,epsclass,setepsclass,"epsilon-insensitivity width eps scale for class d"                   );

    // ---------------------------

    auto m_ml_gp = m_ml.def_submodule("gp","Gaussian Process specific options.");

    QGETSETD(m_ml_gp,getMLType,ssetMLTypeClean,"type","settype",    "GP type. Types are:\n"
                                                                    "                                                                               \n"
                                                                    "gpc - GPR: gaussian process binary classification (unreliable).                \n"
                                                                    "gpr - GPR: gaussian process scalar regression.                                 \n"
                                                                    "gpv - GPR: gaussian process vector regression.                                 \n"
                                                                    "gpa - GPR: gaussian process anionic regression.                                \n"
                                                                    "gpg - GPR: gaussian process gentype regression.                                \n"
                                                                    "gpC - GPR: gaussian process binary classify RFF (kernels 3,4,13,19).           \n"
                                                                    "gpR - GPR: gaussian process scalar regression RFF (kernels 3,4,13,19).         ");
    QGETSETD(m_ml_gp,getBmethod,setBmethod,"typeBias","settypeBias","set GP bias method.  Methods are:\n"
                                                                    "                                                                               \n"
                                                                    "var - variable bias.                                                           \n"
                                                                    "fix - zero bias (default).                                                     ");
    QGETSETD(m_ml_gp,getEmethod,setEmethod,"approxType","setapproxType","set GP inequality constraint/classifier approximation method.\n"
                                                                        "Methods are:                                                               \n"
                                                                        "                                                                           \n"
                                                                        "Naive   - enforce alpha sign constraint (posterior variance will be wrong).\n"
                                                                        "EP      - expectation propogation (CURRENTLY NOT WORKING).                 \n"
                                                                        "LapNorm - Laplace approximation using normal CDF likelihood (default).     \n"
                                                                        "LapLog  - Laplace approximation using Logit likelihood.                    ");

    QGETSET(m_ml_gp,sigma,    setsigma,    "measurement noise variance sigma"                 );
    QGETSET(m_ml_gp,sigma_cut,setsigma_cut,"measurement noise variance scale for JIT sampling");

//    QGETSETCLA(m_ml_gp,sigmaclass,setsigmaclass,"measurement noise variance scale factor for class d (sigma -> sigma/Cd)");

    // ---------------------------

    auto m_ml_knn = m_ml.def_submodule("knn", "Kernel nearest neighbours specific options.");

    QGETSETD(m_ml_knn,getMLType,ssetMLTypeClean,"type","settype","KNN type. Types are:\n"
                                                                 "                                                                               \n"
                                                                 "knc - KNN: binary classification.                                              \n"
                                                                 "knm - KNN: multiclass classification.                                          \n"
                                                                 "knr - KNN: scalar regression.                                                  \n"
                                                                 "knv - KNN: vector regression.                                                  \n"
                                                                 "kna - KNN: anionic regression.                                                 \n"
                                                                 "kng - KNN: gentype regression.                                                 \n"
                                                                 "knp - KNN: density estimation.                                                 ");

    QGETCLA(m_ml_knn,NNC,"number of active training vectors for class d");

    QGETSET(m_ml_knn,k,  setk,  "number of neighbours"     );
    QGETSET(m_ml_knn,ktp,setktp,"weight function (see -K).");

    // ---------------------------

    auto m_ml_imp = m_ml.def_submodule("imp","Impulse model specific options.");

    QGETSETD(m_ml_imp,getMLType,ssetMLTypeClean,"type","settype","IMP type. Types are:\n"
                                                                 "                                                                               \n"
                                                                 "ei  - IMP: expected (hypervolume) improvement.                                 \n"
                                                                 "svm - IMP: 1-norm 1-class modded SVM mono-surrogate.                           \n"
                                                                 "rls - IMP: Random linear scalarisation.                                        \n"
                                                                 "rns - IMP: Random draw from GP xformed into an increasing function on [0,1]^d. ");

    // ---------------------------

    auto m_ml_blk = m_ml.def_submodule("blk","Miscellaneous model specific options.");

    QGETSETD(m_ml_blk,getMLType,ssetMLTypeClean,"type","settype","BLK type. Types are:\n"
                                                                 "                                                                               \n"
                                                                 "nop - BLK: no-operation machine (holds data but does nothing with it, lets     \n"
                                                                 "      mu(x),var(x),cov(x,x') remain the prior mean, variance and covariance).  \n"
                                                                 "mer - BLK: Mercer kernel inheritance block.                                    \n"
                                                                 "con - BLK: consensus machine.                                                  \n"
                                                                 "fna - BLK: user function machine (elementwise).*                               \n"
                                                                 "fnb - BLK: user function machine (vectorwise).*                                \n"
                                                                 "mxa - BLK: mex function machine (elementwise).                                 \n"
                                                                 "mxb - BLK: mex function machine (vectorwise).                                  \n"
                                                                 "io  - BLK: user I/O machine.                                                   \n"
                                                                 "sys - BLK: system call machine.                                                \n"
                                                                 "avr - BLK: scalar averaging machine.                                           \n"
                                                                 "avv - BLK: vector averaging machine.                                           \n"
                                                                 "ava - BLK: anionic averaging machine.                                          \n"
                                                                 "fcb - BLK: function callback (do not use).                                     \n"
                                                                 "ber - BLK: Bernstein basis polynomial.                                         \n"
                                                                 "bat - BLK: Lead-acid battery model.**                                          \n"
                                                                 "ker - BLK: kernel specialisation.***                                           \n"
                                                                 "mba - BLK: multi-block sum.                                                    ");



















    // ---------------------------
    // ---------------------------
    // ---------------------------
    // ---------------------------

    auto m_opt = m.def_submodule("opt","Built-in Optimisation Methods.");

    auto m_opt_grid       = m_opt.def_submodule("grid",      "Grid Optimisation"       );
    auto m_opt_DIRect     = m_opt.def_submodule("DIRect",    "DIRect Optimisation"     );
    auto m_opt_NelderMead = m_opt.def_submodule("NelderMead","Nelder-Mead Optimisation");
    auto m_opt_Bayesian   = m_opt.def_submodule("BO",        "Bayesian Optimisation"   );

    m_opt_grid.def(      "opt", &mloptgrid,       "Optimise (minimise) fn : [0,1]^dim to [0,1] using grid optimiser i.",       py::arg("i")=0,py::arg("dim")=1,py::arg("numreps")=1,py::arg("fn"),py::arg("callback")=py::none());
    m_opt_DIRect.def(    "opt", &mloptDIRect,     "Optimise (minimise) fn : [0,1]^dim to [0,1] using DIRect optimiser i.",     py::arg("i")=0,py::arg("dim")=1,py::arg("numreps")=1,py::arg("fn"),py::arg("callback")=py::none());
    m_opt_NelderMead.def("opt", &mloptNelderMead, "Optimise (minimise) fn : [0,1]^dim to [0,1] using Nelder-Mead optimiser i.",py::arg("i")=0,py::arg("dim")=1,py::arg("numreps")=1,py::arg("fn"),py::arg("callback")=py::none());
    m_opt_Bayesian.def(  "opt", &mloptBayesian,   "Optimise (minimise) fn : [0,1]^dim to [0,1] using Bayesian optimiser i.",   py::arg("i")=0,py::arg("dim")=1,py::arg("numreps")=1,py::arg("fn"),py::arg("callback")=py::none());

    auto m_opt_Bayesian_model = m_opt_Bayesian.def_submodule("models","Model Options"               );
    auto m_opt_Bayesian_tune  = m_opt_Bayesian.def_submodule("models","Model Tuning"                );
    auto m_opt_Bayesian_gpucb = m_opt_Bayesian.def_submodule("gpucb", "GP-UCB related constants"    );
    auto m_opt_Bayesian_ei    = m_opt_Bayesian.def_submodule("ei",    "EI related constants"        );
    auto m_opt_Bayesian_cgt   = m_opt_Bayesian.def_submodule("cgt",   "Constraints"                 );
    auto m_opt_Bayesian_fid   = m_opt_Bayesian.def_submodule("fid",   "Multi-fidelity"              );
    auto m_opt_Bayesian_vis   = m_opt_Bayesian.def_submodule("vis",   "Visualization and Plotting"  );
    auto m_opt_Bayesian_ts    = m_opt_Bayesian.def_submodule("ts",    "Thompson Sampling"           );
    auto m_opt_Bayesian_tl    = m_opt_Bayesian.def_submodule("tl",    "Transfer learning"           );
    auto m_opt_Bayesian_moo   = m_opt_Bayesian.def_submodule("moo",   "Multi-objective learning"    );
    auto m_opt_Bayesian_mr    = m_opt_Bayesian.def_submodule("mr",    "Multi-recommendation methods");

    // All optimisers

    QGETSETOPTALL(m_opt,optname,     "base-string from which all logfile names are derived"    );
    QGETSETOPTALL(m_opt,maxtraintime,"maximum training time in seconds (default 0, unlimited)" );
    QGETSETOPTALL(m_opt,softmin,     "soft minimum (clip if found) on objective (default -inf)");
    QGETSETOPTALL(m_opt,softmax,     "soft maximum (clip if found) on objective (default +inf)");
    QGETSETOPTALL(m_opt,hardmin,     "hard minimum (stop if found) on objective (default -inf)");
    QGETSETOPTALL(m_opt,hardmax,     "hard maximum (stop if found) on objective (default +inf)");

    // Grid options

    m_opt_grid.def("selgridopt",&selgridopt,"Select grid optimiser i > 0. If i=0 then return current grid optimizer (default\n"
                                            "1) without modification. You can have arbitrarily many grid optimizers at any  \n"
                                            "given time. Set rst = 1 to also reset.                                         ",
                                            py::arg("i")=0,py::arg("rst")=0);

    QGETSETOPTB(m_opt,numZooms,grid,"number of grid-zooms (default 0)"                                              );
    QGETSETOPTB(m_opt,zoomFact,grid,"scaling (zoom) factor for each grid-zoom (default 0.333)"                      );
    QGETSETOPTB(m_opt,numPts,  grid,"grid-definition vector. Each index gives the number of points along that axis.");

    // DIRect options

    m_opt_DIRect.def("selDIRectopt",&selDIRectopt,"Select DIRect optimiser i > 0. If i=0 then return current DIRect optimizer     \n"
                                                  "(default 1) without modification. You can have arbitrarily many DIRect         \n"
                                                  "optimizers at any given time. Set rst = 1 to also reset.                       ",
                                                  py::arg("i")=0,py::arg("rst")=0);

    QGETSETOPTB(m_opt,maxits,  DIRect,"maximum cube divisions (default 1000)."      );
    QGETSETOPTB(m_opt,maxevals,DIRect,"maximum function evaluations (default 5000).");
    QGETSETOPTB(m_opt,eps,     DIRect,"epsilon factor (default 1e-4)."              );

    // Nelder-Mead options

    m_opt_NelderMead.def("selNelderMeadopt",&selNelderMeadopt,"Select NelderMead optimiser i > 0. If i=0 then return current NelderMead       \n"
                                                              "optimizer (default 1) without modification. You can have arbitrarily many      \n"
                                                              "NelderMead optimizers at any given time. Set rst = 1 to also reset.            ",
                                                              py::arg("i")=0,py::arg("rst")=0);

    QGETSETOPTB(m_opt,minf_max,NelderMead,"maximum f val (default -HUGE_VAL)."                  );
    QGETSETOPTB(m_opt,ftol_rel,NelderMead,"relative tolerance of function value (default 0)."   );
    QGETSETOPTB(m_opt,ftol_abs,NelderMead,"abolute tolerance of function value (default 0)."    );
    QGETSETOPTB(m_opt,xtol_rel,NelderMead,"relative tolerance of x value (default 0)."          );
    QGETSETOPTB(m_opt,xtol_abs,NelderMead,"abolute tolerance of x value (default 0)."           );
    QGETSETOPTB(m_opt,maxeval, NelderMead,"max number of f evaluations (default 1000)."         );
    QGETSETOPTB(m_opt,method,  NelderMead,"0 is subplex, 1 is original Nelder-Mead (default 0).");




    // Bayesian options

    m_opt_Bayesian.def("selBOopt",&selBayesianopt,"Select BO optimiser i > 0. If i=0 then return current BO optimizer (default 1) \n"
                                                  "without modification. You can have arbitrarily many BO optimizers at any given \n"
                                                  "time. Set rst = 1 to also reset.                                               ",
                                                  py::arg("i")=0,py::arg("rst")=0);

    QGETSETOPTB(m_opt,acq,    Bayesian,"Bayesian optimisation acquisition function:\n"
                                       "                                                                               \n"
                                       " 0 - MO (pure exploitation, mean only minimisation).                           \n"
                                       " 1 - EI (expected improvement - default).                                      \n"
                                       " 2 - PI (probability of improvement).                                          \n"
                                       " 3 - GP-UCB as per Brochu (recommended GP-UCB).*                               \n"
                                       " 4 - GP-UCB |D| finite as per Srinivas.                                        \n"
                                       " 5 - GP-UCB |D| infinite as per Srinivas.                                      \n"
                                       " 6 - GP-UCB p based on Brochu.                                                 \n"
                                       " 7 - GP-UCB p |D| finite based on Srinivas.                                    \n"
                                       " 8 - GP-UCB p |D| infinite based on Srinivas.                                  \n"
                                       " 9 - PE (variance-only maximisation).                                          \n"
                                       "10 - PEc (total variance-only, including constraints in variance maximisation).\n"
                                       "11 - Multi-strategy learning or user-defined.                                  \n"
                                       "12 - Thompson sampling.#                                                       \n"
                                       "13 - GP-UCB RKHS as per Srinivas.                                              \n"
                                       "14 - GP-UCB RKHS as Chowdhury.#                                                \n"
                                       "15 - GP-UCB RKHS as Bogunovic.~                                                \n"
                                       "16 - Thompson sampling (unity scaling on variance).                            \n"
                                       "17 - GP-UCB as per Kandasamy (multifidelity 2017).                             \n"
                                       "18 - Human will be prompted to input x (x=NaN).                                \n"
                                       "19 - HE (human-level exploitation beta = 0.01).                                \n"
                                       "20 - GP-UCB as per BO-Muse (single AI).  Typically combined with human prompt. \n"
                                       "21 - Random experiments (generated as per start-points).                       \n"
                                       "                                                                               \n"
                                       "* beta_n = 2.log((n^{2+dim/2}).(pi^2)/(3.delta))                               \n"
                                       "$ variance of model only.                                                      \n"
                                       "@ total variance of model and contraints.                                      \n"
                                       "# Chowdhury, On Kernelised Multi-Arm Bandits, Algorithm 2.                     \n"
                                       "~ Bogunovic, Misspecified GP Bandit Optim., Lemma 1.                           \n"
                                       "^ Intendid to be combined with human prompt.                                   ");
    QGETSETOPTB(m_opt,betafn, Bayesian,"user-defined beta for acq 11. You can make this a function with the vars:\n"
                                       "                                                                               \n"
                                       "- x_0  = iteration number.                                                     \n"
                                       "- x_1  = x dimension.                                                          \n"
                                       "- x_2  = delta.                                                                \n"
                                       "- x_3  = |D|.                                                                  \n"
                                       "- x_4  = a.                                                                    \n"
                                       "- x_5  = b.                                                                    \n"
                                       "- x_6  = r.                                                                    \n"
                                       "- x_7  = p.                                                                    \n"
                                       "- x_8  = batch size (inner).                                                   \n"
                                       "- x_9  = R.                                                                    \n"
                                       "- x_10 = B.                                                                    \n"
                                       "- x_11 = mig.                                                                  \n"
                                       "- x_12 = RKHS norm.                                                            \n"
                                       "- x_13 = kappa0.                                                               \n"
                                       "- x_14 = lengthscale.                                                          \n"
                                       "- x_15 = sigma.                                                                \n"
                                       "- x_16 = ell1.                                                                 \n"
                                       "- x_17 = pi.                                                                   \n"
                                       "                                                                               \n"
                                       "You can also use [ [ f1 ] [ f2 ] ... ], where f1,f2,... define acquisition     \n"
                                       "function (see acq variable). This will generate multiple recommendations in a  \n"
                                       "single iteration, one for each of the acq f1,f2,... given.                     ");
    QGETSETOPTB(m_opt,PIscale,Bayesian,"PI scaling: set 0 for standard operation, 1 to scale aquisition function by    \n"
                                       "the PI (probability of improvement) acquisition function.                      ");

    QGETSETOPTB(m_opt,sigmuseparate,Bayesian,"posterior separation:\n"
                                             "                                                                               \n"
                                             "0 - use same model for posterior mean and variance (default).                  \n"
                                             "1 - use separate models. This is required e.g. for hallucinated samples in.    \n"
                                             "    some multi-recommendation methods, where the variance updates after each   \n"
                                             "    individual recommendation but the posterior mean updates only after the    \n"
                                             "    whole batch.                                                               ");

    QGETSETOPTB(m_opt,startpoints,Bayesian,"number of initial (random, uniform) seeds points. Use -1 (default) for d+1.");
    QGETSETOPTB(m_opt,totiters,   Bayesian,"number of iterations. Use 0 for unlimited, -1 (default) for 15d, -2 for\n"
                                           "frequentist mode (stop when min_x err(x) < err).");
    QGETSETOPTB(m_opt,err,        Bayesian,"when totiters=-2, the frequentist stopping condition is min_x err(x) < err.");
    QGETSETOPTB(m_opt,minstdev,   Bayesian,"added to the posterior variance in acquisition function (default 0).");
    QGETSETOPTB(m_opt,ztol,       Bayesian,"zero tolerance factor.");

    QGETSETOPTB(m_opt,startseed,Bayesian,"seed for RNG immediately prior to generating random seeds. -1 for no seed,\n"
                                         "-2 to seed with time. If >=0 then this is incremented on each use (default 42).");
    QGETSETOPTB(m_opt,algseed,  Bayesian,"seed for RNG immediately prior to the main algorithm loop. -1 for no seed,\n"
                                         "-2 to seed with time. If >=0 then this is incremented on each use (default 42)");


    QGETSETOPT(m_opt,delta,Bayesian,gpucb,"delta factor used in GP-UCB (default 0.1).");
    QGETSETOPT(m_opt,nu,   Bayesian,gpucb,"nu factor Srinivas GP-UCB (default 0.2, see Srivinas)");
    QGETSETOPT(m_opt,modD, Bayesian,gpucb,"|D| (grid siez) for GP-UCB finite (deflt -1: size of grid or 10)");
    QGETSETOPT(m_opt,a,    Bayesian,gpucb,"a constant for Srinivas |D|-infinite gpUCB (default 1)");
    QGETSETOPT(m_opt,b,    Bayesian,gpucb,"b constant for Srinivas |D|-infinite gpUCB (default 1)");
    QGETSETOPT(m_opt,r,    Bayesian,gpucb,"r constant for Srinivas |D|-infinite gpUCB (default 1)");
    QGETSETOPT(m_opt,p,    Bayesian,gpucb,"p value for GP-UCB p variants (default 2)");
    QGETSETOPT(m_opt,R,    Bayesian,gpucb,"R constant for acquisition functions 12,13,14 (default 1)");
    QGETSETOPT(m_opt,B,    Bayesian,gpucb,"B constant for acquisition functions 12,13,14 (default 1)");


    QGETSETOPT(m_opt,zeta, Bayesian,ei,"zeta factor in EI (default 0). 0.01 works ok).");


    QGETSETOPT(m_opt,TSmode,     Bayesian,ts,"Thompson sampling mode:\n"
                                             "\n"
                                             "1 - use regular grid or random samples.\n"
                                             "3 - use JIT sampling (default).");
    QGETSETOPT(m_opt,TSNsamp,    Bayesian,ts,"number of samples for sample mode 1. Set >0 for fixed grid, <0 for random grid,\n"
                                             "0 (default) for random grid of 10.j.dim^2 random points (here j is the number  \n"
                                             "of prior samples in model).");
    QGETSETOPT(m_opt,TSsampType, Bayesian,ts,"sample type for sample mode 1:\n"
                                             "                                                                               \n"
                                             "0   - unbounded draw (default).                                                \n"
                                             "1   - positive (definite/symm) draw by clip max(0,y).                          \n"
                                             "2   - positive (definite/symm) draw by flip |y|.                               \n"
                                             "3   - negative (definite/symm) draw by clip min(0,y).                          \n"
                                             "4   - negative (definite/symm) draw by flip -|-y|.                             \n"
                                             "5   - unbounded (symmetric) draw.                                              \n"
                                             "1x  - As above, but force alpha >= 0 after sample.                             \n"
                                             "2x  - As above, but force alpha <= 0 after sample.                             \n"
                                             "1xx - As above, but square the function afterwards. Existing observations      \n"
                                             "      treated as observations on the function before squaring.                 \n"
                                             "2xx - Like 1xx,  but existing observations treated as observations on the      \n"
                                             "      squared function.                                                        \n"
                                             "3xx - Like  2xx, but  return the  function *before* squaring occurs.           ");
    QGETSETOPT(m_opt,TSxsampType,Bayesian,ts,"x sample type for sample mode 1:\n"
                                             "                                                                               \n"
                                             "0 - \"True\" pseudo-random (default).                                          \n"
                                             "1 - pre-defined sequence, sequentially generated.                              \n"
                                             "2 - pre-defined sequence, same every time.                                     \n"
                                             "3 - grid of Nsamp^dim samples                                                  ");
    QGETSETOPT(m_opt,sigma_cut,  Bayesian,ts,"variance scale for JIT TS (default 0.1).");


    QGETSETOPT(m_opt,numcgt,   Bayesian,cgt,"number of constraints enforced (default 0).");
    QGETSETOPT(m_opt,cgtmethod,Bayesian,cgt,"constraint method:\n"
                                            "                                                                               \n"
                                            "0 - calculate P(c(x))>=0 and scale acquisition function by this (default).     \n"
                                            "1 - optimise f(x).ind(c(x)>=0), so that the mean/variance of c are built into  \n"
                                            "    the posterior mean/variance before calculating acquisition function.       ");
    QGETSETOPT(m_opt,cgtmargin,Bayesian,cgt,"safety margin for enforcing inequality constraints in the acq fn (default 0.1)");


    QGETSETOPT(m_opt,moodim,Bayesian,moo,"number of objectives (default 1, single-objective).");

    m_opt_Bayesian_moo.def("setimp",&boSetimpmeas,"For BO, set improvement measure (IMP). This is required for multi-objective BO,\n"
                                                  "and defines how improvement is measured (vector to scalar). Essentially, mean  \n"
                                                  "imp(mean,var). Processing is done with the IMP given. Note that the acquisition\n"
                                                  "function defined by -gbH still be applied after this (to do passthrough use    \n"
                                                  "acquisition function 0. Some IMPs have a concept of posterior variance, some   \n"
                                                  "don't. For EHI use acquisition function 0, for SVM-type mono-surrogate use for \n"
                                                  "for example acquition function 3.                                              ",
                                                  py::arg("j"));


    QGETSETOPT(m_opt,numfids,   Bayesian,fid,"number of fidelity levels per axis (default 0, no fidelity variables)."       );
    QGETSETOPT(m_opt,dimfid,    Bayesian,fid,"number of fidelity axis (default 1, but meaningless unless numfids>0)."       );
    QGETSETOPT(m_opt,fidbudget, Bayesian,fid,"fidelity budget (default -1, unlimited)."                                     );
    QGETSETOPT(m_opt,fidpenalty,Bayesian,fid,"fidelity penalty function f(z), where z is the fidelity vector."              );
    QGETSETOPT(m_opt,fidvar,    Bayesian,fid,"fidelity additive variance function n(z), added to measurement vari (dflt 0).");
    QGETSETOPT(m_opt,fidmode,   Bayesian,fid,"");
    QGETSETOPT(m_opt,fidover,   Bayesian,fid,"optional fidelity overwrite:\n"
                                            "                                                                               \n"
                                             "0 - use fidelity generated by algorithm.                                      \n"
                                             "2 - randomly select fidelity <= recommended fidelity.                         ");


    QGETSETOPT(m_opt,intrinbatch,      Bayesian,mr,"intrinsic batch size (default 1)."                                             );
    QGETSETOPT(m_opt,intrinbatchmethod,Bayesian,mr,"intrinsic batch recommendation method:"
                                                   "                                                                               \n"
                                                   "0 - use max mean, det(covar)^(1/(2*ibs)) (default).                            \n"
                                                   "1 - use ave mean, det(covar)^(1/(2*ibs)).                                      \n"
                                                   "2 - use min mean, det(covar)^(1/(2*ibs)).                                      \n"
                                                   "3 - use max mean, sqrt(ibs/Tr(inv(covar))).                                    \n"
                                                   "4 - use ave mean, sqrt(ibs/Tr(inv(covar))).                                    \n"
                                                   "5 - use min mean, sqrt(ibs/Tr(inv(covar))).                                    ");


    m_opt_Bayesian_tl.def("setkxfersrc",&boSetkernapproxsource,"kernel transfer learning source ML.",py::arg("j"));

    QGETSETOPT(m_opt,tranmeth,Bayesian,tl,"Transfer learning data treatment:\n"
                                          "                                                                               \n"
                                          "0 - assume data from target model (default).                                   \n"
                                          "1 - use env-GP as per Joy1/Shi21.                                              \n"
                                          "2 - use diff-GP as per Shi21.                                                  ");
    QGETSETOPT(m_opt,alpha0,  Bayesian,tl,"alpha0 value for env-GP.");
    QGETSETOPT(m_opt,beta0,   Bayesian,tl,"beta0 value for env-GP.");
    QGETSETOPT(m_opt,kxfnum,  Bayesian,tl,"kernel transfer learning method:\n"
                                          "                                                                               \n"
                                          "800 - trivial K(x,y) = Kj(x,y).                                                \n"
                                          "801 - m-norm (free kernel) transfer (default).                                 \n"
                                          "802 - moment (Der and Lee) transfer.                                           \n"
                                          "804 - K-learn transfer.                                                        \n"
                                          "805 - K2-learn transfer.                                                       \n"
                                          "806 - Multi-layer transfer.                                                    ");
    QGETSETOPT(m_opt,kxfnorm, Bayesian,tl,"kernel transfer normalization:\n"
                                          "                                                                               \n"
                                          "0 = no normalization.                                                          \n"
                                          "0 = use normalization (default).                                               ");


    QGETSETOPT(m_opt,modelname,     Bayesian,vis,"model basename when plotting/logging (default smbomodel)"            );
    QGETSETOPT(m_opt,modeloutformat,Bayesian,vis,"format for plotting posterior (0 terminal, 1 ps, 2 pdf (default))"   );
    QGETSETOPT(m_opt,plotfreq,      Bayesian,vis,"plotting frequency for posterior (0 none (default), -1 only on exit)");
    QGETSETOPT(m_opt,modelbaseline, Bayesian,vis,"baseline function f(x) for posterior plots (empty for none)"         );

    QGETSETOPT(m_opt,tunemu,     Bayesian,tune,"Tuning for objective model (0 none, 1 max-log-like (default), 2 LOO, 3 recall." );
    QGETSETOPT(m_opt,tunecgt,    Bayesian,tune,"Tuning for constraint model (0 none, 1 max-log-like (default), 2 LOO, 3 recall.");
    QGETSETOPT(m_opt,tunesigma,  Bayesian,tune,"Tuning for noise model (0 none, 1 max-log-like (default), 2 LOO, 3 recall."     );
    QGETSETOPT(m_opt,tunesrcmod, Bayesian,tune,"Tuning for source model (0 none, 1 max-log-like (default), 2 LOO, 3 recall."    );
    QGETSETOPT(m_opt,tunediffmod,Bayesian,tune,"Tuning for difference model (0 none, 1 max-log-like (default), 2 LOO, 3 recall.");
//    QGETSETOPT(m_opt,tuneaugxmod,Bayesian,tune,"Tuning for x augmentation (side-channel) model (0 none, 1 max-log-likelihood (default), 2 LOO, 3 recall.");


    m_opt_Bayesian_model.def("setgridsrc",&boSetgridsource, "For BO, set grid source.",py::arg("j"));
    m_opt_Bayesian_model.def("selmu",    &selmlmuapprox,    "select objective model for BO i to use like any ML (see also selml).",py::arg("i")=0,py::arg("k")=0);
    m_opt_Bayesian_model.def("selcgt",   &selmlcgtapprox,   "select constraint model for BO i to use like any ML (see also selml).",py::arg("i")=0,py::arg("k")=0);
//    m_opt_Bayesian_model.def("selaug",   &selmlaugxapprox,  "select x augmentation (side-channel) model for BO i to use like any ML (see also selml).",py::arg("i")=0,py::arg("k")=0);
    m_opt_Bayesian_model.def("selsigma", &selmlsigmaapprox, "select noise model for BO i to use like any ML (see also selml).",py::arg("i")=0);
    m_opt_Bayesian_model.def("selsrc",   &selmlsrcmodel,    "select source model for BO i to use like any ML (see also selml).",py::arg("i")=0);
    m_opt_Bayesian_model.def("seldiff",  &selmldiffmodel,   "select src->destination difference model (transfer learning) for BO i to use   \n"
                                                            "like any ML (see also selml).",py::arg("i")=0);

    m_opt_Bayesian_model.def("selmu_prior",    &selmlmuapprox_prior,    "select prior objective model for BO i to use like any ML (see also selml).",py::arg("i")=0);
    m_opt_Bayesian_model.def("selcgt_prior",   &selmlcgtapprox_prior,   "select prior constraint model for BO i to use like any ML (see also selml).",py::arg("i")=0);
//    m_opt_Bayesian_model.def("selaug_prior",   &selmlaugxapprox_prior,  "select prior x augmentation (side-channel) model for BO i to use like any ML (see also selml).",py::arg("i")=0);
    m_opt_Bayesian_model.def("selsigma_prior", &selmlsigmaapprox_prior, "select prior noise model for BO i to use like any ML (see also selml).",py::arg("i")=0);
    m_opt_Bayesian_model.def("selsrc_prior",   &selmlsrcmodel_prior,    "select prior source model for BO i to use like any ML (see also selml).",py::arg("i")=0);
    m_opt_Bayesian_model.def("seldiff_prior",  &selmldiffmodel_prior,   "select prior src->destination difference model (transfer learning) for BO i to \n"
                                                                        "use like any ML (see also selml).",py::arg("i")=0);


















    // ---------------------------
    // ---------------------------
    // ---------------------------
    // ---------------------------

    auto m_ml_kern = m_ml.def_submodule("kern","Kernel Options for Model.");

    auto m_ml_kern_UU  = m_ml_kern.def_submodule("UU", "Output kernel Options for Model."        );
    auto m_ml_kern_RFF = m_ml_kern.def_submodule("RFF","RFF similarity kernel Options for Model.");

    QGETSETKERQD(m_ml_kern,cType,   setType, "type", "settype", "kernel type, for example (z = <x,x'>, d=||x-x'||_2):\n"
                                                                "                                                                               \n"
                                                                "   0: Constant:                 Kq(x,x') = rq_1                                \n"
                                                                "   1: Linear:                   Kq(x,x') = z/(rq_0.rq_0)                       \n"
                                                                "   2: Polynomial:               Kq(x,x') = ( rq_1 + z/(rq_0.rq_0) )^iq_0       \n"
                                                                "   3: Gaussian (RBF):           Kq(x,x') = exp(-d/(2.rq_0.rq_0)-rq_1)          \n"
                                                                "   4: Laplacian:                Kq(x,x') = exp(-sqrt(d)/rq_0-rq_1)             \n"
                                                                "   5: Polynoise:            Kq(x,x') = exp(-sqrt(d)^rq_1/(rq_1*rq_0^rq_1)-rq_2)\n"
                                                                "   6: ANOVA:                    Kq(x,x') = sum_k exp(-rq_4*((x_k/rq_0)^rq_1-...\n"
                                                                "                                                   (x'_k/rq_0)^rq_1)^rq_2)^rq_3\n"
                                                                "   7: Sigmoid:#                 Kq(x,x') = tanh( z/(rq_0.rq_0) + rq_1 )        \n"
                                                                "   8: Rational quadratic:       Kq(x,x') = ( 1 + d/(2*rq_0*rq_0*rq_1) )^(-rq_1)\n"
                                                                "   9: Multiquadratic:%          Kq(x,x') = sqrt( d/(rq_0.rq_0) + rq_1^2 )      \n"
                                                                "  10: Inverse multiquadric:     Kq(x,x') = 1/sqrt( d/(rq_0.rq_0) + rq_1^2 )    \n"
                                                                "  11: Circular:*                Kq(x,x') = 2/pi * arccos(-sqrt(d)/rq_0) - ...  \n"
                                                                "                                       2/pi * sqrt(d)/rq_0 * sqrt(1 - d/rq_0^2)\n"
                                                                "  12: Sperical:+                Kq(x,x') = 1 - 3/2 * sqrt(d)/rq_0 + 1/2 * ...  \n"
                                                                "                                                               sqrt(d)^3/rq_0^3\n"
                                                                "  13: Wave:                     Kq(x,x') = sinc(sqrt(d)/rq_0)                  \n"
                                                                "  14: Power:                    Kq(x,x') = -sqrt(d/(rq_0.rq_0))^rq_1           \n"
                                                                "  15: Log:#                     Kq(x,x') = -log(sqrt(d/(rq_0.rq_0))^rq_1 + 1)  \n"
                                                                "  16: Spline:                   Kq(x,x') = prod_k ( 1 + (x_k/rq_0).(x'_k/rq_0) \n"
                                                                "                         ... + (x_k/rq_0).(x'_k/rq_0).min(x_k/rq_0,x'_k/rq_0) -\n"
                                                                "                       ... ((x_k/rq_0+x'_k/rq_0).min(x_k/rq_0,x'_k/rq_0)^2)/2 +\n"
                                                                "                                                (min(x_k/rq_0,x'_k/rq_0)^3)/3 )\n"
                                                                "  17: B-Spline:                Kq(x,x') = sum_k B_(2iq_0+1)(x_k/rq_0-x'_k/rq_0)\n"
                                                                "  19: Cauchy:                   Kq(x,x') = 1/(1+(d/(rq_0.rq_0)))               \n"
                                                                "  20: Chi-square:             Kq(x,x') = 1 - sum_k (2((x_k/rq_0).(x'_k/rq_0)))/\n"
                                                                "                                                       ... (x_k/rq_0+x'_k/rq_0)\n"
                                                                "  21: Histogram:                Kq(x,x') = sum_k min(x_k/rq_0,x'_k/rq_0)       \n"
                                                                "  22: Generalised histogram:    Kq(x,x') = sum_k min(|x_k/rq_0|^rq_1,|x'_k/... \n"
                                                                "                                                                    rq_0|^rq_2)\n"
                                                                "  23: Generalised T-student:    Kq(x,x') = 1/(1+(sqrt(d)/rq_0)^rq_1)           \n"
                                                                "  24: Vovk's real:    Kq(x,x') = (1-((z/(rq_0.rq_0))^iq_0))/(1-(z/(rq_0.rq_0)))\n"
                                                                "  25: Weak fourier:             Kq(x,x') = pi.cosh(pi-(sqrt(d)/rq_0))          \n"
                                                                "  26: Thin spline 1:            Kq(x,x') = ((d/rq_0)^(rq_1+0.5))               \n"
                                                                "  27: Thin spline 2:            Kq(x,x') = ((d/rq_0)^rq_1).ln(sqrt(d/rq_0))    \n"
                                                                "  28: Generic:                  Kq(x,x') = (user defined by hyperparam rq_10)  \n"
                                                                "                          args: a_0: m                                         \n"
                                                                "                                a_1: <x,x'>/(rq_0.rq_0)                        \n"
                                                                "                                a_2: <x',x>/(rq_0.rq_0)                        \n"
                                                                "                                a_3: ||x-x'||^2/(rq_0.rq_0)                    \n"
                                                                "                                a_4: [ rq_0 rq_1 ... rq_8 ]                    \n"
                                                                "                                a_5: [ ||x||^2/(rq_0.rq_0) ||x'||^2/... ... ]  \n"
                                                                "                                a_6: [ x/rq_0 x'/rq_0 ... ]                    \n"
                                                                "                                a_7: [ Ki0 Ki1 ... : rq_9 = [ i0 i1 ... ] ]    \n"
                                                                "  29: Arc-cosine:               Kq(x,x') = (1/pi) (rq_0.sqrt(a))^iq_0 ...      \n"
                                                                "                            (rq_0.sqrt(b))^iq_0 Jn(arccos(z/(sqrt(a).sqrt(b))))\n"
                                                                "  30: Chaotic logistic:         Kq(x,x') = Kn(x,x') = <phi_{sigma,n}(x/rq_0),..\n"
                                                                "                                                        phi_{sigma,n}(x'/rq_0)>\n"
                                                                "  31: Summed chaotic logistic:  Kq(x,x') = sum_{0,n} Kn(x,x')                  \n"
                                                                "  32: Diagonal:                 Kq(x,x') = rq_1 if i == j, 0 otherwise         \n"
                                                                "  33: Uniform:               Kq(x,x') = 1/(2.rq_0) ( 1 if real(sqrt(d)) < rq_0,\n"
                                                                "                                                                  0 otherwise )\n"
                                                                "  34: Triangular:     Kq(x,x') = (1-sqrt(d)/rq_0)/rq_0 if real(sqrt(d)) < rq_0,\n"
                                                                "                                                                  0 otherwise )\n"
                                                                "  35: Even-integer Matern:      Kq(x,x') = ((2^(1-iq_0))/gamma(iq_0)).((sqrt...\n"
                                                                "                 (2.iq_0).sqrt(d)/rq_0)^iq_0).K_rq_1(sqrt(2.iq_0).sqrt(d)/rq_0)\n"
                                                                "  36: Weiner:                   Kq(x,x') = prod_i min(x_i/rq_0,x'_i/rq_0)      \n"
                                                                "  37: Half-integer Matern:  Kq(x,x') = exp(-(sqrt(2.(iq_0+1/2))/rq_0).sqrt(d)).\n"
                                                                "                       (gamma(iq_0+1)/gamma((2.iq_0)+1)) . sum_{i=0,1,...,iq_0}\n"
                                                                " ( ((iq_0+1)!/(i!.(iq_0-i)!)) . pow((sqrt(8.(iq_0+1/2))/rq_0).sqrt(d),iq_0-i) )\n"
                                                                "  38: 1/2-Matern:               Kq(x,x') = exp(-sqrt(d)/rq_0)                  \n"
                                                                "  39: 3/2-Matern:               Kq(x,x') = (1+((sqrt(3)/rq_0).sqrt(d))) . ...  \n"
                                                                "                                                   exp(-(sqrt(3)/rq_0).sqrt(d))\n"
                                                                "  40: 5/2-Matern:               Kq(x,x') = (1+((sqrt(5)/rq_0).sqrt(d))+((5/(3. \n"
                                                                "                                 rq_0*rq_0))*d)) . exp(-(sqrt(5)/rq_0).sqrt(d))\n"
                                                                "  41: RBF-rescale:              Kq(x,x') = exp(log(z)/(2.rq_0.rq_0))           \n"
                                                                "  42: Inverse Gudermannian:     Kq(x,x') = igd(z/(rq_0.rq_0))                  \n"
                                                                "  43: Log ratio:            Kq(x,x') = log((1+z/(rq_0.rq_0))/(1-z/(rq_0.rq_0)))\n"
                                                                "  44: Exponential:***           Kq(x,x') = exp(z/(rq_0.rq_0)-rq_1)             \n"
                                                                "  45: Hyperbolic sine:          Kq(x,x') = sinh(z/(rq_0.rq_0))                 \n"
                                                                "  46: Hyperbolic cosine:        Kq(x,x') = cosh(z/(rq_0.rq_0))                 \n"
                                                                "  47: Sinc Kernel (Tobar):      Kq(x,x') = sinc(sqrt(d)/rq_0).cos(2*pi*sqrt(d)/\n"
                                                                "                                                               ... (rq_0.rq_1))\n"
                                                                "  48: LUT kernel:               Kq(x,x') = rq_1((int) x, (int) x') if rq_1 is a\n"
                                                                "                              matrix, otherwise (rq_1 if x != x', 1 if x == x')\n"
                                                                "  49: Gaussian Harmonic:  Kq(x,x')=(1-rq_2)/(1-rq_2.exp(-d/(2.rq_0.rq_0)-rq_1))\n"
                                                                "  50: Alt arc-cosine:           Kq(x,x') = pi - arccos(z/rq_0.rq_0)            \n"
                                                                "  51: Vovk-like:                Kq(x,x') = 1/(2-(z/rq_0.rq_0))                 \n"
                                                                "  52: Radius (see Bock):        Kq(x,x') = ((a.b)^(1/m))/(rq_0.rq_0) (just use \n"
                                                                "                             a normalised linear kernel for the angular kernel)\n"
                                                                "  53: Radius (see Bock):        Kq(x,x') = (((1-(1-a^rq_1)^rq_2).(1-(1-b^rq_1)^\n"
                                                                "                              rq_2))^(1/m))/(rq_0.rq_0) (just use a normalised \n"
                                                                "                                          linear kernel for the angular kernel)\n"
                                                                "1003: Gaussian dense deriv:%    Kq(x,x') = (prod_k d/dx_k) exp(-d/(2.rq_0.rq_0)\n"
                                                                "                                                                         -rq_1)\n"
                                                                "1038: 1/2-Matern dense deriv:%@ Kq(x,x') = (prod_k d/dx_k) exp(-sqrt(d)/rq_0)  \n"
                                                                "1039: 3/2-Matern dense deriv:%@ Kq(x,x') = (prod_k d/dx_k) (1+((sqrt(3)/rq_0). \n"
                                                                "                                       sqrt(d))) . exp(-(sqrt(3)/rq_0).sqrt(d))\n"
                                                                "2003: Gaussian dense integ:%    Kq(x,x') = (prod_k int_{x_k=0}^infty dx_k) ... \n"
                                                                "                                                     exp(-d/(2.rq_0.rq_0)-rq_1)\n"
                                                                "2038: 1/2-Matern dense integ:%@ Kq(x,x') = (prod_k int_{x_k=0}^infty dx_k) ... \n"
                                                                "                                                             exp(-sqrt(d)/rq_0)\n"
                                                                "2039: 3/2-Matern dense integ:%@ Kq(x,x') = (prod_k int_{x_k=0}^infty dx_k) ... \n"
                                                                "                    (1+((sqrt(3)/rq_0).sqrt(d))) . exp(-(sqrt(3)/rq_0).sqrt(d))\n"
                                                                "                                                                               \n"
                                                                "Notes: % non-positive-definite kernel                                          \n"
                                                                "       # conditionally positive definite kernel                                \n"
                                                                "       * kernel is only positive definite in R^2                               \n"
                                                                "       + kernel is only positive definite in R^3                               \n"
                                                                "       @ kernel is only defined if set to operate product-wise                 ");
    QGETSETKERD(m_ml_kern, getTypes,setTypes,"types","settypes","Kernel types [ t0, t1, ... ] (by default the overall kernel is the weighted sum\n"
                                                                "K(x,x') = sum_i wi Ki(x,x') of these, but you can change this (see (mul)split).");

    QGETSETKERQD(m_ml_kern,getRealConstZero,setRealConstZero,"l","setl","lengthscale rq_0=l");
    QGETSETKERQD(m_ml_kern,getIntConstZero, setIntConstZero, "d","setd","order iq_0=d"      );

    QGETSETKERQD(m_ml_kern,cWeight,       setWeight,       "weight","setweight","weight wq"                                 );
    QGETSETKERQD(m_ml_kern,cRealConstants,setRealConstants,"hyperR","sethyperR","hyper-parameters [rq_0, rq_1, ...]"        );
    QGETSETKERQD(m_ml_kern,cIntConstants, setIntConstants, "hyperZ","sethyperZ","integer hyper-parameters [iq_0, iq_1, ...]");

    QGETSETKERD(m_ml_kern,getHyper,       setHyper,        "hyperWRs","sethyperWRs","weights, hyper-params [ [ w0,r0_0,r0_1, ...], [ w1,r1_0,r1_1, ...], ...]");
    QGETSETKERD(m_ml_kern,getIntConstants,setIntConstantss,"hyperZs", "sethyperZs", "integer hyper-params [ [ i0_0,i0_1, ... ], [ i1_0,i1_1, ... ], ... ]");

    // ---------------------------

    auto m_ml_kern_obscure = m_ml_kern.def_submodule("obscure","More Obscure Kernel Options for Model.");

    auto m_ml_kern_obscure_UU  = m_ml_kern_obscure.def_submodule("UU", "More obscure output kernel Options for Model."        );
    auto m_ml_kern_obscure_RFF = m_ml_kern_obscure.def_submodule("RFF","More obscure RFF similarity kernel Options for Model.");

    QGETSETKERD(m_ml_kern_obscure,isFullNorm,setFullNorm,"fullnorm","setfullnorm","overall normalization (0 normal, 1 K(x,y) = K(x,y)/sqrt(K(x,x).K(y,y)))");
    QGETSETKERD(m_ml_kern_obscure,isSymmSet,setSymmSet,  "symm",    "setsymm",    "similarity 2-kernel symmetrization (0 normal, 1 K([x1~x2],[x3~x4]) = sqrt(K(x1,x3).K(x1,x4).K(x2,x3).K(x2.x4)))");
    QGETSETKERD(m_ml_kern_obscure,isProd,   setProd,     "prod",    "setprod",    "product-wise kernel (0 normal, 1 K(x,y) = prod_i K(x_i,y_i))");
    QGETSETKERD(m_ml_kern_obscure,isAltDiff,setAltDiff,  "altdiff", "setaltdiff", "metric:\n""0:   ||x-y||_2^2    -> ||x||_m^m + ||x'||_m^m + ... - m.<<x,x',...>>_m\n""1:   ||x-y||_2^2    -> ||x||_2^2 + ||x'||_2^2 + ... - 2.<<x,x',...>>_m (default)\n""2:   2*(||x-y||_2^2 -> ||x||_2^2 + ||x'||_2^2 + ... - (1/m).(sum_{ij} <xi,xj>))\n""     (the RBF has additional scaling as per paper - see Kbase)\n""5:   ||x-y||_2^2    -> ||x-x'||_2^2 + ||x''-x'''||_2^2 + ...\n""                     = ||x||_2^2 + ||x'||_x^2 + ... - 2<x,x'> - 2<x'',x'''> - ...\n""                     = ||x||_2^2 + ||x'||_x^2 + ... - 2 sum_{i=0,2,...} <x_i,x_{i+1}>\n""103: K(...) -> 1/2^{m-1} sum_{s = [ +-1 +-1 ... ] in R^m : |i:si=+1| + |i:si=-1| in 4Z_+} K(||sum_i s_i x_i ||_2^2)\n""104: K(...) -> 1/m!      sum_{s = [ +-1 +-1 ... ] in R^m : |i:si=+1| = |i:si=-1|         } K(||sum_i s_i x_i ||_2^2)\n""203: like 103, but kernel expansion occurs over first kernel in chain only\n""204: like 104, but kernel expansion occurs over first kernel in chain only\n""300: true moment-kernel expension to 2-kernels\n");
    QGETSETKERD(m_ml_kern_obscure,rankType, setrankType, "ranktype","setranktype","rank type (0: normal phi(x,x') = phi(x)-phi(x'), 1: phi(x,x') = phi(x)+phi(x'), 2: phi(x,x') = phi(x) otimes phi(x') - phi(x') otimes phi(x), 3: phi(x,x') = phi(x) otimes phi(x') - phi(x') otimes phi(x))");

    QGETSETKERD(m_ml_kern_obscure,denseZeroPoint,setdenseZeroPoint,"iz","setiz","zero point for dense integration");

    QGETSETKERD(m_ml_kern_obscure,getlinGradOrd, setlinGradOrd, "linGradOrd", "setlinGradOrd", "Order of linear gradient constraints (see Jidling)"      );
    QGETSETKERD(m_ml_kern_obscure,getlinGradScal,setlinGradScal,"linGradScal","setlinGradScal","Matrix part of linear gradient constraints (see Jidling)");

    QGETSETKERD(m_ml_kern_obscure,getlinParity,    setlinParity,    "linParity",    "setlinParity",    "Linear parity constraint"       );
    QGETSETKERD(m_ml_kern_obscure,getlinParityOrig,setlinParityOrig,"linParityOrig","setlinParityOrig","Linear parity constraint origin");

    QGETSETKERD(m_ml_kern_obscure,numSamples,        setnumSamples,        "sN","setsN","number of samples if interpretting functions as distributions");
    QGETSETKERD(m_ml_kern_obscure,sampleDistribution,setSampleDistribution,"sD","setsD","distribution type if interpretting functions as distributions");
    QGETSETKERD(m_ml_kern_obscure,sampleIndices,     setSampleIndices,     "sI","setsI","indices on which to interpret functions as distributions"     );

    QGETSETKERD(m_ml_kern_obscure,getChained, setChained, "chain",   "setchain",   "controls chaining ([0, 0, ...] normal, e.g. [ 0, 1, 1, 0, ... ] K(...) = K_0(...) + K_3(K_2(K_1(...))) + K_4(...) + ...)");
    QGETSETKERD(m_ml_kern_obscure,getSplit,   setSplit,   "split",   "setsplit",   "controls splitting (termwise 0,1,2... see code)"                                                                         );
    QGETSETKERD(m_ml_kern_obscure,getMulSplit,setMulSplit,"mulsplit","setmulsplit","controls multiplicative splitting (termwise 0,1,2... see code)"                                                          );

    QGETSETKERD(m_ml_kern_obscure,getIsNormalised,setIsNormalised,"norms","setnorms","individual normalizations [n_0, n_1, ...] (n_q=0 normal, n_q=1 Kq(x,y) = Kq(x,y)/sqrt(Kq(x,x).Kq(y,y)))");
    QGETSETKERD(m_ml_kern_obscure,getIsMagTerm,   setIsMagTerm,   "mags", "setmags", "individual magnitudization [m_0, m_1, ...] (n_q=0 normal, n_q=1 Kq(x,y) = Kq(x,x).Kq(y,y))"             );

    QGETSETKERD(m_ml_kern_obscure,getRealOverwrites,setRealOverwrites,"hyperRovrs","sethyperRovrs","hyper-parameters substitutions [i_0, i_1, ...]"        );
    QGETSETKERD(m_ml_kern_obscure,getIntOverwrites, setIntOverwrites, "hyperZovrs","SethyperZovrs","integer hyper-parameters substitutions [i_0, i_1, ...]");

    QGETSETKERQD(m_ml_kern_obscure,isNormalised,setisNormalised,"norm","setnorm","individual normalization (0 normal, 1 Kq(x,y) = Kq(x,y)/sqrt(Kq(x,x).Kq(y,y)))");
    QGETSETKERQD(m_ml_kern_obscure,isMagTerm,   setisMagTerm,   "mag", "setmag", "individual magnitudization (termwise 0 normal, 1 Kq(x,y) = Kq(x,x).Kq(y,y))"   );

    QGETSETKERQD(m_ml_kern_obscure,cRealOverwrite,setRealOverwrite,"hyperRovr","sethyperRovr","hyper-parameters substitution vector"        );
    QGETSETKERQD(m_ml_kern_obscure,cIntOverwrite, setIntOverwrite, "hyperZovr","sethyperZovr","integer hyper-parameters substitution vector");

    QGETSETKERD(m_ml_kern,getHyperLB,       setHyperLB,        "hyperWRLBs","sethyperWRLBs","nominal lower bounds for weights and hyper-parameters"   );
    QGETSETKERD(m_ml_kern,getIntConstantsLB,setIntConstantssLB,"hyperZLBs", "sethyperZLBs", "nominal lower bounds for integer hyper-parameters vector");

    QGETSETKERQD(m_ml_kern,cWeightLB,       setWeightLB,       "weightLB","setweightLB","nominal lower bound for weight"                   );
    QGETSETKERQD(m_ml_kern,cRealConstantsLB,setRealConstantsLB,"hyperRLB","sethyperRLB","nominal lower bounds for hyper-parameters"        );
    QGETSETKERQD(m_ml_kern,cIntConstantsLB, setIntConstantsLB, "hyperZLB","sethyperZLB","nominal lower bounds for integer hyper-parameters");

    QGETSETKERQD(m_ml_kern,getRealConstZeroLB,setRealConstZeroLB,"lLB","setlLB","nominal lower bound for lengthscale");
    QGETSETKERQD(m_ml_kern,getIntConstZeroLB, setIntConstZeroLB, "dLB","setdLB","nominal lower bounds for order"     );

    QGETSETKERD(m_ml_kern,getHyperUB,       setHyperUB,        "hyperWRUBs","sethyperWRUBs","nominal upper bounds for weights and hyper-parameters"   );
    QGETSETKERD(m_ml_kern,getIntConstantsUB,setIntConstantssUB,"hyperZUBs", "sethyperZUBs", "nominal upper bounds for integer hyper-parameters vector");

    QGETSETKERQD(m_ml_kern,cWeightUB,       setWeightUB,       "weightUB","setweightUB","nominal upper bound for weight"                   );
    QGETSETKERQD(m_ml_kern,cRealConstantsUB,setRealConstantsUB,"hyperRUB","sethyperRUB","nominal upper bounds for hyper-parameters"        );
    QGETSETKERQD(m_ml_kern,cIntConstantsUB, setIntConstantsUB, "hyperZUB","sethyperZUB","nominal upper bounds for integer hyper-parameters");

    QGETSETKERQD(m_ml_kern,getRealConstZeroUB,setRealConstZeroUB,"lUB","setlUB","nominal upper bound for lengthscale");
    QGETSETKERQD(m_ml_kern,getIntConstZeroUB, setIntConstZeroUB, "dUB","setdUB","nominal upper bounds for order"     );
}

//void logit(const std::string logstr) { errstream() << "python: " << logstr << "\n"; }
void logit(py::object logstr) { errstream() << "python: " << convFromPy(logstr) << "\n"; }

py::object mlopt(GlobalOptions &optimiser, int dim, int numreps, py::object &objfn, py::object &callback);

py::object mloptgrid(      int i, int dim, int numreps, py::object objfn, py::object callback) { i = glob_gridInd      (i); return mlopt(getgridref      (i),dim,numreps,objfn,callback); }
py::object mloptDIRect(    int i, int dim, int numreps, py::object objfn, py::object callback) { i = glob_DIRectInd    (i); return mlopt(getDIRectref    (i),dim,numreps,objfn,callback); }
py::object mloptNelderMead(int i, int dim, int numreps, py::object objfn, py::object callback) { i = glob_NelderMeadInd(i); return mlopt(getNelderMeadref(i),dim,numreps,objfn,callback); }
py::object mloptBayesian(  int i, int dim, int numreps, py::object objfn, py::object callback) { i = glob_BayesianInd  (i); return mlopt(getBayesianref  (i),dim,numreps,objfn,callback); }

void internobjfn(gentype &res, Vector<gentype> &x, void *arg)
{
    py::object **argp = (py::object **) arg;

    // In the evaluate case x is a non-zero sized vector, and res is the result of evaluating argp[0] on x
    // In the callback case x is empty, argp[1] is callable and res is an integer (callback type); and we evaluate argp[1] on res

    if      ( x.size() )                                            { convFromPy(res,(*(argp[0]))(convToPy(x  ))); }
    else if ( !isValNone(*(argp[1])) && isValCallable(*(argp[1])) ) {                (*(argp[1]))(convToPy(res));  }
}

py::object mlopt(GlobalOptions &optimiser, int dim, int numreps, py::object &objfn, py::object &callback)
{
    dostartup();

    // Check arguments make sense

    if ( dim <= 0              ) { return makeError("dim must be a positive integer"); }
    if ( !isValCallable(objfn) ) { return makeError("objfn must be a callable");       }

    // Set up standard bounds

    Vector<gentype> xmin(dim,0.0_gent);
    Vector<gentype> xmax(dim,1.0_gent);

    // Return values for callback

    Vector<gentype> xres;
    Vector<gentype> rawxres;
    gentype         fres;
    int             ires = 0;

    int mInd = 0;       // ignore

    Vector<Vector<gentype> > allxres;
    Vector<Vector<gentype> > allrawxres;
    Vector<gentype>          allfres;
    Vector<Vector<gentype> > allcres;
    Vector<gentype>          allmres;
    Vector<gentype>          allsres;
    Vector<double>           s_score;

    gentype meanfres;
    gentype varfres;
    gentype meanires;
    gentype varires;
    gentype meantres;
    gentype vartres;
    gentype meanTres;
    gentype varTres;

    Vector<gentype> meanallfres;
    Vector<gentype> varallfres;
    Vector<gentype> meanallmres;
    Vector<gentype> varallmres;

    Vector<int> distMode(dim,0); // linear range
    Vector<int> varsType(dim,1); // double

    // Actual call to optim

    svmvolatile int dummy = 0;

    py::object *optargs[2] = { &objfn, &callback };

    //optimiser.ispydirect = true;
    int retcode = optimiser.optim(dim,
                                  xres,rawxres,fres,ires,
                                  mInd,
                                  allxres,allrawxres,allfres,allcres,allmres,allsres,s_score,
                                  xmin,xmax,distMode,varsType,
                                  &internobjfn,optargs,dummy,numreps,
                                  meanfres,varfres,meanires,varires,meantres,vartres,meanTres,varTres,meanallfres,varallfres,meanallmres,varallmres);
    //optimiser.ispydirect = false;

    // Setup return dictionary

    py::dict res(py::arg("x")         = convToPy(xres),
                 py::arg("y")         = convToPy(fres),
                 py::arg("i")         = convToPy(ires),
                 py::arg("allx")      = convToPy(allxres),
                 py::arg("ally")      = convToPy(allfres),
                 py::arg("allgt")     = convToPy(allcres),
                 py::arg("allhv")     = convToPy(allmres),
                 py::arg("alls")      = convToPy(allsres),
                 py::arg("allsscore") = convToPy(allsres),
                 py::arg("retcode")   = convToPy(retcode),
                 py::arg("meanf")     = convToPy(meanfres),
                 py::arg("varf")      = convToPy(varfres),
                 py::arg("meani")     = convToPy(meanires),
                 py::arg("vari")      = convToPy(varires),
                 py::arg("meant")     = convToPy(meantres),
                 py::arg("vart")      = convToPy(vartres),
                 py::arg("meanT")     = convToPy(meanTres),
                 py::arg("varT")      = convToPy(varTres),
                 py::arg("meanallf")  = convToPy(meanallfres),
                 py::arg("varallf")   = convToPy(varallfres),
                 py::arg("meanallm")  = convToPy(meanallmres),
                 py::arg("varallm")   = convToPy(varallmres));

    // ...and we're done

    return res;
}

#define COMMA ,
#define RECURSE_ARG(Q_x,Q_fn,Q_prefn,Q_postfn)            \
{                                                         \
    py::tuple Q_xx = py::cast<py::tuple>(Q_x);            \
    int Q_size = (int) Q_xx.size();                       \
    py::list Q_res(Q_size);                               \
                                                          \
    for ( int Q_i = 0 ; Q_i < Q_size ; ++Q_i )            \
    {                                                     \
        Q_res[Q_i] = Q_fn ( Q_prefn Q_xx[Q_i] Q_postfn ); \
    }                                                     \
                                                          \
    return Q_res;                                         \
}
#define RECURSE_ARG_B(Q_x,Q_fn,Q_prefn,Q_postfn)          \
{                                                         \
    py::tuple Q_xx = py::cast<py::tuple>(Q_x);            \
    int Q_size = (int) Q_xx.size();                       \
    int Q_res = 0;                                        \
                                                          \
    for ( int Q_i = 0 ; Q_i < Q_size ; ++Q_i )            \
    {                                                     \
        Q_res += Q_fn ( Q_prefn Q_xx[Q_i] Q_postfn );     \
    }                                                     \
                                                          \
    return Q_res;                                         \
}
#define RECURSE_ARG_C(Q_x,Q_fn,Q_prefn,Q_postfn)          \
{                                                         \
    py::tuple Q_xx = py::cast<py::tuple>(Q_x);            \
    int Q_size = (int) Q_xx.size();                       \
    int Q_res = 0;                                        \
                                                          \
    for ( int Q_i = 0 ; Q_i < Q_size ; ++Q_i )            \
    {                                                     \
        Q_res += Q_fn ( Q_prefn Q_xx[Q_size-(Q_i+1)] Q_postfn ); \
    }                                                     \
                                                          \
    return Q_res;                                         \
}

#define DRECURSE_ARG(Q_z,Q_x,Q_fn,Q_prefn,Q_postfn)       \
{                                                         \
    py::tuple Q_zz = py::cast<py::tuple>(Q_z);            \
    py::tuple Q_xx = py::cast<py::tuple>(Q_x);            \
    int Q_size = (int) Q_xx.size();                       \
    StrucAssert( (int) Q_zz.size() == Q_size );           \
    py::list Q_res(Q_size);                               \
                                                          \
    for ( int Q_i = 0 ; Q_i < Q_size ; ++Q_i )            \
    {                                                     \
        Q_res[Q_i] = Q_fn ( Q_prefn Q_zz[Q_i],Q_xx[Q_i] Q_postfn ); \
    }                                                     \
                                                          \
    return Q_res;                                         \
}
#define DRECURSE_ARG_B(Q_z,Q_x,Q_fn,Q_prefn,Q_postfn)     \
{                                                         \
    py::tuple Q_zz = py::cast<py::tuple>(Q_z);            \
    py::tuple Q_xx = py::cast<py::tuple>(Q_x);            \
    int Q_size = (int) Q_xx.size();                       \
    StrucAssert( (int) Q_zz.size() == Q_size );           \
    int Q_res = 0;                                        \
                                                          \
    for ( int Q_i = 0 ; Q_i < Q_size ; ++Q_i )            \
    {                                                     \
        Q_res += Q_fn ( Q_prefn Q_zz[Q_i],Q_xx[Q_i] Q_postfn ); \
    }                                                     \
                                                          \
    return Q_res;                                         \
}
#define DRECURSE_ARG_C(Q_z,Q_x,Q_fn,Q_prefn,Q_postfn)     \
{                                                         \
    py::tuple Q_zz = py::cast<py::tuple>(Q_z);            \
    py::tuple Q_xx = py::cast<py::tuple>(Q_x);            \
    int Q_size = (int) Q_xx.size();                       \
    StrucAssert( (int) Q_zz.size() == Q_size );           \
    int Q_res = 0;                                        \
                                                          \
    for ( int Q_i = 0 ; Q_i < Q_size ; ++Q_i )            \
    {                                                     \
        Q_res += Q_fn ( Q_prefn Q_zz[Q_size-(Q_i+1)],Q_xx[Q_size-(Q_i+1)] Q_postfn ); \
    }                                                     \
                                                          \
    return Q_res;                                         \
}

void boSetgridsource      (int j) { dostartup(); int i = glob_BayesianInd(0); j = glob_MLInd(j); getBayesianref(i).gridsource = &getMLref(j);          }
void boSetkernapproxsource(int j) { dostartup(); int i = glob_BayesianInd(0); j = glob_MLInd(j); getBayesianref(i).kernapprox = &getMLref(j);          }
void boSetimpmeas         (int j) { dostartup(); int i = glob_BayesianInd(0); j = glob_MLInd(j); getBayesianref(i).impmeasu   = &getMLref(j).getIMP(); }

double mltuneKernel(int method, double xwidth, int tuneK, int tuneP)
{
    dostartup();
    int i = glob_MLInd(0);

    return getMLref(i).tuneKernel(method,xwidth,tuneK,tuneP);
}

// Non-trivial python module function layer

const SparseVector<gentype> &getvec(py::object xa, SparseVector<gentype> &xxa);
const SparseVector<gentype> &getvec(py::object xa, SparseVector<gentype> &xxa)
{
    dostartup();
    int i = glob_MLInd(0);

    if      ( isValInteger(xa)   ) { return getMLrefconst(i).x(toInt(xa));     }
    else if ( convFromPy(xxa,xa) ) { throw("Can't convert x to sparsevector"); }

    return xxa;
}

py::object mlK0(void)
{
    dostartup();
    int i = glob_MLInd(0);

    gentype res;
    SparseVector<gentype> xxa;

    getMLrefconst(i).K0(res);

    return convToPy(res);
}

py::object mlK1(py::object xa)
{
    if ( isValTuple(xa) ) { RECURSE_ARG(xa,mlK1,,); }

    if ( isValNone(xa) ) { return mlK0(); }

    dostartup();
    int i = glob_MLInd(0);

    gentype res;
    SparseVector<gentype> xxa;

    if ( isValInteger(xa) ) { getMLrefconst(i).K1(res,toInt(xa)     ); }
    else                    { getMLrefconst(i).K1(res,getvec(xa,xxa)); }

    return convToPy(res);
}

py::object mlK2(py::object xa, py::object xb)
{
    if      ( isValTuple(xa) ) { RECURSE_ARG(xa,mlK2,,COMMA xb); }
    else if ( isValTuple(xb) ) { RECURSE_ARG(xb,mlK2,xa COMMA,); }

    if ( isValNone(xa) ) { return mlK1(xb); }
    if ( isValNone(xb) ) { return mlK1(xa); }

    dostartup();
    int i = glob_MLInd(0);

    gentype res;
    SparseVector<gentype> xxa,xxb;

    if ( isValInteger(xa) && isValInteger(xb) ) { getMLref(i).K2(res,toInt(xa),     toInt(xb)     ); }
    else                                        { getMLref(i).K2(res,getvec(xa,xxa),getvec(xb,xxb)); }

    return convToPy(res);
}

py::object mlK3(py::object xa, py::object xb, py::object xc)
{
    if      ( isValTuple(xa) ) { RECURSE_ARG(xa,mlK3,,COMMA xb COMMA xc); }
    else if ( isValTuple(xb) ) { RECURSE_ARG(xb,mlK3,xa COMMA, COMMA xc); }
    else if ( isValTuple(xc) ) { RECURSE_ARG(xc,mlK3,xa COMMA xb COMMA,); }

    if ( isValNone(xa) ) { return mlK2(xb,xc); }
    if ( isValNone(xb) ) { return mlK2(xa,xc); }
    if ( isValNone(xc) ) { return mlK2(xa,xb); }

    dostartup();
    int i = glob_MLInd(0);

    gentype res;
    SparseVector<gentype> xxa,xxb,xxc;

    if ( isValInteger(xa) && isValInteger(xb) && isValInteger(xc) ) { getMLref(i).K3(res,toInt(xa),     toInt(xb),     toInt(xc)     ); }
    else                                                            { getMLref(i).K3(res,getvec(xa,xxa),getvec(xb,xxb),getvec(xc,xxc)); }

    return convToPy(res);
}

py::object mlK4(py::object xa, py::object xb, py::object xc, py::object xd)
{
    if      ( isValTuple(xa) ) { RECURSE_ARG(xa,mlK4,,COMMA xb COMMA xc COMMA xd); }
    else if ( isValTuple(xb) ) { RECURSE_ARG(xb,mlK4,xa COMMA, COMMA xc COMMA xd); }
    else if ( isValTuple(xc) ) { RECURSE_ARG(xc,mlK4,xa COMMA xb COMMA, COMMA xd); }
    else if ( isValTuple(xd) ) { RECURSE_ARG(xd,mlK4,xa COMMA xb COMMA xc COMMA,); }

    if ( isValNone(xa) ) { return mlK3(xb,xc,xd); }
    if ( isValNone(xb) ) { return mlK3(xa,xc,xd); }
    if ( isValNone(xc) ) { return mlK3(xa,xb,xd); }
    if ( isValNone(xd) ) { return mlK3(xa,xb,xc); }

    dostartup();
    int i = glob_MLInd(0);

    gentype res;
    SparseVector<gentype> xxa,xxb,xxc,xxd;

    if ( isValInteger(xa) && isValInteger(xb) && isValInteger(xc) && isValInteger(xd) ) { getMLref(i).K4(res,toInt(xa),     toInt(xb),     toInt(xc),     toInt(xd)     ); }
    else                                                                                { getMLref(i).K4(res,getvec(xa,xxa),getvec(xb,xxb),getvec(xc,xxc),getvec(xd,xxd)); }

    return convToPy(res);
}

py::object svmtest(int i, std::string type)
{
    dostartup();

    static bool firstrun = true;
    static SparseVector<gentype> gentestfn;

    if ( firstrun )
    {
        for ( int j = 1 ; j <= NUMOPTTESTFNS ; j++ )
        {
            { gentype jj(j);      gentype x("x"); gentestfn("&",j)      = testfn(jj,x); } // this is a function of x
            { gentype jj(2000+j); gentype x("x"); gentestfn("&",2000+j) = testfn(jj,x); } // this is a function of x
        }

        firstrun = false;
    }

    return convToPy(gentestfn(i+( ( type == "norm" ) ? 2000 : 0 )));
}

void callintercalc(void)
{
    dostartup();

    intercalc(outstream(),std::cin);
}

py::object svmeval(py::object f, py::object x)
{
    dostartup();

    gentype ff,xx;

    if      ( isValString(f)   ) { std::string xf; convFromPy(xf,f); ff = xf; } // convert string to equation
    else if ( convFromPy(ff,f) ) { throw("Error: can't convert f to gentype function stub in svmeval."); }

    if ( convFromPy(xx,x) ) { return convToPy(nan("")); }

    return convToPy(ff(xx)); // evaluate f(x) and convert back to python friendly form
}

void callsnakes(int wide, int high) { dostartup(); snakes(wide,high,(20*high)/23,5,20,100000); }

void svmheavy(int method, int permode, const std::string commstr, int wml);

void svmheavya(void)                      { static std::string dummy; svmheavy(1,-1,dummy,-1);      }
void svmheavyb(int permode)               { static std::string dummy; svmheavy(2,permode,dummy,-1); }
void svmheavyc(const std::string commstr) {                           svmheavy(3,-1,commstr,-1);    }

double mlcalcLOO   (void)                          { dostartup(); int i = glob_MLInd(0); return calcLOO(getMLref(i));                   }
double mlcalcRecall(void)                          { dostartup(); int i = glob_MLInd(0); return calcRecall(getMLref(i));                }
double mlcalcCross (int m, int rndit, int numreps) { dostartup(); int i = glob_MLInd(0); return calcCross(getMLref(i),m,rndit,numreps); }

int selgridopt      (int i, int rst) { dostartup(); int res = glob_gridInd      (i,1); if ( rst ) { getgridref      (i).reset(); } return res; }
int selDIRectopt    (int i, int rst) { dostartup(); int res = glob_DIRectInd    (i,1); if ( rst ) { getDIRectref    (i).reset(); } return res; }
int selNelderMeadopt(int i, int rst) { dostartup(); int res = glob_NelderMeadInd(i,1); if ( rst ) { getNelderMeadref(i).reset(); } return res; }
int selBayesianopt  (int i, int rst) { dostartup(); int res = glob_BayesianInd  (i,1); if ( rst ) { getBayesianref  (i).reset(); } return res; }

int selml                 (int i)        { dostartup();                          return glob_MLInd(                                   i,   1); }
int selmlcgtapprox        (int i, int k) { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_cgtapprox        (i,k),1); }
int selmlmuapprox         (int i, int k) { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_muapprox         (i,k),1); }
//int selmlaugxapprox       (int i, int k) { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_augxapprox       (i,k),1); }
int selmlsigmaapprox      (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_sigmaapprox      (i  ),1); }
int selmldiffmodel        (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_diffmodel        (i  ),1); }
int selmlsrcmodel         (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_srcmodel         (i  ),1); }
int selmlcgtapprox_prior  (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_cgtapprox_prior  (i  ),1); }
int selmlmuapprox_prior   (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_muapprox_prior   (i  ),1); }
//int selmlaugxapprox_prior (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_augxapprox_prior (i  ),1); }
int selmlsigmaapprox_prior(int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_sigmaapprox_prior(i  ),1); }
int selmldiffmodel_prior  (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_diffmodel_prior  (i  ),1); }
int selmlsrcmodel_prior   (int i)        { dostartup(); i = glob_BayesianInd(i); return glob_MLInd(MLIndForBayesian_srcmodel_prior   (i  ),1); }

void swapml  (int i, int j) { dostartup(); i = glob_MLInd(i); j = glob_MLInd(j); StrucAssert(i != j); qswap(getMLref(i),getMLref(j)); }
void copyml  (int i, int j) { dostartup(); i = glob_MLInd(i); j = glob_MLInd(j); StrucAssert(i != j); getMLref(i) = getMLrefconst(j); }
void assignml(int i, int j) { dostartup(); i = glob_MLInd(i); j = glob_MLInd(j); StrucAssert(i != j); getMLref(j) = getMLrefconst(i); }

int setpriml(int j)
{
    dostartup();
    int i = glob_MLInd(0);
        j = glob_MLInd(j);

    StrucAssert( i != j );

    return getMLref(i).setpriml(&(getMLrefconst(j).getMLconst()));
}

int makeMonot(int n, int t, py::object xb, py::object xlb, py::object xub, int d, py::object y, double Cweight, double epsweight, int j)
{
    dostartup();
    int i = glob_MLInd(0);
        j = glob_MLInd(j);

    SparseVector<gentype> zxb;
    SparseVector<double> zxlb;
    SparseVector<double> zxub;

    gentype zy;

    int errcode = 0;

    errcode |= convFromPy(zxb,xb);
    errcode |= convFromPy(zxlb,xlb);
    errcode |= convFromPy(zxub,xub);
    errcode |= convFromPy(zy,y);

    if ( !errcode )
    {
        makeMonotone(getMLref(i),n,t,zxb,zxlb,zxub,d,zy,Cweight,epsweight,((t==2)?&getMLref(j):nullptr));
    }

    return errcode;
}

int addTrainingVectorml(py::object x, py::object z, int j)
{
    // Can have z a tuple and x not a tuple if target is tuple.

    if      ( isValTuple(x) && isValTuple(z) && ( py::cast<py::tuple>(z).size() == py::cast<py::tuple>(x).size() ) && ( j == -1 ) ) { DRECURSE_ARG_B(x,z,addTrainingVectorml,,COMMA j); }
    else if ( isValTuple(x) && isValTuple(z) && ( py::cast<py::tuple>(z).size() == py::cast<py::tuple>(x).size() )                ) { DRECURSE_ARG_C(x,z,addTrainingVectorml,,COMMA j); }
    else if ( isValTuple(x)                                                                                        && ( j == -1 ) ) { RECURSE_ARG_B(x,addTrainingVectorml,,COMMA z COMMA j); }
    else if ( isValTuple(x)                                                                                                       ) { RECURSE_ARG_C(x,addTrainingVectorml,,COMMA z COMMA j); }

    dostartup();
    int i = glob_MLInd(0);

    if ( j == -1 ) { j = getMLref(i).N(); }

    SparseVector<gentype> xx;
    gentype zz;
    double cw = 1.0;
    double ew = 1.0;
    int dd = 2;

    int errcode = 0;

    errcode |= convFromPy(xx,x);
    errcode |= convFromPy(zz,z);

    if ( isValDict(z) )
    {
        SparseVector<gentype> xdefault(xx);

        py::dict altz = py::cast<py::dict>(z);

        for ( auto elm : altz )
        {
            std::string keyz;

            errcode |= convFromPy(keyz,elm.first);

            if ( ( keyz == "y"  ) && ( errcode |= convFromPy(zz,elm.second) ) ) { zz.makeNone(); }
            if ( ( keyz == "cw" ) && ( errcode |= convFromPy(cw,elm.second) ) ) { cw = 1.0;      }
            if ( ( keyz == "ew" ) && ( errcode |= convFromPy(ew,elm.second) ) ) { ew = 1.0;      }
            if ( ( keyz == "d"  ) && ( errcode |= convFromPy(dd,elm.second) ) ) { dd = 2;        }
            if ( ( keyz == "x"  ) && ( errcode |= convFromPy(xx,elm.second) ) ) { xx = xdefault; }

            xdefault = xx;
       }
    }

    if ( errcode )
    {
        return 0;
    }

    return getMLref(i).addTrainingVector(j,zz,xx,cw,ew,dd);
}

int faddTrainingVectorml(const std::string &fname, int ignoreStart, int imax, int reverse, int j)
{
    dostartup();
    int i = glob_MLInd(0);

    if ( j == -1 ) { j = getMLref(i).N(); }

    return addtrainingdata(getMLref(i),fname.c_str(),reverse,ignoreStart,imax,j);
}

int removeTrainingVectorml(int j, int num)
{
    dostartup();
    int i = glob_MLInd(0);

    if ( j == -1 ) { j = getMLref(i).N()-1; }

    return getMLref(i).removeTrainingVector(j,num);
}

py::object muml(py::object xa)
{
    if ( isValTuple(xa) ) { RECURSE_ARG(xa,muml,,); }

    dostartup();
    int i = glob_MLInd(0);

    gentype resh,resg;
    SparseVector<gentype> xx;

    if ( isValInteger(xa) ) { getMLref(i).gh(resh,resg,toInt(xa)    ); }
    else                    { getMLref(i).gh(resh,resg,getvec(xa,xx)); }

    return convToPy(resh);
}

py::object mugml(py::object xa, int fmt)
{
    if ( isValTuple(xa) ) { RECURSE_ARG(xa,mugml,, COMMA fmt); }

    dostartup();
    int i = glob_MLInd(0);

    gentype resh,resg;
    SparseVector<gentype> xx;

    if ( isValInteger(xa) ) { getMLref(i).gh(resh,resg,toInt(xa),    fmt); }
    else                    { getMLref(i).gh(resh,resg,getvec(xa,xx),fmt); }

    return convToPy(resg); // this is the difference from muml
}

py::object varml(py::object xa)
{
    if ( isValTuple(xa) ) { RECURSE_ARG(xa,varml,,); }

    dostartup();
    int i = glob_MLInd(0);

    gentype resv,resmu;
    SparseVector<gentype> xx;

    if ( isValInteger(xa) ) { getMLref(i).var(resv,resmu,toInt(xa)    ); }
    else                    { getMLref(i).var(resv,resmu,getvec(xa,xx)); }

    return convToPy(resv);
}

py::object covml(py::object xa, py::object xb)
{
    if      ( isValTuple(xa) ) { RECURSE_ARG(xa,covml,, COMMA xb); }
    else if ( isValTuple(xb) ) { RECURSE_ARG(xb,covml,xa COMMA ,); }

    dostartup();
    int i = glob_MLInd(0);

    gentype resv,resmu;
    SparseVector<gentype> xx,yy;

    if ( isValInteger(xa) && isValInteger(xb) ) { getMLref(i).cov(resv,resmu,toInt(xa),    toInt(xb)    ); }
    else                                        { getMLref(i).cov(resv,resmu,getvec(xa,xx),getvec(xb,yy)); }

    return convToPy(resv);
}

py::object predvarml(py::object xa, py::object pp, py::object sigw)
{
    if      ( isValTuple(xa)                      ) { RECURSE_ARG(xa,predvarml,, COMMA pp COMMA sigw); }
    else if ( isValTuple(pp) && !isValTuple(sigw) ) { RECURSE_ARG(pp,predvarml, xa COMMA, COMMA sigw); }
    else if ( isValTuple(pp) &&  isValTuple(sigw) ) { DRECURSE_ARG(pp,sigw,predvarml,xa COMMA,);       }

    dostartup();
    int i = glob_MLInd(0);

    StrucAssert( isValCastableToReal(sigw) );

    gentype resv_pred,resv,resmu;
    SparseVector<gentype> xx;
    SparseVector<gentype> yy;
    double s = 1.0; convFromPy(s,sigw);

    if ( isValInteger(xa) && isValInteger(xa) ) { getMLref(i).predvar(resv_pred,resv,resmu,toInt(xa)    ,toInt(pp)    ,s); }
    else                                        { getMLref(i).predvar(resv_pred,resv,resmu,getvec(xa,xx),getvec(pp,yy),s); }

    return convToPy(resv_pred);
}

py::object predcovml(py::object xa, py::object xb, py::object pp, py::object sigw)
{
    if      ( isValTuple(xa)                      ) { RECURSE_ARG( xa,predcovml,, COMMA xb COMMA pp COMMA sigw);    }
    else if (                      isValTuple(xb) ) { RECURSE_ARG( xb,predcovml, xa COMMA, COMMA pp COMMA sigw);    }
    else if ( isValTuple(pp) && !isValTuple(sigw) ) { RECURSE_ARG( pp,predcovml,     xa COMMA xb COMMA,COMMA sigw); }
    else if ( isValTuple(pp) &&  isValTuple(sigw) ) { DRECURSE_ARG(pp,sigw,predcovml,xa COMMA xb COMMA,);           }

    dostartup();
    int i = glob_MLInd(0);

    StrucAssert( isValCastableToReal(sigw) );

    gentype resv_pred,resv,resmu;
    SparseVector<gentype> xx,yy,zz;
    double s = 1.0; convFromPy(s,sigw);

    if ( isValInteger(xa) && isValInteger(xb) && isValInteger(pp) ) { getMLref(i).predcov(resv_pred,resv,resmu,toInt(xa),    toInt(xb),    toInt(pp)    ,s); }
    else                                                            { getMLref(i).predcov(resv_pred,resv,resmu,getvec(xa,xx),getvec(xb,yy),getvec(pp,zz),s); }

    return convToPy(resv_pred);
}

py::object mlalpha(void)
{
    dostartup();
    int i = glob_MLInd(0);

    Vector<gentype> res;
    int type = getMLref(i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { res = getMLref(i).alpha();    }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { res = getMLref(i).muWeight(); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { res = getMLref(i).gamma();    }

    return convToPy(res);
}

py::object mlbias(void)
{
    dostartup();
    int i = glob_MLInd(0);

    gentype res('N');
    int type = getMLref(i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { res = getMLref(i).bias  (); }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { res = getMLref(i).muBias(); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { res = getMLref(i).delta (); }

    return convToPy(res);
}

py::object mlGp(void)
{
    dostartup();
    int i = glob_MLInd(0);

    gentype res('N');
    int type = getMLref(i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { res = getMLref(i).Gp   (); }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { res = getMLref(i).gprGp(); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { res = getMLref(i).lsvGp(); }

    return convToPy(res);
}

int mlsetalpha(py::object src)
{
    dostartup();
    int i = glob_MLInd(0);

    int errcode = 0;

    Vector<gentype> altsrc;

    if ( ( errcode = convFromPy(altsrc,src) ) ) { return errcode; }

    int type = getMLref(i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { getMLref(i).setAlpha   (altsrc); }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { getMLref(i).setmuWeight(altsrc); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { getMLref(i).setgamma   (altsrc); }
    else                                           { return 1;                        }

    return 0;
}

int mlsetbias(py::object src)
{
    dostartup();
    int i = glob_MLInd(0);

    int errcode = 0;

    gentype altsrc('N');

    if ( ( errcode = convFromPy(altsrc,src) ) ) { return errcode; }

    int type = getMLref(i).type();

         if ( ( type >= 0   ) && ( type <= 99  ) ) { getMLref(i).setBias  (altsrc); }
    else if ( ( type >= 400 ) && ( type <= 499 ) ) { getMLref(i).setmuBias(altsrc); }
    else if ( ( type >= 500 ) && ( type <= 599 ) ) { getMLref(i).setdelta (altsrc); }
    else                                           { return 1;                      }

    return 0;
}







































// Convert C++ types to python

py::object convToPy(const gentype &src)
{
    // Order is important - keep objects as close to source type as possible (especially none)

    if      ( src.isValNone()    ) { return py::none();                                    }
    else if ( src.isValInteger() ) { return convToPy((int)                           src); }
    else if ( src.isValReal()    ) { return convToPy((double)                        src); }
    else if ( src.isValComplex() ) { return convToPy((std::complex<double>)          src); }
    else if ( src.isValString()  ) { return convToPy((const std::string &)           src); }
    else if ( src.isValVector()  ) { return convToPy((const Vector<gentype> &)       src); }
    else if ( src.isValMatrix()  ) { return convToPy((const Matrix<gentype> &)       src); }
    else if ( src.isValSet()     ) { return convToPy((const Set<gentype> &)          src); }
    else if ( src.isValDict()    ) { return convToPy((const Dict<gentype,dictkey> &) src); }
    else if ( src.isValEqnDir()  ) { return py::cpp_function([src](const py::object &x) { return convToPy(src(convFromPy(x))); },py::arg("x")); }
    else if ( src.isValError()   ) { return py::cast(nan(((const std::string &) src).c_str())); }
    else if ( src.isValAnion()   ) { return py::cast(nan("Gentype type anion (hypercomplex) has no python equivalent.")); }
    else if ( src.isValDgraph()  ) { return py::cast(nan("Gentype type directed graph has no python equivalent."));       }

    return py::cast(nan("Type error in gentype->python conversion."));
}

// Convert python to C++ types

int convFromPy(gentype &res, const py::object &src)
{
    int errcode = 4096;
    std::complex<double> tmpres;

    // Order is important - keep objects as close to source type as possible (especially none/null)

    if      ( isValNone(src)     ) { errcode = 0; res.force_none();                              }
    else if ( isValInteger(src)  ) { errcode = convFromPy(res.force_int(),   src);               }
    else if ( isValReal(src)     ) { errcode = convFromPy(res.force_double(),src);               }
    else if ( isValComplex(src)  ) { errcode = convFromPy(tmpres,            src); res = tmpres; }
    else if ( isValString(src)   ) { errcode = convFromPy(res.force_string(),src);               }
    else if ( isValList(src)     ) { errcode = convFromPy(res.force_vector(),src);               }
    else if ( isValDict(src)     ) { errcode = convFromPy(res.force_dict(),  src);               }
    else if ( isValTuple(src)    ) { errcode = convFromPy(res.force_set(),   src);               }
    //else if ( isValMatrix(src)   ) { errcode = convFromPy(res.force_matrix(),src); } - can't disambiguate between array of vectors and matrices at present
    else if ( isValCallable(src) ) { errcode = gentype_function(gentype(const gentype &),res,[src](const gentype &x) { return convFromPy(src(convToPy(x))); }); }
    else                           { errcode = 2048; }

    if ( errcode )
    {
        std::string errstr;
        errstr  = "Couldn't convert python object to gentype (error code ";
        errstr += std::to_string(errcode);
        errstr += ")";
        res.makeError(errstr);
    }

    return errcode ? (errcode+4096) : 0;
}

gentype convFromPy(const py::object &src)
{
    gentype res;
    convFromPy(res,src);
    return res;
}

// Convert C++ types to python

                   py::object convToPy(int                         src) { return py::cast(src);         }
                   py::object convToPy(double                      src) { return py::cast(src);         }
                   py::object convToPy(std::complex<double>        src) { return py::cast(src);         }
                   py::object convToPy(const d_anion              &src) { if ( src.order() <= 1 ) { return py::cast((std::complex<double>) src); } return py::cast(nan("Cant cast hypercomplex anion to python object")); }
                   py::object convToPy(const std::string          &src) { return py::cast(src.c_str()); }
template <class T> py::object convToPy(int vsize, const T         *src) { py::list  res(vsize);         for ( int i = 0 ; i < vsize         ; ++i ) { res[i] = convToPy(src[i]);       } return res; }
template <>        py::object convToPy(const Vector<double>       &src) { py::list  res(src.size());    for ( int i = 0 ; i < src.size()    ; ++i ) { res[i] = py::cast(src.v(i));     } return res; }
template <class T> py::object convToPy(const Vector<T>            &src) { py::list  res(src.size());    for ( int i = 0 ; i < src.size()    ; ++i ) { res[i] = convToPy(src(i));       } return res; }
template <class T> py::object convToPy(const Set<T>               &src) { py::tuple res(src.size());    for ( int i = 0 ; i < src.size()    ; ++i ) { res[i] = convToPy(src.all()(i)); } return res; }
template <class T> py::object convToPy(const Dict<T,dictkey>      &src) { py::dict  res;                for ( int i = 0 ; i < src.size()    ; ++i ) { res[convToPy(src.key(i))] = convToPy(src.val(i)); } return res; }
template <class T> py::object convToPy(const Matrix<T>            &src) { py::list  res(src.numRows()); for ( int i = 0 ; i < src.numRows() ; ++i ) { retVector<T> tmpa,tmpb; res[i] = convToPy(src(i,tmpa,tmpb)); } return res; }
template <class T> py::object convToPy(const SparseVector<T>      &src) { gentype altsrc(src); return convToPy(altsrc); }

// Convert python to C++ types
//
// naive: don't check types, just assume

                   int naivePyToInt(int                   &res, const py::object &src);
                   int naivePyToDbl(double                &res, const py::object &src);
                   int naivePyToCpl(std::complex<double>  &res, const py::object &src);
                   int naivePyToCpl(d_anion               &res, const py::object &src);
                   int naivePyToStr(std::string           &res, const py::object &src);
template <class T> int naivePyToVec(Vector<T>             &res, const py::object &src);
template <class T> int naivePyToMat(Matrix<T>             &res, const py::object &src);
template <class T> int naivePyToSet(Set<T>                &res, const py::object &src);
template <class T> int naivePyToDct(Dict<T,dictkey>       &res, const py::object &src);
template <class T> int naivePyToSpv(SparseVector<T>       &res, const py::object &src);
template <>        int naivePyToSpv(SparseVector<gentype> &res, const py::object &src);
                   int naivePyToEqn(gentype               &res, const py::object &src);

template <class T> int convFromPy(T                     &res, const py::handle &src) { return convFromPy(res,py::reinterpret_borrow<py::object>(src)); }
                   int convFromPy(int                   &res, const py::object &src) { if ( isValCastableToInteger(src) ) { return naivePyToInt(res,src); } res = 0;                                return 1;   }
                   int convFromPy(double                &res, const py::object &src) { if ( isValCastableToReal   (src) ) { return naivePyToDbl(res,src); } res = nan("Not castable to double");    return 2;   }
                   int convFromPy(std::complex<double>  &res, const py::object &src) { if ( isValCastableToComplex(src) ) { return naivePyToCpl(res,src); } res = nan("Not castable to complex");   return 4;   }
                   int convFromPy(d_anion               &res, const py::object &src) { if ( isValCastableToComplex(src) ) { return naivePyToCpl(res,src); } res = nan("Not castable to anion");     return 4;   }
                   int convFromPy(std::string           &res, const py::object &src) { if ( isValString           (src) ) { return naivePyToStr(res,src); } res = "";                               return 8;   }
template <class T> int convFromPy(Vector<T>             &res, const py::object &src) { if ( isValList             (src) ) { return naivePyToVec(res,src); } res.resize(0);                          return 16;  }
template <class T> int convFromPy(Matrix<T>             &res, const py::object &src) { if ( isValList             (src) ) { return naivePyToMat(res,src); } res.resize(0,0);                        return 32;  }
template <class T> int convFromPy(Set<T>                &res, const py::object &src) { if ( isValTuple            (src) ) { return naivePyToSet(res,src); } Set<T> temp; res = temp;                return 64;  }
template <class T> int convFromPy(Dict<T,dictkey>       &res, const py::object &src) { if ( isValDict             (src) ) { return naivePyToDct(res,src); } Dict<gentype,dictkey> temp; res = temp; return 128; }
template <>        int convFromPy(SparseVector<gentype> &res, const py::object &src) { if ( isValList             (src) ) { return naivePyToSpv(res,src); } res.zero();                             return 256; }
template <class T> int convFromPy(SparseVector<T>       &res, const py::object &src) { if ( isValList             (src) ) { return naivePyToSpv(res,src); } res.zero();                             return 512; }

int naivePyToInt(int &res, const py::object &src)
{
    if      ( isValNone(src)    ) { res =       0;          }
    else if ( isValInteger(src) ) { res =       toInt(src); }
    else                          { res = (int) toDbl(src); }

    return 0;
}

int naivePyToDbl(double &res, const py::object &src)
{
    if      ( isValNone(src)    ) { res =          0.0;        }
    else if ( isValInteger(src) ) { res = (double) toInt(src); }
    else                          { res =          toDbl(src); }

    return 0;
}

int naivePyToCpl(std::complex<double> &res, const py::object &src)
{
    if      ( isValNone(src)    ) { res =          0.0;        }
    else if ( isValComplex(src) ) { res =          toCpl(src); }
    else if ( isValInteger(src) ) { res = (double) toInt(src); }
    else                          { res =          toDbl(src); }

    return 0;
}

int naivePyToCpl(d_anion &res, const py::object &src)
{
    if      ( isValNone(src)    ) { res =          0.0;        }
    else if ( isValComplex(src) ) { res =          toCpl(src); }
    else if ( isValInteger(src) ) { res = (double) toInt(src); }
    else                          { res =          toDbl(src); }

    return 0;
}

int naivePyToStr(std::string &res, const py::object &src)
{
    res = py::cast<py::str>(src);

    return 0;
}

template <class T>
int naivePyToVec(Vector<T> &res, const py::object &src)
{
    py::list altsrc = py::cast<py::list>(src);
    int errcode = 0;
    int i = 0;

    res.resize((int) altsrc.size());

    for ( auto elm : altsrc )
    {
        errcode |= convFromPy(res("&",i),elm);
        ++i;
    }

    return errcode ? (errcode+1024) : 0;
}

template <class T>
int naivePyToMat(Matrix<T> &res, const py::object &src)
{
    py::list altsrc = py::cast<py::list>(src);
    int errcode = 0;
    int i = 0;

    retVector<T> tmpa;
    retVector<T> tmpb;
    Vector<T> altres;

    for ( auto elm : altsrc )
    {
        errcode |= convFromPy(altres,elm);

        if ( !i ) { res.resize(1,(int) altres.size()); }
        else      { res.addRow(i);                     }

        res("&",i,tmpa,tmpb) = altres;
        ++i;
    }

    return errcode ? (errcode+2048) : 0;
}

template <class T>
int naivePyToSet(Set<T> &res, const py::object &src)
{
    Set<T> temp;
    int errcode = 0;

    res = temp;

    py::tuple altsrc = py::cast<py::tuple>(src);

    for ( auto elm : altsrc )
    {
        T tg;

        errcode |= convFromPy(tg,elm);
        res.add(tg); // will add to end
    }

    return errcode ? (errcode+4096) : 0;
}

template <class T>
int naivePyToDct(Dict<T,dictkey> &res, const py::object &src)
{
    Dict<gentype,dictkey> temp;
    int errcode = 0;

    res = temp;

    py::dict altsrc = py::cast<py::dict>(src);

    for ( auto elm : altsrc )
    {
        dictkey ti;
        T tg;

        errcode |= convFromPy(ti,elm.first);
        errcode |= convFromPy(tg,elm.second);

        res("&",ti) = tg;
    }

    return errcode ? (errcode+8192) : 0;
}

template <>
int naivePyToSpv(SparseVector<gentype> &res, const py::object &src)
{
/* old version that relies on conversion from gentype.cc
   this doesn't work right for eg [ 1.2 3.4 None ]. The final None is
   converted to none, which is removed by gentype to leave [ 1.2 3.4 ],
   which is not what we want when None could be a placedholder

    gentype altsrc;

    int errcode = convFromPy(altsrc,src); // sparsevector gets encoded into altres
errstream() << "phantomxyz presparse: " << altsrc << "\n";

    if ( !errcode )
    {
        if ( altsrc.isCastableToVectorWithoutLoss() )
        {
            SparseVector<gentype> tmpres = (const SparseVector<gentype> &) altsrc;
errstream() << "phantomxyz presparse conv: " << tmpres << "\n";

            res.zero();

            for ( int i = 0 ; i < tmpres.indsize() ; ++i )
            {
                res("&",tmpres.ind(i)) = (T) tmpres.direcref(i);
            }
errstream() << "phantomxyz presparse res: " << res << "\n";
        }

        else
        {
            errcode = 1024;
        }
    }

    if ( errcode )
    {
        res.zero();
        return errcode+2048;
    }

    return 0;
*/

    res.zero();

    py::list altsrc = py::cast<py::list>(src);
    int errcode = 0;
    int fnum = 0;
    int unum = 0;
    int iv = 0;
    gentype tmpres;

    for ( auto elm : altsrc )
    {
        bool issep = false;

        errcode |= convFromPy(tmpres,elm);

        if ( tmpres.isValString() )
        {
                 if ( ((const std::string &) tmpres) == "~"    ) { unum++;             iv = 0; issep = true; }
            else if ( ((const std::string &) tmpres) == "::::" ) { unum = 0; fnum = 4; iv = 0; issep = true; }
            else if ( ((const std::string &) tmpres) == ":::"  ) { unum = 0; fnum = 3; iv = 0; issep = true; }
            else if ( ((const std::string &) tmpres) == "::"   ) { unum = 0; fnum = 2; iv = 0; issep = true; }
            else if ( ((const std::string &) tmpres) == ":"    ) { unum = 0; fnum = 1; iv = 0; issep = true; }
        }

        if ( !issep )
        {
            if      ( fnum == 0 ) { res.n ("&",iv,unum) = tmpres; }
            else if ( fnum == 1 ) { res.f1("&",iv,unum) = tmpres; }
            else if ( fnum == 2 ) { res.f2("&",iv,unum) = tmpres; }
            else if ( fnum == 3 ) { res.f3("&",iv,unum) = tmpres; }
            else if ( fnum == 4 ) { res.f4("&",iv,unum) = tmpres; }

            iv++;
        }
    }

    return errcode ? (errcode+16384) : 0;
}

template <class T>
int naivePyToSpv(SparseVector<T> &res, const py::object &src)
{
    SparseVector<gentype> altres;
    int errcode = convFromPy(altres,src);

    res.zero();

    for ( int i = 0 ; i < altres.indsize() ; ++i )
    {
        res("&",altres.ind(i)) = (T) altres.direcref(i);
    }

    return errcode ? (errcode+32796) : 0;
}

//int naivePyToEqn(gentype &res, const py::object &src)
//{
//    // Method: make res a call to pycall, and then set altpycall (which
//    // will override the usual functionality of pycall) to a
//    // std::function wrapper around a lambda that captures src
//
//    res = "pycall(0,x)";
//    res.altpycall = new std::function<gentype(const gentype &)>([src](const gentype &x) { gentype altres; convFromPy(altres,src(convToPy(x))); return altres; });
//
////Old version: store a copy, make pycall caller to retrieve it and evaluate through eval functionality
////             (not a good ideal to rely on eval!)
////    py::object *altsrc = new py::object(src);
////    (*altsrc).inc_ref(); // want the copy to hang around forever!
////
////    int i = pyosetsrc(-1,altsrc);
////
////    res = "pycall(y,x)";
////    SparseVector<SparseVector<gentype>> ii;
////    ii("&",0)("&",1) = i; // this is y
////    res.substitute(ii);
//
//    return 0;
//}
































// pycall support - allow gentype to directly call python functions (named or
// stored on heap via pyosetsrc) via eval
//
// Data translation, heap store

py::object &setgetsrc(int &i, int doset, py::object *val = nullptr);
py::object &setgetsrc(int &i, int doset, py::object *val)
{
    static thread_local SparseVector<py::object *> xval;
    static thread_local SparseVector<int> useind;

    if ( doset == 1 )
    {
        // Adding element to store

        StrucAssert(val);

        if ( i == -1 )
        {
            i = 0;

            while ( useind.isindpresent(i) ) { ++i; } // This is shared between all stores, so indices are unique.
        }

        xval("&",i) = val;
        useind("&",i) = 1;
    }

    if ( doset == -1 )
    {
        // Clear and delete stored objects

        StrucAssert(val);

        useind.zero();

        for ( int j = 0 ; j < xval.indsize() ; ++j )
        {
            if ( xval.direref(j) )
            {
                MEMDEL(xval.direref(j));
            }
        }

        return *val;
    }

    // return object reference

    StrucAssert( i >= 0 );
    StrucAssert( useind.isindpresent(i) );

    return *(xval(i));
}

int         pyosetsrc(int k, const py::object &src) { py::object *newsrc; MEMNEW(newsrc,py::object(src)); setgetsrc(k,1,newsrc); return k; }
py::object &pyogetsrc(int k)                        { return setgetsrc(k,0);                                                               }
void        pyoclrsrc(void)                         { int k = 0; py::object dummy = py::cast(1); setgetsrc(k,-1,&dummy);                   }

// drop-in replacement for pycall function in gentype.cc
// (the gentype version, which uses a system call, is disabled by the macro PYLOCAL)
//
// NB these aren't really necessary now that gentype conversion uses lambdas, but could be handy for other purposes

void pycall_x(const std::string &fn, gentype &res, py::object &xx);
void pycall_x(const std::string &fn, gentype &res, py::object &xx)
{
    dostartup();

    // Store in transfer indices (will never be deleted)

    int i = pyosetsrc(-1,xx);

    // Construct run command

    std::string evalfn;

    evalfn =  fn;
    evalfn += "(pyheavy.internal.pyogetsrc(";
    evalfn += std::to_string(i);
    evalfn += "))";

    // Evaluated run command

    py::object resobj = pyeval()(evalfn);

    // Retrieve results of operation

    convFromPy(res,resobj);

    return;
}

void pycall_x(int fni, gentype &res, py::object &xx);
void pycall_x(int fni, gentype &res, py::object &xx)
{
    if ( fni >= 0 )
    {
        dostartup();

        convFromPy(res,pyogetsrc(fni)(xx));
    }

    else
    {
        res.force_null();
    }

    return;
}

template <> void pycall(const std::string &fn, gentype &res, const SparseVector<gentype> &x) { dostartup(); py::object xx = convToPy(x); pycall_x(fn, res,xx); }
template <> void pycall(int fni,               gentype &res, const SparseVector<gentype> &x) { dostartup(); py::object xx = convToPy(x); pycall_x(fni,res,xx); }

void pycall(const std::string &fn, gentype &res, int size, const double *x)
{
    dostartup();

    py::list xx(size);

    for ( int i = 0 ; i < size ; ++i )
    {
        xx[i] = py::cast(x[i]);
    }

    pycall_x(fn,res,xx);
}

void pycall(int fni, gentype &res, int size, const double *x)
{
    dostartup();

    py::list xx(size);

    for ( int i = 0 ; i < size ; ++i )
    {
        xx[i] = py::cast(x[i]);
    }

    pycall_x(fni,res,xx);
}

                   void pycall(const std::string &fn, gentype &res,       int                   x) { dostartup(); py::object xx = py::cast(x);         pycall_x(fn,res,xx); }
                   void pycall(const std::string &fn, gentype &res,       double                x) { dostartup(); py::object xx = py::cast(x);         pycall_x(fn,res,xx); }
                   void pycall(const std::string &fn, gentype &res,       std::complex<double>  x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }
                   void pycall(const std::string &fn, gentype &res, const d_anion              &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }
                   void pycall(const std::string &fn, gentype &res, const std::string          &x) { dostartup(); py::object xx = py::cast(x.c_str()); pycall_x(fn,res,xx); }
template <class T> void pycall(const std::string &fn, gentype &res, const Vector<T>            &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }
template <class T> void pycall(const std::string &fn, gentype &res, const Matrix<T>            &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }
template <class T> void pycall(const std::string &fn, gentype &res, const Set<T>               &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }
template <class T> void pycall(const std::string &fn, gentype &res, const Dict<T,dictkey>      &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }
template <class T> void pycall(const std::string &fn, gentype &res, const SparseVector<T>      &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }
                   void pycall(const std::string &fn, gentype &res, const gentype              &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fn,res,xx); }

                   void pycall(int fni, gentype &res,       int                   x) { dostartup(); py::object xx = py::cast(x);         pycall_x(fni,res,xx); }
                   void pycall(int fni, gentype &res,       double                x) { dostartup(); py::object xx = py::cast(x);         pycall_x(fni,res,xx); }
                   void pycall(int fni, gentype &res,       std::complex<double>  x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }
                   void pycall(int fni, gentype &res, const d_anion              &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }
                   void pycall(int fni, gentype &res, const std::string          &x) { dostartup(); py::object xx = py::cast(x.c_str()); pycall_x(fni,res,xx); }
template <class T> void pycall(int fni, gentype &res, const Vector<T>            &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }
template <class T> void pycall(int fni, gentype &res, const Matrix<T>            &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }
template <class T> void pycall(int fni, gentype &res, const Set<T>               &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }
template <class T> void pycall(int fni, gentype &res, const Dict<T,dictkey>      &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }
template <class T> void pycall(int fni, gentype &res, const SparseVector<T>      &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }
                   void pycall(int fni, gentype &res, const gentype              &x) { dostartup(); py::object xx = convToPy(x);         pycall_x(fni,res,xx); }

//void pycall(const std::string &fn, gentype &res, int size, const double *x)
//{
//    dostartup();
//
//    py::object xx = convToPy(size,x);
//    pycall_x(fn,res,xx);
//}




































// CLI-emulation

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

int atexitblock(void (*)(void));
int atexitblock(void (*)(void))
{
    return 0;
}

void dostartup(void)
{
    static bool firstrun = true;

    if ( firstrun )
    {
        void(*xcliCharPrintErr)(char c) = cliCharPrintErr;
        static LoggingOstreamErr clicerr(xcliCharPrintErr);
        seterrstream(&clicerr);

        void(*xcliCharPrintOut)(char c) = cliCharPrintOut;
        static LoggingOstreamOut clicout(xcliCharPrintOut);
        setoutstream(&clicout);

        // atexit doesn't work like we want, so do this instead
        // auto atexitfn = py::module_::import("atexit");
        // atexitfn.attr("register")(py::cpp_function([]() {
        //     // perform cleanup here -- this function is called with the GIL held
        //     //exitgentype();
        //     pyosrcreset();
        //     svm_atexit(nullptr,nullptr,0); // manual cleanup
        // }));
        // force prevent double-call
        //svm_setatexitfn(atexitblock);

        firstrun = false;

        //suppresserrstreamcout();
        pyAllowPrintErr(0);
    }

    return;
}


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

        dostartup();

//        void(*xcliCharPrintErr)(char c) = cliCharPrintErr;
//        static LoggingOstreamErr clicerr(xcliCharPrintErr);
//        seterrstream(&clicerr);
//
//        void(*xcliCharPrintOut)(char c) = cliCharPrintOut;
//        static LoggingOstreamOut clicout(xcliCharPrintOut);
//        setoutstream(&clicout);
//
////DEBUG        suppresserrstreamcout();
////DEBUG        pyAllowPrintErr(0);

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

//        static thread_local int MLInd = 0;
        static thread_local SVMThreadContext *svmContext;
        MEMNEW(svmContext,SVMThreadContext(glob_MLInd()));
        errstream() << "{";

        // Now that everything has been set up so we can run the actual code.

        SparseVector<SparseVector<int> > returntag;

        runsvm(svmContext,commstack,globargvariables,cligetsetExtVar,returntag);

        // Unlock the thread, signalling that the context can be deleted etc

        errstream() << "}";

        MEMDEL(commstack); commstack = nullptr;

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

            deleteMLs();
            deletegrids();
            deleteDIRects();
            deleteNelderMeads();
            deleteBayess();

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
    static int allowprint = 1;
//DEBUG    static int allowprint = 0;

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
            //static int bstring = 0;

            if ( c != '\b' )
            {
                //bstring = 0;

                (*errlog) << c;
                (*errlog).flush();
            }

            //else if ( !bstring )
            //{
            //    bstring = 1;
            //
            //    (*errlog) << '\n';
            //    (*errlog).flush();
            //}
        }
    }

    return;
}














