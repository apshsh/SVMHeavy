
//
// Mutable ML
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ml_mutable.hpp"
#include <iostream>
#include <sstream>
#include <string>
//#ifdef ENABLE_THREADS
//#include <mutex>
//#endif

#include "svm_single.hpp"
#include "svm_binary.hpp"
#include "svm_scalar.hpp"
#include "svm_multic.hpp"
#include "svm_vector.hpp"
#include "svm_anions.hpp"
#include "svm_densit.hpp"
#include "svm_pfront.hpp"
#include "svm_biscor.hpp"
#include "svm_scscor.hpp"
#include "svm_gentyp.hpp"
#include "svm_planar.hpp"
#include "svm_mvrank.hpp"
#include "svm_mulbin.hpp"
#include "svm_cyclic.hpp"
#include "svm_simlrn.hpp"
#include "svm_kconst.hpp"
#include "svm_scalar_rff.hpp"
#include "svm_binary_rff.hpp"
#include "blk_nopnop.hpp"
#include "blk_consen.hpp"
#include "blk_avesca.hpp"
#include "blk_avevec.hpp"
#include "blk_aveani.hpp"
#include "blk_usrfna.hpp"
#include "blk_usrfnb.hpp"
#include "blk_userio.hpp"
#include "blk_calbak.hpp"
#include "blk_mexfna.hpp"
#include "blk_mexfnb.hpp"
#include "blk_mercer.hpp"
#include "blk_conect.hpp"
#include "blk_system.hpp"
#include "blk_kernel.hpp"
#include "blk_bernst.hpp"
#include "blk_batter.hpp"
#include "knn_densit.hpp"
#include "knn_binary.hpp"
#include "knn_multic.hpp"
#include "knn_gentyp.hpp"
#include "knn_scalar.hpp"
#include "knn_vector.hpp"
#include "knn_anions.hpp"
#include "gpr_scalar.hpp"
#include "gpr_vector.hpp"
#include "gpr_anions.hpp"
#include "gpr_gentyp.hpp"
#include "gpr_binary.hpp"
#include "gpr_scalar_rff.hpp"
#include "gpr_binary_rff.hpp"
#include "lsv_scalar.hpp"
#include "lsv_vector.hpp"
#include "lsv_anions.hpp"
#include "lsv_scscor.hpp"
#include "lsv_gentyp.hpp"
#include "lsv_planar.hpp"
#include "lsv_mvrank.hpp"
#include "lsv_binary.hpp"
#include "lsv_scalar_rff.hpp"
#include "imp_expect.hpp"
#include "imp_parsvm.hpp"
#include "imp_rlsamp.hpp"
#include "imp_nlsamp.hpp"
#include "mlm_scalar.hpp"
#include "mlm_binary.hpp"
#include "mlm_vector.hpp"





//
// xfer:    transfers data from src to dest, leaving src empty
// assign:  calls member assign function.  If *dest type does not match it
//          is deleted and a new one constructed prior to calling.  Note
//          that the address pointed to by dest is not changed.
//
// NB: for non-trivially different classes there will be some loss
//     of information in the data transfer.  For example, anomaly
//     classes may be not from SVM_MultiC, and class-wise weights
//     may be lost when moving from a classifier to a regressor.
//

ML_Base &xfer(ML_Base &dest, ML_Base &src);
//Assign defined above because function definition required at that point
//ML_Base &assign(ML_Base **dest, const ML_Base *src, int onlySemiCopy = 0);

void xferInfo(ML_Base &dest, const ML_Base &src);


ML_Mutable::ML_Mutable() : ML_Base()
{
    mlType = 1; // Default to binary SVM classifier
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = makeNewML(mlType);

    isdelable = 1;

    setaltx(nullptr);

    setinterobs(&interobs);

    return;
}

ML_Mutable::ML_Mutable(int type) : ML_Base()
{
    mlType = type;
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = makeNewML(mlType);

    isdelable = 1;

    setaltx(nullptr);

    setinterobs(&interobs);

    return;
}

ML_Mutable::ML_Mutable(const char *dummy, ML_Base *xtheML)
{
    (void) dummy;

    NiceAssert( xtheML );

    mlType = (*xtheML).type();
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = xtheML;

    isdelable = 0; // this prevents deletion of xtheML

    setaltx(nullptr);

    setinterobs(&interobs);

    return;
}

void ML_Mutable::settheMLdirect(ML_Base *xtheML)
{
    resizetheML(0);

    NiceAssert( xtheML );

    mlType = (*xtheML).type();
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = xtheML;

    isdelable = 0; // this prevents deletion of xtheML

    setaltx(nullptr);

    return;
}

ML_Mutable::ML_Mutable(const ML_Mutable &src) : ML_Base()
{
    mlType = 1; // Default to binary SVM classifier
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = makeNewML(mlType);

    isdelable = 1;

    assign(src,0);
    setaltx(nullptr);

    setinterobs(&interobs);

    return;
}

ML_Mutable::ML_Mutable(const ML_Mutable &src, const ML_Base *xsrc) : ML_Base()
{
    mlType = 1; // Default to binary SVM classifier
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = makeNewML(mlType);

    isdelable = 1;

    setaltx(xsrc);
    assign(src,-1);

    setinterobs(&interobs);

    return;
}

ML_Mutable::~ML_Mutable()
{
    resizetheML(0);

    return;
}

void ML_Mutable::setMLTypeMorph(int newmlType)
{
    if ( mlType != newmlType )
    {
        // Assume a simple ML, must deal with general case elsewhere

        mlType = newmlType;
        mlind  = 0;

        resizetheML(1);

        ML_Base *src  = theML("&",mlind);
        ML_Base *dest = makeNewML(mlType);

        xfer(*dest,*src);

        MEMDEL(src); src = nullptr;
    }

    return;
}

void ML_Mutable::setMLTypeClean(int newmlType)
{
//    if ( mlType != newmlType ) - we *want* to trigger reset in this case!
    {
        // Assume a simple ML, must deal with general case elsewhere

        resizetheML(0);  // deletes everything
        theML.resize(1); // naive resize, don't allocate anything yet

        mlType = newmlType;
        mlind  = 0;

        theML("&",mlind) = makeNewML(mlType);
    }

//    else
//    {
//        restart();
//    }

    return;
}

int ML_Mutable::ssetMLTypeMorph(const std::string &xtype)
{
         if ( xtype == "r"   ) { setMLTypeMorph(0);   return 0; }
    else if ( xtype == "c"   ) { setMLTypeMorph(1);   return 0; }
    else if ( xtype == "s"   ) { setMLTypeMorph(2);   return 0; }
    else if ( xtype == "m"   ) { setMLTypeMorph(3);   return 0; }
    else if ( xtype == "v"   ) { setMLTypeMorph(4);   return 0; }
    else if ( xtype == "a"   ) { setMLTypeMorph(5);   return 0; }
    else if ( xtype == "p"   ) { setMLTypeMorph(7);   return 0; }
    else if ( xtype == "t"   ) { setMLTypeMorph(8);   return 0; }
    else if ( xtype == "l"   ) { setMLTypeMorph(12);  return 0; }
    else if ( xtype == "o"   ) { setMLTypeMorph(13);  return 0; }
    else if ( xtype == "g"   ) { setMLTypeMorph(15);  return 0; }
    else if ( xtype == "i"   ) { setMLTypeMorph(16);  return 0; }
    else if ( xtype == "h"   ) { setMLTypeMorph(17);  return 0; }
    else if ( xtype == "j"   ) { setMLTypeMorph(18);  return 0; }
    else if ( xtype == "b"   ) { setMLTypeMorph(19);  return 0; }
    else if ( xtype == "u"   ) { setMLTypeMorph(20);  return 0; }
    else if ( xtype == "d"   ) { setMLTypeMorph(21);  return 0; }
    else if ( xtype == "R"   ) { setMLTypeMorph(22);  return 0; }
    else if ( xtype == "B"   ) { setMLTypeMorph(23);  return 0; }

    else if ( xtype == "knp" ) { setMLTypeMorph(300); return 0; }
    else if ( xtype == "knc" ) { setMLTypeMorph(301); return 0; }
    else if ( xtype == "kng" ) { setMLTypeMorph(302); return 0; }
    else if ( xtype == "knr" ) { setMLTypeMorph(303); return 0; }
    else if ( xtype == "knv" ) { setMLTypeMorph(304); return 0; }
    else if ( xtype == "kna" ) { setMLTypeMorph(305); return 0; }
    else if ( xtype == "knm" ) { setMLTypeMorph(307); return 0; }

    else if ( xtype == "gpr" ) { setMLTypeMorph(400); return 0; }
    else if ( xtype == "gpv" ) { setMLTypeMorph(401); return 0; }
    else if ( xtype == "gpa" ) { setMLTypeMorph(402); return 0; }
    else if ( xtype == "gpg" ) { setMLTypeMorph(408); return 0; }
    else if ( xtype == "gpc" ) { setMLTypeMorph(409); return 0; }
    else if ( xtype == "gpR" ) { setMLTypeMorph(410); return 0; }
    else if ( xtype == "gpC" ) { setMLTypeMorph(411); return 0; }

    else if ( xtype == "mlr" ) { setMLTypeMorph(800); return 0; }
    else if ( xtype == "mlc" ) { setMLTypeMorph(801); return 0; }
    else if ( xtype == "mlv" ) { setMLTypeMorph(802); return 0; }

    else if ( xtype == "lsr" ) { setMLTypeMorph(500); return 0; }
    else if ( xtype == "lsv" ) { setMLTypeMorph(501); return 0; }
    else if ( xtype == "lsa" ) { setMLTypeMorph(502); return 0; }
    else if ( xtype == "lso" ) { setMLTypeMorph(505); return 0; }
    else if ( xtype == "lsg" ) { setMLTypeMorph(508); return 0; }
    else if ( xtype == "lsi" ) { setMLTypeMorph(509); return 0; }
    else if ( xtype == "lsh" ) { setMLTypeMorph(510); return 0; }
    else if ( xtype == "lsc" ) { setMLTypeMorph(511); return 0; }
    else if ( xtype == "lsR" ) { setMLTypeMorph(512); return 0; }

    else if ( xtype == "nop" ) { setMLTypeMorph(200); return 0; }
    else if ( xtype == "con" ) { setMLTypeMorph(201); return 0; }
    else if ( xtype == "fna" ) { setMLTypeMorph(203); return 0; }
    else if ( xtype == "io"  ) { setMLTypeMorph(204); return 0; }
    else if ( xtype == "avr" ) { setMLTypeMorph(202); return 0; }
    else if ( xtype == "avv" ) { setMLTypeMorph(205); return 0; }
    else if ( xtype == "ava" ) { setMLTypeMorph(206); return 0; }
    else if ( xtype == "fnb" ) { setMLTypeMorph(207); return 0; }
    else if ( xtype == "fcb" ) { setMLTypeMorph(208); return 0; }
    else if ( xtype == "mxa" ) { setMLTypeMorph(209); return 0; }
    else if ( xtype == "mxb" ) { setMLTypeMorph(210); return 0; }
    else if ( xtype == "mer" ) { setMLTypeMorph(211); return 0; }
    else if ( xtype == "mba" ) { setMLTypeMorph(212); return 0; }
    else if ( xtype == "sys" ) { setMLTypeMorph(213); return 0; }
    else if ( xtype == "ker" ) { setMLTypeMorph(214); return 0; }
    else if ( xtype == "ber" ) { setMLTypeMorph(215); return 0; }
    else if ( xtype == "bat" ) { setMLTypeMorph(216); return 0; }

    else if ( xtype == "ei"  ) { setMLTypeMorph(600); return 0; }
    else if ( xtype == "svm" ) { setMLTypeMorph(601); return 0; }
    else if ( xtype == "rls" ) { setMLTypeMorph(602); return 0; }
    else if ( xtype == "rns" ) { setMLTypeMorph(603); return 0; }

    return 1;
}

int ML_Mutable::ssetMLTypeClean(const std::string &xtype)
{
         if ( xtype == "r"   ) { setMLTypeClean(0);   return 0; }
    else if ( xtype == "c"   ) { setMLTypeClean(1);   return 0; }
    else if ( xtype == "s"   ) { setMLTypeClean(2);   return 0; }
    else if ( xtype == "m"   ) { setMLTypeClean(3);   return 0; }
    else if ( xtype == "v"   ) { setMLTypeClean(4);   return 0; }
    else if ( xtype == "a"   ) { setMLTypeClean(5);   return 0; }
    else if ( xtype == "p"   ) { setMLTypeClean(7);   return 0; }
    else if ( xtype == "t"   ) { setMLTypeClean(8);   return 0; }
    else if ( xtype == "l"   ) { setMLTypeClean(12);  return 0; }
    else if ( xtype == "o"   ) { setMLTypeClean(13);  return 0; }
    else if ( xtype == "g"   ) { setMLTypeClean(15);  return 0; }
    else if ( xtype == "i"   ) { setMLTypeClean(16);  return 0; }
    else if ( xtype == "h"   ) { setMLTypeClean(17);  return 0; }
    else if ( xtype == "j"   ) { setMLTypeClean(18);  return 0; }
    else if ( xtype == "b"   ) { setMLTypeClean(19);  return 0; }
    else if ( xtype == "u"   ) { setMLTypeClean(20);  return 0; }
    else if ( xtype == "d"   ) { setMLTypeClean(21);  return 0; }
    else if ( xtype == "R"   ) { setMLTypeClean(22);  return 0; }
    else if ( xtype == "B"   ) { setMLTypeClean(23);  return 0; }

    else if ( xtype == "knp" ) { setMLTypeClean(300); return 0; }
    else if ( xtype == "knc" ) { setMLTypeClean(301); return 0; }
    else if ( xtype == "kng" ) { setMLTypeClean(302); return 0; }
    else if ( xtype == "knr" ) { setMLTypeClean(303); return 0; }
    else if ( xtype == "knv" ) { setMLTypeClean(304); return 0; }
    else if ( xtype == "kna" ) { setMLTypeClean(305); return 0; }
    else if ( xtype == "knm" ) { setMLTypeClean(307); return 0; }

    else if ( xtype == "gpr" ) { setMLTypeClean(400); return 0; }
    else if ( xtype == "gpv" ) { setMLTypeClean(401); return 0; }
    else if ( xtype == "gpa" ) { setMLTypeClean(402); return 0; }
    else if ( xtype == "gpg" ) { setMLTypeClean(408); return 0; }
    else if ( xtype == "gpc" ) { setMLTypeClean(409); return 0; }
    else if ( xtype == "gpR" ) { setMLTypeClean(410); return 0; }
    else if ( xtype == "gpC" ) { setMLTypeClean(411); return 0; }

    else if ( xtype == "mlr" ) { setMLTypeClean(800); return 0; }
    else if ( xtype == "mlc" ) { setMLTypeClean(801); return 0; }
    else if ( xtype == "mlv" ) { setMLTypeClean(802); return 0; }

    else if ( xtype == "lsr" ) { setMLTypeClean(500); return 0; }
    else if ( xtype == "lsv" ) { setMLTypeClean(501); return 0; }
    else if ( xtype == "lsa" ) { setMLTypeClean(502); return 0; }
    else if ( xtype == "lso" ) { setMLTypeClean(505); return 0; }
    else if ( xtype == "lsg" ) { setMLTypeClean(508); return 0; }
    else if ( xtype == "lsi" ) { setMLTypeClean(509); return 0; }
    else if ( xtype == "lsh" ) { setMLTypeClean(510); return 0; }
    else if ( xtype == "lsc" ) { setMLTypeClean(511); return 0; }
    else if ( xtype == "lsR" ) { setMLTypeClean(512); return 0; }

    else if ( xtype == "nop" ) { setMLTypeClean(200); return 0; }
    else if ( xtype == "con" ) { setMLTypeClean(201); return 0; }
    else if ( xtype == "fna" ) { setMLTypeClean(203); return 0; }
    else if ( xtype == "io"  ) { setMLTypeClean(204); return 0; }
    else if ( xtype == "avr" ) { setMLTypeClean(202); return 0; }
    else if ( xtype == "avv" ) { setMLTypeClean(205); return 0; }
    else if ( xtype == "ava" ) { setMLTypeClean(206); return 0; }
    else if ( xtype == "fnb" ) { setMLTypeClean(207); return 0; }
    else if ( xtype == "fcb" ) { setMLTypeClean(208); return 0; }
    else if ( xtype == "mxa" ) { setMLTypeClean(209); return 0; }
    else if ( xtype == "mxb" ) { setMLTypeClean(210); return 0; }
    else if ( xtype == "mer" ) { setMLTypeClean(211); return 0; }
    else if ( xtype == "mba" ) { setMLTypeClean(212); return 0; }
    else if ( xtype == "sys" ) { setMLTypeClean(213); return 0; }
    else if ( xtype == "ker" ) { setMLTypeClean(214); return 0; }
    else if ( xtype == "ber" ) { setMLTypeClean(215); return 0; }
    else if ( xtype == "bat" ) { setMLTypeClean(216); return 0; }

    else if ( xtype == "ei"  ) { setMLTypeClean(600); return 0; }
    else if ( xtype == "svm" ) { setMLTypeClean(601); return 0; }
    else if ( xtype == "rls" ) { setMLTypeClean(602); return 0; }
    else if ( xtype == "rns" ) { setMLTypeClean(603); return 0; }

    else if ( xtype == "ser" ) { setMLTypeClean(-2);  return 0; }
    else if ( xtype == "par" ) { setMLTypeClean(-3);  return 0; }

    return 1;
}

const std::string &ML_Mutable::getMLType(void) const
{
         if ( type() == 0   ) { const static std::string res("r");   return res; }
    else if ( type() == 1   ) { const static std::string res("c");   return res; }
    else if ( type() == 2   ) { const static std::string res("s");   return res; }
    else if ( type() == 3   ) { const static std::string res("m");   return res; }
    else if ( type() == 4   ) { const static std::string res("v");   return res; }
    else if ( type() == 5   ) { const static std::string res("a");   return res; }
    else if ( type() == 7   ) { const static std::string res("p");   return res; }
    else if ( type() == 8   ) { const static std::string res("t");   return res; }
    else if ( type() == 12  ) { const static std::string res("l");   return res; }
    else if ( type() == 13  ) { const static std::string res("o");   return res; }
    else if ( type() == 15  ) { const static std::string res("g");   return res; }
    else if ( type() == 16  ) { const static std::string res("i");   return res; }
    else if ( type() == 17  ) { const static std::string res("h");   return res; }
    else if ( type() == 18  ) { const static std::string res("j");   return res; }
    else if ( type() == 19  ) { const static std::string res("b");   return res; }
    else if ( type() == 20  ) { const static std::string res("u");   return res; }
    else if ( type() == 21  ) { const static std::string res("d");   return res; }
    else if ( type() == 22  ) { const static std::string res("R");   return res; }
    else if ( type() == 23  ) { const static std::string res("N");   return res; }

    else if ( type() == 300 ) { const static std::string res("knp"); return res; }
    else if ( type() == 301 ) { const static std::string res("knc"); return res; }
    else if ( type() == 302 ) { const static std::string res("kng"); return res; }
    else if ( type() == 303 ) { const static std::string res("knr"); return res; }
    else if ( type() == 304 ) { const static std::string res("knv"); return res; }
    else if ( type() == 305 ) { const static std::string res("kna"); return res; }
    else if ( type() == 307 ) { const static std::string res("knm"); return res; }

    else if ( type() == 400 ) { const static std::string res("gpr"); return res; }
    else if ( type() == 401 ) { const static std::string res("gpv"); return res; }
    else if ( type() == 402 ) { const static std::string res("gpa"); return res; }
    else if ( type() == 408 ) { const static std::string res("gpg"); return res; }
    else if ( type() == 409 ) { const static std::string res("gpc"); return res; }
    else if ( type() == 410 ) { const static std::string res("gpR"); return res; }
    else if ( type() == 411 ) { const static std::string res("gpC"); return res; }

    else if ( type() == 800 ) { const static std::string res("mlr"); return res; }
    else if ( type() == 801 ) { const static std::string res("mlc"); return res; }
    else if ( type() == 802 ) { const static std::string res("mlv"); return res; }

    else if ( type() == 500 ) { const static std::string res("lsr"); return res; }
    else if ( type() == 501 ) { const static std::string res("lsv"); return res; }
    else if ( type() == 502 ) { const static std::string res("lsa"); return res; }
    else if ( type() == 505 ) { const static std::string res("lso"); return res; }
    else if ( type() == 508 ) { const static std::string res("lsg"); return res; }
    else if ( type() == 509 ) { const static std::string res("lsi"); return res; }
    else if ( type() == 510 ) { const static std::string res("lsh"); return res; }
    else if ( type() == 511 ) { const static std::string res("lsc"); return res; }
    else if ( type() == 512 ) { const static std::string res("lsR"); return res; }

    else if ( type() == 200 ) { const static std::string res("nop"); return res; }
    else if ( type() == 201 ) { const static std::string res("con"); return res; }
    else if ( type() == 202 ) { const static std::string res("avr"); return res; }
    else if ( type() == 203 ) { const static std::string res("fna"); return res; }
    else if ( type() == 204 ) { const static std::string res("io");  return res; }
    else if ( type() == 205 ) { const static std::string res("avv"); return res; }
    else if ( type() == 206 ) { const static std::string res("ava"); return res; }
    else if ( type() == 207 ) { const static std::string res("fnb"); return res; }
    else if ( type() == 208 ) { const static std::string res("fcb"); return res; }
    else if ( type() == 209 ) { const static std::string res("mxa"); return res; }
    else if ( type() == 210 ) { const static std::string res("mxb"); return res; }
    else if ( type() == 211 ) { const static std::string res("mer"); return res; }
    else if ( type() == 212 ) { const static std::string res("mba"); return res; }
    else if ( type() == 213 ) { const static std::string res("sys"); return res; }
    else if ( type() == 214 ) { const static std::string res("ker"); return res; }
    else if ( type() == 215 ) { const static std::string res("ber"); return res; }
    else if ( type() == 216 ) { const static std::string res("bat"); return res; }

    else if ( type() == 600 ) { const static std::string res("ei");  return res; }
    else if ( type() == 601 ) { const static std::string res("svm"); return res; }
    else if ( type() == 602 ) { const static std::string res("rls"); return res; }
    else if ( type() == 603 ) { const static std::string res("rns"); return res; }

    else if ( type() == -2  ) { const static std::string res("ser"); return res; }
    else if ( type() == -3  ) { const static std::string res("par"); return res; }

    const static std::string res("?");
    return res;
}

int ML_Mutable::setVmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "once" ) { setatonce(); return 0; }
        else if ( method == "red"  ) { setredbin(); return 0; }
    }

    return 1;
}

int ML_Mutable::setCmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "1vsA"   ) { set1vsA();    return 0; }
        else if ( method == "1vs1"   ) { set1vs1();    return 0; }
        else if ( method == "DAG"    ) { setDAGSVM();  return 0; }
        else if ( method == "MOC"    ) { setMOC();     return 0; }
        else if ( method == "maxwin" ) { setmaxwins(); return 0; }
        else if ( method == "recdiv" ) { setrecdiv();  return 0; }
    }

    return 1;
}

int ML_Mutable::setOmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "sch" ) { setsingmethod(0); return 0; }
        else if ( method == "tax" ) { setsingmethod(1); return 0; }
    }

    return 1;
}

int ML_Mutable::setAmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "svc" ) { setClassifyViaSVM(); return 0; }
        else if ( method == "svr" ) { setClassifyViaSVR(); return 0; }
    }

    return 1;
}

int ML_Mutable::setRmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "l" ) { setLinearCost();    setusefuzzt(0); return 0; }
        else if ( method == "q" ) { setQuadraticCost(); setusefuzzt(0); return 0; }
        else if ( method == "o" ) { set1NormCost();     setusefuzzt(0); return 0; }
        else if ( method == "g" ) { setLinearCost();    setusefuzzt(1); return 0; }
        else if ( method == "G" ) { setQuadraticCost(); setusefuzzt(1); return 0; }
    }

    return 1;
}

int ML_Mutable::setTmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "f" ) { setFixedTube();  return 0; }
        else if ( method == "s" ) { setShrinkTube(); return 0; }
    }

    return 1;
}

int ML_Mutable::setEmethod(const std::string &method)
{
    if ( isGPR(*this) )
    {
             if ( method == "Naive"   ) { setNaiveConst();    return 0; }
        else if ( method == "EP"      ) { setEPConst();       return 0; }
        else if ( method == "LapNorm" ) { setLaplaceConst(1); return 0; }
        else if ( method == "LapLog"  ) { setLaplaceConst(2); return 0; }
    }

    return 1;
}

int ML_Mutable::setBmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "var" ) { setVarBias();    return 0; }
        else if ( method == "pos" ) { setPosBias();    return 0; }
        else if ( method == "neg" ) { setNegBias();    return 0; }
        else if ( method == "fix" ) { setFixedBias(0); return 0; }
    }

    else if ( isGPR(*this) )
    {
             if ( method == "var" ) { setVarmuBias();  return 0; }
        else if ( method == "fix" ) { setZeromuBias(); return 0; }
    }

    else if ( isLSV(*this) )
    {
             if ( method == "var" ) { setVardelta();  return 0; }
        else if ( method == "fix" ) { setZerodelta(); return 0; }
    }

    return 1;
}

int ML_Mutable::setMmethod(const std::string &method)
{
    if ( isSVM(*this) )
    {
             if ( method == "n" ) { setNoMonotonicConstraints();    return 0; }
        else if ( method == "i" ) { setForcedMonotonicIncreasing(); return 0; }
        else if ( method == "d" ) { setForcedMonotonicDecreasing(); return 0; }
    }

    return 1;
}



const std::string &ML_Mutable::getVmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( isatonce() ) { const static std::string res("once"); return res; }
        else if ( isredbin() ) { const static std::string res("red");  return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getCmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( is1vsA()    ) { const static std::string res("1vsA");   return res; }
        else if ( is1vs1()    ) { const static std::string res("1vs1");   return res; }
        else if ( isDAGSVM()  ) { const static std::string res("DAG");    return res; }
        else if ( isMOC()     ) { const static std::string res("MOC");    return res; }
        else if ( ismaxwins() ) { const static std::string res("maxwin"); return res; }
        else if ( isrecdiv()  ) { const static std::string res("recdiv"); return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getOmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( singmethod() == 0 ) { const static std::string res("sch"); return res; }
        else if ( singmethod() == 1 ) { const static std::string res("tax"); return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getAmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( isClassifyViaSVM() ) { const static std::string res("svc"); return res; }
        else if ( isClassifyViaSVR() ) { const static std::string res("svr"); return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getRmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( isLinearCost()    && ( usefuzzt() == 0 ) ) { const static std::string res("l"); return res; }
        else if ( isQuadraticCost() && ( usefuzzt() == 0 ) ) { const static std::string res("q"); return res; }
        else if ( is1NormCost()     && ( usefuzzt() == 0 ) ) { const static std::string res("o"); return res; }
        else if ( isLinearCost()    && ( usefuzzt() == 1 ) ) { const static std::string res("g"); return res; }
        else if ( isQuadraticCost() && ( usefuzzt() == 1 ) ) { const static std::string res("G"); return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getTmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( isFixedTube()  ) { const static std::string res("f"); return res; }
        else if ( isShrinkTube() ) { const static std::string res("s"); return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getEmethod(void) const
{
    if ( isGPR(*this) )
    {
             if ( isNaiveConst()        ) { const static std::string res("Naive");   return res; }
        else if ( isEPConst()           ) { const static std::string res("EP");      return res; }
        else if ( isLaplaceConst() == 1 ) { const static std::string res("LapNorm"); return res; }
        else if ( isLaplaceConst() == 2 ) { const static std::string res("LapLog");  return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getBmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( isVarBias()   ) { const static std::string res("var"); return res; }
        else if ( isPosBias()   ) { const static std::string res("pos"); return res; }
        else if ( isNegBias()   ) { const static std::string res("neg"); return res; }
        else if ( isFixedBias() ) { const static std::string res("fix"); return res; }
    }

    else if ( isGPR(*this) )
    {
             if ( isVarmuBias()  ) { const static std::string res("var"); return res; }
        else if ( isZeromuBias() ) { const static std::string res("fix"); return res; }
    }

    else if ( isLSV(*this) )
    {
             if ( isVardelta()  ) { const static std::string res("var"); return res; }
        else if ( isZerodelta() ) { const static std::string res("fix"); return res; }
    }

    const static std::string res("?");
    return res;
}

const std::string &ML_Mutable::getMmethod(void) const
{
    if ( isSVM(*this) )
    {
             if ( isNoMonotonicConstraints()    ) { const static std::string res("n"); return res; }
        else if ( isForcedMonotonicIncreasing() ) { const static std::string res("i"); return res; }
        else if ( isForcedMonotonicDecreasing() ) { const static std::string res("d"); return res; }
    }

    const static std::string res("?");
    return res;
}



std::istream &ML_Mutable::inputstream(std::istream &input)
{
    std::string keytype;

    // Mutate the type based on the first word in the stream (the removal
    // of which will not effect the underlying class), then let polymorphism
    // take care of the rest.

    input >> keytype;

    setMLTypeClean(convIDToType(keytype));

    return getML().inputstream(input);
}

void ML_Mutable::resizetheML(int newsize)
{
    NiceAssert( newsize >= 0 );

    int oldsize = theML.size();

    if ( newsize > oldsize )
    {
        theML.resize(newsize);

        int i;
        int newmlType = oldsize ? (*(theML(mlind))).type() : 1;

        for ( i = oldsize ; i < newsize ; ++i )
        {
            theML("&",i) = makeNewML(newmlType);
        }
    }

    else if ( newsize < oldsize )
    {
        int i;

        for ( i = newsize ; i < oldsize ; ++i )
        {
            if ( isdelable )
            {
                MEMDEL(theML("&",i)); theML("&",i) = nullptr;
            }
        }

        theML.resize(newsize);
    }

    return;
}





// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
//
// Begin helper functions
//
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

// IMPORTANT: if the machine is mutable then we are actually dealing
// with class ML_Mutable, although it will return type indicating
// the encased type.  Now, attempting to dynamic cast from ML_Mutable
// to SVM_Scalar (or whatever) will fail at runtime because you can't
// (apparently) safely cast to sibling cast.  To get around this, use
// the funciton getMLconst

int convIDToType(const std::string &keytype)
{
    int type = -1;

         if ( keytype == "SVM_Scalar"     ) { type =   0; }
    else if ( keytype == "SVM_Binary"     ) { type =   1; }
    else if ( keytype == "SVM_Single"     ) { type =   2; }
    else if ( keytype == "SVM_MultiC"     ) { type =   3; }
    else if ( keytype == "SVM_Vector"     ) { type =   4; }
    else if ( keytype == "SVM_Anions"     ) { type =   5; }
    else if ( keytype == "SVM_Densit"     ) { type =   7; }
    else if ( keytype == "SVM_PFront"     ) { type =   8; }
    else if ( keytype == "SVM_BiScor"     ) { type =  12; }
    else if ( keytype == "SVM_ScScor"     ) { type =  13; }
    else if ( keytype == "SVM_Gentyp"     ) { type =  15; }
    else if ( keytype == "SVM_Planar"     ) { type =  16; }
    else if ( keytype == "SVM_MvRank"     ) { type =  17; }
    else if ( keytype == "SVM_MulBin"     ) { type =  18; }
    else if ( keytype == "SVM_SimLrn"     ) { type =  19; }
    else if ( keytype == "SVM_Cyclic"     ) { type =  20; }
    else if ( keytype == "SVM_KConst"     ) { type =  21; }
    else if ( keytype == "SVM_Scalar_rff" ) { type =  22; }
    else if ( keytype == "SVM_Binary_rff" ) { type =  23; }

    else if ( keytype == "BLK_Nopnop" ) { type = 200; }
    else if ( keytype == "BLK_Consen" ) { type = 201; }
    else if ( keytype == "BLK_AveSca" ) { type = 202; }
    else if ( keytype == "BLK_UsrFnA" ) { type = 203; }
    else if ( keytype == "BLK_UserIO" ) { type = 204; }
    else if ( keytype == "BLK_AveVec" ) { type = 205; }
    else if ( keytype == "BLK_AveAni" ) { type = 206; }
    else if ( keytype == "BLK_UsrFnB" ) { type = 207; }
    else if ( keytype == "BLK_CalBak" ) { type = 208; }
    else if ( keytype == "BLK_MexFnA" ) { type = 209; }
    else if ( keytype == "BLK_MexFnA" ) { type = 210; }
    else if ( keytype == "BLK_Mercer" ) { type = 211; }
    else if ( keytype == "BLK_Conect" ) { type = 212; }
    else if ( keytype == "BLK_System" ) { type = 213; }
    else if ( keytype == "BLK_Kernel" ) { type = 214; }
    else if ( keytype == "BLK_Bernst" ) { type = 215; }
    else if ( keytype == "BLK_Batter" ) { type = 216; }

    else if ( keytype == "KNN_Densit" ) { type = 300; }
    else if ( keytype == "KNN_Binary" ) { type = 301; }
    else if ( keytype == "KNN_Gentyp" ) { type = 302; }
    else if ( keytype == "KNN_Scalar" ) { type = 303; }
    else if ( keytype == "KNN_Vector" ) { type = 304; }
    else if ( keytype == "KNN_Anions" ) { type = 305; }
    else if ( keytype == "KNN_MultiC" ) { type = 307; }

    else if ( keytype == "GPR_Scalar"     ) { type = 400; }
    else if ( keytype == "GPR_Vector"     ) { type = 401; }
    else if ( keytype == "GPR_Anions"     ) { type = 402; }
    else if ( keytype == "GPR_Gentyp"     ) { type = 408; }
    else if ( keytype == "GPR_Binary"     ) { type = 409; }
    else if ( keytype == "GPR_Scalar_rff" ) { type = 410; }
    else if ( keytype == "GPR_Binary_rff" ) { type = 411; }

    else if ( keytype == "LSV_Scalar"     ) { type = 500; }
    else if ( keytype == "LSV_Vector"     ) { type = 501; }
    else if ( keytype == "LSV_Anions"     ) { type = 502; }
    else if ( keytype == "LSV_ScScor"     ) { type = 505; }
    else if ( keytype == "LSV_Gentyp"     ) { type = 508; }
    else if ( keytype == "LSV_Planar"     ) { type = 509; }
    else if ( keytype == "LSV_MvRank"     ) { type = 510; }
    else if ( keytype == "LSV_Binary"     ) { type = 511; }
    else if ( keytype == "LSV_Scalar_rff" ) { type = 512; }

    else if ( keytype == "IMP_Expect" ) { type = 600; }
    else if ( keytype == "IMP_ParSVM" ) { type = 601; }
    else if ( keytype == "IMP_RLSamp" ) { type = 602; }
    else if ( keytype == "IMP_NLSamp" ) { type = 603; }

    else if ( keytype == "MLM_Scalar" ) { type = 800; }
    else if ( keytype == "MLM_Binary" ) { type = 801; }
    else if ( keytype == "MLM_Vector" ) { type = 802; }

    else
    {
        NiceThrow("Error: unrecognised ID string");
    }

    return type;
}

int convTypeToID(std::string &res, int id)
{
    switch ( id )
    {
        case  -3: { res = "MercerKernel"; break; }
        case  -2: { res = "KernPrecursor"; break; }
        case  -1: { res = "ML_Base"; break; }

        case   0: { res = "SVM_Scalar";     break; }
        case   1: { res = "SVM_Binary";     break; }
        case   2: { res = "SVM_Single";     break; }
        case   3: { res = "SVM_MultiC";     break; }
        case   4: { res = "SVM_Vector";     break; }
        case   5: { res = "SVM_Anions";     break; }
        case   7: { res = "SVM_Densit";     break; }
        case   8: { res = "SVM_PFront";     break; }
        case  12: { res = "SVM_BiScor";     break; }
        case  13: { res = "SVM_ScScor";     break; }
        case  15: { res = "SVM_Gentyp";     break; }
        case  16: { res = "SVM_Planar";     break; }
        case  17: { res = "SVM_MvRank";     break; }
        case  18: { res = "SVM_MulBin";     break; }
        case  19: { res = "SVM_SimLrn";     break; }
        case  20: { res = "SVM_Cyclic";     break; }
        case  21: { res = "SVM_KConst";     break; }
        case  22: { res = "SVM_Scalar_rff"; break; }
        case  23: { res = "SVM_Binary_rff"; break; }

        case 200: { res = "BLK_Nopnop"; break; }
        case 201: { res = "BLK_Consen"; break; }
        case 202: { res = "BLK_AveSca"; break; }
        case 203: { res = "BLK_UsrFnA"; break; }
        case 204: { res = "BLK_UserIO"; break; }
        case 205: { res = "BLK_AveVec"; break; }
        case 206: { res = "BLK_AveAni"; break; }
        case 207: { res = "BLK_UsrFnB"; break; }
        case 208: { res = "BLK_CalBak"; break; }
        case 209: { res = "BLK_MexFnA"; break; }
        case 210: { res = "BLK_MexFnB"; break; }
        case 211: { res = "BLK_Mercer"; break; }
        case 212: { res = "BLK_Conect"; break; }
        case 213: { res = "BLK_System"; break; }
        case 214: { res = "BLK_Kernel"; break; }
        case 215: { res = "BLK_Bernst"; break; }
        case 216: { res = "BLK_Batter"; break; }

        case 300: { res = "KNN_Densit"; break; }
        case 301: { res = "KNN_Binary"; break; }
        case 302: { res = "KNN_Gentyp"; break; }
        case 303: { res = "KNN_Scalar"; break; }
        case 304: { res = "KNN_Vector"; break; }
        case 305: { res = "KNN_Anions"; break; }
        case 307: { res = "KNN_MultiC"; break; }

        case 400: { res = "GPR_Scalar";     break; }
        case 401: { res = "GPR_Vector";     break; }
        case 402: { res = "GPR_Anions";     break; }
        case 408: { res = "GPR_Gentyp";     break; }
        case 409: { res = "GPR_Binary";     break; }
        case 410: { res = "GPR_Scalar_rff"; break; }
        case 411: { res = "GPR_Binary_rff"; break; }

        case 500: { res = "LSV_Scalar";     break; }
        case 501: { res = "LSV_Vector";     break; }
        case 502: { res = "LSV_Anions";     break; }
        case 505: { res = "LSV_ScScor";     break; }
        case 508: { res = "LSV_Gentyp";     break; }
        case 509: { res = "LSV_Planar";     break; }
        case 510: { res = "LSV_MvRank";     break; }
        case 511: { res = "LSV_Binary";     break; }
        case 512: { res = "LSV_Scalar_rff"; break; }

        case 600: { res = "IMP_Expect"; break; }
        case 601: { res = "IMP_ParSVM"; break; }
        case 602: { res = "IMP_RLSamp"; break; }
        case 603: { res = "IMP_NLSamp"; break; }

        case 800: { res = "MLM_Scalar"; break; }
        case 801: { res = "MLM_Binary"; break; }
        case 802: { res = "MLM_Vector"; break; }

        default:
        {
            NiceThrow("Error: unrecognised ID");
            break;
        }
    }

    return id;
}

ML_Base *makeNewML(int type, int subtype)
{
    ML_Base *res = nullptr;

    switch ( type )
    {
        case   0: { MEMNEW(res,SVM_Scalar    ()); break; }
        case   1: { MEMNEW(res,SVM_Binary    ()); break; }
        case   2: { MEMNEW(res,SVM_Single    ()); break; }
        case   3: { MEMNEW(res,SVM_MultiC    ()); break; }
        case   4: { MEMNEW(res,SVM_Vector    ()); break; }
        case   5: { MEMNEW(res,SVM_Anions    ()); break; }
        case   7: { MEMNEW(res,SVM_Densit    ()); break; }
        case   8: { MEMNEW(res,SVM_PFront    ()); break; }
        case  12: { MEMNEW(res,SVM_BiScor    ()); break; }
        case  13: { MEMNEW(res,SVM_ScScor    ()); break; }
        case  15: { MEMNEW(res,SVM_Gentyp    ()); break; }
        case  16: { MEMNEW(res,SVM_Planar    ()); break; }
        case  17: { MEMNEW(res,SVM_MvRank    ()); break; }
        case  18: { MEMNEW(res,SVM_MulBin    ()); break; }
        case  19: { MEMNEW(res,SVM_SimLrn    ()); break; }
        case  20: { MEMNEW(res,SVM_Cyclic    ()); break; }
        case  21: { MEMNEW(res,SVM_KConst    ()); break; }
        case  22: { MEMNEW(res,SVM_Scalar_rff()); break; }
        case  23: { MEMNEW(res,SVM_Binary_rff()); break; }

        case 200: { MEMNEW(res,BLK_Nopnop()); break; }
        case 201: { MEMNEW(res,BLK_Consen()); break; }
        case 202: { MEMNEW(res,BLK_AveSca()); break; }
        case 203: { MEMNEW(res,BLK_UsrFnA()); break; }
        case 204: { MEMNEW(res,BLK_UserIO()); break; }
        case 205: { MEMNEW(res,BLK_AveVec()); break; }
        case 206: { MEMNEW(res,BLK_AveAni()); break; }
        case 207: { MEMNEW(res,BLK_UsrFnB()); break; }
        case 208: { MEMNEW(res,BLK_CalBak()); break; }
        case 209: { MEMNEW(res,BLK_MexFnA()); break; }
        case 210: { MEMNEW(res,BLK_MexFnB()); break; }
        case 211: { MEMNEW(res,BLK_Mercer()); break; }
        case 212: { MEMNEW(res,BLK_Conect()); break; }
        case 213: { MEMNEW(res,BLK_System()); break; }
        case 214: { MEMNEW(res,BLK_Kernel()); break; }
        case 215: { MEMNEW(res,BLK_Bernst()); break; }
        case 216: { MEMNEW(res,BLK_Batter()); break; }

        case 300: { MEMNEW(res,KNN_Densit()); break; }
        case 301: { MEMNEW(res,KNN_Binary()); break; }
        case 302: { MEMNEW(res,KNN_Gentyp()); break; }
        case 303: { MEMNEW(res,KNN_Scalar()); break; }
        case 304: { MEMNEW(res,KNN_Vector()); break; }
        case 305: { MEMNEW(res,KNN_Anions()); break; }
        case 307: { MEMNEW(res,KNN_MultiC()); break; }

        case 400: { MEMNEW(res,GPR_Scalar    ()); break; }
        case 401: { MEMNEW(res,GPR_Vector    ()); break; }
        case 402: { MEMNEW(res,GPR_Anions    ()); break; }
        case 408: { MEMNEW(res,GPR_Gentyp    ()); break; }
        case 409: { MEMNEW(res,GPR_Binary    ()); break; }
        case 410: { MEMNEW(res,GPR_Scalar_rff()); break; }
        case 411: { MEMNEW(res,GPR_Binary_rff()); break; }

        case 500: { MEMNEW(res,LSV_Scalar    ()); break; }
        case 501: { MEMNEW(res,LSV_Vector    ()); break; }
        case 502: { MEMNEW(res,LSV_Anions    ()); break; }
        case 505: { MEMNEW(res,LSV_ScScor    ()); break; }
        case 508: { MEMNEW(res,LSV_Gentyp    ()); break; }
        case 509: { MEMNEW(res,LSV_Planar    ()); break; }
        case 510: { MEMNEW(res,LSV_MvRank    ()); break; }
        case 511: { MEMNEW(res,LSV_Binary    ()); break; }
        case 512: { MEMNEW(res,LSV_Scalar_rff()); break; }

        case 600: { MEMNEW(res,IMP_Expect()); break; }
        case 601: { MEMNEW(res,IMP_ParSVM()); break; }
        case 602: { MEMNEW(res,IMP_RLSamp()); break; }
        case 603: { MEMNEW(res,IMP_NLSamp()); break; }

        case 800: { MEMNEW(res,MLM_Scalar()); break; }
        case 801: { MEMNEW(res,MLM_Binary()); break; }
        case 802: { MEMNEW(res,MLM_Vector()); break; }

        default: { NiceThrow("Error: type unknown in makeNewML."); break; }
    }

    NiceAssert(res);

    if ( subtype != -42 )
    {
        res->setsubtype(subtype);
    }

    return res;
}

ML_Base *makeDupML(const ML_Base &src, const ML_Base *srcx)
{
  ML_Base *res = nullptr;
  int type = src.type();

  if ( srcx )
  {
    switch ( type )
    {
        case   0: { MEMNEW(res,SVM_Scalar    (dynamic_cast<const SVM_Scalar     &>(src.getMLconst()),srcx)); break; }
        case   1: { MEMNEW(res,SVM_Binary    (dynamic_cast<const SVM_Binary     &>(src.getMLconst()),srcx)); break; }
        case   2: { MEMNEW(res,SVM_Single    (dynamic_cast<const SVM_Single     &>(src.getMLconst()),srcx)); break; }
        case   3: { MEMNEW(res,SVM_MultiC    (dynamic_cast<const SVM_MultiC     &>(src.getMLconst()),srcx)); break; }
        case   4: { MEMNEW(res,SVM_Vector    (dynamic_cast<const SVM_Vector     &>(src.getMLconst()),srcx)); break; }
        case   5: { MEMNEW(res,SVM_Anions    (dynamic_cast<const SVM_Anions     &>(src.getMLconst()),srcx)); break; }
        case   7: { MEMNEW(res,SVM_Densit    (dynamic_cast<const SVM_Densit     &>(src.getMLconst()),srcx)); break; }
        case   8: { MEMNEW(res,SVM_PFront    (dynamic_cast<const SVM_PFront     &>(src.getMLconst()),srcx)); break; }
        case  12: { MEMNEW(res,SVM_BiScor    (dynamic_cast<const SVM_BiScor     &>(src.getMLconst()),srcx)); break; }
        case  13: { MEMNEW(res,SVM_ScScor    (dynamic_cast<const SVM_ScScor     &>(src.getMLconst()),srcx)); break; }
        case  15: { MEMNEW(res,SVM_Gentyp    (dynamic_cast<const SVM_Gentyp     &>(src.getMLconst()),srcx)); break; }
        case  16: { MEMNEW(res,SVM_Planar    (dynamic_cast<const SVM_Planar     &>(src.getMLconst()),srcx)); break; }
        case  17: { MEMNEW(res,SVM_MvRank    (dynamic_cast<const SVM_MvRank     &>(src.getMLconst()),srcx)); break; }
        case  18: { MEMNEW(res,SVM_MulBin    (dynamic_cast<const SVM_MulBin     &>(src.getMLconst()),srcx)); break; }
        case  19: { MEMNEW(res,SVM_SimLrn    (dynamic_cast<const SVM_SimLrn     &>(src.getMLconst()),srcx)); break; }
        case  20: { MEMNEW(res,SVM_Cyclic    (dynamic_cast<const SVM_Cyclic     &>(src.getMLconst()),srcx)); break; }
        case  21: { MEMNEW(res,SVM_KConst    (dynamic_cast<const SVM_KConst     &>(src.getMLconst()),srcx)); break; }
        case  22: { MEMNEW(res,SVM_Scalar_rff(dynamic_cast<const SVM_Scalar_rff &>(src.getMLconst()),srcx)); break; }
        case  23: { MEMNEW(res,SVM_Binary_rff(dynamic_cast<const SVM_Binary_rff &>(src.getMLconst()),srcx)); break; }

        case 200: { MEMNEW(res,BLK_Nopnop(dynamic_cast<const BLK_Nopnop &>(src.getMLconst()),srcx)); break; }
        case 201: { MEMNEW(res,BLK_Consen(dynamic_cast<const BLK_Consen &>(src.getMLconst()),srcx)); break; }
        case 202: { MEMNEW(res,BLK_AveSca(dynamic_cast<const BLK_AveSca &>(src.getMLconst()),srcx)); break; }
        case 203: { MEMNEW(res,BLK_UsrFnA(dynamic_cast<const BLK_UsrFnA &>(src.getMLconst()),srcx)); break; }
        case 204: { MEMNEW(res,BLK_UserIO(dynamic_cast<const BLK_UserIO &>(src.getMLconst()),srcx)); break; }
        case 205: { MEMNEW(res,BLK_AveVec(dynamic_cast<const BLK_AveVec &>(src.getMLconst()),srcx)); break; }
        case 206: { MEMNEW(res,BLK_AveAni(dynamic_cast<const BLK_AveAni &>(src.getMLconst()),srcx)); break; }
        case 207: { MEMNEW(res,BLK_UsrFnB(dynamic_cast<const BLK_UsrFnB &>(src.getMLconst()),srcx)); break; }
        case 208: { MEMNEW(res,BLK_CalBak(dynamic_cast<const BLK_CalBak &>(src.getMLconst()),srcx)); break; }
        case 209: { MEMNEW(res,BLK_MexFnA(dynamic_cast<const BLK_MexFnA &>(src.getMLconst()),srcx)); break; }
        case 210: { MEMNEW(res,BLK_MexFnB(dynamic_cast<const BLK_MexFnB &>(src.getMLconst()),srcx)); break; }
        case 211: { MEMNEW(res,BLK_Mercer(dynamic_cast<const BLK_Mercer &>(src.getMLconst()),srcx)); break; }
        case 212: { MEMNEW(res,BLK_Conect(dynamic_cast<const BLK_Conect &>(src.getMLconst()),srcx)); break; }
        case 213: { MEMNEW(res,BLK_System(dynamic_cast<const BLK_System &>(src.getMLconst()),srcx)); break; }
        case 214: { MEMNEW(res,BLK_Kernel(dynamic_cast<const BLK_Kernel &>(src.getMLconst()),srcx)); break; }
        case 215: { MEMNEW(res,BLK_Bernst(dynamic_cast<const BLK_Bernst &>(src.getMLconst()),srcx)); break; }
        case 216: { MEMNEW(res,BLK_Batter(dynamic_cast<const BLK_Batter &>(src.getMLconst()),srcx)); break; }

        case 300: { MEMNEW(res,KNN_Densit(dynamic_cast<const KNN_Densit &>(src.getMLconst()),srcx)); break; }
        case 301: { MEMNEW(res,KNN_Binary(dynamic_cast<const KNN_Binary &>(src.getMLconst()),srcx)); break; }
        case 302: { MEMNEW(res,KNN_Gentyp(dynamic_cast<const KNN_Gentyp &>(src.getMLconst()),srcx)); break; }
        case 303: { MEMNEW(res,KNN_Scalar(dynamic_cast<const KNN_Scalar &>(src.getMLconst()),srcx)); break; }
        case 304: { MEMNEW(res,KNN_Vector(dynamic_cast<const KNN_Vector &>(src.getMLconst()),srcx)); break; }
        case 305: { MEMNEW(res,KNN_Anions(dynamic_cast<const KNN_Anions &>(src.getMLconst()),srcx)); break; }
        case 307: { MEMNEW(res,KNN_MultiC(dynamic_cast<const KNN_MultiC &>(src.getMLconst()),srcx)); break; }

        case 400: { MEMNEW(res,GPR_Scalar    (dynamic_cast<const GPR_Scalar     &>(src.getMLconst()),srcx)); break; }
        case 401: { MEMNEW(res,GPR_Vector    (dynamic_cast<const GPR_Vector     &>(src.getMLconst()),srcx)); break; }
        case 402: { MEMNEW(res,GPR_Anions    (dynamic_cast<const GPR_Anions     &>(src.getMLconst()),srcx)); break; }
        case 408: { MEMNEW(res,GPR_Gentyp    (dynamic_cast<const GPR_Gentyp     &>(src.getMLconst()),srcx)); break; }
        case 409: { MEMNEW(res,GPR_Binary    (dynamic_cast<const GPR_Binary     &>(src.getMLconst()),srcx)); break; }
        case 410: { MEMNEW(res,GPR_Scalar_rff(dynamic_cast<const GPR_Scalar_rff &>(src.getMLconst()),srcx)); break; }
        case 411: { MEMNEW(res,GPR_Binary_rff(dynamic_cast<const GPR_Binary_rff &>(src.getMLconst()),srcx)); break; }

        case 500: { MEMNEW(res,LSV_Scalar    (dynamic_cast<const LSV_Scalar     &>(src.getMLconst()),srcx)); break; }
        case 501: { MEMNEW(res,LSV_Vector    (dynamic_cast<const LSV_Vector     &>(src.getMLconst()),srcx)); break; }
        case 502: { MEMNEW(res,LSV_Anions    (dynamic_cast<const LSV_Anions     &>(src.getMLconst()),srcx)); break; }
        case 505: { MEMNEW(res,LSV_ScScor    (dynamic_cast<const LSV_ScScor     &>(src.getMLconst()),srcx)); break; }
        case 508: { MEMNEW(res,LSV_Gentyp    (dynamic_cast<const LSV_Gentyp     &>(src.getMLconst()),srcx)); break; }
        case 509: { MEMNEW(res,LSV_Planar    (dynamic_cast<const LSV_Planar     &>(src.getMLconst()),srcx)); break; }
        case 510: { MEMNEW(res,LSV_MvRank    (dynamic_cast<const LSV_MvRank     &>(src.getMLconst()),srcx)); break; }
        case 511: { MEMNEW(res,LSV_Binary    (dynamic_cast<const LSV_Binary     &>(src.getMLconst()),srcx)); break; }
        case 512: { MEMNEW(res,LSV_Scalar_rff(dynamic_cast<const LSV_Scalar_rff &>(src.getMLconst()),srcx)); break; }

        case 600: { MEMNEW(res,IMP_Expect(dynamic_cast<const IMP_Expect &>(src.getMLconst()),srcx)); break; }
        case 601: { MEMNEW(res,IMP_ParSVM(dynamic_cast<const IMP_ParSVM &>(src.getMLconst()),srcx)); break; }
        case 602: { MEMNEW(res,IMP_RLSamp(dynamic_cast<const IMP_RLSamp &>(src.getMLconst()),srcx)); break; }
        case 603: { MEMNEW(res,IMP_NLSamp(dynamic_cast<const IMP_NLSamp &>(src.getMLconst()),srcx)); break; }

        case 800: { MEMNEW(res,MLM_Scalar(dynamic_cast<const MLM_Scalar &>(src.getMLconst()),srcx)); break; }
        case 801: { MEMNEW(res,MLM_Binary(dynamic_cast<const MLM_Binary &>(src.getMLconst()),srcx)); break; }
        case 802: { MEMNEW(res,MLM_Vector(dynamic_cast<const MLM_Vector &>(src.getMLconst()),srcx)); break; }

        default: { NiceThrow("Error: type unknown in makeNewML."); break; }
    }
  }

  else
  {
    switch ( type )
    {
        case   0: { MEMNEW(res,SVM_Scalar    (dynamic_cast<const SVM_Scalar     &>(src.getMLconst()))); break; }
        case   1: { MEMNEW(res,SVM_Binary    (dynamic_cast<const SVM_Binary     &>(src.getMLconst()))); break; }
        case   2: { MEMNEW(res,SVM_Single    (dynamic_cast<const SVM_Single     &>(src.getMLconst()))); break; }
        case   3: { MEMNEW(res,SVM_MultiC    (dynamic_cast<const SVM_MultiC     &>(src.getMLconst()))); break; }
        case   4: { MEMNEW(res,SVM_Vector    (dynamic_cast<const SVM_Vector     &>(src.getMLconst()))); break; }
        case   5: { MEMNEW(res,SVM_Anions    (dynamic_cast<const SVM_Anions     &>(src.getMLconst()))); break; }
        case   7: { MEMNEW(res,SVM_Densit    (dynamic_cast<const SVM_Densit     &>(src.getMLconst()))); break; }
        case   8: { MEMNEW(res,SVM_PFront    (dynamic_cast<const SVM_PFront     &>(src.getMLconst()))); break; }
        case  12: { MEMNEW(res,SVM_BiScor    (dynamic_cast<const SVM_BiScor     &>(src.getMLconst()))); break; }
        case  13: { MEMNEW(res,SVM_ScScor    (dynamic_cast<const SVM_ScScor     &>(src.getMLconst()))); break; }
        case  15: { MEMNEW(res,SVM_Gentyp    (dynamic_cast<const SVM_Gentyp     &>(src.getMLconst()))); break; }
        case  16: { MEMNEW(res,SVM_Planar    (dynamic_cast<const SVM_Planar     &>(src.getMLconst()))); break; }
        case  17: { MEMNEW(res,SVM_MvRank    (dynamic_cast<const SVM_MvRank     &>(src.getMLconst()))); break; }
        case  18: { MEMNEW(res,SVM_MulBin    (dynamic_cast<const SVM_MulBin     &>(src.getMLconst()))); break; }
        case  19: { MEMNEW(res,SVM_SimLrn    (dynamic_cast<const SVM_SimLrn     &>(src.getMLconst()))); break; }
        case  20: { MEMNEW(res,SVM_Cyclic    (dynamic_cast<const SVM_Cyclic     &>(src.getMLconst()))); break; }
        case  21: { MEMNEW(res,SVM_KConst    (dynamic_cast<const SVM_KConst     &>(src.getMLconst()))); break; }
        case  22: { MEMNEW(res,SVM_Scalar_rff(dynamic_cast<const SVM_Scalar_rff &>(src.getMLconst()))); break; }
        case  23: { MEMNEW(res,SVM_Binary_rff(dynamic_cast<const SVM_Binary_rff &>(src.getMLconst()))); break; }

        case 200: { MEMNEW(res,BLK_Nopnop(dynamic_cast<const BLK_Nopnop &>(src.getMLconst()))); break; }
        case 201: { MEMNEW(res,BLK_Consen(dynamic_cast<const BLK_Consen &>(src.getMLconst()))); break; }
        case 202: { MEMNEW(res,BLK_AveSca(dynamic_cast<const BLK_AveSca &>(src.getMLconst()))); break; }
        case 203: { MEMNEW(res,BLK_UsrFnA(dynamic_cast<const BLK_UsrFnA &>(src.getMLconst()))); break; }
        case 204: { MEMNEW(res,BLK_UserIO(dynamic_cast<const BLK_UserIO &>(src.getMLconst()))); break; }
        case 205: { MEMNEW(res,BLK_AveVec(dynamic_cast<const BLK_AveVec &>(src.getMLconst()))); break; }
        case 206: { MEMNEW(res,BLK_AveAni(dynamic_cast<const BLK_AveAni &>(src.getMLconst()))); break; }
        case 207: { MEMNEW(res,BLK_UsrFnB(dynamic_cast<const BLK_UsrFnB &>(src.getMLconst()))); break; }
        case 208: { MEMNEW(res,BLK_CalBak(dynamic_cast<const BLK_CalBak &>(src.getMLconst()))); break; }
        case 209: { MEMNEW(res,BLK_MexFnA(dynamic_cast<const BLK_MexFnA &>(src.getMLconst()))); break; }
        case 210: { MEMNEW(res,BLK_MexFnB(dynamic_cast<const BLK_MexFnB &>(src.getMLconst()))); break; }
        case 211: { MEMNEW(res,BLK_Mercer(dynamic_cast<const BLK_Mercer &>(src.getMLconst()))); break; }
        case 212: { MEMNEW(res,BLK_Conect(dynamic_cast<const BLK_Conect &>(src.getMLconst()))); break; }
        case 213: { MEMNEW(res,BLK_System(dynamic_cast<const BLK_System &>(src.getMLconst()))); break; }
        case 214: { MEMNEW(res,BLK_Kernel(dynamic_cast<const BLK_Kernel &>(src.getMLconst()))); break; }
        case 215: { MEMNEW(res,BLK_Bernst(dynamic_cast<const BLK_Bernst &>(src.getMLconst()))); break; }
        case 216: { MEMNEW(res,BLK_Batter(dynamic_cast<const BLK_Batter &>(src.getMLconst()))); break; }

        case 300: { MEMNEW(res,KNN_Densit(dynamic_cast<const KNN_Densit &>(src.getMLconst()))); break; }
        case 301: { MEMNEW(res,KNN_Binary(dynamic_cast<const KNN_Binary &>(src.getMLconst()))); break; }
        case 302: { MEMNEW(res,KNN_Gentyp(dynamic_cast<const KNN_Gentyp &>(src.getMLconst()))); break; }
        case 303: { MEMNEW(res,KNN_Scalar(dynamic_cast<const KNN_Scalar &>(src.getMLconst()))); break; }
        case 304: { MEMNEW(res,KNN_Vector(dynamic_cast<const KNN_Vector &>(src.getMLconst()))); break; }
        case 305: { MEMNEW(res,KNN_Anions(dynamic_cast<const KNN_Anions &>(src.getMLconst()))); break; }
        case 307: { MEMNEW(res,KNN_MultiC(dynamic_cast<const KNN_MultiC &>(src.getMLconst()))); break; }

        case 400: { MEMNEW(res,GPR_Scalar    (dynamic_cast<const GPR_Scalar     &>(src.getMLconst()))); break; }
        case 401: { MEMNEW(res,GPR_Vector    (dynamic_cast<const GPR_Vector     &>(src.getMLconst()))); break; }
        case 402: { MEMNEW(res,GPR_Anions    (dynamic_cast<const GPR_Anions     &>(src.getMLconst()))); break; }
        case 408: { MEMNEW(res,GPR_Gentyp    (dynamic_cast<const GPR_Gentyp     &>(src.getMLconst()))); break; }
        case 409: { MEMNEW(res,GPR_Binary    (dynamic_cast<const GPR_Binary     &>(src.getMLconst()))); break; }
        case 410: { MEMNEW(res,GPR_Scalar_rff(dynamic_cast<const GPR_Scalar_rff &>(src.getMLconst()))); break; }
        case 411: { MEMNEW(res,GPR_Binary_rff(dynamic_cast<const GPR_Binary_rff &>(src.getMLconst()))); break; }

        case 500: { MEMNEW(res,LSV_Scalar    (dynamic_cast<const LSV_Scalar     &>(src.getMLconst()))); break; }
        case 501: { MEMNEW(res,LSV_Vector    (dynamic_cast<const LSV_Vector     &>(src.getMLconst()))); break; }
        case 502: { MEMNEW(res,LSV_Anions    (dynamic_cast<const LSV_Anions     &>(src.getMLconst()))); break; }
        case 505: { MEMNEW(res,LSV_ScScor    (dynamic_cast<const LSV_ScScor     &>(src.getMLconst()))); break; }
        case 508: { MEMNEW(res,LSV_Gentyp    (dynamic_cast<const LSV_Gentyp     &>(src.getMLconst()))); break; }
        case 509: { MEMNEW(res,LSV_Planar    (dynamic_cast<const LSV_Planar     &>(src.getMLconst()))); break; }
        case 510: { MEMNEW(res,LSV_MvRank    (dynamic_cast<const LSV_MvRank     &>(src.getMLconst()))); break; }
        case 511: { MEMNEW(res,LSV_Binary    (dynamic_cast<const LSV_Binary     &>(src.getMLconst()))); break; }
        case 512: { MEMNEW(res,LSV_Scalar_rff(dynamic_cast<const LSV_Scalar_rff &>(src.getMLconst()))); break; }

        case 600: { MEMNEW(res,IMP_Expect(dynamic_cast<const IMP_Expect &>(src.getMLconst()))); break; }
        case 601: { MEMNEW(res,IMP_ParSVM(dynamic_cast<const IMP_ParSVM &>(src.getMLconst()))); break; }
        case 602: { MEMNEW(res,IMP_RLSamp(dynamic_cast<const IMP_RLSamp &>(src.getMLconst()))); break; }
        case 603: { MEMNEW(res,IMP_NLSamp(dynamic_cast<const IMP_NLSamp &>(src.getMLconst()))); break; }

        case 800: { MEMNEW(res,MLM_Scalar(dynamic_cast<const MLM_Scalar &>(src.getMLconst()))); break; }
        case 801: { MEMNEW(res,MLM_Binary(dynamic_cast<const MLM_Binary &>(src.getMLconst()))); break; }
        case 802: { MEMNEW(res,MLM_Vector(dynamic_cast<const MLM_Vector &>(src.getMLconst()))); break; }

        default: { NiceThrow("Error: type unknown in makeNewML."); break; }
    }
  }

  NiceAssert(res);

  return res;
}

ML_Base &assign(ML_Base **dest, const ML_Base *src, int onlySemiCopy)
{
    if ( (*dest)->type() != src->type() )
    {
        MEMDEL(*dest); *dest = nullptr;
        *dest = makeNewML(src->type());
    }

    switch ( src->type() )
    {
        case   0: { dynamic_cast<SVM_Scalar &>((**dest).getML()).assign(dynamic_cast<const SVM_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case   1: { dynamic_cast<SVM_Binary &>((**dest).getML()).assign(dynamic_cast<const SVM_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case   2: { dynamic_cast<SVM_Single &>((**dest).getML()).assign(dynamic_cast<const SVM_Single &>((*src).getMLconst()),onlySemiCopy); break; }
        case   3: { dynamic_cast<SVM_MultiC &>((**dest).getML()).assign(dynamic_cast<const SVM_MultiC &>((*src).getMLconst()),onlySemiCopy); break; }
        case   4: { dynamic_cast<SVM_Vector &>((**dest).getML()).assign(dynamic_cast<const SVM_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case   5: { dynamic_cast<SVM_Anions &>((**dest).getML()).assign(dynamic_cast<const SVM_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case   7: { dynamic_cast<SVM_Densit &>((**dest).getML()).assign(dynamic_cast<const SVM_Densit &>((*src).getMLconst()),onlySemiCopy); break; }
        case   8: { dynamic_cast<SVM_PFront &>((**dest).getML()).assign(dynamic_cast<const SVM_PFront &>((*src).getMLconst()),onlySemiCopy); break; }
        case  12: { dynamic_cast<SVM_BiScor &>((**dest).getML()).assign(dynamic_cast<const SVM_BiScor &>((*src).getMLconst()),onlySemiCopy); break; }
        case  13: { dynamic_cast<SVM_ScScor &>((**dest).getML()).assign(dynamic_cast<const SVM_ScScor &>((*src).getMLconst()),onlySemiCopy); break; }
        case  15: { dynamic_cast<SVM_Gentyp &>((**dest).getML()).assign(dynamic_cast<const SVM_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }
        case  16: { dynamic_cast<SVM_Planar &>((**dest).getML()).assign(dynamic_cast<const SVM_Planar &>((*src).getMLconst()),onlySemiCopy); break; }
        case  17: { dynamic_cast<SVM_MvRank &>((**dest).getML()).assign(dynamic_cast<const SVM_MvRank &>((*src).getMLconst()),onlySemiCopy); break; }
        case  18: { dynamic_cast<SVM_MulBin &>((**dest).getML()).assign(dynamic_cast<const SVM_MulBin &>((*src).getMLconst()),onlySemiCopy); break; }
        case  19: { dynamic_cast<SVM_SimLrn &>((**dest).getML()).assign(dynamic_cast<const SVM_SimLrn &>((*src).getMLconst()),onlySemiCopy); break; }
        case  20: { dynamic_cast<SVM_Cyclic &>((**dest).getML()).assign(dynamic_cast<const SVM_Cyclic &>((*src).getMLconst()),onlySemiCopy); break; }
        case  21: { dynamic_cast<SVM_KConst &>((**dest).getML()).assign(dynamic_cast<const SVM_KConst &>((*src).getMLconst()),onlySemiCopy); break; }
        case  22: { dynamic_cast<SVM_Scalar_rff &>((**dest).getML()).assign(dynamic_cast<const SVM_Scalar_rff &>((*src).getMLconst()),onlySemiCopy); break; }
        case  23: { dynamic_cast<SVM_Binary_rff &>((**dest).getML()).assign(dynamic_cast<const SVM_Binary_rff &>((*src).getMLconst()),onlySemiCopy); break; }

        case 200: { dynamic_cast<BLK_Nopnop &>((**dest).getML()).assign(dynamic_cast<const BLK_Nopnop &>((*src).getMLconst()),onlySemiCopy); break; }
        case 201: { dynamic_cast<BLK_Consen &>((**dest).getML()).assign(dynamic_cast<const BLK_Consen &>((*src).getMLconst()),onlySemiCopy); break; }
        case 202: { dynamic_cast<BLK_AveSca &>((**dest).getML()).assign(dynamic_cast<const BLK_AveSca &>((*src).getMLconst()),onlySemiCopy); break; }
        case 203: { dynamic_cast<BLK_UsrFnA &>((**dest).getML()).assign(dynamic_cast<const BLK_UsrFnA &>((*src).getMLconst()),onlySemiCopy); break; }
        case 204: { dynamic_cast<BLK_UserIO &>((**dest).getML()).assign(dynamic_cast<const BLK_UserIO &>((*src).getMLconst()),onlySemiCopy); break; }
        case 205: { dynamic_cast<BLK_AveVec &>((**dest).getML()).assign(dynamic_cast<const BLK_AveVec &>((*src).getMLconst()),onlySemiCopy); break; }
        case 206: { dynamic_cast<BLK_AveAni &>((**dest).getML()).assign(dynamic_cast<const BLK_AveAni &>((*src).getMLconst()),onlySemiCopy); break; }
        case 207: { dynamic_cast<BLK_UsrFnB &>((**dest).getML()).assign(dynamic_cast<const BLK_UsrFnB &>((*src).getMLconst()),onlySemiCopy); break; }
        case 208: { dynamic_cast<BLK_CalBak &>((**dest).getML()).assign(dynamic_cast<const BLK_CalBak &>((*src).getMLconst()),onlySemiCopy); break; }
        case 209: { dynamic_cast<BLK_MexFnA &>((**dest).getML()).assign(dynamic_cast<const BLK_MexFnA &>((*src).getMLconst()),onlySemiCopy); break; }
        case 210: { dynamic_cast<BLK_MexFnB &>((**dest).getML()).assign(dynamic_cast<const BLK_MexFnB &>((*src).getMLconst()),onlySemiCopy); break; }
        case 211: { dynamic_cast<BLK_Mercer &>((**dest).getML()).assign(dynamic_cast<const BLK_Mercer &>((*src).getMLconst()),onlySemiCopy); break; }
        case 212: { dynamic_cast<BLK_Conect &>((**dest).getML()).assign(dynamic_cast<const BLK_Conect &>((*src).getMLconst()),onlySemiCopy); break; }
        case 213: { dynamic_cast<BLK_System &>((**dest).getML()).assign(dynamic_cast<const BLK_System &>((*src).getMLconst()),onlySemiCopy); break; }
        case 214: { dynamic_cast<BLK_Kernel &>((**dest).getML()).assign(dynamic_cast<const BLK_Kernel &>((*src).getMLconst()),onlySemiCopy); break; }
        case 215: { dynamic_cast<BLK_Bernst &>((**dest).getML()).assign(dynamic_cast<const BLK_Bernst &>((*src).getMLconst()),onlySemiCopy); break; }
        case 216: { dynamic_cast<BLK_Batter &>((**dest).getML()).assign(dynamic_cast<const BLK_Batter &>((*src).getMLconst()),onlySemiCopy); break; }

        case 300: { dynamic_cast<KNN_Densit &>((**dest).getML()).assign(dynamic_cast<const KNN_Densit &>((*src).getMLconst()),onlySemiCopy); break; }
        case 301: { dynamic_cast<KNN_Binary &>((**dest).getML()).assign(dynamic_cast<const KNN_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case 302: { dynamic_cast<KNN_Gentyp &>((**dest).getML()).assign(dynamic_cast<const KNN_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }
        case 303: { dynamic_cast<KNN_Scalar &>((**dest).getML()).assign(dynamic_cast<const KNN_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 304: { dynamic_cast<KNN_Vector &>((**dest).getML()).assign(dynamic_cast<const KNN_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case 305: { dynamic_cast<KNN_Anions &>((**dest).getML()).assign(dynamic_cast<const KNN_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case 307: { dynamic_cast<KNN_MultiC &>((**dest).getML()).assign(dynamic_cast<const KNN_MultiC &>((*src).getMLconst()),onlySemiCopy); break; }

        case 400: { dynamic_cast<GPR_Scalar     &>((**dest).getML()).assign(dynamic_cast<const GPR_Scalar     &>((*src).getMLconst()),onlySemiCopy); break; }
        case 401: { dynamic_cast<GPR_Vector     &>((**dest).getML()).assign(dynamic_cast<const GPR_Vector     &>((*src).getMLconst()),onlySemiCopy); break; }
        case 402: { dynamic_cast<GPR_Anions     &>((**dest).getML()).assign(dynamic_cast<const GPR_Anions     &>((*src).getMLconst()),onlySemiCopy); break; }
        case 408: { dynamic_cast<GPR_Gentyp     &>((**dest).getML()).assign(dynamic_cast<const GPR_Gentyp     &>((*src).getMLconst()),onlySemiCopy); break; }
        case 409: { dynamic_cast<GPR_Binary     &>((**dest).getML()).assign(dynamic_cast<const GPR_Binary     &>((*src).getMLconst()),onlySemiCopy); break; }
        case 410: { dynamic_cast<GPR_Scalar_rff &>((**dest).getML()).assign(dynamic_cast<const GPR_Scalar_rff &>((*src).getMLconst()),onlySemiCopy); break; }
        case 411: { dynamic_cast<GPR_Binary_rff &>((**dest).getML()).assign(dynamic_cast<const GPR_Binary_rff &>((*src).getMLconst()),onlySemiCopy); break; }

        case 500: { dynamic_cast<LSV_Scalar &>((**dest).getML()).assign(dynamic_cast<const LSV_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 501: { dynamic_cast<LSV_Vector &>((**dest).getML()).assign(dynamic_cast<const LSV_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case 502: { dynamic_cast<LSV_Anions &>((**dest).getML()).assign(dynamic_cast<const LSV_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case 505: { dynamic_cast<LSV_ScScor &>((**dest).getML()).assign(dynamic_cast<const LSV_ScScor &>((*src).getMLconst()),onlySemiCopy); break; }
        case 508: { dynamic_cast<LSV_Gentyp &>((**dest).getML()).assign(dynamic_cast<const LSV_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }
        case 509: { dynamic_cast<LSV_Planar &>((**dest).getML()).assign(dynamic_cast<const LSV_Planar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 510: { dynamic_cast<LSV_MvRank &>((**dest).getML()).assign(dynamic_cast<const LSV_MvRank &>((*src).getMLconst()),onlySemiCopy); break; }
        case 511: { dynamic_cast<LSV_Binary &>((**dest).getML()).assign(dynamic_cast<const LSV_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case 512: { dynamic_cast<LSV_Scalar_rff &>((**dest).getML()).assign(dynamic_cast<const LSV_Scalar_rff &>((*src).getMLconst()),onlySemiCopy); break; }

        case 600: { dynamic_cast<IMP_Expect &>((**dest).getML()).assign(dynamic_cast<const IMP_Expect &>((*src).getMLconst()),onlySemiCopy); break; }
        case 601: { dynamic_cast<IMP_ParSVM &>((**dest).getML()).assign(dynamic_cast<const IMP_ParSVM &>((*src).getMLconst()),onlySemiCopy); break; }
        case 602: { dynamic_cast<IMP_RLSamp &>((**dest).getML()).assign(dynamic_cast<const IMP_RLSamp &>((*src).getMLconst()),onlySemiCopy); break; }
        case 603: { dynamic_cast<IMP_NLSamp &>((**dest).getML()).assign(dynamic_cast<const IMP_NLSamp &>((*src).getMLconst()),onlySemiCopy); break; }

        case 800: { dynamic_cast<MLM_Scalar &>((**dest).getML()).assign(dynamic_cast<const MLM_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 801: { dynamic_cast<MLM_Binary &>((**dest).getML()).assign(dynamic_cast<const MLM_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case 802: { dynamic_cast<MLM_Vector &>((**dest).getML()).assign(dynamic_cast<const MLM_Vector &>((*src).getMLconst()),onlySemiCopy); break; }

        default:
        {
            NiceThrow("Error: unknown type error in assignxfer assign");

            break;
        }
    }

    return **dest;
}

// Notes on xfer functions:
//
// - by using qaddtrainingvector, we remove the actual data from the source
//   and place it into the dest, leaving a placeholder.  As the added vector
//   is by default set zero, and the Gp matrix is a callback cache, the memory
//   load from all this adding is minimal.  We then simply resize the src to
//   zero to remove all the data.

ML_Base &xfer(ML_Base &dest, ML_Base &src)
{
    if ( dest.type() == src.type() )
    {
        switch ( dest.type() )
        {
            case   0: { dynamic_cast<SVM_Scalar &>(dest) = dynamic_cast<const SVM_Scalar &>(src); break; }
            case   1: { dynamic_cast<SVM_Binary &>(dest) = dynamic_cast<const SVM_Binary &>(src); break; }
            case   2: { dynamic_cast<SVM_Single &>(dest) = dynamic_cast<const SVM_Single &>(src); break; }
            case   3: { dynamic_cast<SVM_MultiC &>(dest) = dynamic_cast<const SVM_MultiC &>(src); break; }
            case   4: { dynamic_cast<SVM_Vector &>(dest) = dynamic_cast<const SVM_Vector &>(src); break; }
            case   5: { dynamic_cast<SVM_Anions &>(dest) = dynamic_cast<const SVM_Anions &>(src); break; }
            case   7: { dynamic_cast<SVM_Densit &>(dest) = dynamic_cast<const SVM_Densit &>(src); break; }
            case   8: { dynamic_cast<SVM_PFront &>(dest) = dynamic_cast<const SVM_PFront &>(src); break; }
            case  12: { dynamic_cast<SVM_BiScor &>(dest) = dynamic_cast<const SVM_BiScor &>(src); break; }
            case  13: { dynamic_cast<SVM_ScScor &>(dest) = dynamic_cast<const SVM_ScScor &>(src); break; }
            case  15: { dynamic_cast<SVM_Gentyp &>(dest) = dynamic_cast<const SVM_Gentyp &>(src); break; }
            case  16: { dynamic_cast<SVM_Planar &>(dest) = dynamic_cast<const SVM_Planar &>(src); break; }
            case  17: { dynamic_cast<SVM_MvRank &>(dest) = dynamic_cast<const SVM_MvRank &>(src); break; }
            case  18: { dynamic_cast<SVM_MulBin &>(dest) = dynamic_cast<const SVM_MulBin &>(src); break; }
            case  19: { dynamic_cast<SVM_SimLrn &>(dest) = dynamic_cast<const SVM_SimLrn &>(src); break; }
            case  20: { dynamic_cast<SVM_Cyclic &>(dest) = dynamic_cast<const SVM_Cyclic &>(src); break; }
            case  21: { dynamic_cast<SVM_KConst &>(dest) = dynamic_cast<const SVM_KConst &>(src); break; }
            case  22: { dynamic_cast<SVM_Scalar_rff &>(dest) = dynamic_cast<const SVM_Scalar_rff &>(src); break; }
            case  23: { dynamic_cast<SVM_Binary_rff &>(dest) = dynamic_cast<const SVM_Binary_rff &>(src); break; }

            case 200: { dynamic_cast<BLK_Nopnop &>(dest) = dynamic_cast<const BLK_Nopnop &>(src); break; }
            case 201: { dynamic_cast<BLK_Consen &>(dest) = dynamic_cast<const BLK_Consen &>(src); break; }
            case 202: { dynamic_cast<BLK_AveSca &>(dest) = dynamic_cast<const BLK_AveSca &>(src); break; }
            case 203: { dynamic_cast<BLK_UsrFnA &>(dest) = dynamic_cast<const BLK_UsrFnA &>(src); break; }
            case 204: { dynamic_cast<BLK_UserIO &>(dest) = dynamic_cast<const BLK_UserIO &>(src); break; }
            case 205: { dynamic_cast<BLK_AveVec &>(dest) = dynamic_cast<const BLK_AveVec &>(src); break; }
            case 206: { dynamic_cast<BLK_AveAni &>(dest) = dynamic_cast<const BLK_AveAni &>(src); break; }
            case 207: { dynamic_cast<BLK_UsrFnB &>(dest) = dynamic_cast<const BLK_UsrFnB &>(src); break; }
            case 208: { dynamic_cast<BLK_CalBak &>(dest) = dynamic_cast<const BLK_CalBak &>(src); break; }
            case 209: { dynamic_cast<BLK_MexFnA &>(dest) = dynamic_cast<const BLK_MexFnA &>(src); break; }
            case 210: { dynamic_cast<BLK_MexFnB &>(dest) = dynamic_cast<const BLK_MexFnB &>(src); break; }
            case 211: { dynamic_cast<BLK_Mercer &>(dest) = dynamic_cast<const BLK_Mercer &>(src); break; }
            case 212: { dynamic_cast<BLK_Conect &>(dest) = dynamic_cast<const BLK_Conect &>(src); break; }
            case 213: { dynamic_cast<BLK_System &>(dest) = dynamic_cast<const BLK_System &>(src); break; }
            case 214: { dynamic_cast<BLK_Kernel &>(dest) = dynamic_cast<const BLK_Kernel &>(src); break; }
            case 215: { dynamic_cast<BLK_Bernst &>(dest) = dynamic_cast<const BLK_Bernst &>(src); break; }
            case 216: { dynamic_cast<BLK_Batter &>(dest) = dynamic_cast<const BLK_Batter &>(src); break; }

            case 300: { dynamic_cast<KNN_Densit &>(dest) = dynamic_cast<const KNN_Densit &>(src); break; }
            case 301: { dynamic_cast<KNN_Binary &>(dest) = dynamic_cast<const KNN_Binary &>(src); break; }
            case 302: { dynamic_cast<KNN_Gentyp &>(dest) = dynamic_cast<const KNN_Gentyp &>(src); break; }
            case 303: { dynamic_cast<KNN_Scalar &>(dest) = dynamic_cast<const KNN_Scalar &>(src); break; }
            case 304: { dynamic_cast<KNN_Vector &>(dest) = dynamic_cast<const KNN_Vector &>(src); break; }
            case 305: { dynamic_cast<KNN_Anions &>(dest) = dynamic_cast<const KNN_Anions &>(src); break; }
            case 307: { dynamic_cast<KNN_MultiC &>(dest) = dynamic_cast<const KNN_MultiC &>(src); break; }

            case 400: { dynamic_cast<GPR_Scalar     &>(dest) = dynamic_cast<const GPR_Scalar     &>(src); break; }
            case 401: { dynamic_cast<GPR_Vector     &>(dest) = dynamic_cast<const GPR_Vector     &>(src); break; }
            case 402: { dynamic_cast<GPR_Anions     &>(dest) = dynamic_cast<const GPR_Anions     &>(src); break; }
            case 408: { dynamic_cast<GPR_Gentyp     &>(dest) = dynamic_cast<const GPR_Gentyp     &>(src); break; }
            case 409: { dynamic_cast<GPR_Binary     &>(dest) = dynamic_cast<const GPR_Binary     &>(src); break; }
            case 410: { dynamic_cast<GPR_Scalar_rff &>(dest) = dynamic_cast<const GPR_Scalar_rff &>(src); break; }
            case 411: { dynamic_cast<GPR_Binary_rff &>(dest) = dynamic_cast<const GPR_Binary_rff &>(src); break; }

            case 500: { dynamic_cast<LSV_Scalar &>(dest) = dynamic_cast<const LSV_Scalar &>(src); break; }
            case 501: { dynamic_cast<LSV_Vector &>(dest) = dynamic_cast<const LSV_Vector &>(src); break; }
            case 502: { dynamic_cast<LSV_Anions &>(dest) = dynamic_cast<const LSV_Anions &>(src); break; }
            case 505: { dynamic_cast<LSV_ScScor &>(dest) = dynamic_cast<const LSV_ScScor &>(src); break; }
            case 508: { dynamic_cast<LSV_Gentyp &>(dest) = dynamic_cast<const LSV_Gentyp &>(src); break; }
            case 509: { dynamic_cast<LSV_Planar &>(dest) = dynamic_cast<const LSV_Planar &>(src); break; }
            case 510: { dynamic_cast<LSV_MvRank &>(dest) = dynamic_cast<const LSV_MvRank &>(src); break; }
            case 511: { dynamic_cast<LSV_Binary &>(dest) = dynamic_cast<const LSV_Binary &>(src); break; }
            case 512: { dynamic_cast<LSV_Scalar_rff &>(dest) = dynamic_cast<const LSV_Scalar_rff &>(src); break; }

            case 600: { dynamic_cast<IMP_Expect &>(dest) = dynamic_cast<const IMP_Expect &>(src); break; }
            case 601: { dynamic_cast<IMP_ParSVM &>(dest) = dynamic_cast<const IMP_ParSVM &>(src); break; }
            case 602: { dynamic_cast<IMP_RLSamp &>(dest) = dynamic_cast<const IMP_RLSamp &>(src); break; }
            case 603: { dynamic_cast<IMP_NLSamp &>(dest) = dynamic_cast<const IMP_NLSamp &>(src); break; }

            case 800: { dynamic_cast<MLM_Scalar &>(dest) = dynamic_cast<const MLM_Scalar &>(src); break; }
            case 801: { dynamic_cast<MLM_Binary &>(dest) = dynamic_cast<const MLM_Binary &>(src); break; }
            case 802: { dynamic_cast<MLM_Vector &>(dest) = dynamic_cast<const MLM_Vector &>(src); break; }

            default:
            {
                NiceThrow("Error: Unknown source/destination type in ML xfer.");

                break;
            }
        }
    }

    else if ( isSVMBinary(dest) && isSVMSingle(src) )
    {
        dynamic_cast<SVM_Binary &>(dest) = dynamic_cast<const SVM_Single &>(src);
    }

    else
    {
        xferInfo(dest,src);

        int i,j;
        int d;
        double Cweight;
        double epsweight;
        gentype y;
        SparseVector<gentype> x;

        for ( i = (src.N())-1 ; i >= 0 ; --i )
        {
            j = dest.N();

            d         = src.isenabled(i);
            Cweight   = (src.Cweight())(i);
            Cweight   = (src.Cweight())(i);
            epsweight = (src.epsweight())(i);

            src.removeTrainingVector(i,y,x);
            dest.qaddTrainingVector(j,y,x,Cweight,epsweight);

            if ( !d )
            {
                dest.disable(j);
            }
        }

        if ( ( isSVMScalar(dest) || isSVMBinary(dest) || isSVMSingle(dest) ) && isSVM(src) )
        {
            (dynamic_cast<SVM_Generic &>(dest)).setCweightfuzz((dynamic_cast<const SVM_Generic &>(src)).Cweightfuzz());
        }
    }

    src.restart();

    return dest;
}

void xferInfo(ML_Base &mldest, const ML_Base &mlsrc)
{
    // Clear the destination

    mldest.restart();
    mldest.setmemsize(mlsrc.memsize());

    // Transfer variables for *all* types of ML

    mldest.setC(mlsrc.C());
    mldest.seteps(mlsrc.eps());

    mldest.setzerotol(mlsrc.zerotol());
    mldest.setOpttol(mlsrc.Opttol());
    mldest.setmaxitcnt(mlsrc.maxitcnt());
    mldest.setmaxtraintime(mlsrc.maxtraintime());
    mldest.settraintimeend(mlsrc.traintimeend());

    mldest.setKernel(mlsrc.getKernel());

    // Type specifics follow

    if ( isSVMScalar(mldest) && isSVM(mlsrc) )
    {
              SVM_Scalar  &dest = dynamic_cast<      SVM_Scalar  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.set1NormCost();     }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMBinary(mldest) && isSVM(mlsrc) )
    {
              SVM_Binary  &dest = dynamic_cast<      SVM_Binary  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMSingle(mldest) && isSVM(mlsrc) )
    {
              SVM_Single  &dest = dynamic_cast<      SVM_Single  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMMultiC(mldest) && isSVM(mlsrc) )
    {
              SVM_MultiC  &dest = dynamic_cast<      SVM_MultiC  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());

        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMVector(mldest) && isSVM(mlsrc) )
    {
              SVM_Vector  &dest = dynamic_cast<      SVM_Vector  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());

        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
    }

    else if ( isSVMAnions(mldest) && isSVM(mlsrc) )
    {
              SVM_Anions  &dest = dynamic_cast<      SVM_Anions  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());

        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
    }

    else if ( isSVMDensit(mldest) && isSVM(mlsrc) )
    {
              SVM_Densit  &dest = dynamic_cast<      SVM_Densit  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMPFront(mldest) && isSVM(mlsrc) )
    {
              SVM_Single  &dest = dynamic_cast<      SVM_Single  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMBiScor(mldest) && isSVM(mlsrc) )
    {
              SVM_BiScor  &dest = dynamic_cast<      SVM_BiScor  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMScScor(mldest) && isSVM(mlsrc) )
    {
              SVM_ScScor  &dest = dynamic_cast<      SVM_ScScor  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMGentyp(mldest) && isSVM(mlsrc) )
    {
              SVM_Gentyp  &dest = dynamic_cast<      SVM_Gentyp  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMPlanar(mldest) && isSVM(mlsrc) )
    {
              SVM_Planar  &dest = dynamic_cast<      SVM_Planar  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMMvRank(mldest) && isSVM(mlsrc) )
    {
              SVM_MvRank  &dest = dynamic_cast<      SVM_MvRank  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMCyclic(mldest) && isSVM(mlsrc) )
    {
              SVM_Cyclic  &dest = dynamic_cast<      SVM_Cyclic  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMMulBin(mldest) && isSVM(mlsrc) )
    {
              SVM_MulBin  &dest = dynamic_cast<      SVM_MulBin  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMSimLrn(mldest) && isSVM(mlsrc) )
    {
              SVM_SimLrn  &dest = dynamic_cast<      SVM_SimLrn  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMKConst(mldest) && isSVM(mlsrc) )
    {
              SVM_KConst  &dest = dynamic_cast<      SVM_KConst  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMScalar_rff(mldest) && isSVM(mlsrc) )
    {
              SVM_Scalar_rff  &dest = dynamic_cast<      SVM_Scalar_rff  &>(mldest);
        const SVM_Generic     &src  = dynamic_cast<const SVM_Generic     &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMBinary_rff(mldest) && isSVM(mlsrc) )
    {
              SVM_Binary_rff  &dest = dynamic_cast<      SVM_Binary_rff  &>(mldest);
        const SVM_Generic     &src  = dynamic_cast<const SVM_Generic     &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isBLKNopnop(mldest) && isBLK(mlsrc) )
    {
              BLK_Nopnop  &dest = dynamic_cast<      BLK_Nopnop  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKConsen(mldest) && isBLK(mlsrc) )
    {
              BLK_Consen  &dest = dynamic_cast<      BLK_Consen  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKAveSca(mldest) && isBLK(mlsrc) )
    {
              BLK_AveSca  &dest = dynamic_cast<      BLK_AveSca  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKUsrFnA(mldest) && isBLK(mlsrc) )
    {
              BLK_UsrFnA  &dest = dynamic_cast<      BLK_UsrFnA  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKUserIO(mldest) && isBLK(mlsrc) )
    {
              BLK_UserIO  &dest = dynamic_cast<      BLK_UserIO  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKAveVec(mldest) && isBLK(mlsrc) )
    {
              BLK_AveVec  &dest = dynamic_cast<      BLK_AveVec  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKAveAni(mldest) && isBLK(mlsrc) )
    {
              BLK_AveAni  &dest = dynamic_cast<      BLK_AveAni  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKUsrFnB(mldest) && isBLK(mlsrc) )
    {
              BLK_UsrFnB  &dest = dynamic_cast<      BLK_UsrFnB  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKCalBak(mldest) && isBLK(mlsrc) )
    {
              BLK_CalBak  &dest = dynamic_cast<      BLK_CalBak  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKMexFnA(mldest) && isBLK(mlsrc) )
    {
              BLK_MexFnA  &dest = dynamic_cast<      BLK_MexFnA  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKMexFnB(mldest) && isBLK(mlsrc) )
    {
              BLK_MexFnB  &dest = dynamic_cast<      BLK_MexFnB  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKMercer(mldest) && isBLK(mlsrc) )
    {
              BLK_Mercer  &dest = dynamic_cast<      BLK_Mercer  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKConect(mldest) && isBLK(mlsrc) )
    {
              BLK_Conect  &dest = dynamic_cast<      BLK_Conect  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKSystem(mldest) && isBLK(mlsrc) )
    {
              BLK_System  &dest = dynamic_cast<      BLK_System  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKKernel(mldest) && isBLK(mlsrc) )
    {
              BLK_Kernel  &dest = dynamic_cast<      BLK_Kernel  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKBernst(mldest) && isBLK(mlsrc) )
    {
              BLK_Bernst  &dest = dynamic_cast<      BLK_Bernst  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKBatter(mldest) && isBLK(mlsrc) )
    {
              BLK_Batter  &dest = dynamic_cast<      BLK_Batter  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isKNNDensit(mldest) && isKNN(mlsrc) )
    {
              KNN_Densit  &dest = dynamic_cast<      KNN_Densit  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNBinary(mldest) && isKNN(mlsrc) )
    {
              KNN_Binary  &dest = dynamic_cast<      KNN_Binary  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNGentyp(mldest) && isKNN(mlsrc) )
    {
              KNN_Gentyp  &dest = dynamic_cast<      KNN_Gentyp  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNScalar(mldest) && isKNN(mlsrc) )
    {
              KNN_Scalar  &dest = dynamic_cast<      KNN_Scalar  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNVector(mldest) && isKNN(mlsrc) )
    {
              KNN_Vector  &dest = dynamic_cast<      KNN_Vector  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNAnions(mldest) && isKNN(mlsrc) )
    {
              KNN_Anions  &dest = dynamic_cast<      KNN_Anions  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNMultiC(mldest) && isKNN(mlsrc) )
    {
              KNN_MultiC  &dest = dynamic_cast<      KNN_MultiC  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isGPRScalar(mldest) && isGPR(mlsrc) )
    {
              GPR_Scalar  &dest = dynamic_cast<      GPR_Scalar  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRVector(mldest) && isGPR(mlsrc) )
    {
              GPR_Scalar  &dest = dynamic_cast<      GPR_Scalar  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRAnions(mldest) && isGPR(mlsrc) )
    {
              GPR_Anions  &dest = dynamic_cast<      GPR_Anions  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRGentyp(mldest) && isGPR(mlsrc) )
    {
              GPR_Gentyp  &dest = dynamic_cast<      GPR_Gentyp  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRBinary(mldest) && isGPR(mlsrc) )
    {
              GPR_Binary  &dest = dynamic_cast<      GPR_Binary  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRScalar_rff(mldest) && isGPR(mlsrc) )
    {
              GPR_Scalar_rff  &dest = dynamic_cast<      GPR_Scalar_rff  &>(mldest);
        const GPR_Generic     &src  = dynamic_cast<const GPR_Generic     &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRBinary_rff(mldest) && isGPR(mlsrc) )
    {
              GPR_Binary_rff  &dest = dynamic_cast<      GPR_Binary_rff  &>(mldest);
        const GPR_Generic     &src  = dynamic_cast<const GPR_Generic     &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isLSVScalar(mldest) && isLSV(mlsrc) )
    {
              LSV_Scalar  &dest = dynamic_cast<      LSV_Scalar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVVector(mldest) && isLSV(mlsrc) )
    {
              LSV_Scalar  &dest = dynamic_cast<      LSV_Scalar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVAnions(mldest) && isLSV(mlsrc) )
    {
              LSV_Scalar  &dest = dynamic_cast<      LSV_Scalar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVScScor(mldest) && isLSV(mlsrc) )
    {
              LSV_ScScor  &dest = dynamic_cast<      LSV_ScScor  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVGentyp(mldest) && isLSV(mlsrc) )
    {
              LSV_Gentyp  &dest = dynamic_cast<      LSV_Gentyp  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVPlanar(mldest) && isLSV(mlsrc) )
    {
              LSV_Planar  &dest = dynamic_cast<      LSV_Planar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVMvRank(mldest) && isLSV(mlsrc) )
    {
              LSV_MvRank  &dest = dynamic_cast<      LSV_MvRank  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVBinary(mldest) && isLSV(mlsrc) )
    {
              LSV_Binary  &dest = dynamic_cast<      LSV_Binary  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVScalar_rff(mldest) && isLSV(mlsrc) )
    {
              LSV_Scalar_rff &dest = dynamic_cast<      LSV_Scalar_rff &>(mldest);
        const LSV_Generic    &src  = dynamic_cast<const LSV_Generic    &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isIMPExpect(mldest) && isIMP(mlsrc) )
    {
              IMP_Expect  &dest = dynamic_cast<      IMP_Expect  &>(mldest);
        const IMP_Generic &src  = dynamic_cast<const IMP_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isIMPParSVM(mldest) && isIMP(mlsrc) )
    {
              IMP_ParSVM  &dest = dynamic_cast<      IMP_ParSVM  &>(mldest);
        const IMP_Generic &src  = dynamic_cast<const IMP_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isIMPRLSamp(mldest) && isIMP(mlsrc) )
    {
              IMP_RLSamp  &dest = dynamic_cast<      IMP_RLSamp  &>(mldest);
        const IMP_Generic &src  = dynamic_cast<const IMP_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isIMPNLSamp(mldest) && isIMP(mlsrc) )
    {
              IMP_NLSamp  &dest = dynamic_cast<      IMP_NLSamp  &>(mldest);
        const IMP_Generic &src  = dynamic_cast<const IMP_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isMLMScalar(mldest) && isMLM(mlsrc) )
    {
              MLM_Scalar  &dest = dynamic_cast<      MLM_Scalar  &>(mldest);
        const MLM_Generic &src  = dynamic_cast<const MLM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isMLMBinary(mldest) && isMLM(mlsrc) )
    {
              MLM_Binary  &dest = dynamic_cast<      MLM_Binary  &>(mldest);
        const MLM_Generic &src  = dynamic_cast<const MLM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isMLMVector(mldest) && isMLM(mlsrc) )
    {
              MLM_Vector  &dest = dynamic_cast<      MLM_Vector  &>(mldest);
        const MLM_Generic &src  = dynamic_cast<const MLM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    return;
}








int ML_Mutable::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    return getMLconst().egetparam(ind,val,xa,ia,xb,ib);
}


int ML_Mutable::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    return getMLconst().getparam(ind,val,xa,ia,xb,ib,desc);
}






// ML examiner functions

void interobs(std::ostream &output, std::istream &input)
{
    {
        int mlid,ind;
        std::string temp;
        std::string linebuff;
        int firstrun = 1;

        gentype val;
        int ia,ib;
        gentype xa,xb;

        MercerKernel dummy; // this is to ensure that the relevant ML list is allocated
        kernPrecursor *dummyk;

        while ( 1 )
        {
            if ( firstrun )
            {
                output << "\n";
                output << "ML object examiner:\n";
                output << "\n";
                output << " ret                       - return to previous menu\n";
                output << " list                      - list details of current MLs\n";
                output << " print i                   - print ML i\n";
                output << " view0 i ind               - view parameter ind for ML i\n";
                output << " view1 i ind xa ia         - view parameter ind for ML i with arguments xa,ia\n";
                output << " view2 i ind xa ia xb ib   - view parameter ind for ML i with arguments xa,ia and xb,ib\n";
                output << " help                      - this screen.\n";
                output << " ?                         - this screen.\n";
                output << " man                       - this screen.\n";
            }

            firstrun = 0;

            output << "> ";
            output << '\0'; // Forces cache flush in matlab

            input >> temp;
            output << temp << "\n";

                 if ( temp == "help"   ) { firstrun = 1; }
            else if ( temp == "man"    ) { firstrun = 1; }
            else if ( temp == "?"      ) { firstrun = 1; }
            else if ( temp == "ret"    ) { return; }

            else if ( temp == "list"   )
            {
               int i,t,st;
               int m = dummy.mllistsize();
               std::string typestring;

               output << "MLID\ttype\tsubtype\ttypestring\taddress\tML_Mutable\t(N)\n";

               for ( i = 0 ; i < m ; ++i )
               {
                    mlid = dummy.mllistind(i);
                    dummy.getaltML(dummyk,mlid);
                    t = (*dummyk).type();
                    st = (*dummyk).subtype();
                    convTypeToID(typestring,t);

                    output << mlid << "\t" << t << "\t" << st << "\t" << typestring << "\t" << std::hex << dummyk << std::dec;

                    if ( t >= 0 )
                    {
                        const ML_Base &dummyML = (dynamic_cast<const ML_Base &>(*dummyk));

                        output << "\t" << dummyML.isMutable() << "\t(" << dummyML.N() << ")";
                    }

                    output << "\n";
                }
            }

            else if ( temp == "print"  )
            {
                input >> mlid;

                if ( dummy.mllistisindpresent(mlid) )
                {
                    dummy.getaltML(dummyk,mlid);
                    (*dummyk).printstream(output,0);
                }

                else
                {
                    output << "ML index not present\n";
                }
            }

            else if ( temp == "view0"  )
            {
                input >> mlid;
                input >> ind;
                ia = 0;
                ib = 0;

                if ( dummy.mllistisindpresent(mlid) )
                {
                    const char *desc;
                    dummy.getaltML(dummyk,mlid);
                    (*dummyk).getparam(ind,val, xa,ia, xb,ib,desc);
                    output << desc << " = " << val << "\n";
               }

                else
                {
                    output << "ML index not present\n";
                }
            }

            else if ( temp == "view1"  )
            {
                input >> mlid;
                input >> ind;
                input >> xa;
                input >> ia;
                ib = 0;

                if ( dummy.mllistisindpresent(mlid) )
                {
                    const char *desc;
                    dummy.getaltML(dummyk,mlid);
                    (*dummyk).getparam(ind,val, xa,ia, xb,ib,desc);
                    output << val << "\n";
                    output << desc << " = " << val << "\n";
                }

                else
                {
                    output << "ML index not present\n";
                }
            }

            else if ( temp == "view2"  )
            {
                input >> mlid;
                input >> ind;
                input >> xa;
                input >> ia;
                input >> xb;
                input >> ib;

                if ( dummy.mllistisindpresent(mlid) )
                {
                    const char *desc;
                    dummy.getaltML(dummyk,mlid);
                    (*dummyk).getparam(ind,val, xa,ia, xb,ib,desc);
                    output << val << "\n";
                    output << desc << " = " << val << "\n";
                }

                else
                {
                    output << "ML index not present\n";
                }
            }

            else
            {
                output << "Unknown command.\n";
                firstrun = 1;
            }
        }
    }

    return;
}

