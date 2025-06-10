
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
#include "mlinter.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "vecfifo.hpp"

namespace py = pybind11;

int                  intvalsrc(int i);
double               dblvalsrc(int i);
std::string          strvalsrc(int i);
std::complex<double> cplvalsrc(int i);

void intpush(int x);
void dblpush(double x);
void strpush(std::string x);
void compush(std::string x);

void svmheavya(void);                               // just get help screen
void svmheavyb(int permode);                        // set persistence mode
void svmheavyc(const std::string commstr);          // execute with string
void svmheavyd(const std::string commstr, int wml); // execute with string and ml number

PYBIND11_MODULE(pyheavy, m) {
    m.def("help", &svmheavya, "Help screen");
    m.def("mode", &svmheavyb, "Set persistence mode");
    m.def("exec", &svmheavyc, "Run with string given");
    m.def("execmod", &svmheavyd, "Run with string given on ML given");
    m.def("intvalsrc", &intvalsrc, "Internal use");
    m.def("dblvalsrc", &dblvalsrc, "Internal use");
    m.def("strvalsrc", &strvalsrc, "Internal use");
    m.def("cplvalsrc", &cplvalsrc, "Internal use");
    m.def("intpush", &intpush, "Internal use");
    m.def("dblpush", &dblpush, "Internal use");
    m.def("strpush", &strpush, "Internal use");
    m.def("compush", &compush, "Internal use");
}

void svmheavy(int method, int permode, const std::string commstr, int wml);

void svmheavya(void)
{
    std::string dummy;

    svmheavy(1,-1,dummy,-1);
}

void svmheavyb(int permode)
{
    std::string dummy;

    svmheavy(2,permode,dummy,-1);
}

void svmheavyc(const std::string commstr)
{
    svmheavy(3,-1,commstr,-1);
}

void svmheavyd(const std::string commstr, int wml)
{
    svmheavy(3,-1,commstr,wml);
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

void cliCharPrintOut(char c);
void cliCharPrintErr(char c);

void cliPrintToOutLog(char c, int mode = 0);
void cliPrintToErrLog(char c, int mode = 0);



// Python function
//
// method: 1 - print help
//         2 - set persistence mode (permode)
//         3 - run commands (in ML wml if wml != -1)

void svmheavy(int method, int permode, const std::string commstr, int wml)
{
    static thread_local int hasbeeninit = 0;
    static thread_local int persistenceset = 0;
    static thread_local int persistencereq = 0;

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
            outstream() << "Turn on persistence:  pyheavy.mode(1)                                         \n";
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
                persistencereq = 1;
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

        // Threaded data.  Each ML is an element in svmContext, with threadInd
        // specifying which is currently in use.  At this point we only have
        // a single ML with index 0.

        static thread_local int threadInd = 0;
        static thread_local int svmInd = 0;
        static thread_local SparseVector<SVMThreadContext *> svmContext;
        static thread_local SparseVector<int> svmThreadOwner;
        static thread_local SparseVector<ML_Mutable *> svmbase;
        MEMNEW(svmContext("&",threadInd),SVMThreadContext(svmInd,threadInd));
        errstream() << "{";

        // Now that everything has been set up so we can run the actual code.

        SparseVector<SparseVector<int> > returntag;

        runsvm(threadInd,svmContext,svmbase,svmThreadOwner,commstack,globargvariables,cligetsetExtVar,returntag);

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

            killallthreads(svmContext,1);

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













void cliCharPrintOut(char c)
{
    cliPrintToOutLog(c);

#ifndef HEADLESS
    if ( !LoggingOstreamOut::suppressStreamCout )
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
    if ( !LoggingOstreamErr::suppressStreamCout )
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

















// To prevent truncation, arguments sent to pycall are stored in a series of
// arrays. These functions maintain static arrays that the c++ code can store
// values in and python can then access.

int                  setintvalsrc(int i, int doset, const int &val);
double               setdblvalsrc(int i, int doset, const double &val);
std::string          setstrvalsrc(int i, int doset, const std::string &val);
std::complex<double> setcplvalsrc(int i, int doset, const std::complex<double> &val);

void intvalsrcreset(void);
void dblvalsrcreset(void);
void strvalsrcreset(void);
void cplvalsrcreset(void);

int                  setintvalsrc(int i, int doset, const int &val)                  { static thread_local SparseVector<int> xval;                   if ( doset == 2 ) { xval.zero(); } else if ( doset ) { xval("&",i) = val; } return xval(i); }
double               setdblvalsrc(int i, int doset, const double &val)               { static thread_local SparseVector<double> xval;                if ( doset == 2 ) { xval.zero(); } else if ( doset ) { xval("&",i) = val; } return xval(i); }
std::string          setstrvalsrc(int i, int doset, const std::string &val)          { static thread_local SparseVector<std::string> xval;           if ( doset == 2 ) { xval.zero(); } else if ( doset ) { xval("&",i) = val; } return xval(i); }
std::complex<double> setcplvalsrc(int i, int doset, const std::complex<double> &val) { static thread_local SparseVector<std::complex<double> > xval; if ( doset == 2 ) { xval.zero(); } else if ( doset ) { xval("&",i) = val; } return xval(i); }

int                  intvalsrc(int i) { const int tmp(0);               return setintvalsrc(i,0,tmp); }
double               dblvalsrc(int i) { const double tmp(0);            return setdblvalsrc(i,0,tmp); }
std::string          strvalsrc(int i) { const std::string tmp;          return setstrvalsrc(i,0,tmp); }
std::complex<double> cplvalsrc(int i) { const std::complex<double> tmp; return setcplvalsrc(i,0,tmp); }

void intvalsrcreset(void) { const int tmp(0);               setintvalsrc(0,2,tmp); }
void dblvalsrcreset(void) { const double tmp(0);            setdblvalsrc(0,2,tmp); }
void strvalsrcreset(void) { const std::string tmp;          setstrvalsrc(0,2,tmp); }
void cplvalsrcreset(void) { const std::complex<double> tmp; setcplvalsrc(0,2,tmp); }

// This function rewrites gentype as a string in python format.
//
// - null is translated to None
// - ints are translated to pyheavy.intvalsrc(i), which is a callback that holds the value for python to grab
// - reals are translated to pyheavy.dblvalsrc(i), which is a callback that holds the value for python to grab
// - complex numbers are translated to pyheavy.cplvalsrc(i), which is a callback that holds the value for python to grab
// - strings are translated to pyheavy.strvalsrc(i), which is a callback that holds the value for python to grab
// - vectors are translated to lists [ a, b, ... ]
// - sets are translated to tuples ( a, b, ... ) (this is because sets require hashes in python, and vectors cannot be hashed)
// - anything else returns non-zero

int convGentype(std::string &res, const gentype &src, int &iint, int &idbl, int &istr, int &icpl);
int convGentype(std::string &res, const gentype &src, int &iint, int &idbl, int &istr, int &icpl)
{
    int errnum = 0;

    if ( src.isValNull() )
    {
        // Map null to None

        res = "None";
    }

    else if ( src.isValInteger() )
    {
        res = "pyheavy.intvalsrc(";
        res += std::to_string(iint);
        res += ")";

        setintvalsrc(iint,1,(int) src);

        iint++;
    }

    else if ( src.isValReal() )
    {
        res = "pyheavy.dblvalsrc(";
        res += std::to_string(idbl);
        res += ")";

        setdblvalsrc(idbl,1,(double) src);

        idbl++;
    }

    else if ( src.isValString() )
    {
        res = "pyheavy.strvalsrc(";
        res += std::to_string(istr);
        res += ")";

        setstrvalsrc(istr,1,(const std::string &) src);

        istr++;
    }

    else if ( ( src.isValAnion() ) && ( src.order() <= 1 ) )
    {
        res = "pyheavy.cplvalsrc(";
        res += std::to_string(icpl);
        res += ")";

        d_anion tmp((const d_anion &) src);

        setcplvalsrc(icpl,1,(std::complex<double>) tmp);

        icpl++;
    }

    else if ( src.isValVector() )
    {
        // Map vectors to lists

        const Vector<gentype> &tmp = (const Vector<gentype> &) src;

        if ( tmp.infsize() )
        {
            return 1;
        }

        res = "[";

        for ( int i = 0 ; i < tmp.size() ; ++i )
        {
            std::string ires;

            if ( ( errnum = convGentype(ires,tmp(i),iint,idbl,istr,icpl) ) )
            {
                return errnum;
            }

            res += ires;

            if ( i < tmp.size()-1 )
            {
                res += ",";
            }
        }

        res += "]";
    }

    else if ( src.isValSet() )
    {
        // Map sets to tuples

        const Vector<gentype> &tmp = ((const Set<gentype> &) src).all();

        res = "(";

        for ( int i = 0 ; i < tmp.size() ; ++i )
        {
            std::string ires;

            if ( ( errnum = convGentype(ires,tmp(i),iint,idbl,istr,icpl) ) )
            {
                return errnum;
            }

            res += ires;

            if ( i < tmp.size()-1 )
            {
                res += ",";
            }
        }

        res += ")";
    }

    else
    {
        errnum = 1;
    }

    return errnum;
}





// To prevent truncation, the return value from pycall is serialised in
// FIFO buffers to be reconstructed afterwards. These functions provide
// python with buffers to push onto and c++ can pull off of.

int         setintvalres(int mode, const int &val);
double      setdblvalres(int mode, const double &val);
std::string setstrvalres(int mode, const std::string &val);
std::string setcomvalres(int mode, const std::string &val);

int         intpop(void);
double      dblpop(void);
std::string strpop(void);
std::string compop(void);

void intvalresreset(void);
void dblvalresreset(void);
void strvalresreset(void);
void comvalresreset(void);

int         setintvalres(int mode, const int &val)         { static thread_local FiFo<int> xval;         int res(0);      if ( mode == 2 ) { xval.resize(0); } else if ( mode == 1 ) { xval.pop(res); } else if ( mode == 0 ) { xval.push(val); } return res; }
double      setdblvalres(int mode, const double &val)      { static thread_local FiFo<double> xval;      double res(0);   if ( mode == 2 ) { xval.resize(0); } else if ( mode == 1 ) { xval.pop(res); } else if ( mode == 0 ) { xval.push(val); } return res; }
std::string setstrvalres(int mode, const std::string &val) { static thread_local FiFo<std::string> xval; std::string res; if ( mode == 2 ) { xval.resize(0); } else if ( mode == 1 ) { xval.pop(res); } else if ( mode == 0 ) { xval.push(val); } return res; }
std::string setcomvalres(int mode, const std::string &val) { static thread_local FiFo<std::string> xval; std::string res; if ( mode == 2 ) { xval.resize(0); } else if ( mode == 1 ) { xval.pop(res); } else if ( mode == 0 ) { xval.push(val); } return res; }

void intpush(int val)         { setintvalres(0,val); }
void dblpush(double val)      { setdblvalres(0,val); }
void strpush(std::string val) { setstrvalres(0,val); }
void compush(std::string val) { setcomvalres(0,val); }

int         intpop(void) { const int tmp(0);      return setintvalres(1,tmp); }
double      dblpop(void) { const double tmp(0);   return setdblvalres(1,tmp); }
std::string strpop(void) { const std::string tmp; return setstrvalres(1,tmp); }
std::string compop(void) { const std::string tmp; return setcomvalres(1,tmp); }

void intvalresreset(void) { const int tmp(0);      setintvalres(2,tmp); }
void dblvalresreset(void) { const double tmp(0);   setdblvalres(2,tmp); }
void strvalresreset(void) { const std::string tmp; setstrvalres(2,tmp); }
void comvalresreset(void) { const std::string tmp; setcomvalres(2,tmp); }

// This funciton interrogates the return stacks to work out what the result was.
// The following helper function in pyheavypy.py does the serialisation on the
// python side which we are reversing here.
//
// def convgen(h):
//    import pyheavy
//    if type(h) is None:
//        pyheavy.compush("none")
//    elif type(h) is int:
//        pyheavy.compush("int")
//        pyheavy.intpush(h)
//    elif type(h) is float:
//        pyheavy.compush("float")
//        pyheavy.dblpush(h)
//    elif type(h) is complex:
//        pyheavy.compush("complex")
//        pyheavy.dblpush(h.real)
//        pyheavy.dblpush(h.imag)
//    elif type(h) is str:
//        pyheavy.compush("string")
//        pyheavy.strpush(h)
//    elif type(h) is list:
//        pyheavy.compush("list")
//        pyheavy.intpush(len(h))
//        for x in h:
//            convgen(x)
//    elif type(h) is tuple:
//        pyheavy.compush("tuple")
//        pyheavy.intpush(len(h))
//        for x in h:
//            convgen(x)
//    else:
//        pyheavy.compush("badtype")
//
//
// mode: 0 push, 1 pop, 2 reset

int convPytoGentype(gentype &res);
int convPytoGentype(gentype &res)
{
    std::string comis = compop();
    int errcode = 0;

    if ( comis == "none" )
    {
        res.force_null();
    }

    else if ( comis == "int" )
    {
        res.force_int() = intpop();
    }

    else if ( comis == "float" )
    {
        res.force_double() = dblpop();
    }

    else if ( comis == "complex" )
    {
        d_anion altres = res.force_anion();

        altres.setorder(1);
        altres("&",0) = dblpop();
        altres("&",1) = dblpop();
    }

    else if ( comis == "string" )
    {
        res.force_string() = strpop();
    }

    else if ( comis == "list" )
    {
        Vector<gentype> &altres = res.force_vector();

        int numadd = intpop();

        altres.resize(numadd);

        for ( int i = 0 ; !errcode && ( i < altres.size() ) ; ++i )
        {
            errcode = convPytoGentype(altres("&",i));
        }
    }

    else if ( comis == "tuple" )
    {
        Set<gentype> temp;
        res.force_set() = temp;
        Set<gentype> &altres = res.force_set();

        int numadd = intpop();

        for ( int i = 0 ; !errcode && ( i < numadd ) ; ++i )
        {
            gentype tg;

            errcode = convPytoGentype(tg);

            altres.add(tg); // will add to end
        }
    }

    else
    {
        errcode = 1;
    }

    return errcode;
}


// drop-in replacement for pycall function in gentype.cc
// (the gentype version, which uses a system call, is disabled by the macro PYLOCAL)

void pycall(const std::string &fn, gentype &res, const gentype &x)
{
    // Load x into holder function so that python, then use eval

    intvalsrcreset();
    dblvalsrcreset();
    strvalsrcreset();
    cplvalsrcreset();

    intvalresreset();
    dblvalresreset();
    strvalresreset();
    comvalresreset();

    std::string xstr;

    // Store arguments for python function and create reconstruction string

    int iint = 0;
    int idbl = 0;
    int istr = 0;
    int icpl = 0;

    if ( convGentype(xstr,x,iint,idbl,istr,icpl) )
    {
        res.makeError("Can't convert to python string");

        return;
    }

    // Construct run command

    std::string evalfn;

    evalfn = "eval('";
    evalfn += "pyheavypy.convgen(";
    evalfn += fn;
    evalfn += "(";
    evalfn += xstr;
    evalfn += "))";
    evalfn += "')";

    //errstream() << "evalfn: " << evalfn << "\n";

    py::object builtins = py::module_::import("builtins");
    py::object eval = builtins.attr("eval");
    auto resultobj = eval(evalfn);

    // Retrieve results of operation

    convPytoGentype(res);

    //errstream() << "result: " << res << "\n";

    return;
}


