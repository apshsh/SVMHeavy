
//
// SVMHeavy CLI
//
// Version: 7
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "mlinter.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"


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


int main(int argc, char *argv[])
{
    try
    {
        // Initialisation of static, overall state, set-once type stuff

        // We do this for speed
        std::ios::sync_with_stdio(false);

        // register main thread ID
        isMainThread(1);
        // not strictly needed but good policy if threading used
        //initgentype(); // - still needed at avoid memory error at exit - UPDATE: NOT ANYMORE!
        // sets callback for calculator in god mode
        //setintercalc(&intercalc); - not needed anymore

        // Print help if no commands given

        if ( argc == 1 )
        {
            outstream() << "SVMheavy 7.0: an SVM implementation by Alistair Shilton.                      \n";
            outstream() << "============                                                                  \n";
            outstream() << "                                                                              \n";
            outstream() << "Copyright: all rights reserved.                                               \n";
            outstream() << "                                                                              \n";
            outstream() << "Usage:         svmheavyv6 {options}                                           \n";
            outstream() << "Basic help:    svmheavyv6 -?                                                  \n";
            outstream() << "Advanced help: svmheavyv6 -??                                                 \n";

            return 0;
        }

        // Register "time 0"

        TIMEABSSEC(TIMECALL);

        // Set up streams (we divert outputs to logfiles)

        void(*xcliCharPrintErr)(char c) = cliCharPrintErr;
        static LoggingOstreamErr clicerr(xcliCharPrintErr);
        seterrstream(&clicerr);

        void(*xcliCharPrintOut)(char c) = cliCharPrintOut;
        static LoggingOstreamOut clicout(xcliCharPrintOut);
        setoutstream(&clicout);

        // Convert the command line arguments into a command string

        std::string commline;

        int i;
        size_t j;
        int isquote;

        for ( i = 1 ; i < argc ; ++i )
        {
            // NB: if string contains spaces but is not enclosed in quotes then
            // dos is being a pita and stripping quotes, so reinstate them.  This
            // will not fix the problem of quotes being stripped from a string
            // not containing spaces, so you need to be wary of that later.  Nor
            // will it fix double-quoted strings

            isquote = 0;

            if ( strlen(argv[i]) )
            {
                if ( ( argv[i][0] != '\"' ) || ( argv[i][strlen(argv[i])-1] != '\"' ) )
                {
                    for ( j = 0 ; j < strlen(argv[i]) ; ++j )
                    {
                        if ( argv[i][j] == ' ' )
                        {
                            isquote = 1;
                            break;
                        }
                    }
                }
            }

            if ( isquote )
            {
                commline += '\"';
                commline += argv[i];
                commline += '\"';
            }

            else
            {
                commline += argv[i];
            }

            if ( i < argc-1 )
            {
                commline += " ";
            }
        }

        // Add -Zx to the end of the command string to ensure that the output
        // stream used by -echo will remain available until the end.

        commline += " -Zx";
//        outstream() << "Running command: " << commline << "\n";

        // Define global variable store

        svmvolatile SparseVector<SparseVector<gentype> > globargvariables;

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

        int svmInd = 1; // not 0 anymore - want to reserve that index for other stuff! 0;
        static thread_local SVMThreadContext *svmContext;
        SparseVector<ML_Mutable *> svmbase;
        MEMNEW(svmContext,SVMThreadContext(svmInd));
        errstream() << "{";

        // Now that everything has been set up so we can run the actual code.

        SparseVector<SparseVector<int> > returntag;

        runsvm(svmContext,svmbase,commstack,globargvariables,cligetsetExtVar,returntag);

        // Unlock the thread, signalling that the context can be deleted etc

        errstream() << "}";

        MEMDEL(commstack);

        // Code not re-entrant, so need to blitz threads, and also delete remaining MLs

        errstream() << "Killing dangling threads.\n";

        killallthreads(svmContext);

        errstream() << "Clearing memory.\n";

        deleteMLs(svmbase);

        cliPrintToOutLog('*',1);
        cliPrintToErrLog('*',1);

        isMainThread(0);
    }

    catch ( const char *errcode )
    {
        errstream() << "Unknown error: " << errcode << ".\n";
        return 1;
    }

    catch ( const std::string errcode )
    {
        errstream() << "Unknown error: " << errcode << ".\n";
        return 1;
    }

    return 0;
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

