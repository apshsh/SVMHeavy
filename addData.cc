
//
// Data loading functions
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <string>
#include <math.h>
#include "addData.hpp"
#include "ml_mutable.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "svm_mvrank.hpp"
#include "lsv_mvrank.hpp"


int addtrainingdata(ML_Base &mlbase, const char *trainfile, int reverse, int ignoreStart, int imax, int ibase, const char *savefile)
{
    std::string tfile(trainfile);
    std::string sfile(savefile ? savefile : "");
    SparseVector<gentype> xtemp;

    return addtrainingdata(mlbase,xtemp,tfile,reverse,ignoreStart,imax,ibase,sfile);
}

int loadFileAndTest(const ML_Base &mlbase, const char *trainfile, int reverse, int ignoreStart, int imax, int ibase,
                    int coercetosingle, int coercefromsingle, int fromsingletarget, int binaryRelabel, int singleDrop, int savex)
{
    std::string tfile(trainfile);
    SparseVector<gentype> xtemp;
    gentype fromsingletarg(fromsingletarget);
    Vector<int> linesread;
    Vector<gentype> ytest;
    Vector<gentype> ytestresh;
    Vector<gentype> ytestresg;
    Vector<gentype> gvarres;
    Vector<int> outkernind;

    return loadFileAndTest(mlbase,xtemp,tfile,reverse,ignoreStart,imax,ibase,
                           coercetosingle,coercefromsingle,fromsingletarg,binaryRelabel,singleDrop,
                           0,linesread,ytest,ytestresh,ytestresg,gvarres,0,outkernind,savex);
}

int loadFileAndSave(const ML_Base &mlbase, const char *trainfile, int reverse, int ignoreStart, int imax, int ibase,
                    int coercetosingle, int coercefromsingle, int fromsingletarget, int binaryRelabel, int singleDrop, const char *savefile)
{
    std::string tfile(trainfile);
    std::string sfile(savefile);
    SparseVector<gentype> xtemp;
    gentype fromsingletarg(fromsingletarget);
    Vector<int> linesread;
    Vector<gentype> ytest;
    Vector<gentype> ytestresh;
    Vector<gentype> ytestresg;
    Vector<gentype> gvarres;
    Vector<int> outkernind;

    return loadFileAndSave(mlbase,xtemp,tfile,reverse,ignoreStart,imax,ibase,
                           coercetosingle,coercefromsingle,fromsingletarg,binaryRelabel,singleDrop,
                           0,linesread,ytest,ytestresh,ytestresg,gvarres,0,outkernind,sfile);
}

// mode = 0: if MLdata == null, load data into xtest,ytest
//        1: load ytest and test xtest
//        2: like 1, but also keep xtest
//        3: like 2, but not testing and save to savefile

int genericMLDataLoad(int binaryRelabel, 
                      int singleDrop, 
                      const ML_Base &mlbase, 
                      const std::string &trainfile, 
                      const std::string &savefile, 
                      int reverse, 
                      int ignoreStart, 
                      int imax, 
                      Vector<SparseVector<gentype> > &xtest, 
                      Vector<gentype> &ytest, 
                      Vector<int> &outkernind,
                      int ibase, 
                      int coercetosingle, 
                      int coercefromsingle, 
                      const gentype &fromsingletarget, 
                      int uselinesvector, 
                      Vector<int> &linesread, 
                      Vector<gentype> &ytestresh, 
                      Vector<gentype> &ytestresg, 
                      Vector<gentype> &gvarres, 
                      const SparseVector<gentype> &xtemplate, 
                      int dovartest = 0, 
                      int mode = 0, 
                      ML_Base *mldest = nullptr);

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, const std::string &savefile)
{
    // These just need to be empty

    gentype fromsingletarget;
    Vector<int> linesread;

    return addtrainingdata(mlbase,
                           xtemplate,
                           trainfile,
                           reverse,
                           ignoreStart,
                           imax,
                           ibase,
                           0,
                           0,
                           fromsingletarget,
                           0,
                           0,
                           0,
                           linesread,
                           savefile);
}

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread, const std::string &savefile)
{
    Vector<SparseVector<gentype> > xtest;
    Vector<gentype> ytest;
    Vector<gentype> ytestresh;
    Vector<gentype> ytestresg;
    Vector<gentype> gvarres;
    Vector<int> outkernind;

    return genericMLDataLoad(binaryRelabel,
                             singleDrop,
                             mlbase,
                             trainfile,
                             savefile,
                             reverse,
                             ignoreStart,
                             imax,
                             xtest,
                             ytest,
                             outkernind,
                             ibase,
                             coercetosingle,
                             coercefromsingle,
                             fromsingletarget,
                             uselinesvector,
                             linesread,
                             ytestresh,
                             ytestresg,
                             gvarres,
                             xtemplate,
                             0,
                             0,
                             static_cast<ML_Base *>(&mlbase));
}

int loadFileForHillClimb(const ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread, Vector<SparseVector<gentype> > &xtest, Vector<gentype> &ytest)
{
    Vector<gentype> ytestresh;
    Vector<gentype> ytestresg;
    Vector<gentype> gvarres;
    int ibase = -1;
    Vector<int> outkernind;
    std::string savefile("");

    return genericMLDataLoad(binaryRelabel,
                             singleDrop,
                             mlbase,
                             trainfile,
                             savefile,
                             reverse,
                             ignoreStart,
                             imax,
                             xtest,
                             ytest,
                             outkernind,
                             ibase,
                             coercetosingle,
                             coercefromsingle,
                             fromsingletarget,
                             uselinesvector,
                             linesread,
                             ytestresh,
                             ytestresg,
                             gvarres,
                             xtemplate,
                             0,
                             0,
                             nullptr);
}

int loadFileAndTest(const ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread, Vector<gentype> &ytest, Vector<gentype> &ytestresh, Vector<gentype> &ytestresg, Vector<gentype> &gvarres, int dovartest, Vector<int> &outkernind, int savex)
{
    Vector<SparseVector<gentype> > xtest;
    std::string savefile("");

    return genericMLDataLoad(binaryRelabel,
                             singleDrop,
                             mlbase,
                             trainfile,
                             savefile,
                             reverse,
                             ignoreStart,
                             imax,
                             xtest,
                             ytest,
                             outkernind,
                             ibase,
                             coercetosingle,
                             coercefromsingle,
                             fromsingletarget,
                             uselinesvector,
                             linesread,
                             ytestresh,
                             ytestresg,
                             gvarres,
                             xtemplate,
                             dovartest,
                             ( savex ? 2 : 1 ),
                             nullptr);
}

int loadFileAndSave(const ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread, Vector<gentype> &ytest, Vector<gentype> &ytestresh, Vector<gentype> &ytestresg, Vector<gentype> &gvarres, int dovartest, Vector<int> &outkernind, const std::string &savefile)
{
    Vector<SparseVector<gentype> > xtest;

    return genericMLDataLoad(binaryRelabel,
                             singleDrop,
                             mlbase,
                             trainfile,
                             savefile,
                             reverse,
                             ignoreStart,
                             imax,
                             xtest,
                             ytest,
                             outkernind,
                             ibase,
                             coercetosingle,
                             coercefromsingle,
                             fromsingletarget,
                             uselinesvector,
                             linesread,
                             ytestresh,
                             ytestresg,
                             gvarres,
                             xtemplate,
                             dovartest,
                             3,
                             nullptr);
}



int genericMLDataLoad(int binaryRelabel,
                      int singleDrop,
                      const ML_Base &mlbase,
                      const std::string &trainfile,
                      const std::string &savefile,
                      int reverse,
                      int ignoreStart,
                      int imax,
                      Vector<SparseVector<gentype> > &xtest,
                      Vector<gentype> &ytest,
                      Vector<int> &outkernind,
                      int ibase,
                      int coercetosingle,
                      int coercefromsingle,
                      const gentype &fromsingletarget,
                      int uselinesvector,
                      Vector<int> &linesread,
                      Vector<gentype> &ytestresh,
                      Vector<gentype> &ytestresg,
                      Vector<gentype> &gvarres,
                      const SparseVector<gentype> &xtemplate,
                      int dovartest,
                      int mode,
                      ML_Base *mldest)
{
    char realtargtype = 'N';

    Vector<int> linesreadatstart(linesread);

    if ( !mldest ) { realtargtype = mlbase.hOutType();    }
    else           { realtargtype = (*mldest).targType(); }

    if ( coercetosingle && !( realtargtype == 'N' ) )
    {
        STRTHROW("Can't use u suffix as ML is not single class.");
    }

    if ( coercefromsingle && ( realtargtype == 'N' ) )
    {
        STRTHROW("Can't use l suffix as ML is single class.");
    }

    int pointsadded = 0;

    if ( ibase == -1 )
    {
        ibase = mlbase.N();
    }

    std::ifstream datfile(trainfile.c_str());

    if ( !datfile.is_open() )
    {
        STRTHROW("Unable to open training file "+trainfile);
    }

    std::ofstream *destfile = nullptr;

    if ( savefile.length() )
    {
        MEMNEW(destfile,std::ofstream(savefile.c_str()));

        if ( !destfile || !(*destfile).is_open() )
        {
            STRTHROW("Unable to open data save file "+savefile);
        }
    }

    SparseVector<gentype> x,y;
    gentype z;
    Vector<double> xCweigh;
    Vector<double> xepsweigh;
    Vector<int> xd;
    Vector<int> xi;
    int xk;

    double Cweight;
    double epsweight;
    int d;
    std::string buffer;
    int ij = 0;
    int goahead;
    int i = ibase;

    while ( !datfile.eof() && ( ( ij < imax+ignoreStart ) || imax == -1 ) )
    {
        goover:

        if ( uselinesvector && ( linesread.size() == 0 ) )
	{
	    break;
	}

	buffer = "";

	while ( ( buffer.length() == 0 ) && !datfile.eof() )
	{
	    getline(datfile,buffer);
	}

	if ( buffer.length() == 0 )
	{
	    break;
	}

        if ( ( buffer.length() >= 2 ) && ( buffer[0] == '/' ) && ( buffer[1] == '/' ) )
        {
            goto goover;
        }

	goahead = 1;

	if ( uselinesvector )
	{
	    if ( ij < linesread(0) )
	    {
		goahead = 0;
	    }

	    else
	    {
                NiceAssert( linesread(0) == ij );
		linesread.remove(0);
	    }
	}

	if ( goahead && ( ij >= ignoreStart ) )
	{
            // Give feedback on progress

//            if ( !(pointsadded%1000) ) { errstream() << "." << pointsadded; }
//            else                       { errstream() << ".";                }

            if ( !(pointsadded%1000) ) { errstream() << "." << pointsadded; }
            else                       { errstreamunlogged() << "."; nullPrint(errstreamunlogged(),pointsadded,-1); }

            // Load training vector from file

            if ( coercefromsingle || ( !coercetosingle && ( realtargtype == 'N' ) ) )
            {
                // No target given in file

                parselineML_Single(x,Cweight,epsweight,buffer,1); //!(mlbase.xspaceSparse()));

                x.fix(); // get it ready.  This avoids possible non-threadsafe access to vectors

                if ( realtargtype == 'N' )
                {
                    // No target for this type, so make target nullptr

                    z.makeNull();
                }

                else
                {
                    // Target not given by file but given by user

                    z = fromsingletarget;
                }

                if ( z.isValEqnDir() )
                {
                    z.scalarfn_setisscalarfn(1);
                }
            }

            else
            {
                // Target given in file - may or may not actually be used

                parselineML_Generic(z,x,Cweight,epsweight,d,buffer,reverse,1); //!(mlbase.xspaceSparse()));

                x.fix();

                if ( coercetosingle )
                {
                    // No target for this type, so make target nullptr

                    z.makeNull();
                }
            }

            addtemptox(x,xtemplate);

            // Binary relabelling (if any)

            if ( binaryRelabel )
            {
                if ( z.isValInteger() )
                {
                    if ( (int) z == binaryRelabel )
                    {
                        z = +1;
                    }

                    else
                    {
                        z = -1;
                    }
                }
            }

            // Class skipping (if any)

            if ( singleDrop )
            {
                if ( z.isValInteger() )
                {
                    if ( (int) z == singleDrop )
                    {
                        // There is surely a nicer way to do this

                        goto goover;
                    }
                }
            }

            // Use for training vector depends on task

            if ( mldest )
            {
                // Task is to add training vectors to machine

                xtest.add(xtest.size());
                ytest.add(ytest.size());
                xCweigh.add(xCweigh.size());
                xepsweigh.add(xepsweigh.size());
                xd.add(xd.size());
                xi.add(xi.size());

                qswap(xtest("&",xtest.size()-1),x);
                ytest.set(ytest.size()-1,z);
                xCweigh.sv(xCweigh.size()-1,Cweight);
                xepsweigh.sv(xepsweigh.size()-1,epsweight);
                xd.sv(xd.size()-1,d);
                xi.sv(xi.size()-1,i);
            }

            else if ( 0 == mode )
            {
                // Task is to load data into (x,y) vectors

                xtest.add(xtest.size());
                ytest.add(ytest.size());

                qswap(xtest("&",xtest.size()-1),x);
                ytest.set(ytest.size()-1,z);
            }

            else if ( ( 1 == mode ) || ( 2 == mode ) )
            {
                // Task is to run tests.  Do not save x as testing file may
                // be very large, but do keep everything else for reporting.

                ytest.add(ytest.size());
                ytestresh.add(ytestresh.size());
                ytestresg.add(ytestresg.size());
                outkernind.add(outkernind.size());

                outkernind.sv(outkernind.size()-1, x.isf4indpresent(3) ? (int) x.f4(3) : -1 );

                if ( dovartest )
                {
                    gvarres.add(gvarres.size());
                }

                ytest.set(ytest.size()-1,z);

                mlbase.gh(ytestresh("&",ytestresh.size()-1),ytestresg("&",ytestresg.size()-1),x);

                if ( dovartest )
                {
                    gentype dummy;

                    mlbase.var(gvarres("&",gvarres.size()-1),dummy,x);
                }

                if ( 2 == mode )
                {
                    xtest.add(xtest.size());
                    qswap(xtest("&",xtest.size()-1),x);
                }
            }

            else
            {
                NiceAssert( 3 == mode );

                // save z,x

                *destfile << z << "\t";
                printnoparen(*destfile,x);
                *destfile << "\n";
            }

            // Update counters and such

            ++pointsadded;
	    ++i;
	}

        // Update counters and such

	++ij;
    }

    datfile.close();

    errstream() << "." << pointsadded << "...";

    // If task is to add training data to ML then do this now blockwise
    // (blockwise preferable as it is much faster if there is variable
    // autosetting setup in the ML)

    if ( mldest && xi.size() )
    {
        if ( isSVMMvRank(mlbase) )
        {
            errstream() << "*";

            // Need to have d set immediately, so use alternative method

            int Nnew = (dynamic_cast<SVM_MvRank &>(mldest->getML())).N() + xi.size();

            if ( (dynamic_cast<SVM_MvRank &>(mldest->getML())).preallocsize() < Nnew )
            {
                (dynamic_cast<SVM_MvRank &>(mldest->getML())).prealloc(Nnew+1);
            }

            for ( i = 0 ; i < xi.size() ; ++i )
            {
                (dynamic_cast<SVM_MvRank &>(mldest->getML())).qaddTrainingVector(xi(i),(double) ytest(i),xtest("&",i),xCweigh(i),xepsweigh(i),xd(i));
            }

            errstream() << "#";
        }

        else if ( isLSVMvRank(mlbase) )
        {
            errstream() << "*";

            // Need to have d set immediately, so use alternative method

            int Nnew = (dynamic_cast<LSV_MvRank &>(mldest->getML())).N() + xi.size();

            if ( (dynamic_cast<LSV_MvRank &>(mldest->getML())).preallocsize() < Nnew )
            {
                (dynamic_cast<LSV_MvRank &>(mldest->getML())).prealloc(Nnew+1);
            }

            for ( i = 0 ; i < xi.size() ; ++i )
            {
                (dynamic_cast<LSV_MvRank &>(mldest->getML())).qaddTrainingVector(xi(i),(double) ytest(i),xtest("&",i),xCweigh(i),xepsweigh(i),xd(i));
            }

            errstream() << "#";
        }

        else
        {
            errstream() << "*";

            xk = xi(0);

            int Nnew = (*mldest).N() + xi.size();

            if ( (*mldest).preallocsize() < Nnew )
            {
                errstream() << "!";
                (*mldest).prealloc(Nnew+1);
                errstream() << ".";
            }

// The following was a test I added while debugging the altcontent acceleration of inner product calculation (see sparsevector.h and gentype.cc).
// It is no longer necessary but I retained it (commented out for speed) as a backstop just in case.  All things being equal it does precisely nothing.
//errstream() << "phantomx fscking start\n";
//for ( i = 0 ; i < xi.size() ; ++i )
//{
//xtest("&",i).makealtcontent();
//}
//errstream() << "phantomx fscking add\n";
            errstream() << "?";
            (*mldest).qaddTrainingVector(xk,ytest,xtest,xCweigh,xepsweigh);
            errstream() << ",";
//errstream() << "phantomx fscking stop\n";

            errstream() << "#";

            //if ( isSVMScalar(mlbase) || isSVMVector(mlbase) || isSVMPlanar(mlbase) || isLSVScalar(mlbase) || isGPRScalar(mlbase) || isSSVScalar(mlbase) || isSVMSimLrn(mlbase) )
            if ( isSVMScalar(mlbase) || isSVMVector(mlbase) || isSVMPlanar(mlbase) || isLSVScalar(mlbase) || isLSVVector(mlbase) || isGPRScalar(mlbase) || isGPRVector(mlbase) || isSVMSimLrn(mlbase) )
            {
                for ( i = 0 ; i < xi.size() ; ++i )
                {
                    xk = xi(i);

                    if ( xd(i) != 2 )
                    {
                        (*mldest).setd(xk,xd(i));
                    }
                }
            }
        }
    }

    errstream() << "\n";

    if ( uselinesvector == 2 )
    {
        linesread = linesreadatstart;
    }

    if ( destfile )
    {
        MEMDEL(destfile); destfile = nullptr;
    }

    return pointsadded;
}

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, Vector<SparseVector<gentype> > &x, const Vector<gentype> &yy, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget)
{
    Vector<gentype> sigmaweight(x.size());

    sigmaweight = 1.0_gent; //onedblgentype();

    return addtrainingdata(mlbase,xtemplate,x,yy,sigmaweight,ibase,coercetosingle,coercefromsingle,fromsingletarget);
}

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, Vector<SparseVector<gentype> > &x, const Vector<gentype> &yy, const Vector<gentype> &sigmaweight, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget)
{
    Vector<gentype> y(yy);

    if ( coercetosingle && !( mlbase.targType() == 'N' ) )
    {
        STRTHROW("Can't use u suffix as ML is not single class.");
    }

    if ( coercefromsingle && ( mlbase.targType() == 'N' ) )
    {
        STRTHROW("Can't use l suffix as ML is single class.");
    }

    if ( coercefromsingle )
    {
        y.resize(x.size());
        y.zero();
        y = fromsingletarget;
    }

    NiceAssert( x.size() == y.size() );

    if ( ibase == -1 )
    {
        ibase = mlbase.N();
    }

    Vector<double> Cweight(sigmaweight.size());
    Vector<double> weightdummy(y.size());

    weightdummy = 1.0;

    if ( x.size() )
    {
        int i;

        for ( i = 0 ; i < x.size() ; ++i )
        {
            if ( x(i).indsize() )
            {
                int iji;

                for ( iji = 0 ; iji < x(i).indsize() ; ++iji )
                {
                    if ( (x(i).direcref(iji)).isValEqnDir() )
                    {
                        (x("&",i).direref(iji)).scalarfn_setisscalarfn(1);
                    }
                }
            }
        }
    }

    if ( y.size() )
    {
        int i;

        for ( i = 0 ; i < y.size() ; ++i )
        {
            if ( y(i).isValEqnDir() )
            {
                y("&",i).scalarfn_setisscalarfn(1);
            }
        }
    }

    if ( sigmaweight.size() )
    {
        int i;

        for ( i = 0 ; i < sigmaweight.size() ; ++i )
        {
            Cweight.sv(i, 1.0/( ( ((double) sigmaweight(i)) < MINSWEIGHT ) ? MINSWEIGHT : ((double) sigmaweight(i)) ) );
        }
    }

    int Nnew = mlbase.N() + x.size();

    if ( mlbase.preallocsize() < Nnew )
    {
        mlbase.prealloc(Nnew+1);
    }

    addtemptox(x,xtemplate);

    mlbase.qaddTrainingVector(ibase,y,x,Cweight,weightdummy);

    return x.size();
}




int addbasisdataUU(ML_Base &dest, const std::string &fname)
{
    int pointsadded = 0;

    std::ifstream  srcfile;

    srcfile.open(fname.c_str(),std::ofstream::in);

    if ( !srcfile.is_open() )
    {
        std::string errstring;
        errstring = "Unable to open basis file "+fname;
        NiceThrow(errstring);
    }

    std::string buffer;
    gentype tempbasevec;

    while ( !srcfile.eof() )
    {
        buffer = "";

        while ( ( buffer.length() == 0 ) && !srcfile.eof() )
        {
            getline(srcfile,buffer);
        }

        if ( buffer.length() == 0 )
        {
            break;
        }

        std::stringstream transit;

        transit << buffer;
        transit >> tempbasevec;

        if ( tempbasevec.isValEqnDir() )
        {
            tempbasevec.scalarfn_setisscalarfn(1);
        }

        dest.addToBasisUU(dest.NbasisUU(),tempbasevec);
        ++pointsadded;
    }

    srcfile.close();

    return pointsadded;
}

int addbasisdataVV(ML_Base &dest, const std::string &fname)
{
    int pointsadded = 0;

    std::ifstream  srcfile;

    srcfile.open(fname.c_str(),std::ofstream::in);

    if ( !srcfile.is_open() )
    {
        std::string errstring;
        errstring = "Unable to open basis file "+fname;
        NiceThrow(errstring);
    }

    std::string buffer;
    gentype tempbasevec;

    while ( !srcfile.eof() )
    {
        buffer = "";

        while ( ( buffer.length() == 0 ) && !srcfile.eof() )
        {
            getline(srcfile,buffer);
        }

        if ( buffer.length() == 0 )
        {
            break;
        }

        std::stringstream transit;

        transit << buffer;
        transit >> tempbasevec;

        if ( tempbasevec.isValEqnDir() )
        {
            tempbasevec.scalarfn_setisscalarfn(1);
        }

        dest.addToBasisVV(dest.NbasisVV(),tempbasevec);
        ++pointsadded;
    }

    srcfile.close();

    return pointsadded;
}




SparseVector<gentype> &addtemptox(SparseVector<gentype> &x, const SparseVector<gentype> &xtemp)
{
    if ( xtemp.indsize() )
    {
        int i,ii;

        for ( i = 0 ; i < xtemp.indsize() ; ++i )
        {
            ii = xtemp.ind(i);

            if ( !(x.isindpresent(ii)) )
            {
                x("&",ii) = xtemp.direcref(i);
            }
        }
    }

    return x;
}

Vector<SparseVector<gentype> > &addtemptox(Vector<SparseVector<gentype> > &x, const SparseVector<gentype> &xtemp)
{
    if ( x.size() )
    {
        int i;

        for ( i = 0 ; i < x.size() ; ++i )
        {
            addtemptox(x("&",i),xtemp);
        }
    }

    return x;
}
