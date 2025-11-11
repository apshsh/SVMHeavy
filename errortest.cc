
//
// Performance/error testing routines
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "errortest.hpp"
#include "qswapbase.hpp"
#include "ml_mutable.hpp"
#include "svm_scalar_rff.hpp"
#include "svm_binary.hpp"
#include "svm_single.hpp"
#include "randfun.hpp"

double calcLOORecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int isLOO, int calcgvarres, int suppressfb = 0, int useThreads = 0);

static void disableVector(int i, ML_Base *activeML);
static void disableVector(const Vector<int> &i, ML_Base *activeML);
static void resetGlobal(ML_Base *activeML);
static void semicopyML(ML_Base *activeML, const ML_Base *srcML);
static void copyML(ML_Base *activeML, const ML_Base *srcML);
static int trainGlobal(int &res, ML_Base *activeML, int islastopt);
static int isVectorActive(int i, ML_Base *activeML);
static int isVectorEnabled(int i, ML_Base *activeML);
static int isTrainedML(ML_Base *activeML);

double calcnegloglikelihood(const ML_Base &baseML, int suppressfb)
{
    (void) suppressfb;

    double res = baseML.loglikelihood();

//    if ( isSVM(baseML) )
//    {
//        res = (dynamic_cast<const SVM_Generic &>(baseML)).quasiloglikelihood();
//    }
//
//    else if ( isGPR(baseML) )
//    {
//        res = (dynamic_cast<const GPR_Generic &>(baseML)).loglikelihood();
//    }
//
//    else if ( isLSV(baseML) )
//    {
//        res = (dynamic_cast<const LSV_Generic &>(baseML)).lsvloglikelihood();
//    }
//
//    else
//    {
//        res = valvnan(); // We want to be able to log the likelihoods in smboopt without worrying about the code throwing an error
//        //NiceThrow("Log-likelihood not defined for this ML type");
//    }

    return -res;
}

double calcmaxinfogain(const ML_Base &baseML, int suppressfb)
{
    (void) suppressfb;

    double res = baseML.maxinfogain();

//    if ( isSVM(baseML) )
//    {
//        res = (dynamic_cast<const SVM_Generic &>(baseML)).quasimaxinfogain();
//    }
//
//    else if ( isGPR(baseML) )
//    {
//        res = (dynamic_cast<const GPR_Generic &>(baseML)).maxinfogain();
//    }
//
//    else if ( isLSV(baseML) )
//    {
//        res = (dynamic_cast<const LSV_Generic &>(baseML)).lsvmaxinfogain();
//    }
//
//    else
//    {
//        res = valvnan(); // We want to be able to log the likelihoods in smboopt without worrying about the code throwing an error
//        //NiceThrow("Log-likelihood not defined for this ML type");
//    }

    return res;
}

double calcRKHSnorm(const ML_Base &baseML, int suppressfb)
{
    (void) suppressfb;

    double res = baseML.RKHSnorm();

//    if ( isSVM(baseML) )
//    {
//        res = (dynamic_cast<const SVM_Generic &>(baseML)).svmRKHSnorm();
//    }
//
//    else if ( isGPR(baseML) )
//    {
//        res = (dynamic_cast<const GPR_Generic &>(baseML)).gprRKHSnorm();
//    }
//
//    else if ( isLSV(baseML) )
//    {
//        res = (dynamic_cast<const LSV_Generic &>(baseML)).lsvRKHSnorm();
//    }
//
//    else
//    {
//        res = valvnan(); // We want to be able to log the likelihoods in smboopt without worrying about the code throwing an error
//        //NiceThrow("Log-likelihood not defined for this ML type");
//    }

    return res;
}


double calcLOO(const ML_Base &baseML, int startpoint, int suppressfb, int useThreads)
{
    Vector<int> cnt;
    Matrix<int> cfm;

    return calcLOO(baseML,cnt,cfm,startpoint,suppressfb,useThreads);
}

double calcRecall(const ML_Base &baseML, int startpoint, int suppressfb, int useThreads)
{
    Vector<int> cnt;
    Matrix<int> cfm;

    return calcRecall(baseML,cnt,cfm,startpoint,suppressfb,useThreads);
}

double calcCross(const ML_Base &baseML, int m, int rndit, int numreps, int startpoint, int suppressfb, int useThreads)
{
    Vector<double> repres;
    Vector<double> cnt;
    Matrix<double> cfm;

    return calcCross(baseML,m,rndit,repres,cnt,cfm,numreps,startpoint,suppressfb,useThreads);
}

double CalcSparSens(const ML_Base &baseML, int minbad, int maxbad, double noisemean, double noisevar, int startpoint, int suppressfb, int useThreads)
{
    Vector<double> repres;
    Vector<double> cnt;
    Matrix<double> cfm;

    return calcSparSens(baseML,repres,cnt,cfm,minbad,maxbad,noisemean,noisevar,startpoint,suppressfb,useThreads);
}

double calcLOO(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, int startpoint, int suppressfb, int useThreads)
{
    Vector<gentype> resh;
    Vector<gentype> resg;
    Vector<gentype> gvarres;

    return calcLOO(baseML,cnt,cfm,resh,resg,gvarres,startpoint,0,suppressfb,useThreads);
}

double calcRecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, int startpoint, int suppressfb, int useThreads)
{
    Vector<gentype> resh;
    Vector<gentype> resg;
    Vector<gentype> gvarres;

    return calcRecall(baseML,cnt,cfm,resh,resg,gvarres,startpoint,0,suppressfb,useThreads);
}

double calcCross(const ML_Base &baseML, int m, int rndit, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, int numreps, int startpoint, int suppressfb, int useThreads)
{
    Vector<Vector<gentype> > resh;
    Vector<Vector<gentype> > resg;
    Vector<Vector<gentype> > gvarres;

    return calcCross(baseML,m,rndit,repres,cnt,cfm,resh,resg,gvarres,numreps,startpoint,0,suppressfb,useThreads);
}

double calcSparSens(const ML_Base &baseML, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, int minbad, int maxbad, double noisemean, double noisevar, int startpoint, int suppressfb, int useThreads)
{
    Vector<Vector<gentype> > resh;
    Vector<Vector<gentype> > resg;
    Vector<Vector<gentype> > gvarres;

    return calcSparSens(baseML,repres,cnt,cfm,resh,resg,gvarres,minbad,maxbad,noisemean,noisevar,startpoint,0,suppressfb,useThreads);
}

double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, int startpoint, int suppressfb, int useThreads)
{
    Vector<int> cnt;
    Matrix<int> cfm;

    return calcTest(baseML,xtest,ytest,cnt,cfm,startpoint,suppressfb,useThreads);
}

double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, Vector<int> &cnt, Matrix<int> &cfm, int startpoint, int suppressfb, int useThreads)
{
    Vector<gentype> resh;
    Vector<gentype> resg;
    Vector<gentype> gvarres;

    return calcTest(baseML,xtest,ytest,cnt,cfm,resh,resg,gvarres,0,startpoint,suppressfb,useThreads);
}

double calcLOO(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int calcvarres, int suppressfb, int useThreads)
{
    return calcLOORecall(baseML,cnt,cfm,resh,resg,gvarres,startpoint,1,calcvarres,suppressfb,useThreads);
}

double calcRecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int calcvarres, int suppressfb, int useThreads)
{
    return calcLOORecall(baseML,cnt,cfm,resh,resg,gvarres,startpoint,0,calcvarres,suppressfb,useThreads);
}




int ishzero(const gentype &h)
{
    int res = 0;

    if ( h.isValInteger() && ( 0 == (int) h ) )
    {
        res = 1;
    }

    return res;
}




















class foldargs;

void qswap(const foldargs *&a, const foldargs *&b);
void qswap(foldargs *&a, foldargs *&b);

class foldargs
{
    public:

    foldargs(ML_Base *locML, bool trainglobcond, Vector<gentype> &kresh, Vector<gentype> &kresg,
             Vector<gentype> &kgvarres, Vector<double> &loctmpcnt, Matrix<double> &loctmpcfm,
             double &loctmpres, int &locNnonz, const Vector<int> &locblockidvect, int Nclasses,
             const Vector<int> &locisenabled, const Vector<gentype> &locy, int startpoint, int calcgvarres,
             const svm_pthread_id &xmainid)
    : xlocML(locML),
      xtrainglobcond(trainglobcond),
      xkresh(kresh),
      xkresg(kresg),
      xkgvarres(kgvarres),
      xloctmpcnt(loctmpcnt),
      xloctmpcfm(loctmpcfm),
      xloctmpres(loctmpres),
      xlocNnonz(locNnonz),
      xlocblockidvect(locblockidvect),
      xNclasses(Nclasses),
      xlocisenabled(locisenabled),
      xlocy(locy),
      xstartpoint(startpoint),
      xcalcgvarres(calcgvarres),
      mainid(xmainid),
      foldres(0)
      { ; }


    ML_Base *xlocML;
    bool xtrainglobcond;
    Vector<gentype> &xkresh;
    Vector<gentype> &xkresg;
    Vector<gentype> &xkgvarres;
    Vector<double> &xloctmpcnt;
    Matrix<double> &xloctmpcfm;
    double &xloctmpres;
    int &xlocNnonz;
    const Vector<int> &xlocblockidvect;
    int xNclasses;
    const Vector<int> &xlocisenabled;
    const Vector<gentype> &xlocy;
    int xstartpoint;
    int xcalcgvarres;
    const svm_pthread_id &mainid;
    int foldres;
};

void qswap(const foldargs *&a, const foldargs *&b)
{
    const foldargs *c(a); a = b; b = c;
}

void qswap(foldargs *&a, foldargs *&b)
{
    foldargs *c(a); a = b; b = c;
}


//inline int *&setident (int *&a);
//inline int *&setzero  (int *&a);
//inline int *&setposate(int *&a);
//inline int *&setnegate(int *&a);
//inline int *&setconj  (int *&a);
//inline int *&setrand  (int *&a);

inline foldargs *&setident (foldargs *&a);
inline foldargs *&setzero  (foldargs *&a);
inline foldargs *&setposate(foldargs *&a);
inline foldargs *&setnegate(foldargs *&a);
inline foldargs *&setconj  (foldargs *&a);
inline foldargs *&setrand  (foldargs *&a);

//inline int *&setident (int *&a) { return a; }
//inline int *&setzero  (int *&a) { return a; }
//inline int *&setposate(int *&a) { return a; }
//inline int *&setnegate(int *&a) { return a; }
//inline int *&setconj  (int *&a) { return a; }
//inline int *&setrand  (int *&a) { return a; }

inline foldargs *&setident (foldargs *&a) { return a; }
inline foldargs *&setzero  (foldargs *&a) { return a = nullptr; }
inline foldargs *&setposate(foldargs *&a) { return a; }
inline foldargs *&setnegate(foldargs *&a) { return a; }
inline foldargs *&setconj  (foldargs *&a) { return a; }
inline foldargs *&setrand  (foldargs *&a) { return a; }



void *doaprep(void *args_ptr);
void *doafold(void *args_ptr);
void *doatest(void *args_ptr);


double calcCross(const ML_Base &baseML, int m, int rndit, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, Vector<Vector<gentype> > &resh, Vector<Vector<gentype> > &resg, 
                 Vector<Vector<gentype> > &gvarres, int numreps, int startpoint, int calcgvarres, int suppressfb, int useThreads)
{

#ifndef ENABLE_THREADS
    useThreads = 0;
#endif

    if ( !baseML.isSolGlob() )
    {
        if ( startpoint == 0 )
        {
            startpoint = 1; // start from reset state no matter what
        }

        if ( startpoint == 2 )
        {
            startpoint = 3; // start from reset state no matter what
        }
    }

    startpoint &= 3;

    NiceAssert( numreps > 0 );

    int oneoff = startpoint & 2;
    startpoint &= 1;

    if ( oneoff )
    {
        // This option is intended to simulate a single random validation set over multiple runs, so it is important that it be the *same* random sequence every time

        //svm_srand(42);
        srand(42);
        double dodum = 0;
        randfill(dodum,'S',42);
    }

    //startpoint = startpoint || isONN(baseML);

    int i,j,k,rndsel;
    double res = 0;
    int N = baseML.N();
    int Nnz = baseML.N()-baseML.NNC(0);
    int Nclasses = baseML.numInternalClasses();

    if ( m > Nnz )
    {
	m = Nnz;
    }

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    repres.resize(numreps);
    repres.zero();

    resg.resize(numreps);
    resh.resize(numreps);

    if ( calcgvarres )
    {
        gvarres.resize(numreps);
    }

    for ( i = 0 ; i < numreps ; ++i )
    {
        resg("&",i).resize(N);
        resg("&",i).zero();

        resh("&",i).resize(N);
        resh("&",i).zero();

        if ( calcgvarres )
        {
            gvarres("&",i).resize(N);
            gvarres("&",i).zero();
        }
    }

    if ( Nnz )
    {
	Vector<double> loccnt(cnt);
	Matrix<double> loccfm(cfm);
        double locres = 0.0;

        ML_Base *locML = (ML_Base *) &baseML;
        ML_Base *srcML = (ML_Base *) &baseML;

        locML = makeNewML(baseML.type(),baseML.subtype());
        NiceAssert( locML );
        semicopyML(locML,&baseML);

        ML_Base *temp = srcML; srcML = locML; locML = temp;
//errstream() << "phantomxyz: " << (*locML).N() << "\n";

        Vector<int> locisenabled(N);
        Vector<gentype> locy = locML->y();

        Vector<gentype> dummygv;

        for ( i = 0 ; i < N ; ++i )
        {
            locisenabled("&",i) = isVectorEnabled(i,locML);
        }

	for ( k = 0 ; k < numreps ; ++k )
	{
	    // Work out cross-fold groups

            loccnt.zero();
            loccfm.zero();
            locres = 0.0;

	    Vector<int> countvect(Nnz);

            j = 0;

            for ( i = 0 ; i < N ; ++i )
	    {
                if ( locisenabled(i) )
		{
                    countvect("&",j++) = i;
		}
	    }

	    Vector<Vector<int> > blockidvect(m);

            int Nnzleft = Nnz;

	    for ( i = 0 ; i < m ; ++i )
	    {
		blockidvect("&",i).resize(Nnzleft/(m-i));

		for ( j = 0 ; j < Nnzleft/(m-i) ; ++j )
		{
		    rndsel = 0;

		    if ( rndit )
		    {
			//rndsel = svm_rand()%(countvect.size());
			rndsel = rand()%(countvect.size());
		    }

		    (blockidvect("&",i))("&",j) = countvect(rndsel);
		    countvect.remove(rndsel);
		}

                Nnzleft -= (Nnzleft/(m-i));
	    }

            NiceAssert( !(countvect.size()) );

            int Nnonz = 0;

            Vector<Vector<double> > vtmpcnt(m);
            Vector<Matrix<double> > vtmpcfm(m);
            Vector<double> vtmpres(m);
            Vector<int> vNnonz(m);


            Vector<ML_Base *> vlocML(m);
            Vector<foldargs *> vxargs(m);
            Vector<svm_pthread_t *> vprepthread(m);
            Vector<svm_pthread_t *> vfoldthread(m);
            Vector<svm_pthread_t *> vtestthread(m);

            svm_pthread_id thisthread = svm_pthread_self();

            vlocML = locML;

            //isMainThread(1); don't do this

            if ( useThreads )
            {
                errstream() << "ALERT: threads enabled\n";

                bool extraFast = false;

                int xvloctype = (*locML).type();

                //if ( ( xvloctype == 0 ) || ( xvloctype == 1 ) || ( xvloctype == 2 ) || ( xvloctype == 7 ) )
                if ( ( xvloctype == 0 ) || ( xvloctype == 1 ) || ( xvloctype == 2 ) || ( xvloctype == 7 ) || ( xvloctype == 22 ) || ( xvloctype == 23 ) )
                {
                    SVM_Scalar &noreallythisone = dynamic_cast<SVM_Scalar &>((*locML).getML());

                    // For svm_scalar and a limited number of variants of it (simple
                    // machines deriving from it, used without some more obscure functionality)
                    // we can operate with a single kcache and a single Gp for added speed.
                    // The following flag is set if we can do this.  See SVM_Scalar (and in
                    // particular the assign( function) for more information on exactly what
                    // gets bypassed, and also below for what extra functions we have to do
                    // manually before and after optimisation to make it work.

                    extraFast = ( !noreallythisone.usefuzzt() &&
                                  ( noreallythisone.m() == 2 ) &&
                                  noreallythisone.isNoMonotonicConstraints() &&
                                  !noreallythisone.is1NormCost() &&
                                  noreallythisone.isFixedTube() );

                    if ( extraFast )
                    {
                        errstream() << "ALERT: fast parallel Gp access enabled\n";

                        (*locML).setmemsize(-1);

                        noreallythisone.delaydownsize = 1; // Trigger Gp redirection

                        if ( ( xvloctype == 22 ) || ( xvloctype == 23 ) )
                        {
                            SVM_Scalar_rff &maybethisone = dynamic_cast<SVM_Scalar_rff &>((*locML).getML());

//need to do this first in errortest so that all versions in cross-fold share the *same* random features (which are stored in x, which is shared)
                            maybethisone.fixupfeatures();
                        }
                    }
                }

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( !i )
                    {
                        vlocML("&",i) = locML;
                    }

                    else
                    {
                        vlocML("&",i) = makeDupML(*locML,locML);
                    }

                    vtmpcnt("&",i) = cnt;
                    vtmpcfm("&",i) = cfm;
                    vtmpres("&",i) = 0.0;
                    vNnonz("&",i)  = 0;

                    bool trainglobcond = ( ( i == m-1 ) && ( k == numreps-1 ) );

                    MEMNEW(vxargs("&",i),foldargs(vlocML("&",i),trainglobcond,resh("&",k),resg("&",k),( calcgvarres ? gvarres("&",k) : dummygv ),vtmpcnt("&",i),vtmpcfm("&",i),vtmpres("&",i),vNnonz("&",i),blockidvect(i),Nclasses,locisenabled,locy,startpoint,calcgvarres,thisthread));
                }

                if ( extraFast )
                {
                    SVM_Scalar &noreallythisone = dynamic_cast<SVM_Scalar &>((*locML).getML());

                    noreallythisone.delaydownsize = 0; // Trigger Gp redirection
                }

                // Fold preparation

                errstream() << "ALERT: parallel fold preparation begun\n";

                // We do the main fold first to make sure that Gp is "fixed" for remaining folds

                if ( extraFast )
                {
                    (*locML).fillCache();
                }

                //doaprep((void *) vxargs("&",0));

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( !suppressfb )
                    {
                        nullPrint(errstream(),"~~~~~",5);
                        nullPrint(errstream(),i,-5);
                    }

                    if ( i )
                    {
                        MEMNEW(vprepthread("&",i),svm_pthread_t);

                        int tfail = svm_pthread_create(vprepthread("&",i),doaprep,(void *) vxargs("&",i));
                        if ( tfail ) { return valvnan(); }
                    }
                }

                doaprep((void *) vxargs("&",0));

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( i )
                    {
                        int tfail = svm_pthread_join(*vprepthread("&",i),nullptr);
                        if ( tfail ) { return valvnan(); }
                    }
	        }

                // Fold pre-training

                if ( extraFast )
                {
                    errstream() << "ALERT: parallel Gp preset begun\n";

                    for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                    {
                        int vloctype = (*(vlocML(i))).type();

                        if ( i )
                        {
                            if ( ( vloctype == 0 ) || ( vloctype == 1 ) || ( vloctype == 2 ) || ( vloctype == 7 ) || ( vloctype == 22 ) || ( vloctype == 23 ) )
                            {
                                SVM_Scalar &noreallythisone = dynamic_cast<SVM_Scalar &>((*(vlocML("&",i))).getML());

                                noreallythisone.delaydownsize = 1;
                            }
                        }

                        else
                        {
                            if ( ( vloctype == 0 ) || ( vloctype == 1 ) || ( vloctype == 2 ) || ( vloctype == 7 ) || ( vloctype == 22 ) || ( vloctype == 23 ) )
                            {
                                SVM_Scalar &noreallythisone = dynamic_cast<SVM_Scalar &>((*locML).getML());

                                noreallythisone.delaydownsize = 1;

                                if ( ( vloctype == 0 ) || ( vloctype == 1 ) || ( vloctype == 2 ) || ( vloctype == 7 ) )
                                {
                                    noreallythisone.upsizenow();
                                }
                            }
                        }
                    }
                }

                // Fold training

                errstream() << "ALERT: parallel fold training begun\n";

                //// We do the main fold first to make sure that Gp is "fixed" for remaining folds
                //
                //doafold((void *) vxargs("&",0));

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( !suppressfb )
                    {
                        nullPrint(errstream(),"@@@@@",5);
                        nullPrint(errstream(),i,-5);
                    }

                    if ( i )
                    {
                        MEMNEW(vfoldthread("&",i),svm_pthread_t);

                        int tfail = svm_pthread_create(vfoldthread("&",i),doafold,(void *) vxargs("&",i));
                        if ( tfail ) { return valvnan(); }
                    }
                }

                doafold((void *) vxargs("&",0));
                if ( vxargs(0)->foldres ) { return valvnan(); }

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( i )
                    {
                        int tfail = svm_pthread_join(*vfoldthread("&",i),nullptr);
                        if ( tfail || vxargs(i)->foldres ) { return valvnan(); }
                    }
	        }

                if ( locres ) { return valvnan(); }

                // Fold cleanup

                if ( extraFast )
                {
                    int xxvloctype = (*locML).type();

                    if ( ( xxvloctype == 0 ) || ( xxvloctype == 1 ) || ( xxvloctype == 2 ) || ( xxvloctype == 7 ) || ( xxvloctype == 22 ) || ( xxvloctype == 23 ) )
                    {
                        SVM_Scalar &noreallythisone = dynamic_cast<SVM_Scalar &>((*locML).getML());

                        if ( ( xxvloctype == 0 ) || ( xxvloctype == 1 ) || ( xxvloctype == 2 ) || ( xxvloctype == 7 ) )
                        {
                            noreallythisone.downsizenow();
                        }

                        noreallythisone.delaydownsize = 0;
                    }
                }

                // Fold testing

                errstream() << "ALERT: parallel fold testing begun\n";

                // We do the main fold first to make sure that Gp is "fixed" for remaining folds

                //doatest((void *) vxargs("&",0));
                //semicopyML(locML,srcML);

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( !suppressfb )
                    {
                        nullPrint(errstream(),"?????",5);
                        nullPrint(errstream(),i,-5);
                    }

                    if ( i )
                    {
                        MEMNEW(vtestthread("&",i),svm_pthread_t);

                        int tfail = svm_pthread_create(vtestthread("&",i),doatest,(void *) vxargs("&",i));
                        if ( tfail ) { return valvnan(); }
                    }
                }

                doatest((void *) vxargs("&",0));
                semicopyML(locML,srcML);

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( i )
                    {
                        int tfail = svm_pthread_join(*vtestthread("&",i),nullptr);
                        if ( tfail ) { return valvnan(); }
                    }
	        }

                semicopyML(locML,srcML);

                // Final cleanup

                errstream() << "ALERT: final cleanup begun\n";

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( i )
                    {
                        MEMDEL(vprepthread("&",i)); vprepthread("&",i) = nullptr;
                        MEMDEL(vfoldthread("&",i)); vfoldthread("&",i) = nullptr;
                        MEMDEL(vlocML("&",i));      vlocML("&",i)      = nullptr;
                    }

                    MEMDEL(vxargs("&",i)); vxargs("&",i) = nullptr;
                }
            }

            else
            {
                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    vtmpcnt("&",i) = cnt;
                    vtmpcfm("&",i) = cfm;
                    vtmpres("&",i) = 0.0;
                    vNnonz("&",i)  = 0;

                    bool trainglobcond = ( ( i == m-1 ) && ( k == numreps-1 ) );

                    MEMNEW(vxargs("&",i),foldargs(vlocML("&",i),trainglobcond,resh("&",k),resg("&",k),( calcgvarres ? gvarres("&",k) : dummygv ),vtmpcnt("&",i),vtmpcfm("&",i),vtmpres("&",i),vNnonz("&",i),blockidvect(i),Nclasses,locisenabled,locy,startpoint,calcgvarres,thisthread));
                }

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    if ( !suppressfb )
                    {
                        nullPrint(errstream(),"~~~~~",5);
                        nullPrint(errstream(),i,-5);
                    }

                    doaprep((void *) vxargs("&",i));
                    doafold((void *) vxargs("&",i));
                    if ( vxargs(0)->foldres ) { return valvnan(); }
                    doatest((void *) vxargs("&",i));
                    semicopyML(locML,srcML);
                }

                for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
                {
                    MEMDEL(vxargs("&",i)); vxargs("&",i) = nullptr;
                }
            }

            // Tally results

            for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; ++i )
            {
		loccnt += vtmpcnt(i);
		loccfm += vtmpcfm(i);
                locres += vtmpres(i);
                Nnonz  += vNnonz(i);
	    }

            locres = Nnonz ? locres/Nnonz : res;

            if ( baseML.isRegression() )
            {
                locres = sqrt(locres);
            }

	    cnt += loccnt;
	    cfm += loccfm;
	    res += locres;

            repres("&",k) = locres;
	}

        if ( !oneoff )
        {
            cnt *= 1/((double) numreps);
            cfm *= 1/((double) numreps);
            res *= 1/((double) numreps);
        }

        { ML_Base *ttemp = srcML; srcML = locML; locML = ttemp; }

        MEMDEL(locML); locML = nullptr;
    }

//    if ( baseML.isRegression() )
//    {
//        res = sqrt(res);
//    }

if ( !suppressfb )
{
errstream() << "\n";
}
    return res;
}





void *doaprep(void *args_ptr)
{
    foldargs &args = *((foldargs *) args_ptr);

    ML_Base *locML = args.xlocML;
    //bool trainglobcond = args.xtrainglobcond;
    //Vector<gentype> &kresh = args.xkresh;
    //Vector<gentype> &kresg = args.xkresg;
    //Vector<gentype> &kgvarres = args.xkgvarres;
    //Vector<double> &loctmpcnt = args.xloctmpcnt;
    //Matrix<double> &loctmpcfm = args.xloctmpcfm;
    //double &loctmpres = args.xloctmpres;
    //int &locNnonz = args.xlocNnonz;
    const Vector<int> &locblockidvect = args.xlocblockidvect;
    //int Nclasses = args.xNclasses;
    //const Vector<int> &locisenabled = args.xlocisenabled;
    //const Vector<gentype> &locy = args.xlocy;
    int startpoint = args.xstartpoint;
    //int calcgvarres = args.xcalcgvarres;
    //const svm_pthread_id &mainid = args.mainid;

//errstream() << "phantomxyz 0: locML = " << locML << " and locML.altxsrc = " << locML->N() << "\n";
        disableVector(locblockidvect,locML);

        if ( startpoint )
        {
            resetGlobal(locML);
        }

    return nullptr;
}



void *doafold(void *args_ptr)
{
    foldargs &args = *((foldargs *) args_ptr);

    ML_Base *locML = args.xlocML;
    bool trainglobcond = args.xtrainglobcond;
    //Vector<gentype> &kresh = args.xkresh;
    //Vector<gentype> &kresg = args.xkresg;
    //Vector<gentype> &kgvarres = args.xkgvarres;
    //Vector<double> &loctmpcnt = args.xloctmpcnt;
    //Matrix<double> &loctmpcfm = args.xloctmpcfm;
    //double &loctmpres = args.xloctmpres;
    //int &locNnonz = args.xlocNnonz;
    //const Vector<int> &locblockidvect = args.xlocblockidvect;
    //int Nclasses = args.xNclasses;
    //const Vector<int> &locisenabled = args.xlocisenabled;
    //const Vector<gentype> &locy = args.xlocy;
    //int startpoint = args.xstartpoint;
    //int calcgvarres = args.xcalcgvarres;
    //const svm_pthread_id &mainid = args.mainid;

        trainGlobal(args.foldres,locML,trainglobcond);

    return nullptr;
}


void *doatest(void *args_ptr)
{
    foldargs &args = *((foldargs *) args_ptr);

    ML_Base *locML = args.xlocML;
    //bool trainglobcond = args.xtrainglobcond;
    Vector<gentype> &kresh = args.xkresh;
    Vector<gentype> &kresg = args.xkresg;
    Vector<gentype> &kgvarres = args.xkgvarres;
    Vector<double> &loctmpcnt = args.xloctmpcnt;
    Matrix<double> &loctmpcfm = args.xloctmpcfm;
    double &loctmpres = args.xloctmpres;
    int &locNnonz = args.xlocNnonz;
    const Vector<int> &locblockidvect = args.xlocblockidvect;
    int Nclasses = args.xNclasses;
    const Vector<int> &locisenabled = args.xlocisenabled;
    const Vector<gentype> &locy = args.xlocy;
    //int startpoint = args.xstartpoint;
    int calcgvarres = args.xcalcgvarres;
    //const svm_pthread_id &mainid = args.mainid;

    int j,l,cla,clb;
    int Nblock = locblockidvect.size();

    loctmpcnt.zero();
    loctmpcfm.zero();
    loctmpres = 0;

    for ( j = 0 ; j < Nblock ; ++j )
    {
        l = locblockidvect(j);

        locML->gh(kresh("&",l),kresg("&",l),l);

        if ( calcgvarres )
        {
            gentype dummy;

            locML->var(kgvarres("&",l),dummy,l);
        }

        // NB: isenabled actually returns d for MLs.  For ML_Scalar,
        //     d sets whether a point is an upper bound, a lower bound
        //     or a standard target type.  This information is needed
        //     to calculate distance.  The isenabled(i) actually returns
        //     d for MLs (d = 0 means disabled).  Hence the following
        //     line passes isenabled(l) to calcDist.

        if ( (*locML).isClassifier() && ishzero(kresh(l)) )
        {
            loctmpres += 1;

            cla = locy(l).isNomConst ? 0 : (*locML).getInternalClass(locy(l));
            clb = Nclasses;
        }

        else
        {
            loctmpres += locML->calcDist(kresh(l),locy(l),l,locisenabled(l));

            cla = locy(l).isNomConst ? 0 : (*locML).getInternalClass(locy(l));
            clb = (*locML).getInternalClass(kresh(l));
        }

        ++(loctmpcnt("&",cla));
        ++(loctmpcfm("&",cla,clb));

        ++locNnonz;
    }

    return nullptr;
}









double calcLOORecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int isLOO, int calcgvarres, int suppressfb, int useThreads)
{
    (void) useThreads; // FIXME: implement later: basically guess number of cores, divide the problem into that many subsets, then set each going as per calcCross

    if ( !baseML.isSolGlob() )
    {
        if ( startpoint == 0 )
        {
            startpoint = 1; // start from reset state no matter what
        }

        if ( startpoint == 2 )
        {
            startpoint = 3; // start from reset state no matter what
        }
    }

    startpoint &= 3;

    int oneoff = startpoint & 2;
    startpoint &= 1;

    NiceAssert( !oneoff );
    (void) oneoff;

    //startpoint = startpoint || isONN(baseML);

    int i,cla,clb,isReTrain;
    double res = 0.0;
    int N = baseML.N();
    int Nnz = baseML.N()-baseML.NNC(0);
    int Nclasses = baseML.numInternalClasses();
    int astategood;

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    resh.resize(N);
    resg.resize(N);

    resh.zero();
    resg.zero();

    if ( calcgvarres )
    {
        gvarres.resize(N);
        gvarres.zero();
    }

    if ( N )
    {
        ML_Base *locML = (ML_Base *) &baseML;
        ML_Base *srcML = (ML_Base *) &baseML;

        if ( isLOO )
	{
            locML = makeNewML(baseML.type(),baseML.subtype());
            NiceAssert( locML );
            semicopyML(locML,&baseML);
	}

        ML_Base *temp = srcML; srcML = locML; locML = temp;

        int locisenable;
        gentype locy;

        for ( i = 0 ; i < N ; ++i )
	{
	    isReTrain = 0;

            locisenable = isVectorEnabled(i,locML);
            locy        = (*locML).y()(i);

            if ( locisenable )
	    {
                astategood = isVectorActive(i,locML);

if ( !suppressfb )
{
nullPrint(errstream(),"~~~~~",5);
nullPrint(errstream(),i,-5);
}
                if ( ( astategood || !isTrainedML(locML) ) && isLOO )
		{
                    disableVector(i,locML);

                    if ( startpoint )
                    {
                        resetGlobal(locML);
                    }

                    int loclocres = 0;
                    trainGlobal(loclocres,locML,( i == N-1 ));
                    if ( loclocres ) { return valvnan(); }

                    isReTrain = 1;
		}
	    }

            else if ( isLOO && ( i == N-1 ) )
            {
                int loclocres = 0;
                trainGlobal(loclocres,locML,( i == N-1 ));
                if ( loclocres ) { return valvnan(); }
            }
//if ( !suppressfb )
//{
//nullPrint(errstream(),".");
//}

            (*locML).gh(resh("&",i),resg("&",i),i);
//errstream() << "phantomx: resh,resg = " << resh(i) << "," << resg(i) << "\n";

            if ( calcgvarres )
            {
                gentype dummy;

                (*locML).var(gvarres("&",i),dummy,i);
            }

//errstream() << "phantomx1: locisenable = " << locisenable << "\n";
            if ( locisenable )
            {
                if ( (*locML).isClassifier() && ishzero(resh(i)) )
                {
//errstream() << "phantomx2: classifier\n";
                    res += 1;

                    cla = locy.isNomConst ? 0 : (*locML).getInternalClass(locy);
                    clb = Nclasses;
                }

                else
                {
//errstream() << "phantomx3: regression: locy = " << locy << "\n";
//errstream() << "phantomx3: regression: dist = " << (*locML).calcDist(resh(i),locy,i,locisenable) << "\n";
                    res += (*locML).calcDist(resh(i),locy,i,locisenable);

                    cla = locy.isNomConst ? 0 : (*locML).getInternalClass(locy);
                    clb = (*locML).getInternalClass((resh)(i));
                }

                ++(cnt("&",cla));
                ++(cfm("&",cla,clb));

                if ( isReTrain && isLOO )
                {
                    semicopyML(locML,srcML);
                }
            }
        }

        if ( isLOO )
        {
            { ML_Base *ttemp = srcML; srcML = locML; locML = ttemp; }

            MEMDEL(locML); locML = nullptr;
        }
    }

    res = Nnz ? res/Nnz : res;

    if ( baseML.isRegression() )
    {
        res = sqrt(res);
    }

if ( !suppressfb )
{
errstream() << "\n";
}
    return res;
}

double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int calcgvarres, int startpoint, int suppressfb, int useThreads)
{
    (void) startpoint;
    (void) suppressfb;
    (void) useThreads;

    NiceAssert( xtest.size() == ytest.size() );

    int i,cla,clb;
    double res = 0.0;
    int N = xtest.size();
    int Nnz = N;
    int Nclasses = baseML.numInternalClasses();

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    resh.resize(N);
    resg.resize(N);

    resh.zero();
    resg.zero();

    if ( calcgvarres )
    {
        gvarres.resize(N);
        gvarres.zero();
    }

    if ( N )
    {
        for ( i = 0 ; i < N ; ++i )
	{
            baseML.gh(resh("&",i),resg("&",i),xtest(i));

            if ( calcgvarres )
            {
                gentype dummy;

                baseML.var(gvarres("&",i),dummy,xtest(i));
            }

            // NB: for svm_planar if f4(3) present then this is expert f4(3).  Otherwise vector output

            if ( baseML.isClassifier() && ishzero(resh(i)) )
            {
                res += 1;

                cla = ytest(i).isNomConst ? 0 : baseML.getInternalClass(ytest(i));
                clb = Nclasses;
            }

            else
            {
                res += baseML.calcDist(resh(i),ytest(i),xtest(i).isf4indpresent(3) ? (int) xtest(i).f4(3) : -1);

                cla = ytest(i).isNomConst ? 0 : baseML.getInternalClass(ytest(i));
                clb = baseML.getInternalClass(resh (i));
            }

            ++(cnt("&",cla));
            ++(cfm("&",cla,clb));
        }
    }

    res = Nnz ? res/Nnz : res;

    if ( baseML.isRegression() )
    {
        res = sqrt(res);
    }

    return res;
}

double calcSparSens(const ML_Base &baseML, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, Vector<Vector<gentype> > &resh, Vector<Vector<gentype> > &resg, Vector<Vector<gentype> > &gvarres, int minbad, int maxbad, double noisemean, double noisevar, int startpoint, int calcgvarres, int suppressfb, int useThreads)
{
    (void) useThreads; // kept for future reference

    NiceAssert( maxbad >= minbad );
    NiceAssert( minbad >= 0 );
    NiceAssert( noisevar >= 0 );

    int numreps = maxbad-minbad+1;

    if ( !baseML.isSolGlob() )
    {
        if ( startpoint == 0 )
        {
            startpoint = 1; // start from reset state no matter what
        }

        if ( startpoint == 2 )
        {
            startpoint = 3; // start from reset state no matter what
        }
    }

    startpoint &= 3;

    //startpoint = startpoint || isONN(baseML);

    int i,j,k;
    double res = 0;
    int N = baseML.N();
    int Nnz = baseML.N()-baseML.NNC(0);
    int Nclasses = baseML.numInternalClasses();
    int xdim = baseML.xspaceDim();

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    repres.resize(numreps);
    repres.zero();

    resg.resize(numreps);
    resh.resize(numreps);

    if ( calcgvarres )
    {
        gvarres.resize(numreps);
    }

    for ( i = 0 ; i < numreps ; ++i )
    {
        resg("&",i).resize(N);
        resg("&",i).zero();

        resh("&",i).resize(N);
        resh("&",i).zero();

        if ( calcgvarres )
        {
            gvarres("&",i).resize(N);
            gvarres("&",i).zero();
        }
    }

    if ( Nnz )
    {
	Vector<double> loccnt(cnt);
	Matrix<double> loccfm(cfm);
        double locres = 0.0;

        ML_Base *locML = (ML_Base *) &baseML;
        ML_Base *srcML = (ML_Base *) &baseML;

        locML = makeNewML(baseML.type(),baseML.subtype());
        NiceAssert( locML );
        copyML(locML,&baseML);

        int locisenabled;
        gentype locy;

        for ( k = 0 ; k < minbad ; ++k )
        {
            for ( j = 0 ; j < N ; ++j )
            {
                SparseVector<gentype> xj = ((*locML).x(j));

                double temp;

                randnfill(temp);

                xj("[]",xdim+1) = noisemean+(temp*noisevar);

                (*locML).setx(j,xj);
            }

            ++xdim;
        }

        for ( k = 0 ; k < numreps ; ++k )
        {
            loccnt.zero();
            loccfm.zero();
            locres = 0.0;

            int cla,clb;
            int Nnonz = 0;

            if ( startpoint )
            {
                resetGlobal(locML);
            }

if ( !suppressfb )
{
nullPrint(errstream(),"~~~~~",5);
nullPrint(errstream(),k,-5); 
}
            int loclocres = 0;
            trainGlobal(loclocres,locML,( k == numreps-1 ));
            if ( loclocres ) { return valvnan(); }

            for ( j = 0 ; j < N ; ++j )
            {
                locisenabled = isVectorEnabled(j,locML);
                locy         = (*locML).y()(j);

                if ( locisenabled )
                {
                    locML->gh(resh("&",k)("&",j),resg("&",k)("&",j),j);

                    if ( calcgvarres )
                    {
                        gentype dummy;

                        locML->var(gvarres("&",k)("&",j),dummy,j);
                    }

                    // NB: isenabled actually returns d for MLs.  For ML_Scalar,
                    //     d sets whether a point is an upper bound, a lower bound
                    //     or a standard target type.  This information is needed
                    //     to calculate distance.  The isenabled(i) actually returns
                    //     d for MLs (d = 0 means disabled).  Hence the following
                    //     line passes isenabled(l) to calcDist.

                    if ( (*locML).isClassifier() && ishzero(resh(k)(j)) )
                    {
                        locres += 1;

                        cla = locy.isNomConst ? 0 : (*locML).getInternalClass(locy);
                        clb = Nclasses;
                    }

                    else
                    {
                        locres += locML->calcDist(resh(k)(j),locy,j,locisenabled);

                        cla = locy.isNomConst ? 0 : (*locML).getInternalClass(locy);
                        clb = (*locML).getInternalClass((resh(k))(j));
                    }

                    ++(loccnt("&",cla));
                    ++(loccfm("&",cla,clb));

                    ++Nnonz;
                }
            }

            semicopyML(locML,srcML);

            locres = Nnonz ? locres/Nnonz : res;

            if ( baseML.isRegression() )
            {
                locres = sqrt(locres);
            }

            cnt += loccnt;
            cfm += loccfm;
            res += locres;

            repres("&",k) = locres;

            for ( j = 0 ; j < N ; ++j )
            {
                SparseVector<gentype> xj = ((*locML).x(j));

                double temp;

                randnfill(temp);

                xj("[]",xdim+1) = noisemean+(temp*noisevar);

                (*locML).setx(j,xj);
            }

            ++xdim;
	}

	cnt *= 1/((double) numreps);
	cfm *= 1/((double) numreps);
        res *= 1/((double) numreps);

        MEMDEL(locML); locML = nullptr;
    }

//    if ( baseML.isRegression() )
//    {
//        res = sqrt(res);
//    }

if ( !suppressfb )
{
errstream() << "\n";
}
    return res;
}


double assessResult(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, const Vector<gentype> &ytestresh, const Vector<gentype> &ytest, const Vector<int> &outkernind)
{
    NiceAssert( ytest.size() == ytestresh.size() );

    double res = 0;
    int i;
    int N = ytest.size();
    int Nclasses = baseML.numInternalClasses();
    int cla,clb;

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt = 0;
    cfm = 0;

    if ( N )
    {
        for ( i = 0 ; i < N ; ++i )
        {
            if ( baseML.isClassifier() && ishzero(ytestresh(i)) )
            {
                res += 1;

                cla = ytest(i).isNomConst ? 0 : baseML.getInternalClass(ytest(i));
                clb = Nclasses;
            }

            else
            {
                res += baseML.calcDist(ytestresh(i),ytest(i),outkernind(i));

                cla = ytest(i).isNomConst ? 0 : baseML.getInternalClass(ytest(i));
                clb = baseML.getInternalClass(ytestresh(i));
            }

            if ( ( cla >= 0 ) && ( clb >= 0 ) && ( cla < Nclasses ) && ( clb < Nclasses+1 ) )
            {
                ++(cnt("&",cla));
                ++(cfm("&",cla,clb));
            }
        }
    }

    res = N ? res/N : res;

    if ( baseML.isRegression() )
    {
        res = sqrt(res);
    }

    return res;
}








static void disableVector(int i, ML_Base *activeML)
{
    activeML->disable(i);

    return;
}

static void disableVector(const Vector<int> &i, ML_Base *activeML)
{
    activeML->disable(i);

    return;
}

static void resetGlobal(ML_Base *activeML)
{
    activeML->reset();

    return;
}

static int trainGlobal(int &res, ML_Base *activeML, int islastopt)
{
    res = 0;

    if ( activeML->type() == 0 )
    {
        (dynamic_cast<SVM_Scalar &>(activeML->getML())).inEmm4Solve = islastopt ? 0 : 2;
    }

    else if ( activeML->type() == 1 )
    {
        (dynamic_cast<SVM_Binary &>(activeML->getML())).inEmm4Solve = islastopt ? 0 : 2;
    }

    else if ( activeML->type() == 2 )
    {
        (dynamic_cast<SVM_Single &>(activeML->getML())).inEmm4Solve = islastopt ? 0 : 2;
    }

    return activeML->train(res);
}

static void semicopyML(ML_Base *activeML, const ML_Base *srcML)
{
    activeML->semicopy(*srcML);

    return;
}

static void copyML(ML_Base *activeML, const ML_Base *srcML)
{
    *activeML = *srcML;

    return;
}

static int isVectorActive(int i, ML_Base *activeML)
{
    return (*activeML).alphaState()(i);
}

static int isVectorEnabled(int i, ML_Base *activeML)
{
    if ( (*activeML).y(i).isNomConst )
    {
        return 0;
    }

    return (*activeML).isenabled(i);
}

static int isTrainedML(ML_Base *activeML)
{
    return (*activeML).isTrained();
}












void measureAccuracy(Vector<double> &res, const Vector<gentype> &resg, const Vector<gentype> &resh, const Vector<gentype> &ytarg, const Vector<int> &dstat, const ML_Base &ml)
{
    NiceAssert( resg.size() == ytarg.size() );
    NiceAssert( resg.size() == dstat.size() );
    
    res.resize(7);

    double &accuracy  = res("&",0);
    double &precision = res("&",1);
    double &recall    = res("&",2);
    double &f1score   = res("&",3);
    double &auc       = res("&",4);
    double &sparsity  = res("&",5);
    double &error     = res("&",6);

    accuracy  = 1;
    precision = 1;
    recall    = 1;
    f1score   = 1;
    auc       = 1;
    sparsity  = ml.sparlvl();
    error     = 0;

    int numClasses = ml.numClasses();

    NiceAssert( numClasses = 2 );

    int clvala = ( numClasses >= 1 ) ? (ml.ClassLabels())(0) : -1;
    int clvalb = ( numClasses >= 2 ) ? (ml.ClassLabels())(1) : clvala+2;

    NiceAssert( clvala != clvalb );

    if ( clvalb < clvala )
    {
        int clvalx;

        clvalx = clvala;
        clvala = clvalb;
        clvalb = clvalx;
    }

    int Nnz = 0;
    int i,j;
    int N = resg.size();

    for ( i = 0 ; i < N ; ++i )
    {
        if ( dstat(i) )
        {
            ++Nnz;
        }
    }

    Vector<double> g(N);
    Vector<int> d(N);
    Vector<int> dpred(N);
    Vector<int> dstatp(dstat);

    int isAUCcalcable = ( numClasses == 2 ) ? 1 : 0;

    if ( N > 1 )
    {
        // Grab data in useable form

        for ( i = N-1 ; i >= 0 ; --i )
        {
            if ( !dstatp(i) )
            {
                g.remove(i);
                d.remove(i);
                dpred.remove(i);
                dstatp.remove(i);
                --N;
            }

            else
            {
                g("&",i) =  resg(i).isCastableToRealWithoutLoss() ? (double) resg(i) : 0;
                d("&",i) = (int) ytarg(i);

                dpred("&",i) = (int) resh(i);

                isAUCcalcable = resg(i).isCastableToRealWithoutLoss() ? isAUCcalcable : 0;
            }
        }

        NiceAssert( N == Nnz );
    }




    if ( N > 1 )
    {
        // Accuracy Calculation

        accuracy = 0;

        for ( i = 0 ; i < N ; ++i )
        {
            if ( dpred(i) == d(i) )
            {
                ++accuracy;
            }
        }

        accuracy /= N;




        // Precision / recall / f1 score Calculation

        if ( numClasses == 2 )
        {
            // Precision/recall Calculation

            int trpos = 0; // true positive
            int fapos = 0; // false positive
            int faneg = 0; // false negative

            for ( i = 0 ; i < N ; ++i )
            {
                if ( ( dpred(i) == clvalb ) && ( d(i) == clvalb ) )
                {
                    ++trpos;
                }

                else if ( ( dpred(i) == clvala ) && ( d(i) == clvalb ) )
                {
                    ++fapos;
                }

                else if ( ( dpred(i) == clvalb ) && ( d(i) == clvala ) )
                {
                    ++faneg;
                }
            }

            precision = trpos+fapos ? ((double) trpos)/((double) trpos+fapos) : 1;
            recall    = trpos+faneg ? ((double) trpos)/((double) trpos+faneg) : 1;

            // f1Score Calculation

            f1score = 2*precision*recall/(precision+recall+1e-10);
        }

        else
        {
            precision = -1;
            recall    = -1;
            f1score   = -1;
        }





        // AUC Calculation

        if ( isAUCcalcable )
        {
            // Sort data from smallest g to largest

            for ( i = 0 ; i < N-1 ; ++i )
            {
                for ( j = 1 ; j < N ; ++j )
                {
                    if ( g(j) < g(i) )
                    {
                        qswap(g("&",j),g("&",i));
                        qswap(dpred("&",j),dpred("&",i));
                        qswap(d("&",j),d("&",i));
                        qswap(dstatp("&",j),dstatp("&",i));
                    }
                }
            }

            // calculate tp/fp

            Vector<int> tp(N);
            Vector<int> fp(N);

            int NP = 0;
            int NN = 0;

            tp("&",0) = 0;
            fp("&",0) = 0;

            for ( i = 1 ; i < N ; ++i )
            {
                tp("&",i) = tp("&",i-1);
                fp("&",i) = fp("&",i-1);

                if ( d(i) == clvalb )
                {
                    ++(tp("&",i));
                    ++NP;
                }

                else
                {
                    ++(fp("&",i));
                    ++NN;
                }
            }

            // Calculate AUC

            auc = 0;

            for ( i = 1 ; i < N ; ++i )
            {
                if ( fp(i) > fp(i-1) )
                {
                    auc += ( (2*(fp(i)-fp(i-1))*NN*NP) != 0 ) ? ((double) (tp(i)+tp(i-1)))/((double) (2*(fp(i)-fp(i-1))*NN*NP)) : 0;
                }
            }
        }

        else
        {
            auc = -1;
        }
    }

    error = 1-accuracy;

    return;
}




void bootstrapML(ML_Base &ml)
{
    if ( ml.N() )
    {
        int j;

        // Start with all indices

        retVector<int> tmpva;

        Vector<int> dnzind(cntintvec(ml.N(),tmpva));

        // Remove all with d == 0

        for ( j = dnzind.size()-1 ; j >= 0 ; --j )
        {
            if ( !((ml.d())(dnzind(j))) )
            {
                dnzind.remove(j);
            }
        }

        if ( dnzind.size() )
        {
            // Construct tally vector

            retVector<int> ttmpva;

            Vector<int> dnztally(zerointvec(dnzind.size(),ttmpva));

            // Randomly "select" elements from dnzind by incrementing the relevant element in tally

            for ( j = 0 ; j < dnzind.size() ; ++j )
            {
                //++(dnztally("&",svm_rand()%(dnzind.size())));
                ++(dnztally("&",rand()%(dnzind.size())));
            }

            // Remove elements *selected* (that way we have the indices of those for which we wish to set d = 0 left)

            for ( j = dnzind.size()-1 ; j >= 0 ; --j )
            {
                if ( dnztally(j) )
                {
                    dnzind.remove(j);
                    dnztally.remove(j);
                }
            }

            // Set d = 0 for unselected elements (remaining dnztally = 0, so just use this as "d")

            ml.setd(dnzind,dnztally);
        }
    }

    return;
}

