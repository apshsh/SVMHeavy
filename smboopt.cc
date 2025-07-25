
//
// Sequential model-based optimisation base class
//
// Date: 12/02/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "smboopt.hpp"
#include "plotml.hpp"
#include "ml_mutable.hpp"


SMBOOptions::SMBOOptions() : GlobalOptions()
    {
        locdim = 0;

        PIscale = 0;

        altmuapprox.ssetMLTypeClean("gpr");
        altcgtapprox.ssetMLTypeClean("gpr");
        altaugxapprox.ssetMLTypeClean("gpr");
        altmuapprox_rff.ssetMLTypeClean("gpR");

        optname = "SMBO Optimisation";

        locires.useTightAllocation();
        locxres.useTightAllocation();
        locxresunconv.useTightAllocation();
        locyres.useTightAllocation();

        //muapproxInd = -1;

        sigmuseparate = 0;
        ismoo         = 0;
        moodim        = 1;
        numcgt        = 0;
        modeltype     = 0;
        modelrff      = 0;
        oracleMode    = 0;

        TSmode      = 3;
        TSNsamp     = 0; //DEFAULT_SAMPLES_SAMPLE;
        TSsampType  = 0;
        TSxsampType = 0;
        sigma_cut   = DEFAULT_SIGMA_CUT;

        modelname      = "smbomodel";
#ifndef USE_MEX
        modeloutformat = 2; // pdf by default
#endif
#ifdef USE_MEX
        modeloutformat = 3; // matlab plot if system is matlab
#endif
        plotfreq       = 0; // don't plot by default
        modelbaseline  = "null";

        tranmeth = 0;
        alpha0   = 0.1;
        beta0    = 1;

        alpha = alpha0;
        beta  = beta0;

        kernapprox = nullptr;
        kxfnum     = 801;
        kxfnorm    = 1;

        tunemu      = 1; //2;
        tunesigma   = 1; //2;
        tunecgt     = 1; //2;
        tunesrcmod  = 1; //2;
        tunediffmod = 1; //2;
        tuneaugxmod = 1; //2;

        usemodelaugx = 0;
        modelnaive   = 0;
        makenoise    = 0;
        ennornaive   = 0;

        srcmodel  = nullptr;
        diffmodel = nullptr;

        srcmodelInd  = -1;
        diffmodelInd = -1;

        //augxapprox = nullptr;

        //extmuapprox = nullptr;
        //extaugxapprox = nullptr;

        //muapprox        = nullptr;
        //sigmaapprox     = nullptr;
        //muapprox_sample = nullptr;
        //augxapprox      = nullptr;

        modelErrOptim   = nullptr;
        ismodelErrLocal = 1;

        sigmaapprox    = nullptr;
        sigmaapproxRaw = nullptr;

        Nbasemu    = 0;
        firsttrain = 0;

        xshortcutenabled = 0;
    }

SMBOOptions::SMBOOptions(const SMBOOptions &src) : GlobalOptions(src)
    {
        modelErrOptim   = nullptr;
        ismodelErrLocal = 1;

        locires.useTightAllocation();
        locxres.useTightAllocation();
        locxresunconv.useTightAllocation();
        locyres.useTightAllocation();

        *this = src;
    }

SMBOOptions &SMBOOptions::operator=(const SMBOOptions &src)
    {
        GlobalOptions::operator=(src);

        killModelErrOptim();

        locdim = src.locdim;

        PIscale = src.PIscale;

        augxapprox = src.augxapprox;

        muapproxInd   = src.muapproxInd;

        extmuapprox   = src.extmuapprox;
        extaugxapprox = src.extaugxapprox;
        extcgtapprox  = src.extcgtapprox;

        sigmuseparate = src.sigmuseparate;
        ismoo         = src.ismoo;
        moodim        = src.moodim;
        numcgt        = src.numcgt;
        modeltype     = src.modeltype;
        modelrff      = src.modelrff;
        oracleMode    = src.oracleMode;

        TSmode      = src.TSmode;
        TSNsamp     = src.TSNsamp;
        TSsampType  = src.TSsampType;
        TSxsampType = src.TSxsampType;
        sigma_cut   = src.sigma_cut;

        modelname      = src.modelname;
        modeloutformat = src.modeloutformat;
        plotfreq       = src.plotfreq;
        modelbaseline  = src.modelbaseline;

        tranmeth = src.tranmeth;
        alpha0   = src.alpha0;
        beta0    = src.beta0;

        kernapprox = src.kernapprox;
        kxfnum     = src.kxfnum;
        kxfnorm    = src.kxfnorm;

        tunemu      = src.tunemu;
        tunesigma   = src.tunesigma;
        tunecgt     = src.tunecgt;
        tunesrcmod  = src.tunesrcmod;
        tunediffmod = src.tunediffmod;
        tuneaugxmod = src.tuneaugxmod;

        usemodelaugx = src.usemodelaugx;
        modelnaive   = src.modelnaive;
        makenoise    = src.makenoise;
        ennornaive   = src.ennornaive;

        xtemplate = src.xtemplate;

        // ======================================

        (xmodprod)         = (src.xmodprod);
        xshortcutenabled = src.xshortcutenabled;
        (xsp)              = (src.xsp);
        (xspp)             = (src.xspp);
        indpremu         = src.indpremu;
        presigweightmu   = src.presigweightmu;

        Nbasemu = src.Nbasemu;
        resdiff = src.resdiff;

        alpha = src.alpha;
        beta  = src.beta;

        srcmodel  = src.srcmodel;
        diffmodel = src.diffmodel;

        srcmodelInd  = src.srcmodelInd;
        diffmodelInd = src.diffmodelInd;

        diffval  = src.diffval;
        predval  = src.predval;
        storevar = src.storevar;

        firsttrain = src.firsttrain;

        (xx) = (src.xx);

        muapprox    = src.muapprox;
        sigmaapprox = src.sigmaapprox;
        augxapprox  = src.augxapprox;
        cgtapprox   = src.cgtapprox;

        muapprox_sample.resize(0);

        altmuapprox         = src.altmuapprox;
        altmuapprox_rff     = src.altmuapprox_rff;
        altaugxapprox       = src.altaugxapprox;
        altcgtapprox        = src.altcgtapprox;

        locires       = src.locires;
        locxres       = src.locxres;
        locxresunconv = src.locxresunconv;
        locyres       = src.locyres;

        if ( src.modelErrOptim )
        {
            modelErrOptim   = (*((src.modelErrOptim))).makeDup();
            ismodelErrLocal = 1;
        }

        return *this;
    }

GlobalOptions *SMBOOptions::makeDup(void) const
    {
        SMBOOptions *newver;

        MEMNEW(newver,SMBOOptions(*this));

        return newver;
    }


void SMBOOptions::consultTheOracle(ML_Mutable &randDir, int dim, const SparseVector<gentype> &locxres, int isFirstAxis)
    {
        if ( ( ( oracleMode == 0 ) || ( ( oracleMode == 3 ) && isFirstAxis ) ) && ( isProjection == 2 ) && useScalarFn && isGPRScalar(randDir) && ( ( modeltype == 0 ) || ( modeltype == 1 ) ) && !sigmuseparate && !ismoo )
        {
            // Construct vector to calculate gradient at locxres

            gentype Ns(xNsamp);

            SparseVector<gentype> czm(locxres);
            czm.f4("&",6) = 1; // This tells models to evaluate gradients

            // Find mean and variance of gradient at czm

            gentype mvec;
            gentype vmat;

            getmuapprox(0).var(vmat,mvec,czm);

            // Sample gradient GP to get direction

            gentype yvec;

            yvec = grand(mvec,vmat);
            yvec *= -1.0; // gradient *descent*

            // Pre-sample GPR (we will overwrite the result)

//FIXMEFIXME fnDim
            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            randDir.getGPR().setSampleMode(1,xmin,xmax,xNsamp,xSampSplit,xSampType,xxSampType,1.0);

            // Overwrite y to finalise sample

            int dummy = 0;

            randDir.getGPR().sety((const Vector<gentype> &) yvec);
            randDir.getGPR().train(dummy);
        }

        else if ( ( ( oracleMode == 1 ) || ( ( oracleMode == 4 ) && isFirstAxis ) ) && ( isProjection == 2 ) && useScalarFn && isGPRScalar(randDir) && ( ( modeltype == 0 ) || ( modeltype == 1 ) ) && !sigmuseparate && !ismoo )
        {
            // Construct vector to calculate gradient at locxres

            gentype Ns(xNsamp);

            SparseVector<gentype> czm(locxres);
            czm.f4("&",6) = 1; // This tells models to evaluate gradients

            // Find mean and variance of gradient at czm

            gentype mvec;

            getmuapprox(0).gg(mvec,czm);

            // We don't sample here, just take the direction

            gentype yvec;

            yvec = mvec;
            yvec *= -1.0; // gradient *descent*

            // Pre-sample GPR (we will overwrite the result)

//FIXMEFIXME fnDim
            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            randDir.getGPR().setSampleMode(1,xmin,xmax,xNsamp,xSampSplit,xSampType,xxSampType,1.0);

            // Overwrite y to finalise sample

            int dummy = 0;

            randDir.getGPR().sety((const Vector<gentype> &) yvec);
            randDir.getGPR().train(dummy);
        }

        else
        {
//RKHSFIXME
            GlobalOptions::consultTheOracle(randDir,dim,locxres,isFirstAxis);
        }
    }

int SMBOOptions::realOptim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &rawxres,
                      gentype &fres,
                      int &ires,
                      int &mres,
                      Vector<int> &muInd,
                      Vector<int> &augxInd,
                      Vector<int> &cgtInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allrawxres,
                      Vector<gentype> &allfres,
                      Vector<Vector<gentype> > &allcres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &supres,
                      Vector<double> &sscore,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
    {
        // Create and register models

        //int muapproxInd    = -1;

        locdim = dim;

        // These need to be passed back

        Vector<int> dummyMLnumbers(9);
        Vector<int> &MLnumbers = MLdefined ? (*((Vector<int> *) ((void **) fnarg)[15])) : dummyMLnumbers;

        // Construct models and register them

        if ( !ismoo )
        {
            muapproxRaw.resize(1);
            muapprox.resize(1);
            muapproxInd.resize(1);
            muInd.resize(1);

            muapproxRaw = nullptr;
            muapprox    = nullptr;
            muapproxInd = -1;
            muInd       = -1;

            if ( extmuapprox.size() && extmuapprox(0) )
            {
                MEMNEW(muapproxRaw("&",0),ML_Mutable);
                (*(muapproxRaw("&",0))).setMLTypeClean((*(extmuapprox(0))).type());

                (*(muapproxRaw("&",0))).getML() = *(extmuapprox(0));
                muapproxInd("&",0) = regML(muapproxRaw("&",0),fnarg,5);

                muapprox("&",0) = &((*(muapproxRaw("&",0))).getML());
            }

            else if ( ( isProjection != 2 ) && !modelrff )
            {
//RKHSFIXME
                MEMNEW(muapproxRaw("&",0),ML_Mutable);
                (*(muapproxRaw("&",0))).setMLTypeClean(altmuapprox.type());

                (*(muapproxRaw("&",0))).getML() = altmuapprox;
                muapproxInd("&",0) = regML(muapproxRaw("&",0),fnarg,5);

                muapprox("&",0) = &((*(muapproxRaw("&",0))).getML());
//RKHSFIXME
                //(*muapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
            }

            else if ( isProjection != 2 )
            {
//RKHSFIXME
                MEMNEW(muapproxRaw("&",0),ML_Mutable);
                (*(muapproxRaw("&",0))).setMLTypeClean(altmuapprox_rff.type());

                (*(muapproxRaw("&",0))).getML() = altmuapprox_rff;
                muapproxInd("&",0) = regML(muapproxRaw("&",0),fnarg,5);

                muapprox("&",0) = &((*(muapproxRaw("&",0))).getML());

                if ( ( modelrff != (*(muapprox(0))).NRff() ) && (*(muapprox(0))).xspaceDim() )
                {
                    (*(muapprox("&",0))).setNRff(modelrff); // needs to be delayed until there are training vectors or RFF will have dim 0, which is wrong)
                }
//RKHSFIXME
                //(*muapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
            }

            else
            {
                MEMNEW(muapproxRaw("&",0),ML_Mutable);
                (*(muapproxRaw("&",0))).setMLTypeClean(altmuapprox.type());

                (*(muapproxRaw("&",0))).getML() = altmuapprox;
                muapproxInd("&",0) = regML(muapproxRaw("&",0),fnarg,5);

                muapprox("&",0) = &((*(muapproxRaw("&",0))).getML());
            }

            muInd = muapproxInd;

            MLnumbers("&",0) = muapproxInd("&",0);
        }

        else
        {
            muapproxRaw.resize(moodim);
            muapprox.resize(moodim);
            muapproxInd.resize(moodim);
            muInd.resize(moodim);

            muapproxRaw = nullptr;
            muapprox    = nullptr;
            muapproxInd = -1;
            muInd       = -1;

            int ii;

            for ( ii = 0 ; ii < moodim ; ii++ )
            {
                if ( ( extmuapprox.size() > ii ) && extmuapprox(ii) )
                {
                    MEMNEW(muapproxRaw("&",ii),ML_Mutable);
                    (*(muapproxRaw("&",ii))).setMLTypeClean((*(extmuapprox(ii))).type());

                    (*(muapproxRaw("&",ii))).getML() = *(extmuapprox(ii));
                    muapproxInd("&",ii) = regML(muapproxRaw("&",ii),fnarg,6);

                    muapprox("&",ii) = &((*(muapproxRaw("&",ii))).getML());
                }

                else
                {
                    MEMNEW(muapproxRaw("&",ii),ML_Mutable);
                    (*(muapproxRaw("&",ii))).setMLTypeClean(altmuapprox.type());

                    (*(muapproxRaw("&",ii))).getML() = altmuapprox;
                    muapproxInd("&",ii) = regML(muapproxRaw("&",ii),fnarg,6);

                    muapprox("&",ii) = &((*(muapproxRaw("&",ii))).getML());
//RKHSFIXME
                    //(*muapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
                }
            }

            muInd = muapproxInd;

            MLnumbers("&",0) = muapproxInd(0); //FIXME
        }

        if ( numcgt )
        {
            cgtapproxRaw.resize(numcgt);
            cgtapprox.resize(numcgt);
            cgtInd.resize(numcgt);

            cgtapproxRaw = nullptr;
            cgtapprox    = nullptr;
            cgtInd       = -1;

            for ( int ii = 0 ; ii < numcgt ; ii++ )
            {
                if ( ( extcgtapprox.size() > ii ) && extcgtapprox(ii) )
                {
                    MEMNEW(cgtapproxRaw("&",ii),ML_Mutable);
                    (*(cgtapproxRaw("&",ii))).setMLTypeClean((*(extcgtapprox(ii))).type());

                    (*(cgtapproxRaw("&",ii))).getML() = *(extcgtapprox(ii));
                    cgtInd("&",ii) = regML(cgtapproxRaw("&",ii),fnarg,14);

                    cgtapprox("&",ii) = &((*(cgtapproxRaw("&",ii))).getML());
                }

                else
                {
                    MEMNEW(cgtapproxRaw("&",ii),ML_Mutable);
                    (*(cgtapproxRaw("&",ii))).setMLTypeClean(altcgtapprox.type());

                    (*(cgtapproxRaw("&",ii))).getML() = altcgtapprox;
                    cgtInd("&",ii) = regML(cgtapproxRaw("&",ii),fnarg,14);

                    cgtapprox("&",ii) = &((*(cgtapproxRaw("&",ii))).getML());
                }

                MLnumbers("&",7) = cgtInd(0);
            }
        }


        {
            NiceAssert( !extaugxapprox.size() || ( extaugxapprox.size() == usemodelaugx ) );

            augxapproxRaw.resize(usemodelaugx);
            augxapprox.resize(usemodelaugx);
            augxInd.resize(usemodelaugx);

            augxapproxRaw = nullptr;
            augxapprox    = nullptr;
            augxInd       = -1;

            int i;

            for ( i = 0 ; i < usemodelaugx ; ++i )
            {
                if ( extaugxapprox.size() && extaugxapprox(i) )
                {
                    MEMNEW(augxapproxRaw("&",i),ML_Mutable);
                    (*(augxapproxRaw("&",i))).setMLTypeClean((*(extaugxapprox(i))).type());

                    (*(augxapproxRaw("&",i))).getML() = *(extaugxapprox(i));
                    augxInd("&",i) = regML(augxapproxRaw("&",i),fnarg,12);

                    augxapprox("&",i) = &((*(augxapproxRaw("&",i))).getML());
                }

                else
                {
                    MEMNEW(augxapproxRaw("&",i),ML_Mutable);
                    (*(augxapproxRaw("&",i))).setMLTypeClean(altaugxapprox.type());

                    (*(augxapproxRaw("&",i))).getML() = altaugxapprox;
                    augxInd("&",i) = regML(augxapproxRaw("&",i),fnarg,12);

                    augxapprox("&",i) = &((*(augxapproxRaw("&",i))).getML());
                }

                MLnumbers("&",6) = augxInd(0);
            }
        }

//RKHSFIXME
        //(*muapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

        sigmaapproxRaw = nullptr;

        if ( sigmuseparate )
        {
            MEMNEW(sigmaapproxRaw ,ML_Mutable);
            (*sigmaapproxRaw).setMLTypeClean(getmuapprox(0).type());

            (*sigmaapproxRaw).getML() = getmuapprox(0);
            sigInd = regML(sigmaapproxRaw,fnarg,7);

            sigmaapprox = &((*sigmaapproxRaw).getML());

            MLnumbers("&",1) = sigInd;

//RKHSFIXME
            //(*sigmaapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
        }

        else
        {
            sigmaapprox = muapprox(0);
            sigInd      = muInd(0);

            MLnumbers("&",1) = -1; //MLnumbers(0);
        }

        // Kernel transfer (must occur before we make copies etc)

        if ( kernapprox )
        {
            errstream() << "realOptim: about to transfer kernel: " << kernapprox << "\n";

            MercerKernel newkern;

            newkern.setAltCall((*kernapprox).MLid());
            newkern.setType(kxfnum);
//RKHSFIXME
            //newkern.setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

            if ( kxfnorm )
            {
                newkern.setNormalised();
            }

            for ( int i = 0 ; i < muapprox.size() ; i++ )
            {
                getmuapprox("&",i).setKernel(newkern);
                getcgtapprox("&",i).setKernel(newkern);
            }

            if ( sigmuseparate )
            {
                getsigmaapprox("&").setKernel(newkern);
            }
        }

        // Record number of points in model for transfer (default: just assume they're from the model we're trying to learn)

        Nbasemu = model_N_mu();

        MLnumbers("&",4) = -1;
        MLnumbers("&",5) = -1;

        // Setup helper variables for env-GP transfer learning

        srcmodel  = nullptr;
        diffmodel = nullptr;

        if ( ( tranmeth == 1 ) && Nbasemu )
        {
            retVector<int> tmpva;

            errstream() << "realOptim: 501 env-GP?\n";

            indpremu = cntintvec(Nbasemu,tmpva);
            presigweightmu.resize(Nbasemu) = 1.0;

            alpha = alpha0;
            beta  = beta0;

            MEMNEW(srcmodel,ML_Mutable);
            (*srcmodel).setMLTypeClean(getmuapprox(0).type());
            (*srcmodel).getML() = getmuapprox(0);
            srcmodelInd = regML(srcmodel ,fnarg,10);

//RKHSFIXME
            //(*srcmodel).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

            errstream() << "realOptim: 501 env-GP: type = " << (*srcmodel).type() << "\n";

            MLnumbers("&",4) = srcmodelInd;
        }

        // Setup helper variables for diff-GP transfer learning

        if ( ( tranmeth == 2 ) && Nbasemu )
        {
            MEMNEW(srcmodel ,ML_Mutable);
            MEMNEW(diffmodel,ML_Mutable);

            (*srcmodel ).setMLTypeClean(getmuapprox(0).type());
            (*diffmodel).setMLTypeClean(getmuapprox(0).type());

            (*srcmodel).getML() = getmuapprox(0);

//RKHSFIXME
            //(*srcmodel).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
            //(*diffmodel).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

            srcmodelInd  = regML(srcmodel ,fnarg,10);
            diffmodelInd = regML(diffmodel,fnarg,11);

            MLnumbers("&",4) = srcmodelInd;
            MLnumbers("&",5) = diffmodelInd;
        }

        srcmodInd  = srcmodelInd;
        diffmodInd = diffmodelInd;

        firsttrain = 1;

        // Optimise

        int res = GlobalOptions::realOptim(dim,xres,rawxres,fres,ires,mres,muInd,augxInd,cgtInd,sigInd,srcmodInd,diffmodInd,allxres,allrawxres,allfres,allcres,allmres,supres,sscore,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch);

        return res;
    }


















































#define NUMLOGSAMP 500


void SMBOOptions::model_log(int stage, double xmin, double xmax, double ymin, double ymax)
{
    std::string stagestr = ( stage == 0 ) ? "pre" : ( ( stage == 1 ) ? "mid" : "post" );

//    static const gentype zg(0.0);
//    int debugit = 0;
//    int debugit = 1;

    if ( !stage )
    {
        // First call - record range for later use!

        log_xmin = xmin;
        log_xmax = xmax;

        log_ymin = ymin;
        log_ymax = ymax;
    }

    else
    {
        // Subsequently retrieve the range

        xmin = log_xmin;
        xmax = log_xmax;

        ymin = log_ymin;
        ymax = log_ymax;
    }

    // Report likelihoods

    //errstream() << "Negative log likelihoods: ";
//errstream() << "phantomxyzxyzxyz Model " << *(muapprox(0)) << "\n";
    errstream() << "NLL: ";

    if ( muapprox.size() )
    {
        int i;

        for ( i = 0 ; i < muapprox.size() ; ++i )
        {
            if ( muapprox(i) )
            {
                errstream() << ", " << calcnegloglikelihood(getmuapprox(i),1) << " (mu " << i << "): ";

                for ( int iii = 0 ; iii < (getmuapprox(i).getKernel()).size() ; iii++ )
                {
                    errstream() << (getmuapprox(i).getKernel()).cWeight(iii) << ", ";
                    errstream() << (getmuapprox(i).getKernel()).cRealConstants(iii) << "\t";
                }

                errstream() << "\n";
            }
        }
    }

    if ( sigmaapprox && ( sigmaapprox != muapprox(0) ) )
    {
        //errstream() << ", " << calcnegloglikelihood(*sigmaapprox,1) << " (sigma)";
    }

    if ( srcmodel )
    {
        //errstream() << ", " << calcnegloglikelihood(*srcmodel,1) << " (src)";
    }

    if ( diffmodel )
    {
        //errstream() << ", " << calcnegloglikelihood(*diffmodel,1) << " (diff)";
    }

    if ( ismodelaug() && augxapprox.size() )
    {
//        int i;
//
//        for ( i = 0 ; i < augxapprox.size() ; ++i )
//        {
//            if ( augxapprox(i) )
//            {
//                errstream() << ", " << calcnegloglikelihood(*augxapprox(i),1) << " (augx " << i << ")";
//            }
//        }
    }

    if ( cgtapprox.size() )
    {
        int i;

        for ( i = 0 ; i < cgtapprox.size() ; ++i )
        {
            if ( cgtapprox(i) )
            {
                errstream() << ", " << calcnegloglikelihood(getcgtapprox(i),1) << " (cgt " << i << "): ";

                for ( int iii = 0 ; iii < (getcgtapprox(i).getKernel()).size() ; iii++ )
                {
                    errstream() << (getcgtapprox(i).getKernel()).cWeight(iii) << ", ";
                    errstream() << (getcgtapprox(i).getKernel()).cRealConstants(iii) << "\t";
                }

                errstream() << "\n";
            }
        }
    }

    //errstream() << "\t";//\n";

    if ( plotfreq && ( ( stage == 0 ) || ( stage == 2 ) || ( ( plotfreq > 0 ) && ( !(model_N_mu()%plotfreq) ) ) ) )
    {
/*
        gentype ytmp,yvartmp;

        Vector<Vector<double> > xplot(ydim);
        Vector<Vector<double> > yplot(ydim);
        Vector<Vector<double> > yvvar(ydim);
        Vector<Vector<double> > ybase(ydim);

        Vector<Vector<double> > xpos(ydim);
        Vector<Vector<double> > ypos(ydim);

        Vector<Vector<double> > xneg(ydim);
        Vector<Vector<double> > yneg(ydim);

        Vector<Vector<double> > xequ(ydim);
        Vector<Vector<double> > yequ(ydim);
*/

        int xind = 0;
        int yind = getdimfid() ? (DEFAULT_TUPLE_INDEX_STEP) : 1; // this is very naive, but as we are only doing 2-dimensions it makes sense
        int j;

//errstream() << "phantomxyz yind = " << yind << "\n";
        //int i;
        //double xindval;

        int ydim = ismoo ? moodim : 1;

        for ( j = 0 ; j < ydim ; j++ )
        {
            gentype baseline(modelbaseline);
            int incbaseline = ( ( modelbaseline == "null" ) ? 0 : 1 );

            model_sublog(getmuapprox(j),baseline,incbaseline,xmin,xmax,ymin,ymax,j,"model",xind,yind,stagestr,-1);
        }

        if ( usemodelaugx )
        {
            for ( j = 0 ; j < augxapprox.size() ; ++j )
            {
                gentype baseline = nullgentype();
                int incbaseline = 0;

                model_sublog(getaugxapprox(j),baseline,incbaseline,xmin,xmax,ymin,ymax,j,"aug",xind,yind,stagestr,+1);
            }
        }


        if ( cgtapprox.size() )
        {
            for ( j = 0 ; j < cgtapprox.size() ; ++j )
            {
                gentype baseline = nullgentype();
                int incbaseline = 0;

                model_sublog(getcgtapprox(j),baseline,incbaseline,xmin,xmax,ymin,ymax,j,"cgt",xind,yind,stagestr,+1);
            }
        }
    }


/*
            xplot("&",j).resize(NUMLOGSAMP);
            yplot("&",j).resize(NUMLOGSAMP);
            yvvar("&",j).resize(NUMLOGSAMP);
            ybase("&",j).resize(NUMLOGSAMP);

            for ( i = 0, xindval = xmin ; xindval <= xmax ; ++i, xindval += (xmax-xmin)/NUMLOGSAMP )
            {
                SparseVector<gentype> x;

                x.zero();
                x("&",xind) = xindval;

                model_muvar(yvartmp,ytmp,x,nullptr,debugit);

                xplot("&",j)("&",i) = xindval;
                yplot("&",j)("&",i) = ismoo ? ( -((double) (ytmp.cast_vector())(j)) ) : ( -((double) ytmp) );
                yvvar("&",j)("&",i) = ismoo ? ( (double) (yvartmp.cast_vector())(j) ) : ( (double) yvartmp );
                ybase("&",j)("&",i) = ( ismoo || baseline.isValNull() ) ? 0.0 : ((double) baseline(x(xind)));
            }

            for ( i = 0 ; i < model_N_mu(j) ; ++i )
            {
                if ( (model_d(j))(i) == +1 )
                {
                    xpos("&",j).add(xpos(j).size()); xpos("&",j)("&",xpos(j).size()-1) = (double) locxresunconv(i)(xind); //(model_x(i))(xind);
                    ypos("&",j).add(ypos(j).size()); ypos("&",j)("&",ypos(j).size()-1) = -((double) (model_y(j))(i));
                }

                if ( (model_d(j))(i) == -1 )
                {
                    xneg("&",j).add(xneg(j).size()); xneg("&",j)("&",xneg(j).size()-1) = (double) locxresunconv(i)(xind); //(model_x(i))(xind);
                    yneg("&",j).add(yneg(j).size()); yneg("&",j)("&",yneg(j).size()-1) = -((double) (model_y(j))(i));
                }

                if ( (model_d())(i) == 2 )
                {
                    xequ("&",j).add(xequ(j).size()); xequ("&",j)("&",xequ(j).size()-1) = (double) locxresunconv(i)(xind); //(model_x(i))(xind);
                    yequ("&",j).add(yequ(j).size()); yequ("&",j)("&",yequ(j).size()-1) = -((double) (model_y(j))(i));
                }
            }

            std::string fname(modelname);
            std::string dname(modelname);
            std::string mlname(modelname);

            fname += "_";
            dname += "_";
            mlname += "_";

            fname += std::to_string(j);
            dname += std::to_string(j);
            mlname += std::to_string(j);

            fname += ".smbo.plot";
            dname += ".smbo.data";
            mlname += ".smbo.gpr";

            std::stringstream ssi;
            std::stringstream ssj;

            ssi << model_N_mu()/plotfreq;
            ssj << stage;

            fname += ssj.str();
            dname += ssj.str();
            mlname += ssj.str();

            fname += "_";
            dname += "_";
            mlname += "_";

            fname += ssi.str();
            dname += ssi.str();
            mlname += ssi.str();

            double ymin = ( testisinf(softmin) || testisinf(softmax) ) ? 1 : -softmax;
            double ymax = ( testisinf(softmin) || testisinf(softmax) ) ? 1 : -softmin;

            int incdata = baseline.isValNull() ? 1 : 3;
            int incvar  = 1;

            //plotit(xplot,yplot,yvvar,ybase,xpos,ypos,xneg,yneg,xequ,yequ,xmin,xmax,1,1,fname,dname,modeloutformat,incdata,incvar);
            plotit(xplot(j),yplot(j),yvvar(j),ybase(j),xpos(j),ypos(j),xneg(j),yneg(j),xequ(j),yequ(j),xmin,xmax,ymin,ymax,fname,dname,modeloutformat,incdata,incvar);
            //plotit(xplot,yplot,yvvar,ybase,xpos,ypos,xneg,yneg,xequ,yequ,0,1,ymin,ymax,fname,dname,modeloutformat,incdata,incvar);

            std::ofstream mlnamefile(mlname);

            mlnamefile << (*(muapprox(j))) << "\n";

            mlnamefile.close();

            if ( ( (*(muapprox(j))).tspaceDim() == 1 ) && ( (*(muapprox(j))).xspaceDim() == 1 ) )
            {
                //double xmin = 0;
                //double xmax = 1;

                //double ymin = 0;
                //double ymax = 1;

                double omin = 1;
                double omax = 0;

                std::string fname(modelname);
                std::string ffname = "mu";
                std::string dname(modelname);
                std::string mlname(modelname);

                fname  += "_";
                ffname += "_";
                dname  += "_";
                mlname += "_";

                fname  += std::to_string(j);
                ffname += std::to_string(j);
                dname  += std::to_string(j);
                mlname += std::to_string(j);

                fname  += "_smbo_model_plot";
                ffname += "_smbo_model_plot";
                dname  += "_smbo_model_data";
                mlname += "_smbo_model_gpr";

                std::stringstream ssi;

                ssi << model_N_mu(); // /plotfreq;

                fname  += stagestr;
                //ffname += stagestr;
                dname  += stagestr;
                mlname += stagestr;

                fname  += "_";
                //ffname += "_";
                dname  += "_";
                mlname += "_";

                fname  += ssi.str();
                //ffname += ssi.str();
                dname  += ssi.str();
                mlname += ssi.str();

                std::ofstream mlnamefile(mlname);
                mlnamefile << (*(muapprox(j))) << "\n";
                mlnamefile.close();

                int incdata = incbaseline ? 3 : 1;
                int incvar  = 1;
                int xusevar = 0;

                SparseVector<gentype> xtemplate;

                plotml((*(muapprox(j))),xind,xmin,xmax,omin,omax,fname, dname,modeloutformat,incdata,baseline,incvar,xusevar,xtemplate,0,0,-1);
                plotml((*(muapprox(j))),xind,xmin,xmax,omin,omax,ffname,dname,modeloutformat,incdata,baseline,incvar,xusevar,xtemplate,0,0,-1);
            }

            else if ( ( (*(muapprox(j))).tspaceDim() == 1 ) && ( (*(muapprox(j))).xspaceDim() == 2 ) )
            {
                //double xmin = 0;
                //double xmax = 1;

                //double ymin = 0;
                //double ymax = 1;

                double omin = 1;
                double omax = 0;

                std::string fname(modelname);
                std::string ffname = "mu";
                std::string dname(modelname);
                std::string mlname(modelname);

                fname  += "_";
                ffname += "_";
                dname  += "_";
                mlname += "_";

                fname  += std::to_string(j);
                ffname += std::to_string(j);
                dname  += std::to_string(j);
                mlname += std::to_string(j);

                fname  += "_smbo_model_plot";
                ffname += "_smbo_model_plot";
                dname  += "_smbo_model_data";
                mlname += "_smbo_model_gpr";

                std::stringstream ssi;

                ssi << model_N_mu(); // /plotfreq;

                fname  += stagestr;
                //ffname += stagestr;
                dname  += stagestr;
                mlname += stagestr;

                fname  += "_";
                //ffname += "_";
                dname  += "_";
                mlname += "_";

                fname  += ssi.str();
                //ffname += ssi.str();
                dname  += ssi.str();
                mlname += ssi.str();

                std::ofstream mlnamefile(mlname);
                mlnamefile << (*(muapprox(j))) << "\n";
                mlnamefile.close();

                int incdata = incbaseline ? 2 : 1;
                int incvar  = 1;
                int xusevar = 0;
                int yusevar = 1;

                SparseVector<gentype> xtemplate;

                plotml((*(muapprox(j))),xind,yind,xmin,xmax,ymin,ymax,omin,omax,fname, dname,modeloutformat,incdata,baseline,incvar,xusevar,yusevar,xtemplate,0,0,-1);
                plotml((*(muapprox(j))),xind,yind,xmin,xmax,ymin,ymax,omin,omax,ffname,dname,modeloutformat,incdata,baseline,incvar,xusevar,yusevar,xtemplate,0,0,-1);
            }

            else
            {
                std::string mlname(modelname);

                mlname += "_";

                mlname += std::to_string(j);

                mlname += ".smbo.model.gpr";

                std::stringstream ssi;

                ssi << model_N_mu(); // /plotfreq;

                mlname += stagestr;

                mlname += "_";

                mlname += ssi.str();

                std::ofstream mlnamefile(mlname);
                mlnamefile << (*(muapprox(j))) << "\n";
                mlnamefile.close();
            }
        }

                if ( ( (*(augxapprox(j))).tspaceDim() == 1 ) && ( (*(augxapprox(j))).xspaceDim() == 1 ) )
                {
                    //double xmin = 0;
                    //double xmax = 1;

                    double omin = 1;
                    double omax = 0;

                    std::string fname(modelname);
                    std::string ffname = "augx";
                    std::string dname(modelname);
                    std::string mlname(modelname);

                    fname  += "_";
                    ffname += "_";
                    dname  += "_";
                    mlname += "_";

                    fname  += std::to_string(j);
                    ffname += std::to_string(j);
                    dname  += std::to_string(j);
                    mlname += std::to_string(j);

                    fname  += "_smbo_aug_plot";
                    ffname += "_smbo_aug_plot";
                    dname  += "_smbo_aug_data";
                    mlname += "_smbo_aug_gpr";

                    std::stringstream ssi;

                    ssi << model_N_mu(); // /plotfreq;

                    fname  += stagestr;
                    //ffname += stagestr;
                    dname  += stagestr;
                    mlname += stagestr;

                    fname  += "_";
                    //ffname += "_";
                    dname  += "_";
                    mlname += "_";

                    fname  += ssi.str();
                    //ffname += ssi.str();
                    dname  += ssi.str();
                    mlname += ssi.str();

                    std::ofstream mlnamefile(mlname);
                    mlnamefile << (*(augxapprox(j))) << "\n";
                    mlnamefile.close();

                    int incdata = 1;
                    int incvar  = 1;
                    int xusevar = 0;

                    SparseVector<gentype> xtemplate;

                    plotml((*(augxapprox(j))),xind,xmin,xmax,omin,omax,fname, dname,modeloutformat,incdata,nullgentype(),incvar,xusevar,xtemplate,0,0,-1);
                    plotml((*(augxapprox(j))),xind,xmin,xmax,omin,omax,ffname,dname,modeloutformat,incdata,nullgentype(),incvar,xusevar,xtemplate,0,0,-1);
                }

                else if ( ( (*(augxapprox(j))).tspaceDim() == 1 ) && ( (*(augxapprox(j))).xspaceDim() == 2 ) )
                {
                    //double xmin = 0;
                    //double xmax = 1;

                    //double ymin = 0;
                    //double ymax = 1;

                    double omin = 1;
                    double omax = 0;

                    std::string fname(modelname);
                    std::string ffname = "augx";
                    std::string dname(modelname);
                    std::string mlname(modelname);

                    fname  += "_";
                    ffname += "_";
                    dname  += "_";
                    mlname += "_";

                    fname  += std::to_string(j);
                    ffname += std::to_string(j);
                    dname  += std::to_string(j);
                    mlname += std::to_string(j);

                    fname  += "_smbo_aug_plot";
                    ffname += "_smbo_aug_plot";
                    dname  += "_smbo_aug_data";
                    mlname += "_smbo_aug_gpr";

                    std::stringstream ssi;

                    ssi << model_N_mu(); // /plotfreq;

                    fname  += stagestr;
                    //ffname += stagestr;
                    dname  += stagestr;
                    mlname += stagestr;

                    fname  += "_";
                    //ffname += "_";
                    dname  += "_";
                    mlname += "_";

                    fname  += ssi.str();
                    //ffname += ssi.str();
                    dname  += ssi.str();
                    mlname += ssi.str();

                    std::ofstream mlnamefile(mlname);
                    mlnamefile << (*(augxapprox(j))) << "\n";
                    mlnamefile.close();

                    int incdata = 1;
                    int incvar  = 1;
                    int xusevar = 0;
                    int yusevar = 1;

                    SparseVector<gentype> xtemplate;

                    plotml((*(augxapprox(j))),xind,yind,xmin,xmax,ymin,ymax,omin,omax,fname, dname,modeloutformat,incdata,nullgentype(),incvar,xusevar,yusevar,xtemplate,0,0,-1);
                    plotml((*(augxapprox(j))),xind,yind,xmin,xmax,ymin,ymax,omin,omax,ffname,dname,modeloutformat,incdata,nullgentype(),incvar,xusevar,yusevar,xtemplate,0,0,-1);
                }

                else
                {
                    std::string mlname(modelname);

                    mlname += "_";

                    mlname += std::to_string(j);

                    mlname += ".smbo.model.gpr";

                    std::stringstream ssi;

                    ssi << model_N_mu(); // /plotfreq;

                    mlname += stagestr;

                    mlname += "_";

                    mlname += ssi.str();

                    std::ofstream mlnamefile(mlname);
                    mlnamefile << (*(augxapprox(j))) << "\n";
                    mlnamefile.close();
                }
            }
*/

    return GlobalOptions::model_log(stage,xmin,xmax,ymin,ymax);
}


void SMBOOptions::model_sublog(const ML_Base &plotmodel, gentype &baselinefn, int incbaselinefn, double xmin, double xmax, double ymin, double ymax, int j, const std::string &nameof, int xind, int yind, const std::string &stagestr, double sf)
{
    double omin = 1;
    double omax = 0;

    std::string fname(modelname);
    std::string ffname = "mu";
    std::string dname(modelname);
    std::string mlname(modelname);

    fname  += "_";
    ffname += "_";
    dname  += "_";
    mlname += "_";

    fname  += std::to_string(j);
    ffname += std::to_string(j);
    dname  += std::to_string(j);
    mlname += std::to_string(j);

    fname  += "_smbo_";
    ffname += "_smbo_";
    dname  += "_smbo_";
    mlname += "_smbo_";

    fname  += nameof;
    ffname += nameof;
    dname  += nameof;
    mlname += nameof;

    fname  += "_plot";
    ffname += "_plot";
    dname  += "_data";
    mlname += "_gpr";

    fname  += stagestr;
    //ffname += stagestr;
    dname  += stagestr;
    mlname += stagestr;

    fname  += "_";
    //ffname += "_";
    dname  += "_";
    mlname += "_";

    fname  += std::to_string(plotmodel.N());
    //ffname += std::to_string(plotmodel.N());
    dname  += std::to_string(plotmodel.N());
    mlname += std::to_string(plotmodel.N());

    std::ofstream mlnamefile(mlname);
    mlnamefile << plotmodel << "\n";
    mlnamefile.close();

//errstream() << "phantomxyz tspaceDim = " << plotmodel.tspaceDim() << "\n";
//errstream() << "phantomxyz xspaceDim = " << plotmodel.xspaceDim() << "\n";
    if ( ( plotmodel.tspaceDim() == 1 ) && plotmodel.N() && ( plotmodel.x()(0).indsize() == 1 ) ) //( plotmodel.xspaceDim() == 1 ) )
    {
        int incdata = incbaselinefn ? 3 : 1;
        int incvar  = 1;
        int xusevar = 0;
//errstream() << "phantomxyz plot model 1d\n";

        SparseVector<gentype> xtemplate;

        plotml(plotmodel,xind,xmin,xmax,omin,omax,fname, dname,modeloutformat,incdata,baselinefn,incvar,xusevar,xtemplate,0,0,sf);
        plotml(plotmodel,xind,xmin,xmax,omin,omax,ffname,dname,modeloutformat,incdata,baselinefn,incvar,xusevar,xtemplate,0,0,sf);
    }

    else if ( ( plotmodel.tspaceDim() == 1 ) && plotmodel.N() && ( plotmodel.x()(0).indsize() == 2 ) ) //( plotmodel.xspaceDim() == 2 ) )
    {
        int incdata = incbaselinefn ? 2 : 1;
        int incvar  = 1;
        int xusevar = 0;
        int yusevar = 1;
//errstream() << "phantomxyz plot model 2d\n";

        SparseVector<gentype> xtemplate;

        plotml(plotmodel,xind,yind,xmin,xmax,ymin,ymax,omin,omax,fname, dname,modeloutformat,incdata,baselinefn,incvar,xusevar,yusevar,xtemplate,0,0,sf);
        plotml(plotmodel,xind,yind,xmin,xmax,ymin,ymax,omin,omax,ffname,dname,modeloutformat,incdata,baselinefn,incvar,xusevar,yusevar,xtemplate,0,0,sf);
    }

    return;
}


void SMBOOptions::model_clear(void)
{
    if ( ( modeltype == 1 ) || ( modeltype == 2 ) )
    {
        int i;

        for ( i = 0 ; i < muapprox.size() ; ++i )
        {
            getmuapprox("&",i).removeTrainingVector(0,getmuapprox(i).N());
        }

        for ( i = 0 ; i < cgtapprox.size() ; ++i )
        {
            getcgtapprox("&",i).removeTrainingVector(0,getcgtapprox(i).N());
        }

        getsigmaapprox("&").removeTrainingVector(0,getsigmaapprox().N());

        for ( i = 0 ; i < augxapprox.size() ; ++i )
        {
            getaugxapprox("&",i).removeTrainingVector(0,getaugxapprox(i).N());
        }
    }
}

void SMBOOptions::model_update(void)
{
#ifndef TURNOFFSHORTCUT
    if ( !ismodelaug() && ( ( modeltype == 0 ) || ( modeltype == 1 ) ) )
    {
        NiceAssert( muapprox == sigmaapprox );

        int N = getmuapprox(0).N();
        int m = getxbasis().size();

        (xmodprod).resize(N,m);

        int i,j;

        for ( i = 0 ; i < N ; ++i )
        {
            for ( j = 0 ; j < m ; ++j )
            {
                innerProduct((xmodprod)("&",i,j),getmuapprox(0).x(i),getxbasis()(j));
            }
        }

        xshortcutenabled = 1;
    }
#endif
}

void SMBOOptions::model_sample(const Vector<double> &qmin, const Vector<double> &qmax, double sampScale)
{
    if ( !model_issample() )
    {
        SparseVector<double> qqmin(qmin);
        SparseVector<double> qqmax(qmax);

        SparseVector<gentype> xmin; xmin.castassign(qqmin);
        SparseVector<gentype> xmax; xmax.castassign(qqmax);

        muapprox_sample.resize(muapprox.size());
        muapprox_sample = nullptr;

        int i;

errstream() << "Sample objective model\n";
        for ( i = 0 ; i < muapprox_sample.size() ; i++ )
        {
            muapprox_sample("&",i) = makeDupML(getmuapprox(i));
        }

        SparseVector<gentype> xxmin;
        SparseVector<gentype> xxmax;

        xxmax = model_convertx(xxmax,xmax);
        xxmin = model_convertx(xxmin,xmin);

        xxmax.makealtcontent();
        xxmin.makealtcontent();

        for ( i = 0 ; i < muapprox_sample.size() ; i++ )
        {
            retVector<gentype> xminrettmp;
            retVector<gentype> xmaxrettmp;

            int sampSplit = 0; // We want true random samples here!

            getmuapprox_sample("&",i).setsigma(SIGMA_ADD); // effectively noiseless samples for practical purposes
            getmuapprox_sample("&",i).setsigma_cut(sigma_cut); // scale factor may be set
            getmuapprox_sample("&",i).setSampleMode(TSmode,xxmin(xminrettmp),xxmax(xmaxrettmp),TSNsamp,sampSplit,TSsampType,TSxsampType,sampScale);
        }
    }

errstream() << "Sampling complete\n";
    return;
}

void SMBOOptions::model_unsample(void)
{
    if ( model_issample() )
    {
        int i;

        for ( i = 0 ; i < muapprox_sample.size() ; i++ )
        {
            MEMDEL(muapprox_sample("&",i));
            muapprox_sample("&",i) = nullptr;
        }
    }

    muapprox_sample.resize(0);

    return;
}

int SMBOOptions::model_issample(void) const
{
    return ( muapprox_sample.size() && muapprox_sample(0) ) ? 1 : 0;
}


int SMBOOptions::initModelDistr(const Vector<int> &sampleInd, const Vector<gentype> &sampleDist)
{
    if ( muapprox.size() )
    {
        int i;

        for ( i = 0 ; i < muapprox.size() ; i++ )
        {
            getmuapprox("&",i).getKernel_unsafe().setSampleDistribution(sampleDist);
            getmuapprox("&",i).getKernel_unsafe().setSampleIndices(sampleInd);
            getmuapprox("&",i).resetKernel();
        }
    }

    if ( cgtapprox.size() )
    {
        int i;

        for ( i = 0 ; i < cgtapprox.size() ; i++ )
        {
            getcgtapprox("&",i).getKernel_unsafe().setSampleDistribution(sampleDist);
            getcgtapprox("&",i).getKernel_unsafe().setSampleIndices(sampleInd);
            getcgtapprox("&",i).resetKernel();
        }
    }

    if ( sigmuseparate )
    {
        getsigmaapprox("&").getKernel_unsafe().setSampleDistribution(sampleDist);
        getsigmaapprox("&").getKernel_unsafe().setSampleIndices(sampleInd);
        getsigmaapprox("&").resetKernel();
    }

    return 1;
}

int SMBOOptions::model_muTrainingVector(gentype &resmu, int imu) const
{
    int res = 0;

    if ( !muapprox.size() )
    {
        resmu.force_null();
    }

    else if ( muapprox.size() == 1 )
    {
        res |= getmuapprox(0).ggTrainingVector(resmu,imu);
    }

    else
    {
        Vector<gentype> &resmuvec = resmu.force_vector(muapprox.size());

        for ( int i = 0 ; i < muapprox.size() ; ++i )
        {
            res |= getmuapprox(i).ggTrainingVector(resmuvec("&",i),imu);
        }
    }

    return res;
}

int SMBOOptions::model_muTrainingVector(Vector<double> &resmu, int imu) const
{
     gentype resgmu;

     int res = model_muTrainingVector(resgmu,imu);

     resmu = (const Vector<double> &) resgmu;

     return res;
}

int SMBOOptions::model_muvarTrainingVector(gentype &resvar, gentype &resmu, int ivar, int imu) const
{
    int resi = 0;

    if ( !muapprox.size() )
    {
        resvar.force_null();
        resmu.force_null();
    }

    else if ( muapprox.size() == 1 )
    {
        if ( !sigmuseparate )
        {
            resi |= getmuapprox_sample(0).varTrainingVector(resvar,resmu,imu);
        }

        else
        {
            gentype dummy;

            resi |= getmuapprox_sample(0).ggTrainingVector(resmu,imu);
            resi |= getsigmaapprox().varTrainingVector(resvar,dummy,ivar);
        }
    }

    else
    {
        Vector<gentype> &resmuvec = resmu.force_vector(muapprox.size());
        Vector<gentype> &resvarvec = resvar.force_vector(muapprox.size());

        for ( int i = 0 ; i < muapprox.size() ; ++i )
        {
            if ( !sigmuseparate )
            {
                resi |= getmuapprox_sample(i).varTrainingVector(resvarvec("&",i),resmuvec("&",i),imu);
            }

            else
            {
                gentype dummy;

                resi |= getmuapprox_sample(i).ggTrainingVector(resmuvec("&",i),imu);
                resi |= getsigmaapprox().varTrainingVector(resvarvec("&",i),dummy,ivar);
            }
        }
    }

    return resi;
}

int SMBOOptions::model_varTrainingVector(gentype &resv, int imu) const
{
    int resi = 0;

    if ( !muapprox.size() )
    {
        resv.force_null();
    }

    else if ( muapprox.size() == 1 )
    {
        if ( !sigmuseparate )
        {
            gentype dummy;

            resi |= getmuapprox_sample(0).varTrainingVector(resv,dummy,imu);
        }

        else
        {
            gentype dummy;

            resi |= getsigmaapprox().varTrainingVector(resv,dummy,imu);
        }
    }

    else
    {
        Vector<gentype> &resvarvec = resv.force_vector(muapprox.size());

        for ( int i = 0 ; i < muapprox.size() ; ++i )
        {
            if ( !sigmuseparate )
            {
                gentype dummy;

                resi |= getmuapprox_sample(i).varTrainingVector(resvarvec("&",i),dummy,imu);
            }

            else
            {
                gentype dummy;

                resi |= getsigmaapprox().varTrainingVector(resvarvec("&",i),dummy,imu);
            }
        }
    }

    return resi;
}

int SMBOOptions::model_muTrainingVector_cgt(Vector<gentype> &resmu, int imu) const
{
    int res = 0;

    resmu.resize(cgtapprox.size());

    for ( int i = 0 ; i < cgtapprox.size() ; ++i )
    {
        res |= getcgtapprox(i).ggTrainingVector(resmu("&",i),imu);
    }

    return res;
}

int SMBOOptions::model_muvarTrainingVector_cgt(Vector<gentype> &resv, Vector<gentype> &resmu, int imu) const
{
    int resi = 0;

    resv.resize(cgtapprox.size());
    resmu.resize(cgtapprox.size());

    for ( int i = 0 ; i < cgtapprox.size() ; ++i )
    {
        resi |= getcgtapprox(i).varTrainingVector(resv("&",i),resmu("&",i),imu);
    }

    return resi;
}

int SMBOOptions::model_train(int &res, svmvolatile int &killSwitch)
{
        int i,ires = 0;

        ires |= modeldiff_int_train(res,killSwitch);

        if ( tunediffmod && diffmodel )
        {
//errstrean() << "phantomxyziii diffmodel\n";
            (*diffmodel).getML().tuneKernel(tunediffmod,getxwidth(),1,0,nullptr);
        }

        ires |= modelaugx_int_train(res,killSwitch);

        if ( tuneaugxmod && ismodelaug() )
        {
            for ( i = 0 ; i < augxapprox.size() ; ++i )
            {
//errstrean() << "phantomxyziii augx " << i << "\n";
                getaugxapprox("&",i).getML().tuneKernel(tuneaugxmod,getxwidth(),1,0,nullptr);
            }
        }

        ires |= modelmu_int_train(res,killSwitch);
        ires |= modelcgt_int_train(res,killSwitch);

        if ( tunemu )
        {
//errstrean() << "phantomxyziii muapprox\n";
            for ( i = 0 ; i < muapprox.size() ; ++i )
            {
                tkBounds tuneBounds(getmuapprox("&",i).getML().getKernel());

                // It is possible for BOCA to get "stuck" with bad parameters:
                // - too little connection between different different fidelities (no connection)
                // - posterior variance over-shrunk (too confindent)

                if ( getmuapprox(i).getMLconst().N() < 40 )
                {
                    tuneBounds.wlb = 0.5;

                    for ( int i = 0 ; i < getdimfid() ; i++ )
                    {
                        tuneBounds.klb("&",i+1) = 0.6;
                    }
// original heuristic: 0.5 if N <= 40, unchange otherwise
                }

                else if ( getmuapprox(i).getMLconst().N() < 50 )
                {
                    double scale = (50.0-getmuapprox("&",i).getML().N())/10.0;

                    tuneBounds.wlb = 0.1+(0.4*scale);

                    for ( int i = 0 ; i < getdimfid() ; i++ )
                    {
                        tuneBounds.klb("&",i+1) = 0.6*scale;
                    }
                }

                getmuapprox("&",i).getML().tuneKernel(tunemu,getxwidth(),1,0,&tuneBounds);
            }
        }

        if ( tunecgt && cgtapprox.size() )
        {
            for ( i = 0 ; i < cgtapprox.size() ; ++i )
            {
                tkBounds tuneBounds(getcgtapprox(i).getMLconst().getKernel());

                if ( getcgtapprox(i).getMLconst().N() < 40 )
                {
                    tuneBounds.wlb = 0.5;

                    for ( int i = 0 ; i < getdimfid() ; i++ )
                    {
                        tuneBounds.klb("&",i+1) = 0.6;
                    }
                }

                else if ( getcgtapprox(i).getMLconst().N() < 50 )
                {
                    double scale = (50.0-getcgtapprox(i).getMLconst().N())/10.0;

                    tuneBounds.wlb = 0.1+(0.4*scale);

                    for ( int i = 0 ; i < getdimfid() ; i++ )
                    {
                        tuneBounds.klb("&",i+1) = 0.6*scale;
                    }
                }

                getcgtapprox("&",i).getML().tuneKernel(tunecgt,getxwidth(),1,0,&tuneBounds);
            }
        }

        ires |= modelsigma_int_train(res,killSwitch);

        if ( tunesigma && sigmuseparate )
        {
//errstrean() << "phantomxyziii sigapprox\n";
            getsigmaapprox("&").getML().tuneKernel(tunesigma,getxwidth(),1,0,nullptr);
        }

        return ires;
}

int SMBOOptions::model_train_sigma(int &res, svmvolatile int &killSwitch)
{
    return modelsigma_int_train(res,killSwitch);
}

int SMBOOptions::model_setd(int imu, int isigma, int nd)
{
    int res = 0;
    int i;

    for ( i = 0 ; i < muapprox.size() ; i++ )
    {
        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
        int nnd = ( nd == -2 ) ? 2 : nd;

        res += getmuapprox("&",i).setd(iimu,nnd);
    }

    if ( sigmuseparate )
    {
        res += getsigmaapprox("&").setd(isigma,nd);
    }

    return res;
}

int SMBOOptions::model_setd_mu(int imu, int nd)
{
    int res = 0;
    int i;

    for ( i = 0 ; i < muapprox.size() ; i++ )
    {
        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
        int nnd = ( nd == -2 ) ? 2 : nd;

        res += getmuapprox("&",i).setd(iimu,nnd);
    }

    return res;
}

int SMBOOptions::model_setd_cgt(int imu, const Vector<int> &nd)
{
    int res = 0;
    int i;

    for ( i = 0 ; i < cgtapprox.size() ; i++ )
    {
        int iimu = ( ( getcgtapprox(i).type() == 212 ) ? (-imu-1) : imu );
        int nnd = ( nd(i) == -2 ) ? 2 : nd(i);

        res += getcgtapprox("&",i).setd(iimu,nnd);
    }

    return res;
}

int SMBOOptions::model_setd_sigma(int isigma, int nd)
{
    int res = 0;

    if ( sigmuseparate )
    {
        res += getsigmaapprox("&").setd(isigma,nd);
    }

    return res;
}

int SMBOOptions::model_setyd(int imu, int isigma, int nd, const gentype &ny, double varadd)
{
    int res = 0;
    int i;

    for ( i = 0 ; i < muapprox.size() ; i++ )
    {
        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
        int nnd = ( nd == -2 ) ? 2 : nd;

        res += getmuapprox("&",i).sety(iimu,ny);
        res += getmuapprox("&",i).setd(iimu,nnd);

        if ( varadd )
        {
            res |= getmuapprox("&",i).setsigmaweight(getmuapprox(i).N()-1,1+(varadd/(getmuapprox(i).sigma())));
        }
    }

    if ( sigmuseparate )
    {
        res += getsigmaapprox("&").sety(isigma,ny);
        res += getsigmaapprox("&").setd(isigma,nd);

        if ( varadd )
        {
            res |= getsigmaapprox("&").setsigmaweight(getsigmaapprox().N()-1,1+(varadd/(getsigmaapprox().sigma())));
        }
    }

    return res;
}

int SMBOOptions::model_setyd_mu(int imu, int nd, const gentype &ny, double varadd)
{
    int res = 0;
    int i;

    for ( i = 0 ; i < muapprox.size() ; i++ )
    {
        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
        int nnd = ( nd == -2 ) ? 2 : nd;

        res += getmuapprox("&",i).sety(iimu,ny);
        res += getmuapprox("&",i).setd(iimu,nnd);

        if ( varadd )
        {
            res |= getmuapprox("&",i).setsigmaweight(getmuapprox(i).N()-1,1+(varadd/(getmuapprox(i).sigma())));
        }
    }

    return res;
}

int SMBOOptions::model_setyd_cgt(int imu, const Vector<int> &nd, const Vector<gentype> &ny, double varadd)
{
    int res = 0;
    int i;

    for ( i = 0 ; i < cgtapprox.size() ; i++ )
    {
        int iimu = ( ( getcgtapprox(i).type() == 212 ) ? (-imu-1) : imu );
        int nnd = ( nd(i) == -2 ) ? 2 : nd(i);

        res += getcgtapprox("&",i).sety(iimu,ny(i));
        res += getcgtapprox("&",i).setd(iimu,nnd);

        if ( varadd )
        {
            res |= getcgtapprox("&",i).setsigmaweight(getcgtapprox(i).N()-1,1+(varadd/(getcgtapprox(i).sigma())));
        }
    }

    return res;
}

int SMBOOptions::model_setyd_sigma(int isigma, int nd, const gentype &ny, double varadd)
{
    int res = 0;

    if ( sigmuseparate )
    {
        res += getsigmaapprox("&").sety(isigma,ny);
        res += getsigmaapprox("&").setd(isigma,nd);

        if ( varadd )
        {
            res |= getsigmaapprox("&").setsigmaweight(getsigmaapprox().N()-1,1+(varadd/(getsigmaapprox().sigma())));
        }
    }

    return res;
}

int SMBOOptions::modelmu_int_train(int &res, svmvolatile int &killSwitch)
{
    int ires = 0;
    int i;

    for ( i = 0 ; i < muapprox.size() ; i++ )
    {
        ires += getmuapprox("&",i).train(res,killSwitch);
    }

    return ires;
}

int SMBOOptions::modelcgt_int_train(int &res, svmvolatile int &killSwitch)
{
    int ires = 0;
    int i;

    for ( i = 0 ; i < cgtapprox.size() ; i++ )
    {
        ires += getcgtapprox("&",i).train(res,killSwitch);
    }

    return ires;
}

int SMBOOptions::modelsigma_int_train(int &res, svmvolatile int &killSwitch)
{
    int ires = 0;

    if ( sigmuseparate )
    {
        ires += getsigmaapprox("&").train(res,killSwitch);
    }

    return ires;
}

int SMBOOptions::model_setsigmaweight_addvar(int imu, int isigma, double addvar)
{
    int i,res = 0;

    for ( i = 0 ; i < muapprox.size() ; i++ )
    {
        res += getmuapprox("&",i).setsigmaweight(imu,(getmuapprox(i).sigma()+addvar)/getmuapprox(i).sigma());
    }

    if ( sigmuseparate )
    {
        res += getsigmaapprox("&").setsigmaweight(isigma,(getsigmaapprox().sigma()+addvar)/getsigmaapprox().sigma());
    }

    return res;
}

int SMBOOptions::model_addTrainingVector_musigma(const gentype &y, const gentype &ypred, const SparseVector<gentype> &x, double varadd)
{
    int ires = 0;

    //SparseVector<gentype> xx;

    locxresunconv.add(locxresunconv.size());
    locxresunconv("&",locxresunconv.size()-1) = x;

    {
        const SparseVector<gentype> &xxx = model_convertx(xx,x);

        ires |= modeldiff_int_addTrainingVector(y,ypred,xxx,2,varadd);
        ires |= modelmu_int_addTrainingVector(y,x,xxx,2,varadd);

        if ( sigmuseparate )
        {
            ires = modelsigma_int_addTrainingVector(y,xxx,2,varadd);
        }
    }

    return ires;
}

int SMBOOptions::model_addTrainingVector_musigma(const gentype &y, const gentype &ypred, const SparseVector<gentype> &x, const Vector<gentype> &xsidechan, int xobstype, double varadd)
{
        int ires = 0;

        locxresunconv.add(locxresunconv.size());
        locxresunconv("&",locxresunconv.size()-1) = x;

        if ( ismodelaug() )
        {
            SparseVector<gentype> xmod;

            // Suppress model to do initial conversion
            int dummy = usemodelaugx;
            usemodelaugx = 0;
            usemodelaugx = dummy;

            ires |= modelaugx_int_addTrainingVector(xsidechan,model_convertx(xmod,x),varadd);
        }

        SparseVector<gentype> xxx(x);

//errstream() << "phantomxyzxyzxyz bayesopt addvec x: " << x << "\n";
//errstream() << "phantomxyzxyzxyz bayesopt addvec xxx: " << xxx << "\n";
        addinxsidechan(xxx,xsidechan);
//errstream() << "phantomxyzxyzxyz bayesopt addvec xxx with side: " << xxx << "\n";

        {
            const SparseVector<gentype> &xxxxx = model_convertx(xx,xxx);
//errstream() << "phantomxyzxyzxyz bayesopt addvec xx: " << xx << "\n";

            ires |= modeldiff_int_addTrainingVector(y,ypred,xxxxx,xobstype,varadd);
//errstream() << "phantomxyzxyzxyz bayesopt addvec xx(b): " << xx << "\n";
            ires |= modelmu_int_addTrainingVector(y,x,xxxxx,xobstype,varadd);

            if ( sigmuseparate )
            {
                NiceAssert( xobstype == 2 );

                ires = modelsigma_int_addTrainingVector(y,xxxxx,xobstype,varadd);
            }
        }

        return ires;
}

int SMBOOptions::model_addTrainingVector_cgt(const Vector<gentype> &y, const SparseVector<gentype> &x, double varadd)
{
    Vector<int> xobstype(y.size());

    xobstype = 2;

    return modelcgt_int_addTrainingVector(y,x,model_convertx(xx,x),xobstype,varadd);
}

int SMBOOptions::model_addTrainingVector_cgt(const Vector<gentype> &y, const SparseVector<gentype> &x, const Vector<gentype> &xsidechan, const Vector<int> &xobstype, double varadd)
{
    SparseVector<gentype> xxx(x);

    addinxsidechan(xxx,xsidechan);

    return modelcgt_int_addTrainingVector(y,x,model_convertx(xx,xxx),xobstype,varadd);
}

int SMBOOptions::model_addTrainingVector_sigmaifsep(const gentype &y, const SparseVector<gentype> &x,double varadd)
{
    int ires = 0;

    if ( sigmuseparate )
    {
        ires = modelsigma_int_addTrainingVector(y,model_convertx(xx,x),2,varadd);
    }

    return ires;
}

int SMBOOptions::model_addTrainingVector_mu_sigmaifsame(const gentype &y, const gentype &ypred, const SparseVector<gentype> &x, const Vector<gentype> &xsidechan, int xobstype, double varadd)
{
        int ires = 0;

        locxresunconv.add(locxresunconv.size());
        locxresunconv("&",locxresunconv.size()-1) = x;

        if ( ismodelaug() )
        {
            SparseVector<gentype> xmod;

            // Suppress model to do initial conversion
            int dummy = usemodelaugx;
            usemodelaugx = 0;
            usemodelaugx = dummy;

            ires |= modelaugx_int_addTrainingVector(xsidechan,model_convertx(xmod,x),varadd);
        }

        SparseVector<gentype> xxx(x);

        addinxsidechan(xxx,xsidechan);

        {
            const SparseVector<gentype> &xxxxxx = model_convertx(xx,xxx);

            ires |= modeldiff_int_addTrainingVector(y,ypred,xxxxxx,xobstype,varadd);
            ires |= modelmu_int_addTrainingVector(y,x,xxxxxx,xobstype,varadd);
        }

        return ires;
}






























const Vector<double> &SMBOOptions::model_xcopy(Vector<double> &resx, int i) const
{
        int j,k;

        for ( j = 0 ; j < locires.size() ; ++j )
        {
            if ( i == locires(j) )
            {
                resx.resize(locxres(j).indsize());

                for ( k = 0 ; k < locxres(j).indsize() ; ++k )
                {
                    resx("&",k) = (double) locxres(j).direcref(k);
                }

                return resx;
            }
        }

        //return backconvertx(resx,((*muapprox).x())(i));
        NiceAssert( !isXconvertNonTrivial() );

        const SparseVector<gentype> &tempresx = (getmuapprox(0).x(i));

        resx.resize(tempresx.indsize());

        for ( j = 0 ; j < tempresx.indsize() ; ++j )
        {
            resx("&",j) = (double) tempresx.direcref(j);
        }

        return resx;
}































int SMBOOptions::default_model_setkernelg(const gentype &nv)
{
        int res = 0;
        int lockernnum = 0;

        Vector<gentype> kernRealConstsa(altmuapprox.getKernel().cRealConstants(lockernnum));
        Vector<gentype> kernRealConstsb(altmuapprox_rff.getKernel().cRealConstants(lockernnum));

        if ( kernRealConstsa(0) != nv )
	{
            kernRealConstsa("&",0) = nv;

            altmuapprox.getKernel_unsafe().setRealConstants(kernRealConstsa,lockernnum);
            altmuapprox.resetKernel(0);
	}

        if ( kernRealConstsb(0) != nv )
	{
            kernRealConstsb("&",0) = nv;

            altmuapprox_rff.getKernel_unsafe().setRealConstants(kernRealConstsb,lockernnum);
            altmuapprox_rff.resetKernel(0);
	}

        return res;
}

int SMBOOptions::default_model_setkernelgg(const SparseVector<gentype> &nv)
{
    int res = 0;

    altmuapprox.getKernel_unsafe().setScale(nv);
    altmuapprox.resetKernel(0);

    altmuapprox_rff.getKernel_unsafe().setScale(nv);
    altmuapprox_rff.resetKernel(0);

    return res;
}

int SMBOOptions::default_modelcgt_setkernelg(const gentype &nv)
{
        int res = 0;
        int lockernnum = 0;

        Vector<gentype> kernRealConstsa(altcgtapprox.getKernel().cRealConstants(lockernnum));

        if ( kernRealConstsa(0) != nv )
	{
            kernRealConstsa("&",0) = nv;

            altcgtapprox.getKernel_unsafe().setRealConstants(kernRealConstsa,lockernnum);
            altcgtapprox.resetKernel(0);
	}

        return res;
}

int SMBOOptions::default_modelcgt_setkernelgg(const SparseVector<gentype> &nv)
{
    int res = 0;

    altcgtapprox.getKernel_unsafe().setScale(nv);
    altcgtapprox.resetKernel(0);

    return res;
}

int SMBOOptions::default_modelaugx_setkernelg(const gentype &nv)
{
    int res = 0;
    int lockernnum = 0;

    Vector<gentype> kernRealConstsa(altaugxapprox.getKernel().cRealConstants(lockernnum));

    if ( kernRealConstsa(0) != nv )
    {
        kernRealConstsa("&",0) = nv;

        altaugxapprox.getKernel_unsafe().setRealConstants(kernRealConstsa,lockernnum);
        altaugxapprox.resetKernel(0);
    }

    return res;
}

int SMBOOptions::default_modelaugx_setkernelgg (const SparseVector<gentype> &nv)
{
    int res = 0;

    altaugxapprox.getKernel_unsafe().setScale(nv);
    altaugxapprox.resetKernel(0);

    return res;
}












































int SMBOOptions::modelmu_int_addTrainingVector(const gentype &y, const SparseVector<gentype> &x, const SparseVector<gentype> &xx, int xobstype, double varadd)
{
        locires.add(locires.size()); locires("&",locires.size()-1) = getmuapprox(0).N();
        locxres.add(locxres.size()); locxres("&",locxres.size()-1) = x;
        locyres.add(locyres.size()); locyres("&",locyres.size()-1) = y;
//errstream() << "phantomxyzxyzxyz bayesopt addvec int: " << x << "\n";

        SparseVector<gentype> xxx(xx);
//errstream() << "phantomxyzxyzxyz bayesopt addvec int xx: " << xx << "\n";
//errstream() << "phantomxyzxyzxyz bayesopt addvec int xxx: " << xxx << "\n";
//errstream() << "phantomxyzxyzxyz bayesopt addvec int xtemplate: " << xtemplate << "\n";

        addtemptox(xxx,xtemplate);
//errstream() << "phantomxyzxyzxyz bayesopt addvec int xxx (b): " << xxx << "\n";
//errstream() << "phantomxyzxyzxyz bayesopt addvec int xtemplate (b): " << xtemplate << "\n";

        int i,ires = 0;
        int dval = xobstype;

        if ( ennornaive )
        {
            SparseVector<gentype> tempx;

            NiceAssert( muapprox.size() == y.size() );

            for ( i = 0 ; i < muapprox.size() ; ++i )
            {
                int locdval = ( getmuapprox(i).type() == 212 ) ? 2 : dval;

                if ( y.isValVector() )
                {
                    const gentype &yi = ((const Vector<gentype> &) y)(i);

                    ires |= getmuapprox("&",i).addTrainingVector(getmuapprox(i).N(),yi,convnearuptonaive(tempx,xxx),1.0,1.0,locdval);

                    if ( varadd )
                    {
                        ires |= getmuapprox("&",i).setsigmaweight(getmuapprox(i).N()-1,1+(varadd/(getmuapprox(i).sigma())));
                    }

                    if ( dval != locdval )
                    {
                        int imu = getmuapprox(i).N()-1;
                        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
                        int ddval = ( dval == -2 ) ? 2 : dval;

                        ires |= getmuapprox("&",i).setd(iimu,ddval);
                    }
                }

                else
                {
                    ires |= getmuapprox("&",i).addTrainingVector(getmuapprox(i).N(),y,convnearuptonaive(tempx,xxx),1.0,1.0,locdval);

                    if ( varadd )
                    {
                        ires |= getmuapprox("&",i).setsigmaweight(getmuapprox(i).N()-1,1+(varadd/(getmuapprox(i).sigma())));
                    }

                    if ( dval != locdval )
                    {
                        int imu = getmuapprox(i).N()-1;
                        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
                        int ddval = ( dval == -2 ) ? 2 : dval;

                        ires |= getmuapprox("&",i).setd(iimu,ddval);
                    }
                }
            }
        }

        else
        {
//errstream() << "phantomxyz muapprox = " << muapprox << "\n";
//errstream() << "phantomxyz y = " << y << "\n";
//errstream() << "phantomxyz xxx = " << xxx << "\n";
            NiceAssert( muapprox.size() == y.size() );

            for ( i = 0 ; i < muapprox.size() ; ++i )
            {
                int locdval = ( getmuapprox(i).type() == 212 ) ? 2 : dval;

                if ( y.isValVector() )
                {
                    const gentype &yi = ((const Vector<gentype> &) y)(i);

                    ires |= getmuapprox("&",i).addTrainingVector(getmuapprox(i).N(),yi,xxx,1.0,1.0,locdval);

                    if ( varadd )
                    {
                        ires |= getmuapprox("&",i).setsigmaweight(getmuapprox(i).N()-1,1+(varadd/(getmuapprox(i).sigma())));
                    }

                    if ( dval != locdval )
                    {
                        int imu = getmuapprox(i).N()-1;
                        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
                        int ddval = ( dval == -2 ) ? 2 : dval;

                        ires |= getmuapprox("&",i).setd(iimu,ddval);
                    }
                }

                else
                {
                    ires |= getmuapprox("&",i).qaddTrainingVector(getmuapprox(i).N(),y,xxx,1.0,1.0,locdval);

                    if ( varadd )
                    {
                        ires |= getmuapprox("&",i).setsigmaweight(getmuapprox(i).N()-1,1+(varadd/(getmuapprox(i).sigma())));
                    }

                    if ( dval != locdval )
                    {
                        int imu = getmuapprox(i).N()-1;
                        int iimu = ( ( getmuapprox(i).type() == 212 ) ? (-imu-1) : imu );
                        int ddval = ( dval == -2 ) ? 2 : dval;

                        ires |= getmuapprox("&",i).setd(iimu,ddval);
                    }
                }
            }
        }

/*
        if ( xobstype != 2 )
        {
            for ( i = 0 ; i < muapprox.size() ; ++i )
            {
                int imu = (*(muapprox(i))).N()-1;
                int iimu = ( ( (*(muapprox(i))).type() == 212 ) ? (-imu-1) : imu );

                ires |= (*(muapprox("&",i))).setd(iimu,xobstype);
            }
        }
*/

        for ( i = 0 ; i < muapprox.size() ; ++i )
        {
            if ( modelrff && ( modelrff != getmuapprox(i).NRff() ) )
            {
                ires |= getmuapprox("&",i).setNRff(modelrff); // needs to be delayed until there are training vectors or RFF will have dim 0, which is wrong)
            }
        }

        return ires;
}

int SMBOOptions::modelcgt_int_addTrainingVector(const Vector<gentype> &y, const SparseVector<gentype> &x, const SparseVector<gentype> &xx, const Vector<int> &xobstype, double varadd)
{
    (void) x;

    SparseVector<gentype> xxx(xx);

    addtemptox(xxx,xtemplate);

    int i,ires = 0;

    if ( ennornaive )
    {
        SparseVector<gentype> tempx;

        NiceAssert( cgtapprox.size() == y.size() );

        for ( i = 0 ; i < cgtapprox.size() ; ++i )
        {
            gentype yval = y(i).isValNull() ? 0.0_gent : y(i);

            {
                ires |= getcgtapprox("&",i).addTrainingVector(getcgtapprox(i).N(),yval,convnearuptonaive(tempx,xxx));

                if ( varadd )
                {
                    ires |= getcgtapprox("&",i).setsigmaweight(getcgtapprox(i).N()-1,1+(varadd/(getcgtapprox(i).sigma())));
                }

                int dval = xobstype.size() ? xobstype(i) : 2;

                if ( dval != 2 )
                {
                    int imu = getcgtapprox(i).N()-1;
                    int iimu = ( ( getcgtapprox(i).type() == 212 ) ? (-imu-1) : imu );
                    int ddval = ( dval == -2 ) ? 2 : dval;

                    ires |= getcgtapprox("&",i).setd(iimu,ddval);
                }
            }
        }
    }

    else
    {
        NiceAssert( cgtapprox.size() == y.size() );

        for ( i = 0 ; i < cgtapprox.size() ; ++i )
        {
            gentype yval = y(i).isValNull() ? 0.0_gent : y(i);

            //if ( !y(i).isValNull() )
            {
                ires |= getcgtapprox("&",i).addTrainingVector(getcgtapprox(i).N(),yval,xxx);

                if ( varadd )
                {
                    ires |= getcgtapprox("&",i).setsigmaweight(getcgtapprox(i).N()-1,1+(varadd/(getcgtapprox(i).sigma())));
                }

                int dval = xobstype.size() ? xobstype(i) : 2;

                if ( dval != 2 )
                {
                    int imu = getcgtapprox(i).N()-1;
                    int iimu = ( ( getcgtapprox(i).type() == 212 ) ? (-imu-1) : imu );
                    int ddval = ( dval == -2 ) ? 2 : dval;

                    ires |= getcgtapprox("&",i).setd(iimu,ddval);
                }
            }
        }
    }

    return ires;
}

int SMBOOptions::modelsigma_int_addTrainingVector(const gentype &y, const SparseVector<gentype> &xx, int xobstype, double varadd)
{
    NiceAssert( sigmuseparate );

    SparseVector<gentype> xxx(xx);

    addtemptox(xxx,xtemplate);

    SparseVector<gentype> tempx;

    int ires = getsigmaapprox("&").addTrainingVector(getsigmaapprox().N(),y,convnearuptonaive(tempx,xxx));

    if ( varadd )
    {
        ires |= getsigmaapprox("&").setsigmaweight(getsigmaapprox().N()-1,1+(varadd/(getsigmaapprox().sigma())));
    }

    if ( xobstype != 2 )
    {
        ires |= getsigmaapprox("&").setd(getsigmaapprox().N()-1,xobstype);
    }

    return ires;
}

int SMBOOptions::modeldiff_int_addTrainingVector(const gentype &y, const gentype &ypred, const SparseVector<gentype> &xx, int xobstype, double varadd)
{
        (void) ypred;
        (void) xobstype;

        int ires = 0;

        if ( ( tranmeth == 1 ) && Nbasemu )
        {
            NiceAssert( xobstype == 2 );

            if ( firsttrain && srcmodel && tunesrcmod )
            {
//outstream() << "Tuning source model\n";
                (*srcmodel).getML().tuneKernel(tunesrcmod,getxwidth(),1,0,nullptr);
                firsttrain = 0;
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            xx.makealtcontent();

            SparseVector<gentype> tempx;

            (*srcmodel).gg(predval,convnearuptonaive(tempx,xx));

            resdiff =  y;
            resdiff -= predval;

//outstream() << "beta := " << beta << " + ( " << y << " - " << predval << " )^2 = ";
            beta += ((double) norm2(resdiff))/2.0;
//outstream() << beta << "\n";
        }

        if ( ( tranmeth == 2 ) && Nbasemu )
        {
            NiceAssert( xobstype == 2 );

            if ( firsttrain && srcmodel && tunesrcmod )
            {
                (*srcmodel).getML().tuneKernel(tunesrcmod,getxwidth(),1,0,nullptr);
                firsttrain = 0;
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            // Predict y based on target (and noise)

            xx.makealtcontent();

            SparseVector<gentype> tempx;

            //(*srcmodel).gg(predval,xx);
            (*srcmodel).var(storevar,predval,convnearuptonaive(tempx,xx));

            // Calculate difference between reality and model

            diffval =  y;
            diffval -= predval;

            // How noisy is the difference?

            double sigmaval;

            sigmaval =  getmuapprox(0).sigma(); // variance of observation
            sigmaval += (double) storevar; // variance from source model

            // Add to difference model

            double sigmaweight = (sigmaval+varadd)/((*diffmodel).sigma());
            double Cweight = 1/sigmaweight;

            SparseVector<gentype> xxx(xx);

            addtemptox(xxx,xtemplate);

            SparseVector<gentype> temmpx;

            ires |= (*diffmodel).addTrainingVector((*diffmodel).N(),diffval,convnearuptonaive(tempx,xxx),Cweight);

//            if ( dval != 2 )
//            {
//                ires |= (*diffmodel).setd((*diffmodel).N()-1,dval);
//            }
        }

        return ires;
}

int SMBOOptions::modeldiff_int_train(int &res, svmvolatile int &killSwitch)
{
        int ires = 0;

        if ( ( tranmeth == 1 ) && Nbasemu )
        {
            NiceAssert( muapprox.size() == 1 );

            if ( firsttrain && srcmodel && tunesrcmod )
            {
                (*srcmodel).getML().tuneKernel(tunesrcmod,getxwidth(),1,0,nullptr);
                firsttrain = 0;
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            int Nmodel = getmuapprox(0).N();

            alpha = alpha0 + ((Nmodel-Nbasemu)/2.0);
            //beta updated incrementally

            getmuapprox("&",0).setsigmaweight(indpremu,( presigweightmu = beta/(alpha+1) ));
//outstream() << "sigma = (" << beta << "/(" << alpha << "+1)) = " << beta/(alpha+1) << "\n";
        }

        if ( ( tranmeth == 2 ) && Nbasemu )
        {
            NiceAssert( muapprox.size() == 1 );

            if ( firsttrain && srcmodel && tunesrcmod )
            {
                (*srcmodel).getML().tuneKernel(tunesrcmod,getxwidth(),1,0,nullptr);
                firsttrain = 0;
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            int i;

            // Train difference model

            ires = (*diffmodel).train(res,killSwitch);

            // Update y and sigma in muapprox

            for ( i = 0 ; i < Nbasemu ; ++i )
            {
                // Calc predicted difference and difference varian

                (*diffmodel).var(storevar,diffval,(*srcmodel).x(i));

                // Calculate bias corrected source y

                predval =  (*srcmodel).y()(i);
                predval += diffval;

                // Calculate bias corrected source y variance

                double sigmaval;

                sigmaval =  ((*srcmodel).sigma())*((*srcmodel).sigmaweight()(i));
                sigmaval += (double) storevar;

                // Set bias corrected source y and variance in muapprox

                getmuapprox("&",0).sety(i,predval);
                getmuapprox("&",0).setsigmaweight(i,(sigmaval/(getmuapprox(0).sigma())));
            }
        }

        return ires;
}



























GlobalOptions &SMBOOptions::getModelErrOptim(GlobalOptions *src) const
{
    if ( src )
    {
        killModelErrOptim();
    }

    if ( !modelErrOptim )
    {
        if ( !src )
        {
            (ismodelErrLocal) = 1;

            MEMNEW(modelErrOptim,DIRectOptions);

            NiceAssert( modelErrOptim );

            *(modelErrOptim) = static_cast<const GlobalOptions &>(*this);
        }

        else
        {
            ismodelErrLocal = 0;
            modelErrOptim   = src;
        }
    }

    return *(modelErrOptim);
}

void SMBOOptions::killModelErrOptim(void) const
{
    if ( modelErrOptim )
    {
        if ( ismodelErrLocal )
        {
            MEMDEL(modelErrOptim);
        }

        modelErrOptim   = nullptr;
        ismodelErrLocal = 1;
    }
}





















































int SMBOOptions::modelaugx_int_addTrainingVector(const Vector<gentype> &y, const SparseVector<gentype> &xx, double varadd)
{
    int ires = 0;
    int i;

    NiceAssert( y.size() == augxapprox.size() );

    for ( i = 0 ; i < y.size() ; ++i )
    {
        ires |= getaugxapprox("&",i).addTrainingVector(getaugxapprox(i).N(),y(i),xx,1,getaugxapprox(i).eps()+varadd);
    }

    return ires;
}

int SMBOOptions::modelaugx_int_train (int &res, svmvolatile int &killSwitch)
{
    int ires = 0;
    int i;

    for ( i = 0 ; i < augxapprox.size() ; ++i )
    {
        ires |= getaugxapprox("&",i).train(res,killSwitch);
    }

    return ires;
}

void SMBOOptions::addinxsidechan(SparseVector<gentype> &x, const Vector<gentype> &xsidechan)
{
        if ( xsidechan.size() )
        {
            int i,j;

            for ( i = 0 ; i < xsidechan.size() ; ++i )
            {
                if ( !(xsidechan(i).isValVector()) )
                {
                    x.n("&",0,i+1) = xsidechan(i);
                }

                else
                {
                    int jdim = xsidechan(i).size();

                    for ( j = 0 ; j < jdim ; ++j )
                    {
                        x.n("&",j,i+1) = xsidechan(i)(j);
                    }
                }
            }
        }

//        if ( xaddranksidechan.size() )
//        {
//            int i,j;
//
//            for ( i = 0 ; i < xaddranksidechan.size() ; ++i )
//            {
//                if ( !(xaddranksidechan(i).isValVector()) )
//                {
//                    x.f1("&",0,i+1) = xaddranksidechan(i);
//                }
//
//                else
//                {
//                    int jdim = xaddranksidechan(i).size();
//
//                    for ( j = 0 ; j < jdim ; ++j )
//                    {
//                        x.f1("&",j,i+1) = xaddranksidechan(i)(j);
//                    }
//                }
//            }
//        }

        return;
}


































































double calcLCB(int dim, const double *x, void *arg);
double calcUCB(int dim, const double *x, void *arg);

double calcLCB(int dim, const double *x, void *arg)
{
    SMBOOptions &caller       = *((SMBOOptions *)           (((void **) arg)[0]));
    SparseVector<gentype> &xx = *((SparseVector<gentype> *) (((void **) arg)[1]));
    gentype &mu               = *((gentype *)               (((void **) arg)[2]));
    gentype &sigmasq          = *((gentype *)               (((void **) arg)[3]));

    int i;

    for ( i = 0 ; i < dim ; ++i )
    {
        xx("&",i) = x[i];
    }

    caller.model_muvar(sigmasq,mu,xx);

    return -((double) mu) - sqrt((double) sigmasq);
}

double calcUCB(int dim, const double *x, void *arg)
{
    SMBOOptions &caller       = *((SMBOOptions *)           (((void **) arg)[0]));
    SparseVector<gentype> &xx = *((SparseVector<gentype> *) (((void **) arg)[1]));
    gentype &mu               = *((gentype *)               (((void **) arg)[2]));
    gentype &sigmasq          = *((gentype *)               (((void **) arg)[3]));

    int i;

    for ( i = 0 ; i < dim ; ++i )
    {
        xx("&",i) = x[i];
    }

    caller.model_muvar(sigmasq,mu,xx);

    return -((double) mu) + sqrt((double) sigmasq);
}

void altcalcLCB(gentype &res, Vector<gentype> &x, void *arg);
void altcalcUCB(gentype &res, Vector<gentype> &x, void *arg);

void altcalcLCB(gentype &res, Vector<gentype> &x, void *arg)
{
    SMBOOptions &caller       = *((SMBOOptions *)           (((void **) arg)[0]));
    SparseVector<gentype> &xx = *((SparseVector<gentype> *) (((void **) arg)[1]));
    gentype &mu               = *((gentype *)               (((void **) arg)[2]));
    gentype &sigmasq          = *((gentype *)               (((void **) arg)[3]));

    int i;
    int dim = x.size();

    for ( i = 0 ; i < dim ; ++i )
    {
        xx("&",i) = (double) x(i);
    }

    caller.model_muvar(sigmasq,mu,xx);

    res.force_double() = -((double) mu) - sqrt((double) sigmasq);

    return;
}

void altcalcUCB(gentype &res, Vector<gentype> &x, void *arg)
{
    SMBOOptions &caller       = *((SMBOOptions *)           (((void **) arg)[0]));
    SparseVector<gentype> &xx = *((SparseVector<gentype> *) (((void **) arg)[1]));
    gentype &mu               = *((gentype *)               (((void **) arg)[2]));
    gentype &sigmasq          = *((gentype *)               (((void **) arg)[3]));

    int i;
    int dim = x.size();

    for ( i = 0 ; i < dim ; ++i )
    {
        xx("&",i) = (double) x(i);
    }

    caller.model_muvar(sigmasq,mu,xx);

    res.force_double() = -((double) mu) + sqrt((double) sigmasq);

    return;
}

double SMBOOptions::model_err(int dim, const Vector<double> &xmin, const Vector<double> &xmax, svmvolatile int &killSwitch)
{
    SparseVector<double> xx;
    gentype mu;
    gentype sigmasq;

    void *modelarg[4];

    modelarg[0] = (void *) this;
    modelarg[1] = (void *) &xx;
    modelarg[2] = (void *) &mu;
    modelarg[3] = (void *) &sigmasq;

    gentype minLCB(0.0); // min_x mu(x) - sigma(x)
    gentype minUCB(0.0); // min_x mu(x) + sigma(x)

    Vector<gentype> xdummy;
    int idummy;
    Vector<Vector<gentype> > allxdummy;
    Vector<gentype> allfdummy;
    Vector<Vector<gentype> > allcdummy;
    Vector<gentype> allmdummy;
    Vector<gentype> allsupdummy;
    Vector<double> allsdummy;
    Vector<gentype> altxmin;
    Vector<gentype> altxmax;

    altxmin.castassign(xmin);
    altxmax.castassign(xmax);

//    DIRectOptions &dopts = getModelErrOptim();
//    static_cast<GlobalOptions &>(dopts) = static_cast<const GlobalOptions &>(*this);
//
//    int dresa = directOpt(dim,xdummy,minLCB,xmin,xmax,calcLCB,(void *) modelarg,dopts,killSwitch);
//    int dresb = directOpt(dim,xdummy,minUCB,xmin,xmax,calcUCB,(void *) modelarg,dopts,killSwitch);

    GlobalOptions &dopts = getModelErrOptim();

    int dresa = dopts.optim(dim,xdummy,minLCB,idummy,allxdummy,allfdummy,allcdummy,allmdummy,allsupdummy,allsdummy,altxmin,altxmax,altcalcLCB,modelarg,killSwitch);
    int dresb = dopts.optim(dim,xdummy,minUCB,idummy,allxdummy,allfdummy,allcdummy,allmdummy,allsupdummy,allsdummy,altxmin,altxmax,altcalcUCB,modelarg,killSwitch);

    std::stringstream resbuffer;
    resbuffer << "Model error codes = " << dresa << "," << dresb << "  ";
    nullPrint(errstream(),resbuffer.str());
    //errstream() << "Model error calculation using DIRect: " << dresa << "," << dresb << "\n";

    return ((double) minUCB)-((double) minLCB);
}



const SparseVector<gentype> &SMBOOptions::convnearuptonaive(SparseVector<gentype> &res, const SparseVector<gentype> &x) const
{
    NiceAssert ( !getdimfid() || !ennornaive );

    if ( !ennornaive || ( ( x.nupsize() <= 1 ) && ( x.f1upsize() <= 1 ) ) )
    {
        return x; // no point messing around
    }

    res.zero();

    int u,i,n,m;

    n = x.nupsize();

    if ( n )
    {
        for ( u = 0 ; u < n ; ++u )
        {
            m = x.nupsize(u);

            for ( i = 0 ; i < m ; ++i )
            {
                res.n("&",i+(u*ennornaive),0) = x.n(i,u);
            }
        }
    }

    n = x.f1upsize();

    if ( n )
    {
        for ( u = 0 ; u < n ; ++u )
        {
            m = x.f1upsize(u);

            for ( i = 0 ; i < m ; ++i )
            {
                res.f1("&",i+(u*ennornaive),0) = x.f1(i,u);
            }
        }
    }

    n = x.f2upsize();

    if ( n )
    {
        for ( u = 0 ; u < n ; ++u )
        {
            m = x.f2upsize(u);

            for ( i = 0 ; i < m ; ++i )
            {
                res.f2("&",i+(u*ennornaive),0) = x.f2(i,u);
            }
        }
    }

    n = x.f4upsize();

    if ( n )
    {
        for ( u = 0 ; u < n ; ++u )
        {
            m = x.f2upsize(u);

            for ( i = 0 ; i < m ; ++i )
            {
                res.f4("&",i+(u*ennornaive),0) = x.f4(i,u);
            }
        }
    }

    return res;
}

