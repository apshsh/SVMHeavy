
//
// Fuzzy weight selection for MLs
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "plotml.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "imp_generic.hpp"
#include "plotbase.hpp"
#include "blk_conect.hpp"
#include <sstream>


#define NUMSAMP 1000
//#define NUMSAMPSURF 1000
#define NUMSAMPSURF 400
//#define STDDEVSCALE 2.0
#define STDDEVSCALE 1.0
// 1 standard deviation seems like the standard here https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html

// Plot a simple line baseline(var), where var is set by xusevar (default x)

int plotfn2d(double xmin, double xmax, double omin, double omax,
             const std::string &fname, const std::string &dname, int outformat, const gentype &baseline,
             int xusevar)
{
    NiceAssert( xmin < xmax );

    int xindi = ( xusevar <= 5 ) ? xusevar : 0;
    int xindj = ( xusevar == 6 ) ? 2 : 0;

    SparseVector<SparseVector<gentype> > evalargs;
    gentype &x = evalargs("&",xindj)("&",xindi);

    // Create the mean/variance datafile to plot

    std::ofstream dnamefile(dname);

    for ( x = xmin ; (double) x <= xmax ; x += (xmax-xmin)/NUMSAMP )
    {
        gentype y = baseline(evalargs);

        if ( !testisvnan(y) && y.isCastableToRealWithoutLoss() )
        {
            dnamefile << x << "\t" << ((double) y) << "\t0\t0\n";
        }
    }

    dnamefile.close();

    return doplot(xmin,xmax,omin,omax,fname,dname,outformat,0,0);
}



// Plot a simple line baseline(var), where var is set by xusevar (default x)

int surffn(double xmin, double xmax, double ymin, double ymax, double omin, double omax,
             const std::string &fname, const std::string &dname, int outformat, const gentype &baseline,
             int xusevar, int yusevar)
{
    NiceAssert( xmin < xmax );

    int xindi = ( xusevar <= 5 ) ? xusevar : 0;
    int xindj = ( xusevar == 6 ) ? 2 : 0;

    int yindi = ( yusevar <= 5 ) ? yusevar : 0;
    int yindj = ( yusevar == 6 ) ? 2 : 0;

    SparseVector<SparseVector<gentype> > evalargs;
    gentype &x = evalargs("&",xindj)("&",xindi);
    gentype &y = evalargs("&",yindj)("&",yindi);

    // Create the mean/variance datafile to plot

    std::ofstream dnamefile(dname);

    for ( x = xmin ; (double) x <= xmax ; x += (xmax-xmin)/NUMSAMPSURF )
    {
        for ( y = ymin ; (double) y <= ymax ; y += (ymax-ymin)/NUMSAMPSURF )
        {
            gentype z = baseline(evalargs);

            if ( !testisvnan(z) && z.isCastableToRealWithoutLoss() )
            {
                dnamefile << x << "\t" << y << "\t" << ((double) z) << "\n";
            }
        }
    }

    dnamefile.close();

    return dosurf(xmin,xmax,ymin,ymax,omin,omax,fname,dname,outformat);
}



// Do a 2-d plot visualisation of ml

int plotml(const ML_Base &ml, int xindex,
           double xmin, double xmax, double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, int incdata, const gentype &baseline, int incvar, int xusevar,
           const SparseVector<gentype> &xtemplate, int plotsq, int plotimp, double scale)
{
    NiceAssert( xmin < xmax );

    int i;

    SparseVector<gentype> x(xtemplate);
    gentype ovar;
    gentype o;

    int xindi = ( xusevar <= 5 ) ? xusevar : 0;
    int xindj = ( xusevar == 6 ) ? 2 : 0;

    SparseVector<SparseVector<gentype> > evalargs;
    gentype &xbase = evalargs("&",xindj)("&",xindi);

    // Create the mean/variance datafile to plot

    std::ofstream dnamefile(dname);
    gentype dummy,dummyb;

    dummyb.force_null(); // This means "no variance on x" when calling imp

    for ( x("&",xindex) = xmin, xbase = xmin ; (double) x(xindex) <= xmax ; x("&",xindex) += (xmax-xmin)/NUMSAMP, xbase = x(xindex) )
    {
        if ( !plotimp && incvar && ( incdata & 2 ) )
        {
            ml.var(ovar,o,x);

            o *= scale;

            dnamefile << x(xindex) << "\t" << o << "\t" << (STDDEVSCALE*sqrt((double) ovar)) << "\t" << baseline(evalargs) << "\n";
        }

        else if ( !plotimp && !incvar && ( incdata & 2 ) )
        {
            //ml.gg(o,x);
            ml.gh(dummy,o,x,( plotsq ? 2 : 0 ));

            o *= scale;

            dnamefile << x(xindex) << "\t" << o << "\t0\t" << baseline(evalargs) << "\n";
        }

        else if ( plotimp && ( incdata & 2 ) )
        {
            (dynamic_cast<const IMP_Generic &>(ml)).imp(o,dummy,x,dummyb);

            dnamefile << x(xindex) << "\t" << o << "\t0\t" << baseline(evalargs) << "\n";
        }

        else if ( !plotimp && incvar && !( incdata & 2 ) )
        {
            ml.var(ovar,o,x);

            o *= scale;

            dnamefile << x(xindex) << "\t" << o << "\t" << (STDDEVSCALE*sqrt((double) ovar)) << "\t0\n";
        }

        else if ( !plotimp && !incvar && !( incdata & 2 ) )
        {
            //ml.gg(o,x);
            ml.gh(dummy,o,x,( plotsq ? 2 : 0 ));

            o *= scale;

            dnamefile << x(xindex) << "\t" << o << "0\t0\n";
        }

        else if ( plotimp && !( incdata & 2 ) )
        {
            (dynamic_cast<const IMP_Generic &>(ml)).imp(o,dummy,x,dummyb);

            dnamefile << x(xindex) << "\t" << o << "0\t0\n";
        }
    }

    dnamefile.close();

    // If we are using them, construct the training data files

    std::string dnamepos = dname+"_pos";
    std::string dnameneg = dname+"_neg";
    std::string dnameequ = dname+"_equ";

    if ( ( incdata & 1 ) )
    {
        std::ofstream dnameposfile(dnamepos);
        std::ofstream dnamenegfile(dnameneg);
        std::ofstream dnameequfile(dnameequ);

        for ( i = 0 ; i < ml.N() ; ++i )
        {
            if ( (ml.d())(i) == +1 )
            {
                //dnameposfile << (ml.y())(i) << "\t" << (ml.x(i))(xindex) << "\n";
                dnameposfile << (ml.x(i))(xindex) << "\t" << (-(ml.y())(i)) << "\n";
            }

            if ( (ml.d())(i) == -1 )
            {
                dnamenegfile << (ml.x(i))(xindex) << "\t" << (-(ml.y())(i)) << "\n";
            }

            if ( (ml.d())(i) == 2 )
            {
                dnameequfile << (ml.x(i))(xindex) << "\t" << (-(ml.y())(i)) << "\n";
            }
        }

        dnameposfile.close();
        dnamenegfile.close();
        dnameequfile.close();
    }

    int ires = doplot(xmin,xmax,omin,omax,fname,dname,outformat,incdata,incvar);

    if ( ( ml.type() == 212 ) && ( dynamic_cast<const BLK_Conect &>(ml).mlqlist().size() ) )
    {
        int numsubs = dynamic_cast<const BLK_Conect &>(ml).mlqlist().size();
        int mlqmode = dynamic_cast<const BLK_Conect &>(ml).getmlqmode();

        for ( int ii = 0 ; ii < numsubs ; ii++ )
        {
            const ML_Base &ml_sub = *(dynamic_cast<const BLK_Conect &>(ml).mlqlist()(ii));

            int incdata_sub = ( ( ii == numsubs-1 ) || ( mlqmode == 0 ) ) ? incdata : ( incdata & 0x0FD );
            int incvar_sub  = ( ( ii == numsubs-1 ) || ( mlqmode == 0 ) ) ? incvar  : 0;

            std::string fname_sub = fname+"_subplot_"+std::to_string(ii);
            std::string dname_sub = dname+"_subplot_"+std::to_string(ii);

            ires |= plotml(ml_sub,xindex,xmin,xmax,omin,omax,fname_sub,dname_sub,outformat,incdata_sub,baseline,incvar_sub,xusevar,xtemplate,plotsq,plotimp,scale);
        }
    }

    return ires;
}

// Do a surface plot visualisation of ml

int plotml(const ML_Base &ml, int xindex, int yindex,
           double xmin, double xmax, double ymin, double ymax, double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, int incdata, const gentype &baseline, int incvar, int xusevar, int yusevar,
           const SparseVector<gentype> &xtemplate, int plotsq, int plotimp, double scale)
{
    NiceAssert( xmin < xmax );
    NiceAssert( ymin < ymax );

    bool ominmaxIsPreset = ( omin < omax ) ? true : false;

    bool firstcheck = true;

    SparseVector<gentype> x(xtemplate);
    gentype o,ovar;

    //double vmin = 0;
    //double vmax = 0;

    int xindi = ( xusevar <= 5 ) ? xusevar : 0;
    int xindj = ( xusevar == 6 ) ? 2 : 0;

    int yindi = ( yusevar <= 5 ) ? yusevar : 0;
    int yindj = ( yusevar == 6 ) ? 2 : 0;

    SparseVector<SparseVector<gentype> > evalargs;
    gentype &xbase = evalargs("&",xindj)("&",xindi);
    gentype &ybase = evalargs("&",yindj)("&",yindi);

    // Create the mean/variance datafile to plot

    std::string dvname(dname); dvname += "_var"; // variance goes here
    std::string dbname(dname); dbname += "_blf"; // baseline goes here

    std::string fvname(fname); fvname += "_var"; // variance goes here
    std::string fbname(fname); fbname += "_blf"; // baseline goes here

    std::ofstream dnamefile(dname);
    std::ofstream dvnamefile(dvname);
    std::ofstream dbnamefile(dbname);

    gentype dummy,dummyb;

    dummyb.force_null();

    for ( x("&",xindex) = xmin, xbase = xmin ; (double) x(xindex) <= xmax ; x("&",xindex) += (xmax-xmin)/GRIDSAMP, xbase = x(xindex) )
    {
        for ( x("&",yindex) = ymin, ybase = ymin ; (double) x(yindex) <= ymax ; x("&",yindex) += (ymax-ymin)/GRIDSAMP, ybase = x(yindex) )
        {
            if ( !plotimp && incvar && ( incdata & 2 ) )
            {
                ml.var(ovar,o,x);

                o *= scale;

//                dnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\n";
                dvnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\t" << (STDDEVSCALE*sqrt(ovar)) << "\n";
                dbnamefile << x(xindex) << "\t" << x(yindex) << "\t" << baseline(evalargs) << "\n";
            }

            else if ( !plotimp && !incvar && ( incdata & 2 ) )
            {
                ml.gh(dummy,o,x,( plotsq ? 2 : 0 ));

                o *= scale;

                dnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\n";
                dbnamefile << x(xindex) << "\t" << x(yindex) << "\t" << baseline(evalargs) << "\n";
            }

            else if ( plotimp && ( incdata & 2 ) )
            {
                (dynamic_cast<const IMP_Generic &>(ml)).imp(o,dummy,x,dummyb);

                dnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\n";
                dbnamefile << x(xindex) << "\t" << x(yindex) << "\t" << baseline(evalargs) << "\n";
            }

            else if ( !plotimp && incvar && !( incdata & 2 ) )
            {
                ml.var(ovar,o,x);

                o *= scale;

//                dnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\n";
                dvnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\t" << (STDDEVSCALE*sqrt(ovar)) << "\n";
            }

            else if ( !plotimp && !incvar && !( incdata & 2 ) )
            {
                ml.gh(dummy,o,x,( plotsq ? 2 : 0 ));

                o *= scale;

                dnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\n";
            }

            else if ( plotimp && !( incdata & 2 ) )
            {
                (dynamic_cast<const IMP_Generic &>(ml)).imp(o,dummy,x,dummyb);

                dnamefile << x(xindex) << "\t" << x(yindex) << "\t" << o << "\n";
            }

            if ( !o.isValNull() && !ominmaxIsPreset )
            {
                double oo = (double) o;
                //double oovar = (double) ovar;

                if ( firstcheck )
                {
                    omin = ( omax = oo );
                    //vmax = sqrt(oovar);

                    firstcheck = false;
                }

                else
                {
                    omin = ( omin < oo ) ? omin : oo;
                    omax = ( omax > oo ) ? omax : oo;

                    //vmax = ( vmax > sqrt(oovar) ) ? omax : sqrt(oovar);
                }
            }
        }
    }

    dnamefile.close();
    dvnamefile.close();
    dbnamefile.close();

    // If we are using them, construct the training data files

    std::string dnamepos = dname+"_pos";
    std::string dnameneg = dname+"_neg";
    std::string dnameequ = dname+"_equ";

    if ( ( incdata & 1 ) )
    {
        std::ofstream dnameposfile(dnamepos);
        std::ofstream dnamenegfile(dnameneg);
        std::ofstream dnameequfile(dnameequ);

        for ( int i = 0 ; i < ml.N() ; ++i )
        {
            if ( (ml.d())(i) == +1 )
            {
                //dnameposfile << (ml.y())(i) << "\t" << (ml.x(i))(xindex) << "\n";
                dnameposfile << (ml.x(i))(xindex) << "\t" << (ml.x(i))(yindex) << "\t" << (-(ml.y())(i)) << "\n";
            }

            if ( (ml.d())(i) == -1 )
            {
                dnamenegfile << (ml.x(i))(xindex) << "\t" << (ml.x(i))(yindex) << "\t" << (-(ml.y())(i)) << "\n";
            }

            if ( (ml.d())(i) == 2 )
            {
                dnameequfile << (ml.x(i))(xindex) << "\t" << (ml.x(i))(yindex) << "\t" << (-(ml.y())(i)) << "\n";
            }
        }

        dnameposfile.close();
        dnamenegfile.close();
        dnameequfile.close();
    }

    int ires = 0;

    if ( incvar )
    {
        ires = dosurfvar(xmin,xmax,ymin,ymax,omin,omax,fname,dvname,outformat);
        //ires = dosurf(xmin,xmax,ymin,ymax,vmin,vmax,fvname,dvname,outformat);
    }

    else
    {
        ires = dosurf(xmin,xmax,ymin,ymax,omin,omax,fname,dname,outformat);
    }

    if ( incdata & 2 )
    {
        ires |= dosurf(xmin,xmax,ymin,ymax,omin,omax,fbname,dbname,outformat);
    }

    return ires;
}


// Do a simple line-plot with optional variance, baseline and datapoints

int plot2d(const Vector<double> &x, const Vector<double> &y, const Vector<double> &yvar, const Vector<double> &ybaseline,
           const Vector<double> &xpos, const Vector<double> &ypos,
           const Vector<double> &xneg, const Vector<double> &yneg,
           const Vector<double> &xequ, const Vector<double> &yequ,
           double xmin, double xmax, double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, int incdata, int incvar)
{
    NiceAssert( x.size() == y.size() );
    NiceAssert( x.size() == yvar.size() );
    NiceAssert( x.size() == ybaseline.size() );

    NiceAssert( xpos.size() == ypos.size() );
    NiceAssert( xneg.size() == yneg.size() );
    NiceAssert( xequ.size() == yequ.size() );

    int i;

    // Create the mean/variance datafile to plot

    std::ofstream dnamefile(dname);

    for ( i = 0 ; i < x.size() ; ++i )
    {
        if ( incvar && ( incdata & 2 ) )
        {
            dnamefile << x(i) << "\t" << y(i) << "\t" << (STDDEVSCALE*sqrt(yvar(i))) << "\t" << ybaseline(i) << "\n";
        }

        else if ( !incvar && ( incdata & 2 ) )
        {
            dnamefile << x(i) << "\t" << y(i) << "\t0\t" << ybaseline(i) << "\n";
        }

        else if ( incvar && !( incdata & 2 ) )
        {
            dnamefile << x(i) << "\t" << y(i) << "\t" << (STDDEVSCALE*sqrt(yvar(i))) << "\t0\n";
        }

        else if ( !incvar && !( incdata & 2 ) )
        {
            dnamefile << x(i) << "\t" << y(i) << "0\t0\n";
        }
    }

    dnamefile.close();

    // If we are using them, construct the training data files

    std::string dnamepos = dname+"_pos";
    std::string dnameneg = dname+"_neg";
    std::string dnameequ = dname+"_equ";

    if ( ( incdata & 1 ) )
    {
        std::ofstream dnameposfile(dnamepos);
        std::ofstream dnamenegfile(dnameneg);
        std::ofstream dnameequfile(dnameequ);

        for ( i = 0 ; i < xpos.size() ; ++i )
        {
            dnameposfile << xpos(i) << "\t" << ypos(i) << "\n";
        }

        for ( i = 0 ; i < xneg.size() ; ++i )
        {
            dnamenegfile << xneg(i) << "\t" << yneg(i) << "\n";
        }

        for ( i = 0 ; i < xequ.size() ; ++i )
        {
            dnameequfile << xequ(i) << "\t" << yequ(i) << "\n";
        }

        dnameposfile.close();
        dnamenegfile.close();
        dnameequfile.close();
    }

    return doplot(xmin,xmax,omin,omax,fname,dname,outformat,incdata,incvar);
}

// Plot multiple graphs (specified by y) on a single 2-d axis

int multiplot2d(const Vector<Vector<gentype> > &y, const Vector<Vector<gentype> > &yvar, Vector<std::string> &plotlabels,
           double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, const std::string &title)
{
    NiceAssert( y.size() == yvar.size() );
    NiceAssert( y.size() == plotlabels.size() );

    if ( !y.size() )
    {
        return 0;
    }

    int m = y.size();
    int i,j,k,q;
    int maxobj = 0; // maximum number of objectives for any repitition

    double xmin = 0;
    double xmax = 0;

    Vector<std::string> dnamelist;
    Vector<int> repind;
    Vector<int> objind;

    for ( j = 0, q = 0 ; j < m ; ++j )
    {
        if ( y(j).size() )
        {
            int numobj = y(j)(0).isValVector() ? y(j)(0).size() : 0;
            // numobj = 0 for single-objective optimisation, otherwise number of objectives

            maxobj = ( numobj > maxobj ) ? numobj : maxobj;

            for ( k = 0 ; ( ( k < numobj ) || ( !k && !numobj ) ) ; ++k )
            {
                dnamelist.add(q);
                repind.add(q);
                objind.add(q);

                dnamelist("&",q) = dname+"_"; // Base for datafile name
                repind("&",q) = j; // Which repitition
                objind("&",q) = k; // Which objective

                //std::stringstream ssj;
                //ssj << j;
                //dnamelist("&",q) += ssj.str();
                dnamelist("&",q) += std::to_string(j);

                if ( numobj )
                {
                    dnamelist("&",q) += "_";

                    //std::stringstream ssk;
                    //ssk << k;
                    //dnamelist("&",q) += ssk.str();
                    dnamelist("&",q) += std::to_string(k);
                }

                dnamelist("&",q) += "_dat";

                std::ofstream dnameqfile(dnamelist(q));

                xmax = ( xmax > y(j).size() ) ? xmax : y(j).size(); // remember xmax is an integer count here!

                for ( i = 0 ; i < y(j).size() ; ++i )
                {
                    double varval = yvar(j)(i).isValNull() ? 0.0 : ( yvar(j)(i).isValVector() ? ((double) yvar(j)(i)(k)) : ((double) yvar(j)(i)) );

                    dnameqfile << i << "\t" << y(j)(i)(k) << "\t" << sqrt(sqrt(varval*varval)) << "\n";
                }

                dnameqfile.close();

                ++q;
            }
        }
    }

    //int numdatfiles = q; // total number of lines to draw

    return domultiplot2d(xmin,xmax,omin,omax,fname,dname,outformat,title,dnamelist,repind,objind,plotlabels,maxobj,1);
}



// Simple scatter plot

int scatterplot2d(const Vector<double> &x, const Vector<double> &y,
                  double xmin, double xmax, double ymin, double ymax,
                  const std::string &fname, const std::string &dname, int outformat)
{
    NiceAssert( x.size() == y.size() );

    if ( !y.size() )
    {
        return 0;
    }

    // create datafile

    std::ofstream datfile(dname.c_str());

    for ( int i = 0 ; i < y.size() ; i++ )
    {
        datfile << x(i) << "\t" << y(i) << "\n";
    }

    datfile.close();

    return doscatterplot2d(xmin,xmax,ymin,ymax,fname,dname,outformat);
}

















