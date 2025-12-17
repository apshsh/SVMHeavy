
//
// Simple plotting front-end to spit out pdfs via gnu-plot and co
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _plotml_h
#define _plotml_h

#include "ml_base.hpp"
#include <string>

// xindex: which x element to use for x data
// yindex: which x element to use for y data
// xtemplate: all x vectors are based on this
//
// xmin: min x value
// xmax: max x value (xmin > xmax for auto, not allowed for ML form)
// ymin: min y value
// ymax: max y value (ymin > ymax for auto, not allowed for ML form)
// omin: min output value
// omax: max output value (omin > omax for auto)
//
// baseline: baseline function f(x)
//
// fname: output file name
// dname: data file created during operation
//        dname.neg is the d=-1 samples (if used)
//        dname.pos is the d=+1 samples (if used)
//        dname.equ is the d=2 samples (if used)
//        dname.plot is the actual plot command for gnuplot
//
// outformat: 0 = dumb terminal output
//            1 = ps
//            2 = pdf
//            3 = plot in matlab
//
// xusevar: variable used for baseline function (x)
//          0 = x = var(0,0)
//          1 = y = var(0,1)
//          2 = z = var(0,2)
//          3 = v = var(0,3)
//          4 = w = var(0,4)
//          5 = g = var(0,5)
//          6 = h = var(2,0)
// yusevar: variable used for baseline function (y)
//
// incdata: 0 = just plot g(x)
//          1 = also include training samples
//          2 = also include baseline
//          3 = include both baseline and samples
// incvar:  0 = just the mean
//          1 = mean and variance
// plotsq:  0 normal
//          1 plot g(x)^2
// plotimp: 0 normal
//          1 imp(x)


// Plot a simple line baseline(var), where var is set by xusevar (default x)

int plotfn2d(double xmin, double xmax, double omin, double omax,
             const std::string &fname, const std::string &dname, int outformat, const gentype &baseline,
             int xusevar = 0);


// Plot a simple surf baseline(xvar,yvar), where xvar,yvar are set by xusevar, yusevar (default x,y)

int surffn(double xmin, double xmax, double ymin, double ymax, double omin, double omax,
             const std::string &fname, const std::string &dname, int outformat, const gentype &baseline,
             int xusevar = 0, int yusevar = 1);


// Do a 2-d plot visualisation of ml

int plotml(const ML_Base &ml, int xindex,
           double xmin, double xmax, double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, int incdata, const gentype &baseline, int incvar, int xusevar,
           const SparseVector<gentype> &xtemplate, int plotsq = 0, int plotimp = 0, double scale = 1, double dscale = 1);


// Do a surface plot visualisation of ml

int plotml(const ML_Base &ml, int xindoes, int yindex,
           double xmin, double xmax, double ymin, double ymax, double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, int incdata, const gentype &baseline, int incvar, int xusevar, int yusevar,
           const SparseVector<gentype> &xtemplate, int plotsq = 0, int plotimp = 0, double scale = 1, double dscale = 1);



// Do a simple line-plot with optional variance, baseline and datapoints

int plot2d(const Vector<double> &x, const Vector<double> &y, const Vector<double> &yvar, const Vector<double> &ybaseline,
           const Vector<Vector<double> > &xpos, const Vector<Vector<double> > &ypos,
           const Vector<Vector<double> > &xneg, const Vector<Vector<double> > &yneg,
           const Vector<Vector<double> > &xequ, const Vector<Vector<double> > &yequ,
           double xmin, double xmax, double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, int incdata, int incvar);


// Plot multiple graphs (specified by y) on a single 2-d axis

int multiplot2d(const Vector<Vector<gentype> > &y, const Vector<Vector<gentype> > &yvar, Vector<std::string> &plotlabels,
           double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat, const std::string &title);


// Simple scatter plot

int scatterplot2d(const Vector<double> &x, const Vector<double> &y,
                  double xmin, double xmax, double ymin, double ymax,
                  const std::string &fname, const std::string &dname, int outformat);

#endif

