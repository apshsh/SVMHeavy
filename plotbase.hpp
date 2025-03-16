
//
// Basic plotting function interface
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _plotbase_h
#define _plotbase_h

#define GRIDSAMP 100
#ifdef ENABLE_THREADS
#define PARSYSCALL false
#endif
#ifndef ENABLE_THREADS
#define PARSYSCALL false
#endif

#include "vector.hpp"
#include <string>



// Do a simple line-plot with optional variance, baseline and datapoints
//
// xmin,xmax: x data range (auto if xmin > xmax)
// omin,omax: y data range (auto if ymin > ymax)
// fname: output filename if used
// dname: datafile with format x y v b
//        x = x value
//        y = y value
//        v = variance
//        b = y baseline
//        dname_neg: upper bound constraints x y
//        dname_pos: lower bound constraints x y
//        dname_neg: equality constraints    x y
// outformat: 0 = dumb terminal output
//            1 = ps
//            2 = pdf
//            3 = plot in matlab
// incdata: 0 = just plot g(x)
//          1 = also include training samples
//          2 = also include baseline
//          3 = include both baseline and samples
// incvar:  0 = just the mean
//          1 = mean and variance

int doplot(double xmin, double xmax,
           double omin, double omax,
           const std::string &fname,
           const std::string &dname,
           int outformat, int incdata, int incvar, int doline = 1);

// Surface plot
//
// xmin,xmax: x data range (auto if xmin > xmax)
// ymin,ymax: y data range (auto if ymin > ymax)
// omin,omax: z data range (auto if zmin > zmax)
// fname: output filename if used
// dname: datafile with format x y z
//        x = x value
//        y = y value
//        v = z value
//        dname_neg: upper bound constraints x y z
//        dname_pos: lower bound constraints x y z
//        dname_neg: equality constraints    x y z
// outformat: 0 = dumb terminal output
//            1 = ps
//            2 = pdf
//            3 = plot in matlab

int dosurf(double xmin, double xmax,
           double ymin, double ymax,
           double omin, double omax,
           const std::string &fname,
           const std::string &dname,
           int outformat);

// Surface plot with variance
//
// xmin,xmax: x data range (auto if xmin > xmax)
// ymin,ymax: y data range (auto if ymin > ymax)
// omin,omax: z data range (auto if zmin > zmax)
// fname: output filename if used
// dname: datafile with format x y z v
//        x = x value
//        y = y value
//        z = z value
//        v = variance
//        dname_neg: upper bound constraints x y z
//        dname_pos: lower bound constraints x y z
//        dname_neg: equality constraints    x y z
// outformat: 0 = dumb terminal output
//            1 = ps
//            2 = pdf
//            3 = plot in matlab

int dosurfvar(double xmin, double xmax,
              double ymin, double ymax,
              double omin, double omax,
              const std::string &fname,
              const std::string &dvname,
              int outformat);

// Plot multiple graphs (specified by y) on a single 2-d axis
//
// xmin,xmax: x data range (auto if xmin > xmax)
// omin,omax: y data range (auto if ymin > ymax)
// fname: output filename if used
// dname: used internally (basename for temp files)
// outformat: 0 = dumb terminal output
//            1 = ps
//            2 = pdf
//            3 = plot in matlab
// title: title for plot
// dnamelist(q): datafile with format x y v b
//               x = x value
//               y = y value
//               v = variance
//               dname_neg: upper bound constraints x y
//               dname_pos: lower bound constraints x y
//               dname_neg: equality constraints    x y
// repind(q): which repetition
// objind(q): which index
// plotlabels(q): label for plot
// maxobj: maximum number of objectives for any repetition
// incvar: 0 = just the mean
//         1 = mean and variance

int domultiplot2d(double xmin, double xmax,
                  double omin, double omax,
                  const std::string &fname,
                  const std::string &dname,
                  int outformat,
                  const std::string &title,
                  const Vector<std::string> &dnamelist,
                  const Vector<int> &repind,
                  const Vector<int> &objind,
                  const Vector<std::string> &plotlabels,
                  int maxobj, int incvar);

// Simple scatter plot
//
// xmin,xmax: x data range (auto if xmin > xmax)
// omin,omax: y data range (auto if ymin > ymax)
// fname: output filename if used
// dname: datafile with format x y
//        x = x value
//        y = y value
// outformat: 0 = dumb terminal output
//            1 = ps
//            2 = pdf
//            3 = plot in matlab

int doscatterplot2d(double xmin, double xmax,
                    double ymin, double ymax,
                    const std::string &fname,
                    const std::string &dname,
                    int outformat);

#endif

