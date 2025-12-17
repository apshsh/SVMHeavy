
//
// Basic plotting function interface
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "plotbase.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include <sstream>
#include <fstream>

//#ifdef NDEBUG
#define DO_CLEANUP
//#endif

//#define DOCROP

//#ifdef ENABLE_THREADS
//#define PARSYSCALL false
//#endif
//#ifndef ENABLE_THREADS
#define PARSYSCALL false
//#endif

// Get preset dash-type strings
const char *dashtypes(int j);



int domultiplot2d(double xmin, double xmax, double omin, double omax,
                  const std::string &fname, const std::string &dname, int outformat, const std::string &title,
                  const Vector<std::string> &dnamelist, const Vector<int> &repind, const Vector<int> &objind, const Vector<std::string> &plotlabels,
                  int maxobj, int incvar)
{
    int numdatfiles = dnamelist.size(); // total number of lines to draw
    int q;

    // Construct the gnuplot script to do the actual plotting

    std::string dnameplot(dname);

    if ( outformat <= 2 )
    {
        dnameplot += ".plot";
    }

    else if ( outformat == 3 )
    {
        dnameplot += ".m";
    }

    std::ofstream dnameplotfile(dnameplot);

    std::string lwcomm("");

    // Header

    if ( outformat == 0 )
    {
        dnameplotfile << "set term dumb\n";
        dnameplotfile << "set size ratio 0.75\n";

        if ( title.length() )
        {
            dnameplotfile << "set title \"" << title << "\"\n";
        }
    }

    else if ( outformat == 1 )
    {
        dnameplotfile << "set terminal postscript portrait enhanced color dashed lw 1 \"DejaVuSans\" 12\n";
        dnameplotfile << "set output \"" << fname << ".ps\"\n";
        dnameplotfile << "set size ratio 0.75\n";

        if ( title.length() )
        {
            dnameplotfile << "set title \"" << title << "\"\n";
        }

        lwcomm = "lw 2";
    }

    else if ( outformat == 2 )
    {
        dnameplotfile << "set terminal pdfcairo enhanced truecolor dashed lw 1\n";
        dnameplotfile << "set output \"" << fname << ".pdf\"\n";
        dnameplotfile << "set size ratio 0.75\n";

        if ( title.length() )
        {
            dnameplotfile << "set title \"" << title << "\"\n";
        }

        lwcomm = "lw 1";
    }

    else if ( outformat == 3 )
    {
        dnameplotfile << "function svmh_fig_" << fname << " = " << dname << "()\n";

        if ( title.length() )
        {
            dnameplotfile << "svmh_fig_" << fname << " = figure('Name','" << title << "');\n";
        }

        else
        {
            dnameplotfile << "svmh_fig_" << fname << " = figure;\n";
        }

        dnameplotfile << "hold on;\n";
    }

    // Axis
    //
    // outformat 3 axis are dealt with after plotting

    if ( outformat <= 2 )
    {
        if ( xmin < xmax )
        {
            dnameplotfile << "set xrange [ " << xmin << " : " << xmax << " ]\n";
        }

        if ( omin < omax )
        {
            dnameplotfile << "set yrange [ " << omin << " : " << omax << " ]\n";
        }

        dnameplotfile << "set border 3\n";
        dnameplotfile << "set xtics nomirror font \",8\"\n";
        dnameplotfile << "set ytics nomirror font \",8\"\n";
        dnameplotfile << "set mxtics 5\n";
        dnameplotfile << "set mytics 5\n";
        dnameplotfile << "set grid mxtics mytics\n";
        dnameplotfile << "set samples 500\n";
        dnameplotfile << "set style fill transparent solid 0.30 noborder\n";
    }

    // Plot data

    if ( outformat <= 2 )
    {
        dnameplotfile << "plot \\\n";

        if ( incvar )
        {
            for ( q = 0 ; q < numdatfiles ; ++q )
            {
                int lt = ( q+1 < 6 ) ? q+1 : q+2; // skip the blue/purple duplicate because I'm blue/purple colour blind

                //dnameplotfile << "'" << dnamelist(q) << "' using 1:($2-$3):($2+$3) with filledcurve fc " << lt << " fs transparent solid 0.30 t \"" << "\", \\\n";
                dnameplotfile << "'" << dnamelist(q) << "' using 1:($2-$3):($2+$3) with filledcurve lc " << lt << " fs transparent solid 0.30 t \"" << "\", \\\n";
            }
        }

        // put the lines over the top for clarity

        for ( q = 0 ; q < numdatfiles ; ++q )
        {
            int lt = ( q+1 < 6 ) ? q+1 : q+2; // skip the blue/purple duplicate because I'm blue/purple colour blind

            std::string plotlab = plotlabels(repind(q));

            if ( maxobj )
            {
                plotlab += " (objective ";
                plotlab += std::to_string(objind(q));
                plotlab += ")";
            }

            dnameplotfile << "'" << dnamelist(q) << "' using 1:2 with lines lt " << lt << " dt " << dashtypes(q+1) << " " << lwcomm << " t \"" << plotlab << ( ( q == numdatfiles-1 ) ? "\"" : "\", \\\n" );
        }
    }

    else if ( outformat == 3 )
    {
        // Construct vector of names to be given to data

        Vector<std::string> xvarnamelist(numdatfiles);
        Vector<std::string> yvarnamelist(numdatfiles);
        Vector<std::string> yvarvarnamelist(numdatfiles);
        Vector<std::string> yminvarnamelist(numdatfiles);
        Vector<std::string> ymaxvarnamelist(numdatfiles);

        for ( q = 0 ; q < numdatfiles ; ++q )
        {
            xvarnamelist("&",q) = "svmh_xdat_";
            xvarnamelist("&",q) += dnamelist(q);

            yvarnamelist("&",q) = "svmh_ydat_";
            yvarnamelist("&",q) += dnamelist(q);

            yvarvarnamelist("&",q) = "svmh_yvardat_";
            yvarvarnamelist("&",q) += dnamelist(q);

            yminvarnamelist("&",q) = "svmh_ymindat_";
            yminvarnamelist("&",q) += dnamelist(q);

            ymaxvarnamelist("&",q) = "svmh_ymaxdat_";
            ymaxvarnamelist("&",q) += dnamelist(q);
        }

        // Load y data

        for ( q = 0 ; q < numdatfiles ; ++q )
        {
            dnameplotfile << "tmpdat = load('" << dname << "');\n";
            dnameplotfile << xvarnamelist(q).c_str() << " = tmpdat(:,1);\n";
            dnameplotfile << yvarnamelist(q).c_str() << " = tmpdat(:,2);\n";
            dnameplotfile << yminvarnamelist(q).c_str() << " = " << yvarnamelist(q).c_str() << " - " << yvarvarnamelist(q).c_str() << ";\n";
            dnameplotfile << ymaxvarnamelist(q).c_str() << " = " << yvarnamelist(q).c_str() << " + " << yvarvarnamelist(q).c_str() << ";\n";
        }

        // Construct x data

        if ( xmin < xmax )
        {
            for ( q = 0 ; q < numdatfiles ; ++q )
            {
                dnameplotfile << xvarnamelist(q).c_str() << " = (" << xvarnamelist(q).c_str() << "*" << xmax-xmin << ")+" << xmin << ";\n";
            }
        }

        // Now plot the data, first the filled regions, then the central lines

        if ( incvar )
        {
            for ( q = 0 ; q < numdatfiles ; ++q )
            {
                char ct = 0;

                switch ( q%6 )
                {
                    case 0:  { ct = 'g'; break; }
                    case 1:  { ct = 'r'; break; }
                    case 2:  { ct = 'b'; break; }
                    case 3:  { ct = 'c'; break; }
                    case 4:  { ct = 'm'; break; }
                    default: { ct = 'y'; break; }
                }

                dnameplotfile << "xvalsare = [" << xvarnamelist(q).c_str() << "', fliplr(" << xvarnamelist(q).c_str() << "')];\n";
                dnameplotfile << "inBetween = [" << yminvarnamelist(q).c_str() << "', fliplr(" << ymaxvarnamelist(q).c_str() << "')];\n";
                dnameplotfile << "h = fill(xvalsare, inBetween, '" << ct << "', 'FaceAlpha', 0.3);\n";
            }
        }

        dnameplotfile << "plot(";

        for ( q = 0 ; q < numdatfiles ; ++q )
        {
            char ct = 0;

            switch ( q%6 )
            {
                case 0:  { ct = 'g'; break; }
                case 1:  { ct = 'r'; break; }
                case 2:  { ct = 'b'; break; }
                case 3:  { ct = 'c'; break; }
                case 4:  { ct = 'm'; break; }
                default: { ct = 'y'; break; }
            }

            std::string lt;

            switch ( (q/6)%4 )
            {
                case 0:  { lt = "-";  break; }
                case 1:  { lt = "--"; break; }
                case 2:  { lt = ":";  break; }
                default: { lt = "-."; break; }
            }

            std::string plotlab = plotlabels(repind(q));

            if ( maxobj )
            {
                plotlab += " (objective ";
                plotlab += std::to_string(objind(q));
                plotlab += ")";
            }

            dnameplotfile << xvarnamelist(q).c_str() << "," << yvarnamelist(q).c_str() << ",'LineStyle','" << lt << "','Color','" << ct << "','DisplayName','" << plotlab << "'";

            if ( q < numdatfiles-1 )
            {
                dnameplotfile << ", ";
            }
        }

        dnameplotfile << ");\n";
    }

    // Axis again (outformat 3)

    if ( outformat == 3 )
    {
        if ( xmin < xmax )
        {
            dnameplotfile << "xlim([ " << xmin << "," << xmax << " ]);\n";
        }

        if ( omin < omax )
        {
            dnameplotfile << "ylim([ " << omin << "," << omax << " ]);\n";
        }

        dnameplotfile << "hold off;\n";
        dnameplotfile << "end\n";
    }

    dnameplotfile << "\n";

    dnameplotfile.close();

    // Construct the shell-script to call gnuplot and do subsequent file conversions, if needed

    std::string dnamesh = dname+"dognuplot.sh";

    if ( outformat <= 2 )
    {
        std::ofstream dnameshfile(dnamesh);

        if ( outformat == 0 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << "\n";
        }

        else if ( outformat == 1 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";
        }

        else if ( outformat == 2 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";

#ifdef DOCROP
            //dnameshfile << "epstopdf " << fname << ".ps  >/dev/null 2>/dev/null\n";
            dnameshfile << "pdfcrop  " << fname << ".pdf >/dev/null 2>/dev/null\n";
            dnameshfile << "mv " << fname << "-crop.pdf " << fname << ".pdf\n";
#endif
        }

        dnameshfile.close();
    }

    else if ( outformat == 3 )
    {
        ;
    }

    // Call the shell script with fingers crossed

    int res = 0;

    if ( outformat <= 2 )
    {
        {
            std::string doplotcomm = "bash ./"+dnamesh;

#ifndef USE_MEX
            res = svm_system(doplotcomm.c_str());
#endif
        }

        // Clean up

#ifndef USE_MEX
#ifdef DO_CLEANUP
        std::string delstringa = "rm "+dnameplot;
        std::string delstringb = "rm "+dnamesh;

        svm_system(delstringa.c_str());
        svm_system(delstringb.c_str());
#endif
#endif
errstream() << "phantomx 8\n";
    }

    else if ( outformat == 3 )
    {
        res = svm_system(dname.c_str());

        // Clean up

#ifndef USE_MEX
#ifdef DO_CLEANUP
        std::string delstringa = "rm "+dnameplot;

        svm_system(delstringa.c_str());
#endif
#endif
    }

    return res;
}






int doplot(double xmin, double xmax, double omin, double omax, const std::string &fname, const std::string &dname, int outformat, int incdata, int incvar, int doline, int numdat)
{
    std::string dnamepos = dname+"_pos";
    std::string dnameneg = dname+"_neg";
    std::string dnameequ = dname+"_equ";
    std::string linetype = ( doline == 1 ) ? "smooth cspline" : ( ( doline == 2 ) ? "with lines" : "" );

    // Construct the gnuplot script to do the actual plotting

    std::string dnameplot(dname);

    if ( outformat <= 2 )
    {
        dnameplot += ".plot";
    }

    else if ( outformat == 3 )
    {
        dnameplot += ".m";
    }

    std::ofstream dnameplotfile(dnameplot);

    std::string lwcomm("lw 1");

    // Header

    if ( outformat == 0 )
    {
        dnameplotfile << "set term dumb\n";
    }

    else if ( outformat == 1 )
    {
        dnameplotfile << "set terminal postscript portrait enhanced color dashed lw 1 \"DejaVuSans\" 12\n";
        dnameplotfile << "set output \"" << fname << ".ps\"\n";
        dnameplotfile << "set size ratio 0.75\n";

        lwcomm = "lw 2";
    }

    else if ( outformat == 2 )
    {
        dnameplotfile << "set terminal pdfcairo enhanced truecolor dashed lw 1\n";
        dnameplotfile << "set output \"" << fname << ".pdf\"\n";
        dnameplotfile << "set size ratio 0.75\n";

        lwcomm = "lw 1";
    }

    else if ( outformat == 3 )
    {
        dnameplotfile << "function svmh_fig_" << fname << " = " << dname << "()\n";

        dnameplotfile << "svmh_fig_" << fname << " = figure;\n";
        dnameplotfile << "hold on;\n";
    }

    // Axis
    //
    // outformat 3 axis are dealt with after plitting

    if ( outformat <= 2 )
    {
        if ( xmin < xmax )
        {
            dnameplotfile << "set xrange [ " << xmin << " : " << xmax << " ]\n";
        }

        if ( omin < omax )
        {
            dnameplotfile << "set yrange [ " << omin << " : " << omax << " ]\n";
        }

        dnameplotfile << "set border 3\n";
        dnameplotfile << "set xtics nomirror font \",8\"\n";
        dnameplotfile << "set ytics nomirror font \",8\"\n";
        dnameplotfile << "set mxtics 5\n";
        dnameplotfile << "set mytics 5\n";
        dnameplotfile << "set grid mxtics mytics\n";
        dnameplotfile << "set samples 500\n";
        dnameplotfile << "set style fill transparent solid 0.30 noborder\n";
        //dnameplotfile << "set style fill solid 0.4 noborder\n";
    }

    // Plot data

    if ( outformat <= 2 )
    {
        if ( numdat )
        {
            dnameplotfile << "set encoding utf8\n";
            dnameplotfile << "symbol(z) = \"•✷+△♠□♣♥♦\"[int(z):int(z)]\n";
        }

        dnameplotfile << "plot \\\n";

        if ( incvar )
        {
            dnameplotfile << "'" << dname << "' using 1:($2-$3):($2+$3) with filledcurve lt 2 t \"\", \\\n'" << dname << "' using 1:2 " << linetype << " lt -1 " << lwcomm << " t \"\"";
        }

        else
        {
            dnameplotfile << "'" << dname << "' using 1:2 " << linetype << " lt -1 " << lwcomm << " t \"\"";
        }

        // Additional data if needed

        if ( ( 2 & incdata ) )
        {
            dnameplotfile << ", \\\n'' using 1:4 " << linetype << " lt -1 dt \".\" " << lwcomm << " t \"\"";
        }

        if ( ( 1 & incdata ) && !numdat )
        {
            dnameplotfile << ", \\\n'" << dnamepos << "' w points lt -1 lw 1 pt 1 ps 0.5 t\"\"";
            dnameplotfile << ", \\\n'" << dnameneg << "' w points lt -1 lw 1 pt 4 ps 0.5 t\"\"";
            dnameplotfile << ", \\\n'" << dnameequ << "' w points lt -1 lw 1 pt 2 ps 0.5 t\"\"";
        }

        else if ( ( 1 & incdata ) && numdat )
        {
            dnameplotfile << ", \\\n'" << dnamepos << "' using 1:2:(symbol($3)) lt -1 lw 1 ps 0.5 t\"\"";
            dnameplotfile << ", \\\n'" << dnameneg << "' using 1:2:(symbol($3)) lt -1 lw 1 ps 0.5 t\"\"";
            dnameplotfile << ", \\\n'" << dnameequ << "' using 1:2:(symbol($3)) lt -1 lw 1 ps 0.5 t\"\"";
        }

        dnameplotfile << "\n";
    }

    else if ( outformat == 3 )
    {
        // Construct vector of names to be given to data

        std::string xvarnamelist;
        std::string yvarnamelist;
        std::string yvarvarnamelist;
        std::string yrefvarnamelist;
        std::string yminvarnamelist;
        std::string ymaxvarnamelist;
        std::string xposvarnamelist;
        std::string yposvarnamelist;
        std::string xnegvarnamelist;
        std::string ynegvarnamelist;
        std::string xequvarnamelist;
        std::string yequvarnamelist;

        xvarnamelist = "svmh_xdat_";
        xvarnamelist += dname;

        yvarnamelist = "svmh_ydat_";
        yvarnamelist += dname;

        yvarvarnamelist = "svmh_ydatvar_";
        yvarvarnamelist += dname;

        yrefvarnamelist = "svmh_ydatref_";
        yrefvarnamelist += dname;

        yminvarnamelist = "svmh_ydatmin_";
        yminvarnamelist += dname;

        ymaxvarnamelist = "svmh_ydatmax_";
        ymaxvarnamelist += dname;

        xposvarnamelist = "svmh_xdatpos_";
        xposvarnamelist += dname;

        yposvarnamelist = "svmh_ydatpos_";
        yposvarnamelist += dname;

        xnegvarnamelist = "svmh_xdatneg_";
        xnegvarnamelist += dname;

        ynegvarnamelist = "svmh_ydatneg_";
        ynegvarnamelist += dname;

        xequvarnamelist = "svmh_xdatequ_";
        xequvarnamelist += dname;

        yequvarnamelist = "svmh_ydatequ_";
        yequvarnamelist += dname;

        // Load y data

        dnameplotfile << "tmpdat = load('" << dname << "');\n";
        dnameplotfile << xvarnamelist.c_str() << " = tmpdat(:,1);\n";
        dnameplotfile << yvarnamelist.c_str() << " = tmpdat(:,2);\n";
        dnameplotfile << yvarvarnamelist.c_str() << " = tmpdat(:,3);\n";
        dnameplotfile << yrefvarnamelist.c_str() << " = tmpdat(:,4);\n";
        dnameplotfile << yminvarnamelist.c_str() << " = " << yvarnamelist.c_str() << " - " << yvarvarnamelist.c_str() << ";\n";
        dnameplotfile << ymaxvarnamelist.c_str() << " = " << yvarnamelist.c_str() << " + " << yvarvarnamelist.c_str() << ";\n";

        // Now plot the data, first the filled regions, then the central lines

        if ( incvar )
        {
            char ct = 'g';

            dnameplotfile << "xvalsare = [" << xvarnamelist.c_str() << "', fliplr(" << xvarnamelist.c_str() << "')];\n";
            dnameplotfile << "inBetween = [" << ymaxvarnamelist.c_str() << "', fliplr(" << yminvarnamelist.c_str() << "')];\n";
            dnameplotfile << "h = fill(xvalsare, inBetween, '" << ct << "', 'FaceAlpha', 0.3);\n";
        }

        {
            char ct = 'k';
            std::string lt = "-";

            dnameplotfile << "plot(" << xvarnamelist.c_str() << "," << yvarnamelist.c_str() << ",'LineStyle','" << lt << "','Color','" << ct << "','DisplayName','" << "');\n";
        }

        if ( ( 2 & incdata ) )
        {
            char ct = 'b';
            std::string lt = "-.";

            dnameplotfile << "plot(" << xvarnamelist.c_str() << "," << yrefvarnamelist.c_str() << ",'LineStyle','" << lt << "','Color','" << ct << "','DisplayName','" << "');\n";
        }

        if ( ( 1 & incdata ) )
        {
            dnameplotfile << "tmpdat = load('" << dnamepos << "');\n";
            dnameplotfile << "if length(tmpdat) > 0\n";
            dnameplotfile << "  " << xposvarnamelist.c_str() << " = tmpdat(:,1);\n";
            dnameplotfile << "  " << yposvarnamelist.c_str() << " = tmpdat(:,2);\n";
            dnameplotfile << "  scatter(" << xposvarnamelist.c_str() << ", " << yposvarnamelist.c_str() << ",'r+');\n";
            dnameplotfile << "end\n";

            dnameplotfile << "tmpdat = load('" << dnameneg << "');\n";
            dnameplotfile << "if length(tmpdat) > 0\n";
            dnameplotfile << "  " << xnegvarnamelist.c_str() << " = tmpdat(:,1);\n";
            dnameplotfile << "  " << ynegvarnamelist.c_str() << " = tmpdat(:,2);\n";
            dnameplotfile << "  scatter(" << xnegvarnamelist.c_str() << ", " << ynegvarnamelist.c_str() << ",'r-');\n";
            dnameplotfile << "end\n";

            dnameplotfile << "tmpdat = load('" << dnameequ << "');\n";
            dnameplotfile << "if length(tmpdat) > 0\n";
            dnameplotfile << "  " << xequvarnamelist.c_str() << " = tmpdat(:,1);\n";
            dnameplotfile << "  " << yequvarnamelist.c_str() << " = tmpdat(:,2);\n";
            dnameplotfile << "  scatter(" << xequvarnamelist.c_str() << ", " << yequvarnamelist.c_str() << ",'rx');\n";
            dnameplotfile << "end\n";
        }
    }

    // Axis again (outformat 3)

    if ( outformat == 3 )
    {
        if ( xmin < xmax )
        {
            dnameplotfile << "xlim([ " << xmin << "," << xmax << " ]);\n";
        }

        if ( omin < omax )
        {
            dnameplotfile << "ylim([ " << omin << "," << omax << " ]);\n";
        }

        dnameplotfile << "hold off;\n";
        dnameplotfile << "end\n";
    }

    dnameplotfile << "\n";

    dnameplotfile.close();

    // Construct the shell-script to call gnuplot and do subsequent file conversions, if needed

    std::string dnamesh = dname+"dognuplot.sh";

    if ( outformat <= 2 )
    {
        std::ofstream dnameshfile(dnamesh);

        if ( outformat == 0 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " > " << fname << ".txt 2>/dev/null\n";
        }

        else if ( outformat == 1 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";
        }

        else if ( outformat == 2 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";

#ifdef DOCROP
            //dnameshfile << "epstopdf " << fname << ".ps  >/dev/null 2>/dev/null\n";
            dnameshfile << "pdfcrop  " << fname << ".pdf >/dev/null 2>/dev/null\n";
            dnameshfile << "mv " << fname << "-crop.pdf " << fname << ".pdf\n";
#endif
        }

        dnameshfile.close();
    }

    else if ( outformat == 3 )
    {
        ;
    }

    // Call the shell script with fingers crossed

    int res = 0;

    if ( outformat <= 2 )
    {
        if ( PARSYSCALL )
        {
//#ifndef ENABLE_THREADS
            std::string doplotcomm = "bash ./"+dnamesh+" &";

#ifndef USE_MEX
            res = svm_system(doplotcomm.c_str());
#endif
//#endif
//#ifdef ENABLE_THREADS
//            std::string *doplotcomm = new std::string("bash ./"+dnamesh);
//
//#ifndef USE_MEX
//            std::thread(svm_system,(*doplotcomm).c_str());
//#endif
//#endif
        }

        else
        {
            std::string doplotcomm = "bash ./"+dnamesh;

#ifndef USE_MEX
            res = svm_system(doplotcomm.c_str());
#endif
        }

        // Clean up

#ifndef USE_MEX
#ifdef DO_CLEANUP
        std::string delstringa = "rm "+dnameplot;
        std::string delstringb = "rm "+dnamesh;

        svm_system(delstringa.c_str());
        svm_system(delstringb.c_str());
#endif
#endif
    }

    else if ( outformat == 3 )
    {
        res = svm_system(dname.c_str());

        // Clean up

#ifndef USE_MEX
#ifdef DO_CLEANUP
        std::string delstringa = "rm "+dnameplot;

        svm_system(delstringa.c_str());
#endif
#endif
    }

    return res;
}





//FIXME up to here
int dosurf(double xmin, double xmax, double ymin, double ymax, double omin, double omax,
           const std::string &fname, const std::string &dname, int outformat)
{
    std::string dnamepos = dname+"_pos";
    std::string dnameneg = dname+"_neg";
    std::string dnameequ = dname+"_equ";


    // Construct the gnuplot script to do the actual plotting

    std::string dnameplot(dname);

    if ( outformat <= 2 )
    {
        dnameplot += ".plot";
    }

    else if ( outformat == 3 )
    {
        dnameplot += ".m";
    }

    std::ofstream dnameplotfile(dnameplot);

    std::string lwcomm("lw 1");

    // Header

    if ( outformat == 0 )
    {
        dnameplotfile << "set term dumb\n";
    }

    else if ( outformat == 1 )
    {
        dnameplotfile << "set terminal postscript portrait enhanced color dashed lw 1 \"DejaVuSans\" 12\n";
        dnameplotfile << "set output \"" << fname << ".ps\"\n";
        dnameplotfile << "set size ratio 0.75\n";

        lwcomm = "lw 2";
    }

    else if ( outformat == 2 )
    {
        dnameplotfile << "set terminal pdfcairo enhanced truecolor dashed lw 1\n";
        dnameplotfile << "set output \"" << fname << ".pdf\"\n";
        dnameplotfile << "set size ratio 0.75\n";

        lwcomm = "lw 1";
    }

    else if ( outformat == 3 )
    {
        dnameplotfile << "function svmh_fig_" << fname << " = " << dname << "()\n";

        dnameplotfile << "svmh_fig_" << fname << " = figure;\n";
        dnameplotfile << "hold on;\n";
    }

    // Axis
    //
    // outformat 3 axis are dealt with after plitting

    if ( outformat <= 2 )
    {
        if ( xmin < xmax )
        {
            dnameplotfile << "set xrange [ " << xmin << " : " << xmax << " ]\n";
        }

        if ( ymin < ymax )
        {
            dnameplotfile << "set yrange [ " << ymin << " : " << ymax << " ]\n";
        }

        if ( omin < omax )
        {
            dnameplotfile << "set zrange [ " << omin << " : " << omax << " ]\n";
        }

        dnameplotfile << "set hidden3d\n";
        dnameplotfile << "set dgrid3d " << GRIDSAMP << "," << GRIDSAMP << " qnorm 2\n";
        dnameplotfile << "set pm3d\n"; // heatmap
        dnameplotfile << "set palette\n";
        dnameplotfile << "unset colorbox\n";

        dnameplotfile << "#set xyplane at 0\n";
        dnameplotfile << "set cntrparam levels 10\n";
        dnameplotfile << "set contour both\n";
        dnameplotfile << "set format x \"\"\n";
        dnameplotfile << "set format y \"\"\n";
        dnameplotfile << "#set format z \"\"\n";
        dnameplotfile << "unset key\n";
    }

    // Plot data

    if ( outformat <= 2 )
    {
        dnameplotfile << "splot '" << dname << "' with lines";
    }

    else if ( outformat == 3 )
    {
        dnameplotfile << "tmpdat = load('" << dname << "');\n";
//        dnameplotfile << "svmh_fig_" << fname << " = plot3(tmpdat(:,1),tmpdat(:,2),tmpdat(:,3),'x');\n";
        dnameplotfile << "svmh_fig_" << fname << " = scatter3(tmpdat(:,1),tmpdat(:,2),tmpdat(:,3),1,tmpdat(:,3));\n";
    }

    // Axis again (outformat 3)

    if ( outformat == 3 )
    {
        if ( xmin < xmax )
        {
            dnameplotfile << "xlim([ " << xmin << "," << xmax << " ]);\n";
        }

        if ( ymin < ymax )
        {
            dnameplotfile << "ylim([ " << ymin << "," << ymax << " ]);\n";
        }

        if ( omin < omax )
        {
            dnameplotfile << "zlim([ " << omin << "," << omax << " ]);\n";
        }

        dnameplotfile << "%hold off;\n";
        dnameplotfile << "end\n";
    }

    dnameplotfile << "\n";

    dnameplotfile.close();

    // Construct the shell-script to call gnuplot and do subsequent file conversions, if needed

    std::string dnamesh = dname+"dognuplot.sh";

    if ( outformat <= 2 )
    {
        std::ofstream dnameshfile(dnamesh);

        if ( outformat == 0 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " > " << fname << ".txt 2>/dev/null\n";
        }

        else if ( outformat == 1 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";
        }

        else if ( outformat == 2 )
        {
            dnameshfile << "#!/usr/bin\n";
            dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";

#ifdef DOCROP
            //dnameshfile << "epstopdf " << fname << ".ps  >/dev/null 2>/dev/null\n";
            dnameshfile << "pdfcrop  " << fname << ".pdf >/dev/null 2>/dev/null\n";
            dnameshfile << "mv " << fname << "-crop.pdf " << fname << ".pdf\n";
#endif
        }

        dnameshfile.close();
    }

    else if ( outformat == 3 )
    {
        ;
    }

    // Call the shell script with fingers crossed

    int res = 0;

    if ( outformat <= 2 )
    {
        if ( PARSYSCALL )
        {
//#ifndef ENABLE_THREADS
            std::string doplotcomm = "bash ./"+dnamesh+" &";

#ifndef USE_MEX
            res = svm_system(doplotcomm.c_str());
#endif
//#endif
//#ifdef ENABLE_THREADS
//            std::string *doplotcomm = new std::string("bash ./"+dnamesh);
//
//#ifndef USE_MEX
//            std::thread(svm_system,(*doplotcomm).c_str());
//#endif
//#endif
        }

        else
        {
            std::string doplotcomm = "bash ./"+dnamesh;

#ifndef USE_MEX
            res = svm_system(doplotcomm.c_str());
#endif
        }

        // Clean up
/*
#ifndef USE_MEX
#ifdef DO_CLEANUP
        std::string delstringa = "rm "+dnameplot;
        std::string delstringb = "rm "+dnamesh;

        svm_system(delstringa.c_str());
        svm_system(delstringb.c_str());
#endif
#endif
*/
    }

    else if ( outformat == 3 )
    {
        res = svm_system(dname.c_str());

        // Clean up

#ifndef USE_MEX
#ifdef DO_CLEANUP
        std::string delstringa = "rm "+dnameplot;

//        svm_system(delstringa.c_str());
#endif
#endif
    }

    return res;
}

int dosurfvar(double xmin, double xmax, double ymin, double ymax, double omin, double omax,
              const std::string &fname, const std::string &dname, int outformat, int incdata)
{
(void) incdata;
    // Construct the gnuplot script to do the actual plotting

    std::string dnamevar = dname+"_var";
    std::string dnamepos = dname+"_pos";
    std::string dnameneg = dname+"_neg";
    std::string dnameequ = dname+"_equ";

    std::string dnameplot(dname);

    if ( outformat <= 2 )
    {
        dnameplot += ".plot";
    }

    else if ( outformat == 3 )
    {
        dnameplot += ".m";
    }

    std::ofstream dnameplotfile(dnameplot);

    if ( outformat == 2 )
    {
        dnameplotfile << "set terminal pdfcairo enhanced truecolor dashed lw 1\n";
        dnameplotfile << "set output \"" << fname << ".pdf\"\n";
    }

    else if ( outformat == 1 )
    {
        dnameplotfile << "set terminal postscript portrait enhanced color dashed lw 1 \"DejaVuSans\" 12\n";
        dnameplotfile << "set output \"" << fname << ".ps\"\n";
    }

    else
    {
        dnameplotfile << "set term dumb\n";
    }

    dnameplotfile << "set size ratio 0.75\n";

    dnameplotfile << "set dgrid3d " << GRIDSAMP << "," << GRIDSAMP << "\n"; // << " qnorm 2\n";
    dnameplotfile << "dataFile=\'" << dnamevar << "\'\n";

    dnameplotfile << "set table dataFile.'.grid'\n";
    dnameplotfile << "splot dataFile u 1:2:3\n";
    dnameplotfile << "unset table\n";

    dnameplotfile << "set table dataFile.'.color'\n";
    dnameplotfile << "splot dataFile u 1:2:4\n";
    dnameplotfile << "unset table\n";

    dnameplotfile << "#set view 60,45\n";
    dnameplotfile << "set hidden3d\n";
    dnameplotfile << "set autoscale cbfix\n";
    dnameplotfile << "set pm3d\n";
    dnameplotfile << "unset dgrid3d\n";
    dnameplotfile << "set ticslevel 0\n";

    dnameplotfile << "set xrange [ " << xmin << " : " << xmax << " ]\n";
    dnameplotfile << "set yrange [ " << ymin << " : " << ymax << " ]\n";
    dnameplotfile << "set zrange [ " << omin << " : " << omax << " ]\n";
    dnameplotfile << "#set format x \"\"\n";
    dnameplotfile << "#set format y \"\"\n";
    dnameplotfile << "#set format z \"\"\n";

    dnameplotfile << "set xtics nomirror font \",8\"\n";
    dnameplotfile << "set ytics nomirror font \",8\"\n";
    dnameplotfile << "set ztics nomirror font \",8\"\n";

    dnameplotfile << "splot sprintf('< paste %s.grid %s.color', dataFile, dataFile) u 1:2:3:7 with pm3d notitle";

//The output is (a) ugly and (b) data is almost all occluded by surface plot
//    if ( ( 1 & incdata ) )
//    {
////        dnameplotfile << ", '" << dnamepos << "' u 1:2:3 w points lt -1 lw 1 pt 1 ps 0.5 t\"\"";
////        dnameplotfile << ", '" << dnameneg << "' u 1:2:3 w points lt -1 lw 1 pt 4 ps 0.5 t\"\"";
////        dnameplotfile << ", '" << dnameequ << "' u 1:2:3 w points lt -1 lw 1 pt 2 ps 0.5 t\"\"";
//        dnameplotfile << ", '" << dnamepos << "' u 1:2:3 w points pt 1 t\"\"";
//        dnameplotfile << ", '" << dnameneg << "' u 1:2:3 w points pt 4 t\"\"";
//        dnameplotfile << ", '" << dnameequ << "' u 1:2:3 w points pt 2 t\"\"";
//    }

    dnameplotfile << "\n";

    dnameplotfile.close();

    // Construct the shell-script to call gnuplot and do subsequent file conversions, if needed

    std::string dnamesh = dname+"dognuplot.sh";

    std::ofstream dnameshfile(dnamesh);

    if ( outformat )
    {
        dnameshfile << "#!/usr/bin\n";
        dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";
    }

    else
    {
        dnameshfile << "#!/usr/bin\n";
        dnameshfile << "gnuplot " << dnameplot << "\n";
    }

    if ( outformat == 2 )
    {
        ;
#ifdef DOCROP
        //dnameshfile << "epstopdf " << fname << ".ps  >/dev/null 2>/dev/null\n";
        dnameshfile << "pdfcrop  " << fname << ".pdf >/dev/null 2>/dev/null\n";
        dnameshfile << "mv " << fname << "-crop.pdf " << fname << ".pdf\n";
#endif
    }

    dnameshfile.close();

    // Call the shell script with fingers crossed

    int res = 0;

    if ( PARSYSCALL )
    {
//#ifndef ENABLE_THREADS
        std::string doplotcomm = "bash ./"+dnamesh+" &";

#ifndef USE_MEX
        res = svm_system(doplotcomm.c_str());
#endif
//#endif
//#ifdef ENABLE_THREADS
//        std::string *doplotcomm = new std::string("bash ./"+dnamesh);
//
//#ifndef USE_MEX
//        std::thread(svm_system,(*doplotcomm).c_str());
//#endif
//#endif
    }

    else
    {
        std::string doplotcomm = "bash ./"+dnamesh;

#ifndef USE_MEX
        res = svm_system(doplotcomm.c_str());
#endif
    }

    // Clean up

#ifndef USE_MEX
#ifdef DO_CLEANUP
    std::string delstringa = "rm "+dnameplot;
    std::string delstringb = "rm "+dnamesh;
    std::string delstringc = "rm "+dnamevar+".color";
    std::string delstringd = "rm "+dnamevar+".grid";

    svm_system(delstringa.c_str());
    svm_system(delstringb.c_str());
    svm_system(delstringc.c_str());
    svm_system(delstringd.c_str());
#endif
#endif

    return res;
}







int doscatterplot2d(double xmin, double xmax, double ymin, double ymax,
                    const std::string &fname, const std::string &dname, int outformat)
{
    std::string dnameplot = dname+".plot";

    std::ofstream dnameplotfile(dnameplot);

    std::string lwcomm("lw 1");

    if ( outformat == 2 )
    {
        dnameplotfile << "set terminal pdfcairo enhanced truecolor dashed lw 1\n";
        dnameplotfile << "set output \"" << fname << ".pdf\"\n";
    }

    else if ( outformat == 1 )
    {
        dnameplotfile << "set terminal postscript portrait enhanced color dashed lw 1 \"DejaVuSans\" 12\n";
        dnameplotfile << "set output \"" << fname << ".ps\"\n";

        lwcomm = "lw 2";
    }

    else
    {
        dnameplotfile << "set term dumb\n";
    }

    if ( xmin < xmax )
    {
        dnameplotfile << "set xrange [ " << xmin << " : " << xmax << " ]\n";
    }

    if ( ymin < ymax )
    {
        dnameplotfile << "set yrange [ " << ymin << " : " << ymax << " ]\n";
    }

    dnameplotfile << "set size ratio 0.75\n";
    dnameplotfile << "set border 3\n";
    dnameplotfile << "#set xtics nomirror font \",8\"\n";
    dnameplotfile << "#set ytics nomirror font \",8\"\n";
    dnameplotfile << "set mxtics 5\n";
    dnameplotfile << "set mytics 5\n";
    dnameplotfile << "set grid mxtics mytics\n";
    dnameplotfile << "#set samples 500\n";
    dnameplotfile << "set style circle radius 0.01\n";
    dnameplotfile << "#set style fill transparent solid 0.30 noborder\n";
    dnameplotfile << "plot \"" << dname << "\" u 1:2 with circles lc rgb \"blue\"\n";
    dnameplotfile << "\n";

    dnameplotfile.close();

    // Construct the shell-script to call gnuplot and do subsequent file conversions, if needed

    std::string dnamesh = dname+"_dognuplot.sh";

    std::ofstream dnameshfile(dnamesh);

    if ( outformat )
    {
        dnameshfile << "#!/usr/bin\n";
        dnameshfile << "gnuplot " << dnameplot << " >/dev/null 2>/dev/null\n";
    }

    else
    {
        dnameshfile << "#!/usr/bin\n";
        dnameshfile << "gnuplot " << dnameplot << "\n";
    }

    if ( outformat == 2 )
    {
        ;
#ifdef DOCROP
        //dnameshfile << "epstopdf " << fname << ".ps  >/dev/null 2>/dev/null\n";
        dnameshfile << "pdfcrop  " << fname << ".pdf >/dev/null 2>/dev/null\n";
        dnameshfile << "mv " << fname << "-crop.pdf " << fname << ".pdf\n";
#endif
    }

    dnameshfile.close();

    // Call the shell script with fingers crossed

    int res = 0;

    if ( PARSYSCALL )
    {
//#ifndef ENABLE_THREADS
        std::string doplotcomm = "bash ./"+dnamesh+" &";

#ifndef USE_MEX
        res = svm_system(doplotcomm.c_str());
#endif
//#endif
//#ifdef ENABLE_THREADS
//        std::string *doplotcomm = new std::string("bash ./"+dnamesh);
//
//#ifndef USE_MEX
//        std::thread(svm_system,(*doplotcomm).c_str());
//#endif
//#endif
    }

    else
    {
        std::string doplotcomm = "bash ./"+dnamesh;

#ifndef USE_MEX
        res = svm_system(doplotcomm.c_str());
#endif
    }

    // Clean up

#ifndef USE_MEX
#ifdef DO_CLEANUP
    std::string delstringa = "rm "+dnameplot;
    std::string delstringb = "rm "+dnamesh;

    svm_system(delstringa.c_str());
    svm_system(delstringb.c_str());
#endif
#endif

    return res;
}





const char *dashtypes(int j)
{
    NiceAssert( j >= 1 );

    // Firt 5 gnuplot dash types are sane, after that we need to work
    // These are kinda random, but the main thing is they can be distinguished where blue/purple cannot be

         if ( j == 1 ) { return "1"; }
    else if ( j == 2 ) { return "2"; }
    else if ( j == 3 ) { return "3"; }
    else if ( j == 4 ) { return "4"; }
    else if ( j == 5 ) { return "5"; }
    else if ( j == 6 ) { return "\'-_\'"; }
    else if ( j == 7 ) { return "\'-._\'"; }
    else if ( j == 8 ) { return "\'--.\'"; }
    else if ( j == 9 ) { return "\'__.\'"; }
    else if ( j == 10 ) { return "\'..-.-\'"; }
    else if ( j == 11 ) { return "\'.._.-\'"; }
    else if ( j == 12 ) { return "\'_-_.\'"; }
    else if ( j == 13 ) { return "\'.-..-\'"; }
    else if ( j == 14 ) { return "\'._..-\'"; }
    else if ( j == 15 ) { return "\'._..._\'"; }

    return dashtypes(j-15);
}
