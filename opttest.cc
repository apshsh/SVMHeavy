
//
// Optimisation test functions as per wikipedia (see opttest.pdf)
//
// Version:
// Date: 1/12/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "opttest.hpp"
#include "numbase.hpp"
#include "matrix.hpp"
#include <math.h>

int evalTestFn(int fnnum, double &res, const Vector<double> &xxx, const Matrix<double> *a)
{
    Vector<double> x(xxx);

    retVector<double> tmpva;
    retVector<double> tmpvb;

    int nonfeas = 0;
    int i;
    int n = x.size();

    double resshift = 0;
    double resscale = 1;

    double xx = 0;
    double yy = 0;
    double tt = 0;

    if ( n >= 1 ) { xx = x(0); }
    if ( n >= 2 ) { yy = x(1); }
    if ( n >= 3 ) { tt = x(2); }

    const static double A1 = 10;

    const static double A2 = 20;
    const static double B2 = 0.2;
    const static double C2 = 2*NUMBASE_PI;

    const static double gamma22 = 1/(5*sqrt(2));

    const static double AA26[5][2] = { {3,5}, {5,2}, {2,1}, {1,4}, {7,9} };
    const static double cc26[5] = { 1,2,5,2,3 };
    const static int m26 = 5;

    const static double b37 = 5.1/(4*NUMBASE_PI*NUMBASE_PI);
    const static double c37 = 5/NUMBASE_PI;
    const static double r37 = 6;
    const static double s37 = 1/(8*NUMBASE_PI);

    const static double beta34 = 0.5;

    res = 0;
    // at end: res = (res-resshift)*resscale);

    // Notes: fall through comments included to make gcc shutup about it

    switch ( fnnum )
    {
        // Rastrigin function
        case 2001: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1001: { x *= 5.12; resscale = 2.0/(80*n); resshift = 0.0; /* tested */ }
        // fall through
        case 1:
        {
            res = A1*n;
            for ( i = 0 ; i < n ; ++i ) { res += ((x(i)*x(i))-(A1*cos(2*NUMBASE_PI*x(i)))); }
            break;
        }

        // Ackley's function
        case 2002: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1002: { x *= 5.12; resscale = 1.0/15.0; resshift = 0.0; }
        // fall through
        case 2:
        {
            for ( i = 0 ; i < n ; ++i ) { res += cos(C2*x(i)); }
            res = -(A2*exp(-B2*sqrt(norm2(x)/n)))-exp(res/n)+A2+NUMBASE_E;
            break;
        }

        // Sphere function
        case 2003: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1003: { x *= 5.12; /* arbitrarily put range -2,2 following wikipedia graph */ resscale = 1.0/(5.12*5.12*n); /* tested */ resshift = 0.0; }
        // fall through
        case 3:
        {
            res = norm2(x);
            break;
        }

        // Rosenbrock function
        case 2004: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1004: { x *= 3.0; /* see 1003 */ resscale = 1.0/(2500.0*5.7664*(n-1)); /*1.0/2500.0; */ resshift = 0.0; }
        // fall through
        case 4:
        {
            NiceAssert( n > 1 );
            for ( i = 0 ; i < n-1 ; ++i ) { res += ((100*(x(i+1)-(x(i)*x(i)))*(x(i+1)-(x(i)*x(i))))+((x(i)-1)*(x(i)-1))); }
            break;
        }

        // Beale function
        case 2005: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1005: { x *= 4.5; resscale = 1.0/200000.0; resshift = 0.0; }
        // fall through
        case 5:
        {
            NiceAssert( n == 2 );
            res  = (1.5-xx+(xx*yy))*(1.5-xx+(xx*yy));
            res += (2.25-xx+(xx*yy*yy))*(2.25-xx+(xx*yy*yy));
            res += (2.625-xx+(xx*yy*yy*yy))*(2.625-xx+(xx*yy*yy*yy));
            break;
        }

        // Goldstein–Price function
        case 2006: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1006: { x *= 2.0; resscale = 1.0/1000000.0; resshift = 3.0; }
        // fall through
        case 6:
        {
            NiceAssert( n == 2 );
            res  = (1+((xx+yy+1)*(xx+yy+1)*(19-(14*xx)+(3*xx*xx)-(14*yy)+(6*xx*yy)+(3*yy*yy))));
            res *= (30+(((2*xx)-(3*yy))*((2*xx)-(3*yy))*(18-(32*xx)+(12*xx*xx)+(48*yy)-(26*xx*yy)+(27*yy*yy))));
            break;
        }

        // Booth's function
        case 2007: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1007: { x *= 10.0; resscale = 1.0/2500.0; resshift = 0.0; }
        // fall through
        case 7:
        {
            NiceAssert( n == 2 );
            res  = ((xx+(2*yy)-7)*(xx+(2*yy)-7));
            res += (((2*xx)+yy-5)*((2*xx)+yy-5));
            break;
        }

        // Bukin function N.6
        case 2008: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1008: { NiceAssert( n == 2 ); x("&",0) *= 3.0; x("&",1) *= 10.0; x("&",1) -= 5.0; resscale = 1.0/390.0; /*1.0/(250*0.704);*/ resshift = 0.0; }
        // fall through
        case 8:
        {
            NiceAssert( n == 2 );
            res = (100*sqrt(abs2(yy-(0.01*xx*xx)))) + (0.01*abs2(xx+10));
            break;
        }

        // Matyas function
        case 2009: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1009: { x *= 10.0; resscale = 1.0/100.0; resshift = 0.0; }
        // fall through
        case 9:
        {
            NiceAssert( n == 2 );
            res = (0.26*((xx*xx)+(yy*yy)))-(0.48*xx*yy);
            break;
        }

        // Levi function N.13
        case 2010: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1010: { x *= 10.0; resscale = 1.0/450.0; resshift = 0.0; }
        // fall through
        case 10:
        {
            NiceAssert( n == 2 );
            res  = (sin(3*NUMBASE_PI*xx)*sin(3*NUMBASE_PI*xx));
            res += (xx-1)*(xx-1)*(1+(sin(3*NUMBASE_PI*yy)*sin(3*NUMBASE_PI*yy)));
            res += (yy-1)*(yy-1)*(1+(sin(2*NUMBASE_PI*yy)*sin(2*NUMBASE_PI*yy)));
            break;
        }

        // Himmelblau's function
        case 2011: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1011: { x *= 5.0; resscale = 1.0/890.0; /*1.0/2000.0;*/ resshift = 0.0; }
        // fall through
        case 11:
        {
            NiceAssert( n == 2 );
            res  = (((xx*xx)+yy-11)*((xx*xx)+yy-11)); // (30-11)*(30-11) = 361
            res += ((xx+(yy*yy)-7)*(xx+(yy*yy)-7));   // 23*23 = 529
            break;
        }

        // Three-hump camel function
        case 2012: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1012: { x *= 5.0; resscale = 1.0/2000.0; resshift = 0.0; }
        // fall through
        case 12:
        {
            NiceAssert( n == 2 );
            res = (2*xx*xx)-(1.05*xx*xx*xx*xx)+(xx*xx*xx*xx*xx*xx/6)+(xx*yy)+(yy*yy);
            break;
        }

        // Easom function
        case 2013: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1013: { x *= 100.0; resscale = 1.0; resshift = -1.0; }
        // fall through
        case 13:
        {
            NiceAssert( n == 2 );
            res = -cos(xx)*cos(yy)*exp(-((xx-NUMBASE_PI)*(xx-NUMBASE_PI))-((yy-NUMBASE_PI)*(yy-NUMBASE_PI)));
            break;
        }

        // Cross-in-tray function
        case 2014: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1014: { x *= 10.0; resscale = 1.0/2.0; /*2.0;*/ resshift = -2.095222; /*-2.06261;*/ }
        // fall through
        case 14:
        {
            NiceAssert( n == 2 );
            res = -0.0001*pow(abs2(sin(xx)*sin(yy)*exp(abs2(100-(abs2(x)/NUMBASE_PI))))+1,0.1);
            break;
        }

        // Eggholder function
        case 2015: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1015: { x *= 512.0; resscale = 1.0/2000.0; resshift = -959.6407; }
        // fall through
        case 15:
        {
            NiceAssert( n == 2 );
            res  = -(yy+47)*sin(sqrt(abs2((xx/2)+yy+47)));
            res -= xx*sin(sqrt(abs2(xx-(yy+47))));
            break;
        }

        // Holder table function
        case 2016: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1016: { x *= 10.0; resscale = 1.0/20.0; resshift = -19.2085; }
        // fall through
        case 16:
        {
            NiceAssert( n == 2 );
            res = -abs2(sin(xx)*cos(yy)*exp(abs2(1-(abs2(x)/NUMBASE_PI))));
            break;
        }

        // McCormick function
        case 2017: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1017: { x *= 3.0; resscale = 1.0/(42.0*1.2122214285714286); resshift = -1.9133; }
        // fall through
        case 17:
        {
            NiceAssert( n == 2 );
            res = sin(xx+yy)+((xx-yy)*(xx-yy))-(1.5*xx)+(2.5*yy)+1;
            break;
        }

        // Schaffer function N. 2
        case 2018: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1018: { x *= 100.0; resscale = 1.0; resshift = 0.0; }
        // fall through
        case 18:
        {
            NiceAssert( n == 2 );
            res = 0.5+(((sin((xx*xx)-(yy*yy))*sin((xx*xx)-(yy*yy)))-0.5)/((1+(0.001*((xx*xx)+(yy*yy))))*(1+(0.001*((xx*xx)+(yy*yy))))));
            break;
        }

        // Schaffer function N. 4
        case 2019: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1019: { x *= 100.0; resscale = 1.0/0.7; resshift = 0.292579; }
        // fall through
        case 19:
        {
            NiceAssert( n == 2 );
            res = 0.5+(((cos(sin((xx*xx)-(yy*yy)))*cos(sin((xx*xx)-(yy*yy))))-0.5)/((1+(0.001*((xx*xx)+(yy*yy))))*(1+(0.001*((xx*xx)+(yy*yy))))));
            break;
        }

        // Styblinski–Tang function
        case 2020: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1020: { x *= 5.0; resscale = 1.0/((125+39)*n); /* 2.0/(250*n); .. 1.0/250.0; .. this is probably wrong */ resshift = -39.16616*n; }
        // fall through
        case 20:
        {
            for ( i = 0 ; i < n ; ++i ) { res += (x(i)*x(i)*x(i)*x(i))-(16*x(i)*x(i))+(5*x(i)); } // max: x=-5,5: y=5^4-16*5^2+5*5 = 625-400+25 = 250   :: min -.39
            res /= 2; //[39n  , 125n]
            break;
        }

        // Stability test function 1
        case 2021: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1021: { x *= 2.0; x -= 1.0; resscale = 1.0; /* nominal */ resshift = -1.3; }
        // fall through
        case 21:
        {
            NiceAssert( n == 1 );
            res = exp(-20*(xx-0.2)*(xx-0.2))+exp(-20*sqrt(0.00001+((xx-0.5)*(xx-0.5))))+exp(2*(xx-0.8));
            break;
        }

        // Stability test function 2
        case 2022: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1022: { x *= 2.0; x -= 1.0; resscale = 1.0; /* nominal */ resshift = -1.0; }
        // fall through
        case 22:
        {
            NiceAssert( n == 1 );
            res = (4*exp(-(xx-1)*(xx-1)/(2*gamma22*gamma22))) + exp(-(xx-0.5)*(xx-0.5)/(2*gamma22*gamma22));
            break;
        }

        // 23: Test function 3: f(x) = sum_i a_{i,0} exp(-||x-x_{i,2:...}||_2^2/(2*a_{i,1}*a_{i,1}))
        case 2023: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1023: { x *= 2.0; x -= 1.0; resscale = 1.0; /* nominal */ resshift = 0.0; /* nominal */ }
        // fall through
        case 23:
        {
            const Matrix<double> &aa = *a;
            const int m = aa.numRows();
            NiceAssert( a );
            NiceAssert( n = aa.numCols()-2 );
            for ( i = 0 ; ( i < m ) && n ; ++i )
            {
                double alpha = aa(i,0);
                double gamma = aa(i,1);
                const Vector<double> &xa = aa(i,2,1,2+n-1,tmpva,tmpvb);
                res += alpha*exp(-norm2(x-xa)/(2*gamma*gamma));
            }
            break;
        }

        // Dropwave function
        case 2024: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1024: { x *= 5.12; resscale = 1.0; resshift = -1.0; }
        // fall through
        case 24:
        {
            res = -(1+cos(12*abs2(x)))/(2+(0.5*norm2(x)));
            break;
        }

        // Grimancy and Lee function
        case 2025: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1025: { x *= 1.0; x += 1.5; resscale = 1.0/6.0625; resshift = -1.0; }
        // fall through
        case 25:
        {
            NiceAssert( n == 1 );
            res = (sin(10*NUMBASE_PI*xx)/(2*xx))+((xx-1)*(xx-1)*(xx-1)*(xx-1));
            break;
        }

        // Langermann function
        case 2026: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1026: { x *= 5.0; x += 5.0; resscale = 1.0/(-3.059239022594964+3.1665725749005245); resshift = -3.1665725749005245; }
        // fall through
        case 26:
        {
            NiceAssert( n == 2 );
            for ( i = 0 ; i < m26 ; i++ )
            {
                double s0 = (xx-AA26[i][0])*(xx-AA26[i][0]);
                double s1 = (yy-AA26[i][1])*(yy-AA26[i][1]);
                res += cc26[i]*exp(-(s0+s1)/NUMBASE_PI)*cos((s0+s1)*NUMBASE_PI);
            }
            break;
        }

        // Griewank function
        case 2027: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1027: { x *= 600.0; resscale = 1.0/(90*n); /* 2.0/(100*n); //1.0/200; */ resshift = -1.0; /* 0.0; */ }
        // fall through
        case 27:
        {
            res = -1; for ( i = 0 ; i < n ; i++ ) { res *= cos(x(i)/sqrt(i+1)); }
            res += 1; for ( i = 0 ; i < n ; i++ ) { res += (x(i)*x(i)/4000);    }
            break;
        }

        // Levy function
        case 2028: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1028: {x *= 10.0; resscale = 1.0/100; resshift = 0.0; }
        // fall through
        case 28:
        {
            for ( i = 0 ; i < n ; i++ )
            {
                double omega = 1 + ((x(i)-1)/4);
                if ( i == 0 )  { res += sin(NUMBASE_PI*omega)*sin(NUMBASE_PI*omega);                                   }
                if ( i < n-1 ) { res += (omega-1)*(omega-1)*(1+(sin(1+(NUMBASE_PI*omega))*sin(1+(NUMBASE_PI*omega)))); }
                else           { res += (omega-1)*(omega-1)*(1+(10*sin(2*NUMBASE_PI*omega)*sin(2*NUMBASE_PI*omega)));  }
            }
            break;
        }

        // Schwefel function
        case 2029: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1029: { x *= 500.0; resscale = 2.0/(1800*n); /* 1.0/1800; */ resshift = 0.0; }
        // fall through
        case 29:
        {
            res = 418.9829*n;
            for ( i = 0 ; i < n ; i++ ) { res += x(i)*sin(sqrt(sqrt(x(i)*x(i)))); }
            break;
        }

        // Shubert function
        case 2030: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1030: { x *= 10.0; resscale = 1.0/(210.48178038742196+186.72187591482512); /* 1.0/500; */ resshift = -186.72187591482512; /* 200.0; */ }
        // fall through
        case 30:
        {
            NiceAssert( n == 2 );
            double resa = 0.0;
            double resb = 0.0;
            for ( i = 1 ; i <= 5 ; i++ ) { resa += i*cos(((i+1)*xx)+i); resb += i*cos(((i+1)*yy)+i); }
            res = resa*resb;
            break;
        }

        // Bohachevsky Function 1
        case 2031: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1031: { x *= 100.0; resscale = 1.0/30070; /* 1.0/300.7; */ resshift = 0.0; }
        // fall through
        case 31:
        {
            NiceAssert( n == 2 );
            res = (xx*xx) + (2*yy*yy) - (0.3*cos(3*NUMBASE_PI*xx)) - (0.4*cos(4*NUMBASE_PI*yy)) + 0.7;
            break;
        }

        // Bohachevsky Function 2
        case 2032: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1032: { x *= 100.0; resscale = 1.0/30070; /* 1.0/300.7; */ resshift = 0.0; }
        // fall through
        case 32:
        {
            NiceAssert( n == 2 );
            res = (x(0)*x(0)) + (2*x(1)*x(1)) - (0.3*cos(3*NUMBASE_PI*x(0))*cos(4*NUMBASE_PI*x(1))) + 0.3;
            break;
        }

        // Bohachevsky Function 3
        case 2033: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1033: { x *= 100.0; resscale = 1.0/30070; /* 1.0/300.7; */ resshift = 0.0; }
        // fall through
        case 33:
        {
            NiceAssert( n == 2 );
            res = (xx*xx) + (2*yy*yy) - (0.3*cos(3*(NUMBASE_PI*xx)+(4*NUMBASE_PI*yy))) + 0.3;
            break;
        }

        // Perm function 0,D,1
        case 2034: { x -= 0.5; x *= 2.0; }
        // fall through
        case 1034: { x *= (double) n; resscale = 1.0/12000.0; resshift = 0.0; }
        // fall through
        case 34:
        {
            for ( i = 0 ; i < n ; i++ )
            {
                double tmpres = 0;
                for ( int j = 0 ; j < n ; j++ ) { tmpres += ( pow(j+1.0,i+1.0) + beta34 )*( pow(x(j)/(j+1.0),i+1.0) - 1); }
                res += (tmpres*tmpres);
            }
            break;
        }

        // Currin muti-fidelity function
        case 2035: { x("&",0,1,1,tmpva) -= 0.5; x("&",0,1,1,tmpva) *= 2.0; }
        // fall through
        case 1035: { x("&",0,1,1,tmpva) /= 2.0; x("&",0,1,1,tmpva) += 0.5; resscale = 1.0/10.8; resshift = 0.0; }
        // fall through
        case 35:
        {
            NiceAssert( n == 3 );
            yy = ( ( yy > -1e-12 ) && ( yy < 1e-12 ) ) ? 1e-12 : yy;
            res = 13.8-( (1-(0.1*(1-tt)*exp(-1/(2*yy)))) * (((2300*xx*xx*xx)+(1900*xx*xx)+(2092*xx)+60)/((100*xx*xx*xx)+(500*xx*xx)+(4*xx)+20)) );
            break;
        }

        // Park muti-fidelity function
        case 2036: { x("&",0,1,1,tmpva) -= 0.5; x("&",0,1,1,tmpva) *= 2.0; }
        // fall through
        case 1036: { x("&",0,1,1,tmpva) /= 2.0; x("&",0,1,1,tmpva) += 0.5; resscale = 1.0/2.5876125; /* Simple calculation */ resshift = 0.0; }
        // fall through
        case 36:
        {
            NiceAssert( n == 3 );
            res = (((xx+(tt/2))*(xx+(tt/2)))+((yy+(tt/2))+(yy+(tt/2))))/2;
            break;
        }

        // Brannin muti-fidelity function
        case 2037: { x("&",0,1,1,tmpva) -= 0.5; x("&",0,1,1,tmpva) *= 2.0; }
        // fall through
        case 1037: { x("&",0,1,1,tmpva) /= 2.0; x("&",0,1,1,tmpva) += 0.5; resscale = 1.0/(7*2.57027082826); resshift = (7*2.29717377815797); }
        // fall through
        case 37:
        {
            NiceAssert( n == 3 );
            xx = 1.5*(xx+1)/2;
            yy = 1.5*(yy+1)/2;
            tt = (tt+1)/2;
            res = yy-((b37-(0.1*(1-tt)))*xx*xx)+(c37*xx)-r37;
            res = (res*res)+(10*(1-s37)*cos(xx))+10;
            break;
        }

        default: { break; }
    }

//OLD VERSION (WHICH ALSO HAD ALL resshifts above negated):    res = (res*resscale)-resshift;
    res = (res-resshift)*resscale;

    return nonfeas;
}

