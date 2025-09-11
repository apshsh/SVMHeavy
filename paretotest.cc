
//
// Pareto Test Functions as per:
//
// Deb, Thiele, Laumanns, Zitzler (DTLZ)
// Scalable Test Problems for Evolutionary Multiobjective Optimisation
//
// Version: 
// Date: 1/12/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "paretotest.hpp"
#include "numbase.hpp"
#include <math.h>

int evalTestFn(int fnnum, int n, int M, Vector<double> &res, const Vector<double> &xx, double alpha)
{
    Vector<double> x(xx);

    int nonfeas = 0;
    int i,j;

    Vector<double> resshift(M);
    Vector<double> resscale(M);

    resshift = 0.0;
    resscale = 1.0;

    NiceAssert( M >= 1 );

    res.resize(M);

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retVector<double> tmpvc;

    const Vector<double> &xL = x(0,1,M-2,tmpva);
    const Vector<double> &xM = x(M-1,1,n-1,tmpvb);

    switch ( fnnum )
    {
        case 2001:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1001:
        {
            x *= 0.5;
            x += 0.5;

            // where: z = [ xM xM+1 ... xn-1 ]
            // g(z) = 100.( #(z) + sum_i ( (zi-0.5)^2 - cos(20*pi*(zi-0.5)) ) )
            //
            // g(z) <= 100.(n-m+1).(1+1/4+1) = 100.(n-M+1).2.25 = 225.(n-M+1)
            // (1+g(z))/2 <= (1+(225.(n-M+1)))/2

            resscale = 2.0/(1.0+(225.0*(n-M+1)));
            resshift = 0.0;
        }
        // fall through
        case 1:
        {
            NiceAssert( M <= n );

            double g = n-M+1;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5)) - cos(20*NUMBASE_PI*(xM(i)-0.5));
            }

            g *= 100;

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;
                res("&",i) *= prod(xL(0,1,M-2-i,tmpvc));
                res("&",i) *= ( i ? (1-xL(M-1-i)) : 1 );
                res("&",i) *= 1+g;
                res("&",i) /= 2;
            }

            break;
        }

        case 2002:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1002:
        {
            x *= 0.5;
            x += 0.5;

            // where: z = [ xM xM+1 ... xn-1 ]
            // g(z) = sum_i (zi-0.5)^2
            // 1+g(z) <= 1+((n-M+1)/4)

            resscale = 1.0/(1+((n-M+1)*0.25));
            resshift = 0.0;
        }
        // fall through
        case 2:
        {
            NiceAssert( M <= n );

            double g = 0;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5));
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        res("&",i) *= cos(NUMBASE_PION2*xL(j));
                    }
                }

                res("&",i) *= ( i ? sin(NUMBASE_PION2*xL(M-1-i)) : 1 );
                res("&",i) *= 1+g;
            }

            break;
        }

        case 2003:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1003:
        {
            x *= 0.5;
            x += 0.5;

            // where: z = [ xM xM+1 ... xn-1 ]
            // g(z) = 100.( #(z) + sum_i ( (zi-0.5)^2 - cos(20*pi*(zi-0.5)) ) )
            // g(z) <= 100.(n-m+1).(1+1/4+1) = 100.(n-M+1).2.25 = 225.(n-M+1)
            // (1+g(z))/2 <= (1+(225.(n-M+1)))/2

            resscale = 1.0/(1.0+(225.0*(n-M+1)));
            resshift = 0.0;
        }
        // fall through
        case 3:
        {
            NiceAssert( M <= n );

            double g = n-M+1;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5)) - cos(20*NUMBASE_PI*(xM(i)-0.5));
            }

            g *= 100;

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        res("&",i) *= cos(NUMBASE_PION2*xL(j));
                    }
                }

                res("&",i) *= ( i ? sin(NUMBASE_PION2*xL(M-1-i)) : 1 );
                res("&",i) *= 1+g;
            }

            break;
        }

        case 2004:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1004:
        {
            x("&",0) *= 0.5;
            x("&",0) += 0.5;

            retVector<double> tmpvaa;

            x("&",1,1,M-1,tmpvaa) *= 5.0;

            // where: z = [ xM xM+1 ... xn-1 ] in [-5,5]
            // g(z) = sum_i (zi-0.5)^2 <= ((n-M+1)*5.5)^2
            // 1+g(z) <= 1+(n-M+1)*5.5^2

            resscale = 1.0/(1+((n-M+1)*5.5*5.5));
            resshift = 0.0;
        }
        // fall through
        case 4:
        {
            NiceAssert( M <= n );

            double g = 0;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5));
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        res("&",i) *= cos(NUMBASE_PION2*pow(xL(j),alpha));
                    }
                }

                res("&",i) *= ( i ? sin(NUMBASE_PION2*pow(xL(M-1-i),alpha)) : 1 );
                res("&",i) *= 1+g;
            }

            break;
        }

        case 2005:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1005:
        {
            x *= 0.5;
            x += 0.5;

            // where: z = [ xM xM+1 ... xn-1 ] in [-1,1]
            // g(z) = sum_i (zi-0.5)^2 <= (n-M+1)*0.25
            // 1+g(z) <= 1+(n-M+1)*0.25

            resscale = 1.0/(1+((n-M+1)*0.25));
            resshift = 0.0;
        }
        // fall through
        case 5:
        {
            NiceAssert( M <= n );

            double g = 0;
            double theta;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5));
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(j)));

                        res("&",i) *= cos(NUMBASE_PION2*theta);
                    }
                }

                if ( i )
                {
                    theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(M-1-i)));

                    res("&",i) *= sin(NUMBASE_PION2*theta);
                }

                res("&",i) *= 1+g;
            }

            break;
        }

        case 2006:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1006:
        {
            x *= 0.5;
            x += 0.5;

            // where: z = [ xM xM+1 ... xn-1 ] in [-1,1]
            // g(z) = sum_i zi^2 <= (n-M+1)
            // 1+g(z) <= 1+(n-M+1)

            resscale = 1.0/(1+(n-M+1));
            resshift = 0.0;
        }
        // fall through
        case 6:
        {
            NiceAssert( M <= n );

            double g = 0;
            double theta;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += pow(xM(i),0.1);
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(j)));

                        res("&",i) *= cos(NUMBASE_PION2*theta);
                    }
                }

                if ( i )
                {
                    theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(M-1-i)));

                    res("&",i) *= sin(NUMBASE_PION2*theta);
                }

                res("&",i) *= 1+g;
            }

            break;
        }

        case 2007:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1007:
        {
            x *= 0.5;
            x += 0.5;

            // where: z = [ xM xM+1 ... xn-1 ]
            // g(z) = 1 + (9/#(z)) sum_i zi <= 10
            // h(z) <= M
            // (1+g).h <= 10.M

            resscale = 1.0/(10*M);
            resshift = 0.0;
        }
        // fall through
        case 7:
        {
            NiceAssert( M <= n );

            double g = 1 + (9*sum(xM)/((double) (n-M+1)));
            double h = M;

            if ( M > 1 )
            {
                for ( i = 0 ; i < M-1 ; i++ )
                {
                    res("&",i) = xL(i);

                    h -= (res(i)/(1+g))*(1+sin(3*NUMBASE_PI*res(i)));
                }
            }

            res("&",M-1) = (1+g)*h;

            break;
        }

        case 2008:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1008:
        {
            x *= 0.5;
            x += 0.5;

            resscale = 1.0/(n/((double) M));
            resshift = 0.0;
        }
        // fall through
        case 8:
        {
            NiceAssert( M < n );

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) = 0;

                for ( j = (int) ((i*n)/((double) M))-1 ; j < (int) (((i+1)*n)/((double) M))-1 ; j++ )
                {
                    // NB: there appears to be a typo in DTLZ here.  They run
                    //     indices from 1..n for decision space, 1..M for
                    //     target space.  But if you consider the lower
                    //     bound on the sum for fj evaluations in 6.26 (and
                    //     6.27) that is:
                    //
                    //     floor((j-1)*(n/M)) = floor(0) = 0
                    //
                    //     which is outside the range of their indices (in our
                    //     ranging 0..n-1 this translates to index -1).  My
                    //     guess is that we just don't include the part of
                    //     the sum that lies outside the range - hence the
                    //     following if statement.

                    NiceAssert( j >= -1 );
                    NiceAssert( j < n );

                    if ( j >= 0 )
                    {
                        res("&",i) += x(j);
                    }
                }

                res *= 1.0/((double) ((int) (n/((double) M))));
            }

            // Feasibility tests

            double g;

            if ( M > 1 )
            {
                for ( j = 0 ; ( j < M-1 ) && !nonfeas ; j++ )
                {
                    g = res(M-1) + (4*res(j)) - 1;

                    if ( g < 0 )
                    {
                        nonfeas = 1;
                    }
                }
            }

            if ( !nonfeas )
            {
                double temp,minsum = 0;

                if ( M > 2 )
                {
                    for ( i = 0 ; i < M-1 ; i++ )
                    {
                        for ( j = 0 ; j < M-1 ; j++ )
                        {
                            temp = res(i)+res(j);

                            if ( ( i != j ) && ( ( !i && ( j == 1 ) ) || ( temp < minsum ) ) )
                            {
                                minsum = temp;
                            }
                        }
                    }
                }

                g = (2*res(M-1)) + minsum - 1;

                if ( g < 0 )
                {
                    nonfeas = 1;
                }
            }

            break;
        }

        case 2009:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1009:
        {
            x *= 0.5;
            x += 0.5;

            resscale = 1.0/(n/((double) M));
            resshift = 0.0;
        }
        // fall through
        case 9:
        {
            NiceAssert( M < n );

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) = 0;

                for ( j = (int) ((i*n)/((double) M))-1 ; j < (int) (((i+1)*n)/((double) M))-1 ; j++ )
                {
                    // NB: there appears to be a typo in DTLZ here.  They run
                    //     indices from 1..n for decision space, 1..M for
                    //     target space.  But if you consider the lower
                    //     bound on the sum for fj evaluations in 6.26 (and
                    //     6.27) that is:
                    //
                    //     floor((j-1)*(n/M)) = floor(0) = 0
                    //
                    //     which is outside the range of their indices (in our
                    //     ranging 0..n-1 this translates to index -1).  My
                    //     guess is that we just don't include the part of
                    //     the sum that lies outside the range - hence the
                    //     following if statement.

                    NiceAssert( j >= -1 );
                    NiceAssert( j < n );

                    if ( j >= 0 )
                    {
                        res("&",i) += pow(x(j),0.1);
                    }
                }
            }

            // Feasibility tests

            double g;

            if ( M > 1 )
            {
                for ( j = 0 ; ( j < M-1 ) && !nonfeas ; j++ )
                {
                    g = (res(M-1)*res(M-1)) + (res(j)*res(j)) - 1;

                    if ( g < 0 )
                    {
                        nonfeas = 1;
                    }
                }
            }

            break;
        }

        case 2010:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1010:
        {
            x *= 4.0;

            // [ 1-exp(-|| x - 1/sqrt(n) ||^2) <= 1 ]
            // [ 1-exp(-|| x + 1/sqrt(n) ||^2) <= 1 ]

            resscale = 1.0;
            resshift = 0.0;
        }
        // fall through
        case 10:
        {
            // FON

            NiceAssert( M == 2 );

            res("&",0) = 1-exp( -sqsum(x)+(2*sum(x)/sqrt((double) n))-1 );
            res("&",1) = 1-exp( -sqsum(x)-(2*sum(x)/sqrt((double) n))-1 );

            break;
        }

        case 2011:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1011:
        {
            x *= 5.0; // so now [-5,5]

            // [ x^2     <= 25 ]
            // [ (x-2)^2 <= 49 ]

            resscale("&",0) = 1.0/25.0;
            resscale("&",1) = 1.0/49.0;
            resshift = 0.0;
        }
        // fall through
        case 11:
        {
case11:
            // SCH1

            NiceAssert( n == 1 );
            NiceAssert( M == 2 );

            res("&",0) = x(0)*x(0);
            res("&",1) = (x(0)-2.0)*(x(0)-2.0);

            break;
        }

        case 2012:
        {
            x -= 0.5;
            x *= 2.0;
        }
        // fall through
        case 1012:
        {
            x *= 7.5;
            x += 2.5;

            // [ { -x     if     x <= 1 in [ -1,5 ] }              ]
            // [ { x-2    if 1 < x <= 3 in [ -1,1 ] } in [ -1,6 ]  ]
            // [ { 4-x    if 3 < x <= 4 in [  0,1 ] }              ]
            // [ { x-4    if 4 < x      in [  0,6 ] }              ]
            // [                                                   ]
            // [ (x-5)^2                              in [ 0,100 ] ]

            resscale("&",0) = 1.0/7.0;
            resscale("&",1) = 0.01;
            resshift("&",0) = -1.0;
            resshift("&",1) = 0.0;
        }
        // fall through
        case 12:
        {
            // SCH2

            NiceAssert( n == 1 );
            NiceAssert( M == 2 );

                 if ( x(0) <= 1 ) { res("&",0) = -x(0);    }
            else if ( x(0) <= 3 ) { res("&",0) = x(0)-2.0; }
            else if ( x(0) <= 4 ) { res("&",0) = 4.0-x(0); }
            else                  { res("&",0) = x(0)-4.0; }

            res("&",1) = (x(0)-5.0)*(x(0)-5.0);

            break;
        }

        default:
        {
            goto case11;

            break;
        }
    }

    res -= resshift;
    res *= resscale;

    res *= -1.0; // BECAUSE WE ARE MINIMISING, NOT MAXIMISING

    return nonfeas;
}

