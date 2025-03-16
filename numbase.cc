
//
// Non-standard functions for reals
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "numbase.hpp"

#ifdef SPECFN_ASUPP
#include <cmath>
double numbase_gamma  (double x) { return std::tgamma(x); }
double numbase_lngamma(double x) { return std::lgamma(x); }
#endif

#ifndef SPECFN_ASUPP
int numbase_gamma_err  (double &res, double x)
int numbase_lngamma_err(double &res, double x)

double numbase_gamma(double x)
{
    double res;

    numbase_gamma_err(res,x);

    return res;
}

double numbase_lngamma(double x)
{
    double res;

    numbase_lngamma_err(res,x);

    return res;
}

int numbase_gamma_err(double &res, double x)
{
    if ( x <= 0.0 )
    {
        res = valvnan("Gamma undefined for x<=0");
        return 1;
    }

    // Split the function domain into three intervals:
    // (0, 0.001), [0.001, 12), and (12, infinity)

    ///////////////////////////////////////////////////////////////////////////
    // First interval: (0, 0.001)
    //
    // For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
    // So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
    // The relative error over this interval is less than 6e-7.

    const double gamma = NUMBASE_EULER;

    if ( x < 0.001 )
    {
        res = 1.0/(x*(1.0+(gamma*x)));
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Second interval: [0.001, 12)

    if (x < 12.0)
    {
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.

        double y = x;
        int n = 0;
        int arg_was_less_than_one = (y < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below

        if ( arg_was_less_than_one )
        {
            y += 1.0;
        }

        else
        {
            n =  ( (int) floor(y) ) - 1;  // will use n later
            y -= n;
        }

        // numerator coefficients for approximation over the interval (1,2)
        const static double p[] =
        {
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        };

        // denominator coefficients for approximation over the interval (1,2)
        const static double q[] =
        {
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        };

        double num = 0.0;
        double den = 1.0;
        int i;

        double z = y - 1;

        for ( i = 0 ; i < 8 ; ++i)
        {
            num = ( num + p[i] )*z;
            den = (den*z) + q[i];
        }

        double result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= (y-1.0);
        }
        else
        {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for (i = 0; i < n; ++i)
            {
                result *= y++; // note post-inc
            }
        }

        res = result;
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Third interval: [12, infinity)

    if (x > 171.624)
    {
        // Correct answer too large to display. Force +infinity.
        res = valpinf();
        return 2;
    }

    int retval = numbase_lngamma_err(res,x);
    res = exp(res);

    return retval;
}

int numbase_lngamma_err(double &res, double x)
{
    if ( x <= 0.0 )
    {
        res = valvnan("Log Gamma undefined for x<=0");
        return 1;
    }

    if ( x < 12.0 )
    {
        int retval = numbase_gamma_err(res,x);
        res = log(fabs(res));

        return retval;
    }

    // Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    const static double c[8] =
    {
        1.0/12.0,
        -1.0/360.0,
        1.0/1260.0,
        -1.0/1680.0,
        1.0/1188.0,
        -691.0/360360.0,
        1.0/156.0,
        -3617.0/122400.0
    };

    double z = 1.0/(x*x);
    double sum = c[7];
    int i;

    for ( i = 6 ; i >= 0 ; --i )
    {
        sum *= z;
        sum += c[i];
    }

    double series = sum/x;
    res = ( x - 0.5 )*log(x) - x + NUMBASE_HALFLOG2PI + series;

    return 0;
}
#endif


// Calculate volume of sphere of radius r in N dimensional space

double spherevol(double rsq, size_t N)
{
    double V = 0;

    // n even: vol = ((2.pi).r^2)^(n/2) / 2.4.....n
    //             = ((2.pi).r^2)^(n/2) / ((1.2.....n/2).(2^(n/2)))
    //             = prod_{i=1,2,...,n/2} (2.pi).r^2 / (2.i)
    //
    // n odd:  vol = (2.r).((2.pi).r^2)^((n-1)/2) / 1.3.....(n-1)
    //             = 2.r prod_{i=1,2,...,(n-1)/2} (2.pi).r^2 / (2.i+1)
    //
    // Note that rsq = r^2

    if ( N == 0 )
    {
        V = 1;
    }

    else if ( N == 1 )
    {
        V = 2*sqrt(rsq);
    }

    else if ( N == 2 )
    {
        V = NUMBASE_PI*rsq;
    }

    else if ( N > 2 )
    {
        size_t i;
        double Vx = 2*NUMBASE_PI*rsq;

        if ( N%2 )
        {
            // N odd

            V = 2*sqrt(rsq);

            for ( i = 1 ; i <= (N-1)/2 ; ++i )
            {
                V *= Vx/((2.0*static_cast<double>(i))+1.0);
            }
        }

        else
        {
            // N even

            V = 1;

            for ( i = 1 ; i <= N/2 ; ++i )
            {
                V *= Vx/(2.0*static_cast<double>(i));
            }
        }
    }

    return V;
}




// signed nth root

double nthrt(double x, int n)
{
    double res = 0.0;

    if ( ( ( x < 0 ) && ( n > 0 ) && ( n%2 ) ) || ( ( x < 0 ) && ( n < 0 ) && ( (-n)%2 ) ) )
    {
        // n is *odd*, so the sign of the result is the sign of x.
        // If n is negative then the result is 1/positive ver

        res = -nthrt(-x,n);
    }

    else if ( n == 0 )
    {
        // This could be ^-inf, but ^+inf seems more intuitive.
        // We let the compiler/standard do the edge case x<0 here.

        res = pow(x,valpinf());
    }

    else
    {
        res = pow(x,1.0/n);
    }

    return res;
}









// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------

// Bessel function code from somewhere on the web

/***********************************************************************
*                                                                      *
*    Program to calculate the first kind Bessel function of integer    *
*    order N, for any REAL X, using the function BESSJ(N,X).           *
*                                                                      *
* -------------------------------------------------------------------- *
*                                                                      *
*    SAMPLE RUN:                                                       *
*                                                                      *
*    (Calculate Bessel function for N=2, X=0.75).                      *
*                                                                      *
*    Bessel function of order  2 for X =  0.7500:                      *
*                                                                      *
*         Y =  0.06707400                                              *
*                                                                      *
* -------------------------------------------------------------------- *
*   Reference: From Numath Library By Tuan Dang Trong in Fortran 77.   *
*                                                                      *
*                               C++ Release 1.0 By J-P Moreau, Paris.  *
*                                        (www.jpmoreau.fr)             *
***********************************************************************/
//#include <math.h>
//#include <stdio.h>

double BESSJ0(double X);
double BESSJ1(double X);
double BESSJ(int N, double X);
double BESSY0(double X);
double BESSY1(double X);
double BESSY(int N, double X);
double BESSYP(int N, double X);


    double BESSJ0 (double X) {
/***********************************************************************
      This subroutine calculates the First Kind Bessel Function of
      order 0, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
************************************************************************/
    const double
          P1=1.0, P2=-0.1098628627E-2, P3=0.2734510407E-4,
          P4=-0.2073370639E-5, P5= 0.2093887211E-6,
          Q1=-0.1562499995E-1, Q2= 0.1430488765E-3, Q3=-0.6911147651E-5,
          Q4= 0.7621095161E-6, Q5=-0.9349451520E-7,
          R1= 57568490574.0, R2=-13362590354.0, R3=651619640.7,
          R4=-11214424.18, R5= 77392.33017, R6=-184.9052456,
          S1= 57568490411.0, S2=1029532985.0, S3=9494680.718,
          S4= 59272.64853, S5=267.8532712, S6=1.0;
    double
          AX,FR,FS,Z,FP,FQ,XX,Y, TMP;

      if (X==0.0) return 1.0;
      AX = fabs(X);
      if (AX < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))));
        TMP = FR/FS;
      }
      else {
        Z = 8./AX;
        Y = Z*Z;
        XX = AX-0.785398164;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        TMP = sqrt(0.636619772/AX)*(FP*cos(XX)-Z*FQ*sin(XX));
      }
      return TMP;
	}

    double Sign(double X, double Y);
    double Sign(double X, double Y) {
      if (Y<0.0) return (-fabs(X));
      else return (fabs(X));
    }

    double BESSJ1 (double X) {
/**********************************************************************
      This subroutine calculates the First Kind Bessel Function of
      order 1, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
***********************************************************************/
    const double  
      P1=1.0, P2=0.183105E-2, P3=-0.3516396496E-4, P4=0.2457520174E-5,
      P5=-0.240337019E-6,  P6=0.636619772,
      Q1= 0.04687499995, Q2=-0.2002690873E-3, Q3=0.8449199096E-5,
      Q4=-0.88228987E-6, Q5= 0.105787412E-6,
      R1= 72362614232.0, R2=-7895059235.0, R3=242396853.1,
      R4=-2972611.439,   R5=15704.48260,  R6=-30.16036606,
      S1=144725228442.0, S2=2300535178.0, S3=18583304.74,
      S4=99447.43394,    S5=376.9991397,  S6=1.0;

	  double AX,FR,FS,Y,Z,FP,FQ,XX, TMP;

      AX = fabs(X);
      if (AX < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))));
        TMP = X*(FR/FS);
      }
      else {
        Z = 8.0/AX;
        Y = Z*Z;
        XX = AX-2.35619491;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        TMP = sqrt(P6/AX)*(cos(XX)*FP-Z*sin(XX)*FQ)*Sign(S6,X);
      }
	  return TMP;
    }

    double BESSJ (int N, double X) {
/************************************************************************
      This subroutine calculates the first kind modified Bessel function
      of integer order N, for any REAL X. We use here the classical
      recursion formula, when X > N. For X < N, the Miller's algorithm
      is used to avoid overflows.
      ----------------------------- 
      REFERENCE:
      C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
      MATHEMATICAL TABLES, VOL.5, 1962.
*************************************************************************/
      const int IACC = 40; 
  	  const double BIGNO = 1e10,  BIGNI = 1e-10;
    
	  double TOX,BJM,BJ,BJP,SUM,TMP;
      int J, JSUM, M;

      if (N == 0) return BESSJ0(X);
      if (N == 1) return BESSJ1(X);
      if (X == 0.0) return 0.0;

      TOX = 2.0/X;
      if (X > 1.0*N) {
        BJM = BESSJ0(X);
        BJ  = BESSJ1(X);
        for (J=1; J<N; ++J) {
          BJP = J*TOX*BJ-BJM;
          BJM = BJ;
          BJ  = BJP;
        }
        return BJ;
      }
      else {
        M = (int) (2*((N+floor(sqrt(1.0*(IACC*N))))/2));
        TMP = 0.0;
        JSUM = 0;
        SUM = 0.0;
        BJP = 0.0;
        BJ  = 1.0;
        for (J=M; J>0; --J) {
          BJM = J*TOX*BJ-BJP;
          BJP = BJ;
          BJ  = BJM;
          if (fabs(BJ) > BIGNO) {
            BJ  = BJ*BIGNI;
            BJP = BJP*BIGNI;
            TMP = TMP*BIGNI;
            SUM = SUM*BIGNI;
          }
          if (JSUM != 0)  SUM += BJ;
          JSUM = 1-JSUM;
          if (J == N)  TMP = BJP;
        }
        SUM = 2.0*SUM-BJ;
        return (TMP/SUM);
      }
    }


/***********************************************************************
*                                                                      *
*    Program to calculate the second kind Bessel function of integer   *
*    order N, for any REAL X, using the function BESSY(N,X).           *
*                                                                      *
* -------------------------------------------------------------------- *
*                                                                      *
*    SAMPLE RUN:                                                       *
*                                                                      *
*    (Calculate Bessel function for N=2, X=0.75).                      *
*                                                                      *
*    Second kind Bessel function of order  2 for X =  0.7500:          *
*                                                                      *
*         Y = -2.62974604                                              *
*                                                                      *
* -------------------------------------------------------------------- *
*   Reference: From Numath Library By Tuan Dang Trong in Fortran 77.   *
*                                                                      *
*                               C++ Release 1.0 By J-P Moreau, Paris.  *
*                                        (www.jpmoreau.fr)             *
***********************************************************************/
//#include <math.h>
//#include <stdio.h>

    double BESSY0 (double X) {
/* --------------------------------------------------------------------
      This subroutine calculates the Second Kind Bessel Function of
      order 0, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
  --------------------------------------------------------------------- */
      const double
		  P1= 1.0, P2=-0.1098628627E-2, P3=0.2734510407E-4,
          P4=-0.2073370639E-5, P5= 0.2093887211E-6,
          Q1=-0.1562499995E-1, Q2= 0.1430488765E-3, Q3=-0.6911147651E-5,
          Q4= 0.7621095161E-6, Q5=-0.9349451520E-7,
          R1=-2957821389.0, R2=7062834065.0, R3=-512359803.6,
          R4= 10879881.29,  R5=-86327.92757, R6=228.4622733,
          S1= 40076544269.0, S2=745249964.8, S3=7189466.438,
          S4= 47447.26470,   S5=226.1030244, S6=1.0;
      double FS,FR,Z,FP,FQ,XX,Y;
 	  if (X == 0.0) return -1e30;
      if (X < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))));
        return (FR/FS+0.636619772*BESSJ0(X)*log(X));
      }
      else {
        Z = 8.0/X;
        Y = Z*Z;
        XX = X-0.785398164;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        return (sqrt(0.636619772/X)*(FP*sin(XX)+Z*FQ*cos(XX)));
      }
    }

    double BESSY1 (double X) {
/* ---------------------------------------------------------------------
      This subroutine calculates the Second Kind Bessel Function of
      order 1, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
  ---------------------------------------------------------------------- */
    const double
      P1= 1.0, P2=0.183105E-2, P3=-0.3516396496E-4,
      P4= 0.2457520174E-5, P5=-0.240337019E-6,
      Q1= 0.04687499995, Q2=-0.2002690873E-3, Q3=0.8449199096E-5,
      Q4=-0.88228987E-6, Q5= 0.105787412E-6,
      R1=-0.4900604943E13, R2= 0.1275274390E13, R3=-0.5153438139E11,
      R4= 0.7349264551E9,  R5=-0.4237922726E7,  R6= 0.8511937935E4,
      S1= 0.2499580570E14, S2= 0.4244419664E12, S3= 0.3733650367E10,
      S4= 0.2245904002E8,  S5= 0.1020426050E6,  S6= 0.3549632885E3, S7=1.0;
    double  FR,FS,Z,FP,FQ,XX, Y;
      if (X == 0.0) return -1e30;
      if (X < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*(S6+Y*S7)))));
        return (X*(FR/FS)+0.636619772*(BESSJ1(X)*log(X)-1.0/X));
      }
      else {
         Z = 8./X;
        Y = Z*Z;
        XX = X-2.356194491;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        return (sqrt(0.636619772/X)*(sin(XX)*FP+Z*cos(XX)*FQ));
      }
    }

	double BESSY (int N, double X) {
/* -----------------------------------------------------------------
      This subroutine calculates the second kind Bessel Function of
      integer order N, for any real X. We use here the classical
      recursive formula. 
  ------------------------------------------------------------------ */
    double TOX,BY,BYM,BYP; int J;
      if (N == 0) return BESSY0(X);
      if (N == 1) return BESSY1(X);
      if (X == 0.0) return -1e30;
      TOX = 2.0/X;
      BY  = BESSY1(X);
      BYM = BESSY0(X);
      for (J=1; J<N; ++J) {
        BYP = J*TOX*BY-BYM;
        BYM = BY;
        BY  = BYP;
      };
      return BY;
	}
// --------------------------------------------------------------------------
    double BESSYP (int N, double X) {
      if (N == 0)
        return (-BESSY(1,X));
      else if (X == 0.0)
        return 1e-30;
      else
        return (BESSY(N-1,X)-(1.0*N/X)*BESSY(N,X));
    }

/***********************************************************************
*                                                                      *
*     Program to calculate the first kind modified Bessel function     *
*  of integer order N, for any REAL X, using the function BESSI(N,X).  *
*                                                                      *
* -------------------------------------------------------------------- *
*    SAMPLE RUN:                                                       *
*                                                                      *
*    (Calculate Bessel function for N=2, X=0.75).                      *
*                                                                      *
*    Bessel function of order 2 for X =  0.7500:                       *
*                                                                      *
*         Y = 0.073667                                                 *
*                                                                      *
* -------------------------------------------------------------------- *
*    Reference: From Numath Library By Tuan Dang Trong in Fortran 77.  *
*                                                                      *
*                               C++ Release 1.1 By J-P Moreau, Paris.  *
*                                        (www.jpmoreau.fr)             *
*                                                                      * 
*    Version 1.1: corected value of P4 in BESSIO (P4=1.2067492 and not *
*                 1.2067429) Aug. 2011.                                *
***********************************************************************/
#include <stdio.h>
#include <math.h>


  double BESSI0(double X);
  double BESSI1(double X);
  double BESSI(int N, double X);

// ---------------------------------------------------------------------
  double BESSI(int N, double X) {
/*----------------------------------------------------------------------
!     This subroutine calculates the first kind modified Bessel function
!     of integer order N, for any REAL X. We use here the classical
!     recursion formula, when X > N. For X < N, the Miller's algorithm
!     is used to avoid overflows. 
!     REFERENCE:
!     C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
!     MATHEMATICAL TABLES, VOL.5, 1962.
------------------------------------------------------------------------*/

      int IACC = 40; 
	  double BIGNO = 1e10, BIGNI = 1e-10;
      double TOX, BIM, BI, BIP, BSI;
      int J, M;

      if (N==0)  return (BESSI0(X));
      if (N==1)  return (BESSI1(X));
      if (X==0.0) return 0.0;

      TOX = 2.0/X;
      BIP = 0.0;
      BI  = 1.0;
      BSI = 0.0;
      M = (int) (2*((N+floor(sqrt(IACC*N)))));
      for (J = M; J>0; --J) {
        BIM = BIP+J*TOX*BI;
        BIP = BI;
        BI  = BIM;
        if (fabs(BI) > BIGNO) {
          BI  = BI*BIGNI;
          BIP = BIP*BIGNI;
          BSI = BSI*BIGNI;
        }
        if (J==N)  BSI = BIP;
      }
      return (BSI*BESSI0(X)/BI);
  }

// ----------------------------------------------------------------------
//  Auxiliary Bessel functions for N=0, N=1
  double BESSI0(double X) {
      double Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX;
      P1=1.0; P2=3.5156229; P3=3.0899424; P4=1.2067492;
      P5=0.2659732; P6=0.360768e-1; P7=0.45813e-2;
      Q1=0.39894228; Q2=0.1328592e-1; Q3=0.225319e-2;
      Q4=-0.157565e-2; Q5=0.916281e-2; Q6=-0.2057706e-1;
      Q7=0.2635537e-1; Q8=-0.1647633e-1; Q9=0.392377e-2;
      if (fabs(X) < 3.75) {
        Y=(X/3.75)*(X/3.75);
        return (P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))));
      }
      else {
        AX=fabs(X);
        Y=3.75/AX;
        BX=exp(AX)/sqrt(AX);
        AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))));
        return (AX*BX);
      }
  }

// ---------------------------------------------------------------------
  double BESSI1(double X) {
      double Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX;
      P1=0.5; P2=0.87890594; P3=0.51498869; P4=0.15084934;
      P5=0.2658733e-1; P6=0.301532e-2; P7=0.32411e-3;
      Q1=0.39894228; Q2=-0.3988024e-1; Q3=-0.362018e-2;
      Q4=0.163801e-2; Q5=-0.1031555e-1; Q6=0.2282967e-1;
      Q7=-0.2895312e-1; Q8=0.1787654e-1; Q9=-0.420059e-2;
      if (fabs(X) < 3.75) {
        Y=(X/3.75)*(X/3.75);
        return(X*(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7)))))));
      }
      else {
        AX=fabs(X);
        Y=3.75/AX;
        BX=exp(AX)/sqrt(AX);
        AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))));
        return (AX*BX);
      }
  }

static double bessk0( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n=0.  */
/*------------------------------------------------------------*/
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(-log(x/2.0)*BESSI0(x))+(-0.57721566+y*(0.42278420
         +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
         +y*(0.10750e-3+y*0.74e-5))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
         +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
         +y*(-0.251540e-2+y*0.53208e-3))))));
   }
   return ans;
}




static double bessk1( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n=1.  */
/*------------------------------------------------------------*/
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(log(x/2.0)*BESSI1(x))+(1.0/x)*(1.0+y*(0.15443144
         +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
         +y*(-0.110404e-2+y*(-0.4686e-4)))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
         +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
         +y*(0.325614e-2+y*(-0.68245e-3)))))));
   }
   return ans;
}




/*
#>            bessk.dc2

Function:     bessk

Purpose:      Evaluate Modified Bessel function Kv(x) of integer order.

Category:     MATH

File:         bessel.c

Author:       M.G.R. Vogelaar

Use:          #include "bessel.hpp"
              double   result; 
              result = bessk( int n,
                              double x )


              bessk    Return the Modified Bessel function Kv(x) of 
                       integer order for input value x.
              n        Integer order of Bessel function.
              x        Double at which the function is evaluated.

                      
Description:  bessk evaluates at x the Modified Bessel function Kv(x) of 
              integer order n.
              This routine is NOT callable in FORTRAN.

Updates:      Jun 29, 1998: VOG, Document created.
#<
*/



double bessk( int n, double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n >= 0*/
/* Note that for x == 0 the functions bessy and bessk are not */
/* defined and a blank is returned.                           */
/*------------------------------------------------------------*/
{
   int j;
   double bk,bkm,bkp,tox;

NiceAssert( n >= 0 );
NiceAssert( x > 0 );

//   if (n < 0 || x == 0.0)
//   {
//      double   dblank;
//      setdblank_c( &dblank );
//      return( dblank );
//   }
   if (n == 0)
      return( bessk0(x) );
   if (n == 1)
      return( bessk1(x) );

   tox=2.0/x;
   bkm=bessk0(x);
   bk=bessk1(x);
   for (j=1;j<n;++j) {
      bkp=bkm+j*tox*bk;
      bkm=bk;
      bk=bkp;
   }
   return bk;
}








#ifdef DJGPP_MATHS
int numbase_j0(double &res,        double x) { res = j0(x);      return 0; }
int numbase_j1(double &res,        double x) { res = j1(x);      return 0; }
int numbase_jn(double &res, int n, double x) { res = jn(n,x);    return 0; }
int numbase_y0(double &res,        double x) { res = y0(x);      return 0; }
int numbase_y1(double &res,        double x) { res = y1(x);      return 0; }
int numbase_yn(double &res, int n, double x) { res = yn(n,x);    return 0; }
int numbase_i0(double &res,        double x) { res = BESSI0(x);  return 0; }
int numbase_i1(double &res,        double x) { res = BESSI1(x);  return 0; }
int numbase_in(double &res, int n, double x) { res = BESSI(n,x); return 0; }
int numbase_k0(double &res,        double x) { res = bessk0(x);  return 0; }
int numbase_k1(double &res,        double x) { res = bessk1(x);  return 0; }
int numbase_kn(double &res, int n, double x) { res = bessk(n,x); return 0; }
#endif

#ifndef DJGPP_MATHS
#ifdef VISUAL_STUDIO_BESSEL
int numbase_j0(double &res,        double x) { res = _j0(x);     return 0; }
int numbase_j1(double &res,        double x) { res = _j1(x);     return 0; }
int numbase_jn(double &res, int n, double x) { res = _jn(n,x);   return 0; }
int numbase_y0(double &res,        double x) { res = _y0(x);     return 0; }
int numbase_y1(double &res,        double x) { res = _y1(x);     return 0; }
int numbase_yn(double &res, int n, double x) { res = _yn(n,x);   return 0; }
int numbase_i0(double &res,        double x) { res = BESSI0(x);  return 0; }
int numbase_i1(double &res,        double x) { res = BESSI1(x);  return 0; }
int numbase_in(double &res, int n, double x) { res = BESSI(n,x); return 0; }
int numbase_k0(double &res,        double x) { res = bessk0(x);  return 0; }
int numbase_k1(double &res,        double x) { res = bessk1(x);  return 0; }
int numbase_kn(double &res, int n, double x) { res = bessk(n,x); return 0; }
#endif

#ifndef VISUAL_STUDIO_BESSEL
int numbase_j0(double &res,        double x) { res = BESSJ0(x);  return 0; }
int numbase_j1(double &res,        double x) { res = BESSJ1(x);  return 0; }
int numbase_jn(double &res, int n, double x) { res = BESSJ(n,x); return 0; }
int numbase_y0(double &res,        double x) { res = BESSY0(x);  return 0; }
int numbase_y1(double &res,        double x) { res = BESSY1(x);  return 0; }
int numbase_yn(double &res, int n, double x) { res = BESSY(n,x); return 0; }
int numbase_i0(double &res,        double x) { res = BESSI0(x);  return 0; }
int numbase_i1(double &res,        double x) { res = BESSI1(x);  return 0; }
int numbase_in(double &res, int n, double x) { res = BESSI(n,x); return 0; }
int numbase_k0(double &res,        double x) { res = bessk0(x);  return 0; }
int numbase_k1(double &res,        double x) { res = bessk1(x);  return 0; }
int numbase_kn(double &res, int n, double x) { res = bessk(n,x); return 0; }
#endif
#endif























// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------

// Inverse error function code from somewhere on the web

#define erfinv_a3 -0.140543331
#define erfinv_a2 0.914624893
#define erfinv_a1 -1.645349621
#define erfinv_a0 0.886226899

#define erfinv_b4 0.012229801
#define erfinv_b3 -0.329097515
#define erfinv_b2 1.442710462
#define erfinv_b1 -2.118377725
#define erfinv_b0 1

#define erfinv_c3 1.641345311
#define erfinv_c2 3.429567803
#define erfinv_c1 -1.62490649
#define erfinv_c0 -1.970840454

#define erfinv_d2 1.637067800
#define erfinv_d1 3.543889200
#define erfinv_d0 1

inline int numbase_erfinv(double &res, double x)
{
    double x2,y;
    int sign_x;
 
    if ( ( x < -1 ) || ( x > 1 ) )
    {
        res = valvnan("erfinv undefined for |x|>1");
        return 1;
    }

    if ( x == 0 )
    {
        res = 0;

        return 0;
    }

    if ( x > 0 )
    {
        sign_x = 1;
    }

    else
    {
        sign_x = -1;
        x = -x;
    }

    if ( x <= 0.7 )
    {
        x2 = x*x;

        res  = x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
        res /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 + erfinv_b1) * x2 + erfinv_b0;
    }

    else
    {
        y  = sqrt (-log ((1 - x) / 2));

        res  = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
        res /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
    }

    res *= sign_x;
    x   *= sign_x;

    res -= ( erf(res) - x ) / (2 / sqrt(NUMBASE_PI) * exp(-res*res));
    res -= ( erf(res) - x ) / (2 / sqrt(NUMBASE_PI) * exp(-res*res));

    return 0;
}

























// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------


// SRC: https://www.radiativetransfer.org/misc/arts-doc/doxygen/html/Faddeeva_8cc_source.html

 /* erfcx(x) = exp(x^2) erfc(x) function, for real x, written by
    Steven G. Johnson, October 2012.
 
    This function combines a few different ideas.
 
    First, for x > 50, it uses a continued-fraction expansion (same as
    for the Faddeeva function, but with algebraic simplifications for z=i*x).
 
    Second, for 0 <= x <= 50, it uses Chebyshev polynomial approximations,
    but with two twists:
 
       a) It maps x to y = 4 / (4+x) in [0,1].  This simple transformation,
          inspired by a similar transformation in the octave-forge/specfun
          erfcx by Soren Hauberg, results in much faster Chebyshev convergence
          than other simple transformations I have examined.
 
       b) Instead of using a single Chebyshev polynomial for the entire
          [0,1] y interval, we break the interval up into 100 equal
          subintervals, with a switch/lookup table, and use much lower
          degree Chebyshev polynomials in each subinterval. This greatly
          improves performance in my tests.
 
    For x < 0, we use the relationship erfcx(-x) = 2 exp(x^2) - erfc(x),
    with the usual checks for overflow etcetera.
 
    Performance-wise, it seems to be substantially faster than either
    the SLATEC DERFC function [or an erfcx function derived therefrom]
    or Cody's CALERF function (from netlib.org/specfun), while
    retaining near machine precision in accuracy.  */
 
 /* Given y100=100*y, where y = 4/(4+x) for x >= 0, compute erfc(x).
 
    Uses a look-up table of 100 different Chebyshev polynomials
    for y intervals [0,0.01], [0.01,0.02], ...., [0.99,1], generated
    with the help of Maple and a little shell script.   This allows
    the Chebyshev polynomials to be of significantly lower degree (about 1/4)
    compared to fitting the whole [0,1] interval with a single polynomial. */


static double erfcx_y100(double y100)
 {
   switch ((int) y100) {
 case 0: {
 double t = 2*y100 - 1;
 return 0.70878032454106438663e-3 + (0.71234091047026302958e-3 + (0.35779077297597742384e-5 + (0.17403143962587937815e-7 + (0.81710660047307788845e-10 + (0.36885022360434957634e-12 + 0.15917038551111111111e-14 * t) * t) * t) * t) * t) * t;
 }
 case 1: {
 double t = 2*y100 - 3;
 return 0.21479143208285144230e-2 + (0.72686402367379996033e-3 + (0.36843175430938995552e-5 + (0.18071841272149201685e-7 + (0.85496449296040325555e-10 + (0.38852037518534291510e-12 + 0.16868473576888888889e-14 * t) * t) * t) * t) * t) * t;
 }
 case 2: {
 double t = 2*y100 - 5;
 return 0.36165255935630175090e-2 + (0.74182092323555510862e-3 + (0.37948319957528242260e-5 + (0.18771627021793087350e-7 + (0.89484715122415089123e-10 + (0.40935858517772440862e-12 + 0.17872061464888888889e-14 * t) * t) * t) * t) * t) * t;
 }
 case 3: {
 double t = 2*y100 - 7;
 return 0.51154983860031979264e-2 + (0.75722840734791660540e-3 + (0.39096425726735703941e-5 + (0.19504168704300468210e-7 + (0.93687503063178993915e-10 + (0.43143925959079664747e-12 + 0.18939926435555555556e-14 * t) * t) * t) * t) * t) * t;
 }
 case 4: {
 double t = 2*y100 - 9;
 return 0.66457513172673049824e-2 + (0.77310406054447454920e-3 + (0.40289510589399439385e-5 + (0.20271233238288381092e-7 + (0.98117631321709100264e-10 + (0.45484207406017752971e-12 + 0.20076352213333333333e-14 * t) * t) * t) * t) * t) * t;
 }
 case 5: {
 double t = 2*y100 - 11;
 return 0.82082389970241207883e-2 + (0.78946629611881710721e-3 + (0.41529701552622656574e-5 + (0.21074693344544655714e-7 + (0.10278874108587317989e-9 + (0.47965201390613339638e-12 + 0.21285907413333333333e-14 * t) * t) * t) * t) * t) * t;
 }
 case 6: {
 double t = 2*y100 - 13;
 return 0.98039537275352193165e-2 + (0.80633440108342840956e-3 + (0.42819241329736982942e-5 + (0.21916534346907168612e-7 + (0.10771535136565470914e-9 + (0.50595972623692822410e-12 + 0.22573462684444444444e-14 * t) * t) * t) * t) * t) * t;
 }
 case 7: {
 double t = 2*y100 - 15;
 return 0.11433927298290302370e-1 + (0.82372858383196561209e-3 + (0.44160495311765438816e-5 + (0.22798861426211986056e-7 + (0.11291291745879239736e-9 + (0.53386189365816880454e-12 + 0.23944209546666666667e-14 * t) * t) * t) * t) * t) * t;
 }
 case 8: {
 double t = 2*y100 - 17;
 return 0.13099232878814653979e-1 + (0.84167002467906968214e-3 + (0.45555958988457506002e-5 + (0.23723907357214175198e-7 + (0.11839789326602695603e-9 + (0.56346163067550237877e-12 + 0.25403679644444444444e-14 * t) * t) * t) * t) * t) * t;
 }
 case 9: {
 double t = 2*y100 - 19;
 return 0.14800987015587535621e-1 + (0.86018092946345943214e-3 + (0.47008265848816866105e-5 + (0.24694040760197315333e-7 + (0.12418779768752299093e-9 + (0.59486890370320261949e-12 + 0.26957764568888888889e-14 * t) * t) * t) * t) * t) * t;
 }
 case 10: {
 double t = 2*y100 - 21;
 return 0.16540351739394069380e-1 + (0.87928458641241463952e-3 + (0.48520195793001753903e-5 + (0.25711774900881709176e-7 + (0.13030128534230822419e-9 + (0.62820097586874779402e-12 + 0.28612737351111111111e-14 * t) * t) * t) * t) * t) * t;
 }
 case 11: {
 double t = 2*y100 - 23;
 return 0.18318536789842392647e-1 + (0.89900542647891721692e-3 + (0.50094684089553365810e-5 + (0.26779777074218070482e-7 + (0.13675822186304615566e-9 + (0.66358287745352705725e-12 + 0.30375273884444444444e-14 * t) * t) * t) * t) * t) * t;
 }
 case 12: {
 double t = 2*y100 - 25;
 return 0.20136801964214276775e-1 + (0.91936908737673676012e-3 + (0.51734830914104276820e-5 + (0.27900878609710432673e-7 + (0.14357976402809042257e-9 + (0.70114790311043728387e-12 + 0.32252476000000000000e-14 * t) * t) * t) * t) * t) * t;
 }
 case 13: {
 double t = 2*y100 - 27;
 return 0.21996459598282740954e-1 + (0.94040248155366777784e-3 + (0.53443911508041164739e-5 + (0.29078085538049374673e-7 + (0.15078844500329731137e-9 + (0.74103813647499204269e-12 + 0.34251892320000000000e-14 * t) * t) * t) * t) * t) * t;
 }
 case 14: {
 double t = 2*y100 - 29;
 return 0.23898877187226319502e-1 + (0.96213386835900177540e-3 + (0.55225386998049012752e-5 + (0.30314589961047687059e-7 + (0.15840826497296335264e-9 + (0.78340500472414454395e-12 + 0.36381553564444444445e-14 * t) * t) * t) * t) * t) * t;
 }
 case 15: {
 double t = 2*y100 - 31;
 return 0.25845480155298518485e-1 + (0.98459293067820123389e-3 + (0.57082915920051843672e-5 + (0.31613782169164830118e-7 + (0.16646478745529630813e-9 + (0.82840985928785407942e-12 + 0.38649975768888888890e-14 * t) * t) * t) * t) * t) * t;
 }
 case 16: {
 double t = 2*y100 - 33;
 return 0.27837754783474696598e-1 + (0.10078108563256892757e-2 + (0.59020366493792212221e-5 + (0.32979263553246520417e-7 + (0.17498524159268458073e-9 + (0.87622459124842525110e-12 + 0.41066206488888888890e-14 * t) * t) * t) * t) * t) * t;
 }
 case 17: {
 double t = 2*y100 - 35;
 return 0.29877251304899307550e-1 + (0.10318204245057349310e-2 + (0.61041829697162055093e-5 + (0.34414860359542720579e-7 + (0.18399863072934089607e-9 + (0.92703227366365046533e-12 + 0.43639844053333333334e-14 * t) * t) * t) * t) * t) * t;
 }
 case 18: {
 double t = 2*y100 - 37;
 return 0.31965587178596443475e-1 + (0.10566560976716574401e-2 + (0.63151633192414586770e-5 + (0.35924638339521924242e-7 + (0.19353584758781174038e-9 + (0.98102783859889264382e-12 + 0.46381060817777777779e-14 * t) * t) * t) * t) * t) * t;
 }
 case 19: {
 double t = 2*y100 - 39;
 return 0.34104450552588334840e-1 + (0.10823541191350532574e-2 + (0.65354356159553934436e-5 + (0.37512918348533521149e-7 + (0.20362979635817883229e-9 + (0.10384187833037282363e-11 + 0.49300625262222222221e-14 * t) * t) * t) * t) * t) * t;
 }
 case 20: {
 double t = 2*y100 - 41;
 return 0.36295603928292425716e-1 + (0.11089526167995268200e-2 + (0.67654845095518363577e-5 + (0.39184292949913591646e-7 + (0.21431552202133775150e-9 + (0.10994259106646731797e-11 + 0.52409949102222222221e-14 * t) * t) * t) * t) * t) * t;
 }
 case 21: {
 double t = 2*y100 - 43;
 return 0.38540888038840509795e-1 + (0.11364917134175420009e-2 + (0.70058230641246312003e-5 + (0.40943644083718586939e-7 + (0.22563034723692881631e-9 + (0.11642841011361992885e-11 + 0.55721092871111111110e-14 * t) * t) * t) * t) * t) * t;
 }
 case 22: {
 double t = 2*y100 - 45;
 return 0.40842225954785960651e-1 + (0.11650136437945673891e-2 + (0.72569945502343006619e-5 + (0.42796161861855042273e-7 + (0.23761401711005024162e-9 + (0.12332431172381557035e-11 + 0.59246802364444444445e-14 * t) * t) * t) * t) * t) * t;
 }
 case 23: {
 double t = 2*y100 - 47;
 return 0.43201627431540222422e-1 + (0.11945628793917272199e-2 + (0.75195743532849206263e-5 + (0.44747364553960993492e-7 + (0.25030885216472953674e-9 + (0.13065684400300476484e-11 + 0.63000532853333333334e-14 * t) * t) * t) * t) * t) * t;
 }
 case 24: {
 double t = 2*y100 - 49;
 return 0.45621193513810471438e-1 + (0.12251862608067529503e-2 + (0.77941720055551920319e-5 + (0.46803119830954460212e-7 + (0.26375990983978426273e-9 + (0.13845421370977119765e-11 + 0.66996477404444444445e-14 * t) * t) * t) * t) * t) * t;
 }
 case 25: {
 double t = 2*y100 - 51;
 return 0.48103121413299865517e-1 + (0.12569331386432195113e-2 + (0.80814333496367673980e-5 + (0.48969667335682018324e-7 + (0.27801515481905748484e-9 + (0.14674637611609884208e-11 + 0.71249589351111111110e-14 * t) * t) * t) * t) * t) * t;
 }
 case 26: {
 double t = 2*y100 - 53;
 return 0.50649709676983338501e-1 + (0.12898555233099055810e-2 + (0.83820428414568799654e-5 + (0.51253642652551838659e-7 + (0.29312563849675507232e-9 + (0.15556512782814827846e-11 + 0.75775607822222222221e-14 * t) * t) * t) * t) * t) * t;
 }
 case 27: {
 double t = 2*y100 - 55;
 return 0.53263363664388864181e-1 + (0.13240082443256975769e-2 + (0.86967260015007658418e-5 + (0.53662102750396795566e-7 + (0.30914568786634796807e-9 + (0.16494420240828493176e-11 + 0.80591079644444444445e-14 * t) * t) * t) * t) * t) * t;
 }
 case 28: {
 double t = 2*y100 - 57;
 return 0.55946601353500013794e-1 + (0.13594491197408190706e-2 + (0.90262520233016380987e-5 + (0.56202552975056695376e-7 + (0.32613310410503135996e-9 + (0.17491936862246367398e-11 + 0.85713381688888888890e-14 * t) * t) * t) * t) * t) * t;
 }
 case 29: {
 double t = 2*y100 - 59;
 return 0.58702059496154081813e-1 + (0.13962391363223647892e-2 + (0.93714365487312784270e-5 + (0.58882975670265286526e-7 + (0.34414937110591753387e-9 + (0.18552853109751857859e-11 + 0.91160736711111111110e-14 * t) * t) * t) * t) * t) * t;
 }
 case 30: {
 double t = 2*y100 - 61;
 return 0.61532500145144778048e-1 + (0.14344426411912015247e-2 + (0.97331446201016809696e-5 + (0.61711860507347175097e-7 + (0.36325987418295300221e-9 + (0.19681183310134518232e-11 + 0.96952238400000000000e-14 * t) * t) * t) * t) * t) * t;
 }
 case 31: {
 double t = 2*y100 - 63;
 return 0.64440817576653297993e-1 + (0.14741275456383131151e-2 + (0.10112293819576437838e-4 + (0.64698236605933246196e-7 + (0.38353412915303665586e-9 + (0.20881176114385120186e-11 + 0.10310784480000000000e-13 * t) * t) * t) * t) * t) * t;
 }
 case 32: {
 double t = 2*y100 - 65;
 return 0.67430045633130393282e-1 + (0.15153655418916540370e-2 + (0.10509857606888328667e-4 + (0.67851706529363332855e-7 + (0.40504602194811140006e-9 + (0.22157325110542534469e-11 + 0.10964842115555555556e-13 * t) * t) * t) * t) * t) * t;
 }
 case 33: {
 double t = 2*y100 - 67;
 return 0.70503365513338850709e-1 + (0.15582323336495709827e-2 + (0.10926868866865231089e-4 + (0.71182482239613507542e-7 + (0.42787405890153386710e-9 + (0.23514379522274416437e-11 + 0.11659571751111111111e-13 * t) * t) * t) * t) * t) * t;
 }
 case 34: {
 double t = 2*y100 - 69;
 return 0.73664114037944596353e-1 + (0.16028078812438820413e-2 + (0.11364423678778207991e-4 + (0.74701423097423182009e-7 + (0.45210162777476488324e-9 + (0.24957355004088569134e-11 + 0.12397238257777777778e-13 * t) * t) * t) * t) * t) * t;
 }
 case 35: {
 double t = 2*y100 - 71;
 return 0.76915792420819562379e-1 + (0.16491766623447889354e-2 + (0.11823685320041302169e-4 + (0.78420075993781544386e-7 + (0.47781726956916478925e-9 + (0.26491544403815724749e-11 + 0.13180196462222222222e-13 * t) * t) * t) * t) * t) * t;
 }
 case 36: {
 double t = 2*y100 - 73;
 return 0.80262075578094612819e-1 + (0.16974279491709504117e-2 + (0.12305888517309891674e-4 + (0.82350717698979042290e-7 + (0.50511496109857113929e-9 + (0.28122528497626897696e-11 + 0.14010889635555555556e-13 * t) * t) * t) * t) * t) * t;
 }
 case 37: {
 double t = 2*y100 - 75;
 return 0.83706822008980357446e-1 + (0.17476561032212656962e-2 + (0.12812343958540763368e-4 + (0.86506399515036435592e-7 + (0.53409440823869467453e-9 + (0.29856186620887555043e-11 + 0.14891851591111111111e-13 * t) * t) * t) * t) * t) * t;
 }
 case 38: {
 double t = 2*y100 - 77;
 return 0.87254084284461718231e-1 + (0.17999608886001962327e-2 + (0.13344443080089492218e-4 + (0.90900994316429008631e-7 + (0.56486134972616465316e-9 + (0.31698707080033956934e-11 + 0.15825697795555555556e-13 * t) * t) * t) * t) * t) * t;
 }
 case 39: {
 double t = 2*y100 - 79;
 return 0.90908120182172748487e-1 + (0.18544478050657699758e-2 + (0.13903663143426120077e-4 + (0.95549246062549906177e-7 + (0.59752787125242054315e-9 + (0.33656597366099099413e-11 + 0.16815130613333333333e-13 * t) * t) * t) * t) * t) * t;
 }
 case 40: {
 double t = 2*y100 - 81;
 return 0.94673404508075481121e-1 + (0.19112284419887303347e-2 + (0.14491572616545004930e-4 + (0.10046682186333613697e-6 + (0.63221272959791000515e-9 + (0.35736693975589130818e-11 + 0.17862931591111111111e-13 * t) * t) * t) * t) * t) * t;
 }
 case 41: {
 double t = 2*y100 - 83;
 return 0.98554641648004456555e-1 + (0.19704208544725622126e-2 + (0.15109836875625443935e-4 + (0.10567036667675984067e-6 + (0.66904168640019354565e-9 + (0.37946171850824333014e-11 + 0.18971959040000000000e-13 * t) * t) * t) * t) * t) * t;
 }
 case 42: {
 double t = 2*y100 - 85;
 return 0.10255677889470089531e0 + (0.20321499629472857418e-2 + (0.15760224242962179564e-4 + (0.11117756071353507391e-6 + (0.70814785110097658502e-9 + (0.40292553276632563925e-11 + 0.20145143075555555556e-13 * t) * t) * t) * t) * t) * t;
 }
 case 43: {
 double t = 2*y100 - 87;
 return 0.10668502059865093318e0 + (0.20965479776148731610e-2 + (0.16444612377624983565e-4 + (0.11700717962026152749e-6 + (0.74967203250938418991e-9 + (0.42783716186085922176e-11 + 0.21385479360000000000e-13 * t) * t) * t) * t) * t) * t;
 }
 case 44: {
 double t = 2*y100 - 89;
 return 0.11094484319386444474e0 + (0.21637548491908170841e-2 + (0.17164995035719657111e-4 + (0.12317915750735938089e-6 + (0.79376309831499633734e-9 + (0.45427901763106353914e-11 + 0.22696025653333333333e-13 * t) * t) * t) * t) * t) * t;
 }
 case 45: {
 double t = 2*y100 - 91;
 return 0.11534201115268804714e0 + (0.22339187474546420375e-2 + (0.17923489217504226813e-4 + (0.12971465288245997681e-6 + (0.84057834180389073587e-9 + (0.48233721206418027227e-11 + 0.24079890062222222222e-13 * t) * t) * t) * t) * t) * t;
 }
 case 46: {
 double t = 2*y100 - 93;
 return 0.11988259392684094740e0 + (0.23071965691918689601e-2 + (0.18722342718958935446e-4 + (0.13663611754337957520e-6 + (0.89028385488493287005e-9 + (0.51210161569225846701e-11 + 0.25540227111111111111e-13 * t) * t) * t) * t) * t) * t;
 }
 case 47: {
 double t = 2*y100 - 95;
 return 0.12457298393509812907e0 + (0.23837544771809575380e-2 + (0.19563942105711612475e-4 + (0.14396736847739470782e-6 + (0.94305490646459247016e-9 + (0.54366590583134218096e-11 + 0.27080225920000000000e-13 * t) * t) * t) * t) * t) * t;
 }
 case 48: {
 double t = 2*y100 - 97;
 return 0.12941991566142438816e0 + (0.24637684719508859484e-2 + (0.20450821127475879816e-4 + (0.15173366280523906622e-6 + (0.99907632506389027739e-9 + (0.57712760311351625221e-11 + 0.28703099555555555556e-13 * t) * t) * t) * t) * t) * t;
 }
 case 49: {
 double t = 2*y100 - 99;
 return 0.13443048593088696613e0 + (0.25474249981080823877e-2 + (0.21385669591362915223e-4 + (0.15996177579900443030e-6 + (0.10585428844575134013e-8 + (0.61258809536787882989e-11 + 0.30412080142222222222e-13 * t) * t) * t) * t) * t) * t;
 }
 case 50: {
 double t = 2*y100 - 101;
 return 0.13961217543434561353e0 + (0.26349215871051761416e-2 + (0.22371342712572567744e-4 + (0.16868008199296822247e-6 + (0.11216596910444996246e-8 + (0.65015264753090890662e-11 + 0.32210394506666666666e-13 * t) * t) * t) * t) * t) * t;
 }
 case 51: {
 double t = 2*y100 - 103;
 return 0.14497287157673800690e0 + (0.27264675383982439814e-2 + (0.23410870961050950197e-4 + (0.17791863939526376477e-6 + (0.11886425714330958106e-8 + (0.68993039665054288034e-11 + 0.34101266222222222221e-13 * t) * t) * t) * t) * t) * t;
 }
 case 52: {
 double t = 2*y100 - 105;
 return 0.15052089272774618151e0 + (0.28222846410136238008e-2 + (0.24507470422713397006e-4 + (0.18770927679626136909e-6 + (0.12597184587583370712e-8 + (0.73203433049229821618e-11 + 0.36087889048888888890e-13 * t) * t) * t) * t) * t) * t;
 }
 case 53: {
 double t = 2*y100 - 107;
 return 0.15626501395774612325e0 + (0.29226079376196624949e-2 + (0.25664553693768450545e-4 + (0.19808568415654461964e-6 + (0.13351257759815557897e-8 + (0.77658124891046760667e-11 + 0.38173420035555555555e-13 * t) * t) * t) * t) * t) * t;
 }
 case 54: {
 double t = 2*y100 - 109;
 return 0.16221449434620737567e0 + (0.30276865332726475672e-2 + (0.26885741326534564336e-4 + (0.20908350604346384143e-6 + (0.14151148144240728728e-8 + (0.82369170665974313027e-11 + 0.40360957457777777779e-13 * t) * t) * t) * t) * t) * t;
 }
 case 55: {
 double t = 2*y100 - 111;
 return 0.16837910595412130659e0 + (0.31377844510793082301e-2 + (0.28174873844911175026e-4 + (0.22074043807045782387e-6 + (0.14999481055996090039e-8 + (0.87348993661930809254e-11 + 0.42653528977777777779e-13 * t) * t) * t) * t) * t) * t;
 }
 case 56: {
 double t = 2*y100 - 113;
 return 0.17476916455659369953e0 + (0.32531815370903068316e-2 + (0.29536024347344364074e-4 + (0.23309632627767074202e-6 + (0.15899007843582444846e-8 + (0.92610375235427359475e-11 + 0.45054073102222222221e-13 * t) * t) * t) * t) * t) * t;
 }
 case 57: {
 double t = 2*y100 - 115;
 return 0.18139556223643701364e0 + (0.33741744168096996041e-2 + (0.30973511714709500836e-4 + (0.24619326937592290996e-6 + (0.16852609412267750744e-8 + (0.98166442942854895573e-11 + 0.47565418097777777779e-13 * t) * t) * t) * t) * t) * t;
 }
 case 58: {
 double t = 2*y100 - 117;
 return 0.18826980194443664549e0 + (0.35010775057740317997e-2 + (0.32491914440014267480e-4 + (0.26007572375886319028e-6 + (0.17863299617388376116e-8 + (0.10403065638343878679e-10 + 0.50190265831111111110e-13 * t) * t) * t) * t) * t) * t;
 }
 case 59: {
 double t = 2*y100 - 119;
 return 0.19540403413693967350e0 + (0.36342240767211326315e-2 + (0.34096085096200907289e-4 + (0.27479061117017637474e-6 + (0.18934228504790032826e-8 + (0.11021679075323598664e-10 + 0.52931171733333333334e-13 * t) * t) * t) * t) * t) * t;
 }
 case 60: {
 double t = 2*y100 - 121;
 return 0.20281109560651886959e0 + (0.37739673859323597060e-2 + (0.35791165457592409054e-4 + (0.29038742889416172404e-6 + (0.20068685374849001770e-8 + (0.11673891799578381999e-10 + 0.55790523093333333334e-13 * t) * t) * t) * t) * t) * t;
 }
 case 61: {
 double t = 2*y100 - 123;
 return 0.21050455062669334978e0 + (0.39206818613925652425e-2 + (0.37582602289680101704e-4 + (0.30691836231886877385e-6 + (0.21270101645763677824e-8 + (0.12361138551062899455e-10 + 0.58770520160000000000e-13 * t) * t) * t) * t) * t) * t;
 }
 case 62: {
 double t = 2*y100 - 125;
 return 0.21849873453703332479e0 + (0.40747643554689586041e-2 + (0.39476163820986711501e-4 + (0.32443839970139918836e-6 + (0.22542053491518680200e-8 + (0.13084879235290858490e-10 + 0.61873153262222222221e-13 * t) * t) * t) * t) * t) * t;
 }
 case 63: {
 double t = 2*y100 - 127;
 return 0.22680879990043229327e0 + (0.42366354648628516935e-2 + (0.41477956909656896779e-4 + (0.34300544894502810002e-6 + (0.23888264229264067658e-8 + (0.13846596292818514601e-10 + 0.65100183751111111110e-13 * t) * t) * t) * t) * t) * t;
 }
 case 64: {
 double t = 2*y100 - 129;
 return 0.23545076536988703937e0 + (0.44067409206365170888e-2 + (0.43594444916224700881e-4 + (0.36268045617760415178e-6 + (0.25312606430853202748e-8 + (0.14647791812837903061e-10 + 0.68453122631111111110e-13 * t) * t) * t) * t) * t) * t;
 }
 case 65: {
 double t = 2*y100 - 131;
 return 0.24444156740777432838e0 + (0.45855530511605787178e-2 + (0.45832466292683085475e-4 + (0.38352752590033030472e-6 + (0.26819103733055603460e-8 + (0.15489984390884756993e-10 + 0.71933206364444444445e-13 * t) * t) * t) * t) * t) * t;
 }
 case 66: {
 double t = 2*y100 - 133;
 return 0.25379911500634264643e0 + (0.47735723208650032167e-2 + (0.48199253896534185372e-4 + (0.40561404245564732314e-6 + (0.28411932320871165585e-8 + (0.16374705736458320149e-10 + 0.75541379822222222221e-13 * t) * t) * t) * t) * t) * t;
 }
 case 67: {
 double t = 2*y100 - 135;
 return 0.26354234756393613032e0 + (0.49713289477083781266e-2 + (0.50702455036930367504e-4 + (0.42901079254268185722e-6 + (0.30095422058900481753e-8 + (0.17303497025347342498e-10 + 0.79278273368888888890e-13 * t) * t) * t) * t) * t) * t;
 }
 case 68: {
 double t = 2*y100 - 137;
 return 0.27369129607732343398e0 + (0.51793846023052643767e-2 + (0.53350152258326602629e-4 + (0.45379208848865015485e-6 + (0.31874057245814381257e-8 + (0.18277905010245111046e-10 + 0.83144182364444444445e-13 * t) * t) * t) * t) * t) * t;
 }
 case 69: {
 double t = 2*y100 - 139;
 return 0.28426714781640316172e0 + (0.53983341916695141966e-2 + (0.56150884865255810638e-4 + (0.48003589196494734238e-6 + (0.33752476967570796349e-8 + (0.19299477888083469086e-10 + 0.87139049137777777779e-13 * t) * t) * t) * t) * t) * t;
 }
 case 70: {
 double t = 2*y100 - 141;
 return 0.29529231465348519920e0 + (0.56288077305420795663e-2 + (0.59113671189913307427e-4 + (0.50782393781744840482e-6 + (0.35735475025851713168e-8 + (0.20369760937017070382e-10 + 0.91262442613333333334e-13 * t) * t) * t) * t) * t) * t;
 }
 case 71: {
 double t = 2*y100 - 143;
 return 0.30679050522528838613e0 + (0.58714723032745403331e-2 + (0.62248031602197686791e-4 + (0.53724185766200945789e-6 + (0.37827999418960232678e-8 + (0.21490291930444538307e-10 + 0.95513539182222222221e-13 * t) * t) * t) * t) * t) * t;
 }
 case 72: {
 double t = 2*y100 - 145;
 return 0.31878680111173319425e0 + (0.61270341192339103514e-2 + (0.65564012259707640976e-4 + (0.56837930287837738996e-6 + (0.40035151353392378882e-8 + (0.22662596341239294792e-10 + 0.99891109760000000000e-13 * t) * t) * t) * t) * t) * t;
 }
 case 73: {
 double t = 2*y100 - 147;
 return 0.33130773722152622027e0 + (0.63962406646798080903e-2 + (0.69072209592942396666e-4 + (0.60133006661885941812e-6 + (0.42362183765883466691e-8 + (0.23888182347073698382e-10 + 0.10439349811555555556e-12 * t) * t) * t) * t) * t) * t;
 }
 case 74: {
 double t = 2*y100 - 149;
 return 0.34438138658041336523e0 + (0.66798829540414007258e-2 + (0.72783795518603561144e-4 + (0.63619220443228800680e-6 + (0.44814499336514453364e-8 + (0.25168535651285475274e-10 + 0.10901861383111111111e-12 * t) * t) * t) * t) * t) * t;
 }
 case 75: {
 double t = 2*y100 - 151;
 return 0.35803744972380175583e0 + (0.69787978834882685031e-2 + (0.76710543371454822497e-4 + (0.67306815308917386747e-6 + (0.47397647975845228205e-8 + (0.26505114141143050509e-10 + 0.11376390933333333333e-12 * t) * t) * t) * t) * t) * t;
 }
 case 76: {
 double t = 2*y100 - 153;
 return 0.37230734890119724188e0 + (0.72938706896461381003e-2 + (0.80864854542670714092e-4 + (0.71206484718062688779e-6 + (0.50117323769745883805e-8 + (0.27899342394100074165e-10 + 0.11862637614222222222e-12 * t) * t) * t) * t) * t) * t;
 }
 case 77: {
 double t = 2*y100 - 155;
 return 0.38722432730555448223e0 + (0.76260375162549802745e-2 + (0.85259785810004603848e-4 + (0.75329383305171327677e-6 + (0.52979361368388119355e-8 + (0.29352606054164086709e-10 + 0.12360253370666666667e-12 * t) * t) * t) * t) * t) * t;
 }
 case 78: {
 double t = 2*y100 - 157;
 return 0.40282355354616940667e0 + (0.79762880915029728079e-2 + (0.89909077342438246452e-4 + (0.79687137961956194579e-6 + (0.55989731807360403195e-8 + (0.30866246101464869050e-10 + 0.12868841946666666667e-12 * t) * t) * t) * t) * t) * t;
 }
 case 79: {
 double t = 2*y100 - 159;
 return 0.41914223158913787649e0 + (0.83456685186950463538e-2 + (0.94827181359250161335e-4 + (0.84291858561783141014e-6 + (0.59154537751083485684e-8 + (0.32441553034347469291e-10 + 0.13387957943111111111e-12 * t) * t) * t) * t) * t) * t;
 }
 case 80: {
 double t = 2*y100 - 161;
 return 0.43621971639463786896e0 + (0.87352841828289495773e-2 + (0.10002929142066799966e-3 + (0.89156148280219880024e-6 + (0.62480008150788597147e-8 + (0.34079760983458878910e-10 + 0.13917107176888888889e-12 * t) * t) * t) * t) * t) * t;
 }
 case 81: {
 double t = 2*y100 - 163;
 return 0.45409763548534330981e0 + (0.91463027755548240654e-2 + (0.10553137232446167258e-3 + (0.94293113464638623798e-6 + (0.65972492312219959885e-8 + (0.35782041795476563662e-10 + 0.14455745872000000000e-12 * t) * t) * t) * t) * t) * t;
 }
 case 82: {
 double t = 2*y100 - 165;
 return 0.47282001668512331468e0 + (0.95799574408860463394e-2 + (0.11135019058000067469e-3 + (0.99716373005509038080e-6 + (0.69638453369956970347e-8 + (0.37549499088161345850e-10 + 0.15003280712888888889e-12 * t) * t) * t) * t) * t) * t;
 }
 case 83: {
 double t = 2*y100 - 167;
 return 0.49243342227179841649e0 + (0.10037550043909497071e-1 + (0.11750334542845234952e-3 + (0.10544006716188967172e-5 + (0.73484461168242224872e-8 + (0.39383162326435752965e-10 + 0.15559069118222222222e-12 * t) * t) * t) * t) * t) * t;
 }
 case 84: {
 double t = 2*y100 - 169;
 return 0.51298708979209258326e0 + (0.10520454564612427224e-1 + (0.12400930037494996655e-3 + (0.11147886579371265246e-5 + (0.77517184550568711454e-8 + (0.41283980931872622611e-10 + 0.16122419680000000000e-12 * t) * t) * t) * t) * t) * t;
 }
 case 85: {
 double t = 2*y100 - 171;
 return 0.53453307979101369843e0 + (0.11030120618800726938e-1 + (0.13088741519572269581e-3 + (0.11784797595374515432e-5 + (0.81743383063044825400e-8 + (0.43252818449517081051e-10 + 0.16692592640000000000e-12 * t) * t) * t) * t) * t) * t;
 }
 case 86: {
 double t = 2*y100 - 173;
 return 0.55712643071169299478e0 + (0.11568077107929735233e-1 + (0.13815797838036651289e-3 + (0.12456314879260904558e-5 + (0.86169898078969313597e-8 + (0.45290446811539652525e-10 + 0.17268801084444444444e-12 * t) * t) * t) * t) * t) * t;
 }
 case 87: {
 double t = 2*y100 - 175;
 return 0.58082532122519320968e0 + (0.12135935999503877077e-1 + (0.14584223996665838559e-3 + (0.13164068573095710742e-5 + (0.90803643355106020163e-8 + (0.47397540713124619155e-10 + 0.17850211608888888889e-12 * t) * t) * t) * t) * t) * t;
 }
 case 88: {
 double t = 2*y100 - 177;
 return 0.60569124025293375554e0 + (0.12735396239525550361e-1 + (0.15396244472258863344e-3 + (0.13909744385382818253e-5 + (0.95651595032306228245e-8 + (0.49574672127669041550e-10 + 0.18435945564444444444e-12 * t) * t) * t) * t) * t) * t;
 }
 case 89: {
 double t = 2*y100 - 179;
 return 0.63178916494715716894e0 + (0.13368247798287030927e-1 + (0.16254186562762076141e-3 + (0.14695084048334056083e-5 + (0.10072078109604152350e-7 + (0.51822304995680707483e-10 + 0.19025081422222222222e-12 * t) * t) * t) * t) * t) * t;
 }
 case 90: {
 double t = 2*y100 - 181;
 return 0.65918774689725319200e0 + (0.14036375850601992063e-1 + (0.17160483760259706354e-3 + (0.15521885688723188371e-5 + (0.10601827031535280590e-7 + (0.54140790105837520499e-10 + 0.19616655146666666667e-12 * t) * t) * t) * t) * t) * t;
 }
 case 91: {
 double t = 2*y100 - 183;
 return 0.68795950683174433822e0 + (0.14741765091365869084e-1 + (0.18117679143520433835e-3 + (0.16392004108230585213e-5 + (0.11155116068018043001e-7 + (0.56530360194925690374e-10 + 0.20209663662222222222e-12 * t) * t) * t) * t) * t) * t;
 }
 case 92: {
 double t = 2*y100 - 185;
 return 0.71818103808729967036e0 + (0.15486504187117112279e-1 + (0.19128428784550923217e-3 + (0.17307350969359975848e-5 + (0.11732656736113607751e-7 + (0.58991125287563833603e-10 + 0.20803065333333333333e-12 * t) * t) * t) * t) * t) * t;
 }
 case 93: {
 double t = 2*y100 - 187;
 return 0.74993321911726254661e0 + (0.16272790364044783382e-1 + (0.20195505163377912645e-3 + (0.18269894883203346953e-5 + (0.12335161021630225535e-7 + (0.61523068312169087227e-10 + 0.21395783431111111111e-12 * t) * t) * t) * t) * t) * t;
 }
 case 94: {
 double t = 2*y100 - 189;
 return 0.78330143531283492729e0 + (0.17102934132652429240e-1 + (0.21321800585063327041e-3 + (0.19281661395543913713e-5 + (0.12963340087354341574e-7 + (0.64126040998066348872e-10 + 0.21986708942222222222e-12 * t) * t) * t) * t) * t) * t;
 }
 case 95: {
 double t = 2*y100 - 191;
 return 0.81837581041023811832e0 + (0.17979364149044223802e-1 + (0.22510330592753129006e-3 + (0.20344732868018175389e-5 + (0.13617902941839949718e-7 + (0.66799760083972474642e-10 + 0.22574701262222222222e-12 * t) * t) * t) * t) * t) * t;
 }
 case 96: {
 double t = 2*y100 - 193;
 return 0.85525144775685126237e0 + (0.18904632212547561026e-1 + (0.23764237370371255638e-3 + (0.21461248251306387979e-5 + (0.14299555071870523786e-7 + (0.69543803864694171934e-10 + 0.23158593688888888889e-12 * t) * t) * t) * t) * t) * t;
 }
 case 97: {
 double t = 2*y100 - 195;
 return 0.89402868170849933734e0 + (0.19881418399127202569e-1 + (0.25086793128395995798e-3 + (0.22633402747585233180e-5 + (0.15008997042116532283e-7 + (0.72357609075043941261e-10 + 0.23737194737777777778e-12 * t) * t) * t) * t) * t) * t;
 }
 case 98: {
 double t = 2*y100 - 197;
 return 0.93481333942870796363e0 + (0.20912536329780368893e-1 + (0.26481403465998477969e-3 + (0.23863447359754921676e-5 + (0.15746923065472184451e-7 + (0.75240468141720143653e-10 + 0.24309291271111111111e-12 * t) * t) * t) * t) * t) * t;
 }
 case 99: {
 double t = 2*y100 - 199;
 return 0.97771701335885035464e0 + (0.22000938572830479551e-1 + (0.27951610702682383001e-3 + (0.25153688325245314530e-5 + (0.16514019547822821453e-7 + (0.78191526829368231251e-10 + 0.24873652355555555556e-12 * t) * t) * t) * t) * t) * t;
 }
   }
   // we only get here if y = 1, i.e. |x| < 4*eps, in which case
   // erfcx is within 1e-15 of 1..
   return 1.0;
 }
 
 double numbase_erfcx(double x)
 {
   if (x >= 0) {
     if (x > 50) { // continued-fraction expansion is faster
       const double ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
       if (x > 5e7) // 1-term expansion, important to avoid overflow
         return ispi / x;
       /* 5-term expansion (rely on compiler for CSE), simplified from:
                 ispi / (x+0.5/(x+1/(x+1.5/(x+2/x))))  */
       return ispi*((x*x) * (x*x+4.5) + 2) / (x * ((x*x) * (x*x+5) + 3.75));
     }
     return erfcx_y100(400/(4+x));
   }
   else
     return x < -26.7 ? HUGE_VAL : (x < -6.1 ? 2*exp(x*x) 
                                    : 2*exp(x*x) - erfcx_y100(400/(4-x)));
 }

