
//
// Common ML functions and definitions
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _mlcommon_h
#define _mlcommon_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "sparsevector.hpp"
#include "anion.hpp"
#include "mercer.hpp"

// MINSWEIGHT: minimum "s" weight in parseline
//
// DEFAULT_ZTOL: default zero tolerance (|a| == 0 if |a| < DEFAULT_ZTOL)
// DEFAULT_EMM: default norm for m-norm optimisation
// DEFAULT_C: default C value (empirical risk tradeoff)
// DEFAULT_SIGMA: default sigma value for GPR blocks
// DEFAULT_SIGMA_CUT: default sigma value for GPR blocks sampling
// DEFAULT_MEMSIZE: default amount of memory allocated to the kernel cache by default
// DEFAULT_OPTTOL: default optimality tolerance
// DEFAULT_LR: default learning rate
// DEFAULT_MAXITCNT: default max iteration count
// DEFAULT_MAXTRAINTIME: default max training time
// DEFAULT_TRAINTIMEEND: default ending training time
// DEFAULT_MAXFAILS: default max number of "bad steps" in optimisation
// DEFAULT_MAXITERFUZZT: default max iterations in SVM fuzzy loop
// DEFAULT_LRFUZZT: default learning rate for SVM fuzzy loop
// DEFAULT_ZTFUZZT: default zero/optimality tolerance for SVM fuzzy loop
// DEFAULT_COSTFNFUZZT: default cost function for SVM fuzzy loop
// DEFAULT_MAXITERMVRANK: default max iterations in outer loop for svm_mvrank training
// DEFAULT_LRMVRANK: default learning rate in outer loop for svm_mvrank training
// DEFAULT_ZTMVRANK: default zero tolerance in outer loop for svm_mvrank training
// DEFAULT_BETARANK: default beta regularisation constant in outer loop for svm_mvrank training
// DEFAULTOUTERSTEPSCALE: default outer loop step scale
// DEFAULTOUTERTOL: default outer loop optimality tolerance
// DEFAULTOUTERMAXITCNT: default outer loop max iteration count
// DEFAULTOUTEROUTERSTEPSCALE: default outer outer loop step scale
// DEFAULTOUTEROUTERTOL: default outer outer loop optimality tolerance
// DEFAULTOUTEROUTERMAXITCNT: default outer outer loop max iteration count
// MINROWDIM: minimum row dimension in kernel cache
// ROWDIMSTEPRATIO: multiplicative increase size when rowdim is changed
// MAXBOUND: maximum upper/lower bound value for alpha
// DEFAULTEPS: default eps value used in regression in terms of default optimal tolerance
// CZEROOFFSET:
// EPSZEROOFFSET:
// MAXPASSIVECHANGE:
// OFFSET_ZERO:
// MULTINORM_OUTERSTEP:
// MULTINORM_OUTERACCUR:
// HARD_MARGIN_LINCUT_VAL: C value past which linear optim works as hard-margin
// DEFAULT_MOOALPHA: default alpha for multi-objective optimiation DTLZ4
// DEFAULT_BAYES_STABPMAX: default pmax for Bayesian optimisation
// DEFAULT_BAYES_STABA: default A for Bayesian optimisation
// DEFAULT_BAYES_STABB: default B for Bayesian optimisation
// DEFAULT_BAYES_STABF: default D for Bayesian optimisation
// DEFAULT_BAYES_STABBAL: default balance for Bayesian optimisation
// DEFAULT_BAYES_STABDELTA: default delta for Bayesian optimisation
// DEFAULT_BAYES_STABZEROPT: default chi for Bayesian optimisation
// DEFAULT_BAYES_STABUSESIG: default use sigmoid compression for Bayesian optimisation
// DEFAULT_BAYES_STABTHRESH: default threshold for sigma
//
// DEFAULT_SAMPLES_SAMPLE: number of samples when sampling GPR.  Also used for use for approximating integrals etc on [0,1] - eg in globalopt.h and FNVector
// SIGMA_ADD: sigma value for samples added during sampling
//
// NN_DEFAULT_C: default C value for NN
// NN_DEFAULT_ZTOL: default zero tolerance for NN
// NN_DEFAULT_OPTTOL: default optimality tolerance for NN
// NN_DEFAULT_MAXITCNT: default max iteration count for NN
// NN_DEFAULT_MAXTRAINTIME: default max training time for NN
// NN_DEFAULT_LR: default learning rate for NN
// NN_DEFAULT_BLOCKSIZE: default block size for NN
// NN_DEFAULT_NCDS:
// NN_DEFAULT_SPARSEWEIGHT: default sparsity weighting for NN
// NN_DEFAULT_SPARSELEVEL: default sparsity level for NN
// NN_DEFAULT_DECREASELR:
//
// GR_DEFAULT_ZTOL: default zero tolerance for GR
// GR_DEFAULT_OPTTOL: default optimality tolerance for GR
// GR_DEFAULT_MAXITCNT: default max iteration count for GR
// GR_DEFAULT_MAXTRAINTIME: default max train time for GR
//
// KNN_DEFAULT_KAY: default "k" value for k-nearest-neighbours
//
// DEFAULT_SERIAL_MAXITCNT: default max iteration count for serial ML
// DEFAULT_SERIAL_MAXTRTIME: default max training time for serial ML
// DEFAULT_SERIAL_OPTTOL: default optimality tolerance for serial ML
// DEFAULT_SERIAL_LR: default learning rate for serial ML
// DEFAULT_SERIAL_SPARSE: default initialisation sparsity for serial ML
//
// IMP_DEFAULT_DELTA: default delta used by GP-UCB
// IMP_DEFAULT_NU: default nu used by GP-UCB


#define MINSWEIGHT                 1e-6

#define DEFAULT_ZTOL               1e-6
#define DEFAULT_EMM                2
#define DEFAULT_NRFF               10
#define DEFAULT_LSRFF              1
#define DEFAULT_REONLY             1
#define DEFAULT_C                  1.0
#define DEFAULT_D                  1.0
#define DEFAULT_SIGMA              0.1
#define DEFAULT_SIGMA_CUT          0.1
#define DEFAULT_MEMSIZE            -1
#define DEFAULT_OPTTOL             0.001
#define DEFAULT_TUNEV              1
#define DEFAULT_INADAM             7
#define DEFAULT_OUTGRAD            5
#define DEFAULT_LR                 0.3
#define DEFAULT_MAXITCNT           0
#define DEFAULT_MAXTRAINTIME       0
#define DEFAULT_TRAINTIMEEND       -1
#define DEFAULT_MAXFAILS           20
#define DEFAULT_MAXITERFUZZT       0
#define DEFAULT_LRFUZZT            0.3
#define DEFAULT_ZTFUZZT            0.01
#define DEFAULT_COSTFNFUZZT        "tanh(x)"
#define DEFAULT_MAXITERMVRANK      0
#define DEFAULT_LRMVRANK           0.3
#define DEFAULT_ZTMVRANK           0.01
#define DEFAULT_BETARANK           1
#define DEFAULTOUTERSTEPSCALE      0.7
#define DEFAULTOUTERSTEPBACK       0.9
#define DEFAULTOUTERDELTA          0.5
#define DEFAULTOUTERMOM            0
#define DEFAULTUSELS               3
#define DEFAULTOUTERTOL            1e-3
#define DEFAULTOUTERMAXITCNT       999999999
#define DEFAULTOUTERMAXTRAINTIME   0
#define DEFAULTOUTERTRAINTIMEEND   -1
#define DEFAULTOUTEROUTERSTEPSCALE 0.2
#define DEFAULTOUTEROUTERTOL       1e-3
#define DEFAULTOUTEROUTERMAXITCNT  999999999
#define DEFAULTOUTERRELTOL         1e-4
#define MINROWDIM                  500
#define ROWDIMSTEPRATIO            1.3
#define MAXBOUND                   1e6
#define DEFAULTEPS                 0.001
#define DEFAULTCYCEPS              0.1
#define CZEROOFFSET                1e-6
#define EPSZEROOFFSET              1e-6
#define MAXPASSIVECHANGE           1e3
#define OFFSET_ZERO                1e-5
#define MULTINORM_OUTERSTEP        0.3
#define MULTINORM_OUTERMOMENTUM    0.05
#define MULTINORM_OUTERMETHOD      3
#define MULTINORM_OUTERACCUR       0.05
#define MULTINORM_FULLCACHE_N      400
#define MULTINORM_MAXITS           1000
#define MULTINORM_OUTEROVSC        0.3
#define DEFAULT_SSV_LR             0.1
#define DEFAULT_SSV_MOM            0.01
#define DEFAULT_SSV_TOL            0.001
#define DEFAULT_SSV_OUTEROVSC      0.5
#define DEFAULT_SSV_MAXITS         10000
#define HARD_MARGIN_LINCUT_VAL     1e12
#define DEFAULT_MOOALPHA           100
#define DEFAULT_REGTYPE            1
#define DEFAULT_MLMLR              0.3
#define DEFAULT_DIFFSTOP           0.02
#define DEFAULT_LSPARSE            1.0
#define DEFAULT_THETA              1.0

//#define DEFAULT_SAMPLES_SAMPLE     30
// SEE ALSO VECTOR.H
#define DEFAULT_SAMPLES_SAMPLE     100
#define SIGMA_ADD                  1e-6

#define NN_DEFAULT_C               1e6
#define NN_DEFAULT_ZTOL            1e-6
#define NN_DEFAULT_OPTTOL          1e-3
#define NN_DEFAULT_MAXITCNT        1000
#define NN_DEFAULT_MAXTRAINTIME    0
#define NN_DEFAULT_TRAINTIMEEND    -1
#define NN_DEFAULT_LR              0.01
#define NN_DEFAULT_BLOCKSIZE       100
#define NN_DEFAULT_NCDS            1
#define NN_DEFAULT_SPARSEWEIGHT    0.1
#define NN_DEFAULT_SPARSELEVEL     0.3
#define NN_DEFAULT_DECREASELR      0.0

#define GR_DEFAULT_ZTOL            1e-6
#define GR_DEFAULT_OPTTOL          1e-3
#define GR_DEFAULT_MAXITCNT        20
#define GR_DEFAULT_MAXTRAINTIME    0
#define GR_DEFAULT_TRAINTIMEEND    -1

#define KNN_DEFAULT_KAY            5
#define KNN_DEFAULT_ZTOL_SQ        1e-12

#define DEFAULT_SERIAL_MAXITCNT    20
#define DEFAULT_SERIAL_MAXTRTIME   0
#define DEFAULT_SERIAL_OPTTOL      1e-3
#define DEFAULT_SERIAL_LR          0.1
#define DEFAULT_SERIAL_SPARSE      0.7

#define IMP_DEFAULT_DELTA          0.1
#define IMP_DEFAULT_NU             1

#define DEFAULT_BAYES_ZTOL         1e-8
#define DEFAULT_BAYES_DELTA        0.1
#define DEFAULT_BAYES_NU           0.2
#define DEFAULT_BAYES_A            1
#define DEFAULT_BAYES_B            1
#define DEFAULT_BAYES_R            1
#define DEFAULT_BAYES_P            2
#define DEFAULT_BAYES_STABPMAX     0
#define DEFAULT_BAYES_STABPMIN     1
#define DEFAULT_BAYES_STABA        0.1
#define DEFAULT_BAYES_STABB        0.1
#define DEFAULT_BAYES_STABF        1
#define DEFAULT_BAYES_STABBAL      0
#define DEFAULT_BAYES_STABZEROPT   -2.0
#define DEFAULT_BAYES_STABDELRREP  50
#define DEFAULT_BAYES_STABDELREP   50
#define DEFAULT_BAYES_STABSTABREP  1000
#define DEFAULT_BAYES_STABUSESIG   0
#define DEFAULT_BAYES_STABTHRESH   0.8
#define DEFAULT_BAYES_HYPER_METHOD  0
#define DEFAULT_BAYES_HYPER_MEASURE 3
#define DEFAULT_BAYES_HYPER_FOLDS   10
#define DEFAULT_BAYES_HYPER_REPS    1
#define DEFAULT_BAYES_HYPER_INCC    0
#define DEFAULT_BAYES_HYPER_INCKR   1
#define DEFAULT_BAYES_HYPER_INCKI   0
#define DEFAULT_BAYES_HYPER_LOWER   1e-2
#define DEFAULT_BAYES_HYPER_UPPER   1e2
#define DEFAULT_BAYES_HYPER_GRID    10
#define DEFAULT_BAYES_HYPER_MODE    0
#define DEFAULT_BAYES_HYPER_LOWERZ  1
#define DEFAULT_BAYES_HYPER_UPPERZ  3
#define DEFAULT_BAYES_HYPER_GRIDZ   3
#define DEFAULT_BAYES_HYPER_MODEZ   1

// DEFZCMAXITCNT - zero crossing maximum iteration count
// DEFINMAXITCNT - inner loop of interior point solver max iteration count
// MUFACTOR - mu factor for t scaling in interior point algorithm

#define DEFZCMAXITCNT 100
#define DEFINMAXITCNT 100




// Parsing functions - used to convert a line of data into usable form
//
// {>,=,!=,<} {y} {tVAL} {TVAL} {eVAL} {EVAL} x
//
// Classification: - {>,=,!=,<} must not be included.
//                 - y gives classification for point (0 for test/unknown).
//                   (y is required).  y=-1,0,+1,+2,...
//                 - {tVAL} or {TVAL} sets the empirical risk scale.
//                 - {eVAL} or {EVAL} sets the distance to surface scale.
//                 - x is the training vector.
//
// Regression: - {>,=,!=,<} defines if this is a lower bound, equality, not
//               used (ie ignore this vector) or upper bound constraint
//               (default is equality).
//             - y gives the target value for this point.
//               (y is required).
//             - {tVAL} or {TVAL} sets the empirical risk scale.
//             - {eVAL} or {EVAL} sets the epsilon insensitivity.
//             - x is the training vector.
//
// Single class: - {>,=,!=,<} must not be included.
//               - {y} must not be included
//               - {tVAL} or {TVAL} sets the empirical risk scale.
//               - {eVAL} or {EVAL} sets the distance to surface scale.
//               - x is the training vector.
//
// The format of the s vector may be either sparse or nonsparse.  Sparse
// vectors have the form:
//
// <feature1>:<valueF1> <feature2>:<valueF2> ... <featureN>:<valueFN>
//
// with all other values being assumed zero, whereas non-sparse vectors have
// the form:
//
// <value1> <value2> ... <valueN>
//
// Commas are treated as whitespace.
//
// d,z,x,Cweight,epsweight: data destinations
// src: source line
// reverse: assumes target at end of line, rather than start

void parselineML_Single (            SparseVector<gentype> &x, double &Cweight, double &epsweight,         std::string &src                 , int countZeros = 0);
void parselineML_Generic(gentype &y, SparseVector<gentype> &x, double &Cweight, double &epsweight, int &r, std::string &src, int reverse = 0, int countZeros = 0);



#endif

//
// Common ML functions and definitions
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "mlcommon.hpp"
#include "vecstack.hpp"
#include <iostream>
#include <sstream>
#include <string>

int isZeroString(const std::string &src);
int isZeroString(const std::string &src)
{
    // Test if zero (sparse form) - that is, 0 or n:0 where n is an int

    int res = 0;

    if ( ( src.length() == 1 ) && ( src == "0" ) )
    {
        res = 1;
    }

    else if ( src.length() <= 1 )
    {
        res = 0;
    }

    else if ( ( src[src.length()-1] == '0' ) && ( src[src.length()-2] == ':' ) )
    {
        // Assume n form correct
        res = 1;
    }

    return res;
}

int decomma(std::string &src, int &elmcnt, int countZeros = 0);
int decomma(std::string &src, int &elmcnt, int countZeros)
{
    // It is assumed that there are no preceeding or postceeding whitespace

    elmcnt = 0;

    if ( src.length() )
    {
	size_t i;
        Stack<char> parenStack;
        int isQuote = 0;
        int isSpace = 0;
        size_t valstart = 0;
        size_t valend = 0;
        bool valend_set = false;
        int isfinalzero = 0; // final element is included regardless to
                             // preserve dimensionality of vector, if needed.
                             // This is important for per-vector normalsiation
                             // in mercer kernels, where the cim
                             // of the vector must be representative, so
                             // for example mean([ 0 0 1 0 ]) = 0.25, rather
                             // then becoming mean([2:1]) = 1, which would
                             // cause problems.
        int isfirstzero = 1; // So that min/max work well in per-vector
                             // normalisation we need to ensure that if
                             // there are any zeros in the vector then at
                             // least one "representative" zero will be kept.

	for ( i = 0 ; i < src.length() ; ++i )
	{
            // Fix spacing, mark start and end of values

            if ( !(parenStack.size()) )
            {
                if ( ( src[i] == ',' ) || ( src[i] == ' ' ) || ( src[i] == '\t' ) )
                {
                    // Overwrite commas, spacing consistency
                    src[i] = ' ';

                    if ( !isSpace )
                    {
                        valend = i-1;
                        valend_set = true;
                        isSpace = 1;
                    }
                }

                else
                {
                    if ( isSpace )
                    {
                        valstart = i;
                        isSpace = 0;
                    }
                }
            }

            if ( i == (src.length())-1 )
            {
                valend = i;
                valend_set = true;
            }

            // Evaluate value - increment if non-zero or countZero

            if ( valend_set && ( valend >= valstart ) )
            {
                isfinalzero = 0;

                if ( countZeros )
                {
                    // counting all, zero or otherwise
                    ++elmcnt;
                }

                else if ( !isZeroString(src.substr(valstart,valend-valstart+1)) )
                {
                    // Only counting non-zero, sparsing out others
                    ++elmcnt;
                }

                else if ( isfirstzero )
                {
                    // Only counting non-zero, sparsing out others
                    // but keep this one as it may be the only zero, and
                    // we may need it to make max/min work
                    ++elmcnt;
                    isfirstzero = 0;
                }

                else
                {
                    isfinalzero = 1;
                }

                valstart = valend+1;
            }

            // Process parenthesis

            if ( !isQuote )
            {
                if ( src[i] == '\"' )
                {
                    isQuote = 1;
                    parenStack.push(src[i]);
                }

                else if ( ( src[i] == '(' ) || ( src[i] == '[' ) || ( src[i] == '{' ) )
                {
                    parenStack.push(src[i]);
                }

                else if ( ( src[i] == ')' ) || ( src[i] == ']' ) || ( src[i] == '}' ) )
                {
                    if ( parenStack.size() )
                    {
                        if ( ( parenStack.accessTop() == '(' ) && ( src[i] == ')' ) )
                        {
                            parenStack.pop();
                        }

                        else if ( ( parenStack.accessTop() == '[' ) && ( src[i] == ']' ) )
                        {
                            parenStack.pop();
                        }

                        else if ( ( parenStack.accessTop() == '{' ) && ( src[i] == '}' ) )
                        {
                            parenStack.pop();
                        }

                        else
                        {
                            return 1;
                        }
                    }
                }
            }

            else if ( src[i] == '\"' )
            {
                NiceAssert( parenStack.size() );
                isQuote = 0;
                parenStack.pop();
            }
        }

        if ( isfinalzero && !countZeros )
        {
            // If vector ends in a zero then this is counted no matter
            // what.  Of course if we are counting zeros then that has
            // already been done, but if we are not counting zeros and
            // the final element was a zero then we need to increment
            // the element count to include it in the count.

            ++elmcnt;
        }
    }

    return 0;
}

void parselineML_Single_all(SparseVector<gentype> &x, double &Cweight, double &epsweight, std::string &src, int countZeros = 0);
void parselineML_Single_all(SparseVector<gentype> &x, double &Cweight, double &epsweight, std::string &src, int countZeros)
{
    //std::string src = xsrc;
    int elmcnt = 0;
    decomma(src,elmcnt,countZeros);
    x.zero();
    x.prealloc(elmcnt);

    int i = 0;

    Cweight = 1;
    epsweight = 1;

    NiceAssert( src.length() );

    while ( isspace(src[i]) )
    {
	++i;

        NiceAssert( i < (int) src.length() );
    }

repover:
    if ( i < (int) src.length() )
    {
	if ( ( src[i] == 't' ) || ( src[i] == 'T' ) )
	{
	    ++i;

            NiceAssert( i < (int) src.length() );
            NiceAssert( !isspace(src[i]) );

	    std::string dsrcb = src.substr(i,src.length());

	    std::istringstream dbufferb;
	    dbufferb.str(dsrcb);
	    dbufferb >> Cweight;

	    while ( !isspace(src[i]) )
	    {
		++i;

		if ( i == (int) src.length() )
		{
		    break;
		}
	    }

	    if ( i < (int) src.length() )
	    {
		while ( isspace(src[i]) )
		{
		    ++i;

		    if ( i == (int) src.length() )
		    {
			break;
		    }
		}
	    }

            goto repover;
	}

	if ( ( src[i] == 's' ) || ( src[i] == 'S' ) )
	{
	    ++i;

            NiceAssert( i < (int) src.length() );
            NiceAssert( !isspace(src[i]) );

	    std::string dsrcb = src.substr(i,src.length());

	    std::istringstream dbufferb;
	    dbufferb.str(dsrcb);
	    dbufferb >> Cweight;

            Cweight = ( Cweight < MINSWEIGHT ) ? (1.0/MINSWEIGHT) : (1/Cweight);

	    while ( !isspace(src[i]) )
	    {
		++i;

		if ( i == (int) src.length() )
		{
		    break;
		}
	    }

	    if ( i < (int) src.length() )
	    {
		while ( isspace(src[i]) )
		{
		    ++i;

		    if ( i == (int) src.length() )
		    {
			break;
		    }
		}
	    }

            goto repover;
	}

	if ( ( src[i] == 'e' ) || ( src[i] == 'E' ) )
	{
	    ++i;

            NiceAssert( i < (int) src.length() );
            NiceAssert( !isspace(src[i]) );

	    std::string dsrcb = src.substr(i,src.length());

	    std::istringstream dbufferb;
	    dbufferb.str(dsrcb);
	    dbufferb >> epsweight;

	    while ( !isspace(src[i]) )
	    {
		++i;

		if ( i == (int) src.length() )
		{
		    break;
		}
	    }

	    if ( i < (int) src.length() )
	    {
		while ( isspace(src[i]) )
		{
		    ++i;

		    if ( i == (int) src.length() )
		    {
			break;
		    }
		}
	    }

            goto repover;
	}
    }

    x.zero();

    if ( i < (int) src.length() )
    {
        std::string xsrc(src.substr(i,src.length()-i));
        std::string lbrace("[ ");
        std::string rbrace(" ]");
        std::string ysrc(lbrace+xsrc+rbrace);

        std::istringstream xbufferc;
        xbufferc.str(ysrc);
        streamItInAlt(xbufferc,x,0,!countZeros);
    }

//    if ( x.indsize() )
//    {
//        int iji;
//
//        for ( iji = 0 ; iji < x.indsize() ; ++iji )
//        {
//            if ( (x.direcref(iji)).isValEqnDir() )
//            {
//                (x.direref(iji)).scalarfn_setisscalarfn(1);
//            }
//        }
//    }

    return;
}

void parselineML_Single(SparseVector<gentype> &x, double &Cweight, double &epsweight, std::string &src, int countZeros)
{
    int i = 0;
    int j = (int) src.length()-1;

    while ( isspace(src[i]) )
    {
	++i;

        NiceAssert( i < (int) src.length() );
    }

    while ( isspace(src[j]) )
    {
	--j;

        NiceAssert( ( j >= i ) && ( j >= 0 ) );
    }

    std::string dsrc = src.substr(i,j-i+1);

    parselineML_Single_all(x,Cweight,epsweight,dsrc,countZeros);

    return;
}


void parselineML_Generic(gentype &z, SparseVector<gentype> &x, double &Cweight, double &epsweight, int &r, std::string &src, int reverse, int countZeros)
{
    int dummy;

    decomma(src,dummy);

    int i = 0;
    int j = (int) src.length()-1;

    NiceAssert( src.length() );

    while ( isspace(src[i]) )
    {
	++i;

        NiceAssert( i < (int) src.length() );
    }

    while ( isspace(src[j]) )
    {
	--j;

        NiceAssert( ( j >= i ) && ( j >= 0 ) );
    }

    r = 2;

    if ( src[i] == '>' )
    {
	r = +1;
	++i;

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    else if ( src[i] == '<' )
    {
	r = -1;
        ++i;

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    else if ( src[i] == '=' )
    {
	r = 2;
	++i;

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    else if ( src[i] == '!' )
    {
	r = 0;
	++i;

        NiceAssert( i < j+1 );

        if ( src[i] == '=' )
        {
            ++i;
        }

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    while ( isspace(src[i]) )
    {
	++i;

        NiceAssert( i < j+1 );
    }

    if ( !reverse )
    {
	std::string dsrc = src.substr(i,j-i+1);

	std::istringstream dbuffer;
	dbuffer.str(dsrc);
        dbuffer >> z;

        int bracketcnt = 0;
        int sqbracketcnt = 0;
        int curlybracketcnt = 0;
        int inquote = 0;

        while ( !( isspace(src[i]) && !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote ) )
	{
            if ( inquote )
            {
                if ( src[i] == '\"' ) { inquote = 0; }
            }

            else
            {
                     if ( src[i] == '('  ) { ++bracketcnt;      }
                else if ( src[i] == '['  ) { ++sqbracketcnt;    }
                else if ( src[i] == '{'  ) { ++curlybracketcnt; }
                else if ( src[i] == ')'  ) { NiceAssert( bracketcnt      ); --bracketcnt;      }
                else if ( src[i] == ']'  ) { NiceAssert( sqbracketcnt    ); --sqbracketcnt;    }
                else if ( src[i] == '}'  ) { NiceAssert( curlybracketcnt ); --curlybracketcnt; }
                else if ( src[i] == '\"' ) { inquote = 1; }
            }

            ++i;

	    if ( i == j+1 )
	    {
                NiceAssert( !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote );

                break;
	    }
	}

	if ( i < j+1 )
	{
	    while ( isspace(src[i]) )
	    {
		++i;

		if ( i == j+1 )
		{
		    break;
		}
	    }
	}
    }

    else
    {
        int bracketcnt = 0;
        int sqbracketcnt = 0;
        int curlybracketcnt = 0;
        int inquote = 0;

        while ( !( isspace(src[j]) && !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote ) )
	{
            if ( inquote )
            {
                if ( src[j] == '\"' ) { inquote = 0; }
            }

            else
            {
                     if ( src[j] == ')'  ) { ++bracketcnt;      }
                else if ( src[j] == ']'  ) { ++sqbracketcnt;    }
                else if ( src[j] == '}'  ) { ++curlybracketcnt; }
                else if ( src[j] == '('  ) { NiceAssert( bracketcnt      ); --bracketcnt;      }
                else if ( src[j] == '['  ) { NiceAssert( sqbracketcnt    ); --sqbracketcnt;    }
                else if ( src[j] == '{'  ) { NiceAssert( curlybracketcnt ); --curlybracketcnt; }
                else if ( src[j] == '\"' ) { inquote = 1; }
            }

	    --j;

	    if ( j == i-1 )
	    {
                NiceAssert( !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote );

		break;
	    }
	}

	std::string dsrc = src.substr(j+1,src.length());

	std::istringstream dbuffer;
	dbuffer.str(dsrc);
        dbuffer >> z;

	if ( j > i-1 )
	{
	    while ( isspace(src[j]) )
	    {
		--j;

		if ( j == i-1 )
		{
		    break;
		}
	    }
	}
    }

    std::string dsrc = src.substr(i,j-i+1);

    parselineML_Single_all(x,Cweight,epsweight,dsrc,countZeros);

//    if ( z.isValEqnDir() )
//    {
//        z.scalarfn_setisscalarfn(1);
//    }

    return;
}



