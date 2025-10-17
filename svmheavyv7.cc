/*
PYTHON INTERFACE EXAMPLE (SEE tmp2 DIRECTORY):

c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cc -o example$(python3-config --extension-suffix)

#include <pybind11/pybind11.h>
#include <iostream>
#include <string>

// General scheme:
//
// if gentype is null then eval without argument
// if gentype is int then cast as int and send to an intvalsrc and use this in eval
// if gentype is double then cast as double and send to an doublevalsrc and use this in eval
// if gentype is complex then cast as double and send to an complexvalsrc and use this in eval
// if gentype is realvector then cast as std::vector and send to an realvectorvalsrc and use this in eval
// if gentype is realmatrix then cast as eigen::matrix and send to an realmatrixvalsrc and use this in eval
// if gentype is string then cast as int and send to an stringvalsrc and use this in eval
// for set, then have as many arguments as there are values in the set
// otherwise act as if double is nan and call with that (including values in vectors, matrices and sets)

namespace py = pybind11;

double setvalsrc(int doset = 0, double val = 0)
{
    static thread_local double xval;

    if ( doset )
    {
        xval = val;
    }

    return xval;
}

double valsrc(void)
{
    return setvalsrc();
}

int intvalsrc(void)
{
    return (int) setvalsrc();
}

void procdata(void)
{
    double x;
    std::string fn;

    std::cout << "Function: "; std::cin >> fn;
    std::cout << "x: "; std::cin >> x;

    setvalsrc(1,x);

    std::string evalfn("eval('");
    evalfn += fn;
    evalfn += "(example.intvalsrc())')";

    //std::cout << "Evaluation string: " << evalfn << "\n";

    py::object builtins = py::module_::import("builtins");
    py::object eval = builtins.attr("eval");
    auto resultobj = eval(evalfn);
    double result = resultobj.cast<double>();

    std::cout << "Result = " << result << "\n";

    return;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("valsrc", &valsrc, "Get double for eval");
    m.def("intvalsrc", &intvalsrc, "Get int for eval");
    m.def("procdata", &procdata, "Apply user function to data stream");
}

*/

//FIXME: in sparsevector, have an optional "information" field that is literally just a text description.
//FIXME: in tuneKernel, test if variables are isNomConst and don't include if they are
//FIXME: remove kernel -kwc, -kwuc etc and use const variables instead!


//FIXME: have posterior variance approximation as distance from the nearest training point scaled (mult or div) by lengthscale, capped at 1 (for SE etc)
//       see lsv_scalar.cc for var/cov functions
//FIXME: make vector, gentype, matrix etc work without basefn where possible (memory and assert in different file?)



//FIXME: add Hermite polynomials to gentype (physicist and statistician, regular and normalised).







//FIXME FIXME: in mercer.h, fix 2x2 implementation as per lsv_scalar.cc, *** line near K2x2

//FIXME: K2x2 should directly call a new fn K2x2 in mercer.h that can do dense integral properly

//FIXME: why is -Stt for 2-d so damn slow

//FIXME: when doing setSampleMode for positive GP eg -St 10, rather than set alpha +ve instead set y greater than sigma everywhere.

//FIXME: need kernel with dense transform of both arguments!  This is required to enable JIT sampling.  Is the dense integral acceptable as a Jidling transform?








//yyybK... - add Chau et al, "Learning Inconsistent Preferences with Gaussian Processes" style kernels (4)





// in lsv_scalar_rff: gamma get/set and Gp get functions need to pad/unpad alpha for the underlying expanded version
//
// RFF for isLinearCost simply doesn't work afaict
//
// Make LS-SVM classes for scalar_rff and binary_rff.  Template:
//./svmheavyv7.exe -z R -Zx -rfs 0 -Zx -R q -nN 500 -trf 0 -lrf .3 -trt 3 -dia 0 -c 10 -AA xor.txt -tr -tl -tmg -s temp2d.svm
//./svmheavyv7.exe -z r -R q -c 10 -kt 3 -kg .3 -AA xor.txt -tr -tl -tmg
//
// FIXME to change the kernel you need to put -trt here or you get a segfault
//./svmheavyv7.exe -z R -Zx -rfs 0 -trt 13 -Zx -R q -nN 500 -trf 0 -lrf .3 -dia 0 -c 10 -AA xor.txt -tr -tl -tmg -s temp2d.svm
//
// FIXME: RFF log likelihood and max information gain are wrong



// Test functions https://www.sfu.ca/~ssurjano/optimization.html - do:
//
// 33 19 Rotated Hyper-Ellipsoid Function
// 34 21 Sum of Different Powers Function
// 35 22 Sum Squares Function
// 36 23 Trid Function
// 37 27 Power Sum Function
// 38 28 Zakharov Function
// 39 30 Six-Hump Camel Function
// 40 31 Dixon-Price Function
// 41 33 De Jong Function N. 5
// 42 35 Michalewicz Function
// 43 37 Branin Function
// 44 38 Colville Function
// 45 39 Forrester et al. (2008) Function
// 46 41 Hartmann 3-D Function
// 47 42 Hartmann 4-D Function
// 48 43 Hartmann 6-D Function
// 49 44 Perm Function d, β
// 50 45 Powell Function
// 51 46 Shekel Function


// Calculator: tertiary operators and functions
// spit out graphs
// long equations go out side
// explanation for each function

// GUI: add a GUI
// in general set/read functions, have ratings for most used, least used, rank by these

// Implement crd (chord) function crd(x) = 2sin(x/2) (see wiki)
// also acrd,crdh,acrdh

// Dual numbers: dualnum.h/cc
// also do split complex number a+b.j, where j.j = 1


/*
Using logistic map, tune lambda to achieve a particular frequency of oscillation.  System
simulates a few thousand steps, correlates with sin wave of appropriate frequency, and
the correlation coeff is the result.  For side-channels we construct a regression model
of the resulting waveform using a fixed kernel.  This is a vector in RKHS space, so it
becomes the side-channel.  Hopefully similar frequencies are similar and different frequencies
or chaos are very different.

What about an electric circuit that can oscillate but goes chaotic if components go out
of range?



for gentype: https://code.launchpad.net/anant
*/

//FOR vector-valued versions, when evaluating gradient, need to correct tspaceDim() to allow for non-directional gradient result dimension inflation!

//TO DO: option to save *all* registered MLs as filenameN.ml, where N is registered ML number
//TO DO: direct, nelder should by default suppress output of experiments! (and flag for others)


/*
temp:  d=10  e=10
temp2: d=.1  e=.1
temp3: d=.1  e=10
temp4: d=10  e=.1
temp5: d=100 e=.1



temp2,temp4,temp5

v gets bigger with increasing d (decreasing Lambda)
alpha barely changes
w gets smaller with increasing d (decreasing Lambda)
gamma has a big influence on the range of v - big values allow large variation
larger Lambda seems to make optimisation quicker
*/

// Kernel inheritance from Scalar_rff should be through the explicit feature map!
// This should be trivial: just map the vectors (however many there are) and take the 1/2/3/4/m product as required.  mercer should take care of the ugly cases (far et al).
// kernel derivatives should also be trivial by simply flipping sin/cos and multiplying by signs as required)

/*
mutex minimisation/removal:
mercer.h: make the mutex a member so that it only locks the vector in question!
sparsevector.h: make the mutex a member so that it only locks the vector in question!
knn_densit.cc: make the mutex a member so that it only locks the vector in question!
*/

//TO DO: mercer.h should have "draw from distribution" function

//./svmheavyv7.exe -qw 3 -z r -kt 2 -kd 3 -Zx -qw 1 -Zx -R q -c 10 -kt 801 -ktx 3 -AA xor.txt -AA xor.txt -Zx -qw 2 -z r -Zx -R q -c 10 -kt 801 -ktx 3 -AA and.txt -AA and.txt -Zx -qw 3 -Zx -xl -1 -xs 1e-6 -xr 2 -xi 400 -xo 3 -xC 2 -x 20 [ 1 2 ] [ 1 1 ] -Zx -qw 1 -tx -s temp1.svm -Zx -qw 2 -tx -s temp2.svm -Zx -qw 3 -x temp3.svm

/*
ML_Base inheritors:
gpr_generic - done
mlm_generic - done

svm_generic inheritors (from ml_base):
svm_multic - done
svm_vector - done

lsv_generic inheritors (from svm_scalar):
lsv_gentyp
lsv_mvrank
lsv_planar

imp_generic inheritors (from ml_base):
imp_parsvm
*/


// Implement hyperkernels eg
// https://papers.nips.cc/paper/3099-gaussian-and-wishart-hyperkernels.pdf

/*

NOTE: - ready to test Bayesian optimisation for hyperkernel design!
      - -z ker -SAA to set lambda (-SAi for individual components), -Ag to add random vectors to dataset
      - for each model, inherit kernel from relevant block
      - -tx on each block, store var(1,1) (result) in var(10,i) for targets, var(100,i) for sensitive variables (eg gender)
      - then -tM var(10,0)+var(10,1)+...-var(100,0)-var(100,1)-...
      - note that the second batch are *maximised*, while the former are *minimised*



1/N sum_i de_i/dg_i dg_i/dbeta_k is the gradient you want to minimise (or maximise for anti-learning)

g_i = sum_j alpha_j K(x_i,x_j)
g_i = sum_j alpha_j sum_kl beta_k beta_l K(x_i,x_j,z_k,z_l) / 2

dg_i/dbeta_k = sum_j alpha_j sum_l beta_l K (x_i,x_j,z_k,z_l)
             = sum_jl alpha_j beta_l K (x_i,x_j,z_k,z_l)

1/N sum_i de_i/dg_i dg_i/dbeta_k = 1/N sum_i de_i/dg_i sum_jl alpha_j beta_l K(x_i,x_j,z_k,z_l)
                                 = 1/N sum_ijl de_i/dg_i alpha_j beta_l K(x_i,x_j,z_k,z_l)

Let s_q be +1 for learning, -1 for anti-learning, with q=1 to m tasks

1/m s_q 1/N sum_i de_qi/dg_qi dg_qi/dbeta_k is the total gradient, negated for anti-learning blocks

                                            = sum_l beta_l ( sum_ij ( 1/(mN) sum_q s_q de_qi/dg_qi alpha_qj )_ij K(x_i,x_j,z_k,z_l) )_kl
                                            = 1/(mN) sum_ij (sum_q s_q de_qi/dg_qi alpha_qj)_ij sum_l beta_l K(x_i,x_j,z_k,z_l)
                                            = sum_l beta_l ( sum_ij ( 1/(mN) sum_q s_q de_qi/dg_qi alpha_qj )_ij K(x_i,x_j,z_k,z_l) )_kl
                                            = sum_l beta_l ( sum_ij B_ij K(x_i,x_j,z_k,z_l) )_kl

B_ij = 1/(mN) sum_q s_q de_qi/dg_qi alpha_qj
     = 1/(mN) sum_q s_q B_qij

B_qij = de_qi/dg_qi alpha_qj = R^{N_s*N_s}


    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res) const { NiceThrow("dedKTrainingVector not implemented for this ML"); return res; }

this gives B_qij

    virtual int getaltML(kernPrecursor *&res, int altMLid) const { return kernPrecursor::getaltML(res,altMLid); }

(consider whether m and/or N scaling factors are needed here)









int Nblk; // number of MLs being optimised for
const double lr(lrKB()); // learning rate
const Vector<int> &altMLids(altMLidsKB()); // IDs of MLs being optimised
const Vector<double> MLweight(MLweightKB()); // weights for different MLs
const double minstep(minstepKB());
const int maxiter(maxiterKB());

Vector<int> alphaState;
Matrix<double> B;
Matrix<double> Bnext;
const ML_Base *firstML;

gentype tmp;
int dummya;
int i,j,k,l;
int notdone = 1;
int firststep = 1;
int itcnt = 0;
int Nblk = altMLids.size();

NiceAssert( MLweight.size() == Nblk );

Matrix<double> lambdaGrad(N());

while ( !killswitch && notdone && ( itcnt < maxiter ) )
{
    ++itcnt;

    if ( !firststep )
    {
        for ( q = 0 ; q < Nblk ; ++q )
        {
            kernPrecursor *MLblk;
            getaltML(MLblk,altMLids(q));
            const ML_Base &MLblock = dynamic_cast<const ML_Base &>(*MLblk);

            if ( !q )
            {
                firstML = &MLblock;

                B = 0.0;
                MLblock.dedKTrainingVector(B) *= MLweight(q);
                alphaState = MLblock.alphaState();
            }

            else
            {
                Bnext = 0.0;
                MLblock.dedKTrainingVector(Bnext) *= MLweight(q);
                alphaState += MLblock.alphaState();

                B += Bnext;
            }
        }

        B *= (1.0/((double) (Nblk*B.numRows())));

        int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
        int Nrep = lambdaKB().numCols();
        int Ninner = B.numRows();

        lambdaGrad = 0.0;

        for ( q = 0 ; q < Nrep ; ++q )
        {
            for ( k = 0 ; k < Nval ; ++k )
            {
                for ( l = 0 ; l < Nval ; ++l )
                {
                    for ( i = 0 ; i < Ninner ; ++i )
                    {
                        if ( alphaState(i) )
                        {
                            for ( j = 0 ; j < Ninner ; ++j )
                            {
                                if ( alphaState(j) )
                                {
                                    lambdaGrad("&",k,q) += lambdaKB()(l,q)*B(i,j)*((double) K4(tmp,(*firstML).x()(i),(*firstML).x()(j),x()(k),x()(l),&((*firstML).xinfo()(i)),&((*firstML).xinfo()(j)),&(xinfo()(k)),&(xinfo()(l))));
                                }
                            }
                        }
                    }
                }
            }
        }

        double stepsize = 0;

        if ( ( itcnt >= 3 ) && ( sqsum(stepsize,lambdaGrad) < minstep ) )
        {
            notdone = firststep;
        }

        // take the step

        lambdaGrad *= -lr;
        lambdaGrad += lambdaKB();

        setlambdaKB(lambdaGrad);
    }

    for ( q = 0 ; q < Nblk ; ++q )
    {
        kernPrecursor *MLblk;
        getaltML(MLblk,altMLids(q));
        const ML_Base &MLblock = dynamic_cast<const ML_Base &>(*MLblk);

        MLblock.resetKernel();
        MLblock.train(dummy);
    }

    firststep = 0;
}

*/


//Vector<ML_Base &>









//NEXT: implement eTrainingVector and dedgTrainingVector for errors where required (error and error gradient wrt g, should be simple)
//NEXT: then can use gradients to define a block that does bi-quadratic multi-task learning and anti-learning

//FIXME: need to define eTrainingVector and dedgTrainingVector in svm_multic_*

//FIXME: block e, gradients on svm_densit,gentyp,pfront,planar

//FIXME: blk_* doesn't implement e or de stuff

//FIXME: svm_vector_redbin.h: other dgTrainingVector variants

//FIXME: alpha in blk_conect.h
//FIXME: as well as y in smboopt.h, have option to use alpha instead.

//FIXME: see rkshfail.sh

//MERCER: FIXME: test gradient implementation

//FIXME xorgrad2.txt does not work!!!

//FIXME phi1,phi3,phi4,phim need to be strung into ml_base tree
//FIXME need option to return RKHSVector for SVM,LS-SVM,GPR

//TO DO: svmmatlab: handling of scalar function passing should do multiple argument functions -> arrays
//TO DO: svmmatlab: handling of non-scalar function passing should make then anonymous functions (eg via str2func('@(x)...') somehow parsed in matlab -> anomymous functions on variables
//TO DO: gentype should have a way of converting to matlab type function string (see str2func in matlab).  Make results work predictably for fucntions with different names in gentype and matlab

/*
TO DO:

1. JMLR - forest (done), DSA (done), gas (data downloaded), adult, oar in: high-dimensional and large-scale anomaly detection using a linear one-class svm with deep learning, sutharshan
2. response to reviews (ARD implemented and run for artificial exps, need to run on fiber and metallurgy - finish and add to paper)
3. response to reviews (implement MKL-GP and add results to paper)
*/

//FIXME: have FNVectors of multiple types: RKHS, Bernstein, Fourier etc

//FIXME: derivatives are actually wrt dScale.*(x-dShift) in mercer.h, so need
// to factor this in when using in ML_Base

//TO DO: incorporate bobyqa

//FIXME (maybe): properly implement Bernstein basis for vectorial x (see blk_bernst for details)
//FIXME: in gentype.cc, some OP_elementwiseDefaultCallA functions need to be modified (excepted) for infsize vectors
//FIXME: in gentype.cc, deref{v,a,m} and collapse need to be modified (excepted) for infsize vectors





//FIXME: in vector/RKHSvector there is *no need* for Vector<T> &i dereference (operator()) as you can just chuck a vector into the T &i type and achieve the same thing!

//FIXME: in gentype, have isinf and inrkhs logical functions
//FIXME: consider *replacing* scalar functions with infinite dimensional vectors
//FIXME: have short-form stream operators for kernels so that RKHS vectors can be readily printed and streamed in, fully automatic detection (eg vector is [ ], RKHS vector is [[ kernelshortform ; x ; a ]] or similar)

//FIXME: mProdPt: need some way to specify noConj, revConj etc etc
//FIXME: pull mProdPt function in ml_base through the whole tree up to ml_mutable
//FIXME: mProdPt in svm_generic.cc, need to implement m-Products (m > 4)


//DONE: inner product between sparsevectors should short based on whichever whichever dimension is lower.  This will significantly simplify the mercer and ml_base code that currently disallows a bunch of optimisations if dimensions differ
//DONE fixes: sparsevector.h (done), gentype.h (done) (specialisations), mercer.h (removes dimensionality tests), ml_base.h (dimensionality tests)
//TO DO: pass xconsist to innerProductDiverted and on to getProduct, so that if altcontent not present can do quick indexed bypass.











//-gc seems to be broken.








//DONE: -gmq 2: purely random oracle






//DONE: -g+ [ ... ] allows you to put penalty terms on p(x)








//TO DO: read published version of lineBO and implement (in addition to current version) their heuristic for choosing epsilon in inner loop (see -gbe)
//
//eps = ( dlogT / 2T )^( (1-2.kappa)/2 ) (for local convergence, where T is some sort of target)
//
//TO DO: how to set range on alpha for line-search?
//TO DO: implement alternative (functional BO paper)






//TO DO: need to be able to return "out of bound" error to code.  Perhaps using nan?  This can be used to restrict alpha for line-search
//TO DO: need to map inf result to large number






//FIXME multistart (at globalopt level, different projections every time)






//FIXME: I honestly can't remember what -x does!






//FIXME: make load work!





//FIXME: 
//
// - add "multiply kernel by scalar" option to aux a_9
// - allow :: e0 ~ e1 ~ ... :::: a0 ~ a1 ~ ...
// - for more general linear operators, kernel evalution is sum of these







//TO DO: gradient based optimiser











//TO DO: each mercer kernel that inherits from another should register with the other so that we can go "upstream" for learning

//TO DO: implement a distributions class.  Base class dist, specialised to various distributions.  Define sample, expectation, variance, support etc in base, specialise where possible
//TO DO: svm with target distribution (actually gentyp should cover this once you incorporate the distribution type)

// LSV_SCALAR: have option to use alternative training schemes for large datasets (SMO, D2C etc) - BUT WHAT ABOUT VARIANCE CALCULATIONS?

//FIXME (old mlinter.cc - may not be relevant): SVM_planar: need VV output kernels included - gradients can then be run through to svm_mvrank (and inheritors)
//FIXME (old mlinter.cc - may not be relevant): sparsevector g(N) - g(N) should return zero element, but doesn't
//FIXME (old mlinter.cc - may not be relevant): now that svm_planar works for arbitrary planes, have option to make basis the standard basis for output (target) space and then do cyclic (rotate template = reflect on line between template and direction)
//FIXME (old mlinter.cc - may not be relevant): no need for mvrank, just have external training method to set VV basis starting from random



//- DO PROJECTION STUFF AT SMBOOPT LAYER 
//- NEED A "DRAW" FUNCTION TO TAKE TWO SAMPLES FROM A GP (IE COPY IT AND SET THE ISSAMPLE BIT) AND THEN COMBINE THEM USING BLK_CONECT 
//- EVALUATING X INVOLVES SETTING WEIGHTS AND THEN PASSING RELEVANT BLK_CONECT TO BASE
//
// BLK_Conect now has weight option
// Therefore make it point to two (or more) samples of an underlying GP and optimise the weights
// This lets you optimise a *function*!
// Simple test case: the norm of || f(x) - sin(x) ||, f(x) should converge to something that approximates a sine wave
// BO needs to be able to create samples of underlying GP, plus BLK_Conect for sums of them, then return ML number of best BLK_Conect (and optionally clear out non-optimal ones)
// The kernel for BO is then defined on the distribution of BLK_Conect objects

/*

THINGS TO FIX:

 - in mercer.h, K4 is fast because it never allocates matrix xy unless absolutely
   necessary, prefering to use xy00, xy01, ... (which has no alloc overhead).  This
   needs to be generalised to K1,K3 (K2 already fast)
 - dg... evaluation: default to function that uses kernel trick a-la cov...
 - add bounds in Cha1 for parameter optimisation -g
 - svm_vector_mredbin: isKVarianceNZ: not implemented properly!
 - really need to implement sparse vector machine - see super-sparse learning in similarity spaces, demontis et al, Dem2
   similar to svm_planar, in scalar version you need to inherit from lsv_scalar but hijack K calculation function to redirect via cheatscache.
 - Need max training time for outer loop in m>4-norm SVM training
 - matrix.h: use LUDP decomposition (see doolittles method on wikipedia) for matrix inversion etc.
 - inequality constraints and subsequently classification for GPs
 - sanitise inputs for all models - inf -> 1/ztol, -inf -> -1/ztol, ditto 0 -> ztol for things like C
 - istrained needs to be made valid for layovers like svm_mvrank.
 - -g nonlinear scaled need to be made to work for *all* variants (int apparently done)!
   and you need to add shift/distort/zoom options
 - -gm/-gi should be able to work for all too: you'll use this for distort/focus
   need a way to specify different measurement noise (scales) for each measurement
 - sort out variable lists everywhere.
 - printing: always print double at max precision (some sort of stream modification?)
 - printing: fix up all classes to print minimal, informative and complete data.
 - printing: title at top to ml_mutable can automatically detect as required.
 - Need an option for persistent streams that just keep trying to connect
   Maybe just add options for max retries (reconnect attempts, -1 means
   try forever) and timeout times (again -1 means no timeout)
 - (low priority) threads need debugging on all systems!
 - numbase - transition away from all GSL dependencies (currently gamma_inc, psi, psi_n, dawson) - see http://www.mymathlib.com/functions maybe?
 - gentype - add functions j0, j1, jn, y0, y1, yn (they are not in numbase)
 - gentype - add derivatives for j0,j1,jn,y0,y1,yn
 - ml_mutable - go through and update morph data transfer (many parts missing)
 - svm_* - vector SVM with spherical cost, can maybe just set
           alpha_i = |alpha_i| u_i and adjust u_i in external loop (this
           cannot lead to non-PSD hessians eg
           U'.U = [ 1 -1 -1 ; -1 1 -1 ; -1 -1 1 ], easy to prove).
 - svm_* - multic_atonce: add tube shrinking.  Do fixed bias option by
           using local calculation of z offset and setting biasdim = 1.
 - svm_* - multic_redbin: add tube shrinking.
 - optstate - need a STEPTRIGGER type bound on cumulative gradient error,
              automatically recalculating gradients when a trigger value
              is reached.
 - sQsLsAsWs - betagradstate says which one of betaGrad(pivBetaF()) is
               nonzero.  However pivBetaF() can change order without
               warning, invalidating betagradstate.  Thus the code assumes
               that the incorrect element of betaGrad(pivBetaF()) is
               is nonzero, and optimisation fails (status: rarely occurs).
 - documentation - matrix + scale does diagonal offset
 - documentation - mchained (deep) kernels and kernel 29
 - documentation - mremoved operators on vector, sparsevector, matrix types
 - documentation - mchaining in mercer
 - documentation - mmany changes in gentype
 - documentation - mrealDeriv, remove derivprod
 - documentation - mthat anything*1 = 1*anything = anything (ie unity always)
 - documentation - mlikewise anything+0 = 0+anything = anything
 - documentation - lscale in vector.h.  Ditto sparsevector
 - documentation - set, dgraph classes, and inclusion in gentype
 - documentation - changes to vector,matrix
 - documentation - mutex in basefn.h
 - documentation - removal of donothing function
 - documentation - posate function
 - documentation - many, many others

 - (low priority) implement SPEA2 and NSGA-II multi-objective optimisation algs
 - (low priority) gradient descent optimisation (using extracted nlopt library)



------------------------------------------------------------------------------------------------------------------------------

 IDEAS:

* distort and focus bayesian optimisation:

- do projection of infinite space onto finite space (stereographic)
- bayesian optimisation on resulting squished space
- iterative update to get optima in centre and scale so main area of interest is relatively undistorted




* Gradient (like Pareto EHI) for stable bayesian optimisaiton

- Look for max improvement based on *optimality* (expectation that gradient zero) rather than just return
- Sharp peaks have uncertain gradients, therefore low probability of being optimal.




* Regression with scoring (Fabrics dataset):

- familiarise self with cheng dataset running and use vector regression with scoring




* Cyclic SVM

- quick implementation of cyclic SVM if you can find a relevant dataset.  Trivially just needs n-1 planar bounds on a vector




* Crosstalk

- Multiple BO algorithms all tuning the same (4-) kernel, so they all talk to each other.  That is, multi-task BO





* GP with cut-off Gaussians (half-gaussian?): use strictly positive (or negative) prior distribution, then observe.  Makes
  the implementation of inequalities trivial





* Bi-quadratic optimisation: outer layer(s) optimise using eg SVM, inner layer optimises for 4-kernel fit

- For this we need each function that is doing kernel transfer from a base to register with that base.  Then that
  base can calculate a "reflected kernel" and do optimisation on that.

------------------------------------------------------------------------------------------------------------------------------

OLD IDEAS:

* Multi-rec: outer loop selects plane, inner loop selects points, outer loop is gradient descent, inner loop is bayesian?

- First rec is GP-UCB, rest are multi-objective recs, give wider span of exploit/explore and bound falls out trivially.
- EXTEND 1: multi-objective recs are constrained, BUT THIS DOES NOT HURT BOUND
- EXTEND 2: as per 1, but


* Auto-encoding SVMs: based around Hinton, generic SVMs, random o_q that is trained with gradient descent.  Interesting 
  to note that we can have *any* inputs as we don't need toevaluate g(x), so don't need addition or scalar multiplication


* Metallurgy problem:

- we have a multi-objective optimisation problem (maximise some phases, minimise others)
- we have constraints.  Want some phases say > 90%, others say < 5%, only interested in solutions that meet this criteria.
- proposed solution: modify the acquisition function:

if conditions met by proposed solution: acq_mod(x) = acq(x) (acquisition function unchanged)
if conditions not met: acq_mod(x) = s.acq(x), s -> 0 as time increases (acquisition function scaled toward zero)

- thus we favour points that meet the constraints
- NB: scale only the mean part, not the variance, as variance expresses an area that has yet to be explored.

*/


// +---------+------+-----+-----+-----+-----+
// | Problem | char | SVM | LSV | GPR | KNN |
// +---------+------+-----+-----+-----+-----+
// |         |      |     |     |     |     |
// | Single  |  s   |  y  |     |     |     |
// | Binary  |  c   |  y  |  y  |  y  |  y  |
// | MultiC  |  m   |  y  |     |     |  y  |
// |         |      |     |     |     |     |
// | Scalar  |  r   |  y  |  y  |  y  |  y  |
// | Vector  |  v   |  y  |  y  |  y  |  y  |
// | Anions  |  a   |  y  |  y  |  y  |  y  |
// | Gentyp  |  g   |  y  |  y  |  y  |  y  |
// |         |      |     |     |     |     |
// | Cyclic  |  u   |  y  |     |     |     |
// |         |      |     |     |     |     |
// | Densit  |  p   |  y  |     |     |  y  |
// | PFront  |  t   |  y  |     |     |     |
// |         |      |     |     |     |     |
// | BiScor  |  l   |  y  |     |     |     |
// | ScScor  |  o   |  y  |  y  |     |     |
// | Planar  |  i   |  y  |  y  |     |     |
// | MvRank  |  h   |  y  |  y  |     |     |
// | MulBin  |  j   |  y  |     |     |     |
// |         |      |     |     |     |     |
// +---------+------+-----+-----+-----+-----+
//
// Key: y == implemented
//      * == to be implemented (TO DO list)
//
// Types: SVM = support vector machine
//        LSV = least-squares support vector machine
//        GPR = gaussian process
//        KNN = K-nearest neighbour
//        BLK = learning block/glue
//        IMP = improvement block
//
// Problems: Single = single-class classifier / anomaly detection
//           Binary = binary classifier
//           MultiC = multi-class classifier
//           Gentyp = gentype regression
//           Scalar = scalar regression (real target)
//           Vector = vectorial regression (vector target)
//           Anions = anionic regression (anionic target)
//           Densit = density estimation
//           PFront = pareto front detection (1 class with gradient restrictions)
//           BiScor = scored regression (ordinal implied by scores)
//           ScScor = scalar regression with scoring (ordinal implied by scores)
//           Planar = planer-constrained vector regression
//           MvRank = multi-expert ranking
//           MulBin = multi-expert binary classification
//
//           Nopnop = does nothing
//           Consen = consensus block: output is value that has most "votes" in x
//           UsrFnA = user-defined function block, elementwise
//           UserIO = user-defined I/O block (interactive via stdin/out)
//           AveSca = averaging block: output is ave(x), restricted scalar
//           AveVec = averaging block: output is ave(x), restricted vectorial
//           AveAni = averaging block: output is ave(x), restricted anionic
//           UsrFnB = user-defined function block, vectorwise
//
// Available (unallocated) chars: bdfknwxyz
//
//
//
// Implementing new ML types
// =========================
//
// ..._...      the ML type itself
// ..._generic: any new functions required for new ... type ML
// ml_mutable:  any new functions in ..._generic need to be transferred here
//              update type list
//              isSVM... etc need to be added
//              copy/assign/transfer functions will need to be updated
// svmheavyv7:  make it visible
// svmmatlab:   make it visible
// Makefile:    add to tree
//
// Implementing new ML_Base functionality
// ======================================
//
// Any functions added to ML_Base need to be added to these headers
//
//  - ml_base.h
//  - ml_base_deref.h
//  - mlm_generic.h
//  - imp_parsvm.h
//  - lsv_gentyp.h
//  - lsv_mvrank.h
//  - lsv_planar.h
//  - svm_generic_deref.h
//  - lsv_generic_deref.h
//  - ml_mutable.h
// - lsv_generic.h (maybe)
// - blk_conect.h (maybe)
// - gpr_generic.h (maybe)
















//
// SVMHeavy CLI
//
// Version: 7
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "mlinter.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"


int cligetsetExtVar(gentype &res, const gentype &src, int num);
int cligetsetExtVar(gentype &res, const gentype &src, int num)
{
    (void) res;
    (void) src;
    (void) num;

    // Do nothing: this means you can throw logs out and not cause issues

    return -1;
}

#define LOGOUTTOFILE 1
#define LOGERRTOFILE 1

void cliCharPrintOut(char c);
void cliCharPrintErr(char c);

void cliPrintToOutLog(char c, int mode = 0);
void cliPrintToErrLog(char c, int mode = 0);


int main(int argc, char *argv[])
{
    try
    {
        // Initialisation of static, overall state, set-once type stuff

        // We do this for speed
        std::ios::sync_with_stdio(false);

        // register main thread ID
        isMainThread(1);
        // not strictly needed but good policy if threading used
        //initgentype(); // - still needed at avoid memory error at exit - UPDATE: NOT ANYMORE!
        // sets callback for calculator in god mode
        //setintercalc(&intercalc); - not needed anymore

        // Print help if no commands given

        if ( argc == 1 )
        {
            outstream() << "SVMheavy 7.0: an SVM implementation by Alistair Shilton.                      \n";
            outstream() << "============                                                                  \n";
            outstream() << "                                                                              \n";
            outstream() << "Copyright: all rights reserved.                                               \n";
            outstream() << "                                                                              \n";
            outstream() << "Usage:         svmheavyv6 {options}                                           \n";
            outstream() << "Basic help:    svmheavyv6 -?                                                  \n";
            outstream() << "Advanced help: svmheavyv6 -??                                                 \n";

            return 0;
        }

        // Register "time 0"

        TIMEABSSEC(TIMECALL);

        // Set up streams (we divert outputs to logfiles)

        void(*xcliCharPrintErr)(char c) = cliCharPrintErr;
        static LoggingOstreamErr clicerr(xcliCharPrintErr);
        seterrstream(&clicerr);

        void(*xcliCharPrintOut)(char c) = cliCharPrintOut;
        static LoggingOstreamOut clicout(xcliCharPrintOut);
        setoutstream(&clicout);

        // Convert the command line arguments into a command string

        std::string commline;

        int i;
        size_t j;
        int isquote;

        for ( i = 1 ; i < argc ; ++i )
        {
            // NB: if string contains spaces but is not enclosed in quotes then
            // dos is being a pita and stripping quotes, so reinstate them.  This
            // will not fix the problem of quotes being stripped from a string
            // not containing spaces, so you need to be wary of that later.  Nor
            // will it fix double-quoted strings

            isquote = 0;

            if ( strlen(argv[i]) )
            {
                if ( ( argv[i][0] != '\"' ) || ( argv[i][strlen(argv[i])-1] != '\"' ) )
                {
                    for ( j = 0 ; j < strlen(argv[i]) ; ++j )
                    {
                        if ( argv[i][j] == ' ' )
                        {
                            isquote = 1;
                            break;
                        }
                    }
                }
            }

            if ( isquote )
            {
                commline += '\"';
                commline += argv[i];
                commline += '\"';
            }

            else
            {
                commline += argv[i];
            }

            if ( i < argc-1 )
            {
                commline += " ";
            }
        }

        // Add -Zx to the end of the command string to ensure that the output
        // stream used by -echo will remain available until the end.

        commline += " -Zx";
//        outstream() << "Running command: " << commline << "\n";

        // Define global variable store

        svmvolatile SparseVector<SparseVector<gentype> > globargvariables;

        // Construct command stack.  All commands must be in awarestream, which
        // is similar to a regular stream but can supply commands from a
        // variety of different sources: for example a string (as here), a stream
        // such as standard input, or various ports etc.  You can then open
        // further awarestreams, which are stored on the stack, with the uppermost
        // stream being the active stream from which current commands are sourced.

        Stack<awarestream *> *commstack;
        MEMNEW(commstack,Stack<awarestream *>);
        std::stringstream *commlinestring;
        MEMNEW(commlinestring,std::stringstream(commline));
        awarestream *commlinestringbox;
        MEMNEW(commlinestringbox,awarestream(commlinestring,1));
        commstack->push(commlinestringbox);

        int svmInd = 1; // not 0 anymore - want to reserve that index for other stuff! 0;
        static thread_local SVMThreadContext *svmContext;
        MEMNEW(svmContext,SVMThreadContext(svmInd));
        errstream() << "{";

        // Now that everything has been set up so we can run the actual code.

        SparseVector<SparseVector<int> > returntag;

        runsvm(svmContext,commstack,globargvariables,cligetsetExtVar,returntag);

        // Unlock the thread, signalling that the context can be deleted etc

        errstream() << "}";

        MEMDEL(commstack); commstack = nullptr;

        // Code not re-entrant, so need to blitz threads, and also delete remaining MLs

        errstream() << "Killing dangling threads.\n";

        killallthreads(svmContext);

        errstream() << "Clearing memory.\n";

        deleteMLs();
        deletegrids();
        deleteDIRects();
        deleteNelderMeads();
        deleteBayess();

        cliPrintToOutLog('*',1);
        cliPrintToErrLog('*',1);

        isMainThread(0);
    }

    catch ( const char *errcode )
    {
        errstream() << "Unknown error: " << errcode << ".\n";
        return 1;
    }

    catch ( const std::string errcode )
    {
        errstream() << "Unknown error: " << errcode << ".\n";
        return 1;
    }

    return 0;
}








void cliCharPrintOut(char c)
{
    cliPrintToOutLog(c);

#ifndef HEADLESS
    if ( !LoggingOstreamOut::suppressStreamCout )
    {
        std::cout << c;
    }
#endif

    return;
}

void cliCharPrintErr(char c)
{
    cliPrintToErrLog(c);

#ifndef HEADLESS
    if ( !LoggingOstreamErr::suppressStreamCout )
    {
        std::cerr << c;
    }
#endif

    return;
}

void cliPrintToOutLog(char c, int mode)
{
    // mode = 0: print char
    //        1: close file for exit

    if ( LOGOUTTOFILE )
    {
        static std::ofstream *outlog = nullptr;

        if ( !mode && !outlog )
        {
            outlog = new std::ofstream;

            NiceAssert(outlog);

            std::string outfname("svmheavy.out.log");
            std::string outfnamebase("svmheavy.out.log");

            int fcnt = 0;

            while ( fileExists(outfname) )
            {
                ++fcnt;

                std::stringstream ss;

                ss << outfnamebase;
                ss << ".";
                ss << fcnt;

                outfname = ss.str();
            }

            (*outlog).open(outfname.c_str());
        }

        if ( mode )
        {
            if ( outlog )
            {
                (*outlog).close();
                delete outlog;
                outlog = nullptr;
            }
        }

        else if ( outlog && !LoggingOstreamOut::suppressStreamFile )
        {
            static int bstring = 0;

            if ( c != '\b' )
            {
                bstring = 0;

                (*outlog) << c;
                (*outlog).flush();
            }

            else if ( !bstring )
            {
                bstring = 1;

                (*outlog) << '\n';
                (*outlog).flush();
            }
        }
    }

    return;
}

void cliPrintToErrLog(char c, int mode)
{
    // mode = 0: print char
    //        1: close file for exit

    if ( LOGERRTOFILE )
    {
        static std::ofstream *errlog = nullptr;

        if ( !mode && !errlog )
        {
            errlog = new std::ofstream;

            NiceAssert(errlog);

            std::string errfname("svmheavy.err.log");
            std::string errfnamebase("svmheavy.err.log");

            int fcnt = 0;

            while ( fileExists(errfname) )
            {
                ++fcnt;

                std::stringstream ss;

                ss << errfnamebase;
                ss << ".";
                ss << fcnt;

                errfname = ss.str();
            }

            (*errlog).open(errfname.c_str());
        }

        if ( mode )
        {
            if ( errlog )
            {
                (*errlog).close();
                delete errlog;
                errlog = nullptr;
            }
        }

        else if ( errlog && !LoggingOstreamErr::suppressStreamFile )
        {
            static int bstring = 0;

            if ( c != '\b' )
            {
                bstring = 0;

                (*errlog) << c;
                (*errlog).flush();
            }

            else if ( !bstring )
            {
                bstring = 1;

                (*errlog) << '\n';
                (*errlog).flush();
            }
        }
    }

    return;
}

