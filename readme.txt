Compilation
===========

basic:     make
optimized: make optmath
debug:     make debug

Executable generated: svmheavyv7.exe

Other options are available: see makefile for information.  Currently it's 
set up for linux/gcc, but other compilers are supported - again, see 
documenation in makefile (in essence uncomment just the one you want to use).
Also you can tune compiler feature use in basefn.hpp (but you probably won't
need to).

Dependencies: I've tried to avoid where possible.  The only one you might want
is gnuplot - the code will work without it, but if it's available then you can
generate nice plots and visualizations as you go.

You can also compile and link to MATLAB using the mex compiler.  Use the 
mexmake.m script.  Depending on mex version you might need to mess with some
flags and/or disable some post-c++20 features (you can do this in basefn.hpp).

Python linkage requires swig and it set up for python 3.5.  You'll probably 
need to mess around with the makefile to set source/library directories to make
this work.



Operation
=========

SVMHeavy is actually a (kernel-based) machine learning library with Bayesian 
optimization library I started writing about 20 years ago during my PhD with 
Bayesian Optimization built onto it later.

There is documentation in instruct.txt (or run svmheavyv7.exe -??).  The part 
relevant the Bayesian Optimization starts on line 2281.  Here are some usage 
examples:

- Quick "is it working" test (classification task):

  ./svmheavyvy.exe -c 1 -kt 3 -kg sqrt\(20\) -tc 5 -AA tr1s.txt



- BO: simple GP-UCB example:

  ./svmheavyv7.exe -gmd 0.01 -gbH 3 -gb 2 "-tM testfn(2002,[ y z ])" "-echo y -echo z" fb 1 0 1 1 fb 2 0 1 1

  This runs BO using the GP-UCB acquisition function.  Flags here are:

  o gmd 0.01: expected error (noise) in the function being optimized.  In this 
    case the function being optimized is the 2d Ackley test function (see 
    instruct.txt, line 722), so this is nominal.
  o -gbH 3 selects the GP-UCB optimizer (see instruct.txt, line 3070)
  o -gb 2 ... optimize 2d function
    ... "-tM testfn(2002,[ y z ])" ... using test function 2, normalized to 
                                       [0,1] input range, [0,1] output range
    ... fb 1 0 1 1 ... where the first optimization variable (first 1 in the 
                       expression) is var(0,1) (shorthand y), varies linearly 
                       (fb) ranging from 0 to 1 (second and third numbers).
                       The final 1 is ununsed in BO but required.
    ... fb 2 0 1 1     specifies the second variable (var(0,2), which is z)
                       its range etc



- BO: simple EI example:

  ./svmheavyv7.exe -gmd 0.01 -gbH 1 -gb 2 "-tM testfn(2002,[ y z ])" "-echo y -echo z" fb 1 0 1 1 fb 2 0 1 1

  same as the first example, except that it uses the EI (expected improvement) 
  acquisition function, as specified by -gbH 1 (see instruct.txt).


(other acquisition functions are available using the -gbH switch, see documentation around line 3070)



- BO-Muse: this example uses BO-Muse for the same task:

  ./svmheavyv7.exe -gmd 0.01 -gbH 11 -gbv \[ [ 20 ] [ 18 ] \] -gb 2 "-tM testfn(2002,[ y z ])" "-echo y -echo z" fb 1 0 1 1 fb 2 0 1 1

  In this example we use -gbH 11 to specify that we want to use multiple 
  acquisition strategies.  The strategies used are specified by the -gbv
  flag: in this case 20 (which is the BO-Muse tuned GP-UCB acquisition 
  function) and 18 (which prompts the user for which point to test).




- Multi-fidelity test: here we reproduce the Currin function test using BOCA in
  Kandasami's multi-fidelity paper (split over multiple lines for readability):

  ./svmheavyv7.exe -L res005 -gba 10 -gbb 100 -gmr -gmd '0.5/(14*14)' -gbH 3 
  -gbfid 3 -gbfp '0.1+(x*x)' -gbfb 50*1.1 -gbfn 1 
  -gmks 2 -gmki 0 -gmkS -gmkt 3 -gmki 1 -gmkt 3 -gmki 0 
  -gb 3 '"-tM' '1-(1-(0.1*(1-v)*exp(-1/(2*z))))*(((2300*y*y*y)+(1900*y*y)+(2092*y)+60)/((100*y*y*y)+(500*y*y)+(4*y)+20))/14' -echo y -echo z -echo 'v"' 
  '"-echo' y -echo z -echo 'v"' fb 1 0 1 1 fb 2 0 1 1 fb 3 0 1 1

  This is a little more involved, so flag-by-flag:
  o -gba 10 -gbb 100          Seed RNGs for repeatable results
  o -gmr -gmd '0.5/(14*14)'   Add artificial noise to the observations with variance 0.5/(14*14)
  o -gbH 3                    Use GP-UCB acquisition function
  o -gbfid 3                  Specifies a multi-fidelity problem with 1 fidelity variable taking 3 possible values (1/3, 2/3 and 1) - instruct.txt, line 3228
  o -gbfp '0.1+(x*x)'         Fidelity penalty for given fidelity (here written as x, ranging [0,1]), as specified in Kandasami's paper.
  o -gbfb 50*1.1              Available fidelity budget.
  o -gbfn 1                   Specifies number of fidelity variables (the fidelity variables are the last whatever variables in the problem) 
  o -gmks 2                   GP kernel has two "parts", where...
  o -gmki 0                   ...the first part...
  o -gmkS                     ...is multiplied with the second part...
  o -gmkt 3                   ...and is an SE kernel...
  o -gmki 1                   ...and the second part...
  o -gmkt 3                   ...is an SE kernel...
  o -gmki 0                   ...and we need to revert back to the first part.
                              (see instruct.txt, after line 1805, for more information.  The prefix "gm" here means "kernel used in BO GP model".
  o -gb 3...                  This specifies the problem as before.  Note that there are 3 variables are: y = var(0,1), z = var(0,2) and v = var(0,3)
                              The optimization problem is the Currin problem from Kandasami's paper, rescaled for convenience.





- Multi-fidelity test with human override: this is the same as the previous 
  example, but with human override option to choose a different fidelity 
  (experimental, currently no proof of convergence).  Node the -gbfo 1 switch.

  ./svmheavyv7.exe -L res005 -gba 10 -gbb 100 -gmr -gmd '0.5/(14*14)' -gbH 3 
  -gbfid 3 -gbfp '0.1+(x*x)' -gbfb 50*1.1 -gbfn 1 -gbfo 1
  -gmks 2 -gmki 0 -gmkS -gmkt 3 -gmki 1 -gmkt 3 -gmki 0 
  -gb 3 '"-tM' '1-(1-(0.1*(1-v)*exp(-1/(2*z))))*(((2300*y*y*y)+(1900*y*y)+(2092*y)+60)/((100*y*y*y)+(500*y*y)+(4*y)+20))/14' -echo y -echo z -echo 'v"' 
  '"-echo' y -echo z -echo 'v"' fb 1 0 1 1 fb 2 0 1 1 fb 3 0 1 1



- Finally, this example combines BO-Muse with BOCA and fidelity override (for BO-Muse + BOCA without override, remove the -gbfo 1 flag:

  ./svmheavyv7.exe -L res005 -gba 10 -gbb 100 -gmr -gmd '0.5/(14*14)' -gbH 11 -gbv \[ [ 20 ] [ 18 ] \]
  -gbfid 3 -gbfp '0.1+(x*x)' -gbfb 50*1.1 -gbfn 1 -gbfo 1
  -gmks 2 -gmki 0 -gmkS -gmkt 3 -gmki 1 -gmkt 3 -gmki 0 
  -gb 3 '"-tM' '1-(1-(0.1*(1-v)*exp(-1/(2*z))))*(((2300*y*y*y)+(1900*y*y)+(2092*y)+60)/((100*y*y*y)+(500*y*y)+(4*y)+20))/14' -echo y -echo z -echo 'v"' 
  '"-echo' y -echo z -echo 'v"' fb 1 0 1 1 fb 2 0 1 1 fb 3 0 1 1







Interfacing
===========

(Note: the following are in the code, but it's been a while since I used
either MATLAB or python interfacing.  If you have any problems just email
me and I'll figure it out).

The code can be compiled to run under MATLAB using mex as described above, 
which allows direct interface with MATLAB-native libraries.  There's some
discussion of how to do this in instruct.txt around line 2361.  This is
probably the most well-defined interface in the code.

You can interface with python similarly, though it's not as well documented.
If you just want to ask for the user to come back with result then there
is the IO block (instruct.txt, line 581).  In either case you can email me
and I'll set up a documented example of usage.







Coding Notes
============

A good amound of the older code here very much predates c++11 and the STL, 
which is why there's a bit of apparent wheel-reinvention (eg vector.h).  In
general the filenames refer to operation (eg chol.hpp is a Cholesky 
factorization class), function (eg bayesopt.hpp is the Bayesian optimization
header) or learning block (eg SVM_Scalar is the support vector machine block
for scalar targets).  mlinter is a somewhat messy frontend, best avoided.









