#include .env - if this worked it would make python much easier
#consider -fuse-ld=mold or -fuse-ld=gold

# Makefile for SVMHeavy v6

# Available targets:
#
# basic: no optimisation, no debugging (default)
# opt: turn on -O3 optimisations
# optmath: turn on -O3 -ffast_math optimisation
# optavx2: turn on -O3 and avx256 intrinsics optimisation
# optmax: turn on all above optimisations
# pybasic: python version
# pyopt: python version
# pyoptmath: python version
# pyoptavx2: python version
# pyoptmax: python version
# debug: basic debugging (-g)
# debugmore: basic debugging + optimiser debugging
# debugdeep: basic debugging + deep optimiser debugging
# debugaddrsan: basic debugging plus address sanitizer
# clean: remove .o files
#
# - Need to uncomment component relevant to target system.
# - New files need to be included in SRC, the DEP tree and a recipe
#
# MLMUTATEFILE: touched to indicate changes in ML tree as make
#     churns and dies on the dependency tree otherwise.
# CBASEFLAGS: over-ride target specifics in basefn.h
# DEBUG_MEM: adds in some extra checks on memory new/delete operations


# absolute bare-bones version - threads, sockets and keyboard interupts disabled
# (for djgpp bare-bones use gxx and del)

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DDISABLE_KB_BY_DEF
#BIGFLAG = -Wall
#LIBFLAGS = -lm

# dos/djgpp version - no sockets or threads but has keyboard interupts

#CC = gxx
#CCC = gxx
#RM = del
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DDJGPP_MATHS -DHAVE_CONIO
#BIGFLAG = -Wall
#LIBFLAGS = -lm












# cygwin/gcc version modern - threadless and socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#BIGFLAG = -Wa,-mbig-obj
#LIBFLAGS = -lm

# cygwin/gcc version modern

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DHAVE_IOCTL -DIS_CPP11 -std=c++11 
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#BIGFLAG = -Wa,-mbig-obj
#LIBFLAGS = -lm

# cygwin/gcc version modern - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#BIGFLAG = -Wa,-mbig-obj
#LIBFLAGS = -lm

# cygwin/gcc version modern - threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#BIGFLAG = -Wa,-mbig-obj
#LIBFLAGS = -lm














# linux/gcc - socketless and threadless

CC = c++
CCC = c++
#CC = g++
#CCC = g++
RM = rm -f
MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS -DDISABLE_KB_BY_DEF -DHAVE_IOCTL -DNOPURGE -DIS_CPP20 -std=c++2a -Wno-overloaded-virtual
CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS -DHAVE_IOCTL -DNOPURGE -DIS_CPP20 -std=c++2a -Wno-overloaded-virtual
#BIGFLAG = -Wa,-mbig-obj
BIGFLAG =
LIBFLAGS = -lm

# linux/gcc version

###CC = clang++
###CCC = clang++
#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DENABLE_THREADS -DHAVE_TERMIOS -DHAVE_IOCTL -DNOPURGE -DIS_CPP20 -std=c++2a -Wno-overloaded-virtual
##BIGFLAG = -Wa,-mbig-obj
#BIGFLAG = 
#LIBFLAGS = -pthread -lm

# linux/gcc - threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DHAVE_TERMIOS -DHAVE_IOCTL -DNOPURGE -DIS_CPP11 -std=c++11
#BIGFLAG = -Wa,-mbig-obj
#LIBFLAGS = -lm

# linux/gcc - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS -DHAVE_IOCTL -DNOPURGE -DIS_CPP20 -std=c++2a -Wno-overloaded-virtual
##BIGFLAG = -Wa,-mbig-obj
#BIGFLAG = 
#LIBFLAGS = -pthread -lm














# visual studio version - socketless and threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DHAVE_CONIO
#BIGFLAG = -big-obj
#LIBFLAGS = -lm

# visual studio version

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DHAVE_CONIO -DENABLE_THREADS -DALLOW_SOCKETS
#BIGFLAG = -big-obj
#LIBFLAGS = -lm

# visual studio version - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DENABLE_THREADS -DHAVE_CONIO
#BIGFLAG = -big-obj
#LIBFLAGS = -lm

# visual studio version - threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DHAVE_CONIO -DALLOW_SOCKETS
#BIGFLAG = -big-obj
#LIBFLAGS = -lm

# visual studio (older) version

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECFN_ASUPP -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DVISUAL_STU_OLD -DVISUAL_STU_NOERF -DHAVE_CONIO -DENABLE_THREADS -DALLOW_SOCKETS
#BIGFLAG = -big-obj
#LIBFLAGS = -lm

# visual studio (older) version - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DVISUAL_STU_OLD -DVISUAL_STU_NOERF -DHAVE_CONIO -DENABLE_THREADS 
#BIGFLAG = -big-obj
#LIBFLAGS = -lm















# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version)

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM ALLOW_SOCKETS ENABLE_THREADS IS_CPP11
#LIBFLAGS = -lm

# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version) - threadless

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM ALLOW_SOCKETS IS_CPP11
#LIBFLAGS = -lm

# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version) - socketless

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM ENABLE_THREADS IS_CPP11
#LIBFLAGS = -lm

# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version) - threadless and socketless

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM IS_CPP11
#LIBFLAGS = -lm










# source files

SRC = basefn.cc randfun.cc clockbase.cc memdebug.cc gslrefs.cc zerocross.cc \
      numbase.cc anion.cc dynarray.cc strfns.cc opttest.cc paretotest.cc idstore.cc smatrix.cc gentype.cc mercer.cc mlcommon.cc kcache.cc FNVector.cc \
      hyper_psc.cc hyper_debug.cc hyper_base.cc hyper_opt.cc hyper_alt.cc \
      optcontext.cc optlinbasecontext.cc optlincontext.cc optlinstate.cc \
      sQbase.cc sQsLsAsWs.cc sQsmo.cc sQd2c.cc sQgraddesc.cc linsolve.cc \
      adam.cc \
      nlopt_base.cc nlopt_direct.cc nlopt_neldermead.cc nlopt_slsqp.cc \
      ml_base.cc \
      svm_generic.cc \
      svm_scalar.cc svm_binary.cc svm_single.cc svm_biscor.cc svm_scscor.cc svm_densit.cc svm_pfront.cc svm_scalar_rff.cc svm_binary_rff.cc \
      svm_vector_atonce_template.cc \
      svm_multic_redbin.cc svm_multic_atonce.cc svm_vector_redbin.cc svm_vector_atonce.cc svm_vector_mredbin.cc svm_vector_matonce.cc \
      svm_multic.cc svm_vector.cc svm_anions.cc svm_gentyp.cc svm_planar.cc svm_mvrank.cc svm_mulbin.cc \
      svm_simlrn.cc svm_cyclic.cc svm_kconst.cc \
      knn_generic.cc \
      knn_scalar.cc knn_binary.cc knn_multic.cc knn_vector.cc knn_anions.cc knn_gentyp.cc knn_densit.cc \
      blk_generic.cc \
      blk_nopnop.cc blk_consen.cc blk_avesca.cc blk_avevec.cc blk_aveani.cc blk_usrfna.cc blk_usrfnb.cc blk_userio.cc blk_calbak.cc \
      blk_mexfna.cc blk_mexfnb.cc blk_mercer.cc blk_conect.cc blk_system.cc blk_kernel.cc blk_bernst.cc blk_batter.cc \
      imp_generic.cc \
      imp_expect.cc imp_parsvm.cc imp_rlsamp.cc imp_nlsamp.cc \
      lsv_generic.cc \
      lsv_scalar.cc lsv_vector.cc lsv_anions.cc lsv_scscor.cc lsv_gentyp.cc lsv_planar.cc lsv_mvrank.cc lsv_binary.cc lsv_scalar_rff.cc \
      gpr_generic.cc \
      gpr_scalar.cc gpr_vector.cc gpr_anions.cc gpr_gentyp.cc gpr_binary.cc gpr_scalar_rff.cc gpr_binary_rff.cc \
      mlm_generic.cc \
      mlm_scalar.cc mlm_binary.cc mlm_vector.cc \
      ml_mutable.cc \
      fuzzyml.cc makemonot.cc plotbase.cc plotml.cc xferml.cc errortest.cc addData.cc analyseAnomaly.cc balc.cc hillclimb.cc \
      globalopt.cc gridopt.cc directopt.cc nelderopt.cc smboopt.cc bayesopt.cc \
      awarestream.cc \
      mlinter.cc
OBJS = $(SRC:.cc=.o)








# Flags and stuff

#Still working through to add -Wshadow
#BASEFLAGS = -W -Wall -Wextra -Wconversion -Wpedantic
BASEFLAGS = -W -Wall -Wextra -Wconversion -Wpedantic -DDISABLE_KB_BY_DEF
#for noise, use -Weffc++
MEHFLAGS = -DIGNOREMEM -DNDEBUG
LIBFLAG = -lm
#OPTFLAGS = -O3 -DIGNOREMEM -DNDEBUG -march=native -mtune=native - NOTE THAT -march=native CAUSES A SUBTLE BUG THAT SOMETIMES (NOT ALWAYS) SCREWS WITH NUMERICAL RESULTS
#FOR EXAMPLE THE FOLLOWING DOES NOT WORK IF -march=native IS USED!: ./svmheavyv7.exe -qw 3 -z d -kt 2 -kd 3 -Zx -qw 1 -Zx -R q -c 10 -kt 801 -ktx 3 -AA xor.txt -AA xor.txt -Zx -qw 2 -z r -Zx -R q -w 0 -c 10 -kt 801 -ktx 3 -AA and.txt -AA and.txt -Zx -qw 3 -Zx -xl -1 -xs 0 -xr 2 -xi 200 -xo 3 -xC 2 -x 20 [ 1 2 ] [ 1 1 ] -Zx -qw 3 -s temp3.svm -Zx -qw 1 -s temp1.svm -Zx -qw 2 -s temp2.svm
#NOTE THAT YOU CAN USE -march=native EVERYWHERE EXCEPT FOR THE COMPILATION OF basefn.h AND EVERYTHING WILL WORK JUST FINE.
OPTFLAGS = -O3 -DIGNOREMEM -DNDEBUG -mtune=native
#OPTFLAGSMATHS = -ffast-math - don't use this!!!!!!  Parts of nlopt rely on isinf, and this flag makes isinf non-functional! Also exp(-inf) can be -inf if this flag is set, and mercer.cc relies on exp(-inf) = 0 when dealing with infinite sets!
OPTFLAGSMATHS =
AVX2FLAGS = -DUSE_PMMINTRIN
DEBUGFLAGS0 = -g
DEBUGFLAGSA = -g -DDEBUG_MEM_CHEAP
DEBUGFLAGSAA = -g -DDEBUG_MEM
#DEBUGFLAGSA = -g -DDEBUG_MEM
DEBUGFLAGSB = -DDEBUGOPT
DEBUGFLAGSC = -DDEBUGDEEP
DEBUGFLAGSX = -g -fsanitize=undefined -fsanitize=address
TARGNAME = svmheavyv7.exe
TARGSRC = svmheavyv7.cc
#PYFLAGS = -fPIC -DMAKENUMPYCOMP -fpermissive
#PYTHONI = -I/usr/include/python3.8/ -I/usr/local/lib/python3.8/dist-packages/numpy/core/include/
#PYTHONL = -Xlinker -export-dynamic
PYFLAGS = -shared -fPIC
PYTHONI = -I/usr/include/python3.8 -I/home/ashilton/.local/lib/python3.8/site-packages/pybind11/include
#PYTHONI = $(python3 -m pybind11 --includes)
PYTARGNAME = pyheavy.cpython-38-x86_64-linux-gnu.so
#PYTARGNAME = pyheavy$(python3-config --extension-suffix)
PYTARGSRC = pyheavy.cc
PYTHONL =
#headless blocks all cout/cerr and system calls for running in batch mode
HEADLESSFLAGS = -DHEADLESS
USERLESSFLAGS = -DUSERLESS



# python working version c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cc -o example$(python3-config --extension-suffix)
#echo $(python3 -m pybind11 --includes)
#-I/usr/include/python3.8 -I/home/ashilton/.local/lib/python3.8/site-packages/pybind11/include
#
#echo $(python3-config --extension-suffix)
#.cpython-38-x86_64-linux-gnu.so








# server - optmath and headless combined

julian: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(OPTFLAGSMATHS) $(USERLESSFLAGS)
julian: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

optmath: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(OPTFLAGSMATHS)
optmath: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

server: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(OPTFLAGSMATHS) $(HEADLESSFLAGS)
server: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

basic: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS)
basic: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

exeonly: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS)
exeonly: $(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

meh: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(MEHFLAGS)
meh: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

opt: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS)
opt: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

optavx2: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(AVX2FLAGS)
optavx2: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

optmax: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(AVX2FLAGS) $(OPTFLAGSMATHS)
optmax: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)







pydebugmore: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSAA) $(PYTHONI) $(PYFLAGS) -DPYLOCAL
pydebugmore: $(OBJS)
	$(CCC) $(PYTHONI) $(PYTARGSRC) -o $(PYTARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS) $(PYTHONL)

pydebugbase: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGS0) $(PYTHONI) $(PYFLAGS) -DPYLOCAL
pydebugbase: $(OBJS)
	$(CCC) $(PYTHONI) $(PYTARGSRC) -o $(PYTARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS) $(PYTHONL)

pybasic: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(PYTHONI) $(PYFLAGS) -DPYLOCAL
pybasic: $(OBJS)
	$(CCC) $(PYTHONI) $(PYTARGSRC) -o $(PYTARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS) $(PYTHONL)

pyoptmath: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(OPTFLAGSMATHS) $(PYTHONI) $(PYFLAGS) -DPYLOCAL
pyoptmath: $(OBJS)
	$(CCC) $(PYTHONI) $(PYTARGSRC) -o $(PYTARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS) $(PYTHONL)

pyserver: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(OPTFLAGSMATHS) $(PYTHONI) $(PYFLAGS) $(HEADLESSFLAGS) -DPYLOCAL
pyserver: $(OBJS)
	$(CCC) $(PYTHONI) $(PYTARGSRC) -o $(PYTARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS) $(PYTHONL)





#pybasic: CFLAGS = $(PYFLAGS) $(CBASEFLAGS) $(PYTHONI)
#pybasic: $(OBJS)
#	swig -c++ -python -o ml_mutable_wrap.cxx ml_mutable.i
#	$(CCC) $(CFLAGS) $(PYTHONI) -c ml_mutable_wrap.cxx -o ml_mutable_wrap.o
#	$(CCC) $(PYTHONL) $(LIBFLAGS) -shared $(OBJS) ml_mutable_wrap.o -o _ml_mutable.so
#
#pyopt: CFLAGS = $(PYFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(PYTHONI)
#pyopt: $(OBJS)
#	swig -c++ -python -o ml_mutable_wrap.cxx ml_mutable.i
#	$(CCC) $(CFLAGS) $(PYTHONI) -c ml_mutable_wrap.cxx -o ml_mutable_wrap.o
#	$(CCC) $(PYTHONL) $(LIBFLAGS) -shared $(OBJS) ml_mutable_wrap.o -o _ml_mutable.so
#
#pyoptmath: CFLAGS = $(PYFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(OPTFLAGSMATHS) $(PYTHONI)
#pyoptmath: $(OBJS)
#	swig -c++ -python -o ml_mutable_wrap.cxx ml_mutable.i
#	$(CCC) $(CFLAGS) $(PYTHONI) -c ml_mutable_wrap.cxx -o ml_mutable_wrap.o
#	$(CCC) $(PYTHONL) $(LIBFLAGS) -shared $(OBJS) ml_mutable_wrap.o -o _ml_mutable.so
#
#pyoptavx2: CFLAGS = $(PYFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(AVX2FLAGS) $(PYTHONI)
#pyoptavx2: $(OBJS)
#	swig -c++ -python -o ml_mutable_wrap.cxx ml_mutable.i
#	$(CCC) $(CFLAGS) $(PYTHONI) -c ml_mutable_wrap.cxx -o ml_mutable_wrap.o
#	$(CCC) $(PYTHONL) $(LIBFLAGS) -shared $(OBJS) ml_mutable_wrap.o -o _ml_mutable.so
#
#pyoptmax: CFLAGS = $(PYFLAGS) $(CBASEFLAGS) $(OPTFLAGS) $(AVX2FLAGS) $(OPTFLAGSMATHS) $(PYTHONI)
#pyoptmax: $(OBJS)
#	swig -c++ -python -o ml_mutable_wrap.cxx ml_mutable.i
#	$(CCC) $(CFLAGS) $(PYTHONI) -c ml_mutable_wrap.cxx -o ml_mutable_wrap.o
#	$(CCC) $(PYTHONL) $(LIBFLAGS) -shared $(OBJS) ml_mutable_wrap.o -o _ml_mutable.so




debugbase: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGS0)
debugbase: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debug: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA)
debug: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugaddrsan: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSX)
debugaddrsan: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS) -static-libasan

debugalt: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSB)
debugalt: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugmore: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA) $(DEBUGFLAGSB)
debugmore: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugopt: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA) $(DEBUGFLAGSB)
debugopt: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugoptdeep: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSB) $(DEBUGFLAGSC)
debugoptdeep: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugdeep: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA) $(DEBUGFLAGSB) $(DEBUGFLAGSC)
debugdeep: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)




.PHONY: clean

clean: 
	$(RM) *.o

cleansvm: 
	$(RM) svm_*.o

mex: CFLAGS = 
mex: $(OBJS)
	$(CCC) svmmatlab.cpp *.obj




# Define macros to simplify include trees for header files only.  Each
# macro specifies the dependencies in the relevant header in terms of
# both headers included and previous macros for headers included from
# said header.  Headers with no dependencies are not defined here

DYNARRAYDEP     = basefn.hpp memdebug.hpp niceassert.hpp qswapbase.hpp
VECTORDEP       = numbase.hpp dynarray.hpp qswapbase.hpp $(DYNARRAYDEP)
SPARSEVECTORDEP = strfns.hpp vector.hpp $(VECTORDEP) 
DICTDEP         = vector.hpp $(VECTORDEP) 
MATRIXDEP       = sparsevector.hpp $(SPARSEVECTORDEP)
VECSTACKDEP     = vector.hpp $(VECTORDEP)
NONZEROINTDEP   = basefn.hpp memdebug.hpp niceassert.hpp
SETDEP          = matrix.hpp $(MATRIXDEP)
DGRAPHDEP       = matrix.hpp $(MATRIXDEP)
ADAMDEP         = vector.hpp $(VECTORDEP)


NUMBASEDEP    = basefn.hpp memdebug.hpp niceassert.hpp gslrefs.hpp
ANIONDEP      = basefn.hpp memdebug.hpp niceassert.hpp numbase.hpp $(NUMBASEDEP)
PARETOTESTDEP = vector.hpp $(VECTORDEP)
OPTTESTDEP    = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP)
IDSTOREDEP    = nonzeroint.hpp $(NONZEROINTDEP) sparsevector.hpp $(SPARSEVECTORDEP)
SMATRIXDEP    = matrix.hpp $(MATRIXDEP)
GENTYPEDEP    = basefn.hpp memdebug.hpp niceassert.hpp numbase.hpp $(NUMBASEDEP) anion.hpp $(ANIONDEP) \
                vector.hpp $(VECTORDEP) sparsevector.hpp $(SPARSEVECTORDEP) \
                matrix.hpp $(MATRIXDEP) set.hpp $(SETDEP) svmdict.hpp $(DICTDEP) dgraph.hpp $(DGRAPHDEP) \
                opttest.hpp $(OPTTESTDEP) paretotest.hpp $(PARETOTESTDEP)
MERCERDEP     = vector.hpp $(VECTORDEP) sparsevector.hpp $(SPARSEVECTORDEP) \
                matrix.hpp $(MATRIXDEP) gentype.hpp $(GENTYPEDEP) numbase.hpp
MLCOMMONDEP   = sparsevector.hpp $(SPARSEVECTORDEP) anion.hpp $(ANIONDEP) \
                mercer.hpp $(MERCERDEP)
KCACHEDEP     = vector.hpp $(VECTORDEP) gentype.hpp $(GENTYPEDEP)
FNVECTORDEP   = vector.hpp $(VECTORDEP) sparsevector.hpp $(SPARSEVECTORDEP) \
                gentype.hpp $(GENTYPEDEP) mercer.hpp $(MERCERDEP) \
                mlcommon.hpp $(MLCOMMONDEP)

CHOLDEP = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) mlcommon.hpp $(MLCOMMONDEP)

OPTCONTEXTDEP        = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) \
                       chol.hpp $(CHOLDEP) numbase.hpp $(NUMBASEDEP)
OPTLINBASECONTEXTDEP = optcontext.hpp $(OPTCONTEXTDEP)
OPTLINCONTEXTDEP     = optlinbasecontext.hpp $(OPTLINBASECONTEXTDEP)
OPTLINSTATEDEP       = optlincontext.hpp $(OPTLINCONTEXTDEP) mlcommon.hpp $(MLCOMMONDEP)
OPTSTATEDEP          = optcontext.hpp $(OPTCONTEXTDEP) mlcommon.hpp $(MLCOMMONDEP)

SQBASEDEP     = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) optstate.hpp $(OPTSTATEDEP) \
                smatrix.hpp $(SMATRIXDEP)
SQSLSQSWSDEP  = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) optstate.hpp $(OPTSTATEDEP) \
                sQbase.hpp $(SQBASEDEP)
SQSMODEP      = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) optstate.hpp $(OPTSTATEDEP) \
                sQbase.hpp $(SQBASEDEP)
SQD2CDEP      = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) optstate.hpp $(OPTSTATEDEP) \
                sQbase.hpp $(SQBASEDEP)
SQSMOVECTDEP  = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) optstate.hpp $(OPTSTATEDEP) \
                smatrix.hpp $(SMATRIXDEP) zerocross.hpp sQbase.hpp $(SQBASEDEP)
SQGRADDESCDEP = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) optstate.hpp $(OPTSTATEDEP) \
                sQbase.hpp $(SQBASEDEP) 
LINSOLVEDEP   = vector.hpp $(VECTORDEP) matrix.hpp $(MATRIXDEP) optstate.hpp $(OPTSTATEDEP) \
                sQbase.hpp $(SQBASEDEP)

NLOPT_BASEDEP       = basefn.hpp memdebug.hpp niceassert.hpp
NLOPT_DIRECTDEP     = basefn.hpp memdebug.hpp niceassert.hpp nlopt_base.hpp
NLOPT_NELDERMEADDEP = basefn.hpp memdebug.hpp niceassert.hpp nlopt_base.hpp
NLOPT_SLSQPDEP      = basefn.hpp memdebug.hpp niceassert.hpp nlopt_base.hpp

ML_BASEDEP = vector.hpp $(VECTORDEP) sparsevector.hpp $(SPARSEVECTORDEP) \
             matrix.hpp $(MATRIXDEP) mercer.hpp $(MERCERDEP) \
             set.hpp $(SETDEP) gentype.hpp $(GENTYPEDEP) \
             mlcommon.hpp $(MLCOMMONDEP) numbase.hpp $(NUMBASEDEP) \
             FNVector.hpp $(FNVECTORDEP) basefn.hpp memdebug.hpp niceassert.hpp globalopt_base.hpp

SVM_GENERICDEP = ml_base.hpp $(ML_BASEDEP)

SVM_SCALARDEP = svm_generic.hpp $(SVM_GENERICDEP) kcache.hpp $(KCACHEDEP) \
                optstate.hpp $(OPTSTATEDEP) sQgraddesc.hpp $(SQGRADDESCDEP)
SVM_BINARYDEP = svm_scalar.hpp $(SVM_SCALARDEP)
SVM_SIMLRNDEP = svm_scalar.hpp $(SVM_SCALARDEP)
SVM_SINGLEDEP = svm_binary.hpp $(SVM_BINARYDEP)
SVM_BISCORDEP = svm_binary.hpp $(SVM_BINARYDEP)
SVM_SCALAR_RFFDEP = svm_scalar.hpp $(SVM_SCALARDEP) adam.hpp $(ADAMDEP)
SVM_BINARY_RFFDEP = svm_scalar_rff.hpp $(SVM_SCALAR_RFFDEP)
SVM_SCSCORDEP = svm_scalar.hpp $(SVM_SCALARDEP)
SVM_DENSITDEP = svm_scalar.hpp $(SVM_SCALARDEP)
SVM_PFRONTDEP = svm_binary.hpp $(SVM_BINARYDEP)

SVM_VECTOR_ATONCE_TEMPLATEDEP = svm_generic.hpp $(SVM_GENERICDEP) kcache.hpp $(KCACHEDEP) \
                                optstate.hpp $(OPTSTATEDEP) sQsLsAsWs.hpp $(SQSLSASWSDEP) \
                                sQsmo.hpp $(SQSMODEP) sQd2c.hpp $(SQD2CDEP) \
                                sQsmoVect.hpp $(SQSMOVECTDEP)

SVM_MULTIC_REDBINDEP  = svm_generic.hpp $(SVM_GENERICDEP) svm_generic_deref.hpp svm_binary.hpp $(SVM_BINARYDEP) \
                        svm_single.hpp $(SVM_SINGLEDEP) kcache.hpp $(KCACHEDEP) \
                        idstore.hpp $(IDSTOREDEP)
SVM_MULTIC_ATONCEDEP  = svm_generic.hpp $(SVM_GENERICDEP) svm_binary.hpp $(SVM_BINARYDEP) \
                        svm_single.hpp $(SVM_SINGLEDEP) kcache.hpp $(KCACHEDEP) \
                        idstore.hpp $(IDSTOREDEP)
SVM_VECTOR_REDBINDEP  = svm_generic.hpp $(SVM_GENERICDEP) svm_scalar.hpp $(SVM_SCALARDEP) \
                        svm_scscor.hpp $(SVM_SCSCORDEP) kcache.hpp $(KCACHEDEP)
SVM_VECTOR_ATONCEDEP  = svm_vector_atonce_template.hpp $(SVM_VECTOR_ATONCE_TEMPLATEDEP)
SVM_VECTOR_MREDBINDEP = svm_generic.hpp $(SVM_GENERICDEP) svm_scalar.hpp $(SVM_SCALARDEP)
SVM_VECTOR_MATONCEDEP = svm_vector_atonce_template.hpp $(SVM_VECTOR_ATONCE_TEMPLATEDEP)

SVM_MULTICDEP = svm_generic.hpp $(SVM_GENERICDEP) \
                svm_multic_redbin.hpp $(SVM_MULTIC_REDBINDEP) svm_multic_atonce.hpp $(SVM_MULTIC_ATONCEDEP)
SVM_VECTORDEP = svm_generic.hpp $(SVM_GENERICDEP) svm_scalar.hpp $(SVM_SCALARDEP) \
                svm_vector_redbin.hpp $(SVM_MULTIC_REDBINDEP) svm_vector_atonce.hpp $(SVM_MULTIC_ATONCEDEP) \
                svm_vector_mredbin.hpp $(SVM_MULTIC_MREDBINDEP) svm_vector_matonce.hpp $(SVM_MULTIC_MATONCEDEP)
SVM_ANIONSDEP = svm_vector.hpp $(SVM_VECTORDEP)
SVM_GENTYPDEP = svm_vector.hpp $(SVM_VECTORDEP)
SVM_PLANARDEP = svm_planar.hpp $(SVM_SCALARDEP)
SVM_MVRANKDEP = svm_mvrank.hpp $(SVM_PLANARDEP)
SVM_MULBINDEP = svm_mulbin.hpp $(SVM_MVRANKDEP)
SVM_CYCLICDEP = svm_cyclic.hpp $(SVM_PLANARDEP)
SVM_KCONSTDEP = svm_kconst.hpp $(SVM_GENERICDEP)

KNN_GENERICDEP = ml_base.hpp $(ML_BASEDEP) kcache.hpp $(KCACHEDEP)

KNN_SCALARDEP = knn_generic.hpp $(KNN_GENERICDEP)
KNN_BINARYDEP = knn_generic.hpp $(KNN_GENERICDEP)
KNN_MULTICDEP = knn_generic.hpp $(KNN_GENERICDEP)
KNN_VECTORDEP = knn_generic.hpp $(KNN_GENERICDEP)
KNN_ANIONSDEP = knn_generic.hpp $(KNN_GENERICDEP)
KNN_GENTYPDEP = knn_generic.hpp $(KNN_GENERICDEP)
KNN_DENSITDEP = knn_generic.hpp $(KNN_GENERICDEP)

BLK_GENERICDEP = ml_base.hpp $(ML_BASEDEP)

BLK_NOPNOPDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_MERCERDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_CONSENDEP = blk_generic.hpp $(BLK_GENERICDEP) idstore.hpp $(IDSTOREDEP)
BLK_BERNSTDEP = blk_generic.hpp $(BLK_GENERICDEP) gpr_generic.hpp $(GPR_GENERICDEP)
BLK_BATTERDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_AVESCADEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_AVEVECDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_AVEANIDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_USRFNADEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_USRFNBDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_USERIODEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_CALBAKDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_MEXFNADEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_MEXFNBDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_CONECTDEP = blk_generic.hpp $(BLK_GENERICDEP) gpr_generic.hpp $(GPR_GENERICDEP)
BLK_SYSTEMDEP = blk_generic.hpp $(BLK_GENERICDEP)
BLK_KERNELDEP = blk_generic.hpp $(BLK_GENERICDEP)

IMP_GENERICDEP = ml_base.hpp $(ML_BASEDEP)

IMP_EXPECTDEP = imp_generic.hpp $(IMP_GENERICDEP) hyper_opt.hpp
IMP_PARSVMDEP = imp_generic.hpp $(IMP_GENERICDEP) svm_pfront.hpp $(SVM_PFRONTDEP)
IMP_RLSAMPDEP = imp_generic.hpp $(IMP_GENERICDEP)
IMP_NLSAMPDEP = imp_generic.hpp $(IMP_GENERICDEP) gpr_scalar.hpp $(GPR_SCALARDEP)

LSV_GENERICDEP = ml_base.hpp $(ML_BASEDEP) svm_scalar.hpp $(SVM_SCALARDEP)

LSV_SCALARDEP = lsv_generic.hpp $(LSV_GENERICDEP)
LSV_VECTORDEP = lsv_generic.hpp $(LSV_GENERICDEP) svm_vector.hpp $(SVM_VECTORDEP) lsv_generic_deref.hpp
LSV_ANIONSDEP = lsv_generic.hpp $(LSV_GENERICDEP)
LSV_SCSCORDEP = lsv_generic.hpp $(LSV_GENERICDEP) svm_scscor.hpp $(SVM_SCSCORDEP)
LSV_GENTYPDEP = lsv_generic.hpp $(LSV_GENERICDEP) svm_gentyp.hpp $(SVM_GENTYPDEP)
LSV_PLANARDEP = lsv_generic.hpp $(LSV_GENERICDEP) svm_planar.hpp $(SVM_PLANARDEP)
LSV_MVRANKDEP = lsv_generic.hpp $(LSV_GENERICDEP) svm_mvrank.hpp $(SVM_MVRANKDEP)
LSV_BINARYDEP = lsv_scalar.hpp $(LSV_SCALARDEP)
LSV_SCALAR_RFFDEP = lsv_generic.hpp $(LSV_GENERICDEP) svm_scalar_rff.hpp $(SVM_SCALAR_RFFDEP) lsv_generic_deref.hpp

GPR_GENERICDEP = ml_base.hpp $(ML_BASEDEP) ml_base_deref.hpp lsv_generic.hpp $(LSV_GENERICDEP)

GPR_SCALARDEP = gpr_generic.hpp $(GPR_GENERICDEP) lsv_scalar.hpp $(LSV_SCALARDEP)
GPR_VECTORDEP = gpr_generic.hpp $(GPR_GENERICDEP) lsv_vector.hpp $(LSV_VECTORDEP)
GPR_ANIONSDEP = gpr_generic.hpp $(GPR_GENERICDEP) lsv_anions.hpp $(LSV_ANIONSDEP)
GPR_GENTYPDEP = gpr_generic.hpp $(GPR_GENERICDEP) lsv_gentyp.hpp $(LSV_GENTYPDEP)
GPR_BINARYDEP = gpr_scalar.hpp $(GPR_SCALARDEP)
GPR_SCALAR_RFFDEP = gpr_generic.hpp $(GPR_GENERICDEP) lsv_scalar_rff.hpp $(LSV_SCALAR_RFFDEP)
GPR_BINARY_RFFDEP = gpr_scalar_rff.hpp $(GPR_SCALAR_RFFDEP)

MLM_GENERICDEP = ml_base.hpp $(ML_BASEDEP) ml_base_deref.hpp svm_generic.hpp $(SVM_GENERICDEP) svm_scalar.hpp $(SVM_SCALARDEP)

MLM_SCALARDEP = mlm_generic.hpp $(MLM_GENERICDEP) svm_scalar.hpp $(SVM_SCALARDEP)
MLM_BINARYDEP = mlm_generic.hpp $(MLM_GENERICDEP) svm_binary.hpp $(SVM_BINARYDEP)
MLM_VECTORDEP = mlm_generic.hpp $(MLM_GENERICDEP) svm_vector.hpp $(SVM_VECTORDEP)

# Simplified version so as to not kill make!

#ML_MUTABLEDEP = $(MLMUTATEFILE)
ML_MUTABLEDEP = ml_base.hpp $(ML_BASEDEP) \
                svm_generic.hpp $(SVM_GENERICDEP) blk_generic.hpp $(BLK_GENERICDEP) \
                knn_generic.hpp $(KNN_GENERICDEP) gpr_generic.hpp $(GPR_GENERICDEP) \
                lsv_generic.hpp $(LSV_GENERICDEP) imp_generic.hpp $(IMP_GENERICDEP) \
                mlm_generic.hpp $(MLM_GENERICDEP)

FUZZYMLDEP        = ml_base.hpp $(ML_BASEDEP)
MAKEMONOTDEP      = makemonot.hpp $(ML_BASEDEP)
PLOTBASEDEP       = basefn.hpp memdebug.hpp niceassert.hpp $(VECTORDEP)
PLOTMLDEP         = ml_base.hpp $(ML_BASEDEP) imp_generic.hpp $(IMP_GENERICDEP) blk_conect.hpp $(BLK_CONECTDEP) plotbase.hpp $(PLOTBASEDEP) basefn.hpp memdebug.hpp niceassert.hpp
XFERMLDEP         = svm_generic.hpp $(SVM_GENERICDEP) ml_base.hpp $(ML_BASEDEP)
ERRORTESTDEP      = ml_base.hpp $(ML_BASEDEP)
ADDDATADEP        = ml_base.hpp $(ML_BASEDEP) basefn.hpp memdebug.hpp niceassert.hpp
ANALYSEANOMALYDEP = svm_generic.hpp $(SVM_GENERICDEP)
BALCDEP           = ml_base.hpp $(ML_BASEDEP)
HILLCLIMBDEP      = ml_base.hpp $(ML_BASEDEP)

GLOBALOPTDEP = FNVector.hpp $(FNVECTORDEP) ml_base.hpp $(ML_BASEDEP) \
               ml_mutable.hpp $(ML_MUTABLE) $(ML_MUTABLEDEP) \
               mlcommon.hpp $(MLCOMMONDEP) blk_usrfnb.hpp $(BLK_USRFNBDEP) \
               blk_bernst.hpp $(BLK_BERNSTDEP) blk_conect.hpp $(BLK_CONECTDEP) \
               globalopt_base.hpp
GRIDOPTDEP   = globalopt.hpp $(GLOBALOPTDEP)
DIRECTOPTDEP = globalopt.hpp $(GLOBALOPTDEP) nlopt_direct.hpp $(NLOPT_DIRECTDEP)
NELDEROPTDEP = globalopt.hpp $(GLOBALOPTDEP) nlopt_neldermead.hpp $(NLOPT_NELDERMEADDEP)
SMBOOPTDEP   = ml_base.hpp $(ML_BASEDEP) ml_mutable.hpp $(ML_MUTABLE) \
               globalopt.hpp $(GLOBALOPTDEP) gridopt.hpp $(GRIDOPTDEP) \
               directopt.hpp $(DIRECTOPTDEP) nelderopt.hpp $(NELDEROPTDEP) \
               gpr_scalar.hpp $(GPR_SCALARDEP) gpr_vector.hpp $(GPR_VECTORDEP) \
               errortest.hpp $(ERRORTESTDEP) addData.hpp $(ADDDATADEP) \
               plotml.hpp $(PLOTMLDEP) gpr_scalar_rff.hpp $(GPR_SCALAR_RFFDEP)
BAYESOPTDEP  = smboopt.hpp $(SMBOOPTDEP) directopt.hpp $(DIRECTOPTDEP) \
               imp_generic.hpp $(IMP_GENERICDEP) imp_nlsamp.hpp $(IMP_NLSAMPDEP) \
               plotml.hpp $(PLOTMLDEP)
AWARESTREAMDEP = basefn.hpp memdebug.hpp niceassert.hpp
MLINTERDEP   = ml_base.hpp $(ML_BASEDEP) mlcommon.hpp $(MLCOMMONDEP) \
               ml_mutable.hpp $(ML_MUTABLEDEP) gentype.hpp $(GENTYPEDEP) \
               ofiletype.hpp $(OFILETYPEDEP) vecstack.hpp $(VECSTACKDEP) \
               awarestream.hpp $(AWARESTREAMDEP) matrix.hpp $(MATRIXDEP)




ML_MUTABLE_CCDEP = svm_single.hpp $(SVM_SINGLEDEP) svm_binary.hpp $(SVM_BINARYDEP) \
               svm_scalar.hpp $(SVM_SCALARDEP) svm_multic.hpp $(SVM_MULTICDEP) \
               svm_vector.hpp $(SVM_VECTORDEP) svm_anions.hpp $(SVM_ANIONSDEP) \
               svm_densit.hpp $(SVM_DENSITDEP) \
               svm_pfront.hpp $(SVM_PFRONTDEP) svm_biscor.hpp $(SVM_BISCORDEP) \
               svm_scscor.hpp $(SVM_SCSCORDEP) svm_gentyp.hpp $(SVM_GENTYPDEP) \
               svm_planar.hpp $(SVM_PLANARDEP) svm_mvrank.hpp $(SVM_MVRANKDEP) \
               svm_mulbin.hpp $(SVM_MULBINDEP) svm_cyclic.hpp $(SVM_CYCLICDEP) \
               svm_simlrn.hpp $(SVM_SIMLRNDEP) svm_kconst.hpp $(SVM_KCONSTDEP) \
               svm_scalar_rff.hpp $(SVM_SCALAR_RFFDEP) svm_binary_rff.hpp $(SVM_BINARY_RFFDEP) \
               blk_nopnop.hpp $(BLK_NOPNOPDEP) blk_consen.hpp $(BLK_CONSENDEP) \
               blk_avesca.hpp $(BLK_AVESCADEP) blk_avevec.hpp $(BLK_AVEVECDEP) \
               blk_aveani.hpp $(BLK_AVEANIDEP) blk_usrfna.hpp $(BLK_USRFNADEP) \
               blk_usrfnb.hpp $(BLK_USRFNBDEP) blk_userio.hpp $(BLK_USERIODEP) \
               blk_calbak.hpp $(BLK_CALBAKDEP) blk_mexfna.hpp $(BLK_MEXFNADEP) \
               blk_mexfnb.hpp $(BLK_MEXFNBDEP) blk_mercer.hpp $(BLK_MERCERDEP) \
               blk_conect.hpp $(BLK_CONECTDEP) blk_system.hpp $(BLK_SYSTEMDEP) \
               blk_kernel.hpp $(BLK_KERNELDEP) blk_bernst.hpp $(BLK_BERNSTDEP) \
               blk_batter.hpp $(BLK_BATTERDEP) \
               knn_densit.hpp $(KNN_DENSITDEP) knn_binary.hpp $(KNN_BINARYDEP) \
               knn_multic.hpp $(KNN_MULTICDEP) knn_gentyp.hpp $(KNN_GENTYPDEP) \
               knn_scalar.hpp $(KNN_SCALARDEP) knn_vector.hpp $(KNN_VECTORDEP) \
               knn_anions.hpp $(KNN_ANIONSDEP) \
               gpr_scalar.hpp $(GPR_SCALARDEP) gpr_vector.hpp $(GPR_VECTORDEP) \
               gpr_anions.hpp $(GPR_ANIONSDEP) gpr_gentyp.hpp $(GPR_GENTYPDEP) \
               gpr_binary.hpp $(GPR_BINARYDEP) gpr_scalar_rff.hpp $(GPR_SCALAR_RFFDEP) \
               gpr_binary_rff.hpp $(GPR_BINARY_RFFDEP) \
               lsv_scalar.hpp $(LSV_SCALARDEP) lsv_vector.hpp $(LSV_VECTORDEP) \
               lsv_anions.hpp $(LSV_ANIONSDEP) lsv_scscor.hpp $(LSV_SCSCORDEP) \
               lsv_gentyp.hpp $(LSV_GENTYPDEP) \
               lsv_planar.hpp $(LSV_PLANARDEP) lsv_mvrank.hpp $(LSV_MVRANKDEP) \
               lsv_binary.hpp $(LSV_BINARYDEP) lsv_scalar_rff.hpp $(LSV_SCALAR_RFFDEP) \
               imp_expect.hpp $(IMP_EXPECTDEP) imp_parsvm.hpp $(IMP_PARSVMDEP) \
               imp_rlsamp.hpp $(IMP_RLSAMPDEP) imp_nlsamp.hpp $(IMP_NLSAMPDEP) \
               mlm_scalar.hpp $(MLM_SCALARDEP) mlm_binary.hpp $(MLM_BINARYDEP) \
               mlm_vector.hpp $(MLM_VECTORDEP)








# Object file creation dependencies.
#
# Basic template for what follows:
#
# x.o: x.cc blah1.hpp blah2.hpp ...
#      $(CC) $< -o $@ -c $(CFLAGS)
#
# These are headers included from the .cc file only.  If DEP macros are
# defined for a given header then this must also be included as it will
# expand to the complete dependency tree.
#
# Note that the ML tree pipes to $(MLMUTATEFILE) as the dependency tree
# is too complicated for make.


basefn.o    : basefn.cc basefn.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
randfun.o   : randfun.cc randfun.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
clockbase.o : clockbase.cc clockbase.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
memdebug.o  : memdebug.cc memdebug.hpp basefn.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
gslrefs.o   : gslrefs.cc gslrefs.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
zerocross.o : zerocross.cc zerocross.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
adam.o : adam.cc adam.hpp basefn.hpp memdebug.hpp niceassert.hpp $(ADAMDEP) sQbase.hpp $(SQBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

numbase.o    : numbase.cc numbase.hpp $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
anion.o      : anion.cc anion.hpp numbase.hpp $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
dynarray.o   : dynarray.cc dynarray.hpp $(DYNARRAYDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
strfns.o     : strfns.cc strfns.hpp numbase.hpp $(NUMBASEDEP) vecstack.hpp $(VECSTACKDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
opttest.o : opttest.cc opttest.hpp $(OPTTESTDEP) numbase.hpp $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
paretotest.o : paretotest.cc paretotest.hpp $(PARETOTESTDEP) numbase.hpp $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
idstore.o    : idstore.cc idstore.hpp $(IDSTOREDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
smatrix.o    : smatrix.cc smatrix.hpp $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
gentype.o    : gentype.cc gentype.hpp $(GENTYPEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
mercer.o     : mercer.cc mercer.hpp $(MERCERDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
mlcommon.o   : mlcommon.cc mlcommon.hpp $(MLCOMMONDEP) vecstack.hpp $(VECSTACKDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
kcache.o     : kcache.cc kcache.hpp $(KCACHEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
FNVector.o   : FNVector.cc FNVector.hpp $(FNVECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

hyper_psc.o   : hyper_psc.cc hyper_psc.hpp basefn.hpp memdebug.hpp niceassert.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_debug.o : hyper_debug.cc hyper_debug.hpp basefn.hpp memdebug.hpp niceassert.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_base.o  : hyper_base.cc hyper_base.hpp hyper_psc.hpp hyper_debug.hpp basefn.hpp memdebug.hpp niceassert.hpp numbase.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_opt.o   : hyper_opt.cc hyper_opt.hpp hyper_base.hpp hyper_psc.hpp basefn.hpp memdebug.hpp niceassert.hpp numbase.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_alt.o   : hyper_alt.cc hyper_alt.hpp hyper_opt.hpp hyper_base.hpp hyper_psc.hpp basefn.hpp memdebug.hpp niceassert.hpp numbase.hpp
	$(CC) $< -o $@ -c $(CFLAGS)

optcontext.o        : optcontext.cc optcontext.hpp $(OPTCONTEXTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
optlinbasecontext.o : optlinbasecontext.cc optlinbasecontext.hpp $(OPTLINBASECONTEXTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
optlincontext.o     : optlincontext.cc optlincontext.hpp $(OPTLINCONTEXTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
optlinstate.o       : optlinstate.cc optlinstate.hpp $(OPTLINSTATEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

sQbase.o : sQbase.cc sQbase.hpp $(SQBASEDEP) kcache.hpp $(KCACHEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQsLsAsWs.o : sQsLsAsWs.cc sQsLsAsWs.hpp $(SQSLSQSWSDEP) kcache.hpp $(KCACHEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQsmo.o     : sQsmo.cc sQsmo.hpp $(SQSMODEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQd2c.o     : sQd2c.cc sQd2c.hpp $(SQD2CDEP) sQsmo.hpp $(SQSMODEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQgraddesc.o : sQgraddesc.cc sQgraddesc.hpp $(SQGRADDESCDEP) sQsLsAsWs.hpp $(SQSLSQSWSDEP) sQd2c.hpp $(SQD2CDEP) smatrix.hpp $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
linsolve.o  : linsolve.cc linsolve.hpp $(LINSOLVEDEP) optlinstate.hpp $(OPTLINSTATEDEP) sQbase.hpp $(SQBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

nlopt_base.o  : nlopt_base.cc nlopt_base.hpp $(NLOPT_BASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
nlopt_direct.o  : nlopt_direct.cc nlopt_direct.hpp $(NLOPT_DIRECTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
nlopt_neldermead.o  : nlopt_neldermead.cc nlopt_neldermead.hpp $(NLOPT_NELDERMEADDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
nlopt_slsqp.o  : nlopt_slsqp.cc nlopt_slsqp.hpp $(NLOPT_SLSQP)
	$(CC) $< -o $@ -c $(CFLAGS)

ml_base.o : ml_base.cc ml_base.hpp $(ML_BASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_generic.o : svm_generic.cc svm_generic.hpp $(SVM_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_scalar.o : svm_scalar.cc svm_scalar.hpp $(SVM_SCALARDEP) sQsLsAsWs.hpp $(SQSLSASWSDEP) sQsmo.hpp $(SQSMODEP) sQd2c.hpp $(SQD2CDEP) sQgraddesc.hpp $(SQGRADDESCDEP) linsolve.hpp $(LINSOLVEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_binary.o : svm_binary.cc svm_binary.hpp $(SVM_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_simlrn.o : svm_simlrn.cc svm_simlrn.hpp $(SVM_SIMLRNDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_single.o : svm_single.cc svm_single.hpp $(SVM_SINGLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_biscor.o : svm_biscor.cc svm_biscor.hpp $(SVM_BISCORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_scalar_rff.o : svm_scalar_rff.cc svm_scalar_rff.hpp $(SVM_SCALAR_RFFDEP) basefn.hpp memdebug.hpp niceassert.hpp adam.hpp $(ADAMDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_binary_rff.o : svm_binary_rff.cc svm_binary_rff.hpp $(SVM_BINARY_RFFDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_scscor.o : svm_scscor.cc svm_scscor.hpp $(SVM_SCSCORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_densit.o : svm_densit.cc svm_densit.hpp $(SVM_DENSITDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_pfront.o : svm_pfront.cc svm_pfront.hpp $(SVM_PFRONTDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_vector_atonce_template.o : svm_vector_atonce_template.cc svm_vector_atonce_template.hpp $(SVM_VECTOR_ATONCE_TEMPLATEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_multic_redbin.o  : svm_multic_redbin.cc svm_multic_redbin.hpp $(SVM_MULTIC_REDBINDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_multic_atonce.o  : svm_multic_atonce.cc svm_multic_atonce.hpp $(SVM_MULTIC_ATONCEDEP) smatrix.hpp $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_redbin.o  : svm_vector_redbin.cc svm_vector_redbin.hpp $(SVM_VECTOR_REDBINDEP) smatrix.hpp $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_atonce.o  : svm_vector_atonce.cc svm_vector_atonce.hpp $(SVM_VECTOR_ATONCEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_mredbin.o : svm_vector_mredbin.cc svm_vector_mredbin.hpp $(SVM_VECTOR_MREDBINDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_matonce.o : svm_vector_matonce.cc svm_vector_matonce.hpp $(SVM_VECTOR_MATONCEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_multic.o : svm_multic.cc svm_multic.hpp $(SVM_MULTICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector.o : svm_vector.cc svm_vector.hpp $(SVM_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_anions.o : svm_anions.cc svm_anions.hpp $(SVM_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_gentyp.o : svm_gentyp.cc svm_gentyp.hpp $(SVM_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_planar.o : svm_planar.cc svm_planar.hpp $(SVM_PLANARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_mvrank.o : svm_mvrank.cc svm_mvrank.hpp $(SVM_MVRANKDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_mulbin.o : svm_mulbin.cc svm_mulbin.hpp $(SVM_MULBINDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_cyclic.o : svm_cyclic.cc svm_cyclic.hpp $(SVM_CYCLICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_kconst.o : svm_kconst.cc svm_kconst.hpp $(SVM_KCONSTDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

knn_generic.o : knn_generic.cc knn_generic.hpp $(KNN_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

knn_scalar.o : knn_scalar.cc knn_scalar.hpp $(KNN_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_binary.o : knn_binary.cc knn_binary.hpp $(KNN_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_multic.o : knn_multic.cc knn_multic.hpp $(KNN_MULTICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_vector.o : knn_vector.cc knn_vector.hpp $(KNN_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_anions.o : knn_anions.cc knn_anions.hpp $(KNN_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_gentyp.o : knn_gentyp.cc knn_gentyp.hpp $(KNN_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_densit.o : knn_densit.cc knn_densit.hpp $(KNN_DENSITDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

blk_generic.o : blk_generic.cc blk_generic.hpp $(BLK_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

blk_nopnop.o : blk_nopnop.cc blk_nopnop.hpp $(BLK_NOPNOPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_consen.o : blk_consen.cc blk_consen.hpp $(BLK_CONSENDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_avesca.o : blk_avesca.cc blk_avesca.hpp $(BLK_AVESCADEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_avevec.o : blk_avevec.cc blk_avevec.hpp $(BLK_AVEVECDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_aveani.o : blk_aveani.cc blk_aveani.hpp $(BLK_AVEANIDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_usrfna.o : blk_usrfna.cc blk_usrfna.hpp $(BLK_USRFNADEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_usrfnb.o : blk_usrfnb.cc blk_usrfnb.hpp $(BLK_USRFNBDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_userio.o : blk_userio.cc blk_userio.hpp $(BLK_USERIODEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_calbak.o : blk_calbak.cc blk_calbak.hpp $(BLK_CALBAKDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_mexfna.o : blk_mexfna.cc blk_mexfna.hpp $(BLK_MEXFNADEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_mexfnb.o : blk_mexfnb.cc blk_mexfnb.hpp $(BLK_MEXFNBDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_mercer.o : blk_mercer.cc blk_mercer.hpp $(BLK_MERCERDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_conect.o : blk_conect.cc blk_conect.hpp $(BLK_CONECTDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_system.o : blk_system.cc blk_system.hpp $(BLK_SYSTEMDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_kernel.o : blk_kernel.cc blk_kernel.hpp $(BLK_KERNELDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_bernst.o : blk_bernst.cc blk_bernst.hpp $(BLK_BERNSTDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_batter.o : blk_batter.cc blk_batter.hpp $(BLK_BATTERDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

imp_generic.o : imp_generic.cc imp_generic.hpp $(IMP_GENERICDEP) hyper_base.hpp 
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

imp_expect.o : imp_expect.cc imp_expect.hpp $(IMP_EXPECTDEP) hyper_alt.hpp hyper_base.hpp
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
imp_parsvm.o : imp_parsvm.cc imp_parsvm.hpp $(IMP_PARSVMDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
imp_rlsamp.o : imp_rlsamp.cc imp_rlsamp.hpp $(IMP_RLSAMPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
imp_nlsamp.o : imp_nlsamp.cc imp_nlsamp.hpp $(IMP_NLSAMPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

lsv_generic.o : lsv_generic.cc lsv_generic.hpp $(LSV_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

lsv_scalar.o : lsv_scalar.cc lsv_scalar.hpp $(LSV_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_vector.o : lsv_vector.cc lsv_vector.hpp $(LSV_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_anions.o : lsv_anions.cc lsv_anions.hpp $(LSV_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_scscor.o : lsv_scscor.cc lsv_scscor.hpp $(LSV_SCSCORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_gentyp.o : lsv_gentyp.cc lsv_gentyp.hpp $(LSV_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_planar.o : lsv_planar.cc lsv_planar.hpp $(LSV_PLANARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_mvrank.o : lsv_mvrank.cc lsv_mvrank.hpp $(LSV_MVRANKDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_binary.o : lsv_binary.cc lsv_binary.hpp $(LSV_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_scalar_rff.o : lsv_scalar_rff.cc lsv_scalar_rff.hpp $(LSV_SCALAR_RFFDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

gpr_generic.o : gpr_generic.cc gpr_generic.hpp $(GPR_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

gpr_scalar.o : gpr_scalar.cc gpr_scalar.hpp $(GPR_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_vector.o : gpr_vector.cc gpr_vector.hpp $(GPR_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_anions.o : gpr_anions.cc gpr_anions.hpp $(GPR_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_gentyp.o : gpr_gentyp.cc gpr_gentyp.hpp $(GPR_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_binary.o : gpr_binary.cc gpr_binary.hpp $(GPR_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_scalar_rff.o : gpr_scalar_rff.cc gpr_scalar_rff.hpp $(GPR_SCALAR_RFFDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_binary_rff.o : gpr_binary_rff.cc gpr_binary_rff.hpp $(GPR_BINARY_RFFDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

mlm_generic.o : mlm_generic.cc mlm_generic.hpp $(MLM_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

mlm_scalar.o : mlm_scalar.cc mlm_scalar.hpp $(MLM_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
mlm_binary.o : mlm_binary.cc mlm_binary.hpp $(MLM_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
mlm_vector.o : mlm_vector.cc mlm_vector.hpp $(MLM_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

ml_mutable.o : ml_mutable.cc ml_mutable.hpp $(ML_MUTABLEDEP) $(ML_MUTABLE_CCDEP)
	$(CC) $< -o $@ -c $(CFLAGS) $(BIGFLAG) > $(MLMUTATEFILE)

fuzzyml.o        : fuzzyml.cc fuzzyml.hpp $(FUZZYMLDEP) svm_single.hpp $(SVM_SINGLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
makemonot.o      : makemonot.cc makemonot.hpp $(MAKEMONOTDEP) ml_base.hpp $(ML_BASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
plotbase.o       : plotbase.cc plotbase.hpp $(PLOTBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
plotml.o         : plotml.cc plotml.hpp $(PLOTMLDEP) plotbase.hpp $(PLOTBASEDEP) imp_generic.hpp $(IMP_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
xferml.o         : xferml.cc xferml.hpp $(XFERMLDEP) svm_scalar.hpp $(SVM_SCALARDEP) svm_kconst.hpp $(SVM_KCONSTDEP) numbase.hpp $(NUMBASEDEP) mlcommon.hpp $(MLCOMMONDEP) nlopt_neldermead.hpp $(NLOPT_NELDERMEADDEP) nlopt_slsqp.hpp $(NLOPT_SLSQPDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
errortest.o      : errortest.cc errortest.hpp $(ERRORTESTDEP) ml_mutable.hpp $(ML_MUTABLEDEP) svm_scalar_rff.hpp $(SVM_SCALAR_RFFDEP) svm_binary.hpp $(SVM_BINARYDEP) svm_single.hpp $(SVM_SINGLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
addData.o        : addData.cc addData.hpp $(ADDDATADEP) ml_mutable.hpp $(ML_MUTABLEDEP) svm_mvrank.hpp $(SVM_MVRANKDEP) lsv_mvrank.hpp $(LSV_MVRANKDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
analyseAnomaly.o : analyseAnomaly.cc analyseAnomaly.hpp $(ANALYSEANOMALYDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
balc.o           : balc.cc balc.hpp $(BALCDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
hillclimb.o      : hillclimb.cc hillclimb.hpp $(HILLCLIMBDEP) errortest.hpp $(ERRORTESTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

globalopt.o : globalopt.cc globalopt.hpp $(GLOBALOPTDEP) hyper_base.hpp
	$(CC) $< -o $@ -c $(CFLAGS)
gridopt.o   : gridopt.cc gridopt.hpp $(GRIDOPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
directopt.o : directopt.cc directopt.hpp $(DIRECTOPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
nelderopt.o : nelderopt.cc nelderopt.hpp $(NELDEROPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
smboopt.o : smboopt.cc smboopt.hpp $(SMBOOPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
bayesopt.o  : bayesopt.cc bayesopt.hpp $(BAYESOPTDEP) ml_mutable.hpp $(ML_MUTABLEDEP) imp_expect.hpp $(IMP_EXPECTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

awarestream.o : awarestream.cc awarestream.hpp $(AWARESTREAMDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

mlinter.o   : mlinter.cc mlinter.hpp $(MLINTERDEP) hillclimb.hpp $(HILLCLIMBDEP) fuzzyml.hpp $(FUZZYMLDEP) makemonot.hpp $(MAKEMONOTDEP) xferml.hpp $(XFERMLDEP) errortest.hpp $(ERRORTESTDEP) addData.hpp $(ADDDATADEP) analyseAnomaly.hpp $(ANALYSEANOMALYDEP) balc.hpp $(BALCDEP) gridopt.hpp $(GRIDOPTDEP) directopt.hpp $(DIRECTOPTDEP) nelderopt.hpp $(NELDEROPTDEP) smboopt.hpp $(SMBOOPTDEP) bayesopt.hpp $(BAYESOPTDEP) globalopt.hpp $(GLOBALOPTDEP) opttest.hpp $(OPTTESTDEP) paretotest.hpp $(PARETOTESTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)



