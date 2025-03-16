
//
// Runs svmheavy as a forked process with background optimisation
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svmfront_h
#define _svmfront_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <string>
#include "numbase.hpp"

#define RESBUFFSIZE 2048
#define DIRBUFFSIZE 2048

class svm_front;
class svm_front
{
    public:

    svm_front();
    ~svm_front();
    svm_front &operator=(const svm_front &src);

    std::string &init(std::string &res, const std::string _type,
                                        const std::string _vtype,
                                        const std::string _mtype,
                                        const std::string _ctype,
                                        const std::string _btype,
                                        const std::string _rtype,
                                        const std::string _ttype);
    std::string &pass(std::string &res, const std::string arg);
    std::string &kill(std::string &res);

    private:

    // status:
    //
    // 0 means uninitialised
    // 1 means running
    //
    // pipefd: streams used to communicate with child SVM
    //
    // type:  SVM type (s,c,m,r,v,a)
    // vtype: vectorial type (once,red)
    // mtype: multiclass type (1vsA,1vsa,DAG,MOC,maxwin,recdiv)
    // ctype: classification type (svc,svr)
    // btype: bias type (f,v,p,n)
    // rtype: risk type (l,q,g,G)
    // ttype: tube type (f,s)

    int status;
    int pipeto[2];
    int pipefrom[2];

    std::string type;
    std::string vtype;
    std::string mtype;
    std::string ctype;
    std::string btype;
    std::string rtype;
    std::string ttype;

    char resbuf[RESBUFFSIZE];

    void clensebuf(void);
};

#endif
