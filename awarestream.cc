
//
// Very slightly smarter stream class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// What is it: a vector with push and pop operations added on top
// and some other stuff

#include "awarestream.hpp"
#include <stdio.h>
//#ifdef ENABLE_THREADS
//#include <mutex>
//#endif

//#ifdef ENABLE_THREADS
//std::mutex awarestream::fifolock;
//#endif
svmvolatile fifolist awarestream::strfifo;

//using namespace std::literals;


awarestream *makeUnixSocket(std::string &sockname, int dellisten, int fixname, int sercli)
{
    if ( !fixname )
    {
        static const std::string svmshort("./svm");
        static const std::string sockshort("./svm");

//        getUniqueFile(sockname,"./svm"s,".sock"s);
        getUniqueFile(sockname,svmshort,sockshort);
    }

    if ( sercli && fileExists(sockname) )
    {
        // Delete file if it exists

        remove(sockname.c_str());
    }

    return new awarestream("&",sockname,SVM_SOCK_STREAM,1,sercli,dellisten);
}

void delUnixSocket(awarestream *sock)
{
    if ( sock )
    {
        delete sock;
    }
}

awarestream *makeTCPIPSocket(const std::string &server_url, int port, int dellisten, int _sercli)
{
    awarestream *res;

    if ( _sercli )
    {
        res = new awarestream(port,SVM_SOCK_STREAM,1,dellisten);
    }

    else
    {
        res = new awarestream(server_url,port,SVM_SOCK_STREAM,2,dellisten);
    }

    return res;
}

void delTCPIPSocket(awarestream *sock)
{
    if ( sock )
    {
        delete sock;
    }
}
