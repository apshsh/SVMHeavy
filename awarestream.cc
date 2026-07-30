
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

svmvolatile fifolist awarestream::strfifo;

//using namespace std::literals;

/*
awarestream *makeUnixSocket(std::string &sockname, int dellisten, int fixname, int sercli)
{
    if ( !fixname )
    {
        const static std::string svmshort("./svm");
        const static std::string sockshort("./svm");

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
    if ( sock ) { delete sock; }
}

awarestream *makeTCPIPSocket(const std::string &server_url, int port, int dellisten, int _sercli)
{
    awarestream *res;

    if ( _sercli ) { res = new awarestream(port,SVM_SOCK_STREAM,1,dellisten);            }
    else           { res = new awarestream(server_url,port,SVM_SOCK_STREAM,2,dellisten); }

    return res;
}

void delTCPIPSocket(awarestream *sock)
{
    if ( sock ) { delete sock; }
}
*/










/// OLD HEADER STUFF

/*
// This function opens a Unix socket server.  The filename (socketname) is
// generated and put in sockname, and a pointer to the socket returned.  If
// fixname then sockname is instead given by the caller and *must* be used!
//
// dellisten = 0 connect now
//             1 connect when used
// fixname   = 0 generate socket name that is not in use
//           = 1 use sockname given
// _sercli   = 0 for client
//             1 for server
//
// TCPIP version sockname == "" means use "127.0.0.1"

//awarestream *makeUnixSocket(std::string &sockname, int dellisten = 0, int fixname = 0, int _sercli = 1);
//void delUnixSocket(awarestream *sock);

//awarestream *makeTCPIPSocket(const std::string &server_url, int port, int dellisten = 0, int _sercli = 1);
//void delTCPIPSocket(awarestream *sock);

    int feedback;  // means no passing back upstream for UDP/TCP, 1 means allow feedback upstream
    int dellisten; // 0 normal, 1 means that server has not yet had a connection established to it so need to listen etc.
    std::stringstream buffer;
    int port;
    int sock;
    int serverorclient; // 1 = server, 0 = client
    std::string server_url;
    int socktype;
    int streamtype;
    std::string sun_path;

    // Constructor for standard istream type
    //
    // _istr       = stream that data will be coming from
    // _ideletable = 0 if stream is to be left open always (eg std::cin)
    //               1 if stream should be closed once done (eg filestream)
    //
    // (_std default was std::cin)

    awarestream(std::istream *_istr = nullptr, int _ideletable = 0) : srcsel(_str ? 1 : 0),
                                                                      istr(_istr),
                                                                      ostr(nullptr),
                                                                      ideletable(_str ? _ideletable : 0),
                                                                      odeletable(0),
                                                                      port(0),
                                                                      sock(0),
                                                                      feedback(0),
                                                                      serverorclient(-1). // 1 = server, 0 = client
                                                                      server_url(""),
                                                                      socktype(-1),
                                                                      streamtype(SVM_AF_INET),
                                                                      sun_path(""),
                                                                      fifoind(-1),
                                                                      dellisten(0) { ; }
    }

    awarestream(const char *, std::ostream *_ostr = nullptr, int _odeletable = 0): srcsel(1),
                                                                                   istr(nullptr),
                                                                                   ostr(_ostr ? _ostr : &outstream()),
                                                                                   ideletable(0),
                                                                                   odeletable(_odeletable),
                                                                                   port(0),
                                                                                   sock(0),
                                                                                   feedback(1),
                                                                                   serverorclient(-1), // 1 = server, 0 = client
                                                                                   server_url(""),
                                                                                   socktype(-1),
                                                                                   streamtype(SVM_AF_INET),
                                                                                   sun_path(""),
                                                                                   fifoind(-1),
                                                                                   dellisten(0) { ; }

    awarestream(std::istream *_istr, std::ostream *_ostr, int _ideletable, int _odeletable) : srcsel(1),
                                                                                              istr(_istr),
                                                                                              ostr(_ostr),
                                                                                              ideletable(_ideletable),
                                                                                              odeletable(_odeletable),
                                                                                              port(0),
                                                                                              sock(0),
                                                                                              feedback(1),
                                                                                              serverorclient(-1), // 1 = server, 0 = client
                                                                                              server_url(""),
                                                                                              socktype(-1),
                                                                                              streamtype(SVM_AF_INET),
                                                                                              sun_path(""),
                                                                                              fifoind(-1),
                                                                                              dellisten(0) { ; }

    // Constructor for shared string fifo (cross-thread comms)
    //
    // _fifoind = which fifo is used

    awarestream(const char *, const char *, int _fifoind) : srcsel(4),
                                                            istr(nullptr),
                                                            ostr(nullptr),
                                                            ideletable(0),
                                                            odeletable(0),
                                                            port(0),
                                                            sock(0),
                                                            feedback(0),
                                                            serverorclient(-1), // 1 = server, 0 = client
                                                            server_url(""),
                                                            socktype(-1),
                                                            streamtype(SVM_AF_INET),
                                                            sun_path(""),
                                                            fifoind(_fifoind),
                                                            dellisten(0) { NiceAssert( _fifoind >= 0 ); }

    // Copy constructor.  This will always throw as there is no defined concept of copy here
    // Constructor for UDP/TCP control.
    //
    // _port     = port to listen to
    // _socktype = SVM_SOCK_STREAM if connecting over TCP (default)
    //           = SVM_SOCK_DGRAM if connecting over UDP
    // _dellisten = 0 wait for client to connect here
    //              1 delay wait until other server call occurs (servers only)
    //
    // If socket cannot be bound then this will revert to no stream and throw an error
    //
    // The first version sets up as a server and waits for connection from a client,
    // the second is a client that attempts to connect to a server at address _server_url.
    //
    // If server_url == "" then it is set "127.0.0.1" (loopback)

    awarestream(int _port, int _socktype = SVM_SOCK_STREAM, int _feedback = 0, int _dellisten = 0)
    {
        srcsel         = 2;
        str            = nullptr;
        ostr           = nullptr;
        ideletable     = 1;
        odeletable     = 1;
        port           = _port;
        sock           = 0;
        feedback       = _feedback;
        serverorclient = 1; // 1 = server, 0 = client
        server_url     = "";
        socktype       = _socktype;
        streamtype     = SVM_AF_INET;
        sun_path       = "";
        fifoind        = -1;
        dellisten      = serverorclient ?_dellisten : 0;

        if ( connectTCPUDP() )
        {
            NiceThrow("Server setup fail.");
        }
    }

    awarestream(const std::string &_server_url, int _port, int _socktype = SVM_SOCK_STREAM, int _feedback = 0, int _dellisten = 0)
    {
        std::string loc_server_url(_server_url);

        if ( loc_server_url == "" )
        {
            loc_server_url = "127.0.0.1";
        }

        srcsel         = 2;
        str            = nullptr;
        ostr           = nullptr;
        ideletable     = 1;
        odeletable     = 1;
        port           = _port;
        sock           = 0;
        feedback       = _feedback;
        serverorclient = 0; // 1 = server, 0 = client
        server_url     = loc_server_url;
        socktype       = _socktype;
        streamtype     = SVM_AF_INET;
        sun_path       = "";
        fifoind        = -1;
        dellisten      = serverorclient ?_dellisten : 0;

        if ( connectTCPUDP() )
        {
            NiceThrow("Client setup fail.");
        }
    }

    // Constructor for unix socket control.
    //
    // _addrpath  = path/filename of unix socket control
    // _socktype  = SVM_SOCK_STREAM if connecting over TCP
    //              SVM_SOCK_DGRAM if connecting over UDP (default)
    // _feedback  = 0 uni-directional
    //              1 bi-directional
    // _sercli    = 0 for client
    //              1 for server
    // _dellisten = 0 wait for client to connect here
    //              1 delay wait until other server call occurs (servers only)
    //
    // If socket cannot be bound then this will revert to no stream and throw an error

    awarestream(const char *, std::string _addrpath, int _socktype = SVM_SOCK_DGRAM, int _feedback = 0, int _sercli = 1, int _dellisten = 0)
    {
        srcsel         = 3;
        str            = nullptr;
        ostr           = nullptr;
        ideletable     = 1;
        odeletable     = 1;
        port           = 0;
        sock           = 0;
        feedback       = _feedback;
        serverorclient = _sercli; // 1 = server, 0 = client
        server_url     = "";
        socktype       = _socktype;
        streamtype     = SVM_AF_UNIX;
        sun_path       = _addrpath;
        fifoind        = -1;
        dellisten      = serverorclient ?_dellisten : 0;

        if ( connectTCPUDP() )
        {
            NiceThrow("Server setup fail.");
        }
    }

    ~awarestream()
    {
        if ( ideletable && ( srcsel == 1 )      ) { MEMDEL(istr); istr = nullptr; }
        if ( odeletable && ( srcsel == 1 )      ) { MEMDEL(ostr); ostr = nullptr; }
        //if ( ( srcsel == 2 ) || ( srcsel == 3 ) ) { disconnectTCPUDP(); }
    }

    int good(void)
    {
        int res = 0;

        if      ( srcsel == 0 ) { res = 0;                                      }
        else if ( srcsel == 1 ) { res = istr->good();                           }
        else if ( srcsel == 2 ) { res = 1;                                      }
        else if ( srcsel == 3 ) { res = 1;                                      }
        else if ( srcsel == 4 ) { res = ( strfifo.size(fifoind) >= 0 ) ? 1 : 0; }

        return res;
    }

    // vogon pipes a string to the output stream. srcsel == 0 acts like /dev/null
    //
    // Success return: 0
    // Fail return: 2 for udp fail
    //              3 for tcp fail
    //              1 for successful push onto killed fifo
    //             -1 for failed push onto fifo
    //             -2 for failed push onto killed fifo

    int vogon(const std::string &src)
    {
        int res = 0;

        // srcsel == 0: acts like /dev/null

        if ( dellisten )
        {
            if ( srcsel != 3 )
            {
                svm_listen(sock,1024);
                clilen = sizeof(*cli_addr);
            }

            if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
            {
                srcsel = 0;
                return 1;
            }

            dellisten = 0;
        }

        if ( ( srcsel == 1 ) && ( ostr != nullptr ) )
        {
            *ostr << src;
            res = 0;
        }

        else if ( ( ( srcsel == 2 ) || ( srcsel == 3 ) ) && feedback )
        {
            if ( svm_send(sock,src.c_str(),strlen(src.c_str()),0) < 0 )
            {
                // Close socket, revert type and return an error

                disconnectTCPUDP();
                res = srcsel;
            }

            else
            {
                res = 0;
            }
        }

        else if ( srcsel == 4 )
        {
            res = strfifo.push(fifoind,src);
        }

        return res;
    }

    // Skim pipes a string off the input stream.
    //
    // Success return: 0
    // Fail return: 2 for udp fail
    //              3 for tcp fail
    //              4 for attempted read from output only stream
    //              1 for successful pop from killed fifo
    //             -1 for failed pop from fifo
    //             -2 for failed pop from killed fifo

    int skim(std::string &dest)
    {
        int res = 0;

        if ( dellisten )
        {
            if ( srcsel != 3 )
            {
                svm_listen(sock,1024);
                clilen = sizeof(*cli_addr);
            }

            if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
            {
                srcsel = 0;
                return 1;
            }

            dellisten = 0;
        }

        if ( srcsel == 0 )
        {
            dest = "";
            res = 4;
        }

        else if ( srcsel == 1 )
        {
            *str >> dest;
            res = 0;
        }

        else if ( ( srcsel == 2 ) || ( srcsel == 3 ) )
        {
            // If buffer is empty then listen until we receive something.  We then shove that into the buffer and continue

            dest = "";

            while ( dest.length() == 0 )
            {
                buffer >> dest;

                if ( !(buffer.good()) && ( dest.length() == 0 ) )
                {
                    buffer.clear();

                    int bytes_read = 0;
                    char recv_data[SVM_UDPBUFFERLEN];
                    struct svm_sockaddr_in client_addr;
                    svm_socklen_t addr_len = sizeof(struct svm_sockaddr);

                    // Wait until we get something (

                    while ( bytes_read == 0 )
                    {
                        if ( ( bytes_read = svm_recvfrom(sock,recv_data,SVM_UDPBUFFERLEN,0,(struct svm_sockaddr *) &client_addr,&addr_len) ) < 0 )
                        {
                            // Close socket, revert type and return an error

                            dest = "";
                            disconnectTCPUDP();
                            res = srcsel;
                            goto exitpoint;
                        }
                    }

                    // nullptr terminate the response and shove it into the buffer

                    recv_data[bytes_read] = '\0';

                    buffer << recv_data;
                }
            }

            res = 0;
        }

        else if ( srcsel == 4 )
        {
            int isdone = 0;

            while ( !isdone )
            {
                int fifosizeval = strfifo.size(fifoind);

                if ( fifosizeval > 0 )
                {
                    // There are strings in the fifo, so pop one of them.

                    res = strfifo.pop(fifoind,dest);

                    isdone = 1;
                }

                else if ( fifosizeval == 0 )
                {
                    // There are no strings in the fifo, but fifo live
                    // Unlock fifo, wait a random interval to give other
                    // threads a chance to push onto fifo or kill it, then
                    // retry.

                    //svm_usleep(FIFO_BASESLEEP+(svm_rand()%FIFO_ADDSLEEP_RAND));
                    svm_usleep(FIFO_BASESLEEP+(rand()%FIFO_ADDSLEEP_RAND));
                }

                else
                {
                    // fifo has kill-flag set, so treat it like a dead
                    // socket.

                    dest = "";
                    res = fifosizeval;

                    isdone = 1;
                }
            }
        }

exitpoint:
        return res;
    }

    int good(void)
    {
        if ( dellisten )
        {
            if ( srcsel != 3 )
            {
                svm_listen(sock,1024);
                clilen = sizeof(*cli_addr);
            }

            if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
            {
                srcsel = 0;
                return 1;
            }

            dellisten = 0;
        }

        int res = 0;

        if      ( srcsel == 0 ) { res = 0;                                      }
        else if ( srcsel == 1 ) { res = str->good();                            }
        else if ( srcsel == 2 ) { res = 1;                                      }
        else if ( srcsel == 3 ) { res = 1;                                      }
        else if ( srcsel == 4 ) { res = ( strfifo.size(fifoind) >= 0 ) ? 1 : 0; }

        return res;
    }

    int connectTCPUDP(void)
    {
        return baseconnectTCPUDP();
    }

    // global because dellisten

    svm_socklen_t clilen;
    struct svm_sockaddr *cli_addr;

    int baseconnectTCPUDP(void)
    {
	struct svm_sockaddr_in server_addr_ip;
	struct svm_sockaddr_un server_addr_un;
	struct svm_sockaddr *server_addr;
        int server_addr_size;

	// Create a socket - domain SVM_AF_INET (IP) or SVM_AF_UNIX (unix sockets)
	//                 - type of service SVM_SOCK_DGRAM (UDP) or SVM_SOCK_STREAM (TCP)
	//                 - protocol 0 (not used)

	if ( ( sock = svm_socket(streamtype,socktype,0) ) < 0 )
	{
	    srcsel = 0;
	    return 1;
	}

	// Identify (name) the socket and bind it

	memset((char *) &server_addr,0,sizeof(server_addr));

	server_addr_ip.sin_family = streamtype;                                     // Address family used
	server_addr_ip.sin_port   = svm_htons(port);                                // Port to listen to
	server_addr_un.sun_family = streamtype;                                     // Address family used
        strncpy_safe_b(server_addr_un.sun_path,sun_path.c_str(),SVM_UNIX_PATH_MAX-1); // path for unix stream

	server_addr      = ( streamtype == SVM_AF_INET ) ? ( (struct svm_sockaddr *) &server_addr_ip ) : ( (struct svm_sockaddr *) &server_addr_un );
	server_addr_size = ( streamtype == SVM_AF_INET ) ? sizeof(struct svm_sockaddr_in) : sizeof(struct svm_sockaddr_un);

        if ( serverorclient )
        {
            // server

#ifndef VISUAL_STU
#ifdef ALLOW_SOCKETS
            server_addr_ip.sin_addr.s_addr = svm_htonl(SVM_INADDR_ANY); // Allow connection from all
#endif
#endif

            if ( svm_bind(sock,server_addr,server_addr_size) )
            {
                srcsel = 0;
                return 1;
            }

            if ( socktype == SVM_SOCK_STREAM )
            {
                //svm_socklen_t clilen;
                struct svm_sockaddr_in cli_addr_ip;
                struct svm_sockaddr_un cli_addr_un;

		//struct svm_sockaddr *cli_addr = ( streamtype == SVM_AF_INET ) ? ( (struct svm_sockaddr *) &cli_addr_ip ) : ( (struct svm_sockaddr *) &cli_addr_un );
		cli_addr = ( streamtype == SVM_AF_INET ) ? ( (struct svm_sockaddr *) &cli_addr_ip ) : ( (struct svm_sockaddr *) &cli_addr_un );


                if ( srcsel == 3 )
                {
                    svm_listen(sock,1024);
                    clilen = sizeof(*cli_addr);
                }

                if ( !dellisten )
                {
                    if ( srcsel != 3 )
                    {
                        svm_listen(sock,1024);
                        clilen = sizeof(*cli_addr);
                    }

                    if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
                    {
                        srcsel = 0;
                        return 1;
                    }
                }
            }
        }

        else
        {
            // client

#ifndef VISUAL_STU
#ifdef ALLOW_SOCKETS
            server_addr_ip.sin_addr.s_addr = svm_inet_addr(server_url.c_str()); // Specify connection to server
#endif
#endif

            if ( socktype == SVM_SOCK_STREAM )
            {
                if ( svm_connect(sock,server_addr,server_addr_size) < 0 )
                {
                    srcsel = 0;
                    return 1;
                }
            }
        }

        return 0;
    }

    void disconnectTCPUDP(void)
    {
        try { svm_shutdown(sock,SVM_SHUT_RDWR); } catch (...) { ; }
        try { svm_close(sock);                  } catch (...) { ; }

        srcsel = 0;
    }
*/


















// OLD BASEFN.HPP STUFF

// ALLOW_SOCKETS:  sockets are used in awarestream for TCP and UDP streams.

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Sockets stuff

// --- If sockets available include relevant libraries ---

/*
#ifdef ALLOW_SOCKETS
#include <errno.h>
#ifdef CYGWIN_BUILD
#include <sys/un.h>
#endif
#ifndef VISUAL_STU
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#ifndef CYGWIN_BUILD
#include <linux/un.h>
#endif
#endif
#endif
*/

/*
#ifdef ALLOW_SOCKETS

#ifdef VISUAL_STU
// windows doesn't define this, but inferring from final argument of recvfrom
#define socklen_t int
//class sockaddr_un;
//class sockaddr_un
//{
//    public:
//
//    int sun_family;
//    char *sun_path;
//};
#define UNIX_PATH_MAX 256
#define SHUT_RDWR     2
struct sockaddr_un
{
    int sun_family;
    char *sun_path;

    sockaddr_un()
    {
        sun_family = 0;
        sun_path = new char[UNIX_PATH_MAX+1];
        sun_path[0] = '\0';
    }

    ~sockaddr_un()
    {
        delete[] sun_path;
    }
};
//#define SHUT_RDWR     SD_BOTH
inline int close(int a);
inline int close(int a) { return closesocket(a); }
#pragma comment(lib, "Ws2_32.lib")
#endif

#define UDPBUFFERLEN 1024

// Alias everything

#define SVM_SOCK_STREAM   SOCK_STREAM
#define SVM_SOCK_DGRAM    SOCK_DGRAM
#define SVM_MAX_RETRIES   MAX_RETRIES
#define SVM_AF_INET       AF_INET
#define SVM_AF_UNIX       AF_UNIX
#define SVM_UNIX_PATH_MAX UNIX_PATH_MAX
#define SVM_INADDR_ANY    INADDR_ANY
#define SVM_SHUT_WR       SHUT_WR
#define SVM_SHUT_RDWR     SHUT_RDWR
#define SVM_UDPBUFFERLEN  UDPBUFFERLEN
#define svm_socklen_t     socklen_t
#define svm_sockaddr_in   sockaddr_in
#define svm_sockaddr_un   sockaddr_un
#define svm_sockaddr      sockaddr

inline int svm_send(int a, const char *b, size_t c, int d);
inline int svm_send(int a, const char *b, size_t c, int d) { return (int) send(a,b,c,d); }

inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f);
inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f) { return (int) recvfrom(a,b,c,d,e,f); }

inline int svm_htons(int a);
inline int svm_htons(int a) { return htons((uint16_t) a); }

inline int svm_htonl(int a);
inline int svm_htonl(int a) { return htonl(a); }

inline int svm_inet_addr(const char *a);
inline int svm_inet_addr(const char *a) { return inet_addr(a); }

inline int svm_shutdown(int a, int b);
inline int svm_shutdown(int a, int b) { return shutdown(a,b); }

inline int svm_close(int a);
inline int svm_close(int a) { return close(a); }

inline int svm_socket(int a, int b, int c);
inline int svm_socket(int a, int b, int c) { return socket(a,b,c); }

inline int svm_bind(int a, svm_sockaddr *b, int c);
inline int svm_bind(int a, svm_sockaddr *b, int c) { return bind(a,b,c); }

inline int svm_accept(int a, svm_sockaddr *b, svm_socklen_t *c);
inline int svm_accept(int a, svm_sockaddr *b, svm_socklen_t *c) { return accept(a,b,c); }

inline int svm_connect(int a, svm_sockaddr *b, int c);
inline int svm_connect(int a, svm_sockaddr *b, int c) { return connect(a,b,c); }

inline int svm_listen(int a, int b);
inline int svm_listen(int a, int b) { return listen(a,b); }

#endif

// --- If sockets not possible define stubs and fake classes to allow ---
// --- compilation and return error codes if sockets used.            ---

#ifndef ALLOW_SOCKETS

#define SVM_SOCK_STREAM   0
#define SVM_SOCK_DGRAM    0
#define SVM_MAX_RETRIES   5
#define SVM_AF_INET       0
#define SVM_AF_UNIX       0
#define SVM_UNIX_PATH_MAX 0
#define SVM_INADDR_ANY    0
#define SVM_SHUT_WR       0
#define SVM_SHUT_RDWR     0
#define SVM_UDPBUFFERLEN  1024
#define svm_socklen_t     int

struct svm_saddr;
struct svm_saddr
{
    public:

    int ws_addr;
    int wS_un; // something windows uses apparently
};

struct svm_sockaddr_in;
struct svm_sockaddr_in
{
    public:

    int sin_family;
    int sin_port;
    struct svm_saddr sin_addr;
};

struct svm_sockaddr_un;
struct svm_sockaddr_un
{
    public:

    int sun_family;
    char *sun_path;
};

struct svm_sockaddr;
struct svm_sockaddr
{
    public:

    int sin_family;
    int sin_port;
    struct svm_saddr sin_addr;
};

inline int svm_send(int a, const char *b, size_t c, int d);
inline int svm_send(int,   const char *,  size_t,   int) { return -1; }

inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f);
inline int svm_recvfrom(int,   char *,  int,   int,   svm_sockaddr *,  svm_socklen_t *)  { return -1; }

inline int svm_htons(int a);
inline int svm_htons(int) { return -1; }

inline int svm_htonl(int a);
inline int svm_htonl(int) { return -1; }

inline int svm_inet_addr(const char *a);
inline int svm_inet_addr(const char *) { return -1; }

inline int svm_shutdown(int a, int b);
inline int svm_shutdown(int,   int) { return -1; }

inline int svm_close(int a);
inline int svm_close(int) { return -1; }

inline int svm_socket(int a, int b, int c);
inline int svm_socket(int,   int,   int) { return -1; }

inline int svm_bind(int a, svm_sockaddr *b, int c);
inline int svm_bind(int,   svm_sockaddr *,  int) { return -1; }

inline int svm_accept(int a, svm_sockaddr *b, svm_socklen_t *c);
inline int svm_accept(int,   svm_sockaddr *,  svm_socklen_t *) { return -1; }

inline int svm_connect(int a, svm_sockaddr *b, int c);
inline int svm_connect(int,   svm_sockaddr *,  int) { return -1; }

inline int svm_listen(int a, int b);
inline int svm_listen(int,   int) { return -1; }

#endif

*/






































