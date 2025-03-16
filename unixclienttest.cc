
#include "awarestream.hpp"

int main(int argc, const char **argv)
{
    std::string sockis("./temp.sock");
    std::string buffer("ping");

    if ( argc == 2 )
    {
        sockis = argv[1];
    }

    awarestream svmsock("&",sockis,SOCK_STREAM,1,0);
//    awarestream svmsock("&",sockis,SVM_SOCK_DGRAM,1,0);

    int hasfb = 0;

    svm_usleep(10000000);

    std::cerr << "Client is awake... ";

    svm_usleep(10000000);

    std::cerr << "sending ping... ";
    std::cerr << svmsock.vogon(buffer) << "... ";

    svm_usleep(10000000);

    std::cerr << "closing.\n";

    return 0;
}


