
#include "awarestream.hpp"

int main(int argc, const char **argv)
{
    std::string sockis("./temp.sock");
    std::string buffer;

    if ( argc == 2 )
    {
        sockis = argv[1];
    }

    awarestream svmsock("&",sockis,SOCK_STREAM,1,0);
//    awarestream svmsock("&",sockis,SVM_SOCK_DGRAM,1,0);

    int hasfb = 0;

    while ( 1 )
    {
	std::cout << "Argument: "; std::getline(std::cin,buffer);

	if ( buffer == "die" )
	{
	    break;
	}

	else if ( buffer == "fbon" )
	{
	    hasfb = 1;
	}

	else if ( buffer == "fboff" )
	{
	    hasfb = 0;
	}

	else
	{
	    std::cerr << "Send string = " << svmsock.vogon(buffer) << "\n";

	    if ( hasfb )
	    {
		std::cout << "Receive string = " << svmsock.skim(buffer) << "\n";
		std::cout << "Feedback: " << buffer << "\n";
	    }
	}
    }

    return 0;
}


