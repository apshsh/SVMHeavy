
#include "awarestream.hpp"

int main()
{
    std::string sockis("./temp.sock");
    std::string buffer;
    try
    {
//        awarestream svmsock("&",sockis,SOCK_STREAM,1,1);
//        awarestream svmsock("&",sockis,SVM_SOCK_DGRAM,1,1);
        awarestream *svmsocket = makeUnixSocket(sockis,1);
        awarestream &svmsock = *svmsocket;
        std::cout << "socket name: " << sockis << "\n";

    int hasfb = 1;

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
    }

    catch ( const char *badstr )
    {
        std::cerr << "Failed with: " << badstr << "\n";
    }

    return 0;
}


