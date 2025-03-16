
#include "awarestream.hpp"

int main()
{
    std::string sockis("./temp.sock");
    std::string buffer;

    try
    {
        awarestream *svmsocket = makeUnixSocket(sockis,1);
        awarestream &svmsock = *svmsocket;

        std::istream svmsockin(&svmsock);
        std::ostream svmsockout(&svmsock);

        std::cout << "socket name: " << sockis << "\n";
        std::string clientstarter("unixclienttest.py ");
        clientstarter += sockis;
        std::cout << "Start client: " << clientstarter << "\n";
        svm_pycall(clientstarter,1);

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
//            std::cerr << "Send string = " << svmsock.vogon(buffer) << "\n";

            std::cerr << "Send string...";
            svmsockout << buffer;
            std::cerr << "done\n";

	    if ( hasfb )
	    {
//                std::cout << "Receive string = " << svmsock.skim(buffer) << "\n";
//                std::cout << "Feedback: " << buffer << "\n";

		std::cout << "Receive string...\n";
                svmsockin >> buffer;
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


