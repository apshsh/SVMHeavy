#include <iostream>

int main()
{
    std::cout << "[";

    for ( int i = 0 ; i <= 1000 ; ++i )
    {
        for ( int j = 0 ; j <= 1000 ; ++j )
        {
            double x = ((double) i)/1000;
            double y = ((double) j)/1000;

            std::cout << "[" << x << "," << y << "]";

            if ( ( i != 1000 ) || ( j != 1000 ) )
            {
                std::cout << ",";
            }
        }
    }

    std::cout << "]\n";

    return 0;
}
