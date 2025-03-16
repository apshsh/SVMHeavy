#include <iostream>
#include <thread>

int main()
{
    unsigned int numThreads = std::thread::hardware_concurrency();

    std::cout << "Number of threads available: " << numThreads << "\n";

    return 0;
}
