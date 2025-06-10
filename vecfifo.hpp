
//
// FIFO buffer class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// What is it: a vector with push and pop operations added on top

#ifndef _vecfifo_h
#define _vecfifo_h

#include "vector.hpp"

template <class T>
class FiFo;

// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const FiFo<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        FiFo<T> &dest);

// Swap function

template <class T> void qswap(FiFo<T> &a, FiFo<T> &b);

// The class itself

template <class T>
class FiFo : public Vector<T>
{
public:

    // Constructors and Destructors

    explicit FiFo() : Vector<T>()
    {
        return;
    }

    FiFo(const FiFo<T> &src) : Vector<T>(static_cast<const Vector<T> &>(src))
    {
        return;
    }

    explicit FiFo(const Vector<T> &src) : Vector<T>(src)
    {
        return;
    }

    // Assignment

    FiFo<T> &operator=(const FiFo<T> &src)
    {
        static_cast<Vector<T> &>(*this) = static_cast<const Vector<T> &>(src);

        return *this;
    }

    FiFo<T> &operator=(const Vector<T> &src)
    {
        static_cast<Vector<T> &>(*this) = src;

        return *this;
    }

    // FiFo operations
    //
    // accessBottom: returns reference to element on bottom of FiFo
    // push: push element onto FiFo (top)
    // pop: pop element from FiFo (bottom)
    // safepop: like pop, but returns 0 if pop successful, 1 if FiFo empty
    // isempty: return 1 if FiFo empty, 0 otherwise

    void push(const T &src)
    {
        (*this).add((*this).size());
	accessTop() = src;

        return;
    }

    int pop(T &dest)
    {
	if ( !isempty() )
	{
	   dest = accessBottom();
           (*this).remove(0); // note that we pop from the bottom for a fifo buffer

	   return 0;
	}

        return 1;
    }

    int pop(void)
    {
	if ( !isempty() )
	{
           (*this).remove(0); // note that we pop from the bottom for a fifo buffer

	   return 0;
	}

        return 1;
    }

    T &accessTop(void)
    {
        return (*this)("&",(*this).size()-1);
    }

    T &accessBottom(void)
    {
        return (*this)("&",0); // note that we pop from the bottom for a fifo buffer
    }

    int isempty(void)
    {
        return ( (*this).size() == 0 );
    }
};

template <class T>
void qswap(FiFo<T> &a, FiFo<T> &b)
{
    qswap(static_cast<Vector<T> &>(a),static_cast<Vector<T> &>(b));

    return;
}

template <class T>
std::ostream &operator<<(std::ostream &output, const FiFo<T> &src)
{
    output << static_cast<const Vector<T> &>(src);

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, FiFo<T> &dest)
{
    input >> static_cast<Vector<T> &>(dest);

    return input;
}

#endif
