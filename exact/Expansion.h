#pragma once

// Released into the public-domain by Robert Bridson, 2009.
// Modified by Tyson Brochu, 2011.
// Simple functions for manipulating multiprecision floating-point
// expansions, with simplicity favoured over speed.

#include <vector>
#include <cstddef>
namespace other {

using std::vector;

// The basic type is essentially a vector of *increasing* and 
// *nonoverlapping* doubles, apart from allowed zeroes anywhere.

class Expansion;

void negative(const Expansion& input, Expansion& output);

int sign( const Expansion& a );

bool
is_zero( const Expansion& a );

void
add(double a, double b, Expansion& sum);

// a and sum may be aliased to the same Expansion for in-place addition
void
add(const Expansion& a, double b, Expansion& sum);

inline void
add(double a, const Expansion& b, Expansion& sum)
{ add(b, a, sum); }

// aliasing a, b and sum is safe
void
add(const Expansion& a, const Expansion& b, Expansion& sum);

void
subtract( const double& a, const double& b, Expansion& difference);

// aliasing a, b and difference is safe
void
subtract(const Expansion& a, const Expansion& b, Expansion& difference);

// aliasing input and output is safe
void
negative(const Expansion& input, Expansion& output);

void
multiply(double a, double b, Expansion& product);

void
multiply(double a, double b, double c, Expansion& product);

void
multiply(double a, double b, double c, double d, Expansion& product);

void
multiply(const Expansion& a, double b, Expansion& product);

inline void
multiply(double a, const Expansion& b, Expansion& product)
{ multiply(b, a, product); }

// Aliasing NOT safe
void
multiply(const Expansion& a, const Expansion& b, Expansion& product);

void compress( const Expansion& e, Expansion& h );

// Aliasing NOT safe
bool divide( const Expansion& x, const Expansion& y, Expansion& q );

void
remove_zeros(Expansion& a);

double
estimate(const Expansion& a);

bool equals( const Expansion& a, const Expansion& b );

void
print_full( const Expansion& e );


// ----------------------------------------------------

class Expansion
{
      
public:
   
   vector<double> v;

   Expansion()
   : v(0)
   {}

   explicit Expansion( double val )
    : v(1, val)
   {}
   
   Expansion( size_t n, double val )
   : v(n,val)
   {}
   
   virtual ~Expansion() {}
   
   Expansion& operator+=(const Expansion &rhs)
   {
      add( *this, rhs, *this );
      return *this;
   }
   
   Expansion& operator-=(const Expansion &rhs)
   {
      subtract( *this, rhs, *this );
      return *this;
   }
   
   Expansion& operator*=(const Expansion &rhs)
   {
      Expansion p;
      multiply( *this, rhs, p );
      *this = p;
      return *this;
   }
   
   inline Expansion operator+(const Expansion &other) const 
   {
      Expansion result = *this;     
      result += other;  
      return result;              
   }
   
   inline Expansion operator-(const Expansion &other) const 
   {
      Expansion result = *this;    
      result -= other;  
      return result;              
   }
   
   
   inline Expansion operator*(const Expansion &other) const
   {
      Expansion result = *this;    
      result *= other;  
      return result;              
   }
   
   inline Expansion operator-( ) const
   {
      Expansion result;
      negative( *this, result );
      return result;
   }
   
   inline double estimate() const
   {
      return other::estimate( *this );
   }
   
   inline bool indefinite_sign() const
   {
      return false;
   }
   
   static void begin_special_arithmetic()
   {}
   
   static void end_special_arithmetic()
   {}
   
   inline void clear()
   {
      v.clear();
   }

   inline void resize( size_t new_size )
   {
      v.resize(new_size);
   }

};


inline void make_expansion( double a, Expansion& e )
{ 
   if(a) 
   {
      e = Expansion(1, a); 
   }
   else
   {
      e.clear();
   }
}

inline void
make_zero(Expansion& e)
{ e.resize(0); }

inline void create_from_double( double a, Expansion& out )
{
   make_expansion( a, out );
}

inline bool certainly_opposite_sign( const Expansion& a, const Expansion& b )
{
   return sign(a) * sign(b)<0;
}

}
