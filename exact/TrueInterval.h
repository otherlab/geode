// ---------------------------------------------------------
//
//  interval.h
//  Tyson Brochu 2011
//
// ---------------------------------------------------------

#pragma once

#include <other/core/utility/rounding.h>
#include <other/core/vector/Vector.h>
namespace other {

#ifndef DEBUG
#define VERIFY() (void)0;
#else
#define VERIFY() assert( -v[0] <= v[1] ); \
assert( v[0] == v[0] ); \
assert( v[1] == v[1] );
#endif

#ifndef DEBUG
#define CHECK_ROUNDING_MODE() (void)0;
#else
#define CHECK_ROUNDING_MODE() assert( fegetround( ) == FE_UPWARD ); 
#endif


// ----------------------------------------
//
// class TrueInterval:
//
// Stores the interval [a,b] as [-a,b] internally.  With proper arithmetic operations, this 
// allows us to use only FE_UPWARD and avoid switching rounding modes over and over.
//
// ----------------------------------------

class TrueInterval
{
   
    static int s_previous_rounding_mode;

public:

   // Internal representation
   double v[2];
         
   TrueInterval( double val );   
   TrueInterval( double left, double right );
   TrueInterval();
   
   double stored_left() const;
   double stored_right() const;

   bool contains_zero() const;
   bool indefinite_sign() const;
   bool is_certainly_negative() const;
   bool is_certainly_positive() const;
   bool is_certainly_zero() const;
   bool certainly_opposite_sign( const TrueInterval& other ) const;

   double estimate() const;
   Vector<double,2> get_actual_interval() const;
   
   TrueInterval& operator+=(const TrueInterval &rhs);
   TrueInterval& operator-=(const TrueInterval &rhs);
   TrueInterval& operator*=(const TrueInterval &rhs);
   
   TrueInterval operator+(const TrueInterval &other) const;
   TrueInterval operator-(const TrueInterval &other) const;
   TrueInterval operator*(const TrueInterval &other) const;
   
   TrueInterval operator-() const;
   
   static void begin_special_arithmetic();
   static void end_special_arithmetic();
   
};

inline void create_from_double( double a, TrueInterval& out );


// ----------------------------------------

inline TrueInterval::TrueInterval( double val )
{
   v[0] = -val;
   v[1] = val;
   VERIFY();
}

// ----------------------------------------

inline TrueInterval::TrueInterval( double left, double right )
{
   assert( left <= right );
   v[0] = -left;
   v[1] = right;
   VERIFY();
}

// ----------------------------------------

inline TrueInterval::TrueInterval()
{
   v[0] = 0;
   v[1] = 0;
   VERIFY();
}

// ----------------------------------------

inline double TrueInterval::stored_left() const
{
   return v[0];
}

// ----------------------------------------

inline double TrueInterval::stored_right() const
{
   return v[1];
}

// ----------------------------------------

inline bool TrueInterval::contains_zero() const
{
   // true if a <= 0 && b >= 0
   // remember v[0] == -a

   return v[0] >= 0 && v[1] >= 0;
}

// ----------------------------------------

inline bool TrueInterval::indefinite_sign() const
{
   // Same as contains_zero; exists only for templatization purposes

   return v[0] >= 0 && v[1] >= 0;
}

// ----------------------------------------

inline bool TrueInterval::is_certainly_negative( ) const
{
   return v[1] < 0;
}

// ----------------------------------------

inline bool TrueInterval::is_certainly_positive( ) const
{
   return v[0] < 0;
}

// ----------------------------------------

inline bool TrueInterval::is_certainly_zero( ) const
{
   return v[0] == 0 && v[1] == 0;
}

// ----------------------------------------

inline bool TrueInterval::certainly_opposite_sign( const TrueInterval& other ) const
{
   return ( is_certainly_negative() && other.is_certainly_positive() )
       || ( is_certainly_positive() && other.is_certainly_negative() );
}

// ----------------------------------------

inline bool certainly_opposite_sign( const TrueInterval& a, const TrueInterval& b )
{
   return a.certainly_opposite_sign( b );
}

// ----------------------------------------

inline double TrueInterval::estimate() const
{
   double est = 0.5 * (v[1] - v[0]);

   if ( est == 0 && !is_certainly_zero() )
   {
      // don't return zero as an estimate if the interval is not identically zero.
      est = 1e-20;
   }

   return est;
}

// ----------------------------------------

inline Vector<double,2> TrueInterval::get_actual_interval() const
{
   return Vector<double,2>( -v[0], v[1] );
}

// ----------------------------------------

inline TrueInterval& TrueInterval::operator+=(const TrueInterval &rhs)
{
   CHECK_ROUNDING_MODE();
   VERIFY();
   v[0] += rhs.v[0];
   v[1] += rhs.v[1];
   VERIFY();
   
   return *this;
}

// ----------------------------------------

inline TrueInterval& TrueInterval::operator-=( const TrueInterval& rhs )
{
   CHECK_ROUNDING_MODE();   
   v[0] += rhs.v[1];
   v[1] += rhs.v[0];
   VERIFY();
   return *this;
}

// ----------------------------------------

inline TrueInterval& TrueInterval::operator*=( const TrueInterval& rhs )
{
   CHECK_ROUNDING_MODE();   
   TrueInterval p = (*this) * rhs;
   *this = p;
   return *this;
}

// ----------------------------------------

inline TrueInterval TrueInterval::operator+(const TrueInterval &other) const 
{
   CHECK_ROUNDING_MODE();   
   double v0 = v[0] + other.v[0];
   double v1 = v[1] + other.v[1  ];
   return TrueInterval(-v0, v1);
}

// ----------------------------------------

inline TrueInterval TrueInterval::operator-(const TrueInterval &other) const 
{
   CHECK_ROUNDING_MODE();   
   double v0 = v[0] + other.v[1];
   double v1 = v[1] + other.v[0];
   return TrueInterval(-v0, v1);              
}

// ----------------------------------------

inline TrueInterval TrueInterval::operator*(const TrueInterval &other) const
{
   CHECK_ROUNDING_MODE();
   
   double neg_a = v[0];
   double b = v[1];
   double neg_c = other.v[0];
   double d = other.v[1];
   
   TrueInterval product;
   
   if ( b <= 0 )
   {
      if ( d <= 0 )
      {
         product.v[0] = -b * d;
         product.v[1] = neg_a * neg_c;
      }
      else if ( -neg_c <= 0 && 0 <= d )
      {
         product.v[0] = neg_a * d;
         product.v[1] = neg_a * neg_c;
      }
      else
      {
         product.v[0] = neg_a * d;
         product.v[1] = b * -neg_c;
      }
   }
   else if ( -neg_a <= 0 && 0 <= b )
   {
      if ( d <= 0 )
      {
         product.v[0] = b * neg_c;
         product.v[1] = neg_a * neg_c;
      }
      else if ( -neg_c <= 0 && 0 <= d )
      {
         product.v[0] = max( neg_a * d, b * neg_c );
         product.v[1] = max( neg_a * neg_c, b * d );
      }
      else
      {
         product.v[0] = neg_a * d;
         product.v[1] = b * d;
      }
      
   }
   else
   {
      if ( d <= 0 )
      {
         product.v[0] = b * neg_c; 
         product.v[1] = -neg_a * d;
      }
      else if ( -neg_c <= 0 && 0 <= d )
      {
         product.v[0] = b * neg_c;
         product.v[1] = b * d;
      }
      else
      {
         product.v[0] = -neg_a * neg_c;
         product.v[1] = b * d;
      }
   }
   
   return product;
   
}


// ----------------------------------------

inline TrueInterval TrueInterval::operator-( ) const
{
   CHECK_ROUNDING_MODE();
   return TrueInterval( -v[1], v[0] );   
}

// ----------------------------------------

inline void TrueInterval::begin_special_arithmetic()
{
   s_previous_rounding_mode = fegetround();
   fesetround( FE_UPWARD );
}

// ----------------------------------------

inline void TrueInterval::end_special_arithmetic()
{
   fesetround( s_previous_rounding_mode );
}


// ----------------------------------------

inline void create_from_double( double a, TrueInterval& out )
{
   out.v[0] = -a;
   out.v[1] = a;   
}

}
