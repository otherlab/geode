// Multiprecision integer arithmetic for exact geometric predicates

#include <other/core/exact/Exact.h>
#include <vector>
namespace other {

using std::vector;

ostream& operator<<(ostream& output, mpz_t x) {
  const int size = gmp_snprintf(0,0,"%Zd",x);
  vector<char> buffer(size+1);
  gmp_snprintf(&buffer[0],size+1,"%Zd",x);
  return output<<&buffer[0];
}

ostream& operator<<(ostream& output, mpq_t x) {
  const int size = gmp_snprintf(0,0,"%Qd",x);
  vector<char> buffer(size+1);
  gmp_snprintf(&buffer[0],size+1,"%Qd",x);
  return output<<&buffer[0];
}

}
