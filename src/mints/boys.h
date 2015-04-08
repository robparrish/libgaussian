#ifndef BOYS_H
#define BOYS_H

#include <vector>

namespace lightspeed {

class Boys {

public:
    Boys(int nmax) { buffer_.resize(nmax+1); }
    virtual void compute(double t, int n);
    const std::vector<double>& buffer() const { return buffer_; }
    
protected:
    std::vector<double> buffer_;

};

} // namespace lightspeed

#endif // header guard
