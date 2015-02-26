#include "diis.h"
#include <math.h>
#include <sstream>

using namespace ambit;

namespace lightspeed {

DIIS::DIIS(
    size_t min_vectors,
    size_t max_vectors,
    bool use_disk) :
    min_vectors_(min_vectors),
    max_vectors_(max_vectors),
    use_disk_(use_disk)
{
    E_ = Tensor::build(kCore,"E",{max_vectors_,max_vectors_});
}
void DIIS::clear()
{
    state_vecs_.clear(); 
    error_vecs_.clear(); 
}
void DIIS::add_iteration(
    const std::vector<Tensor>& state_vec, 
    const std::vector<Tensor>& error_vec)
{
    // => Data Copy <= //
    
    std::vector<Tensor> state;
    for (size_t ind = 0; ind < state_vec.size(); ind++) {
        state.push_back(state_vec[ind].clone(use_disk() ? kDisk : state_vec[ind].type()));
    }
    std::vector<Tensor> error;
    for (size_t ind = 0; ind < error_vec.size(); ind++) {
        error.push_back(error_vec[ind].clone(use_disk() ? kDisk : error_vec[ind].type()));
    }

    // => Index Replacement <= //

    size_t index;
    if (cur_vectors() < max_vectors()) {
        index = cur_vectors();
        state_vecs_.push_back(state);
        error_vecs_.push_back(error);
    } else {
        double maxE = 0.0;
        double* Ep = E_.data().data();
        for (size_t ind = 0; ind < max_vectors_; ind++) {
            if (Ep[ind * max_vectors_ + ind] >= maxE) {
                maxE = Ep[ind * max_vectors_ + ind];
                index = ind;
            }
        }
        state_vecs_[index] = state;
        error_vecs_[index] = error;
    }

    // => Error Inner Product <= //

    std::vector<double> Econt(cur_vectors(),0.0);
    for (size_t t = 0; t < error.size(); t++) {
        Tensor T = (error[t].type() == kDisk ? error[t].clone(kCore) : error[t]);
        // Needed for arbitrary-rank dot (avert your eyes)
        std::stringstream ss;
        std::vector<std::string> inds = {"i","j","k","l","m","n","o","p","q","r","s","t","u","v"};
        for (int r = 0; r < T.rank(); r++) {
            ss << inds[r];
        }
        std::string cinds = ss.str();
        for (size_t ind = 0; ind < cur_vectors(); ind++) {
            Tensor U = (error_vecs_[ind][t].type() == kDisk ? error_vecs_[ind][t].clone(kCore) : error_vecs_[ind][t]);
            Econt[ind] += T(cinds) * U(cinds);
        }
    }
    double* Ep = E_.data().data();
    for (size_t ind = 0; ind < cur_vectors(); ind++) {
        Ep[ind * max_vectors_ + index] = 
        Ep[index * max_vectors_ + ind] = 
        Econt[ind];
    }
}
bool DIIS::extrapolate( 
    std::vector<Tensor>& state_vec)
{
    if (cur_vectors() < min_vectors()) return false;

    // => Raw B Matrix <= //
    
    size_t cur = cur_vectors();
    Tensor B = Tensor::build(kCore,"B",{cur+1,cur+1});
    double* Bp = B.data().data();
    double* Ep = E_.data().data();
    for (size_t i = 0; i < cur; i++) {
        for (size_t j = 0; j < cur; j++) {
            Bp[i * (cur + 1) + j] = Ep[i * max_vectors() + j];
        }
        Bp[i * (cur + 1) + cur] = 
        Bp[cur * (cur + 1) + i] = 
        1.0;
    }

    Tensor f = Tensor::build(kCore,"f",{cur+1});
    double* fp = f.data().data();
    fp[cur] = 1.0;

    Tensor c = Tensor::build(kCore,"c",{cur+1});
    double* cp = c.data().data();
    
    // => Balancing Registers <= //

    Tensor d = Tensor::build(kCore,"d",{cur+1});
    double* dp = d.data().data();
    
    Tensor s = Tensor::build(kCore,"s",{cur+1});
    double* sp = s.data().data();

    // => Zero/Negative Trapping <= // 

    bool is_zero = false;
    for (size_t i = 0; i < cur; i++) {
        if(Bp[i * (cur + 1) + i] <= 0.0) {
            is_zero = true;
        }
    }
    
    // => Balancing <= //
    
    for (size_t i = 0; i < cur; i++) {
        sp[i] = (is_zero ? 1.0 : pow(Bp[i * (cur + 1) + i],-1.0/2.0));
    }
    sp[cur] = 1.0;

    for (size_t i = 0; i < cur + 1; i++) {
        for (size_t j = 0; j < cur + 1; j++) {
            Bp[i * (cur + 1) + j] *= sp[i]*sp[j];
        }
    }

    // => Inversion <= //

    Tensor Binv = B.power(-1.0,1E-12);
    c("i") = B("ij") * f("j");
    d("i") = s("i") * c("i");

    // => Extrapolation <= //

    for (size_t t = 0; t < state_vec.size(); t++) {
        Tensor T = (state_vec[t].type() == kDisk ? state_vec[t].clone(kCore) : state_vec[t]);
        T.zero();
        // Needed for arbitrary-rank dot (avert your eyes)
        std::stringstream ss;
        std::vector<std::string> inds = {"i","j","k","l","m","n","o","p","q","r","s","t","u","v"};
        for (int r = 0; r < T.rank(); r++) {
            ss << inds[r];
        }
        std::string cinds = ss.str();
        for (size_t i = 0; i < cur; i++) {
            Tensor U = (state_vecs_[i][t].type() == kDisk ? state_vecs_[i][t].clone(kCore) : state_vecs_[i][t]);
            T(cinds) += dp[i] * U(cinds);    
        }
        if (state_vec[t].type() == kDisk) {
            state_vec[t]() = T();
        }
    }
    
    return true;
}

} // namespace lightspeed
