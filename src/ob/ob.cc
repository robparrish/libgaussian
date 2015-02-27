#include <math.h>
#include <core/molecule.h>
#include <core/basisset.h>
#include <mints/schwarz.h>
#include <mints/int2c.h>
#include "ob.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace ambit;

namespace lightspeed {

OneBody::OneBody(const std::shared_ptr<SchwarzSieve>& sieve) :
    sieve_(sieve),
    basis1_(sieve->basis1()),
    basis2_(sieve->basis2())
{
}
void OneBody::compute_S(
    Tensor& S,
    double scale) const
{
    bool symm = is_symmetric();
    size_t nbf1 = basis1_->nfunction();
    size_t nbf2 = basis2_->nfunction();

    if (S.type() != kCore) throw std::runtime_error("S must be kCore");
    if (S.rank() != 2) throw std::runtime_error("S must be rank-2");
    if (S.dim(0) != nbf1 || S.dim(1) != nbf2) throw std::runtime_error("S must be nbf1 x nbf2");

    #if defined(_OPENMP)
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    std::vector<std::shared_ptr<OverlapInt2C>> Sints;
    for (int t = 0; t < nthread; t++) {
        Sints.push_back(std::shared_ptr<OverlapInt2C>(new OverlapInt2C(basis1_,basis2_,0)));
    }

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();

    double* Sp = S.data().data();

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind = 0; ind < shell_pairs.size(); ind++) {
        int P = shell_pairs[ind].first;
        int Q = shell_pairs[ind].second;
        int nP = basis1_->shell(P).nfunction();
        int nQ = basis2_->shell(Q).nfunction();
        int oP = basis1_->shell(P).function_index();
        int oQ = basis2_->shell(Q).function_index();
        #if defined(_OPENMP)
        int t = omp_get_thread_num();
        #else
        int t = 0;
        #endif
        Sints[t]->compute_shell(P,Q);
        double* Sbuffer = Sints[t]->buffer();
        if (symm) {
            double Sperm = (P == Q ? 0.5 : 1.0) * scale;
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Sp[(p + oP)*nbf2 + (q + oQ)] +=
                    Sperm * Sbuffer[p*nQ + q];
                    Sp[(q + oQ)*nbf2 + (p + oP)] +=
                    Sperm * Sbuffer[p*nQ + q];
                }
            }
        } else {
            double Sperm = scale;
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Sp[(p + oP)*nbf2 + (q + oQ)] +=
                    Sperm * Sbuffer[p*nQ + q];
                }
            }
        }
    }
}
void OneBody::compute_T(
    Tensor& T,
    double scale) const
{
    bool symm = is_symmetric();
    size_t nbf1 = basis1_->nfunction();
    size_t nbf2 = basis2_->nfunction();

    if (T.type() != kCore) throw std::runtime_error("T must be kCore");
    if (T.rank() != 2) throw std::runtime_error("T must be rank-2");
    if (T.dim(0) != nbf1 || T.dim(1) != nbf2) throw std::runtime_error("T must be nbf1 x nbf2");

    #if defined(_OPENMP)
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    std::vector<std::shared_ptr<KineticInt2C>> Tints;
    for (int t = 0; t < nthread; t++) {
        Tints.push_back(std::shared_ptr<KineticInt2C>(new KineticInt2C(basis1_,basis2_,0)));
    }

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();

    double* Tp = T.data().data();

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind = 0; ind < shell_pairs.size(); ind++) {
        int P = shell_pairs[ind].first;
        int Q = shell_pairs[ind].second;
        int nP = basis1_->shell(P).nfunction();
        int nQ = basis2_->shell(Q).nfunction();
        int oP = basis1_->shell(P).function_index();
        int oQ = basis2_->shell(Q).function_index();
        #if defined(_OPENMP)
        int t = omp_get_thread_num();
        #else
        int t = 0;
        #endif
        Tints[t]->compute_shell(P,Q);
        double* Tbuffer = Tints[t]->buffer();
        if (symm) {
            double Tperm = (P == Q ? 0.5 : 1.0) * scale;
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Tp[(p + oP)*nbf2 + (q + oQ)] +=
                    Tperm * Tbuffer[p*nQ + q];
                    Tp[(q + oQ)*nbf2 + (p + oP)] +=
                    Tperm * Tbuffer[p*nQ + q];
                }
            }
        } else {
            for (int p = 0; p < nP; p++) {
                double Tperm = scale;
                for (int q = 0; q < nQ; q++) {
                    Tp[(p + oP)*nbf2 + (q + oQ)] +=
                    Tperm * Tbuffer[p*nQ + q];
                }
            }
        }
    }
}
void OneBody::compute_X(
    std::vector<Tensor>& X,
    const std::vector<double>& origin,
    const std::vector<double>& scale) const
{
    bool symm = is_symmetric();
    size_t nbf1 = basis1_->nfunction();
    size_t nbf2 = basis2_->nfunction();

    if (origin.size() != 3) throw std::runtime_error("origin must have 3 entries");
    if (X.size() != 3) throw std::runtime_error("X must have 6 entries");
    if (scale.size() != 3) throw std::runtime_error("scale must have 3 entries");
    for (int i = 0; i < 3; i++) {
        if (X[i].type() != kCore) throw std::runtime_error("X must be kCore");
        if (X[i].rank() != 2) throw std::runtime_error("X must be rank-2");
        if (X[i].dim(0) != nbf1 || X[i].dim(1) != nbf2) throw std::runtime_error("X must be nbf1 x nbf2");
    }

    #if defined(_OPENMP)
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    std::vector<std::shared_ptr<DipoleInt2C>> Xints;
    for (int t = 0; t < nthread; t++) {
        Xints.push_back(std::shared_ptr<DipoleInt2C>(new DipoleInt2C(basis1_,basis2_,0)));
        Xints[t]->set_x(origin[0]);
        Xints[t]->set_y(origin[1]);
        Xints[t]->set_z(origin[2]);
    }

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();

    double* Xp = X[0].data().data();
    double* Yp = X[1].data().data();
    double* Zp = X[2].data().data();

    size_t chunk_size = Xints[0]->chunk_size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind = 0; ind < shell_pairs.size(); ind++) {
        int P = shell_pairs[ind].first;
        int Q = shell_pairs[ind].second;
        int nP = basis1_->shell(P).nfunction();
        int nQ = basis2_->shell(Q).nfunction();
        int oP = basis1_->shell(P).function_index();
        int oQ = basis2_->shell(Q).function_index();
        #if defined(_OPENMP)
        int t = omp_get_thread_num();
        #else
        int t = 0;
        #endif
        Xints[t]->compute_shell(P,Q);
        double* Xbuffer = Xints[t]->buffer();
        double* Ybuffer = Xbuffer + chunk_size;
        double* Zbuffer = Ybuffer + chunk_size;
        if (symm) {
            double Xperm = (P == Q ? 0.5 : 1.0) * scale[0];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Xp[(p + oP)*nbf2 + (q + oQ)] +=
                    Xperm * Xbuffer[p*nQ + q];
                    Xp[(q + oQ)*nbf2 + (p + oP)] +=
                    Xperm * Xbuffer[p*nQ + q];
                }
            }
            double Yperm = (P == Q ? 0.5 : 1.0) * scale[1];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Yp[(p + oP)*nbf2 + (q + oQ)] +=
                    Yperm * Ybuffer[p*nQ + q];
                    Yp[(q + oQ)*nbf2 + (p + oP)] +=
                    Yperm * Ybuffer[p*nQ + q];
                }
            }
            double Zperm = (P == Q ? 0.5 : 1.0) * scale[2];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Zp[(p + oP)*nbf2 + (q + oQ)] +=
                    Zperm * Zbuffer[p*nQ + q];
                    Zp[(q + oQ)*nbf2 + (p + oP)] +=
                    Zperm * Zbuffer[p*nQ + q];
                }
            }
        } else {
            double Xperm = scale[0];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Xp[(p + oP)*nbf2 + (q + oQ)] +=
                    Xperm * Xbuffer[p*nQ + q];
                }
            }
            double Yperm = scale[1];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Yp[(p + oP)*nbf2 + (q + oQ)] +=
                    Yperm * Ybuffer[p*nQ + q];
                }
            }
            double Zperm = scale[2];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Zp[(p + oP)*nbf2 + (q + oQ)] +=
                    Zperm * Zbuffer[p*nQ + q];
                }
            }
        }
    }
}
void OneBody::compute_Q(
    std::vector<Tensor>& Q,
    const std::vector<double>& origin,
    const std::vector<double>& scale) const
{
    bool symm = is_symmetric();
    size_t nbf1 = basis1_->nfunction();
    size_t nbf2 = basis2_->nfunction();

    if (origin.size() != 3) throw std::runtime_error("origin must have 3 entries");
    if (Q.size() != 6) throw std::runtime_error("Q must have 6 entries");
    if (scale.size() != 6) throw std::runtime_error("scale must have 6 entries");
    for (int i = 0; i < 6; i++) {
        if (Q[i].type() != kCore) throw std::runtime_error("Q must be kCore");
        if (Q[i].rank() != 2) throw std::runtime_error("Q must be rank-2");
        if (Q[i].dim(0) != nbf1 || Q[i].dim(1) != nbf2) throw std::runtime_error("Q must be nbf1 x nbf2");
    }

    #if defined(_OPENMP)
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    std::vector<std::shared_ptr<QuadrupoleInt2C>> Qints;
    for (int t = 0; t < nthread; t++) {
        Qints.push_back(std::shared_ptr<QuadrupoleInt2C>(new QuadrupoleInt2C(basis1_,basis2_,0)));
        Qints[t]->set_x(origin[0]);
        Qints[t]->set_y(origin[1]);
        Qints[t]->set_z(origin[2]);
    }

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();

    double* XXp = Q[0].data().data();
    double* XYp = Q[1].data().data();
    double* XZp = Q[2].data().data();
    double* YYp = Q[3].data().data();
    double* YZp = Q[4].data().data();
    double* ZZp = Q[5].data().data();

    size_t chunk_size = Qints[0]->chunk_size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind = 0; ind < shell_pairs.size(); ind++) {
        int P = shell_pairs[ind].first;
        int Q = shell_pairs[ind].second;
        int nP = basis1_->shell(P).nfunction();
        int nQ = basis2_->shell(Q).nfunction();
        int oP = basis1_->shell(P).function_index();
        int oQ = basis2_->shell(Q).function_index();
        #if defined(_OPENMP)
        int t = omp_get_thread_num();
        #else
        int t = 0;
        #endif
        Qints[t]->compute_shell(P,Q);
        double* XXbuffer = Qints[t]->buffer();
        double* XYbuffer = XXbuffer + 1L * chunk_size;
        double* XZbuffer = XXbuffer + 2L * chunk_size;
        double* YYbuffer = XXbuffer + 3L * chunk_size;
        double* YZbuffer = XXbuffer + 4L * chunk_size;
        double* ZZbuffer = XXbuffer + 5L * chunk_size;
        if (symm) {
            double XXperm = (P == Q ? 0.5 : 1.0) * scale[0];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    XXp[(p + oP)*nbf2 + (q + oQ)] +=
                    XXperm * XXbuffer[p*nQ + q];
                    XXp[(q + oQ)*nbf2 + (p + oP)] +=
                    XXperm * XXbuffer[p*nQ + q];
                }
            }
            double XYperm = (P == Q ? 0.5 : 1.0) * scale[1];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    XYp[(p + oP)*nbf2 + (q + oQ)] +=
                    XYperm * XYbuffer[p*nQ + q];
                    XYp[(q + oQ)*nbf2 + (p + oP)] +=
                    XYperm * XYbuffer[p*nQ + q];
                }
            }
            double XZperm = (P == Q ? 0.5 : 1.0) * scale[2];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    XZp[(p + oP)*nbf2 + (q + oQ)] +=
                    XZperm * XZbuffer[p*nQ + q];
                    XZp[(q + oQ)*nbf2 + (p + oP)] +=
                    XZperm * XZbuffer[p*nQ + q];
                }
            }
            double YYperm = (P == Q ? 0.5 : 1.0) * scale[3];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    YYp[(p + oP)*nbf2 + (q + oQ)] +=
                    YYperm * YYbuffer[p*nQ + q];
                    YYp[(q + oQ)*nbf2 + (p + oP)] +=
                    YYperm * YYbuffer[p*nQ + q];
                }
            }
            double YZperm = (P == Q ? 0.5 : 1.0) * scale[4];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    YZp[(p + oP)*nbf2 + (q + oQ)] +=
                    YZperm * YZbuffer[p*nQ + q];
                    YZp[(q + oQ)*nbf2 + (p + oP)] +=
                    YZperm * YZbuffer[p*nQ + q];
                }
            }
            double ZZperm = (P == Q ? 0.5 : 1.0) * scale[5];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    ZZp[(p + oP)*nbf2 + (q + oQ)] +=
                    ZZperm * ZZbuffer[p*nQ + q];
                    ZZp[(q + oQ)*nbf2 + (p + oP)] +=
                    ZZperm * ZZbuffer[p*nQ + q];
                }
            }
        } else {
            double XXperm = scale[0];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    XXp[(p + oP)*nbf2 + (q + oQ)] +=
                    XXperm * XXbuffer[p*nQ + q];
                }
            }
            double XYperm = scale[1];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    XYp[(p + oP)*nbf2 + (q + oQ)] +=
                    XYperm * XYbuffer[p*nQ + q];
                }
            }
            double XZperm = scale[2];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    XZp[(p + oP)*nbf2 + (q + oQ)] +=
                    XZperm * XZbuffer[p*nQ + q];
                }
            }
            double YYperm = scale[3];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    YYp[(p + oP)*nbf2 + (q + oQ)] +=
                    YYperm * YYbuffer[p*nQ + q];
                }
            }
            double YZperm = scale[4];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    YZp[(p + oP)*nbf2 + (q + oQ)] +=
                    YZperm * YZbuffer[p*nQ + q];
                }
            }
            double ZZperm = scale[5];
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    ZZp[(p + oP)*nbf2 + (q + oQ)] +=
                    ZZperm * ZZbuffer[p*nQ + q];
                }
            }
        }
    }
}
void OneBody::compute_V(
    Tensor& V,
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& zs,
    const std::vector<double>& Zs,
    double a,
    double b,
    double w,
    double scale) const
{
    bool symm = is_symmetric();
    size_t nbf1 = basis1_->nfunction();
    size_t nbf2 = basis2_->nfunction();

    if (V.type() != kCore) throw std::runtime_error("V must be kCore");
    if (V.rank() != 2) throw std::runtime_error("V must be rank-2");
    if (V.dim(0) != nbf1 || V.dim(1) != nbf2) throw std::runtime_error("V must be nbf1 x nbf2");

    #if defined(_OPENMP)
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    std::vector<std::shared_ptr<PotentialInt2C>> Vints;
    for (int t = 0; t < nthread; t++) {
        Vints.push_back(std::shared_ptr<PotentialInt2C>(new PotentialInt2C(basis1_,basis2_,0,a,b,w)));
        Vints[t]->xs().clear();
        Vints[t]->ys().clear();
        Vints[t]->zs().clear();
        Vints[t]->Zs().clear();
        Vints[t]->xs().insert(Vints[t]->xs().begin(),xs.begin(),xs.end());
        Vints[t]->ys().insert(Vints[t]->ys().begin(),ys.begin(),ys.end());
        Vints[t]->zs().insert(Vints[t]->zs().begin(),zs.begin(),zs.end());
        Vints[t]->Zs().insert(Vints[t]->Zs().begin(),Zs.begin(),Zs.end());
    }

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();

    double* Vp = V.data().data();

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind = 0; ind < shell_pairs.size(); ind++) {
        int P = shell_pairs[ind].first;
        int Q = shell_pairs[ind].second;
        int nP = basis1_->shell(P).nfunction();
        int nQ = basis2_->shell(Q).nfunction();
        int oP = basis1_->shell(P).function_index();
        int oQ = basis2_->shell(Q).function_index();
        #if defined(_OPENMP)
        int t = omp_get_thread_num();
        #else
        int t = 0;
        #endif
        Vints[t]->compute_shell(P,Q);
        double* Vbuffer = Vints[t]->buffer();
        if (symm) {
            double Vperm = (P == Q ? 0.5 : 1.0) * scale;
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Vp[(p + oP)*nbf2 + (q + oQ)] +=
                    Vperm * Vbuffer[p*nQ + q];
                    Vp[(q + oQ)*nbf2 + (p + oP)] +=
                    Vperm * Vbuffer[p*nQ + q];
                }
            }
        } else {
            double Vperm = scale;
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Vp[(p + oP)*nbf2 + (q + oQ)] +=
                    Vperm * Vbuffer[p*nQ + q];
                }
            }
        }
    }
}
void OneBody::compute_V_nuclear(
    Tensor& V,
    const std::shared_ptr<SMolecule> mol,
    bool use_nuclear,
    double a,
    double b,
    double w,
    double scale) const
{
    std::vector<double> xs(mol->natom());
    std::vector<double> ys(mol->natom());
    std::vector<double> zs(mol->natom());
    std::vector<double> Zs(mol->natom());
    for (size_t ind = 0; ind < mol->natom(); ind++) {
        xs[ind] = mol->atom(ind).x();
        ys[ind] = mol->atom(ind).y();
        zs[ind] = mol->atom(ind).z();
        Zs[ind] = (use_nuclear ?
            mol->atom(ind).Z() :
            mol->atom(ind).Q());
    }
    compute_V(V,xs,ys,zs,Zs,a,b,w,scale);
}

} // namespace lightspeed
