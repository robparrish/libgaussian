#include <math.h>
#include <core/basisset.h>
#include <mints/schwarz.h>
#include "jk.h"

using namespace tensor;

namespace libgaussian {

DirectJK::DirectJK(const std::shared_ptr<SchwarzSieve>& sieve) :
    JK(sieve)
{
}
void DirectJK::print(FILE* fh) const 
{
    fprintf(fh, "  DirectJK:\n\n");
    fprintf(fh, "    Primary Basis   = %s\n", primary_->name().c_str());
    fprintf(fh, "    Doubles         = %zu\n", doubles_);
    fprintf(fh, "    Compute J       = %s\n", compute_J_ ? "Yes" : "No");
    fprintf(fh, "    Compute K       = %s\n", compute_K_ ? "Yes" : "No");
    fprintf(fh, "    Operator a      = %14.6E\n", a_);
    fprintf(fh, "    Operator b      = %14.6E\n", b_);
    fprintf(fh, "    Operator w      = %14.6E\n", w_);
    fprintf(fh, "    Product Cutoff  = %11.3E\n", product_cutoff_);
    fprintf(fh, "    Integral Cutoff = %11.3E\n", sieve_->cutoff());
    fprintf(fh, "\n");
}
void DirectJK::compute_JK_from_C(
    const std::vector<tensor::Tensor>& L,
    const std::vector<tensor::Tensor>& R,
    std::vector<tensor::Tensor>& J,
    std::vector<tensor::Tensor>& K,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    std::vector<Tensor> D(L.size());
    std::vector<bool> symm(L.size());
    for (size_t ind = 0; ind < L.size(); ind++) {
        D[ind] = Tensor::build(kCore,"D",{L[ind].dim(0),R[ind].dim(0)});
        D[ind]("pq") = L[ind]("pi") * R[ind]("qi");
        symm[ind] = L[ind] == R[ind];
    }
    compute_JK_from_D(D,symm,J,K,scaleJ,scaleK);
}
void DirectJK::compute_JK_from_D(
    const std::vector<tensor::Tensor>& D,
    const std::vector<bool>& symm,
    std::vector<tensor::Tensor>& J,
    std::vector<tensor::Tensor>& K,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    // => Default Arguments <= //

    if (!scaleJ.size()) {
    } 
    // TODO

    // => Symmetric? <= //
    
    bool lr_symmetric = true;
    for (sval : symm) {
        lr_symmetric = lr_symmetric && sval;
    }

    // => Sizing <= //

    int nbf     = primary_->nf(unction);
    int nshell  = primary_->nshell();
    int nthread = omp_get_max_threads();

    // => Integrals <= //

    std::vector<std::shared_ptr<PotentialInt4C>> ints;
    for (t = 0; t < nthread; t++) {
        ints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(
            primary_,primary_,primary_,primary_,0,a_,b_w_));
    }

    // => Task Blocking <= //

    std::vector<int> task_shells;
    std::vector<int> task_starts;

    // > Atomic Blocking < //

    // TODO: Make this explicit
    int atomic_ind = -1;
    for (int P = 0; P < nshell; P++) {
        if (primary_->shell(P).atomic_index() > atomic_ind) {
            task_starts.push_back(P);
            atomic_ind++;
        }
        task_shells.push_back(P);
    }
    task_starts.push_back(nshell);

    // < End Atomic Blocking > //

    size_t ntask = task_starts.size() - 1;
    size_t ntask2 = ntask * ntask;
    size_t ntask4 = ntask * ntask * ntask * ntask;

    std::vector<int> task_offsets;
    task_offsets.push_back(0);
    for (int P2 = 0; P2 < primary_->nshell(); P2++) {
        task_offsets.push_back(task_offsets[P2] + primary_->shell(task_shells[P2]).nfunction());
    }

    size_t max_task = 0L;
    for (int task = 0; task < ntask; task++) {
        size_t size = 0L;
        for (int P2 = task_starts[task]; P2 < task_starts[task+1]; P2++) {
            size += primary_->shell(task_shells[P2]).nfunction();
        }
        max_task = (max_task >= size ? max_task : size);
    }

    #if 0
        fprintf(outfile, "  ==> DirectJK: Task Blocking <==\n\n");
        for (int task = 0; task < ntask; task++) {
            fprintf(outfile, "  Task: %3d, Task Start: %4d, Task End: %4d\n", task, task_starts[task], task_starts[task+1]);
            for (int P2 = task_starts[task]; P2 < task_starts[task+1]; P2++) {
                int P = task_shells[P2];
                int size = primary_->shell(P).nfunction();
                int off  = primary_->shell(P).function_index();
                int off2 = task_offsets[P2];
                fprintf(outfile, "    Index %4d, Shell: %4d, Size: %4d, Offset: %4d, Offset2: %4d\n", P2, P, size, off, off2);
            }
        }
        fprintf(outfile, "\n");
    #endif

    // => Significant Task Pairs (PQ|-style <= //

    std::vector<std::pair<int, int> > task_pairs;
    for (int Ptask = 0; Ptask < ntask; Ptask++) {
        for (int Qtask = 0; Qtask < ntask; Qtask++) {
            if (Qtask > Ptask) continue;
            bool found = false;
            for (int P2 = task_starts[Ptask]; P2 < task_starts[Ptask+1]; P2++) {
                for (int Q2 = task_starts[Qtask]; Q2 < task_starts[Qtask+1]; Q2++) {
                    int P = task_shells[P2];
                    int Q = task_shells[Q2];
                    if (sieve_->shell_pair_significant(P,Q)) {
                        found = true;
                        task_pairs.push_back(std::pair<int,int>(Ptask,Qtask));
                        break;
                    }
                }
                if (found) break;
            }
        }
    }
    size_t ntask_pair = task_pairs.size();
    size_t ntask_pair2 = ntask_pair * ntask_pair;

    // => Intermediate Buffers <= //

    std::vector<std::vector<boost::shared_ptr<Matrix> > > JKT;
    for (int thread = 0; thread < nthread; thread++) {
        std::vector<boost::shared_ptr<Matrix> > JK2;
        for (int ind = 0; ind < D.size(); ind++) {
            JK2.push_back(boost::shared_ptr<Matrix>(new Matrix("JKT", (lr_symmetric ? 6 : 10) * max_task, max_task)));
        }
        JKT.push_back(JK2);
    }

    // => Benchmarks <= //

    size_t computed_shells = 0L;

    // ==> Master Task Loop <== //

    #pragma omp parallel for num_threads(nthread) schedule(dynamic) reduction(+: computed_shells)
    for (size_t task = 0L; task < ntask_pair2; task++) {

        size_t task1 = task / ntask_pair;
        size_t task2 = task % ntask_pair;

        int Ptask = task_pairs[task1].first;
        int Qtask = task_pairs[task1].second;
        int Rtask = task_pairs[task2].first;
        int Stask = task_pairs[task2].second;

        // GOTCHA! Thought this should be RStask > PQtask, but
        // H2/3-21G: Task (10|11) gives valid quartets (30|22) and (31|22)
        // This is an artifact that multiple shells on each task allow
        // for for the Ptask's index to possibly trump any RStask pair,
        // regardless of Qtask's index
        if (Rtask > Ptask) continue;

        //printf("Task: %2d %2d %2d %2d\n", Ptask, Qtask, Rtask, Stask);

        int nPtask = task_starts[Ptask + 1] - task_starts[Ptask];
        int nQtask = task_starts[Qtask + 1] - task_starts[Qtask];
        int nRtask = task_starts[Rtask + 1] - task_starts[Rtask];
        int nStask = task_starts[Stask + 1] - task_starts[Stask];

        int P2start = task_starts[Ptask];
        int Q2start = task_starts[Qtask];
        int R2start = task_starts[Rtask];
        int S2start = task_starts[Stask];

        int dPsize = task_offsets[P2start + nPtask] - task_offsets[P2start];
        int dQsize = task_offsets[Q2start + nQtask] - task_offsets[Q2start];
        int dRsize = task_offsets[R2start + nRtask] - task_offsets[R2start];
        int dSsize = task_offsets[S2start + nStask] - task_offsets[S2start];

        int thread = 0;
        #ifdef _OPENMP
            thread = omp_get_thread_num();
        #endif

        // => Master shell quartet loops <= //

        bool touched = false;
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
        for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
            if (Q2 > P2) continue;
            int P = task_shells[P2];
            int Q = task_shells[Q2];
            if (!sieve_->shell_pair_significant(P,Q)) continue;
        for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
        for (int S2 = S2start; S2 < S2start + nStask; S2++) {
            if (S2 > R2) continue;
            int R = task_shells[R2];
            int S = task_shells[S2];
            if (R2 * nshell + S2 > P2 * nshell + Q2) continue;
            if (!sieve_->shell_pair_significant(R,S)) continue;
            if (!sieve_->shell_significant(P,Q,R,S)) continue;

            //printf("Quartet: %2d %2d %2d %2d\n", P, Q, R, S);

            ints[thread]->compute_shell(P,Q,R,S)
            computed_shells++;

            double* buffer = ints[thread]->buffer();

            int Psize = primary_->shell(P).nfunction();
            int Qsize = primary_->shell(Q).nfunction();
            int Rsize = primary_->shell(R).nfunction();
            int Ssize = primary_->shell(S).nfunction();

            int Poff = primary_->shell(P).function_index();
            int Qoff = primary_->shell(Q).function_index();
            int Roff = primary_->shell(R).function_index();
            int Soff = primary_->shell(S).function_index();

            int Poff2 = task_offsets[P2] - task_offsets[P2start];
            int Qoff2 = task_offsets[Q2] - task_offsets[Q2start];
            int Roff2 = task_offsets[R2] - task_offsets[R2start];
            int Soff2 = task_offsets[S2] - task_offsets[S2start];

            for (int ind = 0; ind < D.size(); ind++) {
                double** Dp = D[ind]->pointer();
                double** JKTp = JKT[thread][ind]->pointer();
                const double* buffer2 = buffer;

                if (!touched) {
                    ::memset((void*) JKTp[0L * max_task], '\0', dPsize * dQsize * sizeof(double));
                    ::memset((void*) JKTp[1L * max_task], '\0', dRsize * dSsize * sizeof(double));
                    ::memset((void*) JKTp[2L * max_task], '\0', dPsize * dRsize * sizeof(double));
                    ::memset((void*) JKTp[3L * max_task], '\0', dPsize * dSsize * sizeof(double));
                    ::memset((void*) JKTp[4L * max_task], '\0', dQsize * dRsize * sizeof(double));
                    ::memset((void*) JKTp[5L * max_task], '\0', dQsize * dSsize * sizeof(double));
                    if (!lr_symmetric_) {
                        ::memset((void*) JKTp[6L * max_task], '\0', dRsize * dPsize * sizeof(double));
                        ::memset((void*) JKTp[7L * max_task], '\0', dSsize * dPsize * sizeof(double));
                        ::memset((void*) JKTp[8L * max_task], '\0', dRsize * dQsize * sizeof(double));
                        ::memset((void*) JKTp[9L * max_task], '\0', dSsize * dQsize * sizeof(double));
                    }
                }

                double* J1p = JKTp[0L * max_task];
                double* J2p = JKTp[1L * max_task];
                double* K1p = JKTp[2L * max_task];
                double* K2p = JKTp[3L * max_task];
                double* K3p = JKTp[4L * max_task];
                double* K4p = JKTp[5L * max_task];
                double* K5p;
                double* K6p;
                double* K7p;
                double* K8p;
                if (!lr_symmetric_) {
                    K5p = JKTp[6L * max_task];
                    K6p = JKTp[7L * max_task];
                    K7p = JKTp[8L * max_task];
                    K8p = JKTp[9L * max_task];
                }

                double prefactor = 1.0;
                if (P == Q)           prefactor *= 0.5;
                if (R == S)           prefactor *= 0.5;
                if (P == R && Q == S) prefactor *= 0.5;

                for (int p = 0; p < Psize; p++) {
                for (int q = 0; q < Qsize; q++) {
                for (int r = 0; r < Rsize; r++) {
                for (int s = 0; s < Ssize; s++) {
                    J1p[(p + Poff2) * dQsize + q + Qoff2] += prefactor * (Dp[r + Roff][s + Soff] + Dp[s + Soff][r + Roff]) * (*buffer2);
                    J2p[(r + Roff2) * dSsize + s + Soff2] += prefactor * (Dp[p + Poff][q + Qoff] + Dp[q + Qoff][p + Poff]) * (*buffer2);
                    K1p[(p + Poff2) * dRsize + r + Roff2] += prefactor * (Dp[q + Qoff][s + Soff]) * (*buffer2);
                    K2p[(p + Poff2) * dSsize + s + Soff2] += prefactor * (Dp[q + Qoff][r + Roff]) * (*buffer2);
                    K3p[(q + Qoff2) * dRsize + r + Roff2] += prefactor * (Dp[p + Poff][s + Soff]) * (*buffer2);
                    K4p[(q + Qoff2) * dSsize + s + Soff2] += prefactor * (Dp[p + Poff][r + Roff]) * (*buffer2);
                    if (!lr_symmetric_) {
                        K5p[(r + Roff2) * dPsize + p + Poff2] += prefactor * (Dp[s + Soff][q + Qoff]) * (*buffer2);
                        K6p[(s + Soff2) * dPsize + p + Poff2] += prefactor * (Dp[r + Roff][q + Qoff]) * (*buffer2);
                        K7p[(r + Roff2) * dQsize + q + Qoff2] += prefactor * (Dp[s + Soff][p + Poff]) * (*buffer2);
                        K8p[(s + Soff2) * dQsize + q + Qoff2] += prefactor * (Dp[r + Roff][p + Poff]) * (*buffer2);
                    }
                    buffer2++;
                }}}}

            }
            touched = true;

        }}}} // End Shell Quartets

        if (!touched) continue;

        // => Stripe out <= //

        for (int ind = 0; ind < D.size(); ind++) {
            double** JKTp = JKT[thread][ind]->pointer();
            double** Jp = J[ind]->pointer();
            double** Kp = K[ind]->pointer();

            double* J1p = JKTp[0L * max_task];
            double* J2p = JKTp[1L * max_task];
            double* K1p = JKTp[2L * max_task];
            double* K2p = JKTp[3L * max_task];
            double* K3p = JKTp[4L * max_task];
            double* K4p = JKTp[5L * max_task];
            double* K5p;
            double* K6p;
            double* K7p;
            double* K8p;
            if (!lr_symmetric_) {
                K5p = JKTp[6L * max_task];
                K6p = JKTp[7L * max_task];
                K7p = JKTp[8L * max_task];
                K8p = JKTp[9L * max_task];
            }

            // > J_PQ < //

            for (int P2 = 0; P2 < nPtask; P2++) {
            for (int Q2 = 0; Q2 < nQtask; Q2++) {
                int P = task_shells[P2start + P2];
                int Q = task_shells[Q2start + Q2];
                int Psize = primary_->shell(P).nfunction();
                int Qsize = primary_->shell(Q).nfunction();
                int Poff =  primary_->shell(P).function_index();
                int Qoff =  primary_->shell(Q).function_index();
                int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                for (int p = 0; p < Psize; p++) {
                for (int q = 0; q < Qsize; q++) {
                    #pragma omp atomic
                    Jp[p + Poff][q + Qoff] += J1p[(p + Poff2) * dQsize + q + Qoff2];
                }}
            }}

            // > J_RS < //

            for (int R2 = 0; R2 < nRtask; R2++) {
            for (int S2 = 0; S2 < nStask; S2++) {
                int R = task_shells[R2start + R2];
                int S = task_shells[S2start + S2];
                int Rsize = primary_->shell(R).nfunction();
                int Ssize = primary_->shell(S).nfunction();
                int Roff =  primary_->shell(R).function_index();
                int Soff =  primary_->shell(S).function_index();
                int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                for (int r = 0; r < Rsize; r++) {
                for (int s = 0; s < Ssize; s++) {
                    #pragma omp atomic
                    Jp[r + Roff][s + Soff] += J2p[(r + Roff2) * dSsize + s + Soff2];
                }}
            }}

            // > K_PR < //

            for (int P2 = 0; P2 < nPtask; P2++) {
            for (int R2 = 0; R2 < nRtask; R2++) {
                int P = task_shells[P2start + P2];
                int R = task_shells[R2start + R2];
                int Psize = primary_->shell(P).nfunction();
                int Rsize = primary_->shell(R).nfunction();
                int Poff =  primary_->shell(P).function_index();
                int Roff =  primary_->shell(R).function_index();
                int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                for (int p = 0; p < Psize; p++) {
                for (int r = 0; r < Rsize; r++) {
                    #pragma omp atomic
                    Kp[p + Poff][r + Roff] += K1p[(p + Poff2) * dRsize + r + Roff2];
                    if (!lr_symmetric_) {
                        #pragma omp atomic
                        Kp[r + Roff][p + Poff] += K5p[(r + Roff2) * dPsize + p + Poff2];
                    }
                }}
            }}

            // > K_PS < //

            for (int P2 = 0; P2 < nPtask; P2++) {
            for (int S2 = 0; S2 < nStask; S2++) {
                int P = task_shells[P2start + P2];
                int S = task_shells[S2start + S2];
                int Psize = primary_->shell(P).nfunction();
                int Ssize = primary_->shell(S).nfunction();
                int Poff =  primary_->shell(P).function_index();
                int Soff =  primary_->shell(S).function_index();
                int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                for (int p = 0; p < Psize; p++) {
                for (int s = 0; s < Ssize; s++) {
                    #pragma omp atomic
                    Kp[p + Poff][s + Soff] += K2p[(p + Poff2) * dSsize + s + Soff2];
                    if (!lr_symmetric_) {
                        #pragma omp atomic
                        Kp[s + Soff][p + Poff] += K6p[(s + Soff2) * dPsize + p + Poff2];
                    }
                }}
            }}

            // > K_QR < //

            for (int Q2 = 0; Q2 < nQtask; Q2++) {
            for (int R2 = 0; R2 < nRtask; R2++) {
                int Q = task_shells[Q2start + Q2];
                int R = task_shells[R2start + R2];
                int Qsize = primary_->shell(Q).nfunction();
                int Rsize = primary_->shell(R).nfunction();
                int Qoff =  primary_->shell(Q).function_index();
                int Roff =  primary_->shell(R).function_index();
                int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                for (int q = 0; q < Qsize; q++) {
                for (int r = 0; r < Rsize; r++) {
                    #pragma omp atomic
                    Kp[q + Qoff][r + Roff] += K3p[(q + Qoff2) * dRsize + r + Roff2];
                    if (!lr_symmetric_) {
                        #pragma omp atomic
                        Kp[r + Roff][q + Qoff] += K7p[(r + Roff2) * dQsize + q + Qoff2];
                    }
                }}
            }}

            // > K_QS < //

            for (int Q2 = 0; Q2 < nQtask; Q2++) {
            for (int S2 = 0; S2 < nStask; S2++) {
                int Q = task_shells[Q2start + Q2];
                int S = task_shells[S2start + S2];
                int Qsize = primary_->shell(Q).nfunction();
                int Ssize = primary_->shell(S).nfunction();
                int Qoff =  primary_->shell(Q).function_index();
                int Soff =  primary_->shell(S).function_index();
                int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                for (int q = 0; q < Qsize; q++) {
                for (int s = 0; s < Ssize; s++) {
                    #pragma omp atomic
                    Kp[q + Qoff][s + Soff] += K4p[(q + Qoff2) * dSsize + s + Soff2];
                    if (!lr_symmetric_) {
                        #pragma omp atomic
                        Kp[s + Soff][q + Qoff] += K8p[(s + Soff2) * dQsize + q + Qoff2];
                    }
                }}
            }}

        } // End stripe out

    } // End master task list

    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->scale(2.0);
        J[ind]->hermitivitize();
        if (lr_symmetric_) {
            K[ind]->scale(2.0);
            K[ind]->hermitivitize();
        }
    }
}

} // namespace libgaussian
