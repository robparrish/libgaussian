#ifndef DIIS_H
#define DIIS_H

#include <vector>
#include <ambit/tensor.h>

namespace lightspeed {

/**!
 * Class DIIS provides uniform and simple interface for performing Pulay's
 * Direct Inversion of the Iterative Subspace (DIIS), using arbitrary tensor
 * types and numbers/sizes of state vectors, and providing for disk swapping of
 * the tensors to free up core memory
 *
 * Note: This DIIS object uses the worst-error removal policy, and performs
 * balancing of the DIIS B matrix to attenuate condition problems near
 * convergence
 *
 * => Example Use (UHF) <=
 *
 * Description:
 *  Use DIIS to extrapolate the Fa and Fb Fock matrices with respect to the
 *  error metrics Ga and Gb, as is often needed in UHF
 *
 * Definitions:
 *  Fa/Fb - alpha and beta Fock matrices
 *  Ga/Gb - alpha and beta orbital gradients [usually X(FDS-SDF)X']
 *
 * Code:
 *
 *  // Create a DIIS object with a subspace of size 6, disk storage, and start
 *  // diis-ing after 1 vector is present (immediately)
 *  DIIS diis(1,6,true);
 *  for (int iter = 0; iter < 50; iter++) {
 *      // Build Fa,Fb (state vectors)
 *      ...
 *      // Build Ga,Gb (error vectors)
 *      ...
 *      // Add the state and error vectors
 *      diis.add_iteration({Fa,Fb},{Ga,Gb});
 *      // Extrapolate the Fock matrices in place
 *      bool diis_done = diis.extrapolate({Fa,Fb});
 *      // Use the extrapolated Fock matrices to get new orbitals
 *      ...
 *  }
 *
 **/
class DIIS final {

public:

    // => Constructors <= //

    /**!
     * Verbatim Constructor, fills fields below
     **/
    DIIS(
        size_t min_vectors,
        size_t max_vectors,
        bool use_disk);

    /// DIIS is noncopyable due to internal data references
    DIIS(const DIIS& other) = delete;
    DIIS& operator=(const DIIS& other) = delete;

    // => Accessors <= //

    /// Minimum number of state vectors before the object begins extrapolating
    size_t min_vectors() const { return min_vectors_; }
    /// Maximum number of state vectors to keep at any point in the DIIS lifetime
    size_t max_vectors() const { return max_vectors_; }
    /// Current number of state vectors in the object
    size_t cur_vectors() const { return state_vecs_.size(); }
    /// Does the object make disk copies of the state/error vectors?
    bool use_disk() const { return use_disk_; }

    // => Methods <= //

    /**
     * Reset the state of the DIIS object and flush all
     * accumulated error vectors
     *
     * @result the DIIS object will look as though it has
     * just been initialized, with cur_vectors() == 0
     **/
    void clear();

    /**!
     * Add an iteration's worth of data to the DIIS object
     *
     * NOTE: These objects are copied internally, possibly
     * striped onto disk. You may overwrite these objects as soon as this call
     * is made without loss.
     *
     * @param state_vec the current iteration's state vector
     * @param error_vec the current iteration's error vector
     * @result the DIIS object is updated with copies of the state and error
     * vector, either by adding a wholly new vector to the DIIS object, or by
     * replacing the DIIS vector with worst diagonal error if the subspace is
     * already saturated to max_vectors
     **/
    void add_iteration(
        const std::vector<ambit::Tensor>& state_vec,
        const std::vector<ambit::Tensor>& error_vec);

    /**!
     * Perform DIIS extrapolation using the current DIIS subspace
     *
     * @param state_vec the state vector to overwrite with the extrapolated
     * state_vector
     * @return true if extrapolation was performed
     * @result returns true and overwrites state_vec if cur_vectors() >=
     * min_vectors(), otherwise returns false and does not modify state_vec
     **/
    bool extrapolate(
        std::vector<ambit::Tensor>& state_vec);

protected:

    size_t min_vectors_;
    size_t max_vectors_;
    bool use_disk_;

    std::vector<std::vector<ambit::Tensor>> state_vecs_;
    std::vector<std::vector<ambit::Tensor>> error_vecs_;

    ambit::Tensor E_;

};

} // namespace lightspeed

#endif

