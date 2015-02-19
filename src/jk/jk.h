#ifndef JK_H
#define JK_H

#include <cstddef>
#include <memory> 
#include <vector>
#include <tensor.h>

namespace libgaussian {

enum class JKType {
    kDirect,
    kDF,
};

class JK {

public:
    JK(const std::shared_ptr<SBasisSet>& primary);
    virtual ~JK() {}    

    virtual JKType type() const = 0;
    virtual TensorType tensor_type() const = 0;

    const std::shared_ptr<SBasisSet>& primary() const { return primary_; }
    size_t doubles() const { return doubles_; }
    bool compute_J() const { return compute_J_; }
    bool compute_K() const { return compute_K_; }
    double a() const { return a_; }
    double b() const { return b_; }
    double w() const { return w_; }
    double product_tolerance() const { return product_tolerance_; }
    double integral_tolerance() const { return integral_tolerance_; }

    void set_doubles(size_t doubles) { doubles_ = doubles; }
    void set_compute_J(bool compute_J) { compute_J_ = compute_J; }
    void set_compute_K(bool compute_K) { compute_K_ = compute_K; }
    void set_a(double a) { a_ = a; }
    void set_b(double b) { b_ = b; }
    void set_w(double w) { w_ = w; }
    void set_product_tolerance(double product_tolerance) { product_tolerance_ = product_tolerance; }
    void set_integral_tolerance(double integral_tolerance) { integral_tolerance_ = integral_tolerance; }

    virtual void initialize() = 0;
    
    virtual void print(
        FILE* fh,
        int level = 1) const = 0;

    virtual void compute_JK_from_C(
        const std::vector<Tensor>& L,
        const std::vector<Tensor>& R,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {});

    virtual void compute_JK_from_D(
        const std::vector<Tensor>& D,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {});

    virtual void finalize() = 0;

    virtual void compute_JK_grad_from_C(
        const std::vector<Tensor>& L1,
        const std::vector<Tensor>& R1,
        const std::vector<Tensor>& L2,
        const std::vector<Tensor>& R2,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {});

    virtual void compute_JK_grad_from_D(
        const std::vector<Tensor>& D1,
        const std::vector<Tensor>& D2,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {});

    virtual void compute_JK_hess_from_C(
        const std::vector<Tensor>& L1,
        const std::vector<Tensor>& R1,
        const std::vector<Tensor>& L2,
        const std::vector<Tensor>& R2,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {});

    virtual void compute_JK_hess_from_D(
        const std::vector<Tensor>& D1,
        const std::vector<Tensor>& D2,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {});

protected:

    std::shared_ptr<SBasisSet> primary_;

    size_t doubles_;
    bool compute_J_;
    bool compute_K_;
    double a_;
    double b_;
    double w_;
    double product_tolerance_;
    double integral_tolerance_;

};

class DirectJK final: public JK {

public:
    DirectJK(const std::shared_ptr<SBasisSet>& primary);
    virtual ~DirectJK() override {}    

    JKType type() const override { return JKType::kDirect; }
    TensorType tensor_type() const override { return kCore; }

    void initialize() override {}
    
    virtual void print(
        FILE* fh,
        int level = 1) const override;

    void compute_JK_from_C(
        const std::vector<Tensor>& L,
        const std::vector<Tensor>& R,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {}) override;

    void compute_JK_from_D(
        const std::vector<Tensor>& D,
        const std::vector<Tensor>& J = {},
        const std::vector<Tensor>& K = {},
        const std::vector<Tensor>& W = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {}) override;

    void finalize() override {}

};

class DFJK final: public JK {

public:
    DFJK(
        std::shared_ptr<SBasisSet>& primary,
        std::shared_ptr<SBasisSet>& auxiliary);
    virtual ~DFJK() override {}    

    JKType type() const override { return JKType::kDF; }
    TensorType tensor_type() const override { return kCore; }

    double metric_condition() const { return metric_condition_; }

    void set_metric_condition(double metric_condition) { metric_condition_ = metric_condition; }

    void initialize() override;
    
    virtual void print(
        FILE* fh,
        int level = 1) const override;

    void compute_JK_from_C(
        const std::vector<Tensor>& L,
        const std::vector<Tensor>& R,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {}) override;

    void compute_JK_from_D(
        const std::vector<Tensor>& D,
        std::vector<Tensor>& J = {},
        std::vector<Tensor>& K = {},
        const std::vector<double>& scaleJ = {},
        const std::vector<double>& scaleK = {}) override;

    void finalize() override;

protected:
    
    std::shared_ptr<SBasisSet> auxiliary_;
    double metric_condition_;

};

} // namespace libgaussian

#endif
