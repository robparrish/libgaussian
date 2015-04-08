#ifndef TB_H
#define TB_H

#include <ambit/tensor.h>

namespace lightspeed {

class SBasisSet;
class SchwarzSieve;

class TwoBody
{
public:

    // => Constructors <= //

    /// Verbatim constructor, copies fields below
    TwoBody(const std::shared_ptr<SchwarzSieve>& sieve12,
            const std::shared_ptr<SchwarzSieve>& sieve34);

    /// Virtual destructor
    virtual ~TwoBody() {}

    // => Accessors <= //

    /// Are the basis sets on center1 and center2 the same?
    bool is_symmetric_12() const { return basis1_ == basis2_; }
    /// Are the basis sets on center3 and center4 the same?
    bool is_symmetric_34() const { return basis3_ == basis4_; }
    /// Basis set for center 1
    const std::shared_ptr<SBasisSet>& basis1() const { return basis1_; }
    /// Basis set for center 2
    const std::shared_ptr<SBasisSet>& basis2() const { return basis2_; }
    /// Basis set for center 3
    const std::shared_ptr<SBasisSet>& basis3() const { return basis3_; }
    /// Basis set for center 4
    const std::shared_ptr<SBasisSet>& basis4() const { return basis4_; }

    // => Methods <= //

    // TODO: AO Basis TwoBody Ints

protected:

    std::shared_ptr<SchwarzSieve> sieve12_;
    std::shared_ptr<SchwarzSieve> sieve34_;

    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;
    std::shared_ptr<SBasisSet> basis3_;
    std::shared_ptr<SBasisSet> basis4_;

    int deriv_;
};

class MOTwoBody : public TwoBody
{
public:

    // => Constructors <= //

    /// Verbatim constructor, copies fields below
    MOTwoBody(const std::shared_ptr<SchwarzSieve> &sieve12,
              const std::shared_ptr<SchwarzSieve> &sieve34);

    /// Virtual destructor
    virtual ~MOTwoBody() {}


    // => Methods <= //

    /// Clear all tasks from this MOTwoBody Object (not needed on first use)
    void clear();

    void add_mo_task(
            const std::string& key,
            const ambit::Tensor& C1,
            const ambit::Tensor& C2,
            const ambit::Tensor& C3,
            const ambit::Tensor& C4,
            const std::string& striping = "1234");

    std::map<std::string, ambit::Tensor> compute_mo_tasks_core() const;

    std::map<std::string, ambit::Tensor> compute_mo_tasks_disk() const;

private:

    std::vector<std::string> keys_;
    std::map<std::string, ambit::Tensor> C1s_;
    std::map<std::string, ambit::Tensor> C2s_;
    std::map<std::string, ambit::Tensor> C3s_;
    std::map<std::string, ambit::Tensor> C4s_;
    std::map<std::string, std::string> stripings_;
};

}

#endif // TB_H
