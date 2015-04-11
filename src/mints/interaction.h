#ifndef INTERACTION_H
#define INTERACTION_H

#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace lightspeed {

class Boys;

class Interaction {

public:
    Interaction() {}

    virtual ~Interaction() {}

    const std::string& name() { return name_; }
    const std::string& description() { return description_; }
    const std::map<std::string, std::string>& parameters() { return parameters_; }

    // => Factories <= //

    static std::shared_ptr<Interaction> coulomb();
    static std::shared_ptr<Interaction> ewald(double a, double b, double w);
    static std::shared_ptr<Interaction> f12(double slater_exponent);

    // => Advanced <= //

    virtual void initialize_fundamental(int Lmax) { buffer_.resize(Lmax+1); }
    virtual void compute_fundamental(double rho, double T, int L) = 0;
    const std::vector<double>& buffer() const { return buffer_; }

protected:

    // => User-Level <= //

    std::string name_;
    std::string description_;
    std::map<std::string, std::string> parameters_;

    // => Advanced <= //

    std::vector<double> buffer_;

};

class EwaldInteraction final : public Interaction {

public:
    EwaldInteraction(double a, double b, double w);

    void initialize_fundamental(int Lmax) override final;
    void compute_fundamental(double rho, double T, int L) override final;

protected:
    double a_;
    double b_;
    double w_;
    std::shared_ptr<Boys> boys_;
    const double* boys_buffer_;
};

class F12Interaction final : public Interaction
{
public:
    F12Interaction(double slater_exponent);
    F12Interaction(const std::vector<double>& cs, const std::vector<double>& es);

    void initialize_fundamental(int Lmax) override final;
    void compute_fundamental(double rho, double T, int L) override final;

protected:
    double slater_exponent_;
    std::vector<double> cs_;
    std::vector<double> es_;
    size_t nparam_;
};

} // namespace lightspeed

#endif
