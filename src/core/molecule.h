#ifndef MOLECULE_H
#define MOLECULE_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace libgaussian {

/**!
 * Class SAtom is a data container for a simple atom
 *
 *  To specify the atom type, three fields are used (mostly for convenience)
 *  - label the user-specified label, e.g., He3 or Gh(He3) (used for printing)
 *  - symbol the capitalized atomic symbol, e.g., HE (or X for dummy)
 *  - N the integral atomic number (or 0 for dummy)
 *
 * To specify coordinates, the usual fields are:
 *  - x the x coordinate in au 
 *  - y the y coordinate in au 
 *  - z the z coordinate in au 
 *
 * To specify the nuclear, alpha electronic, and beta electronic charges, three
 * fields are used:
 *  - Z the nuclear charge, e.g., the number of protons
 *  - Ya the alpha electronic charge, e.g., the number of alpha electrons
 *  - Yb the beta electronic charge, e.g., the number of beta electrons
 *
 * These fields should all be specified as positive. The total charge Q is
 * computed as Z - Ya - Yb. The spin charge is computed as S = Ya - Yb. Note
 * that these fields may all be fractional. For Z, this allows one to perform
 * computations with an effective nuclear charge which is different from Z,
 * e.g., for applying psuedopotentials or for performing alchemical studies.
 * The Ya and Yb fields are intended to provide for the ability to the user to
 * specify the quasiclassical atomic charge and its spin polarization, e.g.,
 * for producing a better superposition of atomic density (SAD) guess.
 * 
 * The index field is a back-reference to the atom's position in its parent
 * molecule, and is used in placing data pertaining to the molecule's atom in
 * various arrays.
 *
 * - Rob Parrish, 15 February, 2015
 **/
class SAtom {

public:

    // => Constructors <= //
    
    /// Verbatim constructor, fills fields below
    SAtom(
        const std::string& label,
        const std::string& symbol,
        int N,
        double x,
        double y,
        double z,
        double Z,
        double Ya,  
        double Yb,
        size_t index) :
        label_(label),
        symbol_(symbol),
        N_(N),
        x_(x),
        y_(y),
        z_(z),
        Z_(Z),
        Ya_(Ya),  
        Yb_(Yb),
        index_(index)
        {}
   
    /// Default constructor, no initialization
    SAtom() {}

    // => Accessors <= //

    /// User-specified atom label, e.g He4 or Gh(He), for printing
    const std::string& label() const { return label_; }
    /// Capitalized atomic symbol, e.g., HE (or X for dummy)
    const std::string& symbol() const { return symbol_; }
    /// True atomic number of element (or 0 for dummy)
    int N() const { return N_; }
    /// x coordinate in au
    double x() const { return x_; }
    /// y coordinate in au
    double y() const { return y_; }
    /// z coordinate in au
    double z() const { return z_; }
    /// Nuclear charge in au (might be not equal to N)
    double Z() const { return Yb_; }
    /// Alpha electronic charge in au 
    double Ya() const { return Ya_; }
    /// Beta electronic charge in au 
    double Yb() const { return Yb_; }
    /// Total electronic charge in au
    double Y() const { return Ya_ + Yb_; }
    /// Total charge in au
    double Q() const { return Z_ - Ya_ - Yb_; }
    /// Total spin charge in au
    double S() const { return Ya_ - Yb_; }
    /// Index of this atom within its containing molecule 
    size_t index() const { return index_; }

    // => Setters <= //

    void set_label(const std::string& label) { label_ = label; }
    void set_symbol(const std::string& symbol) { symbol_ = symbol; }
    void set_N(int N) { N_ = N; }
    void set_x(double x) { x_ = x; }
    void set_y(double y) { y_ = y; }
    void set_z(double z) { z_ = z; }
    void set_Ya(double Ya) { Ya_ = Ya; }
    void set_Yb(double Yb) { Yb_ = Yb; }
    void set_index(size_t index) { index_ = index; }

    // => Methods <= //

    /// Return the distance to SAtom other in au
    double distance(const SAtom& other) const; 

private:
    std::string label_;
    std::string symbol_;
    int N_;
    double x_;
    double y_;
    double z_;
    double Z_;
    double Ya_;
    double Yb_;
    size_t index_;   

};

/**!
 * Class SMolecule is a simple wrapper around a vector of atoms, plus some utility
 * functions. The class is deliberately immutable, to prevent clever bastards
 * from reorienting your molecule halfway through the computation!
 *
 * - Rob Parrish, 15 February, 2015
 **/ 
class SMolecule {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below and resets indices in atoms to
     * the ordering of this molecule
     **/
    SMolecule(
        const std::string& name,
        const std::vector<SAtom>& atoms);

    /// Default constructor, no initialization
    SMolecule() {}

    // => Accessors <= //

    /// The molecule's name
    const std::string& name() const { return name_; }
    /// Number of atoms in this molecule
    size_t natom() const { return atoms_.size(); }
    /// Return the A-th atom in this molecule
    const SAtom& atom(int A) const { return atoms_[A]; }
    /// The array of atoms which comprise this molecule
    const std::vector<SAtom>& atoms() const { return atoms_; }       
    
    // => Methods <= //

    /// Return the nuclear repulsion energy in au for this molecule
    double nuclear_repulsion_energy(
        bool use_nuclear,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0) const;

    /// Return the nuclear repulsion energy in au between this and other molecule
    double nuclear_repulsion_energy(
        const std::shared_ptr<SMolecule>& other,
        bool use_nuclear_this = true,
        bool use_nuclear_other = true,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0) const;
     
private:
    std::string name_;
    std::vector<SAtom> atoms_; 

};


} // namespace libgaussian

#endif
