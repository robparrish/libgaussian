#!/usr/bin/env python
import sys, os, re, copy, math, time

# ambit package must be in the PYTHONPATH for this
#import ambit as at
# pylightspeed.so must be in the PYTHONPATH for this
from pylightspeed import * 
import ambit as am

# PSI4's bohr to angstrom conversion
pc_bohr2ang_ = 0.52917720859 

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

# Atoms may be QM (quantum), GH (ghost), MM (molecular mechanics), or DM (dummy)
AtomType = Enum(["QM", "GH", "MM", "DM"])

# Map from atom name to charge, vDW radius, nfrozen
atom_data_ = {
'X'  :  [0,  0.000, 0],
'H'  :  [1,  0.402, 0],
'HE' :  [2,  0.700, 0],
'LI' :  [3,  1.230, 1],
'BE' :  [4,  0.900, 1],
'B'  :  [5,  1.000, 1],
'C'  :  [6,  0.762, 1],
'N'  :  [7,  0.676, 1],
'O'  :  [8,  0.640, 1],
'F'  :  [9,  0.630, 1],
'NE' :  [10, 0.700, 1],
'NA' :  [11, 1.540, 5],
'MG' :  [12, 1.360, 5],
'AL' :  [13, 1.180, 5],
'SI' :  [14, 1.300, 5],
'P'  :  [15, 1.094, 5],
'S'  :  [16, 1.253, 5],
'CL' :  [17, 1.033, 5],
'AR' :  [18, 1.740, 5],
}

# Shell name to L
shell_data_ = {
'S' : 0,
'P' : 1,
'D' : 2,
'F' : 3,
'G' : 4,
'H' : 5,
'I' : 6,
'K' : 7}

class Atom:
    
    def __init__(self, atype, label, symbol, x, y, z): 
        self.atype = atype
        self.label = label
        self.symbol = symbol
        self.N = int(atom_data_[self.symbol][0])
        self.x = x
        self.y = y
        self.z = z
        self.Z = float(self.N)
        self.Ya = 0.5 * float(self.Z)
        self.Yb = 0.5 * float(self.Z)

class Molecule:

    def __init__(self, name):
        self.name = name
        self.atoms = []    
        self.charge = 0.0
        self.multiplicity = 1.0 

    @staticmethod
    def from_xyz_file(filename, angstroms = True):
        s = 1.0
        if angstroms:
            s = 1.0 / pc_bohr2ang_

        fh = open(filename, 'r')
        lines = fh.readlines();
        fh.close()

        name = re.match(r'^(\S+?)(\.xyz)?$', os.path.basename(os.path.normpath(filename))).group(1)

        mol = Molecule(name)
        N = int(re.match(r'^\s*(\d+)\s*$', lines[0]).group(1))
        lines = lines[2:]
        for A in range(N):
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', lines[A])
            mol.atoms.append(Atom(
                AtomType.QM,
                mobj.group(1),
                mobj.group(1).upper(), # TODO
                s*float(mobj.group(2)),
                s*float(mobj.group(3)),
                s*float(mobj.group(4))))
        return mol

    def SMolecule(self):
        SAtoms = SAtomVec()
        for atom in self.atoms: 
            SAtoms.append(SAtom(
                atom.label,
                atom.symbol,
                atom.N,
                atom.x,
                atom.y,
                atom.z,
                atom.Z,
                atom.Ya,
                atom.Yb))
        return SMolecule(self.name, SAtoms)
        
    def natom(self):
        return len(self.atoms)

    def nalpha(self):
        
        # Electron count
        Y = 0.0
        for atom in self.atoms:
            Y += atom.Z
        Y -= self.charge
        
        # Alpha overage
        S = self.multiplicity - 1.0

        # Paired/matched electron count
        E = Y - S;

        if (abs(2.0 * int(E / 2.0) - E) > 1.0E-9):
            raise Exception('Charge/Multiplicty of %f %f is impossible for this system' % (self.charge,self.multiplicity))

        return E / 2.0 + S

    def nbeta(self):
        
        # Electron count
        Y = 0.0
        for atom in self.atoms:
            Y += atom.Z
        Y -= self.charge
        
        # Alpha overage
        S = self.multiplicity - 1.0

        # Paired/matched electron count
        E = Y - S;

        if (abs(2.0 * int(E / 2.0) - E) > 1.0E-9):
            raise Exception('Charge/Multiplicty of %f %f is impossible for this system' % (self.charge,self.multiplicity))

        return E / 2.0
        
class GaussianShell:

    def __init__(self, x, y, z, is_ghost, is_spherical, am, N, ws, cs, es):
        self.x = x
        self.y = y
        self.z = z
        self.is_ghost = is_ghost
        self.is_spherical = is_spherical
        self.am = am
        self.N = N
        self.ws = copy.deepcopy(ws)
        self.cs = copy.deepcopy(cs)
        self.es = copy.deepcopy(es)

    @staticmethod
    def build_from_gbs(is_spherical,am,N,ws,es):

        pi32 = math.pow(math.pi,1.5)
        twoL = math.pow(2,am)
        dfact = 1
        for l in range(1,am+1):
            dfact *= 2*l-1

        K = len(ws)
        cs = [x for x in ws]
        for k in range(K):
            cs[k] *= math.sqrt(twoL * math.pow(es[k] + es[k],am + 1.5) / (pi32 * dfact))

        V = 0.0
        for k1 in range(K):  
            for k2 in range(K):
                V += math.pow(math.sqrt(4*es[k1]*es[k2])/(es[k1]+es[k2]),am+1.5) * ws[k1] * ws[k2]
        V = math.sqrt(N) * math.pow(V,-1.0/2.0)
        cs = [V * x for x in cs]

        return GaussianShell(
            0.0,
            0.0,
            0.0,
            False,
            is_spherical,
            am,
            N,
            ws,
            cs,
            es)

    def ncartesian(self):
        return (self.am + 1) * (self.am + 2) / 2

    def nfunction(self):
        if self.is_spherical:
            return 2 * self.am + 1
        else:
            return (self.am + 1) * (self.am + 2) / 2

    def nprimitive(self):
        return len(self.cs)

class BasisSet:

    def __init__(self, name):
        self.name = name
        self.shells = []
        self.atoms_to_shell_inds = []

    @staticmethod
    def from_gbs_file(molecule, filename):

        name = re.match(r'^(\S+?)(\.gbs)?$', os.path.basename(os.path.normpath(filename))).group(1)

        fh = open(filename, 'r')
        lines = fh.readlines();
        fh.close()

        lines2 = []
        for line in lines:
            if re.match(r'^\s*$', line):
                continue
            if re.match(r'^\s*!', line):
                continue
            lines2.append(line) 

        spherical = False
        if (re.match(r'^\s*cartesian\s*$', lines2[0], re.IGNORECASE)):
            spherical = False
        elif (re.match(r'^\s*spherical\s*$', lines2[0], re.IGNORECASE)):
            spherical = True
        else:
            raise Exception("Where is the cartesian/spherical line?")
         
        lines2 = lines2[1:]

        star_inds = [0]
        for ind in range(len(lines2)):
            line = lines2[ind]
            if re.match('^\s*\*\*\*\*\s*$', line):
                star_inds.append(ind)
        star_inds.append(len(lines2))

        atom_gbs = {}
        for k in range(len(star_inds) - 1):
            ind1 = star_inds[k] + 1
            ind2 = star_inds[k+1]
            if (ind2 - ind1) <= 0:
                continue
            mobj = re.match(r'^\s*(\S+)\s+(\d+)\s*$', lines2[ind1])
            if mobj == None:
                raise Exception("Where is the ID V line?")
            if mobj.group(2) != '0':
                continue
            atom_gbs[mobj.group(1).upper()] = lines2[ind1+1:ind2]

        atom_shells = {}
        for key, atom in atom_gbs.items():
            atom_stuff = []
            
            ind = 0;
            while ind < len(atom):
                mobj = re.match(r'^\s*(\S+)\s+(\d+)\s+(\S+)\s*$', atom[ind])
                if mobj == None:
                    raise Exception("Where is the L K N line?")
                ID = mobj.group(1).upper()
                K = int(mobj.group(2))
                N = float(mobj.group(3))
                ind=ind+1
                if ID == 'SPD':
                    E  = []
                    W0 = []
                    W1 = []
                    W2 = []
                    for k in range(K):
                        mobj = re.match('^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', re.sub(r'[Dd]', 'E', atom[ind + k]))
                        if mobj == None:
                            raise Exception("Where is the E W0 W1 W2 line?")
                        E.append(float(mobj.group(1)))
                        W0.append(float(mobj.group(2)))
                        W1.append(float(mobj.group(3)))
                        W2.append(float(mobj.group(4)))
                    atom_stuff.append(GaussianShell.build_from_gbs(spherical,0,N,W0,E))
                    atom_stuff.append(GaussianShell.build_from_gbs(spherical,1,N,W1,E))
                    atom_stuff.append(GaussianShell.build_from_gbs(spherical,2,N,W2,E))
                    ind=ind+K
                elif ID == 'SP':
                    E  = []
                    W0 = []
                    W1 = []
                    for k in range(K):
                        mobj = re.match('^\s*(\S+)\s+(\S+)\s+(\S+)\s*$', re.sub(r'[Dd]', 'E', atom[ind + k]))
                        if mobj == None:
                            raise Exception("Where is the E W0 W1 line?")
                        E.append(float(mobj.group(1)))
                        W0.append(float(mobj.group(2)))
                        W1.append(float(mobj.group(3)))
                    atom_stuff.append(GaussianShell.build_from_gbs(spherical,0,N,W0,E))
                    atom_stuff.append(GaussianShell.build_from_gbs(spherical,1,N,W1,E))
                    ind=ind+K
                else:
                    L = shell_data_[ID]
                    E  = []
                    W0 = []
                    for k in range(K):
                        mobj = re.match('^\s*(\S+)\s+(\S+)\s*$', re.sub(r'[Dd]', 'E', atom[ind + k]))
                        if mobj == None:
                            raise Exception("Where is the E W line?")
                        E.append(float(mobj.group(1)))
                        W0.append(float(mobj.group(2)))
                    atom_stuff.append(GaussianShell.build_from_gbs(spherical,L,N,W0,E))
                    ind=ind+K
                    
            atom_shells[key] = atom_stuff

        bas = BasisSet(name)
        shell_index = 0
        for A in range(mol.natom()):
            atom = mol.atoms[A]
            shells = atom_shells[atom.symbol]
            bas.atoms_to_shell_inds.append([])
            for shell in shells:
                gs = copy.deepcopy(shell)
                gs.x = atom.x
                gs.y = atom.y
                gs.z = atom.z
                bas.shells.append(gs)
                bas.atoms_to_shell_inds[A].append(shell_index)
                shell_index = shell_index + 1

        return bas

    def SBasisSet(self):
        SShells = SGaussianShellVecVec()
        for A in range(len(self.atoms_to_shell_inds)):
            SShells.append(SGaussianShellVec())
            for ind in self.atoms_to_shell_inds[A]:
                shell = self.shells[ind]
                c2s = DoubleVec()
                e2s = DoubleVec()
                for k in range(shell.nprimitive()): 
                    c2s.append(shell.cs[k])
                    e2s.append(shell.es[k])
                SShells[A].append(SGaussianShell(
                        shell.x, 
                        shell.y, 
                        shell.z, 
                        shell.is_spherical,
                        shell.am,
                        c2s,
                        e2s))
        return SBasisSet(self.name, SShells)

def build_dimension(vals):
    dim = Size_tVec()
    for val in vals:
        dim.append(val)
    return dim

def build_indices(vals):
    dim = StringVec()
    for val in vals:
        dim.append(val)
    return dim

def build_index_range(vals):
    
    ind = Size_tVecVec()
    for val in vals:
        iv = Size_tVec()
        for v in val:
            iv.append(v)
        ind.append(iv)
    return ind

tic = time.time()

# TODO
#Tensor.set_scratch_path('/scratch/parrish')

mol = Molecule.from_xyz_file('geom.xyz', False)
smol = mol.SMolecule()
smol.printf()

na = mol.nalpha()
nb = mol.nbeta()
if (na != nb):
    raise Exception('Cannot run RHF on this system')
nocc = int(na)

bas = BasisSet.from_gbs_file(mol, 'sto-3g.gbs')
sbas = bas.SBasisSet()
sbas.printf()

nbf = sbas.nfunction()

aux = BasisSet.from_gbs_file(mol, 'cc-pvtz-jkfit.gbs')
saux = aux.SBasisSet()
saux.printf()

minao = BasisSet.from_gbs_file(mol, 'cc-pvdz-minao.gbs')
sminao = minao.SBasisSet()
sminao.printf()

schwarz = SchwarzSieve(sbas,sbas,1.0E-12)
schwarz.printf()

ob = OneBody(schwarz)

d = []
d.append(sbas.nfunction())
d.append(sbas.nfunction())

print '  Overlap'
S = am.Tensor(am.TensorType.kCore, "S", d)
ob.compute_S(S.tensor)
#S.printf()

#print '  Dipole'
#X = TensorVec()
#X.append(Tensor.build(TensorType.kCore, "X", d))
#X.append(Tensor.build(TensorType.kCore, "Y", d))
#X.append(Tensor.build(TensorType.kCore, "Z", d))
#ob.compute_X(X)

print '  Kinetic'
T = am.Tensor.build(am.TensorType.kCore, "T", d)
ob.compute_T(T.tensor)
#T.printf()

print '  Potential'
V = am.Tensor.build(am.TensorType.kCore, "V", d)
ob.compute_V_nuclear(V.tensor, smol)
#V.printf()
    
print ''

H = am.Tensor.build(am.TensorType.kCore, "H", d)
H["i,j"] += T["i,j"]
H["i,j"] += V["i,j"]

F = am.Tensor(am.TensorType.kCore, "F", d)
Ft1 = am.Tensor(am.TensorType.kCore, "Ft1", d)
Ft2 = am.Tensor(am.TensorType.kCore, "Ft2", d)
C = am.Tensor(am.TensorType.kCore, "C", d)
D = am.Tensor(am.TensorType.kCore, "D", d)
Cocc = am.Tensor(am.TensorType.kCore, "Cocc", [nbf,nocc])
J = am.Tensor(am.TensorType.kCore, "D", d)
K = am.Tensor(am.TensorType.kCore, "K", d)
G = am.Tensor(am.TensorType.kCore, "G", d)
E = am.Tensor(am.TensorType.kCore, "E", [])

X = S.power(-1.0/2.0, 1.0E-12)

sad = SAD(smol,sbas,sminao)
sad.printf()
Csad = am.Tensor(existing=sad.compute_C())
D["pq"] = Csad["pi"] * Csad["qi"]

F["ij"] = H["ij"]
Ft1["iq"] = X["ip"] * H["pq"]
Ft2["ij"] = H["iq"] * X["qj"]
data = Ft2.tensor.syev(am.EigenvalueOrder.kAscending)
F2 = am.Tensor(existing=data['eigenvectors'])
C["pi"] = X["pj"] * F2["ij"]
Cocc.slice(C,build_index_range([[0,nbf],[0,nocc]]),build_index_range([[0,nbf],[0,nocc]]),1.0,0.0)
#Cocc["pq"] = C[:, :nocc] 
D["pq"] = Cocc["pi"] * Cocc["qi"]

#jk = DirectJK(schwarz)
jk = DFJK(schwarz,saux)
jk.set_doubles(int(1E9))
jk.printf()
jk.initialize()

#jk2.set_compute_J(False)
#jk.set_compute_K(False)

diis = DIIS(1,6,True)

print '  Nuclear Repulsion Energy %18.10f\n' % smol.nuclear_repulsion_energy()
#S.printf()
#X.printf()
#T.printf()
#V.printf()
#H.printf()
#C.printf()
#D.printf()

converged = False
Enuc = smol.nuclear_repulsion_energy();
#E.contract(D,H,build_indices([]),build_indices(["p","q"]),build_indices(["p","q"]),1.0,0.0)
#E.contract(D,F,build_indices([]),build_indices(["p","q"]),build_indices(["p","q"]),1.0,1.0)
#Eelec = E.data()[0]
Eold = 0.0;


for ind in range(0,50):

    J.zero()
    K.zero()

    Cvec = TensorVec()
    if (ind == 0):
        Cvec.append(Csad.tensor)
    else:
        Cvec.append(Cocc.tensor)
    Dvec = TensorVec()
    Dvec.append(D.tensor)
    Jvec = TensorVec()
    Jvec.append(J.tensor)
    Kvec = TensorVec()
    Kvec.append(K.tensor)

    TrueVec = BoolVec()
    TrueVec.append(True)

    jk.compute_JK_from_C(Cvec,Cvec,Jvec,Kvec)    

    F["ij"] = H["ij"]
    F["ij"] += J["ij"]
    F["ij"] += J["ij"]
    F["ij"] -= K["ij"]

    E[""] = D["pq"] * H["pq"]
    E[""] += D["pq"] * F["pq"]
    Eelec = E.data()[0]

    print "  @RHF iter %5d: %20.14lf" % (ind, Enuc + Eelec);

    if (abs(Eelec - Eold) < 1.0E-8):
        converged = True
        break
    Eold = Eelec

    # Orbital Gradient
    Ft1.contract(F,D,build_indices(["p","s"]),build_indices(["p","q"]),build_indices(["q","s"]),1.0,0.0)
    G.contract(Ft1,S,build_indices(["p","s"]),build_indices(["p","q"]),build_indices(["q","s"]),1.0,0.0)
    Ft1.contract(S,D,build_indices(["p","s"]),build_indices(["p","q"]),build_indices(["q","s"]),1.0,0.0)
    G.contract(Ft1,F,build_indices(["p","s"]),build_indices(["p","q"]),build_indices(["q","s"]),-1.0,1.0)
    Ft1.contract(X,G,build_indices(["p","s"]),build_indices(["p","q"]),build_indices(["q","s"]),1.0,0.0)
    G.contract(Ft1,X,build_indices(["p","s"]),build_indices(["p","q"]),build_indices(["q","s"]),1.0,0.0)

    #G2.slice(G,build_index_range([[0,nbf],[0,nbf]]),build_index_range([[0,nbf],[0,nbf]]),1.0,0.0)

    if (ind > 0):
        Fvec = TensorVec()
        Fvec.append(F.tensor)
        Gvec = TensorVec()
        Gvec.append(G.tensor)
        diis.add_iteration(Fvec,Gvec)
        diis.extrapolate(Fvec)

    Ft1["iq"] = X["ip"] * F["pq"]
    Ft2["ij"] = Ft1["iq"] * X["qj"]
    data = Ft2.tensor.syev(am.EigenvalueOrder.kAscending)
    F2 = am.Tensor(existing=data['eigenvectors'])
    C["pi"] = X["pj"] * F2["ij"]
    Cocc.slice(C,build_index_range([[0,nbf],[0,nocc]]),build_index_range([[0,nbf],[0,nocc]]),1.0,0.0)
    D["pq"] = Cocc["pi"] * Cocc["qi"]

print ''
jk.finalize()

toc = time.time()
print '  Elapsed Time: %11.3E [s]' % (toc - tic)
