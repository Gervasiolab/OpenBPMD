descriptn=\
'''OpenBPMD - An open source implementation of Binding Pose Metadynamics (BPMD)
   with OpenMM. Replicates the protocol as described by Clark et al. 2016 
   (DOI: 10.1021/acs.jctc.6b00201).

   Runs a ten 10 ns metadynamics simulations that biase the RMSD of the ligand.

   The stability of the ligand is calculated using the ligand RMSD (PoseScore) 
   and the persistence of the original hydrogen bonds between the protein and 
   the ligand (ContactScore). Stable poses have low RMSD and high fraction of 
   the h-bonds seen in the beginning of the simulation.

   A composite score is calculated using the following formula:
   CompScore = PoseScore - 5 * ContactScore
'''
# OpenMM
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from simtk.openmm.app.metadynamics import *

# The rest
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms, contacts
import mdtraj as md
import pandas as pd
import parmed as pmd 
import os

# Parse the CLI arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=descriptn)

parser.add_argument("-s", type=str, default='solvated.rst7', help='input structure file name (default: %(default)s)')
parser.add_argument("-p", type=str, default='solvated.prm7', help='input topology file name (default: %(default)s)')
parser.add_argument("-lig_resname", type=str, default='MOL', help='the name of the ligand (default: %(default)s)')
parser.add_argument("-nreps", type=int, default=10, help="number of OpenBPMD repeats (default: %(default)i)")
parser.add_argument("-hill_height", type=float, default=0.3, help="the hill height in kcal/mol (default: %(default)i)")

args = parser.parse_args()

coords_file = args.s
parm_file = args.p
lig_resname = args.lig_resname
nreps = args.nreps
set_hill_height = args.hill_height


def get_contact_score(structure_file, trajectory_file, lig_resname = 'MOL'):
    u = mda.Universe(structure_file, trajectory_file)

    sel_donor = f"resname {lig_resname} and not name *H*"
    sel_acceptor = f"protein and not name H* and around 5 resname {lig_resname}"

    # reference groups (first frame of the trajectory, but you could also use a
    # separate PDB, eg crystal structure)
    h_donors = u.select_atoms(sel_donor)
    h_acceptors = u.select_atoms(sel_acceptor)

    con_analysis = contacts.Contacts(u, select=(sel_donor, sel_acceptor),
                            refgroup=(h_donors, h_acceptors), radius=3.5)

    con_analysis.run()
    # print number of averave contacts
    first_ns_mean = np.mean(con_analysis.timeseries[1:10, 1])
    if first_ns_mean == 0:
        normed_contacts = con_analysis.timeseries[1:, 1]
    else:
        normed_contacts = con_analysis.timeseries[1:, 1]/first_ns_mean

    return normed_contacts
    
def minimize(parm, input_positions):
    
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
        # hydrogenMass=4*amu
    )

    # Define platform properties
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}

    # Set up the simulation parameters
    # Langevin integrator at 300 K w/ 1 ps^-1 friction coefficient and a 2-fs timestep
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform)
    simulation.context.setPositions(input_positions)

    # Minimize the system - no predefined number of steps
    simulation.minimizeEnergy()

    # Write out the minimized system to use w/ MDAnalysis
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open('minimized_system.pdb', 'w'))
    
    return positions

def equilibrate(parm, input_positions):

    """
    500 ps NVT equilibration with position restraints 
    on solute heavy atoms using a 2 fs timestep
    """
    # Get the solute heavy atom indices to use for defining position restraints during equilibration
    universe = mda.Universe('minimized_system.pdb', format='XPDB', in_memory=True)
    solute_heavy_atom_idx = universe.select_atoms('not resname WAT and\
                                                   not resname SOL and\
                                                   not resname HOH and\
                                                   not resname CL and \
                                                   not resname NA and \
                                                   not name H*').indices
    # Necessary conversion to int from numpy.int64, b/c it breaks OpenMM C++ function
    solute_heavy_atom_idx = [int(idx) for idx in solute_heavy_atom_idx]

    # Add the restraints.
    # While this step is typically simple in most other programs, it needs more effort in OpenMM.
    # Code taken from: https://github.com/openmm/openmm/issues/2262#issuecomment-464157489
    # It adds a dummy atom with no mass, which is therefore unaffected by any kind of scaling done by barostat (if used).
    # And the atoms are harmonically restrained to the dummy atom.
    # I have to redefine the system, b/c I'm adding new particles and this would clash with modeller.topology.
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
        # hydrogenMass=4*amu
    )
    restraint = HarmonicBondForce()
    restraint.setUsesPeriodicBoundaryConditions(True)
    system.addForce(restraint)
    nonbonded = [force for force in system.getForces() if isinstance(force, NonbondedForce)][0]
    dummyIndex = []
    positions = input_positions
    for i in solute_heavy_atom_idx:
        j = system.addParticle(0)
        nonbonded.addParticle(0, 1, 0)
        nonbonded.addException(i, j, 0, 1, 0)
        restraint.addBond(i, j, 0*nanometers, 5*kilocalories_per_mole/angstrom**2)
        dummyIndex.append(j)
        input_positions.append(positions[i])

    # Equilibration (according to Schrödinger, short NVT?)
    # For now, I'll perform a 500-ps NVT equilibration with position restraints on solute heavy atoms.
    # If needed, another eq. step in NPT ensemble can easily be added.
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision':'mixed'}
    simulation = Simulation(parm.topology, system, integrator, platform, properties)
    simulation.context.setPositions(input_positions)
    integrator.step(250000)  # run 500 ps of equilibration, scientific notation not accepted
    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()[:dummyIndex[0]]
    PDBFile.writeFile(simulation.topology, positions[:dummyIndex[0]], open('equilibrated_system.pdb', 'w'))
    
    return positions

#########
# Begin #
#########

if '.gro' in coords_file:
    coords = pmd.load_file(coords_file)
    parm = pmd.load_file(parm_file)
    parm.box = coords.box[:]
else:
    coords = AmberInpcrdFile(coords_file)
    parm = AmberPrmtopFile(parm_file)

if not os.path.isfile('minimized_system.pdb'):
    print("Minimizing...")
    min_pos = minimize(parm, coords.positions)
else:
    min_pos = PDBFile('minimized_system.pdb').getPositions()

if not os.path.isfile('equilibrated_system.pdb'):
    print("Equilibrating...")
    input_pos = equilibrate(parm, min_pos)
    mdu = md.load('equilibrated_system.pdb',top=parm_file)
    mdu.image_molecules()
    mdu.save_pdb('centred_equilibrated_system.pdb')
else:
    input_pos = PDBFile('equilibrated_system.pdb').getPositions()

print('Starting production metadynamics simulations')

for idx in range(0, nreps):
    if not os.path.isdir(f'rep_{idx}'):
        os.mkdir(f'rep_{idx}')
        
    if os.path.isfile(f'rep_{idx}/bpm_results.csv'):
        continue
    else:
        os.chdir(f'rep_{idx}')
        
    # and now the metadynamics production run
    # prepare system and integrator

    # Get the anchor atoms by ...
    universe = mda.Universe('../equilibrated_system.pdb', format='XPDB', in_memory=True)
    # ... finding the protein's COM ...
    prot_com = universe.select_atoms('protein').center_of_mass()
    x,y,z = prot_com[0], prot_com[1], prot_com[2]
    # ... and taking the heavy backbone atoms within 5A of the COM
    anchor_atoms = universe.select_atoms(f'point {x} {y} {z} 5 and backbone and not name H*')
    # ... or 10 angstrom
    if len(anchor_atoms) == 0:
        anchor_atoms = universe.select_atoms(f'point {x} {y} {z} 10 and backbone and not name H*')

    anchor_atom_idx = anchor_atoms.indices.tolist()

    # Get indices of ligand heavy atoms
    lig = universe.select_atoms(f'resname {lig_resname} and not name H*')

    lig_ha_idx = lig.indices.tolist()

    # Set up the system to run metadynamics
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
        hydrogenMass=4*amu
    )
    
    # add a flat-bottom restraint to fix the issue with PBC
    k = 0*kilojoules_per_mole
    upper_wall = 10.00*nanometer

    upper_wall_rest = CustomCentroidBondForce(2, '(k/2)*max(distance(g1,g2) - upper_wall, 0)^2')
    upper_wall_rest.addGroup(lig_ha_idx)
    upper_wall_rest.addGroup(anchor_atom_idx)
    upper_wall_rest.addBond([0,1])
    upper_wall_rest.addGlobalParameter('k', k)
    upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    # dont forget to actually add the force to the system
    system.addForce(upper_wall_rest)

    alignment_indices = lig_ha_idx + anchor_atom_idx

    rmsd = RMSDForce(input_pos, alignment_indices)

    grid_min, grid_max = 0.0, 1.0 # nm
    hill_height = set_hill_height*kilocalories_per_mole
    hill_width = 0.002 # nm

    grid_width = hill_width / 5
    grid = int(abs(grid_min - grid_max) / grid_width)

    rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width, False, gridWidth = grid)

    # define the metadynamics object 
    # deposit bias every 1 ps, BF = 4
    meta = Metadynamics(system, [rmsd_cv], 300.0*kelvin, 4.0, hill_height, 250, 
                        biasDir = '.', saveFrequency = 250000) # write bias every ns

    # Set up and run metadynamics
    integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.004*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision':'mixed'}

    simulation = Simulation(parm.topology, system, integrator, platform, properties)
    simulation.context.setPositions(input_pos)

    trj_name = 'trj.dcd'    
        
    sim_time = 10 # ns
    steps = 250000 * sim_time

    simulation.reporters.append(DCDReporter(trj_name, 25000)) # every 100 ps
    simulation.reporters.append(StateDataReporter(
                               'sim_log.csv', 250000, step=True,
                               temperature=True,progress=True,
                               remainingTime=True,speed=True,
                               totalSteps=steps,separator=',')) # every 1 ns

    colvar_array = np.array([meta.getCollectiveVariables(simulation)])
    for i in range(0,int(steps),500):
        if i%25000 == 0:
            # log the stored COLVAR every 100ps
            np.save('COLVAR.npy', colvar_array)
        meta.step(simulation, 500)
        current_cvs = meta.getCollectiveVariables(simulation)
        # record the CVs every 2 ps
        colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    np.save('COLVAR.npy', colvar_array)
    
    del simulation, platform, properties, system, meta, integrator 
   
    # center everything, to fix any PBC imaging issues
    mdu = md.load('trj.dcd',top='../solvated.prm7')
    mdu.image_molecules()
    mdu.save('trj.dcd')

    # do the analysis of the sim
    # lets use the trj to get the RMSD of the ligand
    mobile = mda.Universe('../centred_equilibrated_system.pdb', 'trj.dcd')

    r = rms.RMSD(mobile, select = 'backbone',
                 groupselections=[f'resname {lig_resname} and not name H*'], 
                 ref_frame=0).run()
    
    PoseScoreArr = r.rmsd[1:,-1]
            
    ContactScoreArr = get_contact_score('../centred_equilibrated_system.pdb', 
                                        'trj.dcd', lig_resname)
    ContactScoreArr = np.where(ContactScoreArr > 1, 1, ContactScoreArr)
    # also calculate the persistence score
    t = md.load_dcd(f'trj.dcd', top = f'../centred_equilibrated_system.pdb')

    # Calculate the CompScore at every frame
    CompScoreArr = np.zeros(99)
    for index in range(ContactScoreArr.shape[0]):
        ContactScore, PoseScore = ContactScoreArr[index], PoseScoreArr[index]
        CompScore = PoseScore - 5 * ContactScore
        CompScoreArr[index] = CompScore

    Scores = np.stack((CompScoreArr, PoseScoreArr, ContactScoreArr), axis=-1)

    # Save a DataFrame to CSV
    df = pd.DataFrame(Scores,columns=['CompScore','PoseScore','ContactScore'])
    df.to_csv(f'bpm_results.csv',index=False) 

    os.chdir('../')

compList = []
contactList = []
poseList = []
for i in range(0,nreps):
    f = f'rep_{i}/bpm_results.csv'
    df = pd.read_csv(f)
    compList.append(df['CompScore'].values[-20:])
    contactList.append(df['ContactScore'].values[-20:])
    poseList.append(df['PoseScore'].values[-20:])

meanCompScore = np.mean(compList)
meanPoseScore = np.mean(poseList)
meanContact = np.mean(contactList)

meanCompScore_std = np.std(compList)
meanPoseScore_std = np.std(poseList)
meanContact_std = np.std(contactList)

d = {'CompScore':[meanCompScore],'CompScoreSD':[meanCompScore_std],
     'PoseScore':[meanPoseScore],'PoseScoreSD':[meanPoseScore_std],
     'ContactScore':[meanContact],'ContactScoreSD':[meanContact_std]}

results_df = pd.DataFrame(data=d)
results_df = results_df.round(3)
results_df.to_csv('results.csv',index=False)
