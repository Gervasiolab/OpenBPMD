#!/usr/bin/env python
descriptn = \
    """
    OpenBPMD - an open source implementation of Binding Pose Metadynamics
    (BPMD) with OpenMM. Replicates the protocol as described by
    Clark et al. 2016 (DOI: 10.1021/acs.jctc.6b00201).

    Runs ten 10 ns metadynamics simulations that biases the RMSD of the ligand.

    The stability of the ligand is calculated using the ligand RMSD (PoseScore)
    and the persistence of the original noncovalent interactions between the
    protein and the ligand (ContactScore). Stable poses have a low RMSD and
    a high fraction of the native contacts preserved until the end of the
    simulation.

    A composite score is calculated using the following formula:
    CompScore = PoseScore - 5 * ContactScore
    """

# OpenMM
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
try:
    from simtk.openmm.app.metadynamics import *
except:
    from openmm.app.metadynamics import *

# The rest
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts
import mdtraj as md
import pandas as pd
import parmed as pmd
import glob
import os

__author__ = "Dominykas Lukauskis"
__version__ = "1.0.0"
__email__ = "dominykas.lukauskis.19@ucl.ac.uk"


def main(args):
    """Main entry point of the app. Takes in argparse.Namespace object as
    a function argument. Carries out a sequence of steps required to obtain a
    stability score for a given ligand pose in the provided structure file.

    1. Load the structure and parameter files.
    2. If absent, create an output folder.
    3. Minimization up to ener. tolerance of 10 kJ/mol.
    4. 500 ps equilibration in NVT ensemble with position
       restraints on solute heavy atoms with the force 
       constant of 5 kcal/mol/A^2
    5. Run NREPs (default=10) of binding pose metadynamics simulations,
       writing trajectory files and a time-resolved BPM scores for each
       repeat.
    6. Collect results from the OpenBPMD simulations and
       write a final score for a given protein-ligand
       structure.

    Parameters
    ----------
    args.structure : str, default='solvated.rst7'
        Name of the structure file, either Amber or Gromacs format.
    args.parameters : str, default='solvated.prm7'
        Name of the parameter or topology file, either Amber or Gromacs
        format.
    args.output : str, default='.'
        Path to and the name of the output directory.
    args.lig_resname : str, default='LIG'
        Residue name of the ligand in the structure/parameter file.
    args.nreps : int, default=10
        Number of repeat OpenBPMD simulations to run in series.
    args.hill_height : float, default=0.3
        Size of the metadynamical hill, in kcal/mol.
    """
    if args.structure.endswith('.gro'):
        coords = GromacsGroFile(args.structure)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = GromacsTopFile(args.parameters, periodicBoxVectors=box_vectors)
    else:
        coords = AmberInpcrdFile(args.structure)
        parm = AmberPrmtopFile(args.parameters)

    if not os.path.isdir(f'{args.output}'):
        os.mkdir(f'{args.output}')

    # Minimize
    min_file_name = 'minimized_system.pdb'
    if not os.path.isfile(os.path.join(args.output,min_file_name)):
        print("Minimizing...")
        #min_pos = minimize(parm, coords.positions, args.output)
        minimize(parm, coords.positions, args.output, min_file_name)
    min_pdb = os.path.join(args.output,min_file_name)

    # Equilibrate
    eq_file_name = 'equil_system.pdb'
    if not os.path.isfile(os.path.join(args.output,eq_file_name)):
        print("Equilibrating...")
        equilibrate(min_pdb, parm, args.output, eq_file_name)
    eq_pdb = os.path.join(args.output,eq_file_name)
    cent_eq_pdb = os.path.join(args.output,'centred_'+eq_file_name)
    if os.path.isfile(eq_pdb) and not os.path.isfile(cent_eq_pdb):
	# mdtraj can't use GMX TOP, so we have to specify the GRO file instead
        if args.structure.endswith('.gro'):
            mdtraj_top = args.structure
        else:
            mdtraj_top = args.parameters
        mdu = md.load(eq_pdb, top=mdtraj_top)
        mdu.image_molecules()
        mdu.save_pdb(cent_eq_pdb)

    # Run NREPS number of production simulations
    for idx in range(0, args.nreps):
        rep_dir = os.path.join(args.output,f'rep_{idx}')
        if not os.path.isdir(rep_dir):
            os.mkdir(rep_dir)

        if os.path.isfile(os.path.join(rep_dir,'bpm_results.csv')):
            continue
        
        produce(args.output, idx, args.lig_resname, eq_pdb, parm, args.parameters,
                args.structure, args.hill_height)
                
        trj_name = os.path.join(rep_dir,'trj.dcd')
                
        PoseScoreArr = get_pose_score(cent_eq_pdb, trj_name, args.lig_resname)

        ContactScoreArr = get_contact_score(cent_eq_pdb, trj_name, args.lig_resname)

        # Calculate the CompScore at every frame
        CompScoreArr = np.zeros(99)
        for index in range(ContactScoreArr.shape[0]):
            ContactScore, PoseScore = ContactScoreArr[index], PoseScoreArr[index]
            CompScore = PoseScore - 5 * ContactScore
            CompScoreArr[index] = CompScore

        Scores = np.stack((CompScoreArr, PoseScoreArr, ContactScoreArr), axis=-1)

        # Save a DataFrame to CSV
        df = pd.DataFrame(Scores, columns=['CompScore', 'PoseScore',
                                           'ContactScore'])
        df.to_csv(os.path.join(rep_dir,'bpm_results.csv'), index=False)
                
    collect_results(args.output, args.output)

    return None
    

def get_contact_score(structure_file, trajectory_file, lig_resname):
    """A function the gets the ContactScore from an OpenBPMD trajectory.

    Parameters
    ----------
    structure_file : str
        The name of the centred equilibrated system PDB file that 
        was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Residue name of the ligand that was biased.

    Returns
    -------
    contact_scores : np.array 
        ContactScore for every frame of the trajectory.
    """
    u = mda.Universe(structure_file, trajectory_file)

    sel_donor = f"resname {lig_resname} and not name *H*"
    sel_acceptor = f"protein and not name H* and \
                     around 5 resname {lig_resname}"

    # reference groups (first frame of the trajectory, but you could also use
    # a separate PDB, eg crystal structure)
    a_donors = u.select_atoms(sel_donor)
    a_acceptors = u.select_atoms(sel_acceptor)

    cont_analysis = contacts.Contacts(u, select=(sel_donor, sel_acceptor),
                                      refgroup=(a_donors, a_acceptors),
                                      radius=3.5)

    cont_analysis.run()
    # print number of average contacts in the first ns
    # NOTE - hard coded number of frames (100 per traj)
    frame_idx_first_ns = int(len(cont_analysis.timeseries)/10)
    first_ns_mean = np.mean(cont_analysis.timeseries[1:frame_idx_first_ns, 1])
    if first_ns_mean == 0:
        normed_contacts = cont_analysis.timeseries[1:, 1]
    else:
        normed_contacts = cont_analysis.timeseries[1:, 1]/first_ns_mean
    contact_scores = np.where(normed_contacts > 1, 1, normed_contacts)

    return contact_scores


def get_pose_score(structure_file, trajectory_file, lig_resname):
    """A function the gets the PoseScore (ligand RMSD) from an OpenBPMD
    trajectory.

    Parameters
    ----------
    'structure_file : str
        The name of the centred equilibrated system
        PDB file that was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Residue name of the ligand that was biased.

    Returns
    -------
    pose_scores : np.array 
        PoseScore for every frame of the trajectory.
    """
    # Load a MDA universe with the trajectory
    u = mda.Universe(structure_file, trajectory_file)
    # Align each frame using the backbone as reference
    # Calculate the RMSD of ligand heavy atoms
    r = rms.RMSD(u, select='backbone',
                 groupselections=[f'resname {lig_resname} and not name H*'],
                 ref_frame=0).run()
    # Get the PoseScores as np.array
    pose_scores = r.rmsd[1:, -1]

    return pose_scores


def minimize(parm, input_positions, out_dir, min_file_name):
    """An energy minimization function down with an energy tolerance
    of 10 kJ/mol.

    Parameters
    ----------
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    input_positions : OpenMM Quantity
        3D coordinates of the equilibrated system.
    out_dir : str
        Directory to write the outputs.
    min_file_name : str
        Name of the minimized PDB file to write.
    """
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
    )

    # Define platform properties
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}

    # Set up the simulation parameters
    # Langevin integrator at 300 K w/ 1 ps^-1 friction coefficient
    # and a 2-fs timestep
    # NOTE - no dynamics performed, but required for setting up
    # the OpenMM system.
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond,
                                    0.002*picoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform,
                            properties)
    simulation.context.setPositions(input_positions)

    # Minimize the system - no predefined number of steps
    simulation.minimizeEnergy()

    # Write out the minimized system to use w/ MDAnalysis
    positions = simulation.context.getState(getPositions=True).getPositions()
    out_file = os.path.join(out_dir,min_file_name)
    PDBFile.writeFile(simulation.topology, positions,
                      open(out_file, 'w'))

    return None


def equilibrate(min_pdb, parm, out_dir, eq_file_name):
    """A function that does a 500 ps NVT equilibration with position
    restraints, with a 5 kcal/mol/A**2 harmonic constant on solute heavy
    atoms, using a 2 fs timestep.

    Parameters
    ----------
    min_pdb : str
        Name of the minimized PDB file.
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    out_dir : str
        Directory to write the outputs to.
    eq_file_name : str
        Name of the equilibrated PDB file to write.
    """
    # Get the solute heavy atom indices to use
    # for defining position restraints during equilibration
    universe = mda.Universe(min_pdb,
                            format='XPDB', in_memory=True)
    solute_heavy_atom_idx = universe.select_atoms('not resname WAT and\
                                                   not resname SOL and\
                                                   not resname HOH and\
                                                   not resname CL and \
                                                   not resname NA and \
                                                   not name H*').indices
    # Necessary conversion to int from numpy.int64,
    # b/c it breaks OpenMM C++ function
    solute_heavy_atom_idx = [int(idx) for idx in solute_heavy_atom_idx]

    # Add the restraints.
    # We add a dummy atoms with no mass, which are therefore unaffected by
    # any kind of scaling done by barostat (if used). And the atoms are
    # harmonically restrained to the dummy atom. We have to redefine the
    # system, b/c we're adding new particles and this would clash with
    # modeller.topology.
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
    )
    # Add the harmonic restraints on the positions
    # of specified atoms
    restraint = HarmonicBondForce()
    restraint.setUsesPeriodicBoundaryConditions(True)
    system.addForce(restraint)
    nonbonded = [force for force in system.getForces()
                 if isinstance(force, NonbondedForce)][0]
    dummyIndex = []
    input_positions = PDBFile(min_pdb).getPositions()
    positions = input_positions
    # Go through the indices of all atoms that will be restrained
    for i in solute_heavy_atom_idx:
        j = system.addParticle(0)
        # ... and add a dummy/ghost atom next to it
        nonbonded.addParticle(0, 1, 0)
        # ... that won't interact with the restrained atom 
        nonbonded.addException(i, j, 0, 1, 0)
        # ... but will be have a harmonic restraint ('bond')
        # between the two atoms
        restraint.addBond(i, j, 0*nanometers,
                          5*kilocalories_per_mole/angstrom**2)
        dummyIndex.append(j)
        input_positions.append(positions[i])

    integrator = LangevinIntegrator(300*kelvin, 1/picosecond,
                                    0.002*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}
    sim = Simulation(parm.topology, system, integrator,
                     platform, properties)
    sim.context.setPositions(input_positions)
    integrator.step(250000)  # run 500 ps of equilibration
    all_positions = sim.context.getState(
        getPositions=True, enforcePeriodicBox=True).getPositions()
    # we don't want to write the dummy atoms, so we only
    # write the positions of atoms up to the first dummy atom index
    relevant_positions = all_positions[:dummyIndex[0]]
    out_file = os.path.join(out_dir,eq_file_name)
    PDBFile.writeFile(sim.topology, relevant_positions,
                      open(out_file, 'w'))

    return None


def produce(out_dir, idx, lig_resname, eq_pdb, parm, parm_file,
            coords_file, set_hill_height):
    """An OpenBPMD production simulation function. Ligand RMSD is biased with
    metadynamics. The integrator uses a 4 fs time step and
    runs for 10 ns, writing a frame every 100 ps.

    Writes a 'trj.dcd', 'COLVAR.npy', 'bias_*.npy' and 'sim_log.csv' files
    during the metadynamics simulation in the '{out_dir}/rep_{idx}' directory.
    After the simulation is done, it analyses the trajectories and writes a
    'bpm_results.csv' file with time-resolved PoseScore and ContactScore.

    Parameters
    ----------
    out_dir : str
        Directory where your equilibration PDBs and 'rep_*' dirs are at.
    idx : int
        Current replica index.
    lig_resname : str
        Residue name of the ligand.
    eq_pdb : str
        Name of the PDB for equilibrated system.
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    parm_file : str
        The name of the parameter or topology file of the system.
    coords_file : str
        The name of the coordinate file of the system.
    set_hill_height : float
        Metadynamic hill height, in kcal/mol.

    """
    # First, assign the replica directory to which we'll write the files
    write_dir = os.path.join(out_dir,f'rep_{idx}')
    # Get the anchor atoms by ...
    universe = mda.Universe(eq_pdb,
                            format='XPDB', in_memory=True)
    # ... finding the protein's COM ...
    prot_com = universe.select_atoms('protein').center_of_mass()
    x, y, z = prot_com[0], prot_com[1], prot_com[2]
    # ... and taking the heavy backbone atoms within 10A of the COM
    sel_str = f'point {x} {y} {z} 10 and backbone and not name H*'
    anchor_atoms = universe.select_atoms(sel_str)
    if len(anchor_atoms) == 0:
        raise ValueError('No Calpha atoms found within 10 ang of the center of mass of the protein. \
                Check your input files.')
    anchor_atom_idx = anchor_atoms.indices.tolist()

    # Get indices of ligand heavy atoms
    lig = universe.select_atoms(f'resname {lig_resname} and not name H*')
    if len(lig) == 0:
        raise ValueError(f"Ligand with resname '{lig_resname}' not found.")

    lig_ha_idx = lig.indices.tolist()

    # Set up the system to run metadynamics
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
        hydrogenMass=4*amu
    )
    # get the atom positions for the system from the equilibrated
    # system
    input_positions = PDBFile(eq_pdb).getPositions()

    # Add an 'empty' flat-bottom restraint to fix the issue with PBC.
    # Without one, RMSDForce object fails to account for PBC.
    k = 0*kilojoules_per_mole  # NOTE - 0 kJ/mol constant
    upper_wall = 10.00*nanometer
    fb_eq = '(k/2)*max(distance(g1,g2) - upper_wall, 0)^2'
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(lig_ha_idx)
    upper_wall_rest.addGroup(anchor_atom_idx)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter('k', k)
    upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    system.addForce(upper_wall_rest)

    alignment_indices = lig_ha_idx + anchor_atom_idx

    rmsd = RMSDForce(input_positions, alignment_indices)
    # Set up the typical metadynamics parameters
    grid_min, grid_max = 0.0, 1.0  # nm
    hill_height = set_hill_height*kilocalories_per_mole
    hill_width = 0.002  # nm, also known as sigma

    grid_width = hill_width / 5
    # 'grid' here refers to the number of grid points
    grid = int(abs(grid_min - grid_max) / grid_width)

    rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width,
                           False, gridWidth=grid)

    # define the metadynamics object
    # deposit bias every 1 ps, BF = 4, write bias every ns
    meta = Metadynamics(system, [rmsd_cv], 300.0*kelvin, 4.0, hill_height,
                        250, biasDir=write_dir,
                        saveFrequency=250000)

    # Set up and run metadynamics
    integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond,
                                    0.004*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}

    simulation = Simulation(parm.topology, system, integrator, platform,
                            properties)
    simulation.context.setPositions(input_positions)

    trj_name = os.path.join(write_dir,'trj.dcd')

    sim_time = 10  # ns
    steps = 250000 * sim_time

    simulation.reporters.append(DCDReporter(trj_name, 25000))  # every 100 ps
    simulation.reporters.append(StateDataReporter(
                                os.path.join(write_dir,'sim_log.csv'), 250000,
                                step=True, temperature=True, progress=True,
                                remainingTime=True, speed=True,
                                totalSteps=steps, separator=','))  # every 1 ns

    colvar_array = np.array([meta.getCollectiveVariables(simulation)])
    for i in range(0, int(steps), 500):
        if i % 25000 == 0:
            # log the stored COLVAR every 100ps
            np.save(os.path.join(write_dir,'COLVAR.npy'), colvar_array)
        meta.step(simulation, 500)
        current_cvs = meta.getCollectiveVariables(simulation)
        # record the CVs every 2 ps
        colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    np.save(os.path.join(write_dir,'COLVAR.npy'), colvar_array)

    # center everything using MDTraj, to fix any PBC imaging issues
    # mdtraj can't use GMX TOP, so we have to specify the GRO file instead
    if coords_file.endswith('.gro'):
        mdtraj_top = coords_file
    else:
        mdtraj_top = parm_file
    mdu = md.load(trj_name, top=mdtraj_top)
    mdu.image_molecules()
    mdu.save(trj_name)

    return None


def collect_results(in_dir, out_dir):
    """A function that collects the time-resolved BPM results,
    takes the scores from last 2 ns of the simulation, averages them
    and writes that average as the final score for a given pose.

    Writes a 'results.csv' file in 'out_dir' directory.
    
    Parameters
    ----------
    in_dir : str
        Directory with 'rep_*' directories.
    out_dir : str
        Directory where the 'results.csv' file will be written
    """
    compList = []
    contactList = []
    poseList = []
    # find how many repeats have been run
    glob_str = os.path.join(in_dir,'rep_*')
    nreps = len(glob.glob(glob_str))
    for idx in range(0, nreps):
        f = os.path.join(in_dir,f'rep_{idx}','bpm_results.csv')
        df = pd.read_csv(f)
        # Since we only want last 2 ns, get the index of
        # the last 20% of the data points
        last_2ns_idx = round(len(df['CompScore'].values)/5)  # round up
        compList.append(df['CompScore'].values[-last_2ns_idx:])
        contactList.append(df['ContactScore'].values[-last_2ns_idx:])
        poseList.append(df['PoseScore'].values[-last_2ns_idx:])

    # Get the means of the last 2 ns
    meanCompScore = np.mean(compList)
    meanPoseScore = np.mean(poseList)
    meanContact = np.mean(contactList)
    # Get the standard deviation of the final 2 ns
    meanCompScore_std = np.std(compList)
    meanPoseScore_std = np.std(poseList)
    meanContact_std = np.std(contactList)
    # Format it the Pandas way
    d = {'CompScore': [meanCompScore], 'CompScoreSD': [meanCompScore_std],
         'PoseScore': [meanPoseScore], 'PoseScoreSD': [meanPoseScore_std],
         'ContactScore': [meanContact], 'ContactScoreSD': [meanContact_std]}

    results_df = pd.DataFrame(data=d)
    results_df = results_df.round(3)
    results_df.to_csv(os.path.join(out_dir,'results.csv'), index=False)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    # Parse the CLI arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=descriptn)

    parser.add_argument("-s", "--structure", type=str, default='solvated.rst7',
                        help='input structure file name (default: %(default)s)')
    parser.add_argument("-p", "--parameters", type=str, default='solvated.prm7',
                        help='input topology file name (default: %(default)s)')
    parser.add_argument("-o", "--output", type=str, default='.',
                        help='output location (default: %(default)s)')
    parser.add_argument("-lig_resname", type=str, default='MOL',
                        help='the name of the ligand (default: %(default)s)')
    parser.add_argument("-nreps", type=int, default=10,
                        help="number of OpenBPMD repeats (default: %(default)i)")
    parser.add_argument("-hill_height", type=float, default=0.3,
                        help="the hill height in kcal/mol (default: %(default)f)")

    args = parser.parse_args()
    main(args)
