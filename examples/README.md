# Example OpenBPMD system and analysis.

This directory contains an example system, ```solvated.pdb```, ```solvated.rst7```, ```solvated.prm7```. This is a complex of a CDK2 protein (PDB 1PXJ) structure and a docking-generated ligand (PDB 1JVP) pose, set in a triclinic box, solvated with TIP3P waters and counter-ions.

If you type ```python ../openbpmd.py -h```, you should see a description that follows the script and some of the following arguments:

```
  -h, --help            show this help message and exit
  -s S                  input structure file name (default: solvated.rst7)
  -p P                  input topology file name (default: solvated.prm7)
  -o O                  output location (default: .)
  -lig_resname LIG_RESNAME
                        the name of the ligand (default: MOL)
  -nreps NREPS          number of OpenBPMD repeats (default: 10)
  -hill_height HILL_HEIGHT
                        the hill height in kcal/mol (default: 0.300000)
```

As an example, if we want to run a BPMD simulation using gromacs format files, for 5 repeats, where the ligand's residue name is 'LIG', writing the files into a directory called 'ligand0_pose0', the CLI command will look like this:

```python openbpmd.py -s gmx.gro -p gmx.top -o ligand0_pose0 -lig_resname LIG -nreps 5```

## Analysis

In order to preserve space, the example ```output``` directory used for analysis will have most of the typically generated files missing. 

Have a look at the ```example_analysis.ipynb``` notebook or the ```example_analysis.py``` script for how to interpret OpenBPMD results.

For running the analysis as more of a blackbox, use ```python analysis_openbpmd.py -i output``` and have a look at the plots it creates.
