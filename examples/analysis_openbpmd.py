#!/usr/bin/env python
# coding: utf-8
descriptn = \
    """
    Analyses OpenBPMD simulations, makes time-resolved plots of the three 
    scores and calculates the final score for the ligand pose given.

    Writes its outputs to '{dir}_results'.
    """

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# args parse bit
# Parse the CLI arguments
parser = argparse.ArgumentParser(
       formatter_class=argparse.RawDescriptionHelpFormatter,
       description=descriptn)

parser.add_argument("-i", "--dir", type=str,
                    help='directory where OpenBPMD simulations were run')
parser.add_argument("--v", action='store_true', default=False,
                    help='Be verbose (default: %(default)s)')

args = parser.parse_args()

input_dir = args.dir
res_dir = input_dir + '_results'  # save results here
if args.v:
    print(f"Reading '{input_dir}', writting to '{res_dir}/'")
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

# read all the reps from 'input_dir'
glob_str = os.path.join(input_dir,'rep_*')
nreps = len(glob.glob(glob_str))

# We'll store the results from 10 repeats in a 10x99 matrix.
CompScores = np.zeros((nreps,99))
PoseScores = np.zeros((nreps,99))
ContactScores = np.zeros((nreps,99))

# Fill those matrices with the scores from each repeat
for idx in range(0,nreps):
    f = os.path.join(input_dir,f'rep_{idx}','bpm_results.csv')
    df = pd.read_csv(f)
    CompScores[idx] = df['CompScore']
    PoseScores[idx] = df['PoseScore']
    ContactScores[idx] = df['ContactScore']

# Average out the scores from all of the repeats,
# giving a mean of the scores at each frame of the trajectory
averagedCompScore = np.array([ np.mean(CompScores[:,i]) for i in range(0,99) ])
averagedPoseScore = [ np.mean(PoseScores[:,i]) for i in range(0,99) ]
averagedContactScore = [ np.mean(ContactScores[:,i]) for i in range(0,99) ]
# Get the standard deviation for the CompScore
CompScore_stddev = np.array([ np.std(CompScores[:,i]) for i in range(0,99) ])
# An array of time steps for plotting the x axis
time_sequence = np.linspace(0,nreps,99)

if args.v:
    print('Making time-resolved score plots')
plt.title(f"CompScore, {nreps} repeats, from '{input_dir}'")
plt.plot(time_sequence,averagedCompScore, color='blue')
# Visualise the standard deviation of CompScore at each frame
plt.fill_between(time_sequence, averagedCompScore-CompScore_stddev, 
                 averagedCompScore+CompScore_stddev, 
                 color='blue', alpha=0.3, lw=0)
plt.xlabel('time(ns)')
plt.ylabel('CompScore')
plt.ylim(-5,5)
plt.savefig(f'{res_dir}/CompScore.png',facecolor='w')
plt.close()

plt.title(f"PoseScore, {nreps} repeats, from '{input_dir}'")
plt.plot(time_sequence,averagedPoseScore,color='darkorange')
plt.xlabel('time(ns)')
plt.ylabel('PoseScore')
plt.ylim(0,5)
plt.savefig(f'{res_dir}/PoseScore.png',facecolor='w')
plt.close()

plt.title(f"ContactScore, {nreps} repeats, from '{input_dir}'")
plt.plot(time_sequence,averagedContactScore,color='green')
plt.xlabel('time(ns)')
plt.ylabel('ContactScore')
plt.ylim(-0.1,1.1)
plt.savefig(f'{res_dir}/ContactScore.png',facecolor='w')
plt.close()

if args.v:
    print('Collating the results from all repeats into a final score')
# In order to get a single number that evaluates a given pose/ligand, we take the scores of the final 2 ns.
compList = []
contactList = []
poseList = []
# Find how many repeats have been run
for idx in range(0, nreps):
    f = os.path.join(input_dir,f'rep_{idx}','bpm_results.csv')
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

results_df.to_csv(os.path.join(res_dir,'results.csv'), index=False)
if args.v:
    print('Done')

