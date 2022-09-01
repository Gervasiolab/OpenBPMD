#!/usr/bin/env python
# coding: utf-8

# # Analysing BPMD simulations
# 
# In this notebook, we'll walk through a simple example of how to open and plot the time-wise BPMD scores, take their averages over multiple replicas and determining the final stability score for the given pose.

# First, we import a few modules.

# In[7]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Let's start by looking at one replica.

# In[2]:


f = os.path.join('output','rep_0','bpm_results.csv')
df = pd.read_csv(f)
print('\nTime-resolved results from one OpenBPMD simulation\n')
print(df)


# As you can see the different scores were calculated for each frame. Let's plot each of the scores for this replica.

# In[3]:


time_sequence = np.linspace(0,10,99)

plt.title('CompScore from 1 repeat')
plt.plot(time_sequence,df['CompScore'])
plt.xlabel('time(ns)')
plt.ylabel('CompScore')
plt.ylim(-5,5)
plt.savefig('single_rep_CompScore.png',facecolor='w')
plt.close()

plt.title('PoseScore from 1 repeat')
plt.plot(time_sequence,df['PoseScore'],color='darkorange')
plt.xlabel('time(ns)')
plt.ylabel('PoseScore')
plt.ylim(0,5)
plt.savefig('single_rep_PoseScore.png',facecolor='w')
plt.close()

plt.title('ContactScore from 1 repeat')
plt.plot(time_sequence,df['ContactScore'],color='green')
plt.xlabel('time(ns)')
plt.ylabel('ContactScore')
plt.ylim(-0.1,1.1)
plt.savefig('single_rep_ContactScore.png',facecolor='w')
plt.close()


# As you can see, the scores are very noisy. This is typical of short metadynamics simulations that haven't yet converged the free energy surface. However, we're not interested in the free energy. We're only want to test the stability of the ligand in the binding pose. 
# 
# To have more confidence in the stability scores, we run multiple repeat simulations. In reporting the final PoseScore and ContactScore, after 10 repeats, we take the mean of the score over the last 2 ns, which also helps with the noise.
# 
# Let's see what the mean of the scores looks when average-out over time.

# In[4]:


# We'll store the results from 10 repeats in a 10x99 matrix.
CompScores = np.zeros((10,99))
PoseScores = np.zeros((10,99))
ContactScores = np.zeros((10,99))

# Fill those matrices with the scores from each repeat
for idx in range(0,10):
    f = os.path.join('output',f'rep_{idx}','bpm_results.csv')
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
time_sequence = np.linspace(0,10,99)

plt.title('CompScore from 10 repeats')
plt.plot(time_sequence,averagedCompScore, color='blue')
# Visualise the standard deviation of CompScore at each frame
plt.fill_between(time_sequence, averagedCompScore-CompScore_stddev, averagedCompScore+CompScore_stddev, 
                 color='blue', alpha=0.3, lw=0)
plt.xlabel('time(ns)')
plt.ylabel('CompScore')
plt.ylim(-5,5)
plt.savefig('multi_rep_CompScore.png',facecolor='w')
plt.close()

plt.title('PoseScore from 10 repeats')
time_sequence = np.linspace(0,10,99)
plt.plot(time_sequence,averagedPoseScore,color='darkorange')
plt.xlabel('time(ns)')
plt.ylabel('PoseScore')
plt.ylim(0,5)
plt.savefig('multi_rep_PoseScore.png',facecolor='w')
plt.close()

plt.title('ContactScore from 10 repeats')
time_sequence = np.linspace(0,10,99)
plt.plot(time_sequence,averagedContactScore,color='green')
plt.xlabel('time(ns)')
plt.ylabel('ContactScore')
plt.ylim(-0.1,1.1)
plt.savefig('multi_rep_ContactScore.png',facecolor='w')
plt.close()


# Much better. You can now see how averaging multiple repeats reduces the noise of the scores. 

# In the case of a large stability screen, we want to automate things. In order to get a single number that evaluates a given pose/ligand, we take the scores of the final 2 ns.

# In[8]:


compList = []
contactList = []
poseList = []
# Find how many repeats have been run
glob_str = os.path.join('output','rep_*')
nreps = len(glob.glob(glob_str))
for idx in range(0, nreps):
    f = os.path.join('output',f'rep_{idx}','bpm_results.csv')
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

print('\nResults from all 10 repeats\n')
print(results_df)


# The snippet of code above is taken directly from the ```collect_results()``` function in the ```openbpmd.py``` script and you should see the ```results.csv``` file in the ```output``` directory.

# A few words of advice on making sense of BPMD results:
# 1. The lower (more negative) the CompScore, the more likely a given pose to be the correct pose. 'Correct' here means a binding pose that is similar to a pose observed in an experimentally determined structure. 
# 2. PoseScore, being a more objective measure than ContactScore, should be given a slighlty higher weight when the CompScores of two poses are very similar.
# 3. The standard deviations of the scores might seem really high when compared to the scores themselves. Short metadynamics simulations are known to be very noisy and this is expected. In my experience, poses with lower SD also tend to be have a lower RMSD to the known pose. Therefore, SD is a useful indicator of confidence.
# 4. CompScores should only be compared between poses of the same ligand, not for comparing stability of different ligands. The correlation between the CompScore of single pose of a ligand and its potency as has not been investigated.
