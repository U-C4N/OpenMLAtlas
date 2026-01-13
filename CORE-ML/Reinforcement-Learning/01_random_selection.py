# ============================================================================
# REINFORCEMENT LEARNING - RANDOM SELECTION (BASELINE)
# ============================================================================
# This file implements the simplest strategy: SELECT A RANDOM AD EVERY ROUND!
#
# Why is this a bad strategy?
# - It doesn't learn which ad is good
# - We rely on luck every time
# - This is our "baseline" model
# - We use this to compare how much better smarter algorithms
#   (UCB, Thompson Sampling) perform.
#
# Scenario: A website has 10 different ads.
# Goal: Maximize clicks from the ads we show to users.
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import random

# 1. Load the Data
# ----------------
# We read the CSV file created by running generate_data.py.
# Each row = 1 user
# Each column = 1 ad (0 or 1: did they win when clicked?)

data = pd.read_csv('ads_data.csv')

print("=== DATASET LOADED ===")
print(f"Shape: {data.shape[0]} users x {data.shape[1]} ads")
print()
print("First 5 rows:")
print(data.head())
print()

# 2. Define Variables
# -------------------
N = 10000   # Total number of rounds (users)
d = 10      # Number of ads

total_reward = 0            # Total reward we earned (clicks)
ads_selected = []           # List to track which ads we selected

# 3. Start the Main Loop - RANDOM SELECTION
# ------------------------------------------
# Each round:
# 1) Select a random ad (0-9)
# 2) Check the result for that ad (0 or 1)
# 3) Add to reward
# 4) Save selected ad (for histogram)

print("=== RANDOM SELECTION STARTING ===")
print("Selecting a random ad each round...")
print()

for n in range(0, N):
    # Select a random ad (from 0 to 9)
    # random.randrange(10) -> picks one from 0,1,2,3,4,5,6,7,8,9
    ad = random.randrange(d)

    # What is the result for the selected ad for this user?
    # data.values[n, ad] -> row n, column ad
    reward = data.values[n, ad]

    # Add reward to total
    total_reward = total_reward + reward

    # Save selected ad for histogram
    ads_selected.append(ad)

# 4. Print Results
# ----------------
print("=== RESULTS ===")
print(f"Total rounds: {N}")
print(f"Total reward earned: {total_reward}")
print(f"Average reward/round: {total_reward / N:.4f}")
print()

# How many times was each ad selected?
print("=== HOW MANY TIMES EACH AD WAS SELECTED ===")
for i in range(d):
    count = ads_selected.count(i)
    print(f"  Ad {i}: selected {count} times")
print()

# 5. Draw Histogram
# -----------------
# Visualize how many times each ad was selected.
# With random selection, all ads should be selected roughly equally (~1000 times).

plt.figure(figsize=(10, 6))
plt.hist(ads_selected, bins=range(0, d+1), edgecolor='black', color='steelblue', align='left')
plt.title('Random Selection - How Many Times Each Ad Was Selected?', fontsize=14)
plt.xlabel('Ad Number', fontsize=12)
plt.ylabel('Selection Count', fontsize=12)
plt.xticks(range(0, d))
plt.grid(axis='y', alpha=0.3)

# Add total reward to the graph
plt.text(0.02, 0.98, f'Total Reward: {total_reward}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()
plt.savefig('random_selection_histogram.png', dpi=150)
plt.show()

print("Histogram saved as 'random_selection_histogram.png'!")

# ============================================================================
# EXPECTED RESULT:
# ============================================================================
# - All ads will be selected roughly equally (~1000 times each)
# - Total reward will be around 1000-1200
# - This is a BAD result! Because we didn't learn the best ad (Ad_4, 30%).
#
# COMPARISON:
# - If we only selected Ad 4: ~3000 reward!
# - With random selection: ~1000-1200 reward
# - Difference: 2000 lost clicks! That's a huge loss.
#
# NEXT STEP: Let's use the UCB algorithm for smarter selection!
# ============================================================================
