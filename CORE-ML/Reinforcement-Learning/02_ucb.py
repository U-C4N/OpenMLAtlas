# ============================================================================
# REINFORCEMENT LEARNING - UCB (Upper Confidence Bound)
# ============================================================================
# This file implements the UCB algorithm. Much smarter than random selection!
#
# UCB LOGIC:
# ----------
# 1) Each ad has an "average reward" (past performance)
# 2) Each ad has an "uncertainty bonus" (less tried = more bonus)
# 3) UCB = Average + Bonus
# 4) Each round, select the ad with the HIGHEST UCB value!
#
# WHY DOES THIS WORK?
# -------------------
# - At the start: Untried ads get high bonus -> Explore
# - Over time: Good ads get high averages -> Exploit
# - This balance helps us LEARN the best ad!
#
# FORMULA:
# --------
# UCB_i = average_i + sqrt(3 * log(n) / selection_count_i)
#         ↑               ↑
#    Exploitation      Exploration
#                    (Discovery Bonus)
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import math

# 1. Load the Data
# ----------------
data = pd.read_csv('ads_data.csv')

print("=== DATASET LOADED ===")
print(f"Shape: {data.shape[0]} users x {data.shape[1]} ads")
print()

# 2. Define Variables
# -------------------
N = 10000   # Total number of rounds
d = 10      # Number of ads

# Values to track for each ad:
numbers_of_selections = [0] * d    # How many times each ad was selected? [0,0,0,0,0,0,0,0,0,0]
sums_of_rewards = [0] * d          # Total reward from each ad?

# Overall results:
total_reward = 0               # Total reward from all rounds
ads_selected = []              # For histogram: which ads were selected?

# 3. Start the Main Loop - UCB ALGORITHM
# ---------------------------------------
print("=== UCB ALGORITHM STARTING ===")
print()

for n in range(0, N):

    # -----------------------------------------------------------------------
    # STEP 3.1: Calculate UCB value for each ad
    # -----------------------------------------------------------------------
    best_ad = 0          # Current best ad (default 0)
    max_upper_bound = 0  # Current highest UCB value

    for i in range(0, d):

        # If ad was never selected -> Give priority (very high UCB)
        # This ensures we try each ad at least once in the first 10 rounds.
        if numbers_of_selections[i] == 0:
            upper_bound = 1e400  # A very large number (practically infinity)

        else:
            # APPLY THE UCB FORMULA:
            # UCB = average + delta

            # Average reward = total reward / selection count
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]

            # Delta (exploration bonus) = sqrt(3 * log(n+1) / selection_count)
            # We use n+1 because log(0) is undefined at n=0.
            # The value 3 is theoretically optimal. (Some sources use 2)
            delta_i = math.sqrt(3 * math.log(n + 1) / numbers_of_selections[i])

            # UCB = Average + Bonus
            upper_bound = average_reward + delta_i

        # Find the highest UCB
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            best_ad = i

    # -----------------------------------------------------------------------
    # STEP 3.2: Select the best ad and get the result
    # -----------------------------------------------------------------------
    ad = best_ad

    # What is the reward for this user for the selected ad? (0 or 1)
    reward = data.values[n, ad]

    # -----------------------------------------------------------------------
    # STEP 3.3: Update statistics
    # -----------------------------------------------------------------------
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward

    # Add to overall total
    total_reward = total_reward + reward

    # Save for histogram
    ads_selected.append(ad)

    # Print progress every 2000 rounds
    if (n + 1) % 2000 == 0:
        print(f"Round {n+1}: Total reward = {total_reward}")

# 4. Print Results
# ----------------
print()
print("=" * 60)
print("=== UCB RESULTS ===")
print("=" * 60)
print(f"Total rounds: {N}")
print(f"Total reward earned: {total_reward}")
print(f"Average reward/round: {total_reward / N:.4f}")
print()

# Detailed stats for each ad
print("=== DETAILED STATISTICS FOR EACH AD ===")
print("-" * 50)
for i in range(d):
    if numbers_of_selections[i] > 0:
        avg = sums_of_rewards[i] / numbers_of_selections[i]
    else:
        avg = 0
    print(f"Ad {i}: selected {numbers_of_selections[i]:5d} times, "
          f"Total reward: {sums_of_rewards[i]:4d}, "
          f"Average: {avg:.3f}")
print()

# Find the most selected ad
most_selected = numbers_of_selections.index(max(numbers_of_selections))
print(f"MOST SELECTED AD: Ad {most_selected} ({max(numbers_of_selections)} times)")
print()

# 5. Draw Histogram
# -----------------
plt.figure(figsize=(10, 6))
plt.hist(ads_selected, bins=range(0, d+1), edgecolor='black', color='forestgreen', align='left')
plt.title('UCB Algorithm - How Many Times Each Ad Was Selected?', fontsize=14)
plt.xlabel('Ad Number', fontsize=12)
plt.ylabel('Selection Count', fontsize=12)
plt.xticks(range(0, d))
plt.grid(axis='y', alpha=0.3)

# Add total reward to the graph
plt.text(0.02, 0.98, f'Total Reward: {total_reward}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))

plt.tight_layout()
plt.savefig('ucb_histogram.png', dpi=150)
plt.show()

print("Histogram saved as 'ucb_histogram.png'!")

# ============================================================================
# EXPECTED RESULT:
# ============================================================================
# - Ad 4 (the best ad) will be selected THE MOST (7000-8000+ times)
# - Other ads will be selected very few times (only during exploration)
# - Total reward: ~2500-2800 (2X MORE than random selection!)
#
# COMPARISON:
# ============================================================================
# | Method          | Total Reward | Explanation                           |
# |-----------------|--------------|---------------------------------------|
# | Random Selection| ~1000-1200   | No learning, relies on luck           |
# | UCB             | ~2500-2800   | Finds best ad and exploits it         |
# | Optimal (Ad 4)  | ~3000        | If we only selected Ad 4              |
# ============================================================================
#
# UCB gets very close to optimal! Small loss = exploration cost
# ============================================================================
