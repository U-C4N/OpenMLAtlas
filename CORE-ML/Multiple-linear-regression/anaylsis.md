                            OLS Regression Results
==============================================================================
Dep. Variable:                  Grade   R-squared:                       0.904
Model:                            OLS   Adj. R-squared:                  0.899
Method:                 Least Squares   F-statistic:                     177.3
Date:                Mon, 01 Dec 2025   Prob (F-statistic):           2.06e-37
Time:                        22:41:10   Log-Likelihood:                -254.09
No. Observations:                  80   AIC:                             518.2
Df Residuals:                      75   BIC:                             530.1
Df Model:                           4
Covariance Type:            nonrobust
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
const              28.1840      2.024     13.922      0.000      24.151      32.217
Gender              1.7909      1.349      1.328      0.188      -0.896       4.478
StudyHours          6.1614      0.235     26.202      0.000       5.693       6.630
City_London         0.0426      1.704      0.025      0.980      -3.351       3.436
City_Manchester    -1.1699      1.812     -0.646      0.520      -4.779       2.440
==============================================================================
Omnibus:                       10.657   Durbin-Watson:                   2.176
Prob(Omnibus):                  0.005   Jarque-Bera (JB):                3.437
Skew:                          -0.068   Prob(JB):                        0.179
Kurtosis:                       1.994   Cond. No.                         24.7
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# The "Magic Formula" for Grades: A Simple Story

Imagine we built a magic model. This model's job is to look at a student and guess exactly what grade they will get on a test.

We gave the model some clues to help it guess:
1. How many hours the student studied?
2. Is the student a boy or a girl?
3. Which city does the student live in?

Here is what the model found out!

### 1. Our Model is a Genius! (R-squared)
* **The Score:** 90% (0.904)
* **What it means:** Our model is super smart! It guessed almost everyone's grade correctly. It understands 90% of the reasons why grades go up or down.

### 2. The Super Power: Studying (Study Hours)
* **Is it real?** YES! (P-value is 0.000)
* **How does it work?** This is the only thing that matters.
* **The Magic Number (6.16):** For every 1 extra hour you study, the model says your grade goes up by 6 points.
    * Study 1 hour? You get +6 points.
    * Study 2 hours? You get +12 points!
    * It is like collecting coins in a video game. More hours = Higher score.

### 3. The Things That Do NOT Matter
The model looked at other clues, but it said: "Nah, these are boring. They don't change anything."

* **Boy or Girl? (Gender):** It doesn't matter. Being a boy or a girl does not change your grade. It is just luck.
* **Where do you live? (City):** It doesn't matter if you live in London or Manchester. Your city does not make you smarter.

### 4. The Big Lesson
If you want a high score, you just have to study. It doesn't matter who you are or where you live. The only secret to winning is putting in the time!