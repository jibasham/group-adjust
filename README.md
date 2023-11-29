# group-adjust

Demeaning a series with multiple labels per entry

# Installation

Navigate to the parent directory of this repository and run:

```bash
git clone https://github.com/jibasham/group-adjust.git .
cd group-adjust
python -m venv venv
source venv/bin/activate
pip install .
```

# The Problem

```
Your task is to implement the 'group_adjust' method as described
below, ensuring that all provided unit tests pass. Your solution
can be pure Python, NumPy, Pandas, or any combination of the three.
There are multiple ways of solving this problem, so feel free to be
creative, but please include comments to explain your code. Bonus
points are given for particularly efficient (fast) implementations!

Group Adjust Method
The algorithm needs to do the following:
1.) For each group-list provided, calculate the means of the values for each
unique group.

  For example:
  vals       = [  1  ,   2  ,   3  ]
  ctry_grp   = ['USA', 'USA', 'USA']
  state_grp  = ['MA' , 'MA' ,  'CT' ]

  There is only 1 country in the ctry_grp list.  So to get the means:
    USA_mean == mean(vals) == 2
    ctry_means = [2, 2, 2]
  There are 2 states, so to get the means for each state:
    MA_mean == mean(vals[0], vals[1]) == 1.5
    CT_mean == mean(vals[2]) == 3
    state_means = [1.5, 1.5, 3]

2.) Using the weights, calculate a weighted average of those group means
  Continuing from our example:
  weights = [.35, .65]
  35% weighted on country, 65% weighted on state
  ctry_means  = [2  , 2  , 2]
  state_means = [1.5, 1.5, 3]
  weighted_means = [2*.35 + .65*1.5, 2*.35 + .65*1.5, 2*.35 + .65*3]

3.) Subtract the weighted average group means from each original value
  Continuing from our example:
  val[0] = 1
  ctry[0] = 'USA' --> 'USA' mean == 2, ctry weight = .35
  state[0] = 'MA' --> 'MA'  mean == 1.5, state weight = .65
  weighted_mean = 2*.35 + .65*1.5 = 1.675
  demeaned = 1 - 1.675 = -0.675
  Do this for all values in the original list.

4.) Return the demeaned values

Hint: See the test cases below for how the calculation should work.
```

# Initial Thoughts to The Prompt

So, we will be computing a weighted average. A good bread and butter data analysis task. 
