
import pandas as pd


# Your task is to implement the 'group_adjust' method as described
# below, ensuring that all provided unit tests pass. Your solution
# can be pure Python, NumPy, Pandas, or any combination of the three.
# There are multiple ways of solving this problem, so feel free to be
# creative, but please include comments to explain your code. Bonus
# points are given for particularly efficient (fast) implementations!

# Group Adjust Method
# The algorithm needs to do the following:
# 1.) For each group-list provided, calculate the means of the values for each
# unique group.
#
#   For example:
#   vals       = [  1  ,   2  ,   3  ]
#   ctry_grp   = ['USA', 'USA', 'USA']
#   state_grp  = ['MA' , 'MA' ,  'CT' ]
#
#   There is only 1 country in the ctry_grp list.  So to get the means:
#     USA_mean == mean(vals) == 2
#     ctry_means = [2, 2, 2]
#   There are 2 states, so to get the means for each state:
#     MA_mean == mean(vals[0], vals[1]) == 1.5
#     CT_mean == mean(vals[2]) == 3
#     state_means = [1.5, 1.5, 3]
#
# 2.) Using the weights, calculate a weighted average of those group means
#   Continuing from our example:
#   weights = [.35, .65]
#   35% weighted on country, 65% weighted on state
#   ctry_means  = [2  , 2  , 2]
#   state_means = [1.5, 1.5, 3]
#   weighted_means = [2*.35 + .65*1.5, 2*.35 + .65*1.5, 2*.35 + .65*3]
#
# 3.) Subtract the weighted average group means from each original value
#   Continuing from our example:
#   val[0] = 1
#   ctry[0] = 'USA' --> 'USA' mean == 2, ctry weight = .35
#   state[0] = 'MA' --> 'MA'  mean == 1.5, state weight = .65
#   weighted_mean = 2*.35 + .65*1.5 = 1.675
#   demeaned = 1 - 1.675 = -0.675
#   Do this for all values in the original list.
#
# 4.) Return the demeaned values

# Hint: See the test cases below for how the calculation should work.


def group_adjust(vals, groups, weights):
    """
    Calculate a group adjustment (demean).

    Parameters
    ----------

    vals    : List of floats/ints

        The original values to adjust

    groups  : List of Lists

        A list of groups. Each group will be a list of ints

    weights : List of floats

        A list of weights for the groupings.

    Returns
    -------

    A list-like demeaned version of the input values
    """
    if len(groups) != len(weights):
        raise ValueError("Length of weights must match the number of groups.")

    # Convert to DataFrame for efficient computation
    df = pd.DataFrame({'vals': pd.to_numeric(vals, errors='coerce')})

    for i, group in enumerate(groups):
        if len(group) != len(vals):
            raise ValueError("Length of each group must match the length of vals.")
        df[f'group_{i}'] = pd.Categorical(group)

    # Pre-calculate group means and store in a dictionary
    group_means = {}
    for i in range(len(groups)):
        group_key = f'group_{i}'
        means = df.groupby(group_key)['vals'].mean()
        weighted_means = means * weights[i]
        group_means[group_key] = df[group_key].map(weighted_means)

    # Sum the weighted means for each row
    weighted_sum = pd.DataFrame(group_means).sum(axis=1)

    # Demean the values
    demeaned_vals = df['vals'] - weighted_sum

    return demeaned_vals.tolist()
