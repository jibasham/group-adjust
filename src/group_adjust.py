from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import polars as pl
from memory_profiler import profile


# Your task is to implement the 'group_adjust' method as described
# below, ensuring that all provided unit tests pass. Your solution
# can be pure Python, NumPy, Pandas, or any combination of the three.
# There are multiple ways of solving this problem, so feel free to be
# creative, but please include comments to explain your code. Bonus
# points are given for particularly efficient (fast) implementations!
#
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
#
# Hint: See the test cases below for how the calculation should work.


@profile
def group_adjust(
    vals: List[float], groups: List[List[Union[str, int]]], weights: List[float]
) -> List[float]:
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
    return group_adjust_polars(vals, groups, weights)


def group_adjust_pandas(
    vals: List[float], groups: List[List[Union[str, int]]], weights: List[float]
) -> List[float]:
    """
    Calculate a group adjustment (demean) using Pandas.

    Pandas of course is the go-to. Everyone knows it since most data science examples
    use it. It's pretty fast. If I am writing for an established codebase, I would
    most likely use Pandas unless there was a good reason not to. It will be more maintainable
    and quicker for others to adopt.

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
    df = pd.DataFrame({"vals": pd.to_numeric(vals, errors="coerce")})

    for i, group in enumerate(groups):
        if len(group) != len(vals):
            raise ValueError("Length of each group must match the length of vals.")
        # We can reduce the size in memory by quite a bit using Categoricals
        # since I expect the number of unique labels to be much smaller than
        # the number of values.
        df[f"group_{i}"] = pd.Categorical(group)

        # Compute the means for each group in a small DataFrame, and then
        # use it as a sort of lookup table to map the weighted means to each row.
        group_key = f"group_{i}"
        means = df.groupby(group_key)["vals"].mean()
        df[f"weighted_mean_{i}"] = df[group_key].map(means).astype(float) * weights[i]

    # Sum the weighted means for each row
    weighted_sum_columns = [f"weighted_mean_{i}" for i in range(len(groups))]
    df["weighted_sum"] = df[weighted_sum_columns].sum(axis=1)

    # Demean the values
    demeaned_vals = df["vals"] - df["weighted_sum"]

    return demeaned_vals.tolist()


def group_adjust_polars(vals, groups, weights):
    """
    Calculate a group adjustment (demean) using Polars.

    Polars is generally more efficient than Pandas, as it tries to use more CPU cores
    and has other optimizations under the hood to reshuffle the order of lazy operations
    and reduce memory usage. Since we are going for speed here, I thought I would give
    it a go and compare to pandas. I am getting 966 ms wall clock and 334 MB memory
    utilization for the "benchmark" test with 20M elements in the DataFrame.

    The implementation here is pretty much the same as for pandas (which I did first),
    but just as a fairly recognizable transcription into polars syntax.

    Note: For the tests to pass in polars, use "None" as the null value
    instead of np.NaN.

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

    # Create a Polars DataFrame
    df = pl.DataFrame({"vals": vals})

    for i, group in enumerate(groups):
        if len(group) != len(vals):
            raise ValueError("Length of each group must match the length of vals.")
        df = df.with_columns(pl.Series(f"group_{i}", group))

    # Calculate and store weighted means in the DataFrame
    for i in range(len(groups)):
        group_key = f"group_{i}"
        df = df.join(
            df.groupby(group_key).agg(pl.col("vals").mean().alias("mean")), on=group_key, how="left"
        )
        df = df.with_columns((pl.col("mean") * weights[i]).alias(f"weighted_mean_{i}"))
        df = df.drop("mean")

    # Sum the weighted means for each row
    weighted_sum_expr = sum([pl.col(f"weighted_mean_{i}") for i in range(len(groups))])
    df = df.with_columns(weighted_sum_expr.alias("weighted_sum"))

    # Demean the values
    df = df.with_columns((pl.col("vals") - pl.col("weighted_sum")).alias("demeaned_vals"))

    return df["demeaned_vals"].to_list()


if __name__ == "__main__":
    vals = 1_000_000 * [1, np.NaN, 3, 5, 8, 7]
    grps_1 = 1_000_000 * [1, 1, 1, 1, 1, 1]
    grps_2 = 1_000_000 * [1, 1, 1, 1, 2, 2]
    grps_3 = 1_000_000 * [1, 2, 2, 3, 4, 5]
    weights = [0.20, 0.30, 0.50]

    start = datetime.now()
    group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    end = datetime.now()
    diff = end - start
    print(diff)
