"""
@author: James Basham
@email: jibasham@gmail.com
"""
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import polars as pl
from memory_profiler import profile


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

    This is the first thing I attempted and what I would call the "vanilla" solution.
    Essentially group by all the unique labels to get the means. It is pretty
    straight-forward to code up, and reasonably fast, although the extra overhead for
    the convenient syntax would slow you down over a pure NumPy solution, since its
    numpy under the hood anyway.

    I am getting 2.4 s wall clock and 540 MB memory utilization for the "benchmark"
    test with 20M elements in the DataFrame. I am a little sad the memory usage
    is so high - I did try to play some tricks to use categorical data types and
    throw away the intermediate values as I accumulate the weighted means.

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
    df["weighted_sum"] = 0.0

    for i, group in enumerate(groups):
        if len(group) != len(vals):
            raise ValueError("Length of each group must match the length of vals.")
        # We can reduce the size in memory by quite a bit using Categoricals
        # since I expect the number of unique labels to be much smaller than
        # the number of values. Gets you down to a uint8 for each entry, so that's
        # 10x less memory to store "PROVIDENCE" than a string.
        group_key = f"group_{i}"
        df[group_key] = pd.Categorical(group)

        # Compute the means for each group in a small DataFrame, and then
        # use it as a sort of lookup table to map the weighted means to each row.
        means = df.groupby(group_key, observed=False)["vals"].mean()
        # Accumulate the weighted means for each group as you go
        # For a large number of groups you could end up using a lot of memory
        # if you were to save the intermediate values.
        df["weighted_sum"] += df[group_key].map(means).astype(float) * weights[i]

    # Demean the values
    demeaned_vals = df["vals"] - df["weighted_sum"]

    return demeaned_vals.tolist()


def group_adjust_polars(
    vals: List[float], groups: List[List[Union[str, int]]], weights: List[float]
) -> List[float]:
    """
    Calculate a group adjustment (demean) using Polars.

    Polars is generally more efficient than Pandas, as it tries to use more CPU cores
    and has other optimizations under the hood to reshuffle the order of lazy operations
    and reduce memory usage. Since we are going for speed here, I thought I would give
    it a go and compare to pandas. I am getting 717 ms wall clock and 250 MB memory
    utilization for the "benchmark" test with 20M elements in the DataFrame.

    The implementation here is pretty much the same as for pandas (which I did first),
    but just as a fairly recognizable transcription into polars syntax. I am storing
    the intermediate values in the DataFrame, which is not necessary, but it does oddly
    make it a bit faster than using a throwaway Series to store the weighted means
    (as I do in the pandas version). And I am assuming faster is more important than
    using a little extra memory.

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
    # This is a new step - for the pandas version we accumulated them as we went.
    weighted_sum_expr = sum([pl.col(f"weighted_mean_{i}") for i in range(len(groups))])
    df = df.with_columns(weighted_sum_expr.alias("weighted_sum"))

    # Demean the values
    df = df.with_columns((pl.col("vals") - pl.col("weighted_sum")).alias("demeaned_vals"))

    return df["demeaned_vals"].to_list()


def group_adjust_numpy(
    vals: List[float], groups: List[List[Union[str, int]]], weights: List[float]
) -> List[float]:
    """
    Calculate a group adjustment (demean) using NumPy.

    To be honest this is faster than I expected. I initially avoided it because I
    could not think of a way to get around for loops, and if there are a lot of groups
    it could get bogged down. But lets assume the number of groups is small, as in the
    test cases, in which case the overhead is small.

     I am getting 942 ms wall clock time
    and 310 MB memory utilization for the "benchmark" test with 20M elements.

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

    vals = np.array(vals)
    adjusted_vals = np.zeros_like(vals, dtype=float)

    for group, weight in zip(groups, weights):
        group = np.array(group)

        # Handle missing values in vals
        valid_mask = ~np.isnan(vals)
        unique_labels = np.unique(group[valid_mask])

        for label in unique_labels:
            group_mask = group == label
            mean_val = np.mean(vals[group_mask & valid_mask])
            adjusted_vals[group_mask] += mean_val * weight

    return vals - adjusted_vals


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
