import pytest

from group_adjust import *


@pytest.mark.parametrize(
    "group_adjust_func",
    [
        (group_adjust_pandas),
        (group_adjust_polars),
        (group_adjust_numpy),
    ],
)
def test_three_groups(group_adjust_func):
    vals = [1, 2, 3, 8, 5]
    grps_1 = ["USA", "USA", "USA", "USA", "USA"]
    grps_2 = ["MA", "MA", "MA", "RI", "RI"]
    grps_3 = ["WEYMOUTH", "BOSTON", "BOSTON", "PROVIDENCE", "PROVIDENCE"]
    weights = [0.15, 0.35, 0.5]

    adj_vals = group_adjust_func(vals, [grps_1, grps_2, grps_3], weights)
    # 1 - (USA_mean*.15 + MA_mean * .35 + WEYMOUTH_mean * .5)
    # 2 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # 3 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # etc ...
    # Plug in the numbers ...
    # 1 - (.15 * 3.8 + .35 * 2.0 + .5 * 1.0) = -0.770
    # 2 - (.15 * 3.8 + .35 * 2.0 + .5 * 2.5) = -0.520
    # 3 - (.15 * 3.8 + .35 * 2.0 + .5 * 2.5) =  0.480
    # etc...

    answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5


@pytest.mark.parametrize(
    "group_adjust_func",
    [
        (group_adjust_pandas),
        (group_adjust_polars),
        (group_adjust_numpy),
    ],
)
def test_two_groups(group_adjust_func):
    vals = [1, 2, 3, 8, 5]
    grps_1 = ["USA", "USA", "USA", "USA", "USA"]
    grps_2 = ["MA", "RI", "CT", "CT", "CT"]
    weights = [0.65, 0.35]

    adj_vals = group_adjust_func(vals, [grps_1, grps_2], weights)
    # 1 - (.65 * 3.8 + .35 * 1.0) = -1.82
    # 2 - (.65 * 3.8 + .35 * 2.0) = -1.17
    # 3 - (.65 * 3.8 + .35 * 5.33333) = -1.33666
    answer = [-1.82, -1.17, -1.33666, 3.66333, 0.66333]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5


@pytest.mark.parametrize(
    "group_adjust_func, null_value",
    [
        (group_adjust_pandas, np.NaN),
        (group_adjust_polars, None),
        (group_adjust_numpy, np.NaN),
    ],
)
def test_missing_vals(group_adjust_func, null_value):
    vals = [1, null_value, 3, 5, 8, 7]
    grps_1 = ["USA", "USA", "USA", "USA", "USA", "USA"]
    grps_2 = ["MA", "RI", "RI", "CT", "CT", "CT"]
    weights = [0.65, 0.35]

    adj_vals = group_adjust_func(vals, [grps_1, grps_2], weights)

    answer = [-2.47, null_value, -1.170, -0.4533333, 2.54666666, 1.54666666]

    for ans, res in zip(answer, adj_vals):
        if ans is None:
            assert res is None
        elif np.isnan(ans):
            assert np.isnan(res)
        else:
            assert abs(ans - res) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "group_adjust_func, null_value",
    [
        (group_adjust_pandas, np.NaN),
        (group_adjust_polars, None),
        (group_adjust_numpy, np.NaN),
    ],
)
def test_weights_len_equals_group_len(group_adjust_func, null_value):
    # Need to have 1 weight for each group

    vals = [1, null_value, 3, 5, 8, 7]
    # vals = [1, None, 3, 5, 8, 7]
    grps_1 = ["USA", "USA", "USA", "USA", "USA", "USA"]
    grps_2 = ["MA", "RI", "RI", "CT", "CT", "CT"]
    weights = [0.65]

    with pytest.raises(ValueError):
        group_adjust_func(vals, [grps_1, grps_2], weights)


@pytest.mark.parametrize(
    "group_adjust_func, null_value",
    [
        (group_adjust_pandas, np.NaN),
        (group_adjust_polars, None),
        (group_adjust_numpy, np.NaN),
    ],
)
def test_group_len_equals_vals_len(group_adjust_func, null_value):
    # The groups need to be same shape as vals
    vals = [1, null_value, 3, 5, 8, 7]
    grps_1 = ["USA"]
    grps_2 = ["MA", "RI", "RI", "CT", "CT", "CT"]
    weights = [0.65]

    with pytest.raises(ValueError):
        group_adjust_func(vals, [grps_1, grps_2], weights)


@pytest.mark.parametrize(
    "group_adjust_func, null_value",
    [
        (group_adjust_pandas, np.NaN),
        (group_adjust_polars, None),
        (group_adjust_numpy, np.NaN),
    ],
)
def test_performance(group_adjust_func, null_value, benchmark):
    # vals = 1000000*[1, None, 3, 5, 8, 7]
    # If you're doing numpy, use the np.NaN instead
    vals = 1000000 * [1, null_value, 3, 5, 8, 7]
    grps_1 = 1000000 * [1, 1, 1, 1, 1, 1]
    grps_2 = 1000000 * [1, 1, 1, 1, 2, 2]
    grps_3 = 1000000 * [1, 2, 2, 3, 4, 5]
    weights = [0.20, 0.30, 0.50]

    start = datetime.now()
    benchmark(group_adjust_func, vals, [grps_1, grps_2, grps_3], weights)
    end = datetime.now()
    diff = end - start
