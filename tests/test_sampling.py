import numpy as np

from quanheng.quantity import Quantity
from quanheng.sampling import from_quantiles, quantiles


def test_stream_is_pinned() -> None:
    # 这个测试失败意味着随机流变了：所有已保存的结果都无法复算，属于破坏性变更。
    np.testing.assert_allclose(
        quantiles(0, "score:a:rent", 3), [0.97302797, 0.41170296, 0.09270841], atol=1e-8
    )


def test_streams_depend_on_seed_and_key_only() -> None:
    assert np.array_equal(quantiles(7, "k", 5), quantiles(7, "k", 5))
    assert not np.array_equal(quantiles(7, "k", 5), quantiles(7, "j", 5))
    assert not np.array_equal(quantiles(7, "k", 5), quantiles(8, "k", 5))


def test_inverse_cdf_values() -> None:
    u = np.array
    assert from_quantiles(Quantity.model_validate(7), u([0.1, 0.9])).tolist() == [7, 7]
    assert from_quantiles(Quantity.model_validate([10, 20]), u([0.0, 0.25])).tolist() == [10, 12.5]
    np.testing.assert_allclose(from_quantiles(Quantity.model_validate([0, 1, 2]), u([0.5])), [1.0])
    np.testing.assert_allclose(from_quantiles(Quantity.model_validate([0, 0, 4]), u([0.75])), [2.0])
    np.testing.assert_allclose(
        from_quantiles(Quantity.model_validate([0, 3, 4]), u([0.5])), [6**0.5]
    )


def test_inverse_cdf_is_monotone_and_within_bounds() -> None:
    u = np.sort(quantiles(1, "m", 1000))
    for spec in ([10, 20], [0, 1, 2], [0, 0, 4], [-5, -1, 0]):
        x = from_quantiles(Quantity.model_validate(spec), u)
        assert np.all(np.diff(x) >= 0) and x.min() >= spec[0] and x.max() <= spec[-1]
