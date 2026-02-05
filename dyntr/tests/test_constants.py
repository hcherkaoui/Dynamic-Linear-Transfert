from dyntr import constants


def test_constants_have_expected_types():
    assert isinstance(constants.N_JOBS, int)
    assert constants.N_JOBS > 0
    assert isinstance(constants.SEED, int)
    assert constants.D > 0
    assert constants.SIGMA >= 0
