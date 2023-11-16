try:
    import pytest
    pytest.register_assert_rewrite("atompaint.vendored.escnn_nn_testing")
except ImportError:
    pass
