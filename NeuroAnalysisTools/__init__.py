import os
__version__ = '3.0.3'

def test():
    import pytest
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    # test_dir = os.path.join(curr_dir, 'test')
    test_dir = curr_dir.replace('\\', '/')
    pytest.main(test_dir)