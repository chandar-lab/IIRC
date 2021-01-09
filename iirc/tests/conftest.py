def pytest_addoption(parser):
    parser.addoption("--imagenet_root", action="store", default="./data")


def pytest_generate_tests(metafunc):
    """ This is called for every test. Only get/set command line arguments
    if the argument is specified in the list of test "fixturenames".
    """
    option_value = metafunc.config.option.imagenet_root
    if 'imagenet_root' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("imagenet_root", [option_value])
