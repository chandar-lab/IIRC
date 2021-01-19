import logging


def print_msg(msg):
    print(msg)
    logging.info(msg)
    return


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict