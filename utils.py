import pickle


def load_pickle_file(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)

    return b


def save_as_pickle_file(file, path):
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=2)
