try:
    import matplotlib
except ImportError:
    pass
else:
    matplotlib.use("Agg")

import sys
import imp
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle


def main():
    cwd = sys.argv[4]
    sys.path.append(cwd)
    pkl_file = sys.argv[1]
    imp.load_source(sys.argv[2], sys.argv[3])
    with open(pkl_file, "rb") as f:
        worker = pickle.load(f)
    worker.main()
