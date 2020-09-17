import numpy as np


def integers_to_one_hot(integer_vector, max_val=None):
    integer_vector = np.squeeze(integer_vector)
    if max_val == None:
        max_val = np.max(integer_vector)
    one_hot = np.zeros((integer_vector.shape[0], max_val + 1))
    for i, integer in enumerate(integer_vector):
        one_hot[i, integer] = 1.0
    return one_hot


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
