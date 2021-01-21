import numpy as np


def styblinski_tang(x, axis=None):
    t1 = np.sum(x ** 4, axis=axis)
    t2 = -16 * np.sum(x ** 2, axis=axis)
    t3 = 5 * np.sum(x, axis=axis)
    return 0.5 * (t1 + t2 + t3)


def batch_styblinski_tang(x):
    return styblinski_tang(x, axis=1)


def sequential_styblinski_tang(x):
    return np.array([styblinski_tang(e) for e in x])


def qing(x):
    return np.sum(np.array([(e ** 2 - i) ** 2 for i, e in enumerate(x)]))


def cos_mixture(x):
    return np.sum(np.cos(x ** 5 * np.pi)) + np.sum(x ** 2)


def schwefel221(x, axis=None):
    return np.max(np.abs(x), axis=axis)


def batch_schwefel221(x):
    return schwefel221(x, axis=1)


def apline1(x, axis=None):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=axis)


def batch_apline1(x):
    return apline1(x, axis=1)
