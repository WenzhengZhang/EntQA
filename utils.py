import collections
import random
import bisect


class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


def sample_range_excluding(n, k, excluding):
    skips = [j - i for i, j in enumerate(sorted(set(excluding)))]
    s = random.sample(range(n - len(skips)), k)
    return [i + bisect.bisect_right(skips, i) for i in s]
