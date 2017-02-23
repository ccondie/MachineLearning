class Node(object):
    def __init__(self, weight=None):
        self.weight = weight
        self.net = 0
        self.out = 0
        self.sigma = None
        self.target = None

