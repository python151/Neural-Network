import copy

def copy_network(func):
    def inner(self, *args, **kwargs):
        network = copy.deepcopy(self.network)
        return func(self, network=network, *args, **kwargs)
    return inner