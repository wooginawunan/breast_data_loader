class SeedShifter:
    def __init__(self, random_seed, number_of_loaders):
        self.random_seed = random_seed
        self.number_of_loaders = number_of_loaders

    def get_seed(self, phase, epoch_number):
        if phase in ("train", "training"):
            return self.random_seed + ((self.number_of_loaders + 1) * epoch_number)
        elif phase in ("val", "validation"):
            return self.random_seed + (10 ** 6) + ((self.number_of_loaders + 1) * epoch_number)
        elif phase in ("test",):
            return self.random_seed + (2 * (10 ** 6)) + ((self.number_of_loaders + 1) * epoch_number)
        else:
            raise KeyError(phase)

    @classmethod
    def from_parameters(cls, parameters):
        return cls(
            random_seed=parameters['random_seed'],
            number_of_loaders=parameters['number_of_loaders'],
        )


