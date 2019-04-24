"""
This file contains the three architectures used for the model ensemble in scaden
The architectures are stored in the dict 'architectures', each element of which is a tuple that consists
of the hidden_unit sizes (element 0) and the respective dropout rates (element 1)
"""

architectures = {'m256':    ([256, 128, 64, 32],    [0, 0, 0, 0]),
                 'm512':    ([512, 256, 128, 64],   [0, 0.3, 0.2, 0.1]),
                 'm1024':   ([1024, 512, 256, 128], [0, 0.6, 0.3, 0.1])}

