"""
Module containing a bunch of random utils

Author: Simon Thomas
Date: June-02-2020


DO TO
    - Add the visualisation tools here eventually.
    - Add FID score calculation

"""

import os


class WeightFileManager:
    """
    Automatically removes old weights.

    Useful for small disk-space limits on servers

    """

    def __init__(self, directory):
        self.dir = directory

    def update(self, n_to_keep):
        """
        Keeps the latest n files.
        """
        files = os.listdir(self.dir)

        to_keep = sorted(files)[-n_to_keep::]

        for file in files:
            if file not in to_keep:
                os.remove(os.path.join(self.dir, file))

