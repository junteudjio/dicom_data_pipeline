import errno
import os

__author__ = 'Junior Teudjio'
__all__ = ['mkdir_p']


def mkdir_p(path):
    '''
    Recursively creates the directories in a given path
    Equivalent to batch cmd mkdir -p.
    Parameters
    ----------
    path : str
        Path to the final directory to create.
    Returns
    -------
    '''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise