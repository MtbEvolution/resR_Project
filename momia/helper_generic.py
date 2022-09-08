__author__ = "jz-rolling"

# OMEGA helper
# 04/07/2021


def set_default_by_kwarg(key,kwarg,default):
    """
    :param key: kwarg key
    :param kwarg: kwargs
    :param default: default value
    """
    if key in kwarg:
        return kwarg[key]
    else:
        return default

