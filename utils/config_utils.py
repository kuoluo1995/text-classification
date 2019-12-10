from utils import yaml_utils


def get_config(name):
    config_ = yaml_utils.read('configs/' + name + '.yaml')
    return config_


# 对比dict的update只能进行下一层的update，无法实现迭代。所以才写了这个函数
def dict_update(_dict, extend_dict):
    for key, value in extend_dict.items():
        if isinstance(value, dict) and key in _dict.keys() and isinstance(_dict[key], dict):
            _dict[key] = dict_update(_dict[key], value)
        else:
            _dict[key] = value
    return _dict


def object2dict(option):
    _dict = dict()
    for key, value in option.__dict__:
        if isinstance(value, Option):
            _dict[key] = object2dict(value)
        else:
            _dict[key] = value
    return _dict


def _dict2object(_dict):
    for key, value in _dict.items():
        if isinstance(value, dict):
            _dict[key] = Option(value)
        else:
            _dict[key] = value
    return _dict


class Option(dict):
    def __init__(self, *args, **kwargs):
        super(Option, self).__init__(*args, **kwargs)
        self.__dict__ = _dict2object(self)
