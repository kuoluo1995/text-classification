import importlib
import numpy as np
from functools import partial


def get_model_class_by_name(name):
    file_name = 'models.' + name
    libs = importlib.import_module(file_name)
    target_cls = name.replace('_', '')
    for key, cls in libs.__dict__.items():
        if target_cls.lower() == key.lower():
            target_cls = cls
            break

    if issubclass(target_cls, str):
        raise NotImplementedError('In {}.py, {} should be  in lowercase.'.format(name, target_cls))

    return target_cls


def get_model_fn(model_type, **_config):
    file_name = 'models.' + model_type + '.' + _config['name']
    libs = importlib.import_module(file_name)
    if 'build_model' in libs.__dict__.keys():
        sub_model = libs.__dict__['build_model']
        sub_model = partial(sub_model, **_config)
        return sub_model
    else:
        raise NotImplementedError('In {}.py, there should be a function with name.'.format(_config['name']))


if __name__ == '__main__':
    model_fn = get_model_fn('generator', **{'name': 'unet', 'filter_channels': 32})
    model_fn(image=np.array([[1]]), out_channels=1)
