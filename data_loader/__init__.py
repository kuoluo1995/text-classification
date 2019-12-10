import importlib


def get_data_loader_by_name(name):
    file_name = 'data_loader.' + name
    libs = importlib.import_module(file_name)
    target_cls = name.replace('_', '')
    for key, cls in libs.__dict__.items():
        if target_cls.lower() == key.lower():
            target_cls = cls
            break

    if issubclass(target_cls, str):
        raise NotImplementedError('In {}.py, {} should be in lowercase.'.format(name, target_cls))
    return target_cls
