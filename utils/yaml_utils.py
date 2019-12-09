import yaml
from pathlib import Path


def read(path):
    with Path(path).open('r', encoding='utf-8') as file:
        params = yaml.load(file, Loader=yaml.Loader)
    return params


def write(path, data, encoding='utf-8'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding=encoding) as file:
        yaml.dump(data, file, allow_unicode=True)


if __name__ == '__main__':
    write('../test.yaml', {1: '你你你'})
