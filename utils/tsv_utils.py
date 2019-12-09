import csv
from pathlib import Path


def read(path):
    with Path(path).open(mode='r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        result = list()
        for row in reader:
            result.append(row)
    return result


def write(path, list_data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(list_data)
