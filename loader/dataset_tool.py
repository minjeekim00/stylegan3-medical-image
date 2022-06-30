import os
import json

from glob import glob
from tqdm import tqdm



def save_bytes(fname: str, data):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as fout:
        if isinstance(data, str):
            data = data.encode('utf8')
        fout.write(data)

def create_labels(dir: str, ext='.dcm'):

    labels = None
    classes_to_idx = None

    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    files = glob(os.path.join(dir, f'**/*{ext}'), recursive=True)
    assert len(files) > 0, print(f"There is no file with a {ext} extension")

    labels = []
    for file in tqdm(files):
        for cls in classes:

            if os.path.join(dir, cls) in file:
                labels.append([file.replace(dir, ''), classes_to_idx[cls]])
                break

    return labels, classes_to_idx


def save_labels(dir: str, ext='.dcm'):

    labels, classes_to_idx = create_labels(dir, ext)

    metadata = {
                'labels': labels if all(x is not None for x in labels) else None,
                'classes': classes_to_idx 
                }
    save_bytes(os.path.join(dir, 'dataset.json'), json.dumps(metadata))
