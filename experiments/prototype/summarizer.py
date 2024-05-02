
import json
from transformers import

if __name__ == '__main__':
    fn = "./TV.json"

    with open(fn) as f:
        data = json.load(f)

    print(data[0])