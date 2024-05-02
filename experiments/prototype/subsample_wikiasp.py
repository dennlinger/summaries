
import json

import numpy as np


def subsample_data(data, fn):

    consideration = []
    for sample in data:
        if len(sample["targets"]) > 2:
            print([el[0] for el in sample["targets"]])
            if all([len(el[1]) > 0 for el in sample["targets"]]):
                consideration.append(sample)
    print(len(consideration))

    np.random.seed(484839)
    subset = list(np.random.choice(consideration, 40, replace=False))

    new_data = {}
    for el in subset:
        curr_inputs = []
        for s in el["inputs"]:
            if len(s) >= 25:
                curr_inputs.append(s)

        new_data[el["exid"]] = {
            "inputs": curr_inputs,
            "targets": el["targets"]
        }

    with open(fn, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    with open("../../../Downloads/TelevisionShow/test.jsonl", "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    subsample_data(data, "./TV_filtered.json")

    with open("../../../Downloads/Company/test.jsonl", "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    subsample_data(data, "./Company_filtered.json")
