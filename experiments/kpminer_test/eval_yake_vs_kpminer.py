"""
Evaluation script, computing F1@5 and @10 for the predictions.
"""
from typing import Set, Tuple
import json

from tqdm import tqdm
import numpy as np

def get_macro_scores(prediction, reference) -> Tuple[float, float, float]:
    prediction = set(prediction)
    reference = set(reference)
    prec = macro_precision(prediction, reference)
    rec = macro_recall(prediction, reference)
    if np.isclose(prec, 0.0) and np.isclose(rec, 0.0):
        f1 = 0.0
    else:
        f1 = macro_f1(prec, rec)

    return prec, rec, f1


def macro_precision(prediction: Set, reference: Set):
    return len(prediction.intersection(reference)) / len(prediction)


def macro_recall(prediction: Set, reference: Set):
    return len(prediction.intersection(reference)) / len(reference)


def macro_f1(prec: float, rec: float):
    return 2 * (prec * rec) / (prec + rec)


def print_scores(predictions, lines, name, k=10):
    all_prec = []
    all_rec = []
    all_f1 = []

    for sample, reference_line in tqdm(zip(predictions, lines)):
        reference = json.loads(reference_line)
        # Slightly different output format
        if name == "yake":
            prediction = sample[name]
        elif name == "kp-miner":
            prediction = [el[0] for el in sample[name]]
        else:
            raise ValueError("Incorrect name of algorithm specified!")
        # Limit to top k predictions
        prediction = prediction[:k]

        # Compute scores and assign
        macro_scores = get_macro_scores(prediction, reference["keyword"].split(";"))
        all_prec.append(macro_scores[0])
        all_rec.append(macro_scores[1])
        all_f1.append(macro_scores[2])

    print(f"{name} macro scores @{k}:")
    print(f"Precision: {np.mean(all_prec) * 100:.2f}%")
    print(f"Recall:    {np.mean(all_rec) * 100:.2f}%")
    print(f"F1:        {np.mean(all_f1) * 100:.2f}%")


if __name__ == '__main__':
    with open("predictions.json") as f:
        predictions = json.load(f)
    with open("kp20k_testing.json") as f:
        lines = f.readlines()

    if len(lines) != len(predictions):
        raise AssertionError("Length of lines and predictions not equal!")

    print_scores(predictions, lines, name="yake", k=10)
    print_scores(predictions, lines, name="kp-miner", k=10)

    print_scores(predictions, lines, name="yake", k=5)
    print_scores(predictions, lines, name="kp-miner", k=5)


