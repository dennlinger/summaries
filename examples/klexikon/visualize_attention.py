"""
Main script to visualize attention on Klexikon articles with mT5.
"""
from datasets import load_dataset

if __name__ == '__main__':
    shortest_article_ids = [260, 1301, 2088, 665, 1572, 436, 1887, 1422, 1506, 474]

    dataset = load_dataset("dennlinger/klexikon")

    for idx in shortest_article_ids:
        sample = dataset["train"][idx]

