import pandas as pd
import matplotlib.pyplot as plt
from fire import Fire

FIGURES_DIR = "figures"

def main(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

if __name__ == "__main__":
    Fire(main)
