#Exploratory Data Analysis 1
#round-2-networks exploring the semantic space
#imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


input_filepath = "Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory"
input_file = "round2networks-2026.xlsx"
input = os.path.join(input_filepath, input_file)
raw = pd.read_excel(input)

