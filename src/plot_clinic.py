import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PlotClinic:
    def __init__(self, df:pd.DataFrame, verbose:bool=True):
        self.df = df
        self.verbose = verbose