# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:14:50 2017

@author: Fina
"""

def get_data_frame():
	import pandas as pd

	path = "Dataset/CencusIncome.data.txt"

	df = pd.read_csv(path)
	return df

