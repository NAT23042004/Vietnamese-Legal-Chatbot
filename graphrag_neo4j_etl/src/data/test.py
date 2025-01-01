import pandas as pd
import sys

graph = pd.read_csv("graph-export.csv")
node = pd.read_csv("node-export.csv")
relationship = pd.read_csv("relationship-export.csv")

print(graph['noidung'])
print(node['name'].head().info)
