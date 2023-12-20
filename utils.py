import pandas
import numpy
from sklearn.linear_model import LinearRegression
from typing import List 
import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from scipy.stats import pearsonr


def plot_graph( G ):
    pos = nx.circular_layout(G)  
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=15, width=2.5, alpha=0.9, edge_color="gray")
    plt.title("Causal Graph")
    
    plt.show()    


def partial_pearson( data: pandas.DataFrame, 
                    x: str, 
                    y: str, 
                    Z : List[str] = [] ,
                    p_value_threshold: float = 0.05):
    
    for elm in Z:
        assert elm in data
    assert x in data
    assert y in data 

    assert x not in Z 
    assert y not in Z

    if len(Z) == 0:
        corr = pearsonr( data[x], data[y] )

    else:
        Z_data = data[Z]
        x_data = data[x]
        y_data = data[y]
        
        model = LinearRegression()
        x_bar = model.fit(Z_data,x_data).predict(Z_data) - x_data
    
        model = LinearRegression()
        y_bar = model.fit(Z_data,y_data).predict(Z_data) - y_data
    
        corr = pearsonr( x_bar, y_bar )

    significant = corr.pvalue < p_value_threshold

    return {'magnitude': corr.correlation, 'pvalue': corr.pvalue, 'is_associated': significant } 

def adjacency_to_graph(adj_matrix, columns):
    """
    Convert an adjacency matrix to a NetworkX graph.
    
    Parameters:
        adj_matrix (np.array): Adjacency matrix.
        columns (list): Column names corresponding to the nodes of the graph.

    Returns:
        G (networkx.Graph): Graph represented by the adjacency matrix.
    """
    if not isinstance(adj_matrix, np.ndarray):
        raise ValueError("adj_matrix must be a numpy ndarray.")
    if len(adj_matrix) != len(columns):
        raise ValueError("The number of columns must match the size of the adjacency matrix.")
    
    G = nx.DiGraph()  # Assuming the graph is directed; change to nx.Graph() for undirected
    G.add_nodes_from(columns)
    
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] != 0:  # Adjust condition if needed
                G.add_edge(columns[i], columns[j])
    return G

