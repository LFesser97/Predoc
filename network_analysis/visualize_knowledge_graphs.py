"""
visualize_knowledge_graphs.py

Created on Sun Mar 19 2023

@author: Lukas

This file contains all functions for visualizing knowledge graphs.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt


# functions for visualizing knowledge graphs

def visualize_knowledge_graph(knowledge_graph: nx.DiGraph) -> None:
    """
    This function visualizes a knowledge graph.

    Parameters
    ----------
    knowledge_graph : nx.DiGraph
        The knowledge graph.

    Returns
    -------
    None.
    """

    # color the nodes according to their type
    node_colors = []
    node_legend = {}

    for node in knowledge_graph.nodes():
        if knowledge_graph.nodes[node]["type"] == "company":
            node_colors.append("orange")
            node_legend.append(("Company", "orange"))

        elif knowledge_graph.nodes[node]["type"] == "Bank/ Individual":
            node_colors.append("lightblue")
            node_legend.append(("Bank/ Individual", "lightblue"))

        else:
            node_colors.append("black")
            print("The node type is not known.")

    # color the edges according to their type
    edge_colors = []
    edge_legend = {}

    for edge in knowledge_graph.edges():
        if knowledge_graph.edges[edge]["type"] == "supplier":
            edge_colors.append("red")
            edge_legend.append(("Supplier", "red"))

        elif knowledge_graph.edges[edge]["type"] == "shareholder":
            edge_colors.append("blue")
            edge_legend.append(("Shareholder", "blue"))

        else:
            edge_colors.append("black")
            print("The edge type is not known.")

    # creat a new pos, make sure edges don't overlap with nodes or legends

    pos = nx.spring_layout(knowledge_graph)
    pos.update((n, (x, y+0.1)) for n, (x, y) in pos.items())
    pos.update((n, (x, y-0.1)) for n, (x, y) in pos.items())

    # draw the graph

    nx.draw(knowledge_graph, pos, with_labels=True, node_size=500, node_color=node_colors, edge_color=edge_colors)
    labels = nx.get_edge_attributes(knowledge_graph, 'weight')
    nx.draw_networkx_edge_labels(knowledge_graph, pos, edge_labels=labels, label_pos=0.3, font_size=8)

    # add two legends, one for the edge colors and one for the node colors -> WILL NEED TO GENERALIZE THIS

    edge_legend = plt.legend([plt.Line2D([0], [0], color=element[1], lw=4) for element in edge_legend], [element[0] for element in edge_legend], loc="upper left", title="Edge Color")
    node_legend = plt.legend([plt.Line2D([0], [0], color=element[1], lw=4) for element in node_legend], [element[0] for element in node_legend], loc="lower right", title="Node Color")
    plt.gca().add_artist(edge_legend)
    plt.show()