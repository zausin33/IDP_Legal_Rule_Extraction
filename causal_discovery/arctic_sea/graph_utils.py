import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx


def create_graph(node_names, adj_matrix, gt_G: nx.DiGraph = None, do_draw: bool = True):
    """
    Creates a graph from a list of node names and an adjacency matrix.
    """
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(node_names)

    # Add edges to the graph based on the adjacency matrix
    num_nodes = len(node_names)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] != 0:  # Change this to 'if adj_matrix[i][j] != 0:' if using a 2D list
                G.add_edge(node_names[i], node_names[j])

    if do_draw:
        draw_graph(G, gt_G)

        # Show the plot
        plt.show()
    return G


def structural_hamming_distance(G1, G2):
    # Ensure that both graphs have the same set of nodes
    if set(G1.nodes) != set(G2.nodes):
        raise ValueError("The two graphs must have the same set of nodes")

    # Calculate the symmetric difference in edges between the two graphs
    edges_difference = set(G1.edges) ^ set(G2.edges)

    # The Structural Hamming Distance is the size of the symmetric difference
    return len(edges_difference)


def nhd(G1, G2):
    # Calculate Structural Hamming Distance
    shd = structural_hamming_distance(G1, G2)

    # Number of nodes
    n = G1.number_of_nodes()

    # Calculate Normalized Hamming Distance
    return round(shd / (n * n), 2)


def nhd_baseline(G1, G2):
    # Calculate the worst-case SHD (sum of number of edges in both graphs)
    shd_worst_case = G1.number_of_edges() + G2.number_of_edges()

    # Number of nodes
    n = G1.number_of_nodes()

    # Calculate the NHD baseline
    return round(shd_worst_case / (n * n), 2)


def nhd_ratio(G1, G2):
    # Calculate NHD and NHD baseline
    actual_nhd = nhd(G1, G2)
    baseline_nhd = nhd_baseline(G1, G2)

    # Calculate NHD ratio
    return round(actual_nhd / baseline_nhd, 2)


def _create_edges_abcd(llm_input, parsed_llm_output, name_variable_mapping, bidirectional=False):
    edges = []
    for el, el_res in zip(llm_input, parsed_llm_output):
        if el_res == "A":
            edges.append((name_variable_mapping[el['X']], name_variable_mapping[el['Y']]))
        elif el_res == "B":
            edges.append((name_variable_mapping[el['Y']], name_variable_mapping[el['X']]))
        elif el_res == "C":
            if bidirectional:
                edges.append((name_variable_mapping[el['X']], name_variable_mapping[el['Y']]))
                edges.append((name_variable_mapping[el['Y']], name_variable_mapping[el['X']]))
            else:
                pass
        elif el_res == "D":
            pass
        else:
            print("Error", el, el_res)

    return edges


def _create_edges_yes_no(llm_input, parsed_llm_output, name_variable_mapping):
    edges = []
    for el, el_res in zip(llm_input, parsed_llm_output):
        if el_res.lower() == "yes":
            edges.append((name_variable_mapping[el['X']], name_variable_mapping[el['Y']]))
    return edges


def draw_graph(graph: nx.DiGraph, gt_G: nx.DiGraph=None):
    plt.figure(figsize=(10, 6))
    graph = graph.copy()

    if gt_G is not None:
        missing_edges = set(gt_G.edges) - set(graph.edges)
        correct_edges = set(gt_G.edges) & set(graph.edges)
        wrong_edges = set(graph.edges) - set(gt_G.edges)
    else:
        missing_edges = []
        correct_edges = set(graph.edges)
        wrong_edges = []

    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=500)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edgelist=missing_edges, style='dotted', edge_color="lightblue")
    nx.draw_networkx_edges(graph, pos, edgelist=wrong_edges, edge_color="red")
    nx.draw_networkx_edges(graph, pos, edgelist=correct_edges, edge_color='black')

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_weight='bold')

    # Create a legend
    missing_line = mlines.Line2D([], [], color='lightblue', linestyle='dotted', label='Missing Edges')
    wrong_line = mlines.Line2D([], [], color='red', label='Wrong Edges')
    correct_line = mlines.Line2D([], [], color='black', label='Correct Edges')

    plt.legend(handles=[missing_line, wrong_line, correct_line], loc='upper left')

    plt.show()


def create_graph_from_llm_output(parsed_llm_output, llm_input, name_variable_mapping, edges_yes_no=False, bidirectional=False, gt_G: nx.DiGraph=None):
    if edges_yes_no:
        edges = _create_edges_yes_no(llm_input, parsed_llm_output, name_variable_mapping)
    else:
        edges = _create_edges_abcd(llm_input, parsed_llm_output, name_variable_mapping, bidirectional=bidirectional)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    G.add_nodes_from(name_variable_mapping.values())
    G.add_edges_from(edges)

    draw_graph(G, gt_G)

    # Show the plot
    plt.show()
    return G


def evaluate(gt_G, G):
    nodes = list(gt_G.nodes)
    # all possible edges
    all_edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(len(nodes)) if i != j]
    # edges in gt_G
    gt_edges = list(gt_G.edges)
    # edges in G
    edges = list(G.edges)
    # true positive
    tp = len(set(gt_edges).intersection(set(edges)))
    # false positive
    fp = len(set(edges).difference(set(gt_edges)))
    # false negative
    fn = len(set(gt_edges).difference(set(edges)))
    # true negative
    tn = len(set(all_edges).difference(set(gt_edges)).difference(set(edges)))
    # accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # recall
    recall = tp / (tp + fn)
    # precision
    precision = tp / (tp + fp)
    # f1
    f1 = 2 * recall * precision / (recall + precision)

    print("NHD:", nhd(gt_G, G))
    print("NHD Baseline:", nhd_baseline(gt_G, G))
    print("NHD Ratio:", nhd_ratio(gt_G, G))
    print("Number of edges:", G.number_of_edges())
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1:", f1)

