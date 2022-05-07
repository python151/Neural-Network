import networkx as nx
import pickle
import matplotlib.pyplot as plt

with open("top_network.obj", "rb") as f:
  network = pickle.load(f)


edge_width = [0.5 * network.network[u][v]['weight'] for u, v in network.network.edges()]
nx.draw_networkx(network.network, width=edge_width, pos=nx.spring_layout(network.network))

print(f"Number of edges: {len(edge_width)}")

plt.savefig("Graph.png", format="PNG")
plt.show()