import networkx as nx

import matplotlib.pyplot as plt 

from dataclasses import dataclass, field
import random


@dataclass
class Edge:
    """An edge of the graph.

    Args:
        travel_time (float): The time it takes to travel the edge. A high value indicates more traffic.
        pheromones: (float): The amount of pheromones deposited by the ants on the edge. Defaults to 1.0.
    """

    travel_time: float
    pheromones: float = 1.0


@dataclass
class Node:
    """A node in the graph.

    Args:
        id (str): The unique ID of the node.
        edges (dict[str, Edge]): Stores all the outgoing edges from this node.
    """

    id: str
    edges: dict[str, Edge] = field(default_factory=dict)

    def add_edge(self, destination: str, travel_time: float):
        self.edges[destination] = Edge(travel_time)


@dataclass
class Graph:
    """A Directed Graph made up of Nodes and Edges.

    Args:
        graph (dict[str, Node]): The actual graph.
        evaporation_rate (float): The evaporation rate of the pheromones. Defaults to 0.1
    """

    graph: dict[str, Node] = field(default_factory=dict)
    evaporation_rate: float = 0.5

    def node_exists(self, id: str) -> bool:
        """Checks if the node exists in the graph.

        Args:
            id (str): The ID of the node.

        Returns:
            bool: True if the node exists in the graph, else False.
        """
        return id in self.graph

    def edge_exists(self, source: str, destination: str) -> bool:
        """Checks if the edge exists in the graph.

        Args:
            source (str): The ID of the source node.
            destination (str): The ID of the destination node.

        Returns:
            bool: True if the edge exists in the graph, else False.
        """
        if not self.node_exists(source) or not self.node_exists(destination):
            return False
        return destination in self.graph[source].edges.keys()

    def add_node(self, id: str) -> None:
        """Add a node to the graph.

        Args:
            id (str): The ID of the node to be added to the graph.
        """
        self.graph[id] = Node(id)

    def add_edge(self, source: str, destination: str, travel_time: float) -> None:
        """Add an edge connecting 2 nodes in the graph.

        Args:
            source (str): The source node of the edge.
            destination (str): The destination node of the edge.
            travel_time (float): The amount of time it takes to travel that edge.
        """
        if not self.node_exists(source):
            self.add_node(source)
        if not self.node_exists(destination):
            self.add_node(destination)
        self.graph[source].add_edge(destination, travel_time)

    def get_all_nodes(self) -> list[str]:
        """Returns a list of all the nodes in the graph.

        Returns:
            list[str]: A list of all the nodes in the graph.
        """
        return list(self.graph.keys())

    def get_all_edges(self) -> list[Edge]:
        """Returns all the edges in the graph.

        Returns:
            list[Tuple[str, str, float]]: A list of Tuples -> (source, destination, travel_time)
        """
        edges: list[Edge] = []
        for source in self.graph:
            for destination in self.graph[source].edges:
                edges.append(self.get_edge(source, destination))
        return edges

    def get_edge(self, source: str, destination: str) -> Edge:
        """Returns the Edge if it exists in the graph.

        Args:
            source (str): The ID of the source node.
            destination (str): The ID of the destination node.

        Returns:
            Edge: The Edge object.
        """
        return self.graph[source].edges[destination]

    def get_node_edges(self, id: str) -> dict[str, Edge]:
        """Returns all the edges of a node.

        Args:
            id (str): The ID of the node.

        Returns:
            dict[str, Edge]: The outgoing edges of the node.
        """
        return self.graph[id].edges

    def get_node(self, id: str) -> Node | None:
        """Returns the Node if it exists in the graph.

        Args:
            id (str): The ID of the Node.

        Returns:
            Optional[Node]: The Node if it exists otherwise returns None.
        """
        if self.node_exists(id):
            return self.graph[id]
        return None

    def get_neighbors(self, id: str) -> list[str]:
        """Returns all the neighbors of a Node in the graph.

        Args:
            id (str): The ID of the Node.

        Returns:
            list[str]: A list of all the neighbors of that Node.
        """
        if not self.node_exists(id):
            return []
        return [neighbor for neighbor in self.graph[id].edges]

    def get_travel_times(self, id: str) -> list[float]:
        """Returns a list of travel times of all the edges of the Node.

        Args:
            id (str): The ID of the Node.

        Returns:
            list[float]: A list of travel times of all the edges of the Node.
        """
        if not self.node_exists(id):
            return []
        travel_times = []
        for _, edge in self.graph[id].edges.items():
            travel_times.append(edge.travel_time)
        return travel_times

    def get_edge_travel_time(self, source: str, destination: str) -> float:
        """Returns the travel time of the specified edge if it exists.

        Args:
            source (str): The source node of the edge.
            destination (str): The destination of the edge.

        Returns:
            float: The travel time of that edge.
        """
        if (
            not self.node_exists(source)
            or not self.node_exists(destination)
            or destination not in self.graph[source].edges
        ):
            return float("inf")
        return self.graph[source].edges[destination].travel_time

    def compute_path_travel_time(self, path: list[str]) -> float:
        """Computes the cost of a path (a list of node IDs).

        Args:
            path (list[str]): The ID of the nodes in the path.

        Returns:
            float: The total travel time of the specified path.
        """
        cost = 0.0
        for i in range(len(path) - 1):
            if self.edge_exists(path[i], path[i + 1]):
                cost += self.get_edge_travel_time(path[i], path[i + 1])
            else:
                return float("inf")
        return cost

    def evaporate(self) -> None:
        """Evaporates the pheromone values of all the edges given the evaporation parameter (rho)."""
        for node_id, node in self.graph.items():
            for neighbor, edge in self.graph[node_id].edges.items():
                edge.pheromones = (1 - self.evaporation_rate) * edge.pheromones

    def deposit_pheromones_on_edge(
        self, source: str, destination: str, new_pheromones: float
    ) -> None:
        """Updates the pheromones on an edge in the graph.

        Args:
            source (str): The source node of the edge.
            destination (str): The destination node of the edge.
            new_pheromones (float): The amount of pheromones to be added to the existing value on the edge.
        """
        self.graph[source].edges[destination].pheromones += new_pheromones

    def deposit_pheromones_along_path(self, path: list[str]) -> None:
        """Updates the pheromones along all the edges in the path.

        Args:
            path (list[str]): The path followed by the ant.
        """
        path_cost = self.compute_path_travel_time(path)
        for i in range(len(path) - 1):
            self.deposit_pheromones_on_edge(path[i], path[i + 1], 1 / path_cost)

    def normalize_graph_for_dijkstra(self) -> dict[str, dict[str, float]]:
        """Normalizes the graph for the Dijkstra's algorithm implementation.

        Returns:
            dict[str, dict[str, float]]: A simple, dictionary-structured graph with only the travel times of the edges.
        """
        dijkstra_graph: dict[str, dict[str, float]] = {}
        for node in self.get_all_nodes():
            dijkstra_graph[node] = {}
            for edge in self.graph[node].edges:
                dijkstra_graph[node][edge] = self.graph[node].edges[edge].travel_time
        return dijkstra_graph

    def update_edge_travel_time(self, edge: Edge, new_travel_time: float) -> None:
        """Updates the travel time of an edge in the graph.

        Args:
            edge (Edge): The Edge object.
            new_travel_time (float): The new travel time.
        """
        if new_travel_time <= 0:
            new_travel_time = 1
        edge.travel_time = new_travel_time

    def update_edges_travel_time(
        self, max_delta_time: int = 2, update_probability: float = 0.7
    ) -> None:
        """Stochastically updates the travel time of the edges in the graph.

        Args:
            max_delta_time (int, optional): The maximum allowed change in travel time of an edge (in positive or negative direction). Defaults to 2.
            update_probability (float, optional): The probability that the travel time of an edge will be updated. Defaults to 0.7.
        """
        for edge in self.get_all_edges():
            if random.random() > update_probability:
                continue
            delta_time = random.choice(
                [i for i in range(-max_delta_time, max_delta_time + 1, 1) if i != 0]
            )
            self.update_edge_travel_time(edge, edge.travel_time + delta_time)

    def __str__(self) -> str:
        """Displays the graph.

        Returns:
            str: The string representation of the graph.
        """
        display = []
        for node_id, node in self.graph.items():
            display.append("---")
            display.append(f"Node {node.id}")
            display.append("")
            display.append("Edges:")
            for edge_id, edge in node.edges.items():
                display.append(
                    f"{node_id} -> {edge_id}, Travel Time: {edge.travel_time}, Pheromones: {edge.pheromones}"
                )
        return "\n".join(display)


@dataclass
class Ant:
    """A class for an Ant that traverses the graph.

    Args:
        graph (Graph): The Graph object.
        source (str): The source node of the ant.
        destination (str): The destination node of the ant.
        alpha (float): The amount of importance given to the pheromone by the ant. Defaults to 0.9.
        beta (float): The amount of importance given to the travel time value by the ant. Defaults to 0.1.
        visited_nodes (Set): A set of nodes that have been visited by the ant.
        path (list[str]): A list of node IDs of the path taken by the ant so far.
        is_fit (bool): A flag which indicates if the ant has reached the destination (fit) or not (unfit). Defaults to False.
    """

    graph: Graph
    source: str
    destination: str
    alpha: float = 0.5
    beta: float = 0.5
    visited_nodes: set = field(default_factory=set)
    path: list[str] = field(default_factory=list)
    is_fit: bool = False

    def __post_init__(self) -> None:
        self.current_node = self.source
        self.path.append(self.source)

    def reached_destination(self) -> bool:
        """Checks if the ant has reached the destination node in the graph.

        Returns:
            bool: True, if the ant has reached the destination.
        """
        return self.current_node == self.destination

    def _get_unvisited_neighbors(
        self, all_neighbors: dict[str, Edge]
    ) -> dict[str, Edge]:
        """Returns a subset of the all the neighbors of the node which are unvisited.

        Args:
            all_neighbors (dict[str, Edge]): A set of all neighbors of the node.

        Returns:
            dict[str, Edge]: A subset of all the unvisited neighbors.
        """
        unvisited_neighbors = {}
        for neighbor, edge in all_neighbors.items():
            if neighbor in self.visited_nodes:
                continue
            unvisited_neighbors[neighbor] = edge
        return unvisited_neighbors

    @staticmethod
    def _calculate_edges_total(
        unvisited_neighbors: dict[str, Edge], alpha: float, beta: float
    ) -> float:
        """Computes the denominator of the transition probability equation for the ant.

        Args:
            unvisited_neighbors (dict[str, Edge]): A set of unvisited neighbors of the current node.
            alpha (float): [description]: The alpha value.
            beta (float): [description]: The beta value.

        Returns:
            float: The summation of all the outgoing edges (to unvisited nodes) from the current node.
        """
        total = 0.0
        for neighbor, edge in unvisited_neighbors.items():
            total += (edge.pheromones**alpha) * ((1 / edge.travel_time) ** beta)
        return total

    @staticmethod
    def _calculate_edge_probabilites(
        unvisited_neighbors: dict[str, Edge], alpha: float, beta: float, total: float
    ) -> dict[str, float]:
        """Computes the transition probabilities of all the edges from the current node.

        Args:
            unvisited_neighbors (dict[str, Edge]): A set of unvisited neighbors of the current node.
            alpha (float): [description]: The alpha value.
            beta (float): [description]: The beta value.
            total (float): [description]: The summation of all the outgoing edges (to unvisited nodes) from the current node.

        Returns:
            dict[str, float]: A dictionary mapping node IDs to their transition probabilities.
        """
        probabilities = {}
        for neighbor, edge in unvisited_neighbors.items():
            probabilities[neighbor] = (
                (edge.pheromones**alpha) * ((1 / edge.travel_time) ** beta)
            ) / total
        return probabilities

    @staticmethod
    def _sort_edge_probabilites(probabilities: dict[str, float]):
        """Sorts the probabilities of the edges in descending order.

        Args:
            probabilities (dict[str, float]): A dictionary mapping the node IDs to their transition probabilities.

        Returns:
            [type]: A sorted dictionary mapping node IDs to their transition probabilities.
        """
        return {
            k: v for k, v in sorted(probabilities.items(), key=lambda item: -item[1])
        }

    @staticmethod
    def _choose_neighbor_using_roulette_wheel(
        sorted_probabilities: dict[str, float]
    ) -> str:
        """Chooses the next node to be visited using the Roulette Wheel selection technique.

        Args:
            sorted_probabilities (dict[str, Edge]): A sorted dictionary mapping node IDs to their transition probabilities.

        Returns:
            str: The ID of the next node to be visited by the ant.
        """
        pick = random.uniform(0, sum(sorted_probabilities.values()))
        current = 0.0
        next_node = ""
        for key, value in sorted_probabilities.items():
            current += value
            if current > pick:
                next_node = key
                break

        return next_node

    def _pick_next_node(
        self, unvisited_neighbors: dict[str, Edge], alpha: float, beta: float
    ) -> str:
        """Chooses the next node to be visited by the ant using the Roulette Wheel selection technique.

        Args:
            unvisited_neighbors (dict[str, Edge]): A set of unvisited neighbors of the current node.
            alpha (float): [description]: The alpha value.
            beta (float): [description]: The beta value.

        Returns:
            str: The ID of the next node to be visited by the ant.
        """

        edges_total = self._calculate_edges_total(unvisited_neighbors, alpha, beta)
        probabilities = self._calculate_edge_probabilites(
            unvisited_neighbors, edges_total, alpha, beta
        )
        sorted_probabilities = self._sort_edge_probabilites(probabilities)
        return self._choose_neighbor_using_roulette_wheel(sorted_probabilities)

    def take_step(self) -> None:
        """This method allows the ant to travel to a neighbor of the current node in the graph."""
        # Mark the node as visited.
        self.visited_nodes.add(self.current_node)

        # Find all the neighboring nodes of the current node.
        all_neighbors = self.graph.get_node_edges(self.current_node)

        # Check if the current node has no neighbors (isolated node).
        if len(all_neighbors) == 0:
            return

        # Find unvisited neighbors of the current node.
        unvisited_neighbors = self._get_unvisited_neighbors(all_neighbors)

        # Pick the next node based on the Roulette Wheel selection technique.
        next_node = self._pick_next_node(unvisited_neighbors, self.alpha, self.beta)

        if not next_node:
            return

        self.path.append(next_node)
        self.current_node = next_node


@dataclass
class ACO:
    graph: Graph

    def _forward_ants(self, ants: list[Ant], max_iterations: int) -> None:
        """Deploys forward search ants in the graph.

        Args:
            ants (list[Ant]): A list of Ants.
            max_iterations (int, optional): The maximum number of steps an ant is allowed is to take in order to reach the destination.
                If it fails to find a path, it is tagged as unfit. Defaults to 50.
        """
        for idx, ant in enumerate(ants):
            for i in range(max_iterations):
                if ant.reached_destination():
                    ant.is_fit = True
                    break
                ant.take_step()

    def _backward_ants(self, ants: list[Ant]) -> None:
        """Sends the ants (which are fit) backwards towards the source while they drop pheromones on the path.

        Args:
            ants (list[Ant]): A list of Ants.
        """
        for idx, ant in enumerate(ants):
            if ant.is_fit:
                self.graph.deposit_pheromones_along_path(ant.path)

    def _deploy_search_ants(
        self,
        source: str,
        destination: str,
        num_ants: int,
        random_spawns: bool = False,
        cycles: int = 100,
        max_iterations: int = 50,
    ) -> None:
        """Deploys search ants which traverse the graph to find the shortest path.

        Args:
            source (str): The source node in the graph.
            destination (str): The destination node in the graph.
            num_ants (int): The number of ants to be spawned.
            random_spawns (bool): A flag to determine if the ants should be spawned at random nodes or always at the source node.
            cycles (int, optional): The number of cycles of generating and deploying ants (forward and backward). Defaults to 100.
            max_iterations (int, optional): The maximum number of steps an ant is allowed is to take in order to reach the destination.
                If it fails to find a path, it is tagged as unfit. Defaults to 50.
        """
        for cycle in range(cycles):
            ants: list[Ant] = []
            for _ in range(num_ants):
                spawn_point = (
                    random.choice(self.graph.get_all_nodes())
                    if random_spawns
                    else source
                )
                ants.append(Ant(self.graph, spawn_point, destination))
            self._forward_ants(ants, max_iterations)
            self.graph.evaporate()
            self._backward_ants(ants)

    def _deploy_solution_ant(
        self,
        source: str,
        destination: str,
        max_iterations: int = 1_000,
    ) -> list[str]:
        """Deploys the final ant that greedily w.r.t. the pheromones finds the shortest path from the source to the destination.

        Args:
            source (str): The source node in the graph.
            destination (str): The destination node in the graph.

        Returns:
            list[str]: The shortest path found by the ants (A list of node IDs).
        """
        # Spawn an ant which favors pheromone values over edge costs.

        paths = []

        for _ in range(max_iterations):
            ant = Ant(self.graph, source, destination, alpha=0.999, beta=0.001)
            for __ in range(1_000):
                if not ant.reached_destination():
                    # print(ant.path)
                    ant.take_step()
                else:
                    if ant.path not in paths:
                        paths.append(ant.path)
                        break

        current_minimum = float("inf")
        shortest_path = []
        for path in paths:
            travel_time = self.graph.compute_path_travel_time(path)
            if travel_time < current_minimum:
                current_minimum = travel_time
                shortest_path = path

        return shortest_path

    def find_shortest_path(
        self, source: str, destination: str
    ) -> tuple[list[str], float]:
        """Finds the shortest path from the source to the destination in the graph using the traditional Ant Colony Optimization technique.

        Args:
            source (str): The source node in the graph.
            destination (str): The destination node in the graph.

        Returns:
            list[str]: The shortest path found by the ants (A list of node IDs).
            float: The total travel time of the shortest path.
        """
        self._deploy_search_ants(
            source,
            destination,
            num_ants=200,
            random_spawns=False,
            cycles=100,
            max_iterations=50,
        )
        shortest_path = self._deploy_solution_ant(source, destination)
        return shortest_path, self.graph.compute_path_travel_time(shortest_path)


graph = Graph()

graph.add_edge("2", "1", travel_time=1655)
graph.add_edge("2", "3", travel_time=3230)
graph.add_edge("2", "11", travel_time=2367)
graph.add_edge("2", "10", travel_time=1368)
graph.add_edge("3", "2", travel_time=3230)
graph.add_edge("3", "10", travel_time=3230)
graph.add_edge("4", "3", travel_time=1213)
graph.add_edge("5", "3", travel_time=2472)
graph.add_edge("5", "6", travel_time=152)
graph.add_edge("6", "7", travel_time=2801)
graph.add_edge("6", "20", travel_time=1319)
graph.add_edge("7", "6", travel_time=2801)
graph.add_edge("7", "8", travel_time=867)
graph.add_edge("7", "19", travel_time=834)
graph.add_edge("9", "8", travel_time=201)
graph.add_edge("9", "10", travel_time=857)
graph.add_edge("9", "19", travel_time=1203)
graph.add_edge("10", "2", travel_time=1368)
graph.add_edge("10", "3", travel_time=2541)
graph.add_edge("10", "9", travel_time=857)
graph.add_edge("11", "2", travel_time=2367)
graph.add_edge("11", "12", travel_time=2368)
graph.add_edge("11", "15", travel_time=1409)
graph.add_edge("11", "19", travel_time=1653)
graph.add_edge("12", "13", travel_time=711)
graph.add_edge("12", "14", travel_time=541)
graph.add_edge("15", "11", travel_time=1409)
graph.add_edge("15", "14", travel_time=1678)
graph.add_edge("15", "16", travel_time=2144)
graph.add_edge("15", "18", travel_time=1487)
graph.add_edge("16", "15", travel_time=2144)
graph.add_edge("16", "17", travel_time=1585)
graph.add_edge("16", "23", travel_time=2557)
graph.add_edge("17", "16", travel_time=1585)
graph.add_edge("17", "18", travel_time=477)
graph.add_edge("17", "25", travel_time=156)
graph.add_edge("18", "15", travel_time=1487)
graph.add_edge("18", "17", travel_time=477)
graph.add_edge("18", "19", travel_time=870)
graph.add_edge("19", "20", travel_time=3977)
graph.add_edge("19", "18", travel_time=870)
graph.add_edge("19", "11", travel_time=1653)
graph.add_edge("19", "9", travel_time=1203)
graph.add_edge("19", "7", travel_time=834)
graph.add_edge("20", "21", travel_time=2945)
graph.add_edge("20", "23", travel_time=3054)
graph.add_edge("20", "19", travel_time=3977)
graph.add_edge("20", "6", travel_time=1319)
graph.add_edge("23", "20", travel_time=3054)
graph.add_edge("23", "22", travel_time=2561)
graph.add_edge("23", "16", travel_time=2557)
graph.add_edge("24", "23", travel_time=984)
graph.add_edge("24", "25", travel_time=2519)

source = "4"
destination = "8"

# print(graph)
aco = ACO(graph)

aco_path1, aco_cost1 = aco.find_shortest_path('5', '14')
aco_path2, aco_cost2 = aco.find_shortest_path('5', '1')
aco_path3, aco_cost3 = aco.find_shortest_path('5', '22')
aco_path4, aco_cost4 = aco.find_shortest_path('5', '13')
aco_path5, aco_cost5 = aco.find_shortest_path('5', '21')
aco_path6, aco_cost6 = aco.find_shortest_path('5', '25')
aco_path7, aco_cost7 = aco.find_shortest_path('5', '8')


print(f"ACO - path: {aco_path1}, cost: {aco_cost1}")
print(f"ACO - path: {aco_path2}, cost: {aco_cost2}")
print(f"ACO - path: {aco_path3}, cost: {aco_cost3}")
print(f"ACO - path: {aco_path4}, cost: {aco_cost4}")
print(f"ACO - path: {aco_path5}, cost: {aco_cost5}")
print(f"ACO - path: {aco_path6}, cost: {aco_cost6}")
print(f"ACO - path: {aco_path7}, cost: {aco_cost7}")




# Create graph('2', '1')
G = nx.Graph()


# Add Nodes 
nodes = [
    '1', '2', '3', '4', '5', '6', '7',
    '8', '9', '10', '11', '12', '13', '14',
    '15', '16', '17', '18', '19', '20', '21',
    '22', '23', '24', '25'
]
G.add_nodes_from(nodes)



node_colors = {}

change_nodes = ['4', '24', '5']
for i in change_nodes:
    node_colors[i] = "green"


change_nodes2 = ['8', '22',]
for i in change_nodes2:
    node_colors[i] = "orange"

change_nodes3 = ['1','14','13','25','21']
for i in change_nodes3:
    node_colors[i] = 'pink'
# Extract and add edges
edges = [
    ('2', '1'), ('2', '3'), ('2', '11'), ('2', '10'), ('3', '2'), ('3', '10'), ('4', '3'), ('5', '3'), ('5', '6'), ('6', '7'), ('6', '20'), ('7', '6'), ('7', '8'), ('7', '19'), ('9', '8'), ('9', '10'), ('9', '19'), ('10', '2'), ('10', '3'), ('10', '9'), ('11', '2'), ('11', '12'), ('11', '15'), ('11', '19'), ('12', '13'), ('12', '14'), ('15', '11'), ('15', '14'), ('15', '16'), ('15', '18'), ('16', '15'), ('16', '17'), ('16', '23'), ('17', '16'), ('17', '18'), ('17', '25'), ('18', '15'), ('18', '17'), ('18', '19'), ('19', '20'), ('19', '18'), ('19', '11'), ('19', '9'), ('19', '7'), ('20', '21'), ('20', '23'), ('20', '19'), ('20', '6'), ('23', '20'), ('23', '22'), ('23', '16'), ('24', '23'), ('24', '25')
]

G.add_edges_from(edges)


# Set positions
# position = {
#     '1': (0,0), '2': (3,0), '3': (3,3), '4': (6,0), '5': (6,3), '6': (3,-3), '7': (6,-3),
#     '8': (0.5,-0.5), '9': (3.5,-0.5), '10': (3.5,2.5), '11': (6.5, -0.5), '12': (6.5,2.5), '13': (3.5,-3.5), '14': (6.5, -3.5), '15': (-0.5,-0.5),
#     '16': (1,-1), '17': (4,-1), '18': (4,2), '19': (7,-1), '20': (7,2), '21': (4,-4), '22': (7,-4), '23': (0,-1),
#     '24': (1.5, -1.5), '25': (4.5,-1.5), '26': (4.5, 1.5), '27': (7.5,-1.5), '28': (7.5, 1.5), '29': (4.5, -4.5), '30': (7.5,-4.5), 'E2': (0.5,-1.5),
#     '31': (5,1), '32': (8,-2), '33': (8,1), '34': (8,-5), 'E1': (2,-2), 'E4': (5,-5), 'E3': (5,-2),
#     'E5': (5.5, 0.5), 'E7': (8.5,0.5), 'E6': (8.5,-2.5), 'E8': (8.5,-5.5), 'E9': (9,-5),
# }


position = {
    '13': (3,3),
    '12': (2.5,2),'14': (3.5,2),
    '11': (2,1), '15': (3.5,1),'16': (6,1),
    '1': (0,0),'2':(1,0), '9':(2,0), '19':(3,0), '18':(4,0), '17': (5,0),
    '10': (1.5,-1),'8':(2.2,-1), '25':(4.5,-1),
    '7':(2.5,-2),'24': (5,-2),'23': (6,-2),'22': (7,-2),
    '3': (1,-3),'5': (2.5,-3),'6': (3,-3),'20': (3.5,-3),
    '4':(0,-4), '21':(4,-4)
}

fig, ax = plt.subplots()
# Draw graph outline
nx.draw(G, position, with_labels=True, node_color = [node_colors.get(i, "lightblue") for i in G.nodes()], node_size=800)

# Highlightes path
color = ['blue','green','orange','purple','cyan','black','red']
num = 0
for shortest_path in [aco_path1,aco_path2,aco_path3,aco_path4,aco_path5,aco_path6,aco_path7]:
    highlighted_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
    nx.draw_networkx_edges(G, position, edgelist= highlighted_edges, edge_color=color[num], width=3, ax = ax, arrows=True, arrowstyle="->", arrowsize=20)
    num += 1 


# plt.savefig("./exit_1.png")
plt.show()


