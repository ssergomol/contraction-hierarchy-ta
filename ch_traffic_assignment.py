import heapq
import math
import time

import networkx as nx
import scipy

from network_import import *
from utils import PathUtils

import heapq
import math
import numpy as np
import networkx as nx

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class FlowTransportNetwork:
    def __init__(self):
        self.linkSet = {}
        self.nodeSet = {}

        self.tripSet = {}
        self.zoneSet = {}
        self.originZones = {}

        self.networkx_graph = None

    
        self.order_of = {}  
        self.node_order = {}    
        self.times_all = {}    

    def to_networkx(self):
        """
        Lazy creation of networkx DiGraph from self.linkSet keys.
        """
        if self.networkx_graph is None:
            edges = [(int(begin), int(end)) for (begin, end) in self.linkSet.keys()]
            self.networkx_graph = nx.DiGraph(edges)
        return self.networkx_graph

    def reset_flow(self):
        for link in self.linkSet.values():
            link.reset_flow()
            self.times_all[(link.init_node, link.term_node)] = link.fft

    def reset(self):
        for link in self.linkSet.values():
            link.reset()
            self.times_all[(link.init_node, link.term_node)] = link.fft

    def local_dijkstra_without_v(self, u: str, v: str, P_max: float):
        """
        A specialized Dijkstra that:
          - Ignores node 'v' altogether
          - Expands until the cost exceeds P_max
        Returns a dict D of minimum distances from 'u' to all other nodes
        (ignoring any node in self.order_of, i.e. contracted nodes).
        """
        vertices = list(self.nodeSet.keys())
        visited = set()
        pq = [(0.0, u)]
        D = {x: math.inf for x in vertices}

        visited.add(v)
        D[u] = 0.0

        while pq:
            cost, n = heapq.heappop(pq)
            if n in visited:
                continue

            if cost > P_max:
                break

            visited.add(n)

            for neighbor in self.nodeSet[n].outLinks:
                if neighbor in self.order_of:
                    continue
                if (n, neighbor) not in self.linkSet:
                    continue

                old_cost = D[neighbor]
                new_cost = D[n] + self.linkSet[(n, neighbor)].cost
                if new_cost < old_cost:
                    D[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))

        return D


    def edge_difference(self, v: str) -> int:
        """
        Calculates how many new shortcuts would be introduced if node v is contracted.
        Formula: -(outdeg + indeg) + (number of actual shortcuts)
        """
        outdeg = len(self.nodeSet[v].outLinks)
        indeg = len(self.nodeSet[v].inLinks)
        dif = - outdeg - indeg

        for u in self.nodeSet[v].inLinks:
            if u in self.order_of:
                continue

            P = {}
            for w in self.nodeSet[v].outLinks:
                if w in self.order_of:
                    continue
                if (u, v) not in self.linkSet or (v, w) not in self.linkSet:
                    continue
                P[w] = self.linkSet[(u, v)].cost + self.linkSet[(v, w)].cost

            if not P:
                continue

            P_max = max(P.values())
            D = self.local_dijkstra_without_v(u, v, P_max)

            for w in P.keys():
                if D[w] > P[w]:
                    dif += 1

        return dif

    def get_node_order_edge_difference(self):
        """
        Builds a priority queue of (edge_difference, nodeId) for all nodes in the network.
        """
        node_pq = []
        for v in self.nodeSet.keys():
            if v in self.order_of:
                continue
            dif = self.edge_difference(v)
            heapq.heappush(node_pq, (dif, v))
        return node_pq


    def preprocess(self):
        """
        Main contraction-hierarchy preprocessing routine:
          - Build a PQ based on edge_difference
          - Iteratively pop the node with smallest difference
          - Recompute difference if necessary (lazy updates)
          - Contract the node: create shortcuts if needed
          - Store the contraction order in self.order_of and self.node_order
        """
        node_pq = self.get_node_order_edge_difference()
        order = 0

        while node_pq:
            _, v = heapq.heappop(node_pq)

            new_dif = self.edge_difference(v)
            if node_pq and new_dif > node_pq[0][0]:
                heapq.heappush(node_pq, (new_dif, v))
                continue

            order += 1
            if order % 500 == 0:
                print(f"..........Contracting {order} nodes so far..........")

            self.order_of[v] = order
            self.node_order[order] = v

            for u in self.nodeSet[v].inLinks:
                if u in self.order_of:
                    continue

                P = {}
                for w in self.nodeSet[v].outLinks:
                    if w in self.order_of:
                        continue
                    if (u, v) not in self.linkSet or (v, w) not in self.linkSet:
                        continue
                    P[w] = self.linkSet[(u, v)].cost + self.linkSet[(v, w)].cost

                if not P:
                    continue

                P_max = max(P.values())
                D = self.local_dijkstra_without_v(u, v, P_max)

                for w in P.keys():
                    if D[w] > P[w]:
                        if (u, w) in self.linkSet:
                            self.linkSet[(u, w)].shortcut_node = v
                        else:
                        
                            self.linkSet[(u, w)] = Link(
                                init_node=u,
                                term_node=w,
                                capacity=999999,   # large capacity
                                length=0,
                                fft=P[w],          # treat P[w] as the free-flow time
                                b=0,
                                power=0,
                                speed_limit=999999,
                                toll=0,
                                linkType='shortcut'
                            )
                            self.linkSet[(u, w)].shortcut_node = v

                            if w not in self.nodeSet[u].outLinks:
                                self.nodeSet[u].outLinks.append(w)
                            if u not in self.nodeSet[w].inLinks:
                                self.nodeSet[w].inLinks.append(u)

                        self.times_all[(u, w)] = P[w]
                        self.linkSet[(u, w)].cost = P[w]

        print('Preprocess Done!')

    def bidirectional_dijkstra_CH(self, source_node: str, target_node: str):
        """
        Performs a bidirectional Dijkstra search on the FlowTransportNetwork,
        respecting contraction hierarchy order (`self.order_of`) to skip nodes,
        and also supporting the reconstruction of shortcuts.

        Returns a tuple (shortest_time, shortest_path):
        - shortest_time: the total cost/distance from source_node to target_node
        - shortest_path: list of nodes representing the route from source_node to target_node
                        (including any necessary shortcut expansions)
        """

        vertices = list(self.nodeSet.keys())

        visited_start = set()
        visited_end = set()

        parents1 = {}
        parents2 = {}

        dist1 = {v: math.inf for v in vertices}
        dist2 = {v: math.inf for v in vertices}

        parents1[source_node] = source_node
        parents2[target_node] = target_node
        dist1[source_node] = 0.0
        dist2[target_node] = 0.0

        pq_start = [(0.0, source_node)]
        pq_end   = [(0.0, target_node)]

        while pq_start or pq_end:
            # ---------- Forward Step ----------
            if pq_start:
                current_dist, current_vertex = heapq.heappop(pq_start)
                if current_vertex in visited_start:
                    pass
                else:
                    visited_start.add(current_vertex)

                    for neighbor in self.nodeSet[current_vertex].outLinks:
                  
                        if neighbor in self.order_of and current_vertex in self.order_of:
                            if self.order_of[neighbor] <= self.order_of[current_vertex]:
                                continue

                        old_cost = dist1[neighbor]
                        if (current_vertex, neighbor) not in self.times_all:
                            continue
                        new_cost = dist1[current_vertex] + self.times_all[(current_vertex, neighbor)]
                        if new_cost < old_cost:
                            dist1[neighbor] = new_cost
                            parents1[neighbor] = current_vertex
                            heapq.heappush(pq_start, (new_cost, neighbor))

            # ---------- Backward Step ----------
            if pq_end:
                current_dist, current_vertex = heapq.heappop(pq_end)
                if current_vertex in visited_end:
                    pass
                else:
                    visited_end.add(current_vertex)

                    for neighbor in self.nodeSet[current_vertex].inLinks:
                        if neighbor in self.order_of and current_vertex in self.order_of:
                            if self.order_of[neighbor] <= self.order_of[current_vertex]:
                                continue

                        old_cost = dist2[neighbor]
                        if (neighbor, current_vertex) not in self.times_all:
                            continue
                        new_cost = dist2[current_vertex] + self.times_all[(neighbor, current_vertex)]
                        if new_cost < old_cost:
                            dist2[neighbor] = new_cost
                            parents2[neighbor] = current_vertex
                            heapq.heappush(pq_end, (new_cost, neighbor))

        reachable_nodes = [v for v in vertices
                        if dist1[v] != math.inf and dist2[v] != math.inf]
        if not reachable_nodes:
            return math.inf, []

        shortest_time = math.inf
        common_node = None
        for v in reachable_nodes:
            total_cost = dist1[v] + dist2[v]
            if total_cost < shortest_time:
                shortest_time = total_cost
                common_node = v

        if common_node is None:
            return 0.0, []

   
        path1 = []
        cur_node = common_node
        while parents1[cur_node] != cur_node:
            tmp_node = parents1[cur_node]
            path_fragment = []
            if (tmp_node, cur_node) in self.linkSet and hasattr(self.linkSet[(tmp_node, cur_node)], 'shortcut_node'):
                if self.linkSet[(tmp_node, cur_node)].shortcut_node != 0:
                    path_fragment = self._generate_shortcut(tmp_node, cur_node)

            path1 = path_fragment + path1
            path1 = [tmp_node] + path1
            cur_node = tmp_node

        path2 = []
        cur_node = common_node
        while parents2[cur_node] != cur_node:
            path2.append(cur_node)
            tmp_node = parents2[cur_node]
            path_fragment = []
            if (cur_node, tmp_node) in self.linkSet and hasattr(self.linkSet[(cur_node, tmp_node)], 'shortcut_node'):
                if self.linkSet[(cur_node, tmp_node)].shortcut_node != 0:
                    path_fragment = self._generate_shortcut(cur_node, tmp_node)
            path2 += path_fragment
            cur_node = tmp_node
        path2.append(cur_node)

        shortest_path = path1 + path2

        return shortest_time, shortest_path
    
    def _generate_shortcut(self, start_node: str, end_node: str) -> list:
        """
        Recursively expands any shortcut on (start_node, end_node).
        If self.linkSet[(start_node, end_node)].shortcut_node != 0,
        we expand from start_node -> shortcut_node -> end_node,
        repeating recursively if needed.

        Returns a list of intermediate node IDs (excluding start_node and end_node themselves).
        """
        if (start_node, end_node) not in self.linkSet:
            return []

        link_obj = self.linkSet[(start_node, end_node)]
        if not hasattr(link_obj, 'shortcut_node'):
            return []  
        shortcut_node = getattr(link_obj, 'shortcut_node', 0)
        if shortcut_node == 0:
            return []

        return (self._generate_shortcut(start_node, shortcut_node) +
                [shortcut_node] +
                self._generate_shortcut(shortcut_node, end_node))

    def update_preprocessing(self):
        """
        Быстро обновляет CH после минимальных изменений в `self.times_all`.
        """

        newLinkSet = self.linkSet.copy()
        for (u, v) in newLinkSet:
            if self.linkSet[(u, v)].linkType == 'shortcut':
                del self.linkSet[(u, v)]
                self.nodeSet[u].outLinks.remove(v)
                self.nodeSet[v].inLinks.remove(u)

        self.preprocess()

   

class Zone:
    def __init__(self, zoneId: str):
        self.zoneId = zoneId

        self.lat = 0
        self.lon = 0
        self.destList = [] 


class Node:
    """
    This class has attributes associated with any node
    """

    def __init__(self, nodeId: str):
        self.Id = nodeId

        self.lat = 0
        self.lon = 0

        self.outLinks = []  # list of node ids (strs)
        self.inLinks = []  # list of node ids (strs)

        # For Dijkstra
        self.label = np.inf
        self.pred = None


class Link:
    """
    This class has attributes associated with any link
    """
    def __init__(self,
                 init_node: str,
                 term_node: str,
                 capacity: float,
                 length: float,
                 fft: float,
                 b: float,
                 power: float,
                 speed_limit: float,
                 toll: float,
                 linkType: str
                 ):
        self.init_node = init_node
        self.term_node = term_node
        self.max_capacity = float(capacity)  # veh per hour
        self.length = float(length)  # Length
        self.fft = float(fft)  # Free flow travel time (min)
        self.beta = float(power)
        self.alpha = float(b)
        self.speedLimit = float(speed_limit)
        self.toll = float(toll)
        self.linkType = linkType

        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.flow = 0.0
        self.cost = self.fft

        self.shortcut_node = 0 

    def modify_capacity(self, delta_percentage: float):
        assert -1 <= delta_percentage <= 1
        self.curr_capacity_percentage += delta_percentage
        self.curr_capacity_percentage = max(0, min(1, self.curr_capacity_percentage))
        self.capacity = self.max_capacity * self.curr_capacity_percentage

    def reset(self):
        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.reset_flow()

    def reset_flow(self):
        self.flow = 0.0
        self.cost = self.fft


class Demand:
    def __init__(self,
                 init_node: str,
                 term_node: str,
                 demand: float
                 ):
        self.fromZone = init_node
        self.toNode = term_node
        self.demand = float(demand)

def DijkstraHeap(origin, network: FlowTransportNetwork):
    """
    Calcualtes shortest path from an origin to all other destinations.
    The labels and preds are stored in node instances.
    """
    for n in network.nodeSet:
        network.nodeSet[n].label = np.inf
        network.nodeSet[n].pred = None
    network.nodeSet[origin].label = 0.0
    network.nodeSet[origin].pred = None
    SE = [(0, origin)]
    while SE:
        currentNode = heapq.heappop(SE)[1]
        currentLabel = network.nodeSet[currentNode].label
        for toNode in network.nodeSet[currentNode].outLinks:
            link = (currentNode, toNode)
            newNode = toNode
            newPred = currentNode
            existingLabel = network.nodeSet[newNode].label
            newLabel = currentLabel + network.linkSet[link].cost
            if newLabel < existingLabel:
                heapq.heappush(SE, (newLabel, newNode))
                network.nodeSet[newNode].label = newLabel
                network.nodeSet[newNode].pred = newPred

def euclidean_distance_to_origin(node_id: str, origin: str, network: FlowTransportNetwork) -> float:
    """
    Example heuristic: Euclidean distance from the current node to the 'origin'.
    This is NOT truly an admissible heuristic for shortest-time or shortest-cost paths
    unless your cost is purely Euclidean distance.

    For multi-destination usage, this is mostly a placeholder that won't help prune
    states as A* normally would. If you had a single 'target', you'd measure distance
    to that 'target' node instead.
    """
    lat1, lon1 = network.nodeSet[origin].lat, network.nodeSet[origin].lon
    lat2, lon2 = network.nodeSet[node_id].lat, network.nodeSet[node_id].lon
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def AStarHeap(origin: str, network: FlowTransportNetwork):
    """
    Calculates the shortest path from the 'origin' to all other nodes in the network
    using an A* approach.  The labels and preds are stored in the node instances:
        - nodeSet[node_id].label = shortest distance from origin to 'node_id'
        - nodeSet[node_id].pred  = predecessor node on the path from origin

    NOTE: Because we do not have a single 'target' here, we use a minimal
    heuristic (Euclidean distance to 'origin' or zero) just to demonstrate
    the A* structure. For real A*, you'd typically use a 'target' node
    and an admissible (and consistent) heuristic from each node to that target.
    """

    for n in network.nodeSet:
        network.nodeSet[n].label = np.inf
        network.nodeSet[n].pred = None

    network.nodeSet[origin].label = 0.0
    network.nodeSet[origin].pred = None

    SE = []

    h_cost_origin = 0  
    heapq.heappush(SE, (h_cost_origin, origin))

    visited = set()

    while SE:
        current_f, currentNode = heapq.heappop(SE)

        if currentNode in visited:
            continue
        visited.add(currentNode)

        currentLabel = network.nodeSet[currentNode].label

        for toNode in network.nodeSet[currentNode].outLinks:
            link = (currentNode, toNode)
            if link not in network.linkSet:
                continue

            newNode = toNode
            newPred = currentNode
            existingLabel = network.nodeSet[newNode].label
            newLabel = currentLabel + network.linkSet[link].cost

            if newLabel < existingLabel:
                network.nodeSet[newNode].label = newLabel
                network.nodeSet[newNode].pred = newPred

                h_cost = euclidean_distance_to_origin(newNode, origin, network)
                f_cost = newLabel + h_cost

                heapq.heappush(SE, (f_cost, newNode))

def BPRcostFunction(optimal: bool,
                    fft: float,
                    alpha: float,
                    flow: float,
                    capacity: float,
                    beta: float,
                    length: float,
                    maxSpeed: float
                    ) -> float:
    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        return fft * (1 + (alpha * math.pow((flow * 1.0 / capacity), beta)) * (beta + 1))
    return fft * (1 + alpha * math.pow((flow * 1.0 / capacity), beta))


def constantCostFunction(optimal: bool,
                         fft: float,
                         alpha: float,
                         flow: float,
                         capacity: float,
                         beta: float,
                         length: float,
                         maxSpeed: float
                         ) -> float:
    if optimal:
        return fft + flow
    return fft


def greenshieldsCostFunction(optimal: bool,
                             fft: float,
                             alpha: float,
                             flow: float,
                             capacity: float,
                             beta: float,
                             length: float,
                             maxSpeed: float
                             ) -> float:
    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        return (length * (capacity ** 2)) / (maxSpeed * (capacity - flow) ** 2)
    return length / (maxSpeed * (1 - (flow / capacity)))


def updateTravelTime(network: FlowTransportNetwork, optimal: bool = False, costFunction=BPRcostFunction, algo='dijkstra'):
    """
    This method updates the travel time on the links with the current flow
    """
    for l in network.linkSet:
        if (network.linkSet[l].linkType != 'shortcut'):
            network.linkSet[l].cost = costFunction(optimal,
                                                network.linkSet[l].fft,
                                                network.linkSet[l].alpha,
                                                network.linkSet[l].flow,
                                                network.linkSet[l].capacity,
                                                network.linkSet[l].beta,
                                                network.linkSet[l].length,
                                                network.linkSet[l].speedLimit
                                                )
            network.times_all[l] = network.linkSet[l].cost
    
    if algo == 'CH':
        network.update_preprocessing()
    
  
def findAlpha(x_bar, network: FlowTransportNetwork, iteration_number, optimal: bool = False, costFunction=BPRcostFunction):
    """
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm
    """

    def df(alpha):
        assert 0 <= alpha <= 1
        sum_derivative = 0  
        for l in network.linkSet:
            if (network.linkSet[l].linkType != 'shortcut'):
                tmpFlow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow
                tmpCost = costFunction(optimal,
                                    network.linkSet[l].fft,
                                    network.linkSet[l].alpha,
                                    tmpFlow,
                                    network.linkSet[l].capacity,
                                    network.linkSet[l].beta,
                                    network.linkSet[l].length,
                                    network.linkSet[l].speedLimit
                                    )
                sum_derivative = sum_derivative + (x_bar[l] - network.linkSet[l].flow) * tmpCost
        return sum_derivative

    sol = scipy.optimize.root_scalar(df, x0=np.array([0.5]), bracket=(0, 1))
    assert 0 <= sol.root <= 1
    return sol.root


def tracePreds(dest, network: FlowTransportNetwork):
    """
    This method traverses predecessor nodes in order to create a shortest path
    """
    prevNode = network.nodeSet[dest].pred
    spLinks = []
    while prevNode is not None:
        spLinks.append((prevNode, dest))
        dest = prevNode
        prevNode = network.nodeSet[dest].pred
    return spLinks

def loadAON_new(network: FlowTransportNetwork, computeXbar: bool = True):
    """
    This method produces auxiliary flows for all or nothing loading,
    but now uses CH-based shortest paths (bidirectional_dijkstra_CH) instead of plain DijkstraHeap.

    For each origin node r:
      1) For each destination node s in network.zoneSet[r].destList:
         - Use bidirectional_dijkstra_CH(r, s) to get (shortest_time, shortest_path)
         - Accumulate total travel time = shortest_time * demand
         - If computeXbar is True and a path exists, distribute flows on each link in the path
    """
    
    x_bar = {l: 0.0 for l in network.linkSet if network.linkSet[l].linkType != 'shortcut'}
    SPTT = 0.0  

    for r in network.originZones:
        for s in network.zoneSet[r].destList:
            dem = network.tripSet[r, s].demand
            if dem <= 0:
                continue  

            shortest_time, shortest_path = network.bidirectional_dijkstra_CH_old(r, s)

            if not shortest_path:
                continue

            SPTT += shortest_time * dem

            if computeXbar and r != s:
                
                for i in range(len(shortest_path) - 2, -1, -1):
                    link_tuple = (shortest_path[i], shortest_path[i+1])
                    x_bar[link_tuple] += dem

    return SPTT, x_bar

def loadAON(network: FlowTransportNetwork, algo: str = "dijkstra", computeXbar: bool = True):
    """
    This method produces auxiliary flows for all or nothing loading.
    """
    SPTT = 0.0
    x_bar = {l: 0.0 for l in network.linkSet if network.linkSet[l].linkType != 'shortcut'}

    if algo == 'dijkstra' or algo == 'A*':
        for r in network.originZones:
            if algo == 'dijkstra':
                DijkstraHeap(r, network=network)
            elif algo == 'A*':
                AStarHeap(r, network=network)

            for s in network.zoneSet[r].destList:
                dem = network.tripSet[r, s].demand

                if dem <= 0:
                    continue

                SPTT = SPTT + network.nodeSet[s].label * dem

                if computeXbar and r != s:
                    for spLink in tracePreds(s, network):
                        x_bar[spLink] = x_bar[spLink] + dem

        return SPTT, x_bar
    elif algo == 'CH':
        for r in network.originZones:
            for s in network.zoneSet[r].destList:
                dem = network.tripSet[r, s].demand
                if dem <= 0:
                    continue  

                shortest_time, shortest_path = network.bidirectional_dijkstra_CH_old(r, s)

                if not shortest_path:
                    continue

                SPTT += shortest_time * dem
                if computeXbar and r != s:
                    if algo == 'CH':
                        for i in range(len(shortest_path) - 2, -1, -1):
                            link_tuple = (shortest_path[i], shortest_path[i+1])
                            x_bar[link_tuple] += dem

    return SPTT, x_bar

def readDemand(demand_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in demand_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        demand = row["demand"]

        network.tripSet[init_node, term_node] = Demand(init_node, term_node, demand)
        if init_node not in network.zoneSet:
            network.zoneSet[init_node] = Zone(init_node)
        if term_node not in network.zoneSet:
            network.zoneSet[term_node] = Zone(term_node)
        if term_node not in network.zoneSet[init_node].destList:
            network.zoneSet[init_node].destList.append(term_node)

    print(len(network.tripSet), "OD pairs")
    print(len(network.zoneSet), "OD zones")


def readNetwork(network_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in network_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        capacity = row["capacity"]
        length = row["length"]
        free_flow_time = row["free_flow_time"]
        b = row["b"]
        power = row["power"]
        speed = row["speed"]
        toll = row["toll"]
        link_type = row["link_type"]

        network.linkSet[init_node, term_node] = Link(
            init_node=init_node,
            term_node=term_node,
            capacity=capacity,
            length=length,
            fft=free_flow_time,
            b=b,
            power=power,
            speed_limit=speed,
            toll=toll,
            linkType=link_type
        )

        if init_node not in network.nodeSet:
            network.nodeSet[init_node] = Node(init_node)
        if term_node not in network.nodeSet:
            network.nodeSet[term_node] = Node(term_node)

        if term_node not in network.nodeSet[init_node].outLinks:
            network.nodeSet[init_node].outLinks.append(term_node)
        if init_node not in network.nodeSet[term_node].inLinks:
            network.nodeSet[term_node].inLinks.append(init_node)

        network.times_all[(init_node, term_node)] = free_flow_time


    print(len(network.nodeSet), "nodes")
    print(len(network.linkSet), "links")
    print(len(network.times_all), "entries in times_all")


def get_TSTT(network: FlowTransportNetwork, costFunction=BPRcostFunction, use_max_capacity: bool = True):
    TSTT = round(sum([network.linkSet[a].flow * costFunction(optimal=False,
                                                             fft=network.linkSet[
                                                                 a].fft,
                                                             alpha=network.linkSet[
                                                                 a].alpha,
                                                             flow=network.linkSet[
                                                                 a].flow,
                                                             capacity=network.linkSet[
                                                                 a].max_capacity if use_max_capacity else network.linkSet[
                                                                 a].capacity,
                                                             beta=network.linkSet[
                                                                 a].beta,
                                                             length=network.linkSet[
                                                                 a].length,
                                                             maxSpeed=network.linkSet[
                                                                 a].speedLimit
                                                             ) for a in
                      network.linkSet if network.linkSet[a].linkType != 'shortcut']), 9)
    return TSTT

def assignment_loop_new(network: FlowTransportNetwork,
                    algorithm: str = "FW",
                    systemOptimal: bool = False,
                    costFunction=BPRcostFunction,
                    accuracy: float = 0.001,
                    maxIter: int = 1000000,
                    maxTime: int = 60,
                    verbose: bool = True,
                    shortestPathAlgo = "dijkstra"):

    network.reset_flow()

    iteration_number = 1
    gap = np.inf
    TSTT = np.inf
    assignmentStartTime = time.time()

    while gap > accuracy:

        SPTT, x_bar = loadAON(network=network, algo=shortestPathAlgo)  # now uses CH for shortest paths

        if algorithm == "MSA" or iteration_number == 1:
            alpha = (1 / iteration_number)
        elif algorithm == "FW":
            alpha = findAlpha(x_bar, network=network, optimal=systemOptimal, costFunction=costFunction)
        else:
            raise TypeError('Algorithm must be MSA or FW')

        for l in network.linkSet:
            if (network.linkSet[l].linkType != 'shortcut'):
                network.linkSet[l].flow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow

        updateTravelTime(network=network, optimal=systemOptimal, costFunction=costFunction, algo=shortestPathAlgo)

        SPTT = round(SPTT, 9)
        TSTT = round(sum([network.linkSet[a].flow * network.linkSet[a].cost for a in network.linkSet if network.linkSet[a].linkType != 'shortcut']), 9)
        gap = (TSTT / SPTT) - 1

        if gap < 0:
            print("Error, negative gap encountered. TSTT < SPTT?")

        iteration_number += 1

    return TSTT


def writeResults(network: FlowTransportNetwork, output_file: str, costFunction=BPRcostFunction,
                 systemOptimal: bool = False, verbose: bool = True):
    outFile = open(output_file, "w")
    TSTT = get_TSTT(network=network, costFunction=costFunction)
    if verbose:
        print("\nTotal system travel time:", f'{TSTT} secs')
    tmpOut = "Total Travel Time:\t" + str(TSTT)
    outFile.write(tmpOut + "\n")
    tmpOut = "Cost function used:\t" + BPRcostFunction.__name__
    outFile.write(tmpOut + "\n")
    tmpOut = ["User equilibrium (UE) or system optimal (SO):\t"] + ["SO" if systemOptimal else "UE"]
    outFile.write("".join(tmpOut) + "\n\n")
    tmpOut = "init_node\tterm_node\tflow\ttravelTime"
    outFile.write(tmpOut + "\n")
    for i in network.linkSet:
        if (network.linkSet[i].linkType != 'shortcut'):
            tmpOut = str(network.linkSet[i].init_node) + "\t" + str(
                network.linkSet[i].term_node) + "\t" + str(
                network.linkSet[i].flow) + "\t" + str(costFunction(False,
                                                                network.linkSet[i].fft,
                                                                network.linkSet[i].alpha,
                                                                network.linkSet[i].flow,
                                                                network.linkSet[i].max_capacity,
                                                                network.linkSet[i].beta,
                                                                network.linkSet[i].length,
                                                                network.linkSet[i].speedLimit
                                                                ))
            outFile.write(tmpOut + "\n")
    outFile.close()
    
def assignment_loop(network: FlowTransportNetwork,
                    algorithm: str = "FW",
                    systemOptimal: bool = False,
                    costFunction=BPRcostFunction,
                    accuracy: float = 0.001,
                    maxIter: int = 1000000,
                    maxTime: int = 60,
                    verbose: bool = True,
                    shortestPathAlgo: str = 'dijkstra'):
    """
    For explaination of the algorithm see Chapter 7 of:
    https://sboyles.github.io/blubook.html
    PDF:
    https://sboyles.github.io/teaching/ce392c/book.pdf
    """
    network.reset_flow()

    iteration_number = 1
    gap = np.inf
    TSTT = np.inf
    assignmentStartTime = time.time()

    while gap > accuracy:

        _, x_bar = loadAON(network=network, algo=shortestPathAlgo)

        if algorithm == "MSA" or iteration_number == 1:
            alpha = (1 / iteration_number)
        elif algorithm == "FW":
            alpha = findAlpha(x_bar,
                              network=network,
                              optimal=systemOptimal,
                              costFunction=costFunction, iteration_number=iteration_number)
        else:
            print("Terminating the program.....")
            print("The solution algorithm ", algorithm, " does not exist!")
            raise TypeError('Algorithm must be MSA or FW')

        for l in network.linkSet:
            if (network.linkSet[l].linkType != 'shortcut'):
                network.linkSet[l].flow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow

        updateTravelTime(network=network,
                         optimal=systemOptimal,
                         costFunction=costFunction, algo=shortestPathAlgo)
        

        SPTT, _ = loadAON(network=network, algo=shortestPathAlgo, computeXbar=False)
        SPTT = round(SPTT, 9)
        TSTT = round(sum([network.linkSet[a].flow * network.linkSet[a].cost for a in
                          network.linkSet if network.linkSet[a].linkType != 'shortcut']), 9)

        gap = (TSTT / SPTT) - 1
        print(TSTT, SPTT, "TSTT, SPTT, Max capacity", max([l.capacity for l in network.linkSet.values()]), gap)

        if gap < 0:
            print("Error, gap is less than 0, this should not happen")
            print("TSTT", "SPTT", TSTT, SPTT)

        TSTT = get_TSTT(network=network, costFunction=costFunction)

        iteration_number += 1
        if iteration_number > maxIter:
            if verbose:
                print(
                    "The assignment did not converge to the desired gap and the max number of iterations has been reached")
                print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
                print("Current gap:", round(gap, 5))
            return TSTT
        if time.time() - assignmentStartTime > maxTime:
            if verbose:
                print("The assignment did not converge to the desired gap and the max time limit has been reached")
                print("Assignment did ", iteration_number, "iterations")
                print("Current gap:", round(gap, 5))
            return TSTT

    if verbose:
        print("Assignment converged in ", iteration_number, "iterations")
        print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")

    return TSTT


def load_network(net_file: str,
                 demand_file: str = None,
                 force_net_reprocess: bool = False,
                 verbose: bool = True
                 ) -> FlowTransportNetwork:
    readStart = time.time()

    if demand_file is None:
        demand_file = '_'.join(net_file.split("_")[:-1] + ["trips.tntp"])

    net_name = net_file.split("/")[-1].split("_")[0]

    if verbose:
        print(f"Loading network {net_name}...")

    net_df, demand_df = import_network(
        net_file,
        demand_file,
        force_reprocess=force_net_reprocess
    )

    network = FlowTransportNetwork()

    readDemand(demand_df, network=network)
    readNetwork(net_df, network=network)

    network.originZones = set([k[0] for k in network.tripSet])

    if verbose:
        print("Network", net_name, "loaded")
        print("Reading the network data took", round(time.time() - readStart, 2), "secs\n")

    return network


def computeAssingment(net_file: str,
                      demand_file: str = None,
                      algorithm: str = "FW",
                      costFunction=BPRcostFunction,
                      systemOptimal: bool = False,
                      accuracy: float = 0.0001,
                      maxIter: int = 1000,
                      maxTime: int = 60,
                      results_file: str = None,
                      force_net_reprocess: bool = False,
                      verbose: bool = True,
                      shortestPathAlgo: str = 'dijkstra'
                      ) -> float:

    network = load_network(net_file=net_file, 
                           demand_file=demand_file, 
                           verbose=verbose, 
                           force_net_reprocess=force_net_reprocess)

    if verbose:
        print("Preprocessing the network...")
    t = time.time()
    
    print("Preprocessing time", time.time() - t)
    if shortestPathAlgo == 'CH':
        network.preprocess()

    if verbose:
        print("Running assignment loop on the preprocessed network...")
    TSTT = assignment_loop(network=network,
                           algorithm=algorithm,
                           systemOptimal=systemOptimal,
                           costFunction=costFunction,
                           accuracy=accuracy,
                           maxIter=maxIter,
                           maxTime=maxTime,
                           verbose=verbose, 
                           shortestPathAlgo=shortestPathAlgo)

    if results_file is None:
        results_file = '_'.join(net_file.split("_")[:-1] + ["flow.tntp"])
    writeResults(network=network,
                 output_file=results_file,
                 costFunction=costFunction,
                 systemOptimal=systemOptimal,
                 verbose=verbose)

    return TSTT


def draw_and_save_graph(G, filename="graph.png", figsize=(8, 6), node_color='lightblue', edge_color='gray', node_size=3000, font_size=15):
    """
    Draws a NetworkX graph (directed or undirected), saves it as an image, and displays it.

    Parameters:
    - G (networkx.Graph): The graph to draw (can be directed or undirected).
    - filename (str): The name of the file to save the graph image (default: "graph.png").
    - figsize (tuple): Size of the figure (default: (8, 6)).
    - node_color (str): Color of the nodes (default: 'lightblue').
    - edge_color (str): Color of the edges (default: 'gray').
    - node_size (int): Size of the nodes (default: 3000).
    - font_size (int): Font size for labels (default: 15).
    """
    plt.figure(figsize=figsize)
    nx.draw(G, with_labels=True, node_color=node_color, edge_color=edge_color, node_size=node_size, font_size=font_size, arrows=True)
    
    plt.savefig(filename)
    print(f"Graph saved as {filename}")
    
    # plt.show()

if __name__ == '__main__':
    net_file = str(PathUtils.berlin_center_net_file)

    print("\nSystem Optimal\n")
    total_system_travel_time_optimal = computeAssingment(net_file=net_file,
                                                         algorithm="FW",
                                                         costFunction=BPRcostFunction,
                                                         systemOptimal=True,
                                                         verbose=True,
                                                         accuracy=0.01,
                                                         maxIter=100000,
                                                         maxTime=6000000, 
                                                         shortestPathAlgo='A*')
    print("\nUser Equilibrium\n")
    total_system_travel_time_equilibrium = computeAssingment(net_file=net_file,
                                                             algorithm="FW",
                                                             costFunction=BPRcostFunction,
                                                             systemOptimal=False,
                                                             verbose=True,
                                                             accuracy=0.01,
                                                             maxIter=100000,
                                                             maxTime=6000000, 
                                                             shortestPathAlgo='A*')

    print("UE", total_system_travel_time_equilibrium)
    print("SO", total_system_travel_time_optimal)
