import time
# import heapq
import numpy as np


from ch_traffic_assignment import DijkstraHeap, AStarHeap


# Тестовый класс узла
class Node:
    def __init__(self, name, x=0, y=0):
        self.name = name
        self.x = x
        self.y = y
        self.label = np.inf
        self.pred = None
        self.outLinks = []


# Тестовый класс сети
class FlowTransportNetwork:
    def __init__(self):
        self.nodeSet = {}
        self.linkSet = {}

    def add_node(self, name, x=0, y=0):
        self.nodeSet[name] = Node(name, x, y)

    def add_link(self, from_node, to_node, cost):
        self.nodeSet[from_node].outLinks.append(to_node)
        self.linkSet[(from_node, to_node)] = Link(cost)


class Link:
    def __init__(self, cost):
        self.cost = cost


# Эвристическая функция
def heuristic(node, target):
    return abs(node.x - target.x) + abs(node.y - target.y)


# Тестовый граф
def create_test_network():
    network = FlowTransportNetwork()
    network.add_node('A', 0, 0)
    network.add_node('B', 1, 1)
    network.add_node('C', 2, 1)
    network.add_node('D', 1, 2)
    network.add_node('E', 2, 2)

    network.add_link('A', 'B', 1)
    network.add_link('A', 'C', 4)
    network.add_link('B', 'C', 2)
    network.add_link('B', 'D', 5)
    network.add_link('C', 'E', 1)
    network.add_link('D', 'E', 2)

    return network


# Тест сравнения
def compare_algorithms():
    network1 = create_test_network()
    network2 = create_test_network()

    origin = 'A'
    target = 'E'

    # Тест Дийкстры
    start_time = time.time()
    DijkstraHeap(origin, network1)
    dijkstra_time = time.time() - start_time
    dijkstra_path = reconstruct_path(network1, target)
    dijkstra_cost = network1.nodeSet[target].label

    # Тест A*
    start_time = time.time()
    AStarHeap(origin, target, network2)
    astar_time = time.time() - start_time
    astar_path = reconstruct_path(network2, target)
    astar_cost = network2.nodeSet[target].label

    print("📊 Результаты сравнения:")
    print(f"Dijkstra Path: {dijkstra_path}, Cost: {dijkstra_cost}, Time: {dijkstra_time:.6f}s")
    print(f"A* Path: {astar_path}, Cost: {astar_cost}, Time: {astar_time:.6f}s")

    assert dijkstra_cost == astar_cost, "❌ Стоимости путей не совпадают"
    assert dijkstra_path == astar_path, "❌ Пути не совпадают"
    print("✅ Алгоритмы возвращают одинаковые пути и стоимости.")


# Восстановление пути
def reconstruct_path(network, target):
    path = []
    current = target
    while current:
        path.append(current)
        current = network.nodeSet[current].pred
    return path[::-1]


# Запуск тестов
if __name__ == '__main__':
    compare_algorithms()
