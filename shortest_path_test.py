import unittest
import math

from assignment import (
    FlowTransportNetwork,
    Node,
    Link,
    Demand,
    Zone,
    DijkstraHeap,
    loadAON,
    tracePreds
)

class TestFlowTransportNetworkShortestPaths(unittest.TestCase):
    def setUp(self):
        self.network = FlowTransportNetwork()
        
        nodes = ['A', 'B', 'C', 'D']
        for node_id in nodes:
            self.network.nodeSet[node_id] = Node(node_id)
        
        edges = [
            ('A', 'B', 1),
            ('B', 'C', 1),
            ('A', 'C', 3),
            ('C', 'D', 1),
            ('B', 'D', 4)
        ]
        
        for u, v, cost in edges:
            link_fw = Link(
                init_node=u,
                term_node=v,
                capacity=1000,
                length=1.0,
                fft=cost,
                b=1.0,
                power=4.0,
                speed_limit=60.0,
                toll=0.0,
                linkType='regular'
            )
            self.network.linkSet[(u, v)] = link_fw
            self.network.nodeSet[u].outLinks.append(v)
            self.network.nodeSet[v].inLinks.append(u)
            self.network.times_all[(u, v)] = cost
            
            link_bw = Link(
                init_node=v,
                term_node=u,
                capacity=1000,
                length=1.0,
                fft=cost,
                b=1.0,
                power=4.0,
                speed_limit=60.0,
                toll=0.0,
                linkType='regular'
            )
            self.network.linkSet[(v, u)] = link_bw
            self.network.nodeSet[v].outLinks.append(u)
            self.network.nodeSet[u].inLinks.append(v)
            self.network.times_all[(v, u)] = cost
        
        self.network.originZones = {'A', 'B', 'C', 'D'}
        
        for origin in self.network.originZones:
            self.network.zoneSet[origin] = Zone(origin)
            self.network.zoneSet[origin].destList = [node for node in nodes if node != origin]
        
        demands = [
            ('A', 'C', 10),
            ('A', 'D', 5),
            ('B', 'A', 8),
            ('B', 'D', 7),
            ('C', 'A', 6),
            ('C', 'D', 4),
            ('D', 'A', 3),
            ('D', 'B', 2),
            ('D', 'C', 1)
        ]
        
        for init, term, dem in demands:
            self.network.tripSet[(init, term)] = Demand(init, term, dem)
        
        self.network.preprocess()

    def reconstruct_path_via_dijkstra(self, origin, destination):
        DijkstraHeap(origin, self.network)
        
        shortest_time = self.network.nodeSet[destination].label
        
        if shortest_time == math.inf:
            return (math.inf, [])
        
        path_links = tracePreds(destination, self.network)
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = self.network.nodeSet[current].pred
        path = path[::-1]  
        
        return (shortest_time, path)
    
    def test_shortest_paths_agreement(self):
        test_cases = [
            ('A', 'D', 3, ['A', 'B', 'C', 'D']),
            ('A', 'C', 2, ['A', 'B', 'C']),
            ('B', 'D', 2, ['B', 'C', 'D']),
            ('C', 'A', 2, ['C', 'B', 'A']),
            ('D', 'A', 3, ['D', 'C', 'B', 'A']),
            ('B', 'A', 1, ['B', 'A']),  
            ('C', 'D', 1, ['C', 'D']),  
            ('D', 'B', 2, ['D', 'C', 'B']),
        ]
        
        for origin, destination, expected_time, expected_path in test_cases:
            with self.subTest(origin=origin, destination=destination):
                dijkstra_time, dijkstra_path = self.reconstruct_path_via_dijkstra(origin, destination)
                
                ch_time, ch_path = self.network.bidirectional_dijkstra_CH(origin, destination)
                
                self.assertAlmostEqual(dijkstra_time, ch_time, places=5,
                                       msg=f"Shortest times differ for {origin} -> {destination}")
                
                self.assertEqual(dijkstra_path, ch_path,
                                 msg=f"Shortest paths differ for {origin} -> {destination}")

    def test_no_path(self):
        """
        Test that both algorithms correctly handle cases where no path exists.
        """
        self.network.nodeSet['E'] = Node('E')
        self.network.originZones.add('E')
        self.network.zoneSet['E'] = Zone('E')
        self.network.zoneSet['E'].destList = []
        
        origin = 'A'
        destination = 'E'
        
        dijkstra_time, dijkstra_path = self.reconstruct_path_via_dijkstra(origin, destination)
        
        ch_time, ch_path = self.network.bidirectional_dijkstra_CH(origin, destination)
        
        self.assertEqual(dijkstra_time, math.inf,
                         msg=f"DijkstraHeap should report no path for {origin} -> {destination}")
        self.assertEqual(ch_time, math.inf,
                         msg=f"bidirectional_dijkstra_CH should report no path for {origin} -> {destination}")
        self.assertEqual(dijkstra_path, [],
                         msg=f"DijkstraHeap should return empty path for {origin} -> {destination}")
        self.assertEqual(ch_path, [],
                         msg=f"bidirectional_dijkstra_CH should return empty path for {origin} -> {destination}")

    def test_same_source_and_destination(self):
        """
        Test that both algorithms handle cases where the source and destination are the same.
        """
        origin = 'A'
        destination = 'A'
        
        dijkstra_time, dijkstra_path = self.reconstruct_path_via_dijkstra(origin, destination)
        
        ch_time, ch_path = self.network.bidirectional_dijkstra_CH(origin, destination)
        
        self.assertEqual(dijkstra_time, 0.0,
                         msg=f"DijkstraHeap should report zero time for {origin} -> {destination}")
        self.assertEqual(ch_time, 0.0,
                         msg=f"bidirectional_dijkstra_CH should report zero time for {origin} -> {destination}")
        self.assertEqual(dijkstra_path, [origin],
                         msg=f"DijkstraHeap should return path with only the source node for {origin} -> {destination}")
        self.assertEqual(ch_path, [origin],
                         msg=f"bidirectional_dijkstra_CH should return path with only the source node for {origin} -> {destination}")

    def test_large_grid_shortest_paths_agreement(self):
        grid_size = 10
        grid_nodes = [f"{i}-{j}" for i in range(grid_size) for j in range(grid_size)]
        
        self.network = FlowTransportNetwork()
        
        for node_id in grid_nodes:
            self.network.nodeSet[node_id] = Node(node_id)
        
        for i in range(grid_size):
            for j in range(grid_size):
                current = f"{i}-{j}"
                neighbors = []
                if i < grid_size - 1:
                    neighbors.append(f"{i+1}-{j}")
                if j < grid_size - 1:
                    neighbors.append(f"{i}-{j+1}")
                
                for neighbor in neighbors:
                    link_fw = Link(
                        init_node=current,
                        term_node=neighbor,
                        capacity=1000,
                        length=1.0,
                        fft=1.0,
                        b=1.0,
                        power=4.0,
                        speed_limit=60.0,
                        toll=0.0,
                        linkType='regular'
                    )
                    self.network.linkSet[(current, neighbor)] = link_fw
                    self.network.nodeSet[current].outLinks.append(neighbor)
                    self.network.nodeSet[neighbor].inLinks.append(current)
                    self.network.times_all[(current, neighbor)] = 1.0
                    
                    link_bw = Link(
                        init_node=neighbor,
                        term_node=current,
                        capacity=1000,
                        length=1.0,
                        fft=1.0,
                        b=1.0,
                        power=4.0,
                        speed_limit=60.0,
                        toll=0.0,
                        linkType='regular'
                    )
                    self.network.linkSet[(neighbor, current)] = link_bw
                    self.network.nodeSet[neighbor].outLinks.append(current)
                    self.network.nodeSet[current].inLinks.append(neighbor)
                    self.network.times_all[(neighbor, current)] = 1.0
        
        self.network.originZones = set(grid_nodes)
        
        for origin in self.network.originZones:
            self.network.zoneSet[origin] = Zone(origin)
            self.network.zoneSet[origin].destList = [node for node in grid_nodes if node != origin]
        
        for origin in self.network.originZones:
            for destination in self.network.zoneSet[origin].destList:
                self.network.tripSet[(origin, destination)] = Demand(origin, destination, 1.0)
        
        self.network.preprocess()
        
        test_od_pairs = [
            ('0-0', '9-9'),  
            ('0-9', '9-0'),  
            ('5-5', '9-9'),  
            ('0-0', '5-5'),  
            ('3-7', '8-2'),  
            ('2-3', '7-8'),  
            ('1-1', '8-8'),  
            ('4-4', '6-6'),  
            ('0-5', '5-0'),  
            ('9-9', '0-0'),  
        ]
        
        for origin, destination in test_od_pairs:
            with self.subTest(origin=origin, destination=destination):
                dijkstra_time, dijkstra_path = self.reconstruct_path_via_dijkstra(origin, destination)
                
                ch_time, ch_path = self.network.bidirectional_dijkstra_CH(origin, destination)
                
                self.assertAlmostEqual(dijkstra_time, ch_time, places=5,
                                       msg=f"Shortest times differ for {origin} -> {destination}")
                
                # # Assert that the shortest paths are identical
                # self.assertEqual(dijkstra_path, ch_path,
                #                  msg=f"Shortest paths differ for {origin} -> {destination}")

    def test_very_large_grid_shortest_paths(self):
        import time

        grid_size = 100
        grid_nodes = [f"{i}-{j}" for i in range(grid_size) for j in range(grid_size)]
        
        self.network = FlowTransportNetwork()
        
        for node_id in grid_nodes:
            self.network.nodeSet[node_id] = Node(node_id)
        
        for i in range(grid_size):
            for j in range(grid_size):
                current = f"{i}-{j}"
                neighbors = []
                if i < grid_size - 1:
                    neighbors.append(f"{i+1}-{j}")
                if j < grid_size - 1:
                    neighbors.append(f"{i}-{j+1}")
                
                for neighbor in neighbors:
                    link_fw = Link(
                        init_node=current,
                        term_node=neighbor,
                        capacity=1000,
                        length=1.0,
                        fft=1.0,
                        b=1.0,
                        power=4.0,
                        speed_limit=60.0,
                        toll=0.0,
                        linkType='regular'
                    )
                    self.network.linkSet[(current, neighbor)] = link_fw
                    self.network.nodeSet[current].outLinks.append(neighbor)
                    self.network.nodeSet[neighbor].inLinks.append(current)
                    self.network.times_all[(current, neighbor)] = 1.0
                    
                    link_bw = Link(
                        init_node=neighbor,
                        term_node=current,
                        capacity=1000,
                        length=1.0,
                        fft=1.0,
                        b=1.0,
                        power=4.0,
                        speed_limit=60.0,
                        toll=0.0,
                        linkType='regular'
                    )
                    self.network.linkSet[(neighbor, current)] = link_bw
                    self.network.nodeSet[neighbor].outLinks.append(current)
                    self.network.nodeSet[current].inLinks.append(neighbor)
                    self.network.times_all[(neighbor, current)] = 1.0
        
        preprocess_start = time.time()
        self.network.preprocess()
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        print(f"Preprocessing time for CH: {preprocess_time:.4f} seconds")
        
        test_od_pairs = [
            ('0-0', '99-99'),  
            ('0-99', '99-0'),  
            ('50-50', '99-99'),
            ('0-0', '50-50'),  
        ]
        
        for origin, destination in test_od_pairs:
            with self.subTest(origin=origin, destination=destination):
                dijkstra_start = time.time()
                dijkstra_time, _ = self.reconstruct_path_via_dijkstra(origin, destination)
                dijkstra_end = time.time()
                dijkstra_query_time = dijkstra_end - dijkstra_start
                print(f"Dijkstra query time for {origin} -> {destination}: {dijkstra_query_time:.4f} seconds")
                
                ch_start = time.time()
                ch_time, _ = self.network.bidirectional_dijkstra_CH(origin, destination)
                ch_end = time.time()
                ch_query_time = ch_end - ch_start
                print(f"CH query time for {origin} -> {destination}: {ch_query_time:.4f} seconds")
                
                self.assertAlmostEqual(dijkstra_time, ch_time, places=5,
                                    msg=f"Shortest times differ for {origin} -> {destination}")



if __name__ == '__main__':
    unittest.main()
