import heapq

class GraphLoader():
    @staticmethod
    def loadUndirectedGraph(filename):
        with open(filename) as f:
            assert(f.readline().strip() == 'UndirectedGraph')
            size, edges = map(int, f.readline().split())
            graph = UndirectedGraph(size)
            for _ in xrange(edges):
                x, y = map(int, f.readline().split())
                graph.connect(x, y)
            return graph

    @staticmethod
    def loadDirectedGraph(filename):
        with open(filename) as f:
            assert(f.readline().strip() == 'DirectedGraph')
            size, edges = map(int, f.readline().split())
            graph = DirectedGraph(size)
            for _ in xrange(edges):
                x, y = map(int, f.readline().split())
                graph.connect(x, y)
            return graph

    @staticmethod
    def loadEdgeWeightedUndirectedGraph(filename):
        with open(filename) as f:
            assert(f.readline().strip() == 'EdgeWeightedUndirectedGraph')
            size, edges = map(int, f.readline().split())
            graph = EdgeWeightedUndirectedGraph(size)
            for _ in xrange(edges):
                x, y, w = map(int, f.readline().split())
                graph.connect(x, y, w)
            return graph

    @staticmethod
    def loadEdgeWeightedDirectedGraph(filename):
        with open(filename) as f:
            assert(f.readline().strip() == 'EdgeWeightedDirectedGraph')
            size, edges = map(int, f.readline().split())
            graph = EdgeWeightedDirectedGraph(size)
            for _ in xrange(edges):
                x, y, w = map(int, f.readline().split())
                graph.connect(x, y, w)
            return graph

    @staticmethod
    def load(filename):
        with open(filename) as f:
            gtype = f.readline().strip()
        try:
            f = getattr(GraphLoader, 'load' + gtype)
            return f(filename)
        except Exception as e:
            raise ValueError("cannot load graph\n" + str(e))


class UndirectedGraph(object):
    def __init__(self, size):
        self.nodes = [[] for _ in xrange(size)]
        self.size = size

    def connect(self, idx1, idx2):
        self.nodes[idx1].append(idx2)
        self.nodes[idx2].append(idx1)

    def adjacent(self, idx):
        return self.nodes[idx]

    def bfs(self, idx):
        visited = [False] * self.size
        queue = [idx]
        visited[idx] = True
        print "visiting ", idx
        while len(queue) != 0:
            node = queue.pop(0)
            for neighbor in self.adjacent(node):
                if not visited[neighbor]:
                    print "visiting ", neighbor
                    visited[neighbor] = True
                    queue.append(neighbor)

    def dfs(self, idx):
        visited = [False] * self.size
        self._dfs(idx, visited)

    def _dfs(self, idx, visited):
        print "visiting ", idx, "neigbors", self.adjacent(idx)
        visited[idx] = True
        for neighbor in self.adjacent(idx):
            if not visited[neighbor]:
                self._dfs(neighbor, visited)

    def __str__(self):
        string = ''
        for idx in xrange(self.size):
            for neighbor in self.adjacent(idx):
                string += ('{0} -> {1}\n'.format(idx, neighbor))
        return string


class DirectedGraph(UndirectedGraph):
    def __init__(self, size):
        super(DirectedGraph, self).__init__(size)

    def connect(self, idx1, idx2):
        self.nodes[idx1].append(idx2)


class Edge(object):
    def __init__(self, weight, fr, to):
        self.weight = weight
        self.fr = fr
        self.to = to

    def any(self):
        return self.fr

    def other(self, vertex):
        if self.fr == vertex:
            return self.to
        else:
            return self.fr


class EdgeWeightedUndirectedGraph(object):
    def __init__(self, size):
        self.nodes = [[] for _ in xrange(size)]
        self.size = size

    def connect(self, idx1, idx2, weight):
        edge = Edge(weight, idx1, idx2)
        self.nodes[idx1].append(edge)
        self.nodes[idx2].append(edge)

    def adjacent(self, idx):
        return self.nodes[idx]

    def __str__(self):
        string = ''
        for idx in xrange(self.size):
            for edge in self.adjacent(idx):
                string += ('{0} -> {1} : {2}\n'
                           .format(idx, edge.other(idx), edge.weight))
        return string


class EdgeWeightedDirectedGraph(EdgeWeightedUndirectedGraph):
    def __init__(self, size):
        super(EdgeWeightedDirectedGraph, self).__init__(size)
        self.rnodes = [[] for _ in xrange(size)]

    def connect(self, idx1, idx2, weight):
        edge = Edge(weight, idx1, idx2)
        self.nodes[idx1].append(edge)
        self.rnodes[idx2].append(edge)

    def adjacent(self, idx, reverse=False):
        if reverse:
            return self.rnodes[idx]
        else:
            return self.nodes[idx]

    def _dfs_postorder(self, reverse=False):
        visited = [False] * self.size
        order = []
        for i in xrange(self.size):
            if not visited[i]:
                self._dfs_postorder_helper(i, visited, order, reverse)
        return order

    def _dfs_postorder_helper(self, idx, visited, order, reverse=False):
        print "visiting ", idx, "neigbors", map(lambda x: x.other(idx), self.adjacent(idx))
        visited[idx] = True
        for neighbor in self.adjacent(idx, reverse):
            if not visited[neighbor.other(idx)]:
                self._dfs_postorder_helper(neighbor.other(idx), visited, order, reverse)
        order.append(idx)

    def topologicalSort(self):
        order = self._dfs_postorder()
        return list(reversed(order))

    def _dfs_reachable(self, idx, visited, reachable):
        visited[idx] = True
        reachable.append(idx)
        for neighbor in self.adjacent(idx):
            if not visited[neighbor.other(idx)]:
                self._dfs_reachable(neighbor.other(idx), visited, reachable)


    def kosaraju(self):
        order = self._dfs_postorder(reverse=True)
        print order
        visited = [False] * self.size
        strong_components = []
        for i in reversed(order):
            if not visited[i]:
                reachable = []
                self._dfs_reachable(i, visited, reachable)
                strong_components.append(reachable)
        return strong_components

    def _next_vertex(self, visited, pq):
        while len(pq) != 0:
            _, vertex = heapq.heappop(pq)
            if not visited[vertex]:
                return vertex
        return None

    def _relax(self, v_from, v_to, new_distance, visited, distances, parents, pq):
        if not visited[v_to] and distances[v_to] > new_distance:
            distances[v_to] = new_distance
            parents[v_to] = v_from
            heapq.heappush(pq, (new_distance, v_to))

    def dijkstra(self, origin):
        visited = [False] * self.size
        distances = [float('inf')] * self.size
        parents = [float('inf')] * self.size
        distances[origin] = 0
        pq = [(0, origin)]
        vertex = origin
        while vertex is not None:
            visited[vertex] = True
            for edge in self.adjacent(vertex):
                self._relax(vertex, edge.other(vertex), distances[vertex] + edge.weight,
                            visited, distances, parents, pq)
            vertex = self._next_vertex(visited, pq)
        return zip(range(self.size), parents, distances)  

    def _relax_bellman(self, distances, parents, from_v, to_v, new_distance):
        if distances[to_v] > new_distance:
            distances[to_v] = new_distance
            parents[to_v] = from_v
            return True
        return False

    def bellmanford(self, origin):
        distances = [float('inf')] * self.size
        parents = [float('inf')] * self.size
        distances[origin] = 0
        for _ in xrange(self.size):
            relaxed = False
            for v in xrange(self.size):
                if distances[v] != float('inf'):
                    for e in self.adjacent(v):
                        if self._relax_bellman(distances, parents, v, e.other(v), distances[v] + e.weight):
                            relaxed = True
            if not relaxed:
                return zip(range(self.size), parents, distances)
                
        relaxed = False
        for v in xrange(self.size):
            if distances[v] != float('inf'):
                for e in self.adjacent(v):
                    if self._relax_bellman(distances, parents, v, e.other(v), distances[v] + e.weight):
                        raise ValueError('The graph contains negative cycles')
        return zip(range(self.size), parents, distances)        

    def _dfs_cycle(self, idx, visited):
        visited[idx] = 1
        ret = False
        for edge in self.adjacent(idx):
            if visited[edge.other(idx)] == 1:
                return True
            if not visited[edge.other(idx)]:
                ret = self._dfs_cycle(edge.other(idx), visited)
        visited[idx] = 2
        return ret

    def has_cycles(self):
        visited = [0] * self.size
        for i in xrange(self.size):
            if not visited[i]:
                if self._dfs_cycle(i, visited):
                    return True
        return False

class DirectedAcyclicGraph(DirectedGraph):
    def __init__(self, size):
        super(DirectedAcyclicGraph, self).__init__(size)

    def _dfs_cycle(self, idx, visited):
        visited[idx] = 1
        ret = False
        for v in self.adjacent(idx):
            if visited[v] == 1:
                return True
            if not visited[v]:
                ret = self._dfs_cycle(v, visited)
        visited[idx] = 2
        return ret

    def _has_cycles(self):
        visited = [0] * self.size
        for i in xrange(self.size):
            if not visited[i]:
                if self._dfs_cycle(i, visited):
                    return True
        return False

    def connect(self, idx1, idx2):
        self.nodes[idx1].append(idx2)
        if self._has_cycles():
            raise ValueError('The Graph has cycles')

