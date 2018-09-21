import math
import numpy as np
import networkx as nx
import gym
from gym import error, spaces, utils
from gym.utils import seeding
MAX_ARRAY_LENGTH = 100
class Problem:
    def __init__(self, input):
        if input[1] = None:
            # random graph
            self.num_nodes = input[0]
            
            self.primalbound = np.inf
            self.r1star = np.inf
            self.finalpath = []
            self.graph = nx.DiGraph()
            
            self.r1 = 'r1'
            self.cost = 'cost'
            H = nx.grid_2d_graph(self.num_nodes/4,4).to_directed()
            node_dict = {}
            for (idx, node) in enumerate(H.nodes):
                node_dict[node] = idx
                self.graph.add_node(idx,visited=0,r11=np.inf,r12=np.inf,r13=np.inf,c1=np.inf,c2=np.inf,c3=np.inf)
            for (u, v) in H.edges:
                res1 = np.random.random()
                cost = np.random.random()
                self.graph.add_edge(node_dict[u],node_dict[v])
                self.graph[node_dict[u]][node_dict[v]][self.r1] = res1
                self.graph[node_dict[u]][node_dict[v]][self.cost] = cost
            OD = np.random.choice(self.num_nodes, 2, replace=False)            
            self.start = OD[0]
            self.dest  = OD[1]
            self.R1underbar = nx.shortest_path_length(self.graph, target=self.dest, weight=self.r1)
            self.Cunderbar = nx.shortest_path_length(self.graph, target=self.dest, weight=self.cost)
            self.maxR1 = self.R1underbar[self.start]*1.2
            self.optimal = None
            for path in nx.shortest_simple_paths(self.graph, self.start, self.dest, weight=self.cost):
                cost = 0.0
                res1 = 0.0
                for (idx, n) in enumerate(path):
                    if idx > 0:
                        cost += self.graph[path[idx-1]][n][self.cost]
                        res1 += self.graph[path[idx-1]][n][self.r1]
                if res1 < self.maxR1:
                    self.optimal = cost
                    break
            
class ProblemPool:
    def __init__(self, filename=None):
        if filename == None:
            #random graphs
            self.num_nodes = 12
            self.maxlevel = 10
            self.xrange = 100
            self.num_probs = 20
            self.problems = []
            for i in range(self.num_probs):
                self.problems.append(Problem((self.num_nodes, None)))
            
    def reset(self):
class CsppEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self):
        self.problempool = # load all the graph examples
        self.prunedistance = 1 # the maximum level to jump when pruning
        self.pointer = 1
        self.ppointer = 0
        self.nodelist = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.levellist = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.r1list = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.costlist = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.path = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.action_space = None
        self.obervation_space = spaces.Box(low=0.0, high=2.0, shape=(self.problempool.num_nodes, 2*self.problempool.num_nodes+4), dtype=np.float32)
        self.viewer = None
    def _draw_obs(self, curnode):
        actionopts = np.zeros((self.problem.num_nodes,), dtype=np.float32)
        for (index, value) in enumerate(self.nodelist[0:self.pointer]):
            if self.levellist[index] >= self.levellist[0:self.pointer][-1]-self.prunedistance:
                # TODO check graph u,v
                actionopts[value] = 1
        nodestate = np.zeros((self.problem.num_nodes,3), dtype=np.float32)
        nodestate[self.problem.dest, 0] = 2.0
        nodestate[:, 1] = self.problem.maxR1
        #TODO change nodestate cost layer
        #for index in range(self.problem.num_nodes):
        #    nodestate[index, 2] = self.problem.mincost[index]/self.problem.mincost[self.problem.dest]
        for (index, value) in enumerate(self.nodelist[0:self.pointer]):
            if value != self.problem.dest:
                nodestate[value, 0] = self.levellist[index]/float(self.problempool.maxlevel)
            else:
                nodestate[value, 0] += self.levellist[index]/float(self.problempool.maxlevel)
            nodestate[value, 1] = self.r1list[index]
            nodestate[value, 2] = self.costlist[index]
        for p in self.path[0:self.ppointer]:
            nodestate[p, 0] = 1.0

        edgestate = self.edgestate
        obs = np.concatenate((nodestate, actionopts[:, np.newaxis], edgestate), axis=1)
        reward = 0.0
        if self.pointer == 0:
            reward = -1.0
        if curnode == self.dest:
            reward = 1.0 + np.exp(-np.abs(self.problem.optimal-self.problem.primalbound))
        #if self.levellist[index] == self.levellist[0:self.pointer][-1]:
        #    reward
        done = False
        if self.pointer == 0 or curnode == self.dest:
            done = True
        info = {}
        return obs, reward, done, info
    def _respond_to_action(self, action):
        assert action in self.nodelist[0:self.pointer], "%r (%s) invalid"%(action, type(action))
        # get the last occurrence, this favors pulse over prune
        index = np.where(self.nodelist[0:self.pointer] == action)[0][-1]
        assert self.levellist[index] >= (self.levellist[0:self.pointer][-1]-self.prunedistance), "prune out of distance"
        self.pointer = index + 1
        return (self.nodelist[self.pointer-1],
                self.levellist[self.pointer-1],
                self.r1list[self.pointer-1],
                self.costlist[self.pointer-1])
    def _pulse(self, curr1, curcost, curlevel, curnode):
        if curnode != self.problem.dest:
            self._change_labels(curr1, curcost, curnode)
            #if self.problem.visited[curnode] == 0:
            assert self.problem.graph.node[curnode]['visited'] == 0 , "Current node is appears on the path, making path a loop"            
            self.ppointer += 1
            self.path[self.ppointer-1] = curnode
            self.problem.visited[curnode] = 1
            for (u, v) in self.problem.graph.out_edges(curnode):
                newr1 = curr1 + self.problem.graph[u][v][self.problem.r1]
                newcost = curcost + self.problem.graph[u][v][self.problem.cost]
                newlevel = curlevel + 1
                if (newr1 <= (self.problem.maxR1-self.problem.R1underbar[v])) and
                    (newcost <= (self.problem.primalbound-self.problem.Cunderbar[v])) and
                    (not _check_labels(newr1, newcost, v)) and
                    (self.problem.graph.node[curnode]['visited'] == 0):
                    # pulse
                    self.pointer += 1
                    self.nodelist[self.pointer-1] = v
                    self.levellist[self.pointer-1] = newlevel
                    self.r1list[self.pointer-1] = newr1
                    self.costlist[self.pointer-1] = newcost
                elif (self.problem.graph.node[curnode]['visited'] == 1):
                    # update visited node on loop
                    self._change_labels(newr1, newcost, v)
      else:
            self.ppointer += 1
            self.path[self.ppointer-1] = curnode
            if (curcost <= self.problem.primalbound) and (curr1 <= self.problem.maxR1):
                self.problem.finalpath = self.path[0:self.ppointer]
                self.problem.r1star = curr1
                self.problem.primalbound = curcost
    def _change_labels(self, r1, c, v):
            if (c <= self.problem.graph.nodes[v]['c1']):
                self.problem.graph.nodes[v]['c1'] = c
                self.problem.graph.nodes[v]['r11'] = r1
            elif (r1 <= self.problem.graph.nodes[v]['r12']):
                self.problem.graph.nodes[v]['c2'] = c
                self.problem.graph.nodes[v]['r12'] = r1
            else:
                self.problem.graph.nodes[v]['c3'] = c
                self.problem.graph.nodes[v]['r13'] = r1
    def _check_labels(self, r1, c, v):
            if ((r1>=self.problem.graph.nodes[v]['r11'] and
                 c>=self.problem.graph.nodes[v]['c1']) or
                (r1>=self.problem.graph.nodes[v]['r12'] and
                 c>=self.problem.graph.nodes[v]['c2']) or
                (r1>=self.problem.graph.nodes[v]['r13'] and
                 c>=self.problem.graph.nodes[v]['c3'])):
                return True
            return False
    def step(self, action):
        curnode, curlevel, curr1, curcost = self._respond_to_action(action)
        if self.ppointer >= curlevel:
            for i in range(curlevel-1, self.ppointer):
                self.problem.visited[self.path[i]] = 0
            self.ppointer = curlevel - 1
        self._pulse(curr1, curcost, curlevel, curnode):
        return self._draw_obs(curnode)
    def reset(self):
        self.problem = self.problempool.reset()
        #populate edgestate
        self.edgestate1 = np.zeros((self.problem.num_nodes, self.problem.num_nodes), dtype=np.float32)
        self.edgestate2 = np.zeros((self.problem.num_nodes, self.problem.num_nodes), dtype=np.float32)
        for (u, v) in self.problem.graph.edges:
            self.edgestate1[u, v] = self.problem.graph[u][v][self.problem.r1]
            self.edgestate2[u, v] = self.problem.graph[u][v][self.problem.cost]
        self.edgestate = np.concatenate((self.edgestate1, self.edgestate2), axis=-1)
        #Reset the graph to initial point
        self.pointer = 1
        self.ppointer = 0
        self.nodelist[self.pointer-1] = self.problem.start
        self.levellist[self.pointer-1] = 1
        self.r1list[self.pointer-1] = 0.0
        self.costlist[self.pointer-1] = 0.0
        curnode = self.nodelist[self.pointer-1]
        curlevel = self.levellist[self.pointer-1]
        curr1 = self.r1list[self.pointer-1]
        curcost = self.costlist[self.pointer-1]
        self.pointer -= 1
        self._pulse(curr1, curcost, curlevel, curnode)
        return self._draw_obs(curnode)
    def render(self, mode='human', close=False):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            world_width = self.problempool.xrange
            super(MyEnv, self).render(mode=mode)
        elif mode == 'rgb_array':
            super(MyEnv, self).render(mode=mode)
        else:
            super(MyEnv, self).render(mode=mode)
