# -*- coding:utf-8 -*-
import numpy as np


class Tree:
    def __init__(self, node: tuple):
        self.key = node[0]
        self.value = node[0]
        self.left_child_node = None
        self.right_child_node = None


class HuffmanTree:
    def __init__(self, nodes: list):
        self.queue = [Tree(node) for node in nodes]
        self.queue.sort(key=lambda node: node.value)

    def add(self, new_node):
        if len(self.queue) == 0:
            return new_node
        for i in range(len(self.queue)):
            if self.queue[i].value >= new_node.value:
                return self.queue[:i]+[new_node]+self.queue[i:]
            return self.queue+[new_node]

    def pop(self):
        while len(self.queue) != 1:
            key = "*"
            value = self.queue[0].value+self.queue[1].value
            new_code = Tree((key, value))
            new_node.left_child_node = self.queue.pop(0)
            new_node.right_child_node = self.queue.pop(0)
            self.queue = self.add(new_code)
        return self.queue.pop(0)
