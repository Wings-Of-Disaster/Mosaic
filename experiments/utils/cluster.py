import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

from mpi4py import MPI
from experiments.utils.base import BaseRunner,BaseNode,BaseContext


class ClusterContext(BaseContext):
    def __init__(self, role_name_size_list, rank):
        super().__init__(role_name_size_list, rank)
        self.comm = MPI.COMM_WORLD

        self.nodes = {} 
        self.__init_nodes__(role_name_size_list)

    def get_node(self, role_name, role_index=0):
        return self.nodes[role_name][role_index]

    def get_node_list(self, role_name):
        return self.nodes[role_name]

    def shutdown_cluster(self):
        for rank in range(self.worker_size):
            self.comm.isend(("__shutdown__", None, None), dest=rank)

    def barrier(self):
        self.comm.barrier()

    def __init_nodes__(self, role_name_size_list):
        rank = 0
        for role_name, role_size in role_name_size_list:
            for _ in range(role_size):
                _node_ = ClusterNode(rank)
                rank += 1
                self.nodes.setdefault(role_name, []).append(_node_)


class ClusterRunner(BaseRunner):
    def __init__(self, server):
        self.server = server
        self.comm = MPI.COMM_WORLD

    def run(self):
        self.server.start()
        while not self.server.end_condition():
            func_name, args, kwargs = self.comm.recv()
            if func_name == "__shutdown__":
                break
            getattr(self.server, func_name)(*args, **kwargs)
        self.server.end()
        self.comm.Barrier()


class _RequestManager:
    def __init__(self):
        self.req_list = []

    def put(self, req):
        self.req_list.append(req)
        self._expire_()

    def _expire_(self):
        self.req_list = [req for req in self.req_list if not req.Get_status()]

_REQ_MANAGER_ = _RequestManager()

class ClusterNode(BaseNode):
    def __init__(self, rank):
        super().__init__(rank)
        self._comm_ = MPI.COMM_WORLD

    def __getattr__(self, func_name):
        return lambda *args, **kwargs: _REQ_MANAGER_.put(self._comm_.isend((func_name, args, kwargs), dest=self._rank_))