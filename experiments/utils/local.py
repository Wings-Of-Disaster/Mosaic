import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

from experiments.utils.base import BaseContext,BaseNode
import copy
import logging
from experiments.utils.base import BaseRunner


MESSAGE_QUEUE = []


class LocalNode(BaseNode):
    def __init__(self, rank):
        super().__init__(rank)

    def __getattr__(self, func_name):

        def _send_func(*args, **kwargs):
            if self._deepcopy:
                args = copy.deepcopy(args)
                kwargs = copy.deepcopy(kwargs)
            MESSAGE_QUEUE.append((self._rank_, func_name, args, kwargs))
            self._reset()

        return _send_func


class LocalContext(BaseContext):
    def __init__(self, role_name_size_list, rank):
        super().__init__(role_name_size_list, rank)
        self.nodes = {} 
        self.__init_nodes__(role_name_size_list)

    def get_node(self, role_name, role_index=0):
        return self.nodes[role_name][role_index]

    def get_node_list(self, role_name):
        return self.nodes[role_name]

    def shutdown_cluster(self):
        for rank in range(self.worker_size):
            MESSAGE_QUEUE.append((rank, "__shutdown__", None, {}))

    def barrier(self):
        pass

    def __init_nodes__(self, role_name_size_list):
        rank = 0
        for role_name, role_size in role_name_size_list:
            for _ in range(role_size):
                _node_ = LocalNode(rank)
                rank += 1
                self.nodes.setdefault(role_name, []).append(_node_)


class LocalRunner(BaseRunner):
    def __init__(self, server_list):
        self.server_list = server_list

    def _start_servers_(self):
        for server in self.server_list:
            server.start()

    def _any_survival_(self):
        return any(self.server_list)

    def _is_survival_(self, rank):
        return self.server_list[rank] is not None

    def _kill_server(self, rank):
        if self._is_survival_(rank):
            self.server_list[rank].end()
            self.server_list[rank] = None

    def _kill_if_finished_(self, rank):
        if self._is_survival_(rank) and self.server_list[rank].end_condition():
            self._kill_server(rank)

    def _kill_finished_servers(self):
        for rank in range(len(self.server_list)):
            self._kill_if_finished_(rank)

    def _kill_all_servers(self):
        for rank in range(len(self.server_list)):
            self._kill_server(rank)

    def run(self):
        self._start_servers_()
        self._kill_finished_servers()
        while self._any_survival_():
            if len(MESSAGE_QUEUE) == 0:
                logging.warning(f"MESSAGE QUEUE is empty but there are some survival servers")
                self._kill_all_servers()
                break

            target_rank, func_name, args, kwargs = MESSAGE_QUEUE.pop(0)
            if func_name == "__shutdown__":
                self._kill_server(target_rank)
            elif self._is_survival_(target_rank):
                getattr(self.server_list[target_rank], func_name)(*args, **kwargs)
                self._kill_if_finished_(target_rank)

        MESSAGE_QUEUE.clear()
