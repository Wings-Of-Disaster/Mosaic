from abc import ABC, abstractmethod
import copy


class BaseServer(ABC):

    def __init__(self, context):
        self._ct_ = context
        self._role_name_ = context.role_name
        self._role_index_ = context.role_index

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def end_condition(self):
        pass


class BaseAgg(ABC):
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def merge(self, other):
        pass

    @abstractmethod
    def put(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    def get_and_clear(self):
        _res = self.get()
        self.clear()
        return _res

    def __repr__(self):
        return str(self.get())

    def __str__(self):
        return self.__repr__()

class ModelStateAvgAgg_BN(BaseAgg):
    def __init__(self):
        self.accum_model_state = None
        self.accum_weight = 0.0

    def put(self, state, weight=1.0):
        if self.accum_model_state is None:
            self.accum_model_state = copy.deepcopy(state)
            if weight != 1.0:
                for k in self.accum_model_state.keys():
                    if 'bn' not in k:
                        self.accum_model_state[k].data *= weight
        else:
            for k in self.accum_model_state.keys():
                if 'bn' not in k:
                    self.accum_model_state[k].data += state[k].data * weight
        self.accum_weight += weight
        return self

    def merge(self, other):
        if self.accum_model_state is None:
            self.accum_model_state = copy.deepcopy(other.accum_model_state)
            self.accum_weight = other.accum_weight
        else:
            for k in self.accum_model_state.keys():
                if 'bn' not in k:
                    self.accum_model_state[k] += other.accum_model_state[k]
            self.accum_weight += other.accum_weight
        return self

    def get(self):
        if self.accum_weight == 0.0:
            return None
        avg_state = copy.deepcopy(self.accum_model_state)
        for k in self.accum_model_state.keys():
            if 'bn' not in k:
                avg_state[k] = (avg_state[k] / self.accum_weight).to(avg_state[k].dtype)
        return avg_state

    def clear(self):
        self.accum_model_state = None
        self.accum_weight = 0.0


class ModelStateAvgAgg(BaseAgg):
    def __init__(self):
        self.accum_model_state = None
        self.accum_weight = 0.0

    def put(self, state, weight=1.0):
        if self.accum_model_state is None:
            self.accum_model_state = copy.deepcopy(state)
            if weight != 1.0:
                for k in self.accum_model_state.keys():
                    self.accum_model_state[k].data *= weight
        else:
            for k in self.accum_model_state.keys():
                self.accum_model_state[k].data += state[k].data * weight
        self.accum_weight += weight
        return self

    def merge(self, other):
        if self.accum_model_state is None:
            self.accum_model_state = copy.deepcopy(other.accum_model_state)
            self.accum_weight = other.accum_weight
        else:
            for k in self.accum_model_state.keys():
                self.accum_model_state[k] += other.accum_model_state[k]
            self.accum_weight += other.accum_weight
        return self

    def get(self):
        if self.accum_weight == 0.0:
            return None
        avg_state = copy.deepcopy(self.accum_model_state)
        for k in self.accum_model_state.keys():
            _data = avg_state[k]
            avg_state[k] = (_data / self.accum_weight).to(_data.dtype)
        return avg_state

    def clear(self):
        self.accum_model_state = None
        self.accum_weight = 0.0

class NumericAvgAgg(BaseAgg):
    def __init__(self):
        self.accum_numeric = 0.0
        self.accum_weight = 0.0

    def put(self, numeric, weight=1.0):
        self.accum_numeric += numeric * weight
        self.accum_weight += weight
        return self

    def merge(self, other):
        self.accum_numeric += other.accum_numeric
        self.accum_weight += other.accum_weight
        return self

    def get(self):
        if self.accum_weight == 0.0:
            return None
        return self.accum_numeric / self.accum_weight

    def clear(self):
        self.accum_numeric = 0.0
        self.accum_weight = 0.0


class BaseContext(ABC):

    def __init__(self, role_name_size_list, rank):
        self.role_size_dict = dict(role_name_size_list)
        self.worker_size = sum([size for _, size in role_name_size_list])
        self.rank = rank

        self.role_name = ""
        self.role_index = 0
        self.__init_role__(role_name_size_list)

    def __init_role__(self, role_name_size_list):
        expended_role_index = [(role_name, role_index) for role_name, role_size in role_name_size_list
                               for role_index in range(role_size)]
        self.role_name, self.role_index = expended_role_index[self.rank]

    def get_role_size(self, role_name):
        return self.role_size_dict[role_name]

    @abstractmethod
    def get_node(self, role_name, role_index=0):
        pass

    @abstractmethod
    def get_node_list(self, role_name):
        pass

    @abstractmethod
    def shutdown_cluster(self):
        pass

    @abstractmethod
    def barrier(self):
        pass


class BaseRunner(ABC):
    @abstractmethod
    def run(self):
        pass


class BaseNode(ABC):
    def __init__(self, rank):
        self._rank_ = rank
        self._deepcopy = True

    def set(self, deepcopy=True):
        self._deepcopy = deepcopy
        return self
    
    def _reset(self):
        self._deepcopy = True

    @abstractmethod
    def __getattr__(self, func_name):
        pass