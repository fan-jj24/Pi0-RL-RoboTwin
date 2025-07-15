from threading import Lock
from typing import Union, Iterable, Dict, List, Any
import numpy as np
import gymnasium as gym
import jax
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
    DynamicNextObsReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys, **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO

class DynamicNextObsReplayBufferDataStore(DynamicNextObsReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        **kwargs,
    ):
        DynamicNextObsReplayBuffer.__init__(
            self, observation_space, action_space, capacity, **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            return super().insert(*args, **kwargs)
        
    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)

    def latest_data_id(self) -> int:
        """返回最新插入数据的索引"""
        return self._latest_index  # DynamicNextObsReplayBuffer 中维护的最新索引

    def get_latest_data(self, from_id: int) -> Dict[str, Any]:
        """
        获取从 from_id 到当前最新数据的连续片段
        返回格式: {
            "observations": ...,
            "actions": ...,
            "rewards": ...,
            "dones": ...,
            "next_observations": ...  # 需要动态生成
        }
        """
        with self._lock:
            # 计算有效数据范围
            if from_id < 0 or from_id >= self._capacity:
                raise ValueError(f"Invalid from_id: {from_id}")
            
            # 获取当前 buffer 中的实际数据范围
            available_ids = self._get_valid_ids(from_id)
            if not available_ids:
                return None
            
            # 1. 构造基础数据
            batch_size = len(available_ids)
            data = {
                "observations": self._slice_data(self.dataset_dict["observations"], available_ids),
                "actions": self.dataset_dict["actions"][available_ids],
                "rewards": self.dataset_dict["rewards"][available_ids],
                "dones": self.dataset_dict["dones"][available_ids],
            }
            
            # 2. 动态生成 next_observations
            # data["next_observations"] = self._generate_next_observations(available_ids)
            
            return data
    
    def _get_valid_ids(self, from_id: int) -> List[int]:
        """计算从 from_id 到当前最新数据的连续索引"""
        current_size = min(self._size, self._capacity)
        if from_id >= current_size:
            return []
        
        # 处理环形缓冲区的索引绕回
        latest = self._latest_index
        ids = []
        for i in range(current_size):
            idx = (latest - i - 1 + self._capacity) % self._capacity
            if idx < from_id:
                break
            ids.append(idx)
        return list(reversed(ids))  # 按时间顺序返回
    
    def _slice_data(self, data_dict: Union[Dict, np.ndarray], indices: List[int]) -> Union[Dict, np.ndarray]:
        """递归切片嵌套结构的数据"""
        if isinstance(data_dict, dict):
            return {
                k: self._slice_data(v, indices) for k, v in data_dict.items()
            }
        elif isinstance(data_dict, np.ndarray):
            return data_dict[indices]
        else:
            raise TypeError(f"Unsupported data type: {type(data_dict)}")




def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
