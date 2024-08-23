from jax.tree_util import tree_map_with_path
from jaxtyping import PyTree
from safetensors.flax import load_file

from .serialise import string2keypath, dict2pytree


load_file = load_file

def load_pytree(path: str) -> PyTree:
    d = load_file(path)
    return dict2pytree(d)

def set_weights(module: PyTree, dictionary: dict) -> PyTree:
    keypath_dict = {
        string2keypath(key): val
        for key, val in dictionary.items()
    }
    def replace_func(path, old_value):
        return keypath_dict.get(path, old_value)
    
    result = tree_map_with_path(replace_func, module)
    return result

def load_into_pytree(module: PyTree, path: str) -> PyTree:
    return set_weights(module, load_file(path))