import sys

from jax.tree_util import tree_map_with_path
from jaxtyping import PyTree
from safetensors.flax import load_file

from .serialise import string2keypath, dict2pytree
from ._shared_types import KeyEntry


load_file = load_file

def load_pytree(path: str) -> PyTree:
    d = load_file(path)
    return dict2pytree(d)

def set_weights(module: PyTree, dictionary: dict, verbose=False) -> PyTree:
    key_map = {
        key: string2keypath(key)
        for key in dictionary.keys()
    }
    reverse_key_map = {
        val: key for key, val in key_map.items()
    }
    keypath_dict = {
        key_map[key]: val
        for key, val in dictionary.items()
    }
    used = {}
    def replace_func(path, old_value):
        # We only understand how to work with DictKey, GetAttryKey, and SequenceKey
        filtered_path = tuple(
            part for part in path
            if isinstance(part, KeyEntry)
        )
        # If a part of the path is not one of these three kinds of keys, warn the user
        if path != filtered_path:
            if verbose:
                print(f"Warning: Coercing {path} to {filtered_path}", file=sys.stderr)
        
        if filtered_path in keypath_dict:
            # Check if the leaf has already been set by this function
            if filtered_path in used:
                if verbose:
                    print(f"Warning: {reverse_key_map[filtered_path]} maps to the same leaf as {used[filtered_path]}", file=sys.stderr)
            else:
                used[filtered_path] = reverse_key_map[filtered_path]
            return keypath_dict[filtered_path]
        return old_value
    
    
    result = tree_map_with_path(replace_func, module)

    # Print out warnings for keys in the dictionary that don't map to any leaves
    for key in dictionary.keys():
        if key_map[key] not in used:
            if verbose:
                print(f"Warning: key {key} is not used", file=sys.stderr)

    return result

def load_into_pytree(module: PyTree, path: str, verbose=False) -> PyTree:
    return set_weights(module, load_file(path), verbose=verbose)