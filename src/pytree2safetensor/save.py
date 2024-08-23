import jax
from jaxtyping import PyTree
from safetensors.flax import save_file
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

def node_key2sep_and_str(part):
    if isinstance(part, GetAttrKey):
        return ".", part.name
    if isinstance(part, DictKey):
        return "@", part.key
    if isinstance(part, SequenceKey):
        return "#", str(part.idx)
    raise TypeError("Unknown kind of path part", part)

def leaf_path2string(path):
    if len(path) == 0:
        return ""
    
    string_builder = [node_key2sep_and_str(path[0])[1]]

    for node_key in path[1:]:
        string_builder.extend(node_key2sep_and_str(node_key))

    return "".join(string_builder)

def tree2dict(tree: PyTree) -> dict:
    path_leaves = jax.tree_util.tree_leaves_with_path(tree)
    result = {leaf_path2string(path): leaf for path, leaf in path_leaves}
    return result

def save_pytree(tree: PyTree, path) -> None:
    d = tree2dict(tree)
    save_file(d, path)