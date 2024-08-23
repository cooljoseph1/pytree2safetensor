from safetensors.flax import load_file

from jax.tree_util import DictKey, GetAttrKey, SequenceKey, register_pytree_node_class, tree_map_with_path
from jaxtyping import PyTree
KeyEntry = DictKey | GetAttrKey | SequenceKey
KeyPath = tuple[KeyEntry]

def string2leaf_path(string: str) -> KeyPath:
    path = []
    word_builder = []
    sep = "."

    def append_word():
        nonlocal word_builder
        word = "".join(word_builder)
        word_builder = []

        if sep == ".":
            node_key = GetAttrKey(word)
        elif sep == "#":
            node_key = SequenceKey(int(word))
        elif sep == "@":
            node_key = DictKey(word)
        else:
            raise ValueError("Unknown separator", sep)
        
        path.append(node_key)

    for char in string:
        if char in {".", "@", "#"}:
            append_word()
            sep = char
        else:
            word_builder.append(char)
    append_word()

    return tuple(path)

@register_pytree_node_class
class PyTreeContainer:
  def __init__(self, attrs: dict = {}):
      for key, value in attrs.items():
          setattr(self, key, value)

  def tree_flatten(self):
    keys, values = zip(*(vars(self).items()))
    return (tuple(values), tuple(keys))
  
  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls()
    for key, value in zip(aux_data, children):
        setattr(obj, key, value)
    return obj
  
  def __repr__(self):
      return f"PyTreeContainer({vars(self)})"
      

def _add_leaf(tree: PyTree, path: KeyPath, leaf: any) -> PyTree:
    """
    Add a leaf in place to the given tree, returning the new tree.
    Warning: This mutates the original tree.
    """
    if len(path) == 0:
        return leaf
    
    node_key = path[0]
    rest_path = path[1:]
    
    if isinstance(node_key, GetAttrKey):
        if tree is None:
            tree = PyTreeContainer()
        assert isinstance(tree, PyTreeContainer)

        subtree = getattr(tree, node_key.name, None)
        setattr(tree, node_key.name, _add_leaf(subtree, rest_path, leaf))
        return tree
    
    if isinstance(node_key, SequenceKey):
        if tree is None:
            tree = []
        assert isinstance(tree, list)
        if len(tree) <= node_key.idx:
            tree.extend([None] * (node_key.idx - len(tree) + 1))

        subtree = tree[node_key.idx]
        tree[node_key.idx] = _add_leaf(subtree, rest_path, leaf)
        return tree

    if isinstance(node_key, DictKey):
        if tree is None:
            tree = {}
        assert isinstance(tree, dict)
        subtree = tree.get(node_key.key, None)
        tree[node_key.key] = _add_leaf(subtree, rest_path, leaf)
        return tree

def dict2tree(dictionary: dict) -> PyTree:
    tree = PyTreeContainer()
    for key, leaf in dictionary.items():
        path = string2leaf_path(key)
        tree = _add_leaf(tree, path, leaf)
    return tree

def load_pytree(path) -> PyTree:
    d = load_file(path)
    return dict2tree(d)

def set_weights(module: PyTree, weight_dict: dict) -> PyTree:
    keypath_dict = {
        string2leaf_path(key): val
        for key, val in weight_dict.items()
    }
    def replace_func(path, old_value):
        return keypath_dict.get(path, old_value)
    
    result = tree_map_with_path(replace_func, module)
    return result

def load_into_pytree(module: PyTree, path) -> PyTree:
    return set_weights(module, load_file(path))