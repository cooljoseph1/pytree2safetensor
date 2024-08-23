import jax
from jaxtyping import PyTree
from jax.tree_util import DictKey, GetAttrKey, SequenceKey, register_pytree_node_class
from ._shared_types import KeyEntry, KeyPath

###################
# Serialisation #
###################

def _node_key2sep_and_str(part: KeyEntry) -> tuple[str, str]:
    if isinstance(part, GetAttrKey):
        return ".", part.name
    if isinstance(part, DictKey):
        return "@", part.key
    if isinstance(part, SequenceKey):
        return "#", str(part.idx)
    raise TypeError("Unknown kind of path part", part)

def keypath2string(path: KeyPath) -> str:
    if len(path) == 0:
        return ""
    

    string_builder = []

    for node_key in path:
        string_builder.extend(_node_key2sep_and_str(node_key))

    if string_builder[0] == ".":
        string_builder = string_builder[1:]

    return "".join(string_builder)

def pytree2dict(tree: PyTree) -> dict:
    path_leaves = jax.tree_util.tree_leaves_with_path(tree)
    result = {keypath2string(path): leaf for path, leaf in path_leaves}
    return result

###################
# Deserialisation #
###################

def string2keypath(string: str) -> KeyPath:
    path = []
    word_builder = []
    sep = string[0] if string[0] in {".", "@", "#"} else "."

    def append_word():
        nonlocal word_builder
        word = "".join(word_builder)
        word_builder = []

        if sep == ".":
            node_key = GetAttrKey(word)
        elif sep == "@":
            node_key = DictKey(word)
        elif sep == "#":
            node_key = SequenceKey(int(word))
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
    
    raise TypeError("Unknown kind of node key", node_key)

def dict2pytree(dictionary: dict) -> PyTree:
    tree = None
    for key, leaf in dictionary.items():
        path = string2keypath(key)
        tree = _add_leaf(tree, path, leaf)
    return tree