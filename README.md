# Pytree2Safetensors
Pytree2Safetensors is a simple package to save and load JAX PyTrees to and from
Safetensors, a popular file format for saving neural network weights.

To install, run

```
pip install --upgrade pytree2safetensors
```

Pytree2Safetensors depends on `jax`, `safetensors`, and `jaxtyping`. You also need
to have at least Python 3.10

## Specification
### Serialising/Deserialising

#### `keypath2string(path: KeyPath) -> str`
Serializes a JAX key path (i.e., a path to a leaf in a pytree) to a string by joining together a string representation of each key in the path. Prefixes of these representation tell what type of key it is. A GetAttryKey is prefixed with ".", a DictKey is prefixed with "@", and a SequenceKey is prefixed with "#". If the initial key is a
GetAttryKey, the initial "." is left off.

Examples:
```python
keypath2string((GetAttrKey("layers"), SequenceKey(10), DictKey("query"),))
# => "layers#10@query
keypath2string((SequenceKey(2), GetAttrKey("layers"), SequenceKey(10), DictKey("query"),))
# => "#2.layers#10@query
```

#### `string2keypath(string: str) -> KeyPath`
Inverse of `keypath2string`

#### `pytree2dict(tree: PyTree) -> dict`
Returns a dictionary of serialized key paths mapping to leaves of the tree.

#### `dict2pytree(dictionary: dict) -> tree`
Inverse of `pytree2dict`, except that it wraps attributes in `PyTreeContainer`s instead of using the original object. This is because there is no way for the deserialiser to know what the original object was. You can use `load_into_pytree` to load weights into an initialized
pytree.

#### `PyTreeContainer`
A class which implements the bare minimum to be a node in a pytree according to JAX.

### Saving
#### `save_pytree(tree: PyTree, path: str) -> None`
Saves the pytree as a safetensors at the given path. Equivalent to
```python
safetensors.flax.save_file(pytree2dict(tree), path)
```

### Loading
#### ```load_file```
Alias of `safetensors.flax.load_file`

#### ```load_pytree(path: str) -> PyTree```
Loads a file and uses `dict2pytree` to convert the safetensors dict to a pytree.

#### ```set_weights(module: PyTree, dictionary: dict) -> PyTree```
Given a pytree module and a safetensors dict, load the weights in the safetensors dict into the module using string2keypath to determine their paths. Returns a new pytree.

#### ```load_into_pytree(module: PyTree, path: str) -> PyTree```
Equivalent to `set_weights(module, load_file(path))`.