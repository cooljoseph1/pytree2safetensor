from jax.tree_util import DictKey, GetAttrKey, SequenceKey

KeyEntry = DictKey | GetAttrKey | SequenceKey
KeyPath = tuple[KeyEntry, ...]