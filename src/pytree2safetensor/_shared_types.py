from jaxtyping import DictKey, GetAttrKey, SequenceKey,

KeyEntry = DictKey | GetAttrKey | SequenceKey
KeyPath = tuple[KeyEntry, ...]