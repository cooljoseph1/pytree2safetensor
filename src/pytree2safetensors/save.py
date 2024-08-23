from jaxtyping import PyTree
from safetensors.flax import save_file

from .serialise import pytree2dict

def save_pytree(tree: PyTree, path: str) -> None:
    d = pytree2dict(tree)
    save_file(d, path)
