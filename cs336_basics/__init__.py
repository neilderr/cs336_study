import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336")
except importlib.metadata.PackageNotFoundError:
    try:
        __version__ = importlib.metadata.version("cs336_basics")
    except importlib.metadata.PackageNotFoundError:
        pass
