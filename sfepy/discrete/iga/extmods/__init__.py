"""
Compiled Cython extension for the ``sfepy.discrete.iga`` sub-package.

The shared object ``igac`` is produced by the build step described in the
top-level ``CMakeLists.txt``.  When it has not been compiled yet,
importing it raises a descriptive :class:`ExtensionImportError` instead
of a bare :class:`ImportError`.
"""

from sfepy._extmods import (
    import_extension,
    ExtensionImportError,
)


def __getattr__(name):
    if name == 'igac':
        return import_extension(name, __name__)
    raise AttributeError(name)


__all__ = ['igac', 'ExtensionImportError', 'import_extension']
