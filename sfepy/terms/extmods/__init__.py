"""
Compiled Cython extension for the ``sfepy.terms`` sub-package.

The actual shared object ``terms`` is produced by the build step
described in the top-level ``CMakeLists.txt``.  When it has not been
compiled yet, importing it raises a descriptive
:class:`ExtensionImportError` instead of a bare :class:`ImportError`.
"""

from sfepy._extmods import (
    import_extension,
    ExtensionImportError,
)


def __getattr__(name):
    if name == 'terms':
        return import_extension(name, __name__)
    raise AttributeError(name)


__all__ = ['terms', 'ExtensionImportError', 'import_extension']
