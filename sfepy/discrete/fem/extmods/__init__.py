"""
Compiled Cython extensions for the ``sfepy.discrete.fem`` sub-package.

The shared objects ``bases`` and ``lobatto_bases`` are produced by the
build step described in the top-level ``CMakeLists.txt``.  When they
have not been compiled yet, importing any of them raises a descriptive
:class:`ExtensionImportError` instead of a bare :class:`ImportError`.
"""

from sfepy._extmods import (
    import_extension,
    ExtensionImportError,
)


def __getattr__(name):
    if name in ('bases', 'lobatto_bases'):
        return import_extension(name, __name__)
    raise AttributeError(name)


__all__ = ['bases', 'lobatto_bases',
           'ExtensionImportError', 'import_extension']
