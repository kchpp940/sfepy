"""
Compiled C/Cython extensions shared by the ``sfepy.discrete`` sub-packages.

The actual shared objects (``_fmfield``, ``cmapping``, ``assemble``,
``cmesh``, ``crefcoors``, ``_geommech``) are produced by the build step
described in the top-level ``CMakeLists.txt``.  When they have not been
compiled yet, importing any of them will raise a descriptive
:class:`ExtensionImportError` instead of a bare :class:`ImportError`.
"""

from sfepy._extmods import (
    import_extension,
    ExtensionImportError,
)


def __getattr__(name):
    if name in ('_fmfield', 'cmapping', 'assemble', 'cmesh',
                'crefcoors', '_geommech'):
        return import_extension(name, __name__)
    raise AttributeError(name)


__all__ = ['_fmfield', 'cmapping', 'assemble', 'cmesh',
           'crefcoors', '_geommech',
           'ExtensionImportError', 'import_extension']
