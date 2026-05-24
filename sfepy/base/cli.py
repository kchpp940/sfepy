"""
Unified CLI support layer for SfePy command-line scripts.

This module consolidates the boilerplate that is repeated across
``sfepy/scripts/*.py``:

* building an ``argparse`` parser with the standard ``--version`` /
  ``--debug`` flags and :class:`RawDescriptionHelpFormatter`,
* configuring the global :func:`sfepy.base.base.output` logger
  (``--log`` / ``--quiet`` / prefix),
* resolving input/output paths against the problem ``output_dir``
  (``--auto-dir`` / ``--same-dir``),
* enabling ``debug_on_error()`` consistently, and
* rendering fatal errors through the same ``output()`` channel so
  that log files capture them together with the rest of the run.

Scripts are expected to call :func:`build_parser` (optionally via
:func:`ProblemConfApp.cli`) and wrap their entry point with
:func:`run_main` (or the convenience decorator :func:`main`).
"""
from __future__ import absolute_import

import os
import sys
import traceback
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    RawDescriptionHelpFormatter,
)
from contextlib import contextmanager

import sfepy
from sfepy.base.base import output

__all__ = [
    'SfepyHelpFormatter',
    'build_parser',
    'add_version_arg',
    'add_debug_arg',
    'add_logging_args',
    'APP_OPTIONS_DEFAULTS',
    'build_app_options',
    'configure_output',
    'setup_debug',
    'resolve_output_path',
    'parse_comma_list',
    'fatal',
    'run_main',
    'main',
    'AppContext',
]


APP_OPTIONS_DEFAULTS = dict(
    output_filename_trunk=None,
    output_format=None,
    save_ebc=False,
    save_ebc_nodes=False,
    save_regions=False,
    save_regions_as_groups=False,
    solve_not=False,
)


def build_app_options(**overrides):
    """Create a Struct with the standard application options.

    The returned Struct carries the fields consumed by
    :class:`PDESolverApp` and related application classes (output
    file naming, save toggles, etc.).  Defaults match what
    ``solve_pde()`` has historically used, and any keyword
    argument overrides the corresponding default.

    Typical use in a script::

        cli_opts = parser.parse_args()
        app_opts = build_app_options(**vars(cli_opts))
        app = PDESolverApp(conf, app_opts, output_prefix)

    Passing ``**vars(cli_opts)`` only overrides fields that exist
    in both the argparse Namespace and the defaults; unknown keys
    are silently ignored.
    """
    from sfepy.base.base import Struct
    known = dict(APP_OPTIONS_DEFAULTS)
    for key, val in overrides.items():
        if key in known:
            known[key] = val
    return Struct(**known)


class SfepyHelpFormatter(RawDescriptionHelpFormatter,
                          ArgumentDefaultsHelpFormatter):
    """argparse formatter that keeps ``__doc__`` formatting *and* prints
    argument defaults (RawDescription does not show defaults by default)."""


def build_parser(description=None, formatter_class=None,
                 add_common=True, **kwargs):
    """Create a pre-configured :class:`ArgumentParser`.

    Parameters
    ----------
    description : str, optional
        Parser description.  Defaults to the caller's module ``__doc__``
        when available.
    formatter_class : type, optional
        Defaults to :class:`SfepyHelpFormatter`.
    add_common : bool
        When True (default) the standard ``--version`` and ``--debug``
        arguments are added automatically.

    Returns
    -------
    parser : ArgumentParser
    """
    if formatter_class is None:
        formatter_class = SfepyHelpFormatter
    parser = ArgumentParser(description=description,
                            formatter_class=formatter_class, **kwargs)
    if add_common:
        add_version_arg(parser)
        add_debug_arg(parser)
    return parser


def add_version_arg(parser):
    """Add ``--version`` wired to ``sfepy.__version__``."""
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + sfepy.__version__)


def add_debug_arg(parser):
    """Add ``--debug`` flag (triggers :func:`debug_on_error`)."""
    parser.add_argument('--debug',
                        action='store_true', dest='debug',
                        default=False,
                        help='automatically start debugger when an '
                             'exception is raised')


def add_logging_args(parser, prefix=None, dest_prefix=''):
    """Add the standard logging flags: ``--log`` and ``-q/--quiet``.

    Parameters
    ----------
    prefix : str, optional
        If given, an extra ``--prefix`` argument is added that overrides
        the ``output.prefix`` string.
    dest_prefix : str
        Optional prefix applied to the ``dest`` names (e.g. ``log_``)
        when a script needs more than one logging group.
    """
    parser.add_argument('--log', metavar='file',
                        action='store', dest=dest_prefix + 'log',
                        default=None,
                        help='log all messages to specified file '
                             '(existing file will be overwritten!)')
    parser.add_argument('-q', '--quiet',
                        action='store_true', dest=dest_prefix + 'quiet',
                        default=False,
                        help='do not print any messages to screen')
    if prefix is not None:
        parser.add_argument('--prefix', metavar='str',
                            action='store',
                            dest=dest_prefix + 'output_prefix',
                            default=prefix,
                            help='prefix string for output messages')


def configure_output(options, prefix=None):
    """Apply ``--log`` / ``--quiet`` / ``--prefix`` to the global logger.

    Parameters
    ----------
    options : argparse.Namespace
        Expected attributes: ``log``, ``quiet``, and optionally
        ``output_prefix``.
    prefix : str, optional
        Fallback prefix when ``options.output_prefix`` is not present.
    """
    log_file = getattr(options, 'log', None)
    quiet = bool(getattr(options, 'quiet', False))
    output.set_output(filename=log_file,
                      quiet=quiet,
                      combined=log_file is not None)
    chosen_prefix = getattr(options, 'output_prefix', None) or prefix
    if chosen_prefix is not None:
        output.prefix = chosen_prefix


def setup_debug(options):
    """Enable :func:`debug_on_error` when ``options.debug`` is True."""
    if getattr(options, 'debug', False):
        from sfepy.base.base import debug_on_error
        debug_on_error()


def resolve_output_path(filename, options, output_dir=None):
    """Resolve an output path taking ``--auto-dir`` / ``--same-dir``
    and ``--output-dir`` into account.

    Parameters
    ----------
    filename : str
        Original (possibly relative) path.
    options : argparse.Namespace
        Must provide ``auto_dir`` and/or ``same_dir`` booleans when the
        corresponding flags were registered.
    output_dir : str, optional
        When ``options.auto_dir`` is True this directory is prepended
        to *filename*.

    Returns
    -------
    str : The resolved absolute path.
    """
    if filename is None:
        return filename
    if getattr(options, 'auto_dir', False) and output_dir:
        filename = os.path.join(output_dir, filename)
    if getattr(options, 'same_dir', False):
        filename = os.path.join(os.path.dirname(filename) or '.',
                                os.path.basename(filename))
    return os.path.normpath(os.path.abspath(filename))


def parse_comma_list(value, converter=str):
    """Parse a ``'a,b,c'`` string into a list, returning ``None`` for
    ``None`` input."""
    if value is None:
        return None
    return [converter(part.strip()) for part in value.split(',') if part]


def fatal(message, exc=None, code=1):
    """Print a fatal error through :func:`output` and exit.

    The message is routed through ``output`` so that it is captured by
    the log file configured via :func:`configure_output`.
    """
    output('fatal error:', message)
    if exc is not None:
        output('exception: %s: %s' % (type(exc).__name__, exc))
        output('traceback:')
        for line in traceback.format_exc().splitlines():
            output('  ' + line)
    sys.exit(code)


def run_main(func, *args, **kwargs):
    """Call ``func(*args, **kwargs)`` with a consistent exception handler.

    Any unhandled exception is reported via :func:`fatal` so that it
    lands in the log file when one was configured.
    """
    try:
        return func(*args, **kwargs)
    except SystemExit:
        raise
    except BaseException as exc:
        fatal('uncaught exception in %s' % getattr(func, '__name__', '?'),
              exc=exc)


def main(func):
    """Decorator equivalent to ``if __name__ == '__main__': run_main(func)``.

    Usage::

        @main
        def main():
            ...
    """
    import functools

    @functools.wraps(func)
    def _wrapper():
        if func.__module__ == '__main__':
            run_main(func)
    return _wrapper


class AppContext(object):
    """Lightweight context manager that wires the common CLI pieces.

    Typical usage in a script entry point::

        with AppContext(options, prefix='mytool:') as ctx:
            ...  # run tool body

    ``setup_debug`` and ``configure_output`` are executed on enter; the
    context also offers convenience helpers for path resolution.
    """

    def __init__(self, options, prefix=None):
        self.options = options
        self.prefix = prefix

    def __enter__(self):
        setup_debug(self.options)
        configure_output(self.options, prefix=self.prefix)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None or exc_type is SystemExit:
            return False
        fatal('uncaught exception', exc=exc_val)
        return True

    def resolve(self, filename, output_dir=None):
        return resolve_output_path(filename, self.options,
                                   output_dir=output_dir)
