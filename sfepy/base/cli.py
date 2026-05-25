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
* enabling ``debug_on_error()`` consistently,
* rendering fatal errors through the same ``output()`` channel so
  that log files capture them together with the rest of the run, and
* registering the shared parameter declarations for
  ``--debug`` / ``--format`` / ``-o`` (output filename) /
  ``--output-dir`` along with the lightweight input/output path
  checks that scripts use to report missing paths early.

Scripts are expected to call :func:`build_parser` (optionally via
:func:`ProblemConfApp.cli`) and wrap their entry point with
:func:`run_main` (or the convenience decorator :func:`main`).

Parameter-declaration helpers (the ``add_*`` family) are designed to
be used directly in ``main()``; their help texts come from the
module-level :data:`helps` dict so that repeated flags always describe
the same behaviour.
"""
from __future__ import absolute_import

import os
import os.path as op
import sys
import traceback
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    ArgumentTypeError,
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
    'add_output_format_arg',
    'add_mesh_format_arg',
    'add_output_dir_arg',
    'add_output_filename_arg',
    'add_logging_args',
    'APP_OPTIONS_DEFAULTS',
    'build_app_options',
    'configure_output',
    'setup_debug',
    'apply_debug_option',
    'set_output_prefix',
    'resolve_output_path',
    'check_input_file',
    'check_output_dir',
    'parse_comma_list',
    'fatal',
    'run_main',
    'main',
    'AppContext',
    'helps',
    'check_cli_patterns',
]


helps = {
    'debug':
    'automatically start debugger when an exception is raised',
    'filename':
    'basename of output file(s) [default: <basename of input file>]',
    'output_filename':
    'output file name [default: %(default)s]',
    'output_format':
    'output figure file format (supported by the matplotlib backend used) '
    '[default: %(default)s]',
    'format':
    'output mesh format (overrides output file name extension)',
    'output_dir':
    'output directory',
}


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
                        default=False, help=helps['debug'])


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


def apply_debug_option(options):
    """Alias of :func:`setup_debug` matching the name used by scripts.

    Kept so that callers written against the newer ``_cli_common`` name
    continue to work without changes.
    """
    setup_debug(options)


def add_output_format_arg(parser, default=None, dest='output_format',
                          metavar='format', short='-f', help=None):
    """Add ``-f/--format`` for figure/output formats.

    Parameters
    ----------
    default : str, optional
    dest : str
        argparse destination name.  Defaults to ``'output_format'``
        to match the historical ``simple.py`` / ``probe.py`` usage.
    short : str or None
        Short option, e.g. ``'-f'``.  Pass ``None`` to omit.
    help : str, optional
        Help text.  Defaults to :data:`helps['output_format']`.
    """
    if help is None:
        help = helps['output_format']
    args = []
    if short is not None:
        args.append(short)
    args.append('--format')
    kwargs = dict(metavar=metavar, action='store', dest=dest,
                  default=default, help=help)
    if default is not None:
        kwargs['type'] = str
    parser.add_argument(*args, **kwargs)


def add_mesh_format_arg(parser, default=None, dest='format',
                        metavar='format', short='-f'):
    """Add ``-f/--format`` used by mesh generators and ``convert_mesh``.

    Unlike :func:`add_output_format_arg` this one defaults to
    ``dest='format'`` and always registers ``type=str``, matching the
    behaviour expected by ``blockgen``/``cylindergen``/``combine_meshes``/
    ``convert_mesh``.
    """
    args = []
    if short is not None:
        args.append(short)
    args.append('--format')
    parser.add_argument(*args, metavar=metavar, action='store', type=str,
                        dest=dest, default=default, help=helps['format'])


def add_output_dir_arg(parser, default=None, dest='output_dir',
                       metavar='path', short='-o'):
    """Add ``-o/--output-dir``."""
    args = []
    if short is not None:
        args.append(short)
    args.append('--output-dir')
    parser.add_argument(*args, metavar=metavar, action='store',
                        dest=dest, default=default, help=helps['output_dir'])


def add_output_filename_arg(parser, default=None, dest='output_filename',
                            metavar='filename', short='-o', help=None):
    """Add ``-o <filename>`` used by mesh generators and probes.

    Parameters
    ----------
    default : str, optional
    dest : str
        argparse destination name.  Defaults to ``'output_filename'``.
        Callers such as ``simple.py`` pass ``dest='output_filename_trunk'``.
    short : str or None
        Short option, e.g. ``'-o'``.  Pass ``None`` to omit.
    help : str, optional
        Help text.  Defaults to :data:`helps['output_filename']`.
    """
    if help is None:
        help = helps['output_filename']
    args = []
    if short is not None:
        args.append(short)
    kwargs = dict(metavar=metavar, action='store', dest=dest,
                  default=default, help=help)
    parser.add_argument(*args, **kwargs)


def set_output_prefix(prefix):
    """Set :data:`sfepy.base.base.output.prefix` to *prefix*.

    This is a thin wrapper kept for scripts that want to set the prefix
    without bringing in the heavier :func:`configure_output`.
    """
    output.prefix = prefix


def check_input_file(filename):
    """Raise :class:`argparse.ArgumentTypeError` if *filename* does not exist.

    Call this early in ``main()`` so that missing input files are reported
    with a clear, uniform message instead of letting the error surface as
    a ``FileNotFoundError`` deeper in the call stack.
    """
    if filename is None:
        return filename
    if not op.exists(filename):
        raise ArgumentTypeError(
            'input file does not exist: %s' % filename)
    return filename


def check_output_dir(dirname, create=False):
    """Raise :class:`argparse.ArgumentTypeError` if *dirname* is not a dir.

    When *create* is ``True``, the directory is created (with
    ``os.makedirs(..., exist_ok=True)``) instead of raising.
    """
    if dirname is None:
        return dirname
    if create:
        os.makedirs(dirname, exist_ok=True)
    elif not op.isdir(dirname):
        raise ArgumentTypeError(
            'output directory does not exist: %s' % dirname)
    return dirname


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


def _strip_string_literals(source):
    """Replace triple-quoted strings and comments with spaces (same
    length) so that regex matches don't fire inside docstrings / comments.
    Regular single/double-quoted strings are **kept** so that patterns
    like ``--version`` / ``--debug`` can still be detected.
    """
    out = []
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        # Triple-quoted strings.
        if c in ('"', "'") and i + 2 < n and source[i:i + 3] in ('"""', "'''"):
            quote = source[i:i + 3]
            j = source.find(quote, i + 3)
            if j == -1:
                j = n
            else:
                j += 3
            out.append(' ' * (j - i))
            i = j
            continue
        # Comments (# ...).
        if c == '#':
            j = source.find('\n', i)
            if j == -1:
                j = n
            out.append(' ' * (j - i))
            i = j
            continue
        out.append(c)
        i += 1
    return ''.join(out)


# Patterns that should be replaced by sfepy.base.cli helpers.
# Each entry is (regex, description, preferred_helper).
_CLI_DRIFT_PATTERNS = [
    # --debug
    (r"parser\.add_argument\s*\(\s*['\"]--debug['\"]",
     'hand-written --debug',
     'add_debug_arg(parser)'),
    # --version
    (r"parser\.add_argument\s*\(\s*['\"]--version['\"]",
     'hand-written --version',
     'add_version_arg(parser)'),
    # --format
    (r"parser\.add_argument\s*\([^)]*['\"]--format['\"]",
     'hand-written --format',
     'add_output_format_arg(parser) / add_mesh_format_arg(parser)'),
    # --output-dir
    (r"parser\.add_argument\s*\([^)]*['\"]--output-dir['\"]",
     'hand-written --output-dir',
     'add_output_dir_arg(parser)'),
    # output.prefix =
    (r"output\.prefix\s*=",
     'hand-written output.prefix assignment',
     'set_output_prefix(...)'),
    # debug_on_error() called directly (not inside setup_debug/apply_debug_option)
    (r"debug_on_error\s*\(",
     'direct debug_on_error() call',
     'apply_debug_option(options) / setup_debug(options)'),
    # -o with output-related dest/help (not -o as an arbitrary short option)
    (r"parser\.add_argument\s*\(\s*['\"]-o['\"][^)]*(?:['\"]--output['\"]|dest\s*=\s*['\"][^'\"]*output|help\s*=\s*[^)]*output)",
     'hand-written -o/--output (file/dir)',
     'add_output_filename_arg(parser) / add_output_dir_arg(parser)'),
]


def check_cli_patterns(scripts_dir=None):
    """Scan ``sfepy/scripts/`` for hand-written CLI declarations that
    should use :mod:`sfepy.base.cli` helpers.

    Prints a report to stdout.  Returns ``True`` if no issues were found,
    ``False`` otherwise.

    Parameters
    ----------
    scripts_dir : str, optional
        Directory to scan.  Defaults to ``<sfepy>/scripts``.
    """
    import re
    import glob

    if scripts_dir is None:
        scripts_dir = op.join(op.dirname(__file__), '..', 'scripts')
    scripts_dir = op.normpath(scripts_dir)

    issues = []
    for filepath in sorted(glob.glob(op.join(scripts_dir, '*.py'))):
        filename = op.basename(filepath)
        if filename in ('__init__.py',):
            continue
        # Skip this module itself.
        if op.abspath(filepath) == op.abspath(__file__):
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as fh:
                raw = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        code = _strip_string_literals(raw)
        file_issues = []
        for regex, desc, helper in _CLI_DRIFT_PATTERNS:
            for m in re.finditer(regex, code):
                lineno = code.count('\n', 0, m.start()) + 1
                file_issues.append((lineno, desc, helper))
        if file_issues:
            issues.append((filename, file_issues))

    if not issues:
        print('check_cli_patterns: no issues found in %s' % scripts_dir)
        return True

    print('check_cli_patterns: %d file(s) with issues' % len(issues))
    print()
    for filename, findings in issues:
        print('  %s:' % filename)
        for lineno, desc, helper in findings:
            print('    line %d: %s  ->  use %s' % (lineno, desc, helper))
        print()
    return False


def _cli_main():
    """Entry point for ``python -m sfepy.base.cli`` — runs the drift
    scan on ``sfepy/scripts``."""
    ok = check_cli_patterns()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    _cli_main()
