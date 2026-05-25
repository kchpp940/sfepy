import sys
import os
import shutil
import sysconfig
from warnings import warn

def get_top_dir():
    """
    Return SfePy installation directory information.
    """
    import os.path as op

    pkg_dir = op.normpath(op.realpath(op.dirname(__file__)))

    candidates = []
    for up_dir in ('..', '.'):
        candidates.append(op.normpath(op.realpath(op.join(pkg_dir, up_dir))))
    for prefix in (sys.prefix, sys.base_prefix):
        candidates.append(op.normpath(op.realpath(op.join(prefix, 'sfepy'))))

    top_dir = None
    in_source_tree = False
    for candidate in candidates:
        if op.isfile(op.join(candidate, 'LICENSE')):
            top_dir = candidate
            rel = op.relpath(candidate, pkg_dir)
            in_source_tree = (rel == '..')
            break

    if top_dir is None:
        print('Warning: cannot determine SfePy top level directory.')
        top_dir = pkg_dir

    return top_dir, in_source_tree

top_dir, in_source_tree = get_top_dir()


msg_unknown_os = """could not determine operating system!
try setting it in site_cfg.py manually, see site_cfg_template.py"""

msg_numpydoc = """could not find numpydoc!
If it is installed in a non-standard location, try setting it in
site_cfg.py manually."""

has_attr = lambda obj, attr: obj and hasattr(obj, attr)


def compose_system_compile_flags(is_posix: bool) -> list:
    """
    Provides a list of compile flags that tries to be as similar as possible
    to what Python itself was built with. This has been done historically by
    numpy.distutils (now deprecated) and a squeezed version of it is brought
    over to here.
    """
    if not is_posix:
        return []

    cflags, configure_cppflags, configure_cflags = sysconfig.get_config_vars(
        'CFLAGS', 'CONFIGURE_CPPFLAGS', 'CONFIGURE_CFLAGS')

    return (cflags + ' ' + configure_cppflags + ' ' + configure_cflags).split()


class Config:
    """
    Read and access the site configuration from the current directory.
    """

    def __init__(self):
        from importlib.machinery import SourceFileLoader
        from sfepy.base.base import sfepy_config_dir

        cwd = os.getcwd()
        config_filename = os.path.join(cwd, 'site_cfg.py')
        if not os.path.exists(config_filename):
            config_filename = os.path.join(sfepy_config_dir, 'site_cfg.py')

        if not os.path.exists(config_filename):
            try:
                shutil.copyfile(os.path.join(top_dir,
                                             'site_cfg_template.py'),
                                config_filename)

            except:
                pass

        try:
            self.site_cfg = (SourceFileLoader('site_cfg', config_filename)
                             .load_module())

        except FileNotFoundError:
            print(f'Warning: SfePy site configuration file ({config_filename})'
                  ' not found.')
            self.site_cfg = None

    def python_version(self):
        return "%d.%d" % tuple(sys.version_info[:2])

    def python_include(self):
        if (has_attr(self.site_cfg, 'python_include')
            and (self.site_cfg.python_include != 'auto')):
            return self.site_cfg.python_include

        else:
            return sysconfig.get_config_var('INCLUDEPY')


    def system(self):
        if (has_attr(self.site_cfg, 'system')
            and self.site_cfg.system is not None):
            return self.site_cfg.system

        else:
            if os.name in ['posix']:
                return 'posix'
            elif os.name in ['nt']:
                return 'windows'
            else:
                raise ValueError(msg_unknown_os)

    def compile_flags(self):
        if has_attr(self.site_cfg, 'compile_flags'):
            flags = self.site_cfg.compile_flags
            if isinstance(flags, str):
                warn('Compile flags should be given as a list of strings.'
                     ' Space-separated strings may be removed in the'
                     ' near future.',
                     DeprecationWarning, stacklevel=2)

                flags = flags.split()

        else:
            flags = ['-g', '-O2']

        return flags + compose_system_compile_flags(self.system() == 'posix')

    def debug_flags(self) -> list:
        if has_attr(self.site_cfg, 'debug_flags'):
            return self.site_cfg.debug_flags
        else:
            return []

    def numpydoc_path(self):
        if (has_attr(self.site_cfg, 'numpydoc_path') and
            (self.site_cfg.numpydoc_path is not None)):
            return self.site_cfg.numpydoc_path

        else:
            try:
                import numpydoc; numpydoc
            except ImportError:
                raise ValueError(msg_numpydoc)

    def is_release(self):
        if has_attr(self.site_cfg, 'is_release'):
            return self.site_cfg.is_release
        else:
            return ''

    def refmap_memory_factor(self):
        if has_attr(self.site_cfg, 'refmap_memory_factor'):
            return self.site_cfg.refmap_memory_factor
        else:
            return None

    def debug_warped_cells(self):
        if has_attr(self.site_cfg, 'debug_warped_cells'):
            return self.site_cfg.debug_warped_cells
        else:
            return False

site_config = Config()
