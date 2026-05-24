"""
Unified output and temporary file management for SfePy.

Provides OutputManager for standardized output directory layout and
TempManager for controlled temporary file creation with automatic cleanup.
"""
import os
import os.path as op
import atexit
import shutil
import tempfile
import datetime
import threading

from sfepy.base.base import output, get_default, Struct


class OutputManager:
    """
    Unified output directory manager.

    Creates and manages the output directory structure with optional
    timestamp-based subdirectories to prevent overwriting results
    between runs.

    Parameters
    ----------
    problem_name : str
        Base name for the output, typically derived from the input file.
    output_root : str, optional
        Root directory for all output. Defaults to ``'./output'``.
    create_run_dir : bool, optional
        If True (default), create a timestamped subdirectory for this run.
    timestamp : str, optional
        Custom timestamp for the run directory. If None, auto-generated.
    extra_tags : list of str, optional
        Additional tags appended to the run directory name.
    conflict_strategy : str, optional
        How to handle existing output files/directories. One of:
        - ``'increment'`` (default): New files get an incrementing suffix
        - ``'overwrite'``: Overwrite existing files in the same location
        - ``'error'``: Raise an error if output location exists

    Examples
    --------
    >>> om = OutputManager('poisson', output_root='./results')
    >>> om.get_path('solution', 'vtk')
    './results/20260524_153000_poisson/poisson_solution.vtk'
    """

    CONFLICT_INCREMENT = 'increment'
    CONFLICT_OVERWRITE = 'overwrite'
    CONFLICT_ERROR = 'error'

    VALID_CONFLICT_STRATEGIES = (
        CONFLICT_INCREMENT,
        CONFLICT_OVERWRITE,
        CONFLICT_ERROR,
    )

    def __init__(self, problem_name, output_root=None, create_run_dir=True,
                 timestamp=None, extra_tags=None, conflict_strategy=None):
        self.problem_name = problem_name
        self.output_root = get_default(output_root, './output')
        self.create_run_dir = create_run_dir
        self._extra_tags = extra_tags or []
        self.conflict_strategy = get_default(
            conflict_strategy, self.CONFLICT_INCREMENT)

        if self.conflict_strategy not in self.VALID_CONFLICT_STRATEGIES:
            raise ValueError(
                'invalid conflict_strategy: %s (valid: %s)'
                % (conflict_strategy, self.VALID_CONFLICT_STRATEGIES))

        if timestamp is None:
            self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = timestamp

        if create_run_dir:
            tags = '_'.join(self._extra_tags) if self._extra_tags else ''
            dir_name = f'{self.timestamp}_{problem_name}'
            if tags:
                dir_name += f'_{tags}'
            self.run_dir = op.join(self.output_root, dir_name)
        else:
            self.run_dir = self.output_root

        self.temp_dir = op.join(self.run_dir, 'temp')
        self._dirs_created = False
        self._increment_counter = {}

    def ensure_dirs(self):
        """
        Create the run and temp directories, handling existing
        directories according to the conflict strategy.
        """
        if not self._dirs_created:
            if op.exists(self.run_dir):
                if self.conflict_strategy == self.CONFLICT_ERROR:
                    raise OSError(
                        'output directory already exists: %s'
                        ' (use conflict_strategy="overwrite" or'
                        ' "increment" to proceed)' % self.run_dir)

                elif self.conflict_strategy == self.CONFLICT_OVERWRITE:
                    if op.isdir(self.run_dir):
                        shutil.rmtree(self.run_dir)

            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.temp_dir, exist_ok=True)
            self._dirs_created = True

    def get_path(self, name, ext=None, subdir=None, handle_conflict=None):
        """
        Generate a standard output file path.

        Parameters
        ----------
        name : str
            File name component (e.g., ``'solution'``, ``'ebc'``).
        ext : str, optional
            File extension without the dot (e.g., ``'vtk'``, ``'h5'``).
        subdir : str, optional
            Optional subdirectory within the run directory.
        handle_conflict : bool, optional
            If True (default), apply the conflict strategy when the
            target file already exists.

        Returns
        -------
        path : str
            Absolute path to the output file.
        """
        self.ensure_dirs()
        filename = f'{self.problem_name}_{name}'
        if ext:
            filename += f'.{ext}'

        if subdir:
            full_dir = op.join(self.run_dir, subdir)
            os.makedirs(full_dir, exist_ok=True)
            full_path = op.join(full_dir, filename)
        else:
            full_path = op.join(self.run_dir, filename)

        if handle_conflict is not False and op.exists(full_path):
            if self.conflict_strategy == self.CONFLICT_ERROR:
                raise OSError(
                    'output file already exists: %s'
                    ' (use conflict_strategy="overwrite" or'
                    ' "increment" to proceed)' % full_path)

            elif self.conflict_strategy == self.CONFLICT_INCREMENT:
                key = full_path
                if key not in self._increment_counter:
                    self._increment_counter[key] = 1
                else:
                    self._increment_counter[key] += 1
                base, ext_part = op.splitext(full_path)
                full_path = (f'{base}_{self._increment_counter[key]:03d}'
                             f'{ext_part}')

        return full_path

    def get_temp_path(self, name, ext=None):
        """
        Generate a temporary file path inside the run's temp directory.

        Parameters
        ----------
        name : str
            Temporary file name component.
        ext : str, optional
            File extension without the dot.

        Returns
        -------
        path : str
            Absolute path to the temporary file.
        """
        self.ensure_dirs()
        filename = f'temp_{name}'
        if ext:
            filename += f'.{ext}'
        return op.join(self.temp_dir, filename)

    def get_run_dir(self):
        """Return the run directory path."""
        self.ensure_dirs()
        return self.run_dir

    def get_temp_dir(self):
        """Return the temp directory path."""
        self.ensure_dirs()
        return self.temp_dir

    def save_config(self, conf_filename):
        """
        Copy the problem configuration file into the run directory
        for reproducibility.

        Parameters
        ----------
        conf_filename : str
            Path to the original problem definition file.
        """
        self.ensure_dirs()
        if conf_filename and op.isfile(conf_filename):
            dest = op.join(self.run_dir, op.basename(conf_filename))
            if not op.exists(dest):
                shutil.copy2(conf_filename, dest)

    def make_latest_link(self):
        """Create a 'latest' symlink pointing to the most recent run."""
        self.ensure_dirs()
        link_path = op.join(self.output_root, 'latest')
        try:
            if op.islink(link_path):
                os.unlink(link_path)
            elif op.exists(link_path):
                if op.isdir(link_path):
                    shutil.rmtree(link_path)
                else:
                    os.remove(link_path)
            os.symlink(self.run_dir, link_path)
        except OSError:
            pass

    def get_output_trunk(self):
        """
        Return the output filename trunk (without extension) for use
        with Problem.setup_output().

        Returns
        -------
        trunk : str
            The full path to the output file without extension,
            i.e., ``<run_dir>/<problem_name>``.
        """
        self.ensure_dirs()
        return op.join(self.run_dir, self.problem_name)

    def get_output_dir(self):
        """
        Return the output directory for use with Problem.setup_output().
        Alias for get_run_dir().
        """
        return self.get_run_dir()


class TempManager:
    """
    Centralized temporary file manager with automatic cleanup.

    All temporary files and directories created through this manager
    are tracked and automatically cleaned up on program exit.

    Class-level state is used so any code path can register temp
    resources without passing an instance around.

    Examples
    --------
    >>> d = TempManager.mkdtemp()
    >>> f = TempManager.mkstemp(suffix='.vtk')
    >>> # Both are automatically removed when the process exits.
    """

    _lock = threading.Lock()
    _temp_dirs = []
    _temp_files = []
    _registered = False

    @classmethod
    def _register_atexit(cls):
        if not cls._registered:
            atexit.register(cls.cleanup)
            cls._registered = True

    @classmethod
    def mkdtemp(cls, prefix='sfepy_', suffix='', dir=None):
        """
        Create a temporary directory that will be cleaned up on exit.

        Parameters
        ----------
        prefix : str, optional
            Directory name prefix. Defaults to ``'sfepy_'``.
        suffix : str, optional
            Directory name suffix.
        dir : str, optional
            Parent directory. If None, uses the system temp directory.

        Returns
        -------
        path : str
            Absolute path to the created directory.
        """
        cls._register_atexit()
        d = tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=dir)
        with cls._lock:
            cls._temp_dirs.append(d)
        return d

    @classmethod
    def mkstemp(cls, prefix='sfepy_', suffix='', dir=None):
        """
        Create a temporary file that will be cleaned up on exit.

        Parameters
        ----------
        prefix : str, optional
            File name prefix. Defaults to ``'sfepy_'``.
        suffix : str, optional
            File name suffix (e.g., ``'.vtk'``).
        dir : str, optional
            Parent directory. If None, uses the system temp directory.

        Returns
        -------
        fd : int
            File descriptor of the temporary file.
        path : str
            Absolute path to the created file.
        """
        cls._register_atexit()
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
        with cls._lock:
            cls._temp_files.append(path)
        return fd, path

    @classmethod
    def gettempdir(cls):
        """
        Return the system temp directory, wrapped for consistency.

        Returns
        -------
        path : str
            System temporary directory path.
        """
        return tempfile.gettempdir()

    @classmethod
    def cleanup(cls):
        """
        Remove all registered temporary directories and files.

        Called automatically on program exit via atexit.
        """
        with cls._lock:
            for path in cls._temp_files:
                try:
                    if op.isfile(path) or op.islink(path):
                        os.unlink(path)
                except OSError:
                    pass
            cls._temp_files.clear()

            for d in cls._temp_dirs:
                try:
                    if op.isdir(d):
                        shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass
            cls._temp_dirs.clear()

    @classmethod
    def register_dir(cls, path):
        """
        Register an existing directory for cleanup.

        Parameters
        ----------
        path : str
            Directory path to remove on exit.
        """
        cls._register_atexit()
        with cls._lock:
            cls._temp_dirs.append(path)

    @classmethod
    def register_file(cls, path):
        """
        Register an existing file for cleanup.

        Parameters
        ----------
        path : str
            File path to remove on exit.
        """
        cls._register_atexit()
        with cls._lock:
            cls._temp_files.append(path)

    @classmethod
    def unregister_dir(cls, path):
        """
        Remove a directory from the cleanup list (e.g., when the caller
        takes ownership of it).

        Parameters
        ----------
        path : str
            Directory path to unregister.
        """
        with cls._lock:
            if path in cls._temp_dirs:
                cls._temp_dirs.remove(path)

    @classmethod
    def unregister_file(cls, path):
        """
        Remove a file from the cleanup list.

        Parameters
        ----------
        path : str
            File path to unregister.
        """
        with cls._lock:
            if path in cls._temp_files:
                cls._temp_files.remove(path)


def create_output_manager_from_conf(conf, options):
    """
    Factory function to create an OutputManager from a ProblemConf
    and command-line options.

    All SfePy entry points **must** call this function to ensure
    consistent output directory behavior across all application kinds.

    Parameters
    ----------
    conf : ProblemConf
        The problem configuration.
    options : argparse.Namespace or Struct
        Command-line / runtime options.

    Returns
    -------
    om : OutputManager or None
        The output manager, or None if output management is disabled.
    """
    opts = conf.options

    problem_name = None
    if hasattr(options, 'output_filename_trunk') and \
       options.output_filename_trunk is not None:
        problem_name = options.output_filename_trunk

    if problem_name is None:
        if conf.get('filename_mesh') is not None:
            from sfepy.base.ioutils import get_trunk
            problem_name = get_trunk(conf.filename_mesh)
        elif conf.get('filename_domain') is not None:
            from sfepy.base.ioutils import get_trunk
            problem_name = get_trunk(conf.filename_domain)

    if problem_name is None:
        return None

    output_root = opts.get('output_dir', None)
    if output_root is None and hasattr(options, 'output_dir') and \
       options.output_dir is not None:
        output_root = options.output_dir

    if output_root is None:
        output_root = './output'

    create_run_dir = opts.get('create_run_subdir', True)

    conflict_strategy = opts.get(
        'output_conflict_strategy',
        OutputManager.CONFLICT_INCREMENT)

    om = OutputManager(
        problem_name=problem_name,
        output_root=output_root,
        create_run_dir=create_run_dir,
        conflict_strategy=conflict_strategy,
    )

    return om


def create_output_manager(problem_name, output_root='./output',
                          create_run_dir=True, conflict_strategy=None):
    """
    Create an OutputManager directly, for entry points that do not
    have a ProblemConf (e.g., convert_mesh, probe).

    Parameters
    ----------
    problem_name : str
        Base name for the output.
    output_root : str, optional
        Root directory for all output. Defaults to ``'./output'``.
    create_run_dir : bool, optional
        If True (default), create a timestamped subdirectory.
    conflict_strategy : str, optional
        One of ``'increment'``, ``'overwrite'``, ``'error'``.

    Returns
    -------
    om : OutputManager
    """
    return OutputManager(
        problem_name=problem_name,
        output_root=output_root,
        create_run_dir=create_run_dir,
        conflict_strategy=conflict_strategy,
    )
