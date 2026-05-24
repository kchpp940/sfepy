from sfepy.base.base import Struct, output, insert_as_static_method, get_default


class Application(Struct):
    """
    Base class for applications.

    Subclasses should implement: __init__(), call().

    Automates parametric studies, see parametrize().

    Parameters
    ----------
    conf : ProblemConf
        The problem configuration.
    options : argparse.Namespace or Struct
        Runtime options.
    output_prefix : str
        Prefix for log output.
    output_manager : OutputManager, optional
        Unified output directory manager. If None means legacy behavior.
    """
    def __init__(self, conf, options, output_prefix, **kwargs):
        output_manager = kwargs.pop('output_manager', None)
        Struct.__init__(self,
                        conf=conf,
                        options=options,
                        output_prefix=output_prefix,
                        output_manager=output_manager)
        output.prefix = self.output_prefix
        self.restore()

    def get_output_dir(self):
        """
        Return the output directory, preferring OutputManager when
        available, falling back to the problem's output_dir.

        Returns
        -------
        output_dir : str
        """
        if self.output_manager is not None:
            return self.output_manager.get_output_dir()
        return get_default(
            getattr(self, 'output_dir', '.'),
            '.',
        )

    def get_output_trunk(self):
        """
        Return the output filename trunk (path).

        Returns
        -------
        trunk : str or None
        """
        if self.output_manager is not None:
            return self.output_manager.get_output_trunk()
        return None

    def get_output_path(self, name, ext=None, subdir=None):
        """
        Generate a standard output file path.

        Parameters
        ----------
        name : str
            File name component.
        ext : str, optional
            File extension without dot.
        subdir : str, optional
            Subdirectory within run directory.

        Returns
        -------
        path : str
        """
        if self.output_manager is not None:
            return self.output_manager.get_path(name, ext=ext, subdir=subdir)
        return None

    def setup_options(self):
        pass

    def __call__(self, **kwargs):
        """
        This is either call_basic() or call_parametrized().
        """
        pass

    def call_basic(self, **kwargs):
        return self.call(**kwargs)

    def call_parametrized(self, **kwargs):
        generator = self.parametric_hook(self.problem)
        for aux in generator:
            if isinstance(aux, tuple) and (len(aux) == 2):
                problem, container = aux
                mode = 'coroutine'
            else:
                problem = aux
                mode = 'simple'
            self.problem = problem

            generator_prefix = output.prefix
            output.prefix = self.output_prefix # Restore default.

            """Application options have to be re-processed here as they can
            change in the parametric hook."""
            self.setup_options()
            out = self.call(**kwargs)

            output.prefix = generator_prefix

            if mode == 'coroutine':
                # Pass application output to the generator.
                container.append(out)
                next(generator)

    def restore(self):
        """
        Remove parametric_hook, restore __call__() to call_basic().
        """
        self.parametric_hook = None
        insert_as_static_method(self.__class__, '__call__',
                                self.call_basic)

    def parametrize(self, parametric_hook):
        """
        Add parametric_hook, set __call__() to call_parametrized().
        """
        if parametric_hook is None: return

        self.parametric_hook = parametric_hook
        insert_as_static_method(self.__class__, '__call__',
                                self.call_parametrized)
