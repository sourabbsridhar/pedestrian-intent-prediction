# Implementation to parse configuration file

import os
import logging
from pathlib import Path
from operator import getitem
from datetime import datetime
from logger import setup_logging
from functools import reduce, partial
from utils import read_json, write_json

class ConfigParser:
    """
    Class implementation for all configuration parsers. 
    """
    def __init__(self, userConfiguration, resume=None, modification=None, runId=None):
        """
        Method to initialize an object of type ConfigParser.

        Parameters
        ----------
        self                : ConfigParser
                              Instance of the class
        userConfiguration   : dict
                              User configuration data
        resume              : bool
                              Set to True in order to resume an experiment (Default Value: None)
        modification        : dict
                              Modifications added to the user configuration data (Default Value: None)
        runId               : str
                              Unique identifer for the experiment (Default Value: None)

        Returns
        -------
        self    : BaseDataLoader
                  Initialized object of class ConfigParser
        """
        # Load Configuration File and Apply Modificaitons
        self.configuration = update_configuration(userConfiguration, modification)
        self.resume = resume

        # Set Save directory where trained model and the log would be saved
        saveDirectory = Path(self.configuration["trainer"]["save_dir"])

        experimentName = self.configuration["name"]
        if runId is None:
            runId = datetime.now().strftime(r"%m%d_%H%M%S")
        self.outputDirectory = saveDirectory / "models" / experimentName / runId
        self.logDirectory = saveDirectory / "log" / experimentName / runId

        # Make directory for saving checkpoints and log
        existOk = (runId == "")
        self.outputDirectory.mkdir(parents=True, exist_ok=existOk)
        self.logDirectory.mkdir(parents=True, exist_ok=existOk)

        # Save updated configuration file to the checkpoint directory
        write_json(self.configuration, self.outputDirectory / "config.json")

        # Configure Logging Module
        setup_logging(self.logDirectory)
        self.loggingLevels = {
                                0: logging.WARNING,
                                1: logging.INFO,
                                2: logging.DEBUG
                            }

    @classmethod
    def from_args(cls, args, options=""):
        """
        Method to create a class from the command line arguments.
        Since this method is defined as a @classmethod, the method is bound to the ConfigParser class and not to its instance.

        Parameters
        ----------
        cls         : ConfigParser
                      Class to which the method is bound
        args        : argparse.ArgumentParser
                      Command line arguments
        options     : str/namedTuple
                      Additional user defined command line arguments                    

        Returns
        -------
        cls     : ConfigParser
                  Uninitialized instance of the class ConfigParser
        """
        # Add additional user defined command line arguments
        for individualOption in options:
            args.add_argument(*individualOption.flags, default=None, type=individualOption.type)

        # Parse all command line arguments if not parsed
        if not isinstance(args, tuple):
            args = args.parse_args()

        # Set up PyTorch device for the experiement
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        # Set up user configuration path for the experiment
        if args.resume is not None:
            resumePath = Path(args.resume)
            userConfigurationPath = resumePath.parent / "config.json"
        else:
            assert args.config is not None, "Configuration path needs to be specified. For example: Add '-c config.json'"
            resumePath = None
            userConfigurationPath = Path(args.config)

        configuration = read_json(userConfigurationPath)

        if args.config and resumePath:
            configuration.update(read_json(args.config))

        # Configure modifications to the command line arguments
        modification = {individualOption.target: getattr(args, get_option_name(individualOption.flags)) for individualOption in options}

        return cls(configuration, resumePath, modification)


    def initialize_object(self, name, module, *args, **kwargs):
        """
        Method to find an object with the name given as 'type' in configuration, and returns the
        instance initialized with corresponding arguments given.

        For example:
        object = config.initialize_object('name', module, a, b=1)

        is equivalent to:
        object = module.name(a, b=1)

        Parameters
        ----------
        self        : ConfigParser
                      Instance of the class
        name        : str
                      Name of the key in the configuration
        module      : module
                      Module corresponding to the key to be initializes
        *args       : tuple
                      Non-keyword arguments
        **kwargs    : dict
                      Keyword arguments

        Returns
        -------
        object      : Multiple
                      Initialized object
        """
        # Get attributes of the instance using the __getitem__() method defined below
        moduleName = self[name]["type"]
        moduleArguments = dict(self[name]["args"])

        assert all([k not in moduleArguments for k in kwargs]), "Overwriting kwargs given in configuration file is not allowed"
        moduleArguments.update(kwargs)

        return getattr(module, moduleName)(*args, **moduleArguments)

    def initialize_function(self, name, module, *args, **kwargs):
        """
        Method to find a function handle with the name given as 'type' in configuration, and returns the
        function with given arguments fixed with functools.partial.

        For example:
        function = config.initialize_function('name', module, a, b=1)

        is equivalent to:
        function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)
        
        Parameters
        ----------
        self        : ConfigParser
                      Instance of the class
        name        : str
                      Name of the key in the configuration
        module      : module
                      Module corresponding to the key to be initializes
        *args       : tuple
                      Non-keyword arguments
        **kwargs    : dict
                      Keyword arguments

        Returns
        -------
        function    : callable
                      Function handle
        """
        # Get attributes of the instance using the __getitem__() method defined below
        moduleName = self[name]["type"]
        moduleArguments = dict(self[name]["args"])

        assert all([k not in moduleArguments for k in kwargs]), "Overwriting kwargs given in configuration file is not allowed"
        moduleArguments.update(kwargs)
        return partial(getattr(module, moduleName), *args, **moduleArguments)

    def __getitem__(self, name):
        """
        Method to access value of a specific key from the configuration property of the instance.
        
        Parameters
        ----------
        self        : ConfigParser
                      Instance of the class
        name        : str
                      Name of the key to be accessed from the configuration property

        Returns
        -------
        value       : Multiple
                      Value of a specific key of the configuration property
        """
        return self.configuration[name]

    def get_logger(self, name, verbosity=2):
        """
        Method to get logger to track events.

        Parameters
        ----------
        self        : ConfigParser
                      Instance of the class
        name        : str
                      Name of the logger
        verbosity   : int
                      Level of verbosity (Default value: 2)

        Returns
        -------
        logger      : logging.Logger
                      Logger with the user defined name and verbosity level
        """
        assert verbosity in self.loggingLevels, "Verbosity Option {} is invalid. Valid Options are {}".format(verbosity, self.loggingLevels.keys())
        logger = logging.getLogger(name)
        logger.setLevel(self.loggingLevels[verbosity])
        return logger

    @property
    def config(self):
        """
        Method to return the configuration property of the class ConfigParser.
        
        Parameters
        ----------
        self            : ConfigParser
                          Instance of the class

        Returns
        -------
        configuration   : dict
                          Configuration property of the class ConfigParser
        """
        return self.configuration

    @property
    def output_directory(self):
        """
        Method to return the outputDirectory property of the class ConfigParser.
        
        Parameters
        ----------
        self            : ConfigParser
                          Instance of the class

        Returns
        -------
        outputDirectory : pathlib.Path
                          outputDirectory property of the class ConfigParser
        """
        return self.outputDirectory

    @property
    def log_directory(self):
        """
        Method to return the logDirectory property of the class ConfigParser.
        
        Parameters
        ----------
        self            : ConfigParser
                          Instance of the class

        Returns
        -------
        logDirectory   : pathlib.Path
                         logDirectory property of the class ConfigParser
        """
        return self.logDirectory

def update_configuration(userConfiguration, modification):
    """
    Function to update user defined configuration.

    Parameters
    ----------
    userConfiguration   : dict
                          User defined configuration to be modified
    modification        : Multiple
                          Modifications to be made

    Returns
    -------
    configuration   : dict
                      Modified user defined configuration
    """
    if modification is None:
        return userConfiguration

    for key, value in modification.items():
        if value is not None:
            set_key_value_by_path(userConfiguration, key, value)
    return userConfiguration

def get_option_name(flags):
    """
    Function to get option names from the user defined arguments.

    Parameters
    ----------
    flags       : list
                  List of user defined arguments
    Returns
    -------
    flags       : list
                  List of option names
    """
    for individualFlag in flags:
        if individualFlag.startswith("--"):
            return individualFlag.replace("--", "")
    return flags[0].replace("--", "")

def set_key_value_by_path(tree, keys, value):
    """
    Function to set a key value pair in a tree by path.

    Parameters
    ----------
    tree        : dict
                  Input tree in which the key value pair must be set
    keys        : Multiple
                  Key for which the value must be set
    value       : Multiple
                  New value to be set for the key

    Returns
    -------
    None
    """
    keys = keys.split(";")
    get_key_by_path(tree, keys[:-1])[keys[-1]] = value

def get_key_by_path(tree, keys):
    """
    Function to get keys from a tree by path.

    Parameters
    ----------
    tree        : dict
                  Instance containing the keys
    keys        : Multiple
                  Key to be extracted

    Returns
    -------
    keys        : Multiple
                  Path extracted key
    """
    return reduce(getitem, keys, tree)
