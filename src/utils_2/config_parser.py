import configparser as cp
import pathlib
import typing


def get_configuration_path() -> str:
    """Returns project root folder."""
    return str(pathlib.Path(__file__).parent.parent.parent) + '/configuration/conf.ini'


class ConfigReader:
    _path = None
    _config = None
    _config_dic = {}
    _sections = []

    def __init__(self):
        """
        Instance initiation, recieve a config file location and reads it.
        """
        self._path = get_configuration_path()
        self._config = cp.ConfigParser()

        self.parse()

    def parse(self):
        """
        Parse and load all configuration, all sections and keys.
        """
        try:
            self._config.read(self._path)
            self._sections = self._config.sections()
            for section in self._sections:
                self._config_dic[section] = dict(self._config.items(section))

        except Exception as ex:
            print(ex)
        finally:
            self._config = None

    def get_section(self, section) -> typing.Optional[dict]:
        """
        Method that provide a configuration params according to given section.
        :param section: section of the configuration.
        :return: all params for the given section, or None in case section doesn't exist.
        """
        if section in self._config_dic:
            return self._config_dic[section]

        return None

    def get_parameter(self, section, key):
        """
        Method to get specific parameter.
        :param section: to read from.
        :param key: the key to the parameter.
        :return: parameter if exist, otherwise None.
        """
        if section in self._config_dic and key in self._config_dic[section]:
            return self._config_dic[section][key]

        return None

    def get_params(self):
        """
        Getter meyhod for entire configurations.
        :return: all configuration stored in the file.
        """
        return self._config_dic