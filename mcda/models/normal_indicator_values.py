# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from ProMCDA.mcda.models.base_model_ import Model
from ProMCDA.mcda import util


class NormalIndicatorValues(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, avg: float=None, std: float=None):  # noqa: E501
        """NormalIndicatorValues - a model defined in Swagger

        :param avg: The avg of this NormalIndicatorValues.  # noqa: E501
        :type avg: float
        :param std: The std of this NormalIndicatorValues.  # noqa: E501
        :type std: float
        """
        self.swagger_types = {
            'avg': float,
            'std': float
        }

        self.attribute_map = {
            'avg': 'avg',
            'std': 'std'
        }
        self._avg = avg
        self._std = std

    @classmethod
    def from_dict(cls, dikt) -> 'NormalIndicatorValues':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The NormalIndicatorValues of this NormalIndicatorValues.  # noqa: E501
        :rtype: NormalIndicatorValues
        """
        return util.deserialize_model(dikt, cls)

    @property
    def avg(self) -> float:
        """Gets the avg of this NormalIndicatorValues.

        Average value of the indicator estimated for a specific alternative.  # noqa: E501

        :return: The avg of this NormalIndicatorValues.
        :rtype: float
        """
        return self._avg

    @avg.setter
    def avg(self, avg: float):
        """Sets the avg of this NormalIndicatorValues.

        Average value of the indicator estimated for a specific alternative.  # noqa: E501

        :param avg: The avg of this NormalIndicatorValues.
        :type avg: float
        """
        if avg is None:
            raise ValueError("Invalid value for `avg`, must not be `None`")  # noqa: E501

        self._avg = avg

    @property
    def std(self) -> float:
        """Gets the std of this NormalIndicatorValues.

        Standard deviation value of the indicator estimated for a specific alternative.  # noqa: E501

        :return: The std of this NormalIndicatorValues.
        :rtype: float
        """
        return self._std

    @std.setter
    def std(self, std: float):
        """Sets the std of this NormalIndicatorValues.

        Standard deviation value of the indicator estimated for a specific alternative.  # noqa: E501

        :param std: The std of this NormalIndicatorValues.
        :type std: float
        """
        if std is None:
            raise ValueError("Invalid value for `std`, must not be `None`")  # noqa: E501

        self._std = std