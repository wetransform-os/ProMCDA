# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from ProMCDA.mcda.models.base_model_ import Model
from ProMCDA.mcda import util


class ExactIndicatorValues(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, val: float=None):  # noqa: E501
        """ExactIndicatorValues - a model defined in Swagger

        :param val: The val of this ExactIndicatorValues.  # noqa: E501
        :type val: float
        """
        self.swagger_types = {
            'val': float
        }

        self.attribute_map = {
            'val': 'val'
        }
        self._val = val

    @classmethod
    def from_dict(cls, dikt) -> 'ExactIndicatorValues':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The ExactIndicatorValues of this ExactIndicatorValues.  # noqa: E501
        :rtype: ExactIndicatorValues
        """
        return util.deserialize_model(dikt, cls)

    @property
    def val(self) -> float:
        """Gets the val of this ExactIndicatorValues.

        The exact value of the indicator for a specific alternative.  # noqa: E501

        :return: The val of this ExactIndicatorValues.
        :rtype: float
        """
        return self._val

    @val.setter
    def val(self, val: float):
        """Sets the val of this ExactIndicatorValues.

        The exact value of the indicator for a specific alternative.  # noqa: E501

        :param val: The val of this ExactIndicatorValues.
        :type val: float
        """
        if val is None:
            raise ValueError("Invalid value for `val`, must not be `None`")  # noqa: E501

        self._val = val