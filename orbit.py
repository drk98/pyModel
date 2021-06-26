import numpy as np
import scipy.optimize as optimize
import astropy.units as u
import astropy.constants as const
from typing import Union, Tuple

INPUT_VALUE = Union[float, int, u.Quantity]


def setUnitIfNone(value: INPUT_VALUE, unit: u.UnitBase):
    if not issubclass(value, unit):
        return value * unit
    return value


class Orbit(object):

    @u.quantity_input
    def __init__(self, a: u.m, e: float, i: u.deg, omega: u.deg, rightAscNode: u.deg,
                 T: u.day, n: u.deg / u.day):
        """

        :param a: The semimajor axis.
        :param e: The eccentricity
        :param i: The inclination.
        :param omega: The XXXX.
        :param rightAscNode:
        :param T: The epoch time.
        :param n: The mean daily motion
        """

        assert a > 0, "\'a\', the semi-major axis, should be positive"
        assert e > 0, "\'e\', the eccentricity, should be positive"
        # assert 0 < i < 360, "\'i\', the inclination, should be between 0 deg and 360 deg"
        # assert -360 < omega < 360, "\'omega\', the argument of periapsis, should be between 0 deg and 360 deg"
        # assert -360 < rightAscNode < 360, "\'rightAscNode\', the Longitude of the ascending node , should be between 0 deg and 360 deg"

        self.a: float = a.to(u.meter)
        self.e: float = e
        self.i: float = i.to(u.deg)
        self.omega: float = omega.to(u.deg)
        self.rightAscNode: float = rightAscNode.to(u.deg)
        self.T: float = T.to(u.s)
        self.n = n.to(u.deg / u.day)

    @property
    def period(self) -> u.day:
        """
        Is the orbital period of the object. Assuming that m_sun >> m_object
        :return: The period
        """
        return np.sqrt((4 * np.pi ** 2) * (self.a.to(u.m)) ** 3 / (const.G * const.M_sun)).to(u.day)

    @u.quantity_input
    def _calcMeanAnomaly(self, t: u.s) -> u.deg:
        t = t.to(u.day)

        return self.n * ((t - self.T.to(u.day)) % self.period.to(u.day))

    @u.quantity_input
    def _calcEccentricAnomaly(self, t: u.s = None, meanAnomaly: u.deg = None) -> u.deg:

        if meanAnomaly is None:
            meanAnomaly = self._calcMeanAnomaly(t)
        conv = lambda ea: ea - (self.e * np.sin(ea)) * u.deg - meanAnomaly
        deriv = lambda ea: 1 + self.e * np.cos(ea).value
        return optimize.newton(conv, meanAnomaly, fprime=deriv)

    @u.quantity_input
    def _calcTrueAnomaly(self, t: u.s = None, eccenticAnomaly: u.deg = None, meanAnomaly: u.deg = None) -> u.deg:

        if eccenticAnomaly is None:
            eccenticAnomaly = self._calcEccentricAnomaly(t, meanAnomaly=meanAnomaly)
        return 2 * np.arctan(np.sqrt(((1 + self.e) / (1 - self.e))) * np.tan(eccenticAnomaly / 2)).to(u.deg)

    @u.quantity_input
    def _calcRadius(self, t: u.s = None, eccenticAnomaly: u.deg = None, meanAnomaly: u.deg = None) -> u.m:

        if eccenticAnomaly is None:
            eccenticAnomaly = self._calcEccentricAnomaly(t, meanAnomaly=meanAnomaly)

        return self.a * (1 - self.e * np.cos(eccenticAnomaly))

    @u.quantity_input
    def calcCartesianCoords(self, t: u.s) -> u.m:

        ma = self._calcMeanAnomaly(t)
        ea = self._calcEccentricAnomaly(meanAnomaly=ma)
        v = self._calcTrueAnomaly(eccenticAnomaly=ea)
        r = self._calcRadius(eccenticAnomaly=ea)

        x = r * (np.cos(self.rightAscNode) * np.cos(self.omega + v) - np.sin(
            self.rightAscNode) * np.sin(self.omega + v) * np.cos(self.i))
        y = r * (np.sin(self.rightAscNode) * np.cos(self.omega + v) + np.cos(
            self.rightAscNode) * np.sin(self.omega + v) * np.cos(self.i))
        z = r * (np.sin(self.i) * np.sin(self.omega + v))
        return [x, y, z] * u.m


earth = Orbit(1 * u.AU, 0.01671022, 0.00005 * u.deg, 102.94719 * u.deg, -11.26064 * u.deg, 0 * u.s,
              (360 / 365.2568983840419) * u.deg / u.day)
print(earth.period)
coords = earth.calcCartesianCoords((2459391.9905 - 2451545) * u.day)
print(coords.to(u.AU))
print(np.sqrt(sum([c ** 2 for c in coords])).to(u.AU))
