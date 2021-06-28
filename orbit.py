import numpy as np
import scipy.optimize as optimize
import astropy.units as u
import astropy.constants as const

mu = 1.32712410041e20 * u.m ** 3 / u.s ** 2


def rotation_matrix(theta, axis):
    c = np.cos(theta)
    s = np.sin(theta)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    return None


def rm(theta, axis):
    """
    Alias for rotation_matrix
    :param theta: The angle to rotate
    :param axis: The axis to rotate on
    :return: The 3x3 rotation matrix
    """
    return rotation_matrix(theta, axis)


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

        self.rotaion_factor = rm(-self.rightAscNode, 'z') @ rm(-self.i, 'x') @ rm(-self.omega, 'z')

    @property
    def period(self) -> u.day:
        """
        Is the orbital period of the object. Assuming that m_sun >> m_object
        :return: The period
        """
        return np.sqrt((4 * np.pi ** 2) * (self.a.to(u.m)) ** 3 / (const.G * const.M_sun)).to(u.day)

    @property
    def specificAngularMomentum(self) -> u.m ** 2 / u.s:
        """
        Calculates the specific angular momentum for the object
        :return: The specific angular momentum
        """
        mu = self.n.to(u.deg / u.s) ** 2 * self.a.to(u.m) ** 3
        mu = mu.value * u.m ** 3 / u.s ** 2
        return np.sqrt(mu * self.a.to(u.m) * (1 - self.e ** 2))

    @u.quantity_input
    def _calcMeanAnomaly(self, t: u.s) -> u.deg:
        """
        Calculates the mean anomaly at time t
        :param t: The time to calculate the mean anomaly
        :return: The mean anomaly
        """
        t = t.to(u.day)

        return self.n * ((t - self.T.to(u.day)) % self.period.to(u.day))

    @u.quantity_input
    def _calcEccentricAnomaly(self, t: u.s = None, meanAnomaly: u.deg = None) -> u.deg:
        """
        Calculates the eccentic anomaly at time t
        :param t: The time to calculate the eccentic anomaly
        :param meanAnomaly: The mean anomaly at time t
        :return: the eccentic anomaly at time t (not needed if mean anomaly is provided)
        """

        if meanAnomaly is None:
            meanAnomaly = self._calcMeanAnomaly(t)
        conv = lambda ea: ea - (self.e * np.sin(ea)) - meanAnomaly.value
        deriv = lambda ea: 1 + self.e * np.cos(ea)
        return optimize.newton(conv, meanAnomaly.value, fprime=deriv) * u.deg


    @u.quantity_input
    def _calcTrueAnomaly(self, t: u.s = None, eccenticAnomaly: u.deg = None, meanAnomaly: u.deg = None) -> u.deg:

        if eccenticAnomaly is None:
            eccenticAnomaly = self._calcEccentricAnomaly(t, meanAnomaly=meanAnomaly)
        return 2 * np.arctan2(np.sqrt(1 + self.e) * np.sin(eccenticAnomaly / 2),
                              np.sqrt(1 - self.e) * np.cos(eccenticAnomaly / 2))

    @u.quantity_input
    def _calcRadius(self, t: u.s = None, eccenticAnomaly: u.deg = None, meanAnomaly: u.deg = None) -> u.m:
        """
        Calculate the distance from the central body to the object
        :param t: The time to calculate the radius (not needed if eccenticAnomaly is provided)
        :param eccenticAnomaly: The eccentric anomaly at time t
        :param meanAnomaly: The mean anomaly at time t (not needed if eccenticAnomaly is provided)
        :return: The radius at time t
        """

        if eccenticAnomaly is None:
            eccenticAnomaly = self._calcEccentricAnomaly(t, meanAnomaly=meanAnomaly)

        return self.a * (1 - self.e * np.cos(eccenticAnomaly))

    @u.quantity_input
    def calcCartesianCoords(self, t: u.s) -> u.m:
        """
        Using equation 9 from https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
        :param t: The time when the position is wanted
        :return: The xyz position vector
        """

        ma = self._calcMeanAnomaly(t)
        ea = self._calcEccentricAnomaly(meanAnomaly=ma)
        v = self._calcTrueAnomaly(eccenticAnomaly=ea)
        r = self._calcRadius(eccenticAnomaly=ea)

        o_t = r * np.array([[np.cos(v)],
                            [np.sin(v)],
                            [0]])

        dist = self.rotaion_factor @ o_t

        return dist.T[0]

    @u.quantity_input
    def calcCartesianVelocities(self, t: u.s) -> u.m / u.s:
        """
        Using equation 10 from https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
        :param t: The time at the velocity of the body is wanted
        :return: the xyz velocity vector
        """

        ma = self._calcMeanAnomaly(t)
        ea = self._calcEccentricAnomaly(meanAnomaly=ma)
        v = self._calcTrueAnomaly(eccenticAnomaly=ea)
        r = self._calcRadius(eccenticAnomaly=ea).to(u.m)

        odot_t = np.sqrt(mu * self.a) / r * np.array([[-np.sin(ea)],
                                                      [np.sqrt(1 - self.e ** 2) * np.cos(ea)],
                                                      [0]])
        r_dot = self.rotaion_factor @ odot_t

        return r_dot.T[0]

    @u.quantity_input
    def calcECFCoords(self, t: u.s) -> u.m:
        """
        Calculate the cartesian coordinates of the object in the Earth Centered and Fixed (ECF) system
        :param t: The time the coordinates are wanted
        :return: The xyz vector in ECF
        """

        earth = Orbit(1.00000011 * u.AU, 0.01671022, 0.00005 * u.deg, 102.94719 * u.deg, -11.26064 * u.deg, 0 * u.s,
                      (360 / 365.2568983840419) * u.deg / u.day)

        earth_coords = earth.calcCartesianCoords(t)
        this_coords = self.calcCartesianCoords(t)

        return this_coords - earth_coords

