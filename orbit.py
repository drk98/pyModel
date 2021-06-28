import numpy as np
import scipy.optimize as optimize
import astropy.units as u
import astropy.constants as const


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

    @property
    def specificAngularMomentum(self) -> u.m ** 2 / u.s:
        mu = self.n.to(u.deg / u.s) ** 2 * self.a.to(u.m) ** 3
        mu = mu.value * u.m ** 3 / u.s ** 2
        return np.sqrt(mu * self.a.to(u.m) * (1 - self.e ** 2))

    @u.quantity_input
    def _calcMeanAnomaly(self, t: u.s) -> u.deg:
        t = t.to(u.day)

        return self.n * ((t - self.T.to(u.day)) % self.period.to(u.day))

    @u.quantity_input
    def _calcEccentricAnomaly(self, t: u.s = None, meanAnomaly: u.deg = None) -> u.deg:

        if meanAnomaly is None:
            meanAnomaly = self._calcMeanAnomaly(t)
        conv = lambda ea: ea - (self.e * np.sin(ea)) - meanAnomaly.value
        deriv = lambda ea: 1 + self.e * np.cos(ea)
        return optimize.newton(conv, meanAnomaly.value, fprime=deriv) * u.deg

        #return optimize.newton(conv, meanAnomaly.value) * u.deg

    @u.quantity_input
    def _calcTrueAnomaly(self, t: u.s = None, eccenticAnomaly: u.deg = None, meanAnomaly: u.deg = None) -> u.deg:

        if eccenticAnomaly is None:
            eccenticAnomaly = self._calcEccentricAnomaly(t, meanAnomaly=meanAnomaly)
        return 2 * np.arctan2(np.sqrt(1+self.e)*np.sin(eccenticAnomaly/2), np.sqrt(1-self.e)*np.cos(eccenticAnomaly/2))

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

    @u.quantity_input
    def calcCartesianVelocities(self, t: u.s) -> u.m / u.s:
        ma = self._calcMeanAnomaly(t)
        ea = self._calcEccentricAnomaly(meanAnomaly=ma)
        v = self._calcTrueAnomaly(eccenticAnomaly=ea)
        r = self._calcRadius(eccenticAnomaly=ea).to(u.m)

        print(f"{ma=}")
        print(f"{ea=}")
        print(f"{v=}")
        print(f"{r=}")
        h = self.specificAngularMomentum
        p = self.a * (1 - self.e ** 2)
        # p = self.a*2*np.pi
        x, y, z = self.calcCartesianCoords(t).to(u.m)
        print(f"{p=}")
        print(f"{h=}")
        v_X = x * h * self.e / (r * p) * np.sin(v)
        v_X = v_X - (h / r) * (np.cos(self.rightAscNode) * np.sin(self.omega + v) + np.sin(self.rightAscNode) * np.cos(
            self.omega + v) * np.cos(self.i))

        v_Y = y * h * self.e / (r * p) * np.sin(v)
        v_Y = v_Y - (h / r) * (np.sin(self.rightAscNode) * np.sin(self.omega + v) + np.cos(self.rightAscNode) * np.cos(
            self.omega + v) * np.cos(self.i))

        v_Z = z * h * self.e / (r * p) * np.sin(v)
        v_Z = v_Z + (h / r) * np.sin(self.i) * np.cos(self.omega + v)

        return [v_X, v_Y, v_Z] * (u.m / u.s)

    @u.quantity_input
    def calcECFCoords(self, t: u.s) -> u.m:

        vx, vy, vz = self.calcCartesianVelocities(t)
        x, y, z = self.calcCartesianCoords(t)

        greenwichHA = 0
        T = np.matrix([[np.cos(greenwichHA), np.sin(greenwichHA), 0],
                       [-np.sin(greenwichHA), np.cos(greenwichHA), 0],
                       [0, 0, 1]])
        earthRotRate = 7.2921158553e-5 / u.s
        tmp = np.matrix([[(vx + earthRotRate * y).value],
                         [(vy - earthRotRate * x).value],  # Check the equation of row
                         [vz.value]])

        return T * tmp * u.m


earth = Orbit(1 * u.AU, 0.01671022, 0.00005 * u.deg, 102.94719 * u.deg, -11.26064 * u.deg, 0 * u.s,
              (360 / 365.2568983840419) * u.deg / u.day)
print(earth.period)
print(earth.specificAngularMomentum)
coords = earth.calcCartesianCoords((2459391.9905 - 2451545) * u.day)
velos = earth.calcCartesianVelocities((2459391.9905 - 2451545 ) * u.day)
# velos = earth.calcCartesianVelocities(3 * u.day)
# coords = earth.calcCartesianCoords(3* u.day)
print(velos.to(u.km / u.s))
print(coords.to(u.AU))
print(np.linalg.norm(coords).to(u.AU))
print(np.linalg.norm(velos).to(u.km / u.s))
# print(earth.calcECFCoords(0*u.day))
exit()
days = np.linspace(0, 365)
vx = []
vy = []
vz = []
v = []
for d in np.linspace(0, 365):
    velos = earth.calcCartesianVelocities(d * u.day)
    vx.append(velos[0].to(u.km / u.s).value)
    vy.append(velos[1].to(u.km / u.s).value)
    vz.append(velos[2].to(u.km / u.s).value)
    # print(np.sqrt(sum([v ** 2 for v in velos])).to(u.km/u.s))
    v.append(np.sqrt(sum([vs ** 2 for vs in velos])).to(u.km / u.s).value)
# velos = np.array(velos)
# print(velos)
import matplotlib.pyplot as plt

# plt.plot(days, vx)
# plt.plot(days, vy)
# plt.plot(days, vz)
plt.plot(days, v)
plt.hlines(29.78, xmin=0, xmax=365)
# plt.show()
