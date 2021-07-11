import sys

import orbit
import numpy as np
from astropy.table import Table, vstack
from astropy import io
import astropy.units as u
import os
from scipy.signal import lombscargle
from subprocess import Popen, PIPE
from astroquery.jplhorizons import Horizons

cuSignalAvail = True
try:
    import cusignal
    import cupy as cp
except ModuleNotFoundError:
    cuSignalAvail = False
    pass

NROWS = 8
MAX_LC = 100  # The max number of light curves as set in minkowski
verbosePrint = lambda *args, **kwargs: None


def periodFinder(data, config, timeColumn: str = "jd", magColumn: str = "H") -> float:
    if config["CONVEXINVPARAMS"].getfloat("initial_period", fallback=-1) > 0:
        return config["CONVEXINVPARAMS"].getfloat("initial_period")

    minPeriod = config["PERIODOPTIONS"].getfloat("min_period") / 2
    maxPeriod = config["PERIODOPTIONS"].getfloat("max_period") / 2

    nFreqs = config["PERIODOPTIONS"].getint("number_frequencies")

    if config["PERIODOPTIONS"].getboolean("even_frequencies", fallback=True):
        freqs = 2 * np.pi * np.linspace(1 / maxPeriod, 1 / minPeriod, nFreqs)

    else:
        freqs = 2 * np.pi / np.linspace(minPeriod, maxPeriod, nFreqs)

    normFactor = 2 / (len(data) * np.std(data[magColumn]) ** 2)

    if config["PERIODOPTIONS"].getboolean("use_cpu", fallback=True) or not cuSignalAvail:
        power = lombscargle(data[timeColumn].to(u.h), data[magColumn] - np.mean(data[magColumn]), freqs) * normFactor
        maxPowerIdx = np.argmax(power)

    else:
        freqs_c = cp.asarray(freqs)
        t = cp.asarray(data[timeColumn].to(u.h))
        mag = cp.asarray(data[magColumn])
        power = cusignal.lombscargle(t, mag - cp.mean(mag), freqs_c) * normFactor
        maxPowerIdx = cp.argmax(power).get()

        power = cp.asnumpy(power)

    maxPower = power[maxPowerIdx]

    assert maxPower > config["PERIODOPTIONS"].getfloat("min_periodigram_power", fallback=0), "Max power too low"

    return 4 * np.pi / freqs[maxPowerIdx]


def getData(file, config):
    data = io.ascii.read(file)
    if data["jd"].unit is None:
        data["jd"] *= u.d
    return data


def getCartesianPositions(times, objectID):
    earth = []
    obj = []
    N = 100
    for i in range(0, len(times), N):
        t = times[i:i + N]
        e = Horizons(id="399", id_type="majorbody", epochs=t)
        o = Horizons(id=objectID, epochs=t)
        earth.append(e.vectors()[["x", "y", "z"]])
        obj.append(o.vectors()[["x", "y", "z"]])
    earth = vstack(earth)
    obj = vstack(obj)

    earth["x"] = (-obj["x"]) - (-earth["x"])
    earth["y"] = (-obj["y"]) - (-earth["y"])
    earth["z"] = (-obj["z"]) - (-earth["z"])

    return obj, earth

    """ephs = getEphemerides(objectID)
    obj = orbit.Orbit(*ephs)
    heliCoords = []
    earthCoords = []
    for t in times:
        t *= u.d
        heliCoords.append(obj.calcCartesianCoords(t).to(u.AU))
        earthCoords.append(obj.calcECFCoords(t).to(u.AU))

    return np.array(heliCoords), np.array(earthCoords)"""


def getEphemerides(objectID) -> tuple:
    return (2.202852949833554 * u.AU, .1688092443224495, 3.470615713183181 * u.deg, 16.39901715768626 * u.deg,
            264.8061873191282 * u.deg, .3014505820392746 * u.deg / u.d, 2459337.728795320658 * u.d,
            318.6322997780391 * u.deg)
    pass


def writeInputFiles(data, config, heliCoords, earthCoords, period):
    writeConvexinvParams(config, period)
    writeLightCurveFile(data, heliCoords, earthCoords, period, config)


def writeLightCurveFile(data, heliCoors, earthCoords, period, config):
    data["heliox"] = -heliCoors["x"]
    data["helioy"] = -heliCoors["y"]
    data["helioz"] = -heliCoors["z"]

    data["earthx"] = earthCoords["x"]
    data["earthy"] = earthCoords["y"]
    data["earthz"] = earthCoords["z"]

    with open(config["FILES"].get("lightcurvedata"), "w+") as file:
        blocks = []
        b = np.floor(data["jd"] * 24 / period)
        u = np.unique(b)
        blocks = [data[b == val] for val in u]

        b = np.floor((data["jd"] - .5) / 1000)
        u = np.unique(b)
        blocks = [data[b == val] for val in u]
        # blocks = [data[0:174], data[174:174 + 401]]

        i = 0
        while len(blocks) > MAX_LC:
            blocks[i] = vstack([blocks[i], blocks[i + 1]])
            del blocks[i + 1]

            i = (i + 1) % len(blocks)

        file.write(f"{len(blocks)}\n")
        for b in blocks:
            file.write(f"{len(b)} 0\n")
            for obs in b:
                file.write(
                    f"{obs['jd']:014f}  {obs['H']:e}   {obs['heliox']:e} {obs['helioy']:e} {obs['helioz']:e}   {obs['earthx']:e} {obs['earthy']:e} {obs['earthz']:e}\n")


def magToFlux(data, heliCoors, earthCoords, mz=0):
    h = np.array([heliCoors["x"], heliCoors["y"], heliCoors["z"]])
    e = np.array([earthCoords["x"], earthCoords["y"], earthCoords["z"]])

    heliDist = np.sqrt((h * h).sum(axis=0))
    earthDist = np.sqrt((e * e).sum(axis=0))


    data["H"] = data["H"] - 5 * np.log10(heliDist * earthDist)
    data["H"] = np.power(10, mz - data["H"])


def writeConvexinvParams(config, period) -> None:
    """
    Write the input parameters file for convexinv
    :param config: The configparser of the config file
    :param period: The rotation period of the object
    :return: None
    """

    ip = config["CONVEXINVPARAMS"]

    # There HAS to be a better way to do this
    file_data = f"""{ip.get('initial_lambda')}		{ip.get('initial_lambda_free')}	inital lambda [deg] (0/1 - fixed/free) 
{ip.get('initial_beta')}		{ip.get('initial_beta_free')}	initial beta [deg] (0/1 - fixed/free)
{period:.6f}		{ip.get('initial_period_free')}	inital period [hours] (0/1 - fixed/free)
{ip.get('zero_time')}			zero time [JD]
{ip.get('initial_rotation_angle')}			initial rotation angle [deg]     
{ip.get('convexity_regularization')}			convexity regularization      
{ip.get('degree_spherical_harmonics_expansion')} {ip.get('order_spherical_harmonics_expansion')}			degree and order of spherical harmonics expansion
{NROWS}			number of rows        
{ip.get('phase_function_a')}		{ip.get('phase_funtion_a_free')}	phase funct. param. 'a' (0/1 - fixed/free)  
{ip.get('phase_function_d')}		{ip.get('phase_funtion_d_free')}	phase funct. param. 'd' (0/1 - fixed/free) 
{ip.get('phase_function_k')}		{ip.get('phase_funtion_k_free')}	phase funct. param. 'k' (0/1 - fixed/free) 
{ip.get('lambert_coefficient')}		{ip.get('lambert_coefficient_free')}	Lambert coefficient 'c' (0/1 - fixed/free)      
{ip.get('itteration_stop')}			iteration stop condition
    """

    with open(config["FILES"].get("inputConfig"), "w+") as file:
        file.write(file_data)


def runFiles(config):
    minkowskiLocation = config["FILES"].get("minkowski")
    standardtriLocation = config["FILES"].get("standardtri")
    convexinvLocation = config["FILES"].get("convexinv")

    lightCurveFile = config["FILES"].get("lightcurvedata", fallback="lcs.txt")
    inputConfigFile = config["FILES"].get("inputConfig", fallback="input_convexinv.txt")
    outputlcFile = config["FILES"].get("outputlc", fallback="output_lcs.txt")

    # Maybe enable saving output after each call?
    p = Popen(["cat", lightCurveFile], stdout=PIPE, stdin=PIPE)
    output, err = p.communicate()

    p = Popen([convexinvLocation, "-s", inputConfigFile, outputlcFile], stdout=PIPE, stdin=PIPE)
    output, err = p.communicate(output)
    verbosePrint("convexinv finished")
    p = Popen([minkowskiLocation], stdout=PIPE, stdin=PIPE)
    output, err = p.communicate(output)
    verbosePrint("minkowski finished")
    p = Popen([standardtriLocation], stdout=PIPE, stdin=PIPE)
    output, err = p.communicate(output)
    verbosePrint("standardtri finished")
    return output.decode("utf-8")


def objFormatFix(shapeModel: str, objectID: str, config):
    """
    Fix the output from standardtri to the .obj format
    :param shapeModel: The output from standardtri. The first line
        is the the number of vertices and number of faces. the next section are the vertices, then faces
    :return: The shape model in the correct obj format
    """

    d = shapeModel.split('\n')

    # Get the number of vertices and face rows in the model
    nrows = d[0].split(' ')
    nver, nface = list(filter(''.__ne__, nrows))

    for i in range(1, int(nver) + 1):
        d[i] = 'v ' + d[i]
    for i in range(int(nver) + 1, int(nface) + int(nver) + 1):
        d[i] = 'f ' + d[i]

    shapeModel = '\n'.join(d)
    file = config["FILES"].get("shapeModelFile", fallback="model.obj")
    file = file.replace("%I", objectID)
    with open(file, 'w+') as f:
        f.write(shapeModel)


def main(args):
    assert os.path.isfile(args.config), f"Config file \"{args.config}\" does not exist"

    import configparser
    config = configparser.ConfigParser()
    config.read(args.config)

    data = getData(args.observationsFile, config)
    verbosePrint(f"Got data with {len(data)} observations")
    period = periodFinder(data, config)
    verbosePrint(f"Period is {period:.3f} h")
    heliCoords, earthCoords = getCartesianPositions(data["jd"], args.objectID)
    verbosePrint(f"Got Cartesian Coordinates")
    if not args.flux:
        verbosePrint("Converting magnitudes to flux")
        magToFlux(data, heliCoords, earthCoords)

    writeInputFiles(data, config, heliCoords, earthCoords, period)
    verbosePrint("Input files writen")

    shapeModel = runFiles(config)
    objFormatFix(shapeModel, args.objectID, config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Python light curve inversion')
    parser.add_argument('observationsFile',
                        help='The observations for inversion in csv format')
    parser.add_argument('objectID', type=str, help="The objects id")
    parser.add_argument('--config', "-c", metavar='config', type=str, default='config.ini',
                        help="The config file to use")
    parser.add_argument("--verbose", "-v", help="increase output verbosity", action="store_true")
    parser.add_argument("--flux", "-f", help="If the input brightness values are flux", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        verbosePrint = print
    # print(args.objectID)
    main(args)
