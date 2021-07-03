import orbit
import numpy as np
from astropy.table import Table
from astropy import io
import os
import scipy

try:
    import cuSignal
except ModuleNotFoundError:
    pass
from subprocess import Popen, PIPE


def periodFinder(data, config):
    if config["periods"].get("period", default=-1) > 0:
        return config["periods"].get("period")

    if config["periods"].getboolean("useCPU", default=True):
        # Use SciPy
        pass
    else:
        # Use cuSignal
        pass


def getData(file, config):
    return io.ascii.read(file)


def getCartesianPositions(times, objectID):
    ephs = getEphemerides(objectID)
    obj = orbit.Orbit(*ephs)
    heliCoords = []
    earthCoords = []
    for t in times:
        heliCoords.append(obj.calcCartesianCoords(t))
        earthCoords.append(obj.calcECFCoords(t))

    return np.array(heliCoords), np.array(earthCoords)


def getEphemerides(objectID):
    pass


def writeInputFiles(data, config, heliCoords, earthCoords):
    pass


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

    p = Popen([minkowskiLocation], stdout=PIPE, stdin=PIPE)
    output, err = p.communicate(output)

    p = Popen([standardtriLocation], stdout=PIPE, stdin=PIPE)
    output, err = p.communicate(output)

    return output.decode("utf-8")


def objFormatFix(shapeModel:str):
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
    with open("model.obj", 'w+') as f:
        f.write(shapeModel)


def main(args):
    assert os.path.isfile(args.config), f"Config file \"{args.config}\" does not exist"

    import configparser
    config = configparser.ConfigParser()
    config.read(args.config)

    shapeModel = runFiles(config)
    objFormatFix(shapeModel)
    exit()
    data = getData(args.observationsFile, config)

    period = periodFinder(data, config)

    heliCoords, earthCoords = getCartesianPositions(data)

    writeInputFiles(data, config, heliCoords, earthCoords)

    runFiles(config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Python light curve inversion')
    parser.add_argument('observations file', metavar='observationsFile',
                        help='The observations for inversion in csv format')
    parser.add_argument('Object ID', metavar='objID', type=str, help="The objects id")
    parser.add_argument('--config', metavar='config', type=str, default='config.ini', help="The config file to use")
    args = parser.parse_args()

    main(args)
