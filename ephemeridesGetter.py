import pandas as pd
import astropy.units as u

orbitEphNeeded = ["a", "e", "i", "Peri", "Node", "n", "Epoch", "M"]
ephUnits = [u.AU, 1, u.deg, u.deg, u.deg, u.deg/u.d, u.d, u.deg]

class MPC():

    def __init__(self, file: str = None):
        self.file: str = file
        if self.file is None:
            import wget
            import gzip
            import shutil
            self.file = "mpcorb_extended.json"
            wget.download("https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz",
                          out=f"{self.file}.gz")
            with gzip.open(f"{self.file}.gz", 'rb') as f_in:
                with open(self.file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        self.data: pd.DataFrame = pd.read_json(self.file)
        self.data["Number"] = self.data["Number"].astype(str)
        self.data["Number"] = self.data["Number"].map(lambda x: x.lstrip('(').rstrip(')'))

    def __getitem__(self, key):
        return self.data[self.data["Number"] == key].iloc[0]

    def getEphemerides(self, key) -> tuple:
        obj = self[key]
        eph = tuple(obj[orbitEphNeeded])
        toReturn = []
        for ele, unit in zip(eph, ephUnits):
            toReturn.append(ele*unit)
        return tuple(toReturn)
