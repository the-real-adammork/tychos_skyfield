"""
Tychosium model implementation in Python.
Uses updated Tychosium code as reference.

Orbital parameters are loaded from orbital_params.json, which is generated
from TSN's celestial-settings.json by scripts/sync_params.py.
"""
from scipy.spatial.transform import Rotation as R
import json
import numpy as np
from pathlib import Path

_params_path = Path(__file__).parent / "orbital_params.json"
with open(_params_path) as _f:
    ORBITAL_PARAMS = json.load(_f)

# Parent-child hierarchy as defined in TSN's SolarSystem.jsx.
# Order matters: parents must be moved before children.
HIERARCHY = [
    ("earth", "polar_axis"),
    ("earth", "sun_def"),
    ("sun_def", "sun"),
    ("earth", "moon_def_a"),
    ("moon_def_a", "moon_def_b"),
    ("moon_def_b", "moon"),
    ("earth", "mercury_def_a"),
    ("mercury_def_a", "mercury_def_b"),
    ("mercury_def_b", "mercury"),
    ("earth", "venus_def_a"),
    ("venus_def_a", "venus_def_b"),
    ("venus_def_b", "venus"),
    ("earth", "mars_def_e"),
    ("mars_def_e", "mars_def_s"),
    ("mars_def_s", "mars"),
    ("mars", "phobos"),
    ("mars", "deimos"),
    ("sun", "jupiter_def"),
    ("jupiter_def", "jupiter"),
    ("sun", "saturn_def"),
    ("saturn_def", "saturn"),
    ("sun", "uranus_def"),
    ("uranus_def", "uranus"),
    ("sun", "neptune_def"),
    ("neptune_def", "neptune"),
    ("sun", "halleys_def"),
    ("halleys_def", "halleys"),
    ("earth", "eros_def_a"),
    ("eros_def_a", "eros_def_b"),
    ("eros_def_b", "eros"),
]

# Ordered list of all objects (controls move order).
ALL_OBJECTS = [
    "earth", "polar_axis", "sun_def", "sun",
    "mercury_def_a", "mercury_def_b", "mercury",
    "moon_def_a", "moon_def_b", "moon",
    "venus_def_a", "venus_def_b", "venus",
    "mars_def_e", "mars_def_s", "mars", "phobos", "deimos",
    "jupiter_def", "jupiter",
    "saturn_def", "saturn",
    "uranus_def", "uranus",
    "neptune_def", "neptune",
    "halleys_def", "halleys",
    "eros_def_a", "eros_def_b", "eros",
]

# Objects that can be observed (planets, not deferents).
OBSERVABLE_OBJECTS = [
    "sun", "mercury", "moon", "venus", "mars", "phobos", "deimos",
    "jupiter", "saturn", "uranus", "neptune", "halleys", "eros",
]


class OrbitCenter:
    """
    Data class to keep orbit center coordinates
    """

    def __init__(self, orbit_center_a=0.0, orbit_center_b=0.0, orbit_center_c=0.0):
        self.x = orbit_center_a
        self.y = orbit_center_c
        self.z = orbit_center_b


class OrbitTilt:
    """
    Data class to keep orbit tilt values
    """

    def __init__(self, orbit_tilt_a=0.0, orbit_tilt_b=0.0):
        self.x = orbit_tilt_a
        self.z = orbit_tilt_b


class PlanetObj:
    """
    Class for planet object definition. It initializes the starting objects parameters
    and allows to calculate rotations, locations and RA/DEC

    Attributes
    ----------
        orbit_radius
        orbit_center
        orbit_tilt
        start_pos
        speed
        children
        rotation
        location
        center
        radius_vec

    Methods
    -------
        move_planet_tt
        move_planet
        move_planet_basic
        add_child
        radec_direct
        location_transformed

    Notes
    -----
    move_planet() method needs to be called for parent first and only then for child.
    speed = 1/period(years) * 2pi, represents rotation of radians per year
    """

    def __init__(self, orbit_radius=100.0, orbit_center=OrbitCenter(),
                 orbit_tilt=OrbitTilt(), start_pos=20.0, speed=0.0):

        self.orbit_radius = orbit_radius
        self.orbit_center = orbit_center
        self.orbit_tilt = orbit_tilt
        self.start_pos = start_pos
        self.speed = speed / (2 * np.pi)
        self.children = []

        self.rotation = None
        self.location = None
        self.center = None
        self.radius_vec = None

        # Factory-state cache: computed once on the first call to
        # initialize_orbit_parameters() and reused on every subsequent call.
        # The cached values depend only on constructor inputs (orbit_tilt,
        # orbit_center, orbit_radius), not on julian_day, so they are constant
        # for the lifetime of the object.
        self._init_rotation_cached = None
        self._init_center_cached = None
        self._init_radius_vec_cached = None

        self.initialize_orbit_parameters()

    def initialize_orbit_parameters(self):
        """
        Initializes the object rotation, location, center position, and radius vector.

        First call computes the values from constructor inputs and caches them.
        Subsequent calls (typically from TychosSystem.move_system before each
        per-JD reset) restore from the cache instead of rebuilding scipy
        Rotation objects, which is the dominant cost in the per-eclipse hot loop.
        """
        if self._init_rotation_cached is None:
            self._init_rotation_cached = (
                R.from_euler('x', self.orbit_tilt.x, degrees=True) *
                R.from_euler('z', self.orbit_tilt.z, degrees=True)
            )
            self._init_center_cached = (
                np.array([self.orbit_center.x, self.orbit_center.y, self.orbit_center.z])
                .astype(np.float64)
            )
            self._init_radius_vec_cached = np.array([self.orbit_radius, 0.0, 0.0])

        # scipy.spatial.transform.Rotation is treated as immutable here:
        # nothing in baselib.py mutates a Rotation in-place — move_planet_basic
        # rebinds self.rotation to a new instance via composition. Sharing the
        # cached reference is therefore safe.
        self.rotation = self._init_rotation_cached
        # numpy arrays must be copied because move_planet mutates self.center
        # and self.radius_vec via in-place operations.
        self.location = np.array([0.0, 0.0, 0.0])
        self.center = self._init_center_cached.copy()
        self.radius_vec = self._init_radius_vec_cached.copy()

    def move_planet_tt(self, time_julian):
        """
        Moves planet to specified Julian time.
        NOTE: only can use function once, as every usage modifies children values.
        :param time_julian: float
            Julian time to which to move the planet
        :return: none
        """

        pos = (time_julian - 2451717.0) / 365.2425 * 360
        # 2451717 is reference Julian Date tt for date 2000-6-21 12:00:00
        self.move_planet(pos)

    def move_planet(self, pos):
        """
        Moves planet by specified degrees around y-axis.
        NOTE: only can use function once, as everytime it modifies children values.
        :param pos: float
            Position in degrees to rotate around y-axis
        :return: none
        """

        self.move_planet_basic(self.speed * pos - self.start_pos)
        for child in self.children:
            child.rotation = self.rotation * child.rotation
            child.center = self.center + self.rotation.apply(self.radius_vec + child.center)

    def move_planet_basic(self, pos, directions='y'):
        """
        Moves planet by specified pos, assuming self.speed = 0 and self.start_pos = 0.
        Can call this function multiple times - it does not modify children.
        :param pos: float or List[float]
            Position(s) in degrees to rotate around 'directions'
        :param directions: [optional] string
            The direction or multiple directions with respect which to move
        :return: none
        """

        self.rotation = self.rotation * R.from_euler(directions, pos, degrees=True)
        radius_rotated = self.rotation.apply(self.radius_vec)
        self.location = self.center + radius_rotated

    def add_child(self, child_obj):
        """
        Add child to the planet.
        NOTE: Order of move_planet() matters for the children, need to move parent first.
        :param child_obj: PlanetObj
            Child object to be added.
        :return: none
        """

        self.children += [child_obj]

    def radec_direct(self, ref_obj, polar_obj=None, epoch='j2000', formatted=True):
        """
        Calculate RA and DEC for the current location of the planet. It uses projects planet
        location to the appropriate ref frame for the epoch
        :param ref_obj: PlanetObj
            reference object with respect to which calculate RA and DEC, typically earth
        :param polar_obj: Optional[PlanetObj] = None
            reference object that contains transformation for polar axis frame which
            is used to calculate RA, DEC.
            Only required for the epoch = 'date'
        :param epoch: Optional[String]: 'j2000'(default), 'j2000June' or 'date'
            epoch specifies which 'time' is used for ra/dec calculation. 'j2000' corresponds
            to J2000 (and roughly to ICRF), 'j2000June' corresponds to the 2000/06/21 12:00:00 date
             and 'date' is frame associated with current time
        :param formatted: Optional[Boolean] = True
            If True, return formatted ra/dec using hours and degrees.
            If False, return ra/dec in radians.
        :return: tuple[String, String, Float] - (ra, dec, dist)
            ra is calculated in hours
            dec is calculated in degrees
            dist is the distance to the planet from the ref_obj in AU
        NOTE: 'j2000' epoch rotation is obtained by manually getting rotation quaternion of
        polar axis for the date 2000/01/01 12:00
        """

        if epoch == 'j2000':
            rot = R.from_quat([-0.1420654722633656, 0.6927306657799285, -0.14519055921306223,
                               0.6919980692126839])
        elif epoch == 'j2000June':
            rot = R.from_euler('zxy', [-23.439062, 0.26, 90], degrees=True)
        elif epoch == 'date':
            rot = polar_obj.rotation
        else:
            raise AttributeError("Unknown epoch provided: " + epoch +
                            ". Only epochs 'j2000', 'j2000June' and 'date' are supported." )

        unit_prime = rot.apply(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        loc_prime = np.dot(unit_prime, self.location - ref_obj.location)
        dist = np.linalg.norm(loc_prime) / 100
        dec = np.pi / 2 - np.arccos(loc_prime[1] / np.sqrt(np.dot(loc_prime, loc_prime)))
        ra = (np.sign(loc_prime[0]) *
              np.arccos(loc_prime[2] / np.sqrt(loc_prime[0] ** 2 + loc_prime[2] ** 2)))
        if ra < 0:
            ra += 2 * np.pi
        if not formatted:
            return ra, dec, dist

        dec_sgn = np.sign(dec)
        dec *= dec_sgn * 180 / np.pi
        dec_str = ("{:+.0f}deg {:02.0f}\' {:02.1f}\""
                   .format(dec_sgn * np.floor(dec), np.floor(dec % np.floor(dec) * 60),
                           np.remainder(dec % np.floor(dec) * 60, 1) * 60))
        ra *= 12 / np.pi
        ra_str = "{:.0f}h {:02.0f}m {:02.2f}s".format(
            np.floor(ra), np.floor(np.remainder(ra, 1) * 60),
            np.remainder(np.remainder(ra, 1) * 60, 1) * 60)
        return ra_str, dec_str, dist

    def location_transformed(self, ref_obj, polar_obj, epoch='j2000'):
        """
        Transform object location to be w.r.t. the ref_obj location and rotate coordinate axis
        to align with either (mostly) with ICRF frame or the frame as defined by polar_obj rotation
        corresponding to the epoch of current time
        :param ref_obj: PlanetObj
            reference object with respect to which calculate new location vector, typically earth
        :param polar_obj: PlanetObj
            reference object that contains transformation for polar axis frame that is used in
            Tychos for the RA/DEC calculation.
            Only required for the epoch = 'date'
        :param epoch: Optional[String]: 'j2000'(default), 'j2000June' or 'date'
            epoch specifies which 'time' is used for reference frame rotation. 'j2000' corresponds
            to J2000 (and roughly to the ICRF frame), while 'date' is frame associated with
            current time polar axis direction
        :return: ndarry[float, float, float]
            Location vector in the new rotated reference frame
        NOTE: 'j2000' epoch rotation is obtained by manually getting rotation quaternion of
        polar axis for the date 2000/01/01 12:00
        """

        if epoch == 'j2000':
            r1 = R.from_quat([-0.1420654722633656, 0.6927306657799285, -0.14519055921306223,
                              0.6919980692126839]).inv()
            r2 = R.from_euler('zy', [90, 90], degrees=True)
        elif epoch == 'j2000June':
            r1 = R.from_euler('ZXY', [-23.439062, 0.26, 90], degrees=True)
            r2 = R.from_euler('yx', [-90, 90], degrees=True)
        elif epoch == 'date':
            r1 = polar_obj.rotation.inv()
            r2 = R.from_euler('zy', [90, 90], degrees=True)
        else:
            raise AttributeError("Unknown epoch provided: " + epoch +
                            ". Only epochs 'j2000', 'j2000June' and 'date' are supported." )

        loc = np.transpose((r2 * r1).apply(self.location - ref_obj.location)) / 100
        return loc


class TychosSystem:
    """
    Class specifying dynamic Tychos planet system

    Attributes
    ----------
    julian_day

    Methods
    -------
    move_system
    get_all_objects
    get_observable_objects

    """

    _all_objects = ALL_OBJECTS
    _observable_objects = OBSERVABLE_OBJECTS

    def __init__(self, julian_day=2451717.0, params=None, bodies=None):
        """
        Parameters
        ----------
        julian_day : float
            Initial Julian Day for the system.
        params : dict, optional
            Orbital parameter overrides (same structure as ORBITAL_PARAMS).
        bodies : list[str], optional
            If given, only initialize these observable bodies (e.g. ["sun", "moon"])
            plus earth and any deferents they depend on. Speeds up move_system()
            by skipping unused bodies.
        """
        self.julian_day = julian_day
        self._objs = {}
        if bodies is not None:
            needed = self._resolve_needed_bodies(bodies)
            self._all_objects = [b for b in ALL_OBJECTS if b in needed]
            self._observable_objects = [b for b in OBSERVABLE_OBJECTS if b in needed]
        self._initialize_objects(params)
        self._set_dependencies()
        self.move_system(julian_day)

    @staticmethod
    def _resolve_needed_bodies(bodies):
        """Given a list of observable body names, return the full set of
        bodies needed (including earth, polar_axis, and all deferents in
        the dependency chain)."""
        needed = {"earth", "polar_axis"}
        needed.update(b.lower() for b in bodies)
        # Walk HIERARCHY to find all ancestors of requested bodies
        # Repeat until stable (handles multi-level chains)
        changed = True
        while changed:
            changed = False
            for parent, child in HIERARCHY:
                if child in needed and parent not in needed:
                    needed.add(parent)
                    changed = True
        return needed

    def __getitem__(self, item):
        item = item.lower()
        try:
            obj = self._objs[item]
            return obj
        except Exception as e:
            raise AttributeError(
                "Unknown object {0}, possible objects: {1}"
                .format(item, self.get_all_objects())) from e

    def _initialize_objects(self, params=None):
        """
        Defines initial parameters for each planet.
        :param params: Optional[dict] override for ORBITAL_PARAMS.
            Same structure as ORBITAL_PARAMS — keys are object names,
            values are dicts with orbit_radius, orbit_center_a/b/c,
            orbit_tilt_a/b, start_pos, speed.
        :return: none
        """
        orbital_params = params if params is not None else ORBITAL_PARAMS
        active = set(self._all_objects)
        for name, p in orbital_params.items():
            if name not in active:
                continue
            self._objs[name] = PlanetObj(
                p["orbit_radius"],
                OrbitCenter(p["orbit_center_a"], p["orbit_center_b"], p["orbit_center_c"]),
                OrbitTilt(p["orbit_tilt_a"], p["orbit_tilt_b"]),
                p["start_pos"],
                p["speed"],
            )

    def _add_child(self, parent, child):
        """
        A wrapper around parent object add_child() to specify parent and child objects as strings
        :param parent: string
        :param child: string
        :return: none
        """

        self._objs[parent].add_child(self._objs[child])

    def _set_dependencies(self):
        """
        Sets the dependencies between the system objects from HIERARCHY.
        :return: none
        """
        for parent, child in HIERARCHY:
            if parent in self._objs and child in self._objs:
                self._add_child(parent, child)

    def move_system(self, julian_day):
        """
        Moves the system to the specified julian time.
        It re-initializes each object parameters before executing the move
        :param julian_day: float
            Julian Day to which move the Tychos object system
        :return: none
        """

        self.julian_day = julian_day
        for p in self._all_objects:
            self._objs[p].initialize_orbit_parameters()

        self._objs["polar_axis"].move_planet_basic([-23.439062, 0.26], 'zx')
        self._objs["earth"].move_planet_basic(90)
        for p in self._all_objects:
            self._objs[p].move_planet_tt(julian_day)

    def get_all_objects(self):
        """
        Returns all objects in this system (may be a subset if bodies= was used).
        :return: list[string]
        """
        return list(self._all_objects)

    def get_observable_objects(self):
        """
        Returns observable objects in this system (may be a subset if bodies= was used).
        :return: list[string]
        """
        return list(self._observable_objects)

    def has_body(self, name: str) -> bool:
        """Check whether a body is available in this system instance."""
        return name.lower() in self._objs
