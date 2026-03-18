"""
modules/metadata_spoof.py

Strips AI generation metadata from output images and injects realistic
EXIF data to mimic photos taken by real cameras / phones.

Requires: piexif>=1.1.3 (already in requirements_versions.txt)

Usage:
    from modules.metadata_spoof import apply_camera_exif

    # Returns bytes to pass to image.save(..., exif=...)
    exif_bytes = apply_camera_exif(
        camera_profile="iPhone 15 Pro Max",
        gps_coords=(40.7128, -74.0060),    # (lat, lon) or None
        gps_jitter=True,                    # add ±0.001° random noise
        custom_photographer=None,
    )
"""

import random
import struct
from datetime import datetime, timedelta

try:
    import piexif
    _PIEXIF_AVAILABLE = True
except ImportError:
    _PIEXIF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Camera profiles
# ---------------------------------------------------------------------------

CAMERA_PROFILES = {
    "iPhone 15 Pro Max": {
        "Make": "Apple",
        "Model": "iPhone 15 Pro Max",
        "Software": "17.4.1",
        "LensMake": "Apple",
        "LensModel": "iPhone 15 Pro Max back triple camera 6.765mm f/1.78",
        "FocalLength": (6765, 1000),          # 6.765 mm
        "FocalLengthIn35mmFilm": 24,
        "FNumber_options": [(178, 100), (200, 100), (280, 100)],   # f/1.78, 2.0, 2.8
        "ISOSpeedRatings_options": [50, 64, 100, 200, 400],
        "ExposureTime_options": [                                    # shutter speed as rational
            (1, 60), (1, 120), (1, 250), (1, 500), (1, 1000), (1, 2000)
        ],
        "ExposureProgram": 2,   # Normal program
        "MeteringMode": 5,      # Pattern
        "Flash": 16,            # Flash did not fire, compulsory mode
        "WhiteBalance": 0,      # Auto
        "ColorSpace": 65535,    # Uncalibrated / sRGB-like
    },
    "Samsung Galaxy S24 Ultra": {
        "Make": "SAMSUNG",
        "Model": "SM-S928B",
        "Software": "S928BXXS3AXD6",
        "LensMake": "Samsung",
        "LensModel": "Samsung Galaxy S24 Ultra Rear Main Camera",
        "FocalLength": (6300, 1000),
        "FocalLengthIn35mmFilm": 23,
        "FNumber_options": [(170, 100), (180, 100), (220, 100)],
        "ISOSpeedRatings_options": [50, 100, 200, 400, 800],
        "ExposureTime_options": [
            (1, 60), (1, 100), (1, 250), (1, 500), (1, 1000)
        ],
        "ExposureProgram": 2,
        "MeteringMode": 5,
        "Flash": 0,
        "WhiteBalance": 0,
        "ColorSpace": 1,
    },
    "Canon EOS R5": {
        "Make": "Canon",
        "Model": "Canon EOS R5",
        "Software": "Firmware Version 2.00",
        "LensMake": "Canon",
        "LensModel": "RF85mm F1.2 L USM",
        "FocalLength": (85000, 1000),
        "FocalLengthIn35mmFilm": 85,
        "FNumber_options": [(120, 100), (140, 100), (180, 100), (200, 100), (280, 100), (400, 100)],
        "ISOSpeedRatings_options": [100, 200, 400, 800],
        "ExposureTime_options": [
            (1, 60), (1, 125), (1, 250), (1, 500), (1, 1000), (1, 2000)
        ],
        "ExposureProgram": 1,   # Manual
        "MeteringMode": 5,
        "Flash": 0,
        "WhiteBalance": 0,
        "ColorSpace": 1,
    },
    "Sony A7IV": {
        "Make": "SONY",
        "Model": "ILCE-7M4",
        "Software": "ILCE-7M4 v2.00",
        "LensMake": "Sony",
        "LensModel": "FE 50mm F1.4 GM",
        "FocalLength": (50000, 1000),
        "FocalLengthIn35mmFilm": 50,
        "FNumber_options": [(140, 100), (180, 100), (200, 100), (280, 100), (400, 100), (560, 100)],
        "ISOSpeedRatings_options": [100, 200, 400, 800],
        "ExposureTime_options": [
            (1, 60), (1, 125), (1, 250), (1, 500), (1, 1000), (1, 2000)
        ],
        "ExposureProgram": 3,   # Aperture priority
        "MeteringMode": 2,      # Center weighted
        "Flash": 0,
        "WhiteBalance": 0,
        "ColorSpace": 1,
    },
}

DEFAULT_CAMERA = "iPhone 15 Pro Max"

# ---------------------------------------------------------------------------
# GPS helpers
# ---------------------------------------------------------------------------

def _to_dms_rational(degrees: float):
    """Convert decimal degrees to (degrees, minutes, seconds) as piexif rationals."""
    d = int(abs(degrees))
    m_float = (abs(degrees) - d) * 60
    m = int(m_float)
    s_float = (m_float - m) * 60
    s_int = round(s_float * 100)
    return ((d, 1), (m, 1), (s_int, 100))


def _build_gps_ifd(lat: float, lon: float):
    lat_ref = b"N" if lat >= 0 else b"S"
    lon_ref = b"E" if lon >= 0 else b"W"
    return {
        piexif.GPSIFD.GPSLatitudeRef: lat_ref,
        piexif.GPSIFD.GPSLatitude: _to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        piexif.GPSIFD.GPSLongitude: _to_dms_rational(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSAltitude: (random.randint(0, 3000), 10),
        piexif.GPSIFD.GPSImgDirectionRef: b"T",
        piexif.GPSIFD.GPSImgDirection: (random.randint(0, 35999), 100),
    }

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def apply_camera_exif(
    camera_profile: str = DEFAULT_CAMERA,
    gps_coords: tuple | None = None,
    gps_jitter: bool = True,
    custom_photographer: str | None = None,
    base_datetime: datetime | None = None,
) -> bytes:
    """
    Build a piexif EXIF blob that mimics a photo from the given camera.

    Parameters
    ----------
    camera_profile : str
        Key from CAMERA_PROFILES dict.
    gps_coords : (lat, lon) float tuple or None
        Decimal degrees. None = omit GPS data.
    gps_jitter : bool
        Add ±0.001° random noise to GPS coords for realism.
    custom_photographer : str or None
        Injected as ImageDescription / Artist IPTC field.
    base_datetime : datetime or None
        Base time for the photo. Random small jitter added.
        Defaults to a random recent datetime.

    Returns
    -------
    bytes
        Raw EXIF bytes suitable for PIL image.save(exif=...).
        Returns b"" if piexif is not installed.
    """
    if not _PIEXIF_AVAILABLE:
        return b""

    profile = CAMERA_PROFILES.get(camera_profile, CAMERA_PROFILES[DEFAULT_CAMERA])

    # Randomised timestamp
    if base_datetime is None:
        # Random date within the last 180 days
        days_ago = random.randint(0, 180)
        hours = random.randint(7, 20)
        minutes = random.randint(0, 59)
        seconds = random.randint(0, 59)
        base_datetime = datetime.now() - timedelta(days=days_ago)
        base_datetime = base_datetime.replace(hour=hours, minute=minutes, second=seconds)

    dt_str = base_datetime.strftime("%Y:%m:%d %H:%M:%S").encode()

    # Randomise from option lists
    fnumber = random.choice(profile["FNumber_options"])
    iso = random.choice(profile["ISOSpeedRatings_options"])
    shutter = random.choice(profile["ExposureTime_options"])

    # Build 0th IFD
    zeroth_ifd = {
        piexif.ImageIFD.Make: profile["Make"].encode(),
        piexif.ImageIFD.Model: profile["Model"].encode(),
        piexif.ImageIFD.Software: profile["Software"].encode(),
        piexif.ImageIFD.DateTime: dt_str,
        piexif.ImageIFD.Orientation: 1,
        piexif.ImageIFD.YCbCrPositioning: 1,
        piexif.ImageIFD.XResolution: (72, 1),
        piexif.ImageIFD.YResolution: (72, 1),
        piexif.ImageIFD.ResolutionUnit: 2,
    }
    if custom_photographer:
        zeroth_ifd[piexif.ImageIFD.ImageDescription] = custom_photographer.encode()
        zeroth_ifd[piexif.ImageIFD.Artist] = custom_photographer.encode()

    # Build Exif IFD
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: dt_str,
        piexif.ExifIFD.DateTimeDigitized: dt_str,
        piexif.ExifIFD.ExposureTime: shutter,
        piexif.ExifIFD.FNumber: fnumber,
        piexif.ExifIFD.ISOSpeedRatings: iso,
        piexif.ExifIFD.FocalLength: profile["FocalLength"],
        piexif.ExifIFD.FocalLengthIn35mmFilm: profile["FocalLengthIn35mmFilm"],
        piexif.ExifIFD.ExposureProgram: profile["ExposureProgram"],
        piexif.ExifIFD.MeteringMode: profile["MeteringMode"],
        piexif.ExifIFD.Flash: profile["Flash"],
        piexif.ExifIFD.WhiteBalance: profile["WhiteBalance"],
        piexif.ExifIFD.ColorSpace: profile["ColorSpace"],
        piexif.ExifIFD.ExposureBiasValue: (0, 10),
        piexif.ExifIFD.MaxApertureValue: (fnumber[0], fnumber[1] * 10),
        piexif.ExifIFD.SubjectDistanceRange: 0,
        piexif.ExifIFD.SceneCaptureType: 0,
        piexif.ExifIFD.ExposureMode: 0,  # Auto exposure
        piexif.ExifIFD.DigitalZoomRatio: (100, 100),
    }

    if "LensModel" in profile:
        exif_ifd[piexif.ExifIFD.LensModel] = profile["LensModel"].encode()
    if "LensMake" in profile:
        exif_ifd[piexif.ExifIFD.LensMake] = profile["LensMake"].encode()

    # Build GPS IFD
    gps_ifd = {}
    if gps_coords is not None and _PIEXIF_AVAILABLE:
        lat, lon = gps_coords
        if gps_jitter:
            lat += random.uniform(-0.001, 0.001)
            lon += random.uniform(-0.001, 0.001)
        gps_ifd = _build_gps_ifd(lat, lon)

    exif_dict = {
        "0th": zeroth_ifd,
        "Exif": exif_ifd,
        "GPS": gps_ifd,
        "1st": {},
    }

    try:
        return piexif.dump(exif_dict)
    except Exception:
        return b""


def is_available() -> bool:
    return _PIEXIF_AVAILABLE


def list_camera_profiles() -> list[str]:
    return list(CAMERA_PROFILES.keys())
