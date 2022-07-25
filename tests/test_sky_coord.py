# See the LICENSE file at the top-level directory of this distribution.

"""Test SkyCoord functions."""

from ska_sdp_func.utility import SkyCoord

try:
    import astropy
    import astropy.coordinates
except ImportError:
    astropy = None


def test_sky_coord():
    """Test basic usage."""
    sky_coord = SkyCoord("J2000", 12.345, 34.567)
    assert sky_coord.type() == "J2000"
    assert sky_coord.value(0) == 12.345
    assert sky_coord.value(1) == 34.567


def test_sky_coord_copy():
    """Test making a copy of an existing SkyCoord."""
    sky_coord1 = SkyCoord("icrs", 56.789, 23.456)
    sky_coord2 = SkyCoord(sky_coord1)
    assert sky_coord2.type() == "icrs"
    assert sky_coord2.value(0) == 56.789
    assert sky_coord2.value(1) == 23.456


def test_astropy_coord():
    """Test creation from astropy SkyCoord class."""
    if astropy:
        astropy_coord = astropy.coordinates.SkyCoord(2.56, 1.28, unit="rad")
        sky_coord = SkyCoord(astropy_coord)
        assert sky_coord.type() == "icrs"
        assert sky_coord.value(0) == 2.56
        assert sky_coord.value(1) == 1.28
