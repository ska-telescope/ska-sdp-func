from ska_sdp_func.utility import SkyCoord

try:
    import astropy.coordinates
except:
    astropy = None

def test_sky_coord():
    sky_coord = SkyCoord("J2000", 12.345, 34.567)
    assert(sky_coord.type() == "J2000")
    assert(sky_coord.value(0) == 12.345)
    assert(sky_coord.value(1) == 34.567)

def test_astropy_coord():
    if astropy:
        astropy_coord = astropy.coordinates.SkyCoord(2, 1, unit="rad")
        sky_coord = SkyCoord(astropy_coord)
        assert(sky_coord.type() == "icrs")
        assert(sky_coord.value(0) == 2)
        assert(sky_coord.value(1) == 1)
