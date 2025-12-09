
import pytest

from mordecai3.geonames import (get_adm1_country_entry)



@pytest.mark.skip()
def test_get_adm1_country_entry():
    nld = get_adm1_country_entry("North Holland", "NLD", conn)
    assert nld is not None
    assert nld["geonameid"] == "2749879"
    
    # Non-existent entry
    xyz = get_adm1_country_entry("NonExistent", None, conn)
    assert xyz is None