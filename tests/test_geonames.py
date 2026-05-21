

def test_get_adm1_country_entry(geonames_service_test_data):
    nld = geonames_service_test_data.get_adm1_country_entry("North Holland", "NLD")
    assert nld is not None
    assert nld["geonameid"] == "2749879"

    # Non-existent entry
    xyz = geonames_service_test_data.get_adm1_country_entry("NonExistent", None)
    assert xyz is None