import pandas as pd

def make_country_dict():
    country = pd.read_csv("wikipedia-iso-country-codes.txt")
    country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
    country_dict["CUW"] = len(country_dict)
    country_dict["XKX"] = len(country_dict)
    country_dict["SCG"] = len(country_dict)
    country_dict["SSD"] = len(country_dict)
    country_dict["BES"] = len(country_dict)
    country_dict["NULL"] = len(country_dict)
    country_dict["NA"] = len(country_dict)
    return country_dict


with open("feature_code_dict.json", "r") as f:
    feature_code_dict = json.load(f)