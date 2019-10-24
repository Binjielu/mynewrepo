import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut
from geopy import geocoders

df = pd.read_excel("house_1.xlsx")


#geolocator = Nominatim(user_agent="Data Scientist", timeout=30)
# location1 = geolocator.geocode("3868 N Canyon Ranch Dr, Tucson, AZ")
#a = str(df["Location"][0]) + ", " + str(df['Ciity'][0])
#print(a)

address = []
for i in range(df.shape[0]):
    a = str(df["Location"][i]) + ", " + str(df['Ciity'][i])
    address.append(a)

df["address"] = address
lat = []
lon = []
geolocator = geocoders.GoogleV3(api_key="AIzaSyCPzoC_At418bJVXguFrnANfYuIYkT7LpQ")
# geolocator = Nominatim(scheme='http', timeout=30)

for i in range(df.shape[0]):
    try:
        a = geolocator.geocode(df["address"][i])
        lat.append(a.latitude)
        lon.append(a.longitude)
        print(f"Number {i} house's latitdue and longitude is {a.latitude} and {a.longitude}.")
    except AttributeError:
        lat.append(0)
        lon.append(0)
        print(f"Number {i} Address: {df['address'][i]} has no latitude and longitude.")
    except GeocoderTimedOut as e:
        lat.append(0)
        lon.append(0)
        print("Error: geocode failed on input %s with message %s" % (df["address"][i], e.message))


cord = [(x,y) for x, y in zip(lat,lon)]
df["coordinate"] = cord



print("")