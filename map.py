import folium
import gmplot
import pandas as pd
from geopy.distance import geodesic
import numpy as np
import pdfkit

level = 12
my_map1 = folium.Map(location=(32.183442,-110.975207), zoom_start=level)

armpit = (32.183442,-110.975207)
airport = (32.139985,-110.969291)
busy = (32.278645,-110.996345)
SE_unique = (32.106581,-110.788638)
airForce = (32.201925,-110.921691)

folium.CircleMarker(location=armpit, fill=True, color='blue', radius=23, tooltip='pit').add_to(my_map1)
folium.CircleMarker(location=airport, fill=True, color='blue', radius=30, tooltip='airport').add_to(my_map1)
folium.CircleMarker(location=busy, fill=True, color='blue', radius=50, tooltip='busy').add_to(my_map1)
folium.CircleMarker(location=SE_unique, fill=True, color='blue', radius=35, tooltip='SE_unique').add_to(my_map1)
folium.CircleMarker(location=airForce, fill=True, color='blue', radius=40, tooltip='airForce').add_to(my_map1)

#loc = [(32.178205, -110.926329), (32.206927, -110.926337), (32.206812, -110.892809)]
#loc1 = [(32.294110, -111.026747), (32.293643, -110.981005), (32.263161, -110.980429), (32.263477, -111.002817)]
#folium.vector_layers.Polygon(locations=loc, popup='airForce').add_to(my_map1)
#folium.vector_layers.Polygon(locations=loc1, popup='busy').add_to(my_map1)

df = pd.read_excel("truliaHouse_clean.xlsx")

armpit_radius_km = 23/90 * 2
airport_radius_km = 30/90 * 2
busy_radius_km = 50/90 * 2
SE_unique_radius_km = 35/90 * 2
airForce_radius_km = 40/90 * 2

# armpit_region = np.array(df.shape[0], 3)
# airport_region = np.array(df.shape[0], 3)
# busy_region = np.array(df.shape[0], 3)
# SE_unique_region = np.array(df.shape[0], 3)
# airForce_region = np.array(df.shape[0], 3)
# regular_region = np.array(df.shape[0], 3)

# for i in range(df.shape[0]):
#     a = (df["Lat"][i], df["Lon"][i])
#     if geodesic(armpit,a).km <= armpit_radius_km:
#         armpit_region[i,0] = df["address"][i]
#         armpit_region[i,1] = df["Lat"][i]
#         armpit_region[i,2] = df["Lon"][i]
#     elif geodesic(airport,a).km <= airport_radius_km:
#         airport_region[i, 0] = df["address"][i]
#         airport_region[i, 1] = df["Lat"][i]
#         airport_region[i, 2] = df["Lon"][i]
#     elif geodesic(busy,a).km <= busy_radius_km:
#         busy_region[i, 0] = df["address"][i]
#         busy_region[i, 1] = df["Lat"][i]
#         busy_region[i, 2] = df["Lon"][i]
#     elif geodesic(SE_unique,a) <= SE_unique_radius_km:
#         SE_unique_region[i, 0] = df["address"][i]
#         SE_unique_region[i, 1] = df["Lat"][i]
#         SE_unique_region[i, 2] = df["Lon"][i]
#     elif geodesic(airForce,a) <= airForce_radius_km:
#         airForce_region[i, 0] = df["address"][i]
#         airForce_region[i, 1] = df["Lat"][i]
#         airForce_region[i, 2] = df["Lon"][i]
#     else:
#         regular_region[i, 0] = df["address"][i]
#         regular_region[i, 1] = df["Lat"][i]
#         regular_region[i, 2] = df["Lon"][i]

region = []

for i in range(df.shape[0]):
    if np.isnan(df["Lat"][i]) and np.isnan(df["Lon"][i]):
        region.append(6)

    else:
        a = (df["Lat"][i], df["Lon"][i])
        if geodesic(armpit, a).km <= armpit_radius_km:
            region.append(1)
        elif geodesic(airport, a).km <= airport_radius_km:
            region.append(2)
        elif geodesic(busy, a).km <= busy_radius_km:
            region.append(3)
        elif geodesic(SE_unique, a).km <= SE_unique_radius_km:
            region.append(4)
        elif geodesic(airForce, a).km <= airForce_radius_km:
            region.append(5)
        else:
            region.append(6)


df["region"] = region

count_armpit = 0
count_airport = 0
count_busy = 0
count_SE_unique = 0
count_airForce = 0

sum_armpit = 0
sum_airport = 0
sum_busy = 0
sum_SE_unique = 0
sum_airForce = 0

for i in range(df.shape[0]):

    if df["region"][i] == 1:
        count_armpit += 1
        sum_armpit += int(df["price"][i])
        folium.Marker(location=(df["Lat"][i],df["Lon"][i]), icon=folium.Icon(color="red")).add_to(my_map1)
    elif df["region"][i] == 2:
        count_airport += 1
        sum_airport += int(df["price"][i])
        folium.Marker(location=(df["Lat"][i], df["Lon"][i]), icon=folium.Icon(color="purple")).add_to(my_map1)
    elif df["region"][i] == 3:
        count_busy += 1
        sum_busy += int(df["price"][i])
        folium.Marker(location=(df["Lat"][i], df["Lon"][i]), icon=folium.Icon(color="green")).add_to(my_map1)
    elif df["region"][i] == 4:
        count_SE_unique += 1
        sum_SE_unique += int(df["price"][i])
        folium.Marker(location=(df["Lat"][i], df["Lon"][i]), icon=folium.Icon(color="gray")).add_to(my_map1)
    elif df["region"][i] == 5:
        count_airForce += 1
        sum_airForce += int(df["price"][i])
        folium.Marker(location=(df["Lat"][i], df["Lon"][i]), icon=folium.Icon(color="blue")).add_to(my_map1)
    else:
        continue
        # if np.isnan(df["Lat"][i]) and np.isnan(df["Lon"][i]):
        #     continue
        # else:
        #     folium.Marker(location=(df["Lat"][i], df["Lon"][i]), icon=folium.Icon(color="black")).add_to(my_map1)

price_ave_armpit = '${:,.2f}'.format(int(sum_armpit / count_armpit))
price_ave_airport = '${:,.2f}'.format(int(sum_airport / count_airport))
price_ave_busy = '${:,.2f}'.format(int(sum_busy / count_busy))
price_ave_SE_unique = '${:,.2f}'.format(int(sum_SE_unique / count_SE_unique))
price_ave_airForce = '${:,.2f}'.format(int(sum_airForce / count_airForce))

folium.Marker(armpit, icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(100, -20),
                                          html='<div style="font-size: 26pt">%s</div>' % price_ave_armpit)).add_to(my_map1)
folium.Marker(airport, icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(100, -20),
                                          html='<div style="font-size: 26pt">%s</div>' % price_ave_airport)).add_to(my_map1)
folium.Marker(busy, icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(100, -50),
                                          html='<div style="font-size: 26pt">%s</div>' % price_ave_busy)).add_to(my_map1)
folium.Marker(SE_unique, icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(100, -30),
                                          html='<div style="font-size: 26pt">%s</div>' % price_ave_SE_unique)).add_to(my_map1)
folium.Marker(airForce, icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(100, -30),
                                          html='<div style="font-size: 26pt">%s</div>' % price_ave_airForce)).add_to(my_map1)

my_map1.save("my_map1.html")

#pdfkit.from_file('my_map1.html', 'my_map1.pdf')
