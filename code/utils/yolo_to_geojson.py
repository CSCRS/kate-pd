# This script converts YOLO TXT annotations to GeoJSON polygons.

import os
import geopandas as gpd
import re
import json
from shapely.geometry import shape
from geojson import Feature, FeatureCollection, Polygon
from tqdm import tqdm

width = 512
height = 512

def parse_yolo_polygon(line):
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    # Convert normalized to pixel coordinates
    pixel_coords = [(coords[i] * width, coords[i+1] * height) for i in range(0, len(coords), 2)]
    return class_id, pixel_coords


maxar_grid = gpd.read_file(f'../data/preprocessing/grid_maxar.geojson')
airbus_grid = gpd.read_file(f'../data/preprocessing/grid_pleidas.geojson')

features = []
data_dir='../data/katepd'
label_dir = os.path.join(data_dir, 'labels')
for partition in os.listdir(label_dir):
    for file in tqdm(os.listdir(os.path.join(label_dir, partition))):
        if not file.endswith('.txt'):
            continue
        # Find the first number in val_tile_maxar_1355_patch_172.txt 
        tile_id = int(re.search(r'_(\d+)_', file).group(1))
        source = 'maxar' if 'maxar' in file else 'airbus'
        if 'maxar' in file:
            polygon = shape(maxar_grid.loc[maxar_grid['id'] == tile_id, 'geometry'].values[0])
        else:
            polygon = shape(airbus_grid.loc[airbus_grid['id'] == tile_id, 'geometry'].values[0])
        minx, miny, maxx, maxy = polygon.bounds  # geographic bounds

        def pixel_to_geo(x, y):
            lon = minx + (x / width) * (maxx - minx)
            lat = maxy - (y / height) * (maxy - miny)  # Y-axis usually reversed
            return lon, lat

        label_path = os.path.join(label_dir, partition, file)
        if not os.path.exists(label_path):
            print(f"Label file not found for {file}")
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            class_id, pixel_coords = parse_yolo_polygon(line)
            geo_coords = [pixel_to_geo(x, y) for x, y in pixel_coords]
            # Ensure polygon is closed
            if geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])
            poly = Polygon([geo_coords])
            features.append(Feature(geometry=poly, properties={"partition": partition, "source": source}))

fc = FeatureCollection(features)
fc['crs'] = {
    "type": "name",
    "properties": {
        "name": "urn:ogc:def:crs:EPSG::32637"
    }
}

with open('../data/preprocessing/katepd_polygons.geojson', 'w') as f:
    json.dump(fc, f, indent=2)


            