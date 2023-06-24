import sys
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

# Function to calculate distance using haversine formula
def distance(points):
    R = 6371000  # Radius of the Earth in meters
    lat1 = np.radians(points['lat'])
    lat2 = lat1.shift(1)
    lon1 = np.radians(points['lon'])
    lon2 = lon1.shift(1)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

# Read GPX file and create DataFrame
def read_gpx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    namespace = {'gpx': 'http://www.topografix.com/GPX/1/0'}
    trkpts = root.iterfind('gpx:trk/gpx:trkseg/gpx:trkpt', namespace)

    data = []
    for trkpt in trkpts:
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        time = pd.to_datetime(trkpt.find('gpx:time', namespace).text, utc=True)
        data.append([time, lat, lon])

    df = pd.DataFrame(data, columns=['datetime', 'lat', 'lon'])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df

# Read CSV file and combine data with GPX DataFrame
def read_csv(file_path, points):
    sensor_data = pd.read_csv(file_path, parse_dates=['datetime']).set_index('datetime')
    points['Bx'] = sensor_data['Bx']
    points['By'] = sensor_data['By']
    return points

# Perform Kalman filtering
def kalman_filter(points):
    initial_state = points.iloc[0]
    observation_covariance = np.diag([5, 5]) ** 2  # Assumed observation covariance
    transition_covariance = np.diag([0, 0]) ** 2  # Assumed transition covariance

    Bx_prev = points.iloc[0]['Bx']
    By_prev = points.iloc[0]['By']
    for i in range(len(points)):
        Bx = points.iloc[i]['Bx']
        By = points.iloc[i]['By']

        points.loc[i, 'lat'] += (5 * 7 * Bx) + (34 * 10 ** -7 * By)
        points.loc[i, 'lon'] += (-49 * 10 ** -7 * Bx) + (9 * 10 ** -7 * By)
        points.loc[i, 'Bx'] = By
        points.loc[i, 'By'] = By

    return points

def main():
    if len(sys.argv) != 3:
        print("Usage: python calc_distance.py <gpx_file_path> <csv_file_path>")
        return

    gpx_file = sys.argv[1]
    csv_file = sys.argv[2]

    # Read GPX file
    points = read_gpx(gpx_file)

    # Read CSV file and combine data
    points = read_csv(csv_file, points)

    # Calculate unfiltered distance
    dist = distance(points)
    print(f'Unfiltered distance: {dist:.2f} meters')

    # Apply Kalman filter
    smoothed_points = kalman_filter(points)

    # Calculate filtered distance
    smoothed_dist = distance(smoothed_points)
    print(f'Filtered distance: {smoothed_dist:.2f} meters')

    # Save smoothed track to GPX file
    output_gpx(smoothed_points, 'out.gpx')

# Function to save DataFrame to GPX file
def output_gpx(data, file_path):
    gpx = ET.Element('gpx', version="1.0")
    trk = ET.SubElement(gpx, 'trk')
    trkseg = ET.SubElement(trk, 'trkseg')

    for i, row in data.iterrows():
        trkpt = ET.SubElement(trkseg, 'trkpt', lat=str(row['lat']), lon=str(row['lon']))
        time = ET.SubElement(trkpt, 'time')
        time.text = str(row['datetime'])

    tree = ET.ElementTree(gpx)
    tree.write(file_path)

if __name__ == '__main__':
    main()
