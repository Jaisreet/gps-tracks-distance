# GPS Track Smoother

**Repository Description:**

GPS Track Smoother is a Python program designed to enhance the accuracy of GPS tracks recorded using smartphones, smartwatches, or other wearable devices. It utilizes Kalman filtering techniques to reduce sensor noise and improve the estimation of actual movement patterns in the GPS data.

## Problem Statement

GPS tracks obtained from mobile devices often contain inherent noise, which can distort the actual movement trajectory. This noise can lead to inaccurate distance calculations and misinterpretation of the traveled path. The goal of this project is to develop a solution that can effectively reduce the noise in GPS tracks and provide a smoother representation of the actual movement.

## Approach

The GPS Track Smoother program takes GPX (latitude/longitude data) and CSV (other collected fields) files as input and performs the following tasks:

1. **Reading the GPX File**: The program parses the GPX file using XML libraries and extracts latitude, longitude, and time information for each track point. It creates a DataFrame to store this data, allowing for further processing and analysis.

2. **Reading the CSV File**: Relevant compass readings (Bx and By) from the CSV file are incorporated into the DataFrame based on matching timestamps. These readings provide additional information about the device orientation during the recorded track, which can aid in improving the accuracy of the GPS data.

3. **Calculating Distances**: The program utilizes the haversine formula, a trigonometric method, to calculate the distances between latitude/longitude points in the DataFrame. This initial distance calculation provides an estimate of the total distance traveled.

4. **Applying Kalman Filtering**: Kalman filtering is a powerful technique used for state estimation, particularly in the presence of noise. The program applies a Kalman filter to the GPS data, taking into account the sensor noise, compass readings, and other parameters. This filtering process reduces the noise and enhances the accuracy of the GPS tracks.

5. **Printing the Distance**: The program prints the unfiltered distance, which represents the distance calculated based on the original GPS tracks. It also prints the filtered distance, which reflects the improved estimation after applying Kalman filtering. Both distances are provided in meters and rounded to two decimal places.

6. **Saving the Smoothed Track**: The program saves the smoothed GPS track as a new GPX file named `out.gpx`. This file contains the filtered and enhanced representation of the GPS data after the application of Kalman filtering. You can visualize this smoothed track using various GPS track visualization tools.

## Getting Started

To run the GPS Track Smoother program, follow these steps:

1. Clone the repository: `git clone https://github.com/jaisreet/gps-track-distance.git`
2. Run the program by providing the paths to the GPX and CSV files as command-line arguments: `python calc_distance.py path/to/gps_track.gpx path/to/sensor_data.csv`

Feel free to experiment with different GPX and CSV files and adjust the Kalman filtering parameters to achieve the best results for noise reduction and track accuracy.

## Example Results

Here's an example of the program's output:

```
Unfiltered distance: 5203.45 meters
Filtered distance: 4936.12 meters
```

The unfiltered distance represents the original distance calculated from the GPS tracks, while the filtered distance reflects the improved estimation after applying Kalman filtering.

## Viewing Results

To visualize the smoothed GPS track, you can use popular GPS track visualization tools such as MyGPSFiles or GpsPrune. Simply load the `out
