import random
import math
from shapely.geometry import Polygon, Point

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def generate_kml_with_balanced_sampling(field_coords, drone_waypoints, coverage_percentage=15, lower_altitude=12, min_distance=0.0005, filename="drone_mission_balanced.kml"):
    """
    Generates a KML file with a polygon field boundary and selects sample points alternately from the top and bottom halves.
    :param field_coords: List of tuples representing the polygon boundary [(lon, lat), ...]
    :param drone_waypoints: List of tuples representing the full set of waypoints [(lon, lat), ...]
    :param coverage_percentage: Percentage of waypoints to be selected for low-altitude imaging (fixed at 15%)
    :param lower_altitude: Altitude in meters where the drone will descend to take pictures
    :param min_distance: Minimum allowed distance between sampled waypoints
    :param filename: Output KML file name
    """
    # Convert coordinates to a polygon
    field_polygon = Polygon(field_coords)
    
    # Ensure waypoints are inside the polygon
    valid_waypoints = [wp for wp in drone_waypoints if field_polygon.contains(Point(wp))]

    # Fix the sampling count at exactly 15% of total waypoints
    num_sampled_waypoints = math.ceil((coverage_percentage / 100) * len(valid_waypoints))

    # Compute mid latitude to divide the field into two halves
    _, y_min, _, y_max = field_polygon.bounds
    mid_latitude = (y_min + y_max) / 2

    # Separate waypoints into top and bottom halves
    top_half_waypoints = [wp for wp in valid_waypoints if wp[1] >= mid_latitude]
    bottom_half_waypoints = [wp for wp in valid_waypoints if wp[1] < mid_latitude]

    sampled_waypoints = []
    attempts = 0
    max_attempts = 10000  # Prevent infinite loops

    while len(sampled_waypoints) < num_sampled_waypoints and attempts < max_attempts:
        # Alternate between top and bottom halves
        if len(sampled_waypoints) % 2 == 0:  
            source_list = top_half_waypoints
        else:  
            source_list = bottom_half_waypoints

        if not source_list:
            continue

        candidate = random.choice(source_list)

        # Ensure the new waypoint is far enough from existing ones
        if all(distance(candidate, existing) > min_distance for existing in sampled_waypoints):
            sampled_waypoints.append(candidate)
            source_list.remove(candidate)

        attempts += 1

    # Generate KML content
    kml_template = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Drone Mission with Balanced Sampling</name>
    <Placemark>
      <name>Field Boundary</name>
      <Style>
        <LineStyle>
          <color>ff0000ff</color>
          <width>3</width>
        </LineStyle>
        <PolyStyle>
          <color>7fff0000</color>
          <fill>1</fill>
          <outline>1</outline>
        </PolyStyle>
      </Style>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {polygon_coords}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
    {waypoints}
  </Document>
</kml>"""
    
    waypoint_template = """
    <Placemark>
      <name>Waypoint {id}</name>
      <Style>
        <IconStyle>
          <color>{color}</color>
        </IconStyle>
      </Style>
      <Point>
        <coordinates>{lon},{lat},{alt}</coordinates>
      </Point>
    </Placemark>"""
    
    # Generate waypoints with standard altitude
    kml_waypoints = "\n".join(
        [waypoint_template.format(id=i+1, lon=lon, lat=lat, alt=80, color="ff0000ff") for i, (lon, lat) in enumerate(valid_waypoints)]
    )
    
    # Generate sampled waypoints with lower altitude for image capture (different color)
    kml_sampled_waypoints = "\n".join(
        [waypoint_template.format(id=i+1+len(valid_waypoints), lon=lon, lat=lat, alt=lower_altitude, color="ff00ff00") for i, (lon, lat) in enumerate(sampled_waypoints)]
    )
    
    final_kml = kml_template.format(
        polygon_coords=" ".join([f"{lon},{lat},0" for lon, lat in field_coords] + [f"{field_coords[0][0]},{field_coords[0][1]},0"]),
        waypoints=kml_waypoints + "\n" + kml_sampled_waypoints
    )
    
    # Save the KML file
    with open(filename, "w") as file:
        file.write(final_kml)

    print(f"Total Waypoints: {len(drone_waypoints)}, Sample points chosen: {len(sampled_waypoints)}")
    print(f"KML File Generated: {filename}")

    return filename

# Example usage:
field_coordinates = [
    (-122.082, 37.422),
    (-122.082, 37.428),
    (-122.087, 37.428),
    (-122.087, 37.422)
]

drone_waypoints = [
    (random.uniform(-122.087, -122.082), random.uniform(37.422, 37.428)) for i in range(300)
]

# Generate the KML file with balanced sampling
generate_kml_with_balanced_sampling(field_coordinates, drone_waypoints)
