import h3


def get_h3_index(lat, long, resolution):
    return h3.latlng_to_cell(lat, long, resolution)


def get_hexagon_boundary_coordinate(h3_index):
    return h3.cell_to_boundary(h3_index)


def get_hexagon_neighbors(h3_index, k):
    return h3.grid_ring(h3_index, k)


def get_center_coordinate(coordinate):
    center_lat = sum(lat for lat, _ in coordinate) / len(coordinate)
    center_lon = sum(lon for _, lon in coordinate) / len(coordinate)
    return center_lat, center_lon
