from tiatoolbox.annotation.storage import SQLiteStore

# Load detection results
store = SQLiteStore('/media/u1910100/data/overlays/CMU-1-Small-Region_pannuke.db')

# Access detection points
for annotation in store.values():
    geometry = annotation.geometry  # Point coordinates
    properties = annotation.properties  # Cell type, confidence, etc.
    print(f"Cell at ({geometry.x}, {geometry.y}): {properties}")

store.close()