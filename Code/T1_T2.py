import csv
import networkx as nx
import matplotlib.pyplot as plt

# --- Data containers ---
locations = {}
roads = []

# --- Class Definitions ---
class Location:
    def __init__(self, loc_id, name=None):
        self.id = loc_id
        self.name = name
        self.neighbors = []

class Road:
    def __init__(self, from_loc, to_loc, distance_km, capacity, condition, traffic=None):
        self.from_loc = from_loc
        self.to_loc = to_loc
        self.distance_km = distance_km
        self.capacity = capacity
        self.condition = condition
        self.traffic = traffic or {}

# --- Load neighborhoods.csv ---
with open("neighborhoods.csv", "r") as file:
    csvreader = csv.reader(file)
    next(csvreader)  # Skip header
    for row in csvreader:
        loc_id = row[0]
        name = row[1]
        if loc_id not in locations:
            locations[loc_id] = Location(loc_id, name)
        else:
            locations[loc_id].name = name

# --- Load existing_roads.csv ---
with open('existing_roads.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        from_id = row['from_id']
        to_id = row['to_id']

        if from_id not in locations:
            locations[from_id] = Location(from_id)
        if to_id not in locations:
            locations[to_id] = Location(to_id)

        road = Road(
            from_loc=locations[from_id],
            to_loc=locations[to_id],
            distance_km=float(row['distance_km']),
            capacity=int(row['capacity']),
            condition=int(row['condition']),
        )

        roads.append(road)
        locations[from_id].neighbors.append(road)

        # Since bidirectional, add reverse road too
        reverse_road = Road(
            from_loc=locations[to_id],
            to_loc=locations[from_id],
            distance_km=float(row['distance_km']),
            capacity=int(row['capacity']),
            condition=int(row['condition']),
        )
        roads.append(reverse_road)
        locations[to_id].neighbors.append(reverse_road)

# --- Load traffic_flow.csv ---
with open('traffic_flow.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        road_id = row['road_id']
        from_id, to_id = road_id.split('-')

        traffic = {
            'morning_peak': int(row['morning_peak']),
            'afternoon': int(row['afternoon']),
            'evening_peak': int(row['evening_peak']),
            'night': int(row['night'])
        }

        for road in roads:
            if (road.from_loc.id == from_id and road.to_loc.id == to_id):
                road.traffic = traffic
            if (road.from_loc.id == to_id and road.to_loc.id == from_id):
                road.traffic = traffic

print(f"Total Locations (Nodes): {len(locations)}")
print(f"Total Roads (Edges): {len(roads)}")


# ----------------- Visualize the Graph -----------------
G = nx.Graph()

# Add nodes
for loc_id, loc in locations.items():
    G.add_node(loc_id, label=loc.name)

# Add edges
for road in roads:
    label = f"{road.distance_km}km"
    G.add_edge(road.from_loc.id, road.to_loc.id, weight=road.distance_km, label=label)

# Draw
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
nx.draw_networkx_edges(G, pos, edge_color='gray')
labels = {node: f"{node}\n{data['label']}" for node, data in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("🌍 City Road Network (Bidirectional)")
plt.axis('off')
plt.tight_layout()
plt.show()

# ----------------- Prim's Algorithm: Build MST -----------------
mst = nx.minimum_spanning_tree(G, algorithm='prim')

# Calculate total cost
cost_per_km = 75_000_000  # 75M EGP per km
total_cost = 0
suggested_roads = []

for u, v, data in mst.edges(data=True):
    distance = data['weight']
    cost = distance * cost_per_km
    total_cost += cost
    suggested_roads.append((u, v, distance, cost))

# Budget constraints
budget = 2_500_000_000  # 2.5B EGP

print("\n🏗️ Suggested New Roads (Prim's MST):")
for from_id, to_id, distance, cost in suggested_roads:
    print(f"{from_id} <--> {to_id}: {distance:.2f} km, Cost: {cost/1e6:.2f}M EGP")

print(f"\n🧮 Total Cost: {total_cost/1e9:.2f}B EGP")

if total_cost <= budget:
    print("✅ The proposed roads fit within the 2.5B EGP budget!")
else:
    print("❌ The proposed roads exceed the 2.5B EGP budget.")
