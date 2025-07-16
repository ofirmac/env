# Let's regenerate the draw.io file using properly structured and escaped XML.
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

filename = "./doc/rl_env_flow_fixed.drawio"

# Function to escape special XML characters and replace newlines
def xml_escape(text):
    return escape(text).replace("\n", "&#xa;")

# Define nodes again with proper escaping
nodes = []
y = 20
spacing = 80
id_counter = 2

def add_node(label, shape):
    global id_counter, y
    if shape == 'ellipse':
        style = "ellipse;whiteSpace=wrap;html=1;aspect=fixed;"
        w, h = 130, 60
    elif shape == 'diamond':
        style = "rhombus;whiteSpace=wrap;html=1;"
        w, h = 140, 80
    else:
        style = "rounded=1;whiteSpace=wrap;html=1;"
        w, h = 180, 60

    label = xml_escape(label)
    node = {
        "id": str(id_counter),
        "label": label,
        "style": style,
        "x": 200,
        "y": y,
        "w": w,
        "h": h
    }
    id_counter += 1
    y += spacing
    nodes.append(node)
    return node["id"]

start  = add_node("Start / Init",           "ellipse")
reset  = add_node("Reset env\n(o0)",        "process")
policy = add_node("Agent policy πθ\nselect a_t", "process")
step   = add_node("Env.step(a_t)\nupdate state", "process")
reward = add_node("Compute reward",         "process")
done   = add_node("done?",                  "diamond")
end    = add_node("End episode\nlog metrics", "process")

edges = []
edge_id = 100

def add_edge(src, dst):
    global edge_id
    edge = {
        "id": str(edge_id),
        "style": "edgeStyle=orthogonalEdgeStyle;endArrow=block;",
        "source": src,
        "target": dst
    }
    edge_id += 1
    edges.append(edge)

# Define edges
add_edge(start, reset)
add_edge(reset, policy)
add_edge(policy, step)
add_edge(step, reward)
add_edge(reward, done)
add_edge(done, end)
add_edge(done, policy)  # loop

# Build the XML
mxfile = ET.Element('mxfile', host="app.diagrams.net")
diagram = ET.SubElement(mxfile, 'diagram', id="RL_ENV", name="RL Flow")
model = ET.SubElement(diagram, 'mxGraphModel', dx='1000', dy='1000', grid='1',
                      gridSize='10', guides='1', tooltips='1', connect='1',
                      arrows='1', fold='1', page='1', pageScale='1')
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')

# Add all nodes
for node in nodes:
    cell = ET.SubElement(root, 'mxCell',
                         id=node["id"],
                         value=node["label"],
                         style=node["style"],
                         vertex='1',
                         parent='1')
    geom = ET.SubElement(cell, 'mxGeometry',
                         x=str(node["x"]),
                         y=str(node["y"]),
                         width=str(node["w"]),
                         height=str(node["h"]))
    geom.set('as', 'geometry')

# Add all edges
for edge in edges:
    cell = ET.SubElement(root, 'mxCell',
                         id=edge["id"],
                         style=edge["style"],
                         edge='1',
                         source=edge["source"],
                         target=edge["target"],
                         parent='1')
    geom = ET.SubElement(cell, 'mxGeometry', relative='1')
    geom.set('as', 'geometry')

# Write to file
ET.ElementTree(mxfile).write(filename, encoding='utf-8', xml_declaration=True)

filename
