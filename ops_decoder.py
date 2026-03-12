import xir

g = xir.Graph.deserialize("Model_0312.xmodel")
root = g.get_root_subgraph()

for sg in root.children:
    print("SUBGRAPH:", sg.get_name())
    print("DEVICE:", sg.get_attr("device"))
    
    for op in sg.get_ops():
        print("   ", op.get_name(), op.get_type())