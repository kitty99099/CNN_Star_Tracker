import argparse
import xir

def main():
    parser = argparse.ArgumentParser(description="List subgraphs and CPU ops in an XIR .xmodel")
    parser.add_argument("xmodel", nargs="?",
                        help="Path to .xmodel file (default: ./quantize_result/Model_V2.xmodel)")
    args = parser.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    root = g.get_root_subgraph()
    for i, sg in enumerate(root.toposort_child_subgraph()):
        dev = sg.get_attr("device") if sg.has_attr("device") else "CPU"
        print(f"#{i} name={sg.get_name()} device={dev}")
        if str(dev).upper() != "DPU":
            ops = [n.get_name() for n in sg.toposort()]
            print("  CPU ops:", ops)

if __name__ == "__main__":
    main()