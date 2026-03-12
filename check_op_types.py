import argparse
import xir


def get_subgraphs(root):
    try:
        return root.toposort_child_subgraph()
    except Exception:
        try:
            return list(root.children)
        except Exception:
            return []


def get_ops(subgraph):
    try:
        return list(subgraph.get_ops())
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmodel", required=True, help="Path to .xmodel")
    args = parser.parse_args()

    graph = xir.Graph.deserialize(args.xmodel)
    root = graph.get_root_subgraph()
    subgraphs = get_subgraphs(root)

    for i, sg in enumerate(subgraphs):
        name = sg.get_name()
        device = sg.get_attr("device") if sg.has_attr("device") else "N/A"
        print(f"\n=== SUBGRAPH #{i} ===")
        print(f"name   : {name}")
        print(f"device : {device}")

        ops = get_ops(sg)
        for op in ops:
            op_name = op.get_name()
            op_type = op.get_type()
            print(f"{op_name}    {op_type}")


if __name__ == "__main__":
    main()