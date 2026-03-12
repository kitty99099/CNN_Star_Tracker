import argparse
import xir


def main():
    parser = argparse.ArgumentParser(description="Inspect subgraphs and ops in an .xmodel file")
    parser.add_argument(
        "--xmodel",
        type=str,
        required=True,
        help="Path to the .xmodel file, e.g. ./DPU_models/Model_0312.xmodel"
    )
    args = parser.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    root = g.get_root_subgraph()

    # 有些版本可直接用 children，有些建議用 toposort_child_subgraph()
    try:
        subgraphs = root.toposort_child_subgraph()
    except Exception:
        subgraphs = root.children

    for sg in subgraphs:
        print("SUBGRAPH:", sg.get_name())

        if sg.has_attr("device"):
            print("DEVICE:", sg.get_attr("device"))
        else:
            print("DEVICE: N/A")

        for op in sg.get_ops():
            print("   ", op.get_name(), op.get_type())

        print("-" * 60)


if __name__ == "__main__":
    main()