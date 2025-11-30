#!/usr/bin/env python3
"""
inspect_xmodel.py

用途：
  檢查 XIR .xmodel 內的 DPU 子圖數量與名稱，協助判斷
  為什麼 pynq_dpu 的 load_model() 會觸發 assert len(subgraphs) == 1。

使用方式：
  python inspect_xmodel.py ./CNN_Star_Tracker_Model_V1.xmodel
  # 若你期望剛好只有 1 個 DPU 子圖，可加 --expect-one 讓腳本幫你檢查
  python inspect_xmodel.py ./model.xmodel --expect-one
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Inspect DPU subgraphs in an XIR .xmodel")
    parser.add_argument("xmodel", help="Path to .xmodel file")
    parser.add_argument("--expect-one", action="store_true",
                        help="Return non-zero exit code if DPU subgraphs != 1")
    args = parser.parse_args()

    xmodel_path = Path(args.xmodel)
    if not xmodel_path.is_file():
        print(f"[ERROR] File not found: {xmodel_path}", file=sys.stderr)
        sys.exit(2)

    try:
        import xir
    except Exception as e:
        print("[ERROR] 無法 import 'xir'。請確認已安裝 Vitis AI runtime (含 XIR) 或相容套件。", file=sys.stderr)
        print(f"       原始錯誤：{e}", file=sys.stderr)
        sys.exit(3)

    try:
        graph = xir.Graph.deserialize(str(xmodel_path))
    except Exception as e:
        print(f"[ERROR] 讀取 xmodel 失敗：{xmodel_path}", file=sys.stderr)
        print(f"       原始錯誤：{e}", file=sys.stderr)
        sys.exit(4)

    def is_dpu(subg):
        return subg.has_attr("device") and str(subg.get_attr("device")).upper().startswith("DPU")

    root = graph.get_root_subgraph()
    subs = [sg for sg in root.toposort_child_subgraph() if is_dpu(sg)]

    print(f"[INFO] XMODEL: {xmodel_path}")
    print(f"[INFO] DPU subgraphs: {len(subs)}")

    for i, sg in enumerate(subs):
        name = sg.get_name()
        device = sg.get_attr("device") if sg.has_attr("device") else "N/A"
        # 嘗試列出此子圖內的節點數量，幫助判斷是否被切段
        try:
            nodes = list(sg.toposort())
            n_ops = len(nodes)
        except Exception:
            n_ops = "?"
        print(f"  - #{i}: name='{name}', device='{device}', ops={n_ops}")

    # 額外提示：如果是 0 或 >1，給出建議
    if len(subs) == 0:
        print("\n[HINT] 沒有任何 DPU 子圖：")
        print("       1) 檢查編譯用的 arch.json 是否與板上 overlay 相符（例如 DPUCZDX8G）。")
        print("       2) 確認 Vitis AI 版本與板上 runtime 相同（例如 Vitis AI 2.5 對應 DPU-PYNQ 2.5.x）。")
        print("       3) 確保模型主幹已量化且可映射到 DPU。")
    elif len(subs) > 1:
        print("\n[HINT] 有多個 DPU 子圖：")
        print("       1) 常見因為前/後處理算子不支援導致圖被切段；請將這些算子移到 CPU。")
        print("       2) 調整網路或重編，使推論主幹合併為單一 DPU 子圖。")

    if args.expect_one and len(subs) != 1:
        sys.exit(5)

if __name__ == "__main__":
    main()