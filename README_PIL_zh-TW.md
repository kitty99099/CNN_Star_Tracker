# Ultra96 Star Tracker PIL 版本

`main_CNN_Star_Tracker_PIL.py` 是由 `main_CNN_Star_Tracker.py` 複製後修改的版本。
原始主程式不變。PIL 入口接收電腦模擬器的灰階影像，在 Ultra96 執行既有
CNN → 質心 → geometric voting → SVD 流程，然後回傳結果。

## 使用前先確認模型

把新主程式與已設定好的 `pil_profile.json` 放到 Ultra96 原本的 CNN_Star_Tracker
資料夾。該資料夾仍需原本的 `geometric_voting.py`、`catalog_data/`，以及板上
可用的 PYNQ DPU overlay、相應 `.hwh` 和 `.xmodel`。不要重裝已運作的 PYNQ 環境。

在該資料夾使用原本能執行 DPU 的 Python。以下模型路徑請換成實際部署檔案：

```bash
python3 main_CNN_Star_Tracker_PIL.py \
  --xmodel /path/to/deployed_model.xmodel --bitfile dpu.bit --inspect-model
```

這會實際載入 overlay/model，列出 H/W、NCHW 或 NHWC 與輸出形狀後結束。
目前沿用原程式的 INT8、單輸入／單輸出、batch=1、雙輸出通道(segmentation、distance)
假設。輸出圖尺寸必須等於影像尺寸；不支援的模型會明確停止，不自動 resize。
`--inspect-model` 仍需要真實 Ultra96/DPU，不是電腦上的模型檢查器。

## 設定相機 profile

`pil_profile.json` 與 `pil_profile.example.json` 已同步為 640×480，與目前電腦端
PIL 範例的尺寸、focal_px、cx、cy、calibration_id 一致，可直接使用正式設定檔。

640×480 依使用者提供的 Ultra96 輸入規格設定；啟動時仍會核對實際載入的模型尺寸。焦距與光學中心
取自原主程式。PIL 使用無畸變針孔影像，因此新 PIL 路徑不套用實拍鏡頭的畸變校正。
改變影像尺寸時也必須重新確認內參，不可只調整陣列大小。

`centroid_offset_px: [du,dv]` 表示「CNN 質心座標加上此偏移後，得到 simulator-native
座標」。預設 [0,0] 只用於功能接入，不代表已校準。原質心使用 +0.5 像素中心，需用
孤立星點核對模擬器積分慣例，再決定偏移；不應直接猜測正負 0.5。相同轉換會用於
姿態計算與星點回傳，避免兩套座標。若更換 profile 定義，也要更換 calibration_id。

星表依原程式使用 `catalog_data/star_catalog_6.1.csv` 與 `star_pair_6.1.csv`。
兩邊 epoch/星表參考系尚需對齊後才能宣稱絕對精度；本版會回報星表 SHA256 供追溯。

## 啟動服務

```bash
python3 main_CNN_Star_Tracker_PIL.py \
  --xmodel /path/to/deployed_model.xmodel --bitfile dpu.bit \
  --profile pil_profile.json --port 5000
```

預設監聽 `0.0.0.0:5000`。若已知道 USB 網路介面 IP，可加 `--bind <該IP>`。
本程式使用 TCP；Ultra96 的 USB gadget 網路及 Windows/WSL 路由仍需先能互通。
程式不會自行修改作業系統 USB 設定。

DPU 與星表只在服務啟動時初始化一次，跨影像連線重用。按 Ctrl+C 停止。
每張結果附加寫入 `pil_board_results.jsonl`，可用 `--log` 指定路徑。
`--timeout` 單位為秒，預設 10；它限制網路收發，不會中斷卡住的硬體推論。

## 電腦端配合

修改 WSL 模擬器的 TCP JSON：

- `host`：Ultra96 USB 網路 IP（實際板端連線不能用 127.0.0.1）。
- `port`：5000 或板端所設定的 port。
- `width/height/focal_px/cx/cy/calibration_id`：與板端 profile 相同。
- `output`：尚不存在的資料夾。
- `timeout_ms`：需要涵蓋傳圖與板端完整處理時間。

然後從模擬器 repo 根目錄執行：

```bash
bin/verification_platform --pil pil/your_board_config.json
```

## 通訊與輸出

完全沿用目前 PC 協定 v1：4-byte big-endian JSON 長度、UTF-8 JSON；IMAGE 後接
width×height 個 row-major GRAY8 bytes。每條連線依序 HELLO → READY → IMAGE → RESULT。
最大 JSON 1 MiB、影像 16 MiB。先驗證尺寸、校正、識別碼、CRC32，再呼叫 CNN。
PC 不傳真值，Ultra96 也不讀取模擬真值。

結果包含 run_id/frame_id、影像 CRC32、processing_ms、centroids、identified HIP：

- `ok`：回傳 `cnn_inertial_to_body` 座標的 `rotation_row_major`，就是原程式
  `A_star @ A` 後的 3×3 矩陣。電腦端已支援此格式並轉換到共同座標。
- `euler_zyx_deg`：保留原本 Z/Y/X 三軸角度，供查看。
- `no_solution`：星點不足、太多、辨識不足或對應有歧義；沒有姿態欄位。
- `input_error`：CRC 錯誤或單張處理失敗；附 reason。CRC 錯誤回報實際接收 CRC，
  PC 可能因此歸類為 invalid_result，仍會保存原始錯誤回覆。

每張都重設暫存辨識資料，失敗不沿用上一張姿態。延用原始少於 30 個質心的 gate，
至少 3 顆唯一匹配星才進行 SVD。新的無畸變幾何計算不繪製視窗，避免阻塞 PIL。
保留原 normalization mean/std 與 INT8 量化，不在電腦端重複正規化。

在複製版中另外修正質心視窗的負索引繞回問題，並略過不足三個樣本／rank不足的
最小平方法；原主程式及其他演算法檔案沒有修改。原影片模式請繼續使用原主程式。

## 電腦端驗證

```bash
python3 -m unittest discover -s . -p test_pil_board.py -v
```

測試使用可注入的偵測／辨識替身與假 DPU runner，驗證已知姿態、清除舊結果、
NHWC/NCHW normalization、尺寸拒絕、分段 TCP、CRC、ID、校正、截斷及例外回覆。
在既有 WSL 環境也會呼叫真正的 C++ simulator 進行跨語言通訊測試；其他環境會跳過
該項測試。這些驗證不等於實際 DPU 推論、USB 連線或星追蹤精度已通過。
