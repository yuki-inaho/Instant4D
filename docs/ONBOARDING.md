# LLMオンボーディングサマリー — Instant4D (Blackwell / uv 移植)

> 新任LLMエージェントがこのリポジトリで作業を始める際の初期資料です。記載内容はリポジトリの実状態（`justfile` / `tests/` / `docs/BLACKWELL_SETUP.md` / 各パッチ済みファイル）を検証して埋めています。一次情報は本書と [`docs/BLACKWELL_SETUP.md`](BLACKWELL_SETUP.md) です。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** Instant4D — 単眼動画からの **4D Gaussian Splatting**（動的シーンの時空間3D再構成）。本作業は、研究コードを **NVIDIA RTX PRO 4000 Blackwell (sm_120)** + **uv 仮想環境**で動作させる移植。
- **最終成果物:** uv環境で `reconstruct → prune → optimize` の3段パイプラインが Blackwell GPU 上で完走し、4DGS を学習して novel-view 動画を出力できる状態。加えて pytest 回帰スイート・`justfile`・セットアップ文書一式。
- **ビジネス背景・価値:** 最新 Blackwell 世代 GPU は CUDA 12.8 / PyTorch cu128 を要求し、既存の conda + cu126 手順では動かない。本移植は「最新GPUで研究パイプラインを再現する」ための実証・知見資産（sm_120 ビルド、依存衝突回避）。
- **現時点の進捗サマリ:** **全段 Blackwell 実機で完走済み**。panda シーンを実データ再構成（1,404,488 点）し 4DGS を 5000 反復学習（per-step PSNR は約 10 → 30 台まで上昇）。CUDA 拡張（simple_knn / pointops2 / diff_gaussian_rasterization(JIT) / droid_backends / lietorch / torch_scatter）すべて sm_120 でビルド・実行確認。pytest 9 件パス。

## 2. クリティカルな要求・制約
> 「壊してはいけない」ライン。多くは `docs/BLACKWELL_SETUP.md` に根拠あり。

- **GPU/ツールチェーン固定:** sm_120 には **CUDA toolkit ≥ 12.8（nvcc）** と **PyTorch 2.8.0+cu128** が必須。CUDA 11.8 や torch cu126/cu121/cu118 では sm_120 をコンパイル/実行できない。バージョンを下げてはならない。
- **ビルド時 env:** CUDA 拡張のビルド/JIT 時に `CUDA_HOME=/usr/local/cuda-12.8`、`TORCH_CUDA_ARCH_LIST=12.0`、`PATH` に nvcc を必ず設定（`justfile` が設定済み）。素の `sm_120` を使い `sm_120a` は使わない。
- **適用済みソースパッチを巻き戻さない**（torch>2.7 / sm_120 / Pillow / headless 対応。詳細は `docs/BLACKWELL_SETUP.md`）:
  - mega-sam/lietorch の `.type()`→`.scalar_type()`（lietorch_gpu.cu 19, lietorch_cpu.cpp 19, correlation_kernels.cu 2, altcorr_kernel.cu 1）
  - `SLAM/mega-sam/base/setup.py` 両 extension ブロックの `-gencode=...compute_120,code=sm_120`
  - `diff-gaussian-rasterization/setup.py` は `-O3`（`-g -G` 禁止）。rasterizer は初回 import で JIT ビルドされる
  - `scene/dataset_readers.py`: 画像は `np.uint8`（`np.byte` 不可、Pillow10+）、`image_path = cam_name`（パス二重結合の解消）
  - `scene/__init__.py`: 動画コーデックは `mp4v`（`avc1`/H264 は opencv-python-headless で書けない）
  - `utils/general_utils.py`: `from pointops2.pointops import …`
  - `SLAM/mega-sam/UniDepth/.../nystrom_attention.py`: xformers import ガード + SDPA フォールバック
  - `script/optimize.py`: 保存時は `scene.render_evaluate_sora()`（`render_evaluate_dycheck` は存在しない）
- **依存の罠:**
  - `torch_scatter` は **ソースビルド必須**（Ubuntu 20.04 = glibc 2.31、PyG prebuilt は glibc 2.32 要求）。`uv pip install --no-build-isolation --no-binary torch_scatter torch_scatter`
  - **`SLAM/mega-sam/UniDepth/requirements.txt` を入れない**（torch 2.2 / xformers 0.0.24 にダウングレードされる）。UniDepth には `wandb` + `h5py` のみ追加。
- **環境固定値:** GPU=RTX PRO 4000 Blackwell / driver 580.159.04(CUDA13.0対応) / Python 3.11(uv `.venv`) / OS Ubuntu 20.04(glibc 2.31)。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| セットアップ仕様（一次情報） | `docs/BLACKWELL_SETUP.md` | Blackwell 移植の全制約・全ソース改変・依存の罠・クイックスタート |
| 本書 | `docs/ONBOARDING.md` | 新任エージェント向けオンボーディング |
| ビルド/実行自動化 | `justfile` | setup / cuda-toolkit / build-ext / build-megasam / download-models / pipeline / test 等 |
| 再構成スクリプト | `script/reconstruct_uv.sh` | uv 版 mega-sam 5 段ジオメトリ回復（conda 不要、絶対パス、`I4D_FRAME_LIMIT` 対応） |
| 段2 / 段3 | `script/prune.py` / `script/optimize.py` | voxel filter → `filtered_cvd.npz` / 4DGS 学習 |
| 例 config | `configs/sora/panda.yaml` | sora(panda) 用ハイパラ（gaussian_dim=4, iterations=5000, rot_4d 等） |
| テスト資産 | `tests/`（+ `pytest.ini`） | `test_environment.py`(sm_120検出), `test_extensions.py`(各CUDAカーネル実行), `test_optimize_smoke.py`(合成シーン学習), `synthetic_scene.py`(生成器) |
| 既知課題・改変履歴 | `docs/BLACKWELL_SETUP.md`「Source modifications / gotchas」節 | 既知の互換問題と適用済み修正 |
| 上流issue根拠 | kaolin#865 / DPVO#100 / gaussian-splatting#1296,#1313 | scalar_type・sm_120 gencode・FLT_MAX の根拠 |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- 既存パイプラインの実行・検証（`just pipeline <scene>` / 個別段の再実行）、新しい例シーンや config の追加。
- 学習品質改善（ハイパラ調整、densification/loss 改良）、テスト追加・回帰拡充。
- `justfile` / `docs` / `tests` の保守、相対パス・env 変数化の踏襲。
- バグ修正（互換性・パス・形状不整合など、`docs/BLACKWELL_SETUP.md` の方針に沿うもの）。

### 任せないタスク（要相談 / 禁止）
- torch / CUDA / Python のバージョン変更や cu128 以外への切替（sm_120 要件を破壊する）。
- 「2. クリティカル制約」のソースパッチの巻き戻し。
- **サブモジュール（`SLAM/mega-sam` とその下位 `base`/`lietorch`、`UniDepth`）内の改変の upstream への push**（所有外リポジトリ。改変は `docs/BLACKWELL_SETUP.md` に記録して再適用する運用）。
- 大容量資産（チェックポイント `*.pth`、`filtered_cvd.npz`、`output/`、`.venv`、`logs/`）のコミット（`.gitignore` 済み）。
- デフォルトブランチ（`main`）への直接 push（ブランチを切って PR）。

## 5. インタラクション方針
- **回答スタイル:** 日本語、見出し＋箇条書き、結論先出し。コマンド/パスはコードブロックで明示。
- **回答手順:** 前提（環境・制約）→ 論点 → 提案/実行 → 検証結果（ログ・テスト）。
- **禁止事項・注意:** 未確定事項を断定しない。ビルド/実行は必ず実ログで裏取りし、PSNR 等の数値はログ実測のみ記載。破壊的・外向き操作（push 等）は明示許可があるときのみ。
- **秘匿情報の扱い:** SSH 鍵、HuggingFace トークン、社内パス等はコミット/外部送信しない。`userEmail`(yoshikawa@inaho.co) 等の個人情報は本書以外に拡散しない。

## 6. 試行タスク（オンボーディング演習）
1. `just gpu-info` を実行し、`device=NVIDIA RTX PRO 4000 Blackwell` / `capability (12,0)` / arch_list に `sm_120` が含まれることを確認する。
2. `just test-smoke`（高速回帰）を実行し全件 PASS を確認 → 続けて `just test`（slow 含む合成シーン学習）も PASS することを確認する。
3. `example/panda/filtered_cvd.npz` が存在する前提で、`I4D_NO_WEBSOCKET=1 I4D_CONFIG=<iterations少のconfig> just optimize panda` 相当を短縮実行し、`Training complete` と PSNR 上昇を確認する（フル再構成は `just pipeline panda`、`I4D_FRAME_LIMIT` で軽量化可）。

## 7. 運用ルール・変更管理
- **ドキュメント更新ルール:** ソースを改変したら必ず `docs/BLACKWELL_SETUP.md` の該当節（Source modifications / gotchas）に「何を・なぜ」を追記。本書の制約一覧も同期する。
- **TBDの扱い:** 未確定は `TBD:` と明記し、断定しない。検証できた事実のみ確定として記載。
- **レビュー/承認フロー:** `main` へ直接コミットせずフィーチャーブランチ → PR。サブモジュール内パッチは push 不可のため docs 記録で代替。
- **その他:** 中間生成物/モデルはコミットしない（`.gitignore` 準拠）。CUDA ビルドは常に `CUDA_HOME` / `TORCH_CUDA_ARCH_LIST=12.0` を設定。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:** `git@github.com:yuki-inaho/Instant4D.git`（origin, branch `main`）。`scene/`(データ/モデル), `gaussian_renderer/`(描画), `submodule/`(simple-knn/pointops2/fussed-ssim), `SLAM/mega-sam/`(再構成エンジン), `script/`(パイプライン), `configs/`, `example/`, `tests/`, `docs/`, `output/`(学習出力)。
- **代表的なコマンド:**
  ```bash
  just setup                 # uv venv + torch cu128 + 依存
  just cuda-toolkit          # CUDA 12.8 (nvcc, 要 sudo)
  just download-models       # Depth-Anything / RAFT (gdown)
  just build-ext             # simple_knn / pointops2 / rasterizer(JIT)
  just build-megasam         # droid_backends + lietorch (sm_120)
  just install-reconstruct-deps   # wandb,h5py,torch_scatter(ソース)
  just pipeline panda        # reconstruct -> prune -> optimize
  just test                  # pytest 回帰
  ```
- **依存ライブラリ:** torch==2.8.0+cu128 / torchvision==0.23.0+cu128 / torch_scatter(source) / numpy<2.0 / opencv-python-headless / open3d / point-cloud-utils / omegaconf / kornia / timm / tensorboard / wandb / h5py、ローカルビルド: simple-knn, pointops2, diff-gaussian-rasterization(JIT), droid_backends, lietorch。
- **3段パイプライン I/O:**
  - reconstruct (`reconstruct_uv.sh`): `example/<scene>/*.png` → UniDepth → Depth-Anything → DROID → RAFT → CVD → `SLAM/mega-sam/outputs_cvd/<scene>_sgd_cvd_hr.npz` + `reconstructions/<scene>/motion_prob.npy`
  - prune (`script.prune`): 上記 npz → `example/<scene>/filtered_cvd.npz` + `transforms_{train,test}.json`
  - optimize (`script.optimize`): `example/<scene>/`（画像+npz+transforms）→ `output/<scene>/`（描画・metrics・tensorboard）
- **連絡先/責任者:** yoshikawa@inaho.co

> ※注: 動画書き出しはコーデック `mp4v` 修正済み（検証で各 ~1MB 出力を確認）。`evaluation_metrics.json` は `--test_iterations`（既定 7000）到達時に PSNR/SSIM が記録される設計のため、5000 反復既定実行では空のままになる点に留意。
