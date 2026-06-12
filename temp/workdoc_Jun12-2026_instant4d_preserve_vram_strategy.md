# 作業計画書 兼 記録書: Instant4D 成果物保存と VRAM 適応型 4DGS 改善方針

---

**日付：** 2026年06月12日  
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/Instant4D` / `Instant4D`  
**作業者：** Codex  

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

* **目標1:** 一時 Docker コンテナ環境が消えても、Instant4D 作業の再開に必要な成果物をカテゴリ別に保存・アップロードできる状態にする。
* **目標2:** LichtFeld Studio の MRNF / IGS+ / PPISP 等の考え方を参考に、Instant4D へ組み込む価値が高い VRAM 適応型・品質維持型の改善方針を記録する。
* **目標3:** `/home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501`、`Instant4D_chkpnt10000.pth`、作業書・ログ・JSONL 等を `documents`、`colmap_data`、`other` に分けて保全する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** 明日以降、別の Docker container でストレージがリセットされても、COLMAP/VGGT データ、Instant4D checkpoint、作業経緯、改善方針を復元し、無駄な再計算を避けたい。
* **明示要求:**
  * さきほど整理した VRAM/品質改善案を作業書に記載する。
  * `/home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501` を保存対象に含める。
  * `10000.pth` はアップロード済みであっても、zstd 圧縮版を作る。
  * 作業書や JSONL ファイルは `https://github.com/yuki-inaho/agent-jsonl-compact` を使える場合は軽量化する。
  * 保存物は `documents`、`colmap_data`、`other` に分類する。
  * 明日作業する他社 Docker container で復元できるよう、アップロード・復元単位を明確にする。
* **暗黙制約:**
  * Python 実行は uv 環境を優先する。
  * GitHub repository 本体に GB 級 binary を直接 commit しない。
  * LichtFeld Studio は GPLv3 なので、Instant4D へはコードコピーではなく設計・論文アイデアを参考に独自実装する。
  * upload する大容量 artifact は GitHub Release asset 等、git history を汚さない手段を優先する。
  * 途中で学習・render・監視プロセスを残さない。
* **非ゴール:**
  * この作業書の時点では MRNF / IGS+ / PPISP の本実装は行わない。
  * この作業書の時点では 20000/30000 step の追加学習は行わない。
  * GitHub LFS の導入や外部クラウドストレージの構築は、明示指示がない限り行わない。
* **成功条件:**
  * `documents`、`colmap_data`、`other` の3分類で archive が作成される。
  * `Instant4D_chkpnt10000.pth.zst` が作成され、元ファイルとの sha256 を manifest に残す。
  * 作業書に VRAM 適応型改善方針、復元手順、アップロード候補が記載される。
  * `git status --short --branch` が clean または意図した workdoc 差分のみである。
  * 不要な `script.optimize`、`render_colmap_trajectory`、監視 shell が残っていない。
* **リスクと前提:**
  * GitHub Release asset は大容量制限があるため、2GB 超の archive は分割 upload が必要になる可能性がある。
  * `Instant4D_chkpnt10000.pth` は PyTorch checkpoint であり、zstd 圧縮しても大幅には小さくならない可能性がある。
  * COLMAP 画像は JPEG 主体のため、zstd 圧縮率は限定的である。
  * `agent-jsonl-compact` の CLI 仕様が不明な場合は README を確認し、使えない場合は JSONL 原本を zstd 圧縮する。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 保存対象を3分類へ整理する | 目標1/目標3 | `instant4d_preserve_2026-06-12/{documents,colmap_data,other}` | `find` と `du -sh` |
| SG-2 | VRAM/品質改善方針を文書化する | 目標2 | 本 workdoc | 本章とフェーズ1の記載 |
| SG-3 | `10000.pth` と COLMAP データを zstd archive 化する | 目標1/目標3 | `.zst` / `.tar.zst` | `zstd -t`、`tar -tf`、sha256 |
| SG-4 | JSONL と作業ログを軽量保存する | 目標1/目標3 | compacted JSONL または `.jsonl.zst`、worklog archive | ファイルサイズと manifest |
| SG-5 | upload/recover 手順を明文化する | 目標1 | manifest、release asset 候補 | `gh release view` または upload dry-run 相当 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | 3分類保存 | フェーズ2 手順4-6 | archive path、manifest |
| TR-2 | VRAM改善案の文書化 | フェーズ1 手順1-3 | 本作業書 |
| TR-3 | checkpoint zstd 圧縮 | フェーズ2 手順6 | `Instant4D_chkpnt10000.pth.zst`、sha256 |
| TR-4 | COLMAP データ保存 | フェーズ2 手順5 | `tva_nyx650_colmap_0501.tar.zst` |
| TR-5 | JSONL 軽量化 | フェーズ2 手順7 | compacted JSONL または `.zst` |
| TR-6 | upload 可能性 | フェーズ3 手順10-11 | GitHub Release asset または upload manifest |

---

## 2. 作業内容

### フェーズ 1: 調査・設計フェーズ (見積: 0.8h)

1. **Instant4D の OOM 原因分析:**
   * **タスク内容:** `output/tva_nyx650_0501_vggt_colmap_30000/run.log` と TensorBoard event から、OOM 時の step と `total_points` を確認する。
   * **目的:** 10000 checkpoint 以降の densify 増加が OOM の主因であることを再確認する。
   * **対応サブゴール/Trace ID:** SG-2 / TR-2
2. **LichtFeld Studio の参照対象整理:**
   * **タスク内容:** `src/training/strategies/improved_gs_plus.*`、`mrnf.*`、`components/ppisp.*`、`kernels/depth_loss.*`、`components/sparsity_optimizer.*` を確認する。
   * **目的:** Instant4D へコードコピーせずに設計だけ参考にする範囲を明確化する。
   * **対応サブゴール/Trace ID:** SG-2 / TR-2
3. **保存カテゴリ設計:**
   * **タスク内容:** `documents`、`colmap_data`、`other` の分類定義と含めるファイルを決める。
   * **目的:** 明日の復元時に迷わず必要物を取得できるようにする。
   * **対応サブゴール/Trace ID:** SG-1 / TR-1

### フェーズ 2: 保存物作成フェーズ (見積: 1.5h)

1. **documents archive 作成:**
   * **タスク内容:** workdoc、docs、worklog、設定 YAML、主要 JSON/CSV/log を `documents` に集約し、`documents_*.tar.zst` を作る。
   * **目的:** 作業経緯と再現コマンドを軽量に残す。
   * **対応サブゴール/Trace ID:** SG-1, SG-4 / TR-1, TR-5
2. **colmap_data archive 作成:**
   * **タスク内容:** `/home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501` を `tva_nyx650_colmap_0501.tar.zst` として保存する。
   * **目的:** 501枚データ、GlueMap/VGGT sparse model、解析 JSON を復元可能にする。
   * **対応サブゴール/Trace ID:** SG-3 / TR-4
3. **other archive 作成:**
   * **タスク内容:** `/home/kasm-user/Desktop/Instant4D_chkpnt10000.pth` を zstd 圧縮し、必要に応じて upload 用に分割する。
   * **目的:** 10000 step の学習成果を再利用可能にする。
   * **対応サブゴール/Trace ID:** SG-3 / TR-3
4. **JSONL 軽量化:**
   * **タスク内容:** `agent-jsonl-compact` が使用可能なら Codex session/history JSONL を compact し、不可なら `.jsonl.zst` として保存する。
   * **目的:** 会話・作業文脈を小さく残す。
   * **対応サブゴール/Trace ID:** SG-4 / TR-5

### フェーズ 3: 検証・アップロード準備フェーズ (見積: 0.7h)

1. **archive 検証:**
   * **タスク内容:** `zstd -t`、`tar -tf`、`sha256sum` を実行する。
   * **目的:** 明日復元できない archive をアップロードしない。
   * **対応サブゴール/Trace ID:** SG-3, SG-4 / TR-3, TR-4, TR-5
2. **GitHub upload 方針確認:**
   * **タスク内容:** asset size を見て、GitHub Release asset に入るか、分割が必要か判断する。
   * **目的:** git history に巨大 binary を載せずに保管する。
   * **対応サブゴール/Trace ID:** SG-5 / TR-6
3. **復元手順記録:**
   * **タスク内容:** `tar -I zstd -xf`、`zstd -d`、`uv sync`、data symlink 再作成コマンドを manifest に残す。
   * **目的:** 別 container で再開する作業者が迷わないようにする。
   * **対応サブゴール/Trace ID:** SG-5 / TR-6

### VRAM 適応型 Instant4D 改善方針

優先度順に、Instant4D へ独自実装する価値が高い案を以下に記録する。

1. **VRAM/点数上限つき adaptive densification**
   * OOM の直接原因は点数増加である。`chkpnt10000.pth` は約 7.83M 点、OOM 直前 TensorBoard は約 15.66M 点だった。
   * `densify_until_num_points` を静的条件にせず、現在点数、候補数、`torch.cuda.mem_get_info()`、直近 iteration time を使って split/clone 数を抑制する。
   * まず `max_gaussians`、`target_gaussians`、`vram_stop_fraction`、`densify_topk_fraction` を config 化する。
2. **IGS+ 風 budget schedule + top-k densify**
   * LichtFeld Studio の `ImprovedGSPlus` は `max_cap`、budget schedule、edge score、error score、free slot reuse を持つ。
   * Instant4D では、全候補 split/clone ではなく `score = normalized_xyz_grad + w_t * normalized_t_grad + w_edge * edge_score` の top-k のみにする。
   * コードコピーは GPLv3 のため避け、論文・設計を参考に Python/PyTorch で独自実装する。
3. **MRNF 風 prune-first / free-slot reuse**
   * LichtFeld Studio の `MRNF` は `free_mask`、`grow_and_split`、`compute_edge_scores`、`refine_weight_max`、`vis_count` を持つ。
   * Instant4D は `torch.cat` と prune により再確保が多く、巨大点数時に VRAM 断片化が起きやすい。
   * 第1段階では free slot reuse ではなく、より簡単な「prune-only window」と「top-k growth cap」から入る。
4. **Long-Axis Split**
   * Instant4D の split は random sampling 寄りで、4D の場合は `XYZT` に対してランダムに散る。
   * まず XYZ の最大 scale 軸に沿って split し、`t` は親の `t` を維持または `scale_t` に応じて小さく分割する。
5. **short-lifespan / low temporal contribution prune**
   * 2026 時点の 4DGS 系では短寿命 Gaussian pruning や active mask によるメモリ削減が有効な方向。
   * Instant4D では `motion`、`_t`、`_scaling_t`、per-view visibility から「時間的にほぼ寄与しない」点を落とせる可能性が高い。
6. **PPISP / 簡易測光補正**
   * PPISP は露出、ビネット、色補正、CRF を学習する品質改善寄りの手法であり、VRAM 削減の主役ではない。
   * まずは per-frame exposure と affine RGB correction の小規模版を実装し、植物・温室映像の露出揺れを geometry が吸収する問題を抑える。
7. **VGGT depth 補助**
   * depth を直接 loss に使うには COLMAP reader への接続が必要。
   * 先に VGGT depth から voxel downsample した初期点群を作り、COLMAP 点群に控えめに追加する方が低リスク。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。*

### フェーズ 1: 調査・設計フェーズ

### 手順 1: 現状プロセスと Git 状態の確認
- [ ] 🖐 **操作**: `pgrep -af "script.optimize|render_colmap_trajectory|monitor20000|sleep 120"` と `git status --short --branch` を実行する。
- [ ] 🔎 **確認**: 不要な学習・render・監視プロセスがなく、Git 差分が意図した workdoc または clean である。
- [ ] 🧪 **テスト**: プロセス確認のため自動テストは不要。出力を manifest または作業記録に残す。
- [ ] 🛠 **エラー時対処**: 不要プロセスが残る場合は PID とコマンドを確認し、成果物保存中でないことを確認してから `kill` する。

### 手順 2: 保存対象サイズの確認
- [ ] 🖐 **操作**: `du -sh /home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501 /home/kasm-user/Desktop/Instant4D_chkpnt10000.pth /home/kasm-user/Desktop/Instant4D` を実行する。
- [ ] 🔎 **確認**: COLMAP データ、checkpoint、repo 作業ディレクトリのサイズが記録されている。
- [ ] 🧪 **テスト**: サイズ確認のため自動テストは不要。manifest に数値を残す。
- [ ] 🛠 **エラー時対処**: パスがない場合は `find /home/kasm-user/Desktop -maxdepth 2 -name '*chkpnt10000*' -o -name 'TVA_NYX650*'` で探索する。

### 手順 3: 改善方針の確認
- [ ] 🖐 **操作**: 本書の「VRAM 適応型 Instant4D 改善方針」を読み、次回実装対象を `max_gaussians` / top-k densify から始めると明記する。
- [ ] 🔎 **確認**: OOM 原因、LichtFeld Studio 参照点、Instant4D への移植優先順位が記載されている。
- [ ] 🧪 **テスト**: 設計フェーズのため自動テストは不要。次回実装時に追加するテスト名を手順12以降で定義する。
- [ ] 🛠 **エラー時対処**: 方針が曖昧な場合は、OOM ログ、TensorBoard `total_points`、`scene/gaussian_model.py` の densify 実装を再確認する。

### フェーズ 2: 保存物作成フェーズ

### 手順 4: documents ディレクトリへ証跡を集約
- [ ] 🖐 **操作**: workdoc、`docs/uv_setup.md`、`configs/local/*.yaml`、worklog、主要 JSON/CSV/log を `instant4d_preserve_2026-06-12/documents/` にコピーする。
- [ ] 🔎 **確認**: `find instant4d_preserve_2026-06-12/documents -type f` で作業書、設定、ログが見える。
- [ ] 🧪 **テスト**: `tar -tf documents_*.tar.zst | head` で archive 内に本 workdoc が含まれることを確認する。
- [ ] 🛠 **エラー時対処**: コピー元が存在しない場合は、存在しないパスを manifest に記録して作業を継続する。

### 手順 5: colmap_data archive を作成
- [ ] 🖐 **操作**: `tar -I 'zstd -T0 -3' -cf instant4d_preserve_2026-06-12/colmap_data/tva_nyx650_colmap_0501.tar.zst -C /home/kasm-user/Desktop TVA_NYX650_2026_06_04_colmap_0501` を実行する。
- [ ] 🔎 **確認**: `.tar.zst` が作成され、`tar -I zstd -tf` で `images/frame_00001.jpg` と `gluemap_vggt_result/gluemap_aba` が見える。
- [ ] 🧪 **テスト**: `zstd -t tva_nyx650_colmap_0501.tar.zst` が成功する。
- [ ] 🛠 **エラー時対処**: 容量不足の場合は `df -h` を確認し、不要な `Instant4D/output/*_smoke*` などを削除候補としてユーザー確認する。

### 手順 6: checkpoint zstd を作成
- [ ] 🖐 **操作**: `zstd -T0 -3 -f /home/kasm-user/Desktop/Instant4D_chkpnt10000.pth -o instant4d_preserve_2026-06-12/other/Instant4D_chkpnt10000.pth.zst` を実行する。
- [ ] 🔎 **確認**: `.pth.zst` が作成され、元 `.pth` と圧縮後サイズが manifest に記録されている。
- [ ] 🧪 **テスト**: `zstd -t Instant4D_chkpnt10000.pth.zst` が成功する。
- [ ] 🛠 **エラー時対処**: GitHub Release asset 制限に近い場合は `split -b 1900M` で分割し、結合コマンドを manifest に記録する。

### 手順 7: JSONL を軽量化または zstd 圧縮
- [ ] 🖐 **操作**: `agent-jsonl-compact` の README/CLI を確認し、使える場合は対象 JSONL を compact、使えない場合は `.jsonl.zst` として保存する。
- [ ] 🔎 **確認**: `other/jsonl/` に compacted JSONL または zstd 圧縮 JSONL が存在する。
- [ ] 🧪 **テスト**: compacted JSONL は `wc -l`、`.zst` は `zstd -t` で確認する。
- [ ] 🛠 **エラー時対処**: compact tool が失敗する場合は原本 JSONL を変更せず `.zst` 保存へ切り替え、失敗理由を manifest に残す。

### 手順 8: documents archive を作成
- [ ] 🖐 **操作**: `tar -I 'zstd -T0 -6' -cf instant4d_preserve_2026-06-12/upload/documents_instant4d_jun12_2026.tar.zst -C instant4d_preserve_2026-06-12 documents` を実行する。
- [ ] 🔎 **確認**: archive が作成され、サイズが manifest に記録されている。
- [ ] 🧪 **テスト**: `tar -I zstd -tf documents_instant4d_jun12_2026.tar.zst | head` が成功する。
- [ ] 🛠 **エラー時対処**: archive に不要な巨大ファイルが混入した場合は `find documents -type f -size +50M` で特定して除外する。

### フェーズ 3: 検証・アップロード準備フェーズ

### 手順 9: sha256 manifest を作成
- [ ] 🖐 **操作**: `sha256sum` を archive と重要原本に対して実行し、`instant4d_preserve_2026-06-12/upload/SHA256SUMS.txt` を作成する。
- [ ] 🔎 **確認**: `SHA256SUMS.txt` に documents、colmap_data、checkpoint zstd、必要なら split file が含まれる。
- [ ] 🧪 **テスト**: `sha256sum -c SHA256SUMS.txt` を upload ディレクトリで実行できる。
- [ ] 🛠 **エラー時対処**: チェック失敗時は対象ファイルの再作成と再計算を行い、古い manifest を使い回さない。

### 手順 10: upload サイズを確認
- [ ] 🖐 **操作**: `ls -lh instant4d_preserve_2026-06-12/upload instant4d_preserve_2026-06-12/colmap_data instant4d_preserve_2026-06-12/other` を実行する。
- [ ] 🔎 **確認**: 2GB を超える asset が分割対象として識別されている。
- [ ] 🧪 **テスト**: 分割した場合は `cat part-* > restored.zst && cmp restored.zst original.zst` で復元可能性を確認する。
- [ ] 🛠 **エラー時対処**: GitHub upload が拒否された場合は asset を 1900MiB 以下に分割し、復元コマンドを README に追記する。

### 手順 11: GitHub Release asset として upload
- [ ] 🖐 **操作**: `gh release create instant4d-preserve-2026-06-12 ... --repo yuki-inaho/Instant4D --target rtx4000-ada-uv-setup --prerelease` または既存 release への `gh release upload` を実行する。
- [ ] 🔎 **確認**: `gh release view instant4d-preserve-2026-06-12 --repo yuki-inaho/Instant4D` で asset が表示される。
- [ ] 🧪 **テスト**: 別ディレクトリで小さい documents archive を download/extract して内容を確認する。
- [ ] 🛠 **エラー時対処**: Release 作成権限やサイズ制限で失敗した場合はエラー全文を manifest に残し、外部ストレージ案に切り替える。

### 手順 12: 次回実装テスト計画を作る
- [ ] 🖐 **操作**: `tests/` が存在しない場合は最小テスト方針を docs に記録し、`max_gaussians` と top-k densify の期待動作を定義する。
- [ ] 🔎 **確認**: OOM 再発防止の受け入れ条件が、点数上限・VRAM閾値・checkpoint保存の3軸で定義されている。
- [ ] 🧪 **テスト**: 次回実装時は `uv run python -m py_compile scene/gaussian_model.py script/optimize.py` と短い resume smoke を実行する。
- [ ] 🛠 **エラー時対処**: GPU テスト不可の場合は CPU で score selection/top-k の純粋関数テストを先に実装する。

---

## 4. 作業に使用するコマンド参考情報

### 基本確認

```bash
cd /home/kasm-user/Desktop/Instant4D
git status --short --branch
pgrep -af "script.optimize|render_colmap_trajectory|monitor20000|sleep 120" || true
du -sh /home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501 /home/kasm-user/Desktop/Instant4D_chkpnt10000.pth
```

### 保存先作成

```bash
mkdir -p /home/kasm-user/Desktop/instant4d_preserve_2026-06-12/documents
mkdir -p /home/kasm-user/Desktop/instant4d_preserve_2026-06-12/colmap_data
mkdir -p /home/kasm-user/Desktop/instant4d_preserve_2026-06-12/other
mkdir -p /home/kasm-user/Desktop/instant4d_preserve_2026-06-12/upload
```

### 圧縮

```bash
tar -I 'zstd -T0 -3' -cf /home/kasm-user/Desktop/instant4d_preserve_2026-06-12/colmap_data/tva_nyx650_colmap_0501.tar.zst \
  -C /home/kasm-user/Desktop TVA_NYX650_2026_06_04_colmap_0501

zstd -T0 -3 -f /home/kasm-user/Desktop/Instant4D_chkpnt10000.pth \
  -o /home/kasm-user/Desktop/instant4d_preserve_2026-06-12/other/Instant4D_chkpnt10000.pth.zst
```

### 復元

```bash
tar -I zstd -xf tva_nyx650_colmap_0501.tar.zst -C /home/kasm-user/Desktop
zstd -d Instant4D_chkpnt10000.pth.zst -o /home/kasm-user/Desktop/Instant4D_chkpnt10000.pth
```

### uv 検証

```bash
cd /home/kasm-user/Desktop/Instant4D
uv lock --check
uv run python -m py_compile scene/gaussian_model.py script/optimize.py script/render_colmap_trajectory.py
```

---

## 5. 作業記録

作業記録では、以下を必ず残すこと。

* 実行したコマンドと実行ディレクトリ。
* 作成された archive のパス、サイズ、sha256。
* upload した場合は release URL または asset URL。
* 失敗したコマンドは省略せず、エラー要約と次の対処を記録する。
* 暗黙 fallback は避け、切り替えた場合は理由を明記する。

### 2026-06-12 初期記録

* 作業ブランチ: `rtx4000-ada-uv-setup`
* 最新 commit: `8f4b0ae Support 30000-step TVA resume`
* 10000 checkpoint: `/home/kasm-user/Desktop/Instant4D_chkpnt10000.pth`
* 元 checkpoint: `/home/kasm-user/Desktop/Instant4D/output/tva_nyx650_0501_vggt_colmap_30000/chkpnt10000.pth`
* COLMAP/VGGT データ: `/home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501`
* 3000 checkpoint 元軌跡 mp4: `/home/kasm-user/Desktop/Instant4D/output/tva_nyx650_0501_vggt_colmap_fullish/chkpnt3000_trajectory/reconstruction.mp4`
* 3000 checkpoint 全501枚平均 PSNR: `17.837925711077844 dB`
* 10000 checkpoint は作成済みだが、10000 checkpoint の元軌跡 mp4 生成は中断済み。
