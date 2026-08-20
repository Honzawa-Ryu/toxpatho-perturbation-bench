# Task A 実装計画: Perturbation Invariance（摂動不変性・自己検索）

- 作成日: 2026-08-19
- 前提ドキュメント: [`benchmark_task_design.md`](./benchmark_task_design.md)
- ステータス: ドラフト（デフォルト値は要レビュー、config.ymlで変更可能な設計にする）

## 0. 現状データの確認

`data/raw_wsi/` に **Aperio (.svs) 形式のWSI 1,000枚（計644GB）** を確認。
`read_me.txt` の記載: 「多目的最適化によって取得してきたTGGATEデータ全体の傾向を反映している縮小データセット」。
所見・化合物・用量等のメタデータは今回のデータには同梱されていない（ファイル名は数値ID `{id}.svs` のみ）。

→ Task Aは自己参照GT（同一パッチ由来かどうか）のみで完結するため、**このメタデータ欠如は問題にならない**。
Task B着手時に別途 TG-GATEs 側の化合物/用量/時点メタデータとの突合が必要になる点は
`benchmark_task_design.md` のOpen Questionsに引き続き残す。

環境チェック結果:

| 項目 | 状態 |
|---|---|
| GPU | RTX A6000 ×4（devノードで確認。実行はSlurm経由） |
| openslide / tiffslide | `pyproject.toml`に追加済み（実インストールは`make uv_sync`待ち） |
| TRIDENT | `pyproject.toml`に`trident[patch-encoders]`として追加済み（git依存、下記参照） |
| 病理向け依存（timm等） | `pyproject.toml`に追加済み |
| HF_TOKEN | **`.env`に`export HF_TOKEN=...`の形で設定済み**（python-dotenvは`export `プレフィックス付き行も解釈可能） |

**MPP/倍率**: 20倍（≈0.5 MPP）で確認済み。TRIDENT公式CLIも`--mag 20`を標準的な指定として使っており、
初期案の`mpp_target: 0.5`と整合しているため、この値で確定とする。

## 1. Phase 0: 環境セットアップ（着手前の前提作業）

### 1-1. TRIDENTのインストール（`pyproject.toml`反映済み）

TRIDENT ([mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT)) はPyPI配布ではなく、
GitHubのソースをeditable installする形で使う（`[tool.poetry]`ベースのパッケージ、
`build-backend = poetry.core.masonry.api`）。**実際のリポジトリの`pyproject.toml`を直接確認**したところ

```
python = ">=3.10,<3.13"  # Support Python 3.10, 3.11, and 3.12.
```

と明記されており、**本リポジトリの`requires-python==3.12.*`と競合しない**ことを確認済み
（当初懸念していた「3.10/3.11推奨」情報は誤りだったため訂正）。よってPython版数起因の隔離venv等の
代替案は不要と判断。

`pyproject.toml`には以下を追加済み:

```toml
dependencies = [..., "tiffslide", "timm>=0.9.16,<2", "einops-exts", "trident[patch-encoders]"]

[tool.uv.sources]
trident = { git = "https://github.com/mahmoodlab/TRIDENT.git" }
```

`patch-encoders` extraには`fairscale`, `timm_ctp`（`ctranspath`用）, `conch`（git依存、CONCH本体）,
`musk`（git依存）が含まれる。**注意**: `gigapath`のpatch encoderはTRIDENT側の`slide-encoders` extra
（`environs`, `sacremoses`, `gigapath`, `madeleine`）に属しており、`patch-encoders`だけでは入らない。
gigapathを使う段になったら`trident[patch-encoders,slide-encoders]`に変更する必要がある
（どのみちgigapathはgatedモデルなので、Open/Unrestrictedのみのパイロットでは未使用）。

実際のvenv構築（`uv sync`）はこのノードでは実行できない（`uv`コマンドはApptainerコンテナ内にしか
無く、`make uv_sync p=<partition>`経由のSlurmジョブとしてのみ実行される構成）。
**このコマンドの実行はユーザー側で行ってください**（`partition`の指定が必要なため）。

### 1-2. 利用可能なpatch encoderの洗い出し

TRIDENTの`encoder_factory`（`trident/patch_encoder_models/load.py`）は33種類のpatch encoderに対応。
HF_TOKEN未設定の現状では **Open/Unrestricted** の列のみが初期パイロットで使用可能。

| 区分 | モデル例（`encoder_factory`に渡す名前） |
|---|---|
| Open/Unrestricted（HF_TOKEN不要） | `resnet50`（ImageNet baseline）, `ctranspath`, `keep`, `openmidnight`, `lunit-vits8`, `genbio-pathfm`, `gemma4-e4b`, `gemma4-26b` |
| Gated（要HFライセンス同意 + `HF_TOKEN`） | `uni_v1`/`uni_v2`, `conch_v1`/`conch_v15`, `phikon`/`phikon_v2`, `virchow`/`virchow2`/`virchow2-cls`, `gigapath`/`gigapath-flash`, `hoptimus0`/`hoptimus1`/`h0-mini`, `kaiko-*`, `hibou_l`, `musk`, `gpfm`, `midnight12k`, `phaet`, `mascaret` |

→ `HF_TOKEN`は`.env`に設定済みだが、**gatedモデルはトークンだけでなくモデルごとの個別ライセンス同意
（HuggingFace上で各リポジトリにアクセス申請・承認）が別途必要**。トークンの有無だけでは
アクセス可否は確定しないため、Exp 0003の初期GRID_VALUESは引き続き**Open/Unrestrictedの列のみ**で
構成し、疎通確認（1-3節）でgatedモデルが実際にロードできるかを個別に確認してから
Gated列を追加する2段階構成とする（詳細は5節）。`resnet50`は病理特化ではないため
「非特化モデルとの比較baseline」の位置づけとして残す。

### 1-3. 疎通確認（完了・2026-08-20実施）

`env.sif`（SIF_PATH）を用意し`make uv_sync p=large-andre01`実行（job 8982, COMPLETED）。
その上でapptainer経由の疎通確認スクリプトを実行し、以下すべてOKを確認:

1. `tiffslide`でWSI読み込み: `data/raw_wsi/10306.svs` → dimensions (53783, 34670),
   level_count=3, **mpp=0.4916, objective-power=20** → Phase 0で確定した`mpp_target: 0.5`/
   20倍設定と実データが一致することを確認
2. `encoder_factory("resnet50")` → OK（8,543,296 params）
3. `encoder_factory("uni_v1")`（gated） → **OK（303,350,784 params）**。`HF_TOKEN`は
   MahmoodLab系gatedモデルに対して有効なアクセス権を持つことを確認

→ Phase 0は完了。**gatedモデルも動作確認が取れたため、Exp 0003のGRID_VALUESはOpen/Unrestrictedに
限定せず候補モデル全体で開始してよい**（Array jobなのでモデルごとのアクセス失敗は該当ジョブのみに
閉じる。他モデルには影響しない。PAIGE/Bioptimus等MahmoodLab以外が配布元のgatedモデル
（`virchow*`, `hoptimus*`, `gigapath*`等）は組織が別なのでアクセス可否は個別に要確認）。
0001の実装に進む。

## 2. 全体構成（実験分割）

テンプレートの「1実験=1フェーズ」原則に従い、Task Aを4つの実験に分割する。後続実験は前段の
`outputs/{exp_name}/{variant_key}/` を読みに行く形で依存させる（config.ymlに上流の実験名を明記）。

```
0001_sample_source_patches      WSI → 代表パッチ抽出（原パッチセット）        [CPU, single]
0002_generate_perturbations     原パッチ → 摂動版パッチ生成                    [CPU, single]
0003_extract_patch_embeddings   原+摂動パッチ → 複数モデルで埋め込み抽出      [GPU, GRID array over model]
0004_eval_retrieval_invariance  埋め込み → Top-K/mAP/劣化カーブ評価           [CPU, single]
```

`lib/perturbations.py` と `lib/patch_io.py`（仮）に共通ロジックを切り出し、0002以降・将来のTask B/Cからも
再利用できるようにする（TEMPLATE_CONCEPTの「重複したら`lib/`へconsolidate」原則）。

---

## 3. Exp 0001: `sample_source_patches`

### 目的
1,000枚のWSIから、組織領域内のパッチを一定基準でサンプリングし、Task Aの「原パッチセット」を作る。

### パラメータ（`config.yml`、初期デフォルト案）

```yaml
seed: 42
patch_size_px: 256        # 256x256固定。多くのpatch encoderの入力に近い解像度
mpp_target: 0.5            # 約20倍相当。TG-GATEsのAperioスキャンは概ね0.25-0.5um/px帯
n_wsi: 100                  # まず100枚でパイプライン検証（1,000枚全部は後段でスケール）
patches_per_wsi: 20         # WSIあたり20枚 → 初期セット 2,000パッチ
tissue_threshold: 0.8       # Otsu二値化後の組織占有率がこの閾値以上のタイルのみ採用
```

最初から1,000枚×多数パッチで走らせず、**まず100 WSI × 20枚 = 2,000パッチでパイプライン全体
（0001→0004）を一周させてから、`n_wsi`を上げてスケールする**方針を推奨（大容量WSI読み込みの
I/Oコストが高いため、早期にバグを潰す狙い）。

### 処理内容
1. `tiffslide`（or openslide）でWSIを開き、`mpp_target`に最も近いレベルを選択
2. 低倍率サムネイルでOtsu二値化 → 組織マスクを作成（背景ガラス領域を除外）
3. マスク内から`patches_per_wsi`枚をランダムサンプリング（座標の重複を避けるnon-overlappingグリッド上で抽選）
4. `patch_size_px`でクロップし、指定`mpp_target`にリサイズ（ネイティブ解像度とズレる場合のみ）
5. WSIをまたいで`patch_id = f"{wsi_id}_{x}_{y}"`を採番

### 出力（`outputs/0001_.../default/`）

```
patches/{patch_id}.png
manifest.parquet   # columns: patch_id, wsi_id, x, y, level, mpp, patch_size_px
```

### CLI/variant_key
Grid化しない単発実行（`variant_key = f"{patch_size_px}px_mpp{mpp_target}"` として、
将来別のpatch_size/mpp設定を並行して試せるようにしておく）。

---

## 4. Exp 0002: `generate_perturbations`

### 目的
0001の原パッチそれぞれに対し、複数系統×複数強度の摂動を適用した版を生成する。

### 摂動レジストリ（`lib/perturbations.py`、案）

```python
PERTURBATION_LEVELS: dict[str, dict[int, float | tuple]] = {
    "rotation":          {1: 15,  2: 45,  3: 90},        # degrees
    "color_jitter":      {1: 0.1, 2: 0.3, 3: 0.5},        # brightness/contrast/saturation factor
    "stain_jitter":      {1: 0.1, 2: 0.2, 3: 0.35},       # HED空間でのjitter alpha
    "jpeg_compression":  {1: 90,  2: 50,  3: 20},         # quality（低いほど強劣化）
    "gaussian_blur":     {1: 0.5, 2: 1.5, 3: 3.0},        # sigma(px)
    "gaussian_noise":    {1: 5,   2: 15,  3: 30},         # std（0-255スケール）
    "downsample_mpp":    {1: 0.75,2: 0.5, 3: 0.25},       # 縮小率（縮小後に元サイズへ再拡大）
    "occlusion":         {1: 0.1, 2: 0.25,3: 0.4},        # マスクする面積比
}
```

各関数シグネチャは `apply_perturbation(image: PIL.Image, kind: str, level: int, seed: int) -> PIL.Image`
に統一し、`(patch_id, kind, level)` から決定論的にseedを導出する（`hash((patch_id, kind, level)) % 2**32`等）。
これにより再実行しても同じ摂動画像が再現される。

### 出力（`outputs/0002_.../default/`）

```
perturbed/{kind}/{level}/{patch_id}.png   # jpeg_compressionのみ実際に.jpgで保存し圧縮を反映
manifest.parquet   # columns: patch_id(=parent), kind, level, seed, output_path
```

### 依存
`config.yml`に `source_exp: "0001_YYYYMMDD_sample_source_patches"` を明記し、
`experiment.py`内で `outputs/{source_exp}/default/manifest.parquet` を読む。

---

## 5. Exp 0003: `extract_patch_embeddings`（GRID array over model）

### 目的
0001の原パッチ + 0002の摂動パッチ全件を、TRIDENT経由で複数のpatch encoderに通し埋め込みを得る。

**重要**: TRIDENTの公式CLI（`run_batch_of_slides.py`）はWSI単位のsegmentation→patching→encodingを
想定しており、単体パッチ画像だけを渡すモードは公式には用意されていない。ただし
`trident/patch_encoder_models/load.py`の`encoder_factory(model_name)`はWSI処理から独立しており、
素の`torch.nn.Module`＋評価用transformを返す設計になっている。よって
**TRIDENTのWSI処理・CLI部分は使わず、`encoder_factory`だけを直接importして自前のバッチ推論ループに
組み込む**構成にする（`lib/embedding_extraction.py`（仮）にラップする）。

### `run_slurm.sh` のGRID設計（Phase 1: Open/Unrestrictedモデルのみで開始）

```bash
BASE_COMMAND="python ${PYTHON_PATH}"
GRID_ARGS=("--model")
GRID_VALUES=(
    "resnet50 ctranspath keep openmidnight lunit-vits8"   # HF_TOKEN不要な列。gated列はTOKEN設定後に追加
)
RUN_MODE="array"
```

1ジョブ=1モデルの全パッチ推論（原則「1ジョブ=1原子単位」）。GPU 1枚を割当。

### 処理内容
1. `--model`で指定されたencoderをTRIDENTのfactory経由でロード、対応する前処理transformを取得
2. 0001 + 0002 のmanifestを結合し、全画像パスのリストを作成
3. バッチ推論 → `patch_id`ごとに埋め込みベクトルを取得
4. モデルごとの出力次元差はそのまま保持（後段では同一モデル内比較のみ行うため次元を揃える必要はない）

### 出力（`outputs/0003_.../{model_short}/`）

```
embeddings.parquet
  # columns: patch_id, source_type(original/perturbed), parent_patch_id,
  #          kind(nullable), level(nullable), embedding(list[float])
```

### 実行前の確認事項
- Phase 0で確定した「アクセス可能モデル一覧」を`GRID_VALUES`に反映
- 2,000（原本）+ 2,000×8種×3段階（摂動、最大48,000）程度の推論量になるため、
  最初は`n_wsi`を絞った小規模セットで1モデルだけ動かし、所要時間・出力サイズを見積もってから
  全モデル×フルセットに展開する

---

## 6. Exp 0004: `eval_retrieval_invariance`

### 目的
モデル×摂動タイプ×強度ごとに、摂動後パッチが元パッチを最近傍として引けるかを評価する。

### 評価プロトコル
- Gallery = 0001由来の原パッチ埋め込み全件（例: 2,000件）
- Query = 各(kind, level)ごとの摂動パッチ埋め込み
- 各queryについて、galleryとのcosine類似度でランキング → 正解（`parent_patch_id`と一致する原パッチ）の順位を取得

### 指標
| 指標 | 定義 |
|---|---|
| Top-K accuracy (K=1,5,10) | 正解がTop-K以内に入った割合 |
| MRR | 正解順位の逆数の平均 |
| Recall@K | Top-Kと同義（候補が1つのみのため） |
| Cosine similarity degradation | `level`ごとの`cos(orig_emb, perturbed_emb)`平均値のカーブ |
| Rank degradation | `level`ごとの正解順位（絶対値/相対値）の平均カーブ |

### 出力（`outputs/0004_.../default/`）

```
metrics_by_model_perturbation.parquet
  # columns: model, kind, level, top1_acc, top5_acc, top10_acc, mrr, mean_cos_sim, mean_rank
figures/heatmap_top1_by_model_kind.png
figures/degradation_curves_{kind}.png
```

集計はモデル一覧（0003のGRID_VALUES）をループして`outputs/0003_.../{model}/embeddings.parquet`
を読み込む単発実行（重い推論を含まないためCPUジョブで十分）。

---

## 7. 実行順序と検証ステップ

```
1. Phase 0完了（依存追加・アクセス可能モデル確認・小規模スモークテスト）
2. make create_exp name=sample_source_patches       → 0001実装 → 小規模(n_wsi=5)でローカル動作確認 → preflight → runx
3. make create_exp name=generate_perturbations       → 0002実装 → 同上
4. make create_exp name=extract_patch_embeddings     → 0003実装 → モデル1つ・パッチ少数でdry run → preflight → runx（array）
5. make create_exp name=eval_retrieval_invariance    → 0004実装 → 結果確認
6. 問題なければ n_wsi / patches_per_wsi をスケールし再実行（0001からやり直し、依存する0002-0004も再実行）
```

各ステップとも、まず小規模（数WSI・数十パッチ）でローカル(`run_local.sh`経由の`runx`)動作確認してから
Slurmでのフルスケール投入に進む。

## 8. 未決事項（実装着手前に確定したいもの）

- [x] `mpp_target` = 0.5（20倍）で確定
- [x] Phase 0初期のモデル一覧 = Open/Unrestricted列（`resnet50, ctranspath, keep, openmidnight,
      lunit-vits8`等）で開始。gated列は疎通確認後に追加
- [x] **TRIDENTのPython対応バージョン**: 公式`pyproject.toml`で`python = ">=3.10,<3.13"`と
      明記されており、本リポジトリの`requires-python==3.12.*`と競合しないことを確認。
      `pyproject.toml`への依存追加（`tiffslide`, `timm`, `einops-exts`, `trident[patch-encoders]`）も完了
- [x] HF_TOKENは`.env`に設定済み。gatedモデル（`uni_v1`で確認）も実際にロードできることを確認済み
- [x] **Phase 0完了**（2026-08-20）: `env.sif`をSIF_PATHに設定し`make uv_sync p=large-andre01`成功
      （job 8982）。疎通確認（1-3節）も全項目OK。`pyproject.toml`/`uv.lock`は変更済みだが
      **未コミット**（コミットタイミングは要相談）
- [ ] `n_wsi` / `patches_per_wsi` の最終スケール（2,000パッチはパイロット規模。本番はいくつにするか）
- [ ] 摂動の強度パラメータ具体値（上表は初期案。実際のH&E画像で視覚的に妥当か確認が必要）
- [ ] `downsample_mpp`摂動を「縮小して同サイズに再拡大」とするか「実際に小さいテンソルのままモデルに渡す」とするか
      （後者の方が実運用のMPP不整合に近いが、モデルによって入力サイズ固定の制約がある点に注意）
