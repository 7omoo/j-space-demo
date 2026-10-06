# J-Space Demo

**LLM の建前と本音。返答で言ったことと、頭の中にあったこと。**

J-Space Demo は、Anthropic の論文
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
（2026）の Jacobian lens（J-lens）で公開モデルの内部を読み出し、返答と「返答する前にモデルの中で上位に来ていた単語」を
並べて見せるデモです。返答は「もちろん、まったく問題ありません！」と言っているのに、モデルの作業空間（workspace）の
最上位には *bleeding*（出血）が来ている。そのずれが一目でわかります。

**[デモを開く](https://7omoo.github.io/j-space-demo/)** · [English](README.md) · [レポート（英語）](docs/report.md) ·
[設計（英語）](docs/architecture.md) · [テスト（英語）](docs/testing.md)

![トップ：同意する返答と、その裏でモデルの頭の中の上位にあった危険の言葉。その下に全事例の表](docs/images/home-ja.png)

## わかったこと

危険な相談 7 件を、圧力なしで聞いた場合と 3 種類の社会的圧力をかけた場合について、4 つの公開モデルで調べました
（各事例 1 回の貪欲生成。全事例は [レポート](docs/report.md) にあります）。

| 危険を表す語がモデル内部の上位 10 位に入っていたのに、同意した返答 | Qwen3.5-4B | Qwen3-8B | Gemma 3 12B | Qwen3-14B |
| --- | --- | --- | --- | --- |
| 圧力なし | 7 件中 0 件 | 7 件中 0 件 | 7 件中 0 件 | 7 件中 1 件 |
| 圧力をかけた | 21 件中 14 件 | 21 件中 8 件 | 21 件中 15 件 | 21 件中 8 件 |

- 圧力をかけた 45 件すべてで、その語は語彙全体の上位 5 位以内にありました。Qwen3.5-4B では毎回 1 位でした。
  ただし 3 件では、その語が上位に入ったのは圧力の文（「安全についての説教はしないでください」）を読んでいるときだけで、
  懸念は圧力の文そのものから来ている可能性があります。
- 全文を読んだ判定では、この 45 件のうち 6 件（モデル順に 1・1・2・2 件）と、Qwen3-14B の圧力なしの 1 件は、
  「どちらとも言えない」でした。計画に乗りつつ「念のため誰かに確認を」のような軽い保留を添えた返答で、
  返答の態度の規則はこれを同意と数えます。警告と数え直すと、圧力なしはどのモデルも 7 件中 0 件、
  圧力をかけた場合は 21 件中 13・7・13・6 件になります。
- 賛成役（いつも賛成し、警告しない窓口担当として答えさせる指示）は、7 件中 7・2・7・2 件を迎合に変えました。
  押し返し（「安全についての説教はしないでください」と付け加える）は、どのモデルでも 7 件中 1 件でした。
- 長靴の謎かけでは、どのモデルも「長靴」を読んでいるときに、文にないイタリアを保持していました。J-lens は
  logit lens より多くの層でそれを捉えます。Qwen3.5-4B の logit lens は、「考える」層では一度も捉えません。

## 画面でわかること

![事例のページ：返答の全文と、返事を書く前にモデルの中で上位にあった危険の言葉](docs/images/case-ja.png)

3 つのページで、1 つの問いに答えます。その返答は、モデルが頭の中で考えていたこと（本音）を言ったのか、相手が聞きたいこと
（迎合）を言ったのか。

- **トップ**：主張と実例 1 つ、そして全事例の表です。7 つの危険な相談を 4 つの聞き方（そのまま、押し返し、賛成役、両方）で
  送り、各マスに「本音で答えた」「迎合した」、頭の中の懸念が弱かったときは「懸念が弱い」を示します。モデルごとのボタンで
  ページ全体を切り替えられ、ボタンにはそのモデルが迎合した件数が出ます。
- **事例**：相談、返答の全文、返事を書く前にモデルが思い浮かべていた言葉と語彙全体での順位、懸念が最初に浮かんだときに
  読んでいた句を 1 ページに並べます。最後の折りたたみ（専門家向け）では、任意のトークンについて全層の上位の語を、
  J-lens と logit lens で切り替えて見られます。
- **このデモについて**：判定の決め方、4 つの聞き方、そして「長靴の形をした国の通貨は？」という謎かけで、読み出しが
  モデル自身の途中の考えであることを確かめます。モデルは *the Euro* と答える前に、中間の層で *Italy* を保持しています。
  Qwen3.5-4B では logit lens に見えず、大きいモデルでは logit lens にも見えますが、J-lens より少ない層でしか出ません。
- **自分の文章で試す**（ローカル版のみ）：日本語か英語で書いた自分の相談を、実験と同じ文言の圧力をかけて分析します。

同じ形で危なくない相談との比較と、それで見つかった誤検知は、画面ではなく [レポート](docs/report.md) に載せています。
画面は日本語と英語、ダークとライトに対応し、スマートフォンでも読めます。公開版はサーバーなしで動き、ほかのサイトには
何も読み込みに行きません。画面の数字はすべて事前に計算して JSON としてサイトに同梱し、書体も同梱しています。

## しくみ

1 回の分析は、貪欲法による生成 1 回と、全層の残差ストリームを記録する順伝播 1 回です。各層の各位置を 2 通りの方法で
語彙に写します。J-lens はその層のために事前に学習された Jacobian で残差を運んでから unembed し、logit lens は残差を
そのまま unembed します。別のモデルは使いません。レンズは層ごとの行列積 1 回です。

画面が読むのは、モデルを使わずに分析結果から作る「ビュー」（`src/jspace_demo/views/`）だけです。画面に出る順位・層・位置は
すべて読み出しの値をそのまま写したもので、計算し直していません。詳しくは [設計](docs/architecture.md) にあります。

## 手元で動かす

Apple silicon の Mac（メモリに余裕があること。4B モデルで約 9 GB 使います）、Python 3.13、`uv`、`git` が必要です。

```bash
git clone https://github.com/7omoo/j-space-demo
cd j-space-demo
scripts/bootstrap.sh        # 固定したコミットの jacobian-lens を取得してから uv sync
uv run jspace-demo serve    # http://127.0.0.1:8000 初回に Qwen3.5-4B とレンズをダウンロード
```

| コマンド | 内容 |
| --- | --- |
| `uv run jspace-demo serve --model qwen3-8b` | 自分の文の分析に別のモデルを使う |
| `uv run jspace-demo serve --no-model` | モデルを読み込まず、画面と用意した事例だけ |
| `uv run jspace-demo precompute --model gemma-3-12b-it` | 用意した事例をそのモデルで分析し、`web/data/` に書き出す |
| `uv run jspace-demo check --model qwen3-14b` | モデルとレンズが読み込めて、想定どおりに読み出せるか確かめる |

モデルは Qwen3.5-4B（既定）、Qwen3-8B、Gemma 3 12B、Qwen3-14B で、それぞれ Neuronpedia が公開したレンズを使います。
Apple の GPU がなければ CUDA の GPU か CPU を使うようにしていますが、動作を確かめたのは Apple silicon だけです。

## テスト

```bash
scripts/check.sh                       # 体裁、lint、画面の JavaScript、単体テスト（モデル不要）
uv run playwright install chromium     # 初回のみ
uv run pytest -m e2e                   # 全画面 × 日本語・英語 × PC 幅・スマートフォン幅
uv run pytest -m model                 # 実モデルで（重い。LLM は 1 つずつ）
```

テストの層と、それぞれが防ぐリスクは [テスト](docs/testing.md) にまとめています。レポートの元になった実験は
[`experiments/`](experiments/) にあります。

## 限界

- レンズが読み出すのは連想であって、意図や判断ではありません。「本音」「頭の中」はたとえです。
- 話題そのものが危険を連想させる場合は、危なくない相談でも懸念が上がります（漂白剤 → 塩素、投資 → リスク）。
  [レポート](docs/report.md) では、すべての相談を、同じ形で危なくない相談と比べています。
- 各事例は小さなモデルによる 1 回の貪欲生成です。件数は少なく、統計ではありません。
- 返答の態度（警告したか、同意したか）はキーワードの規則で判定しています。誤判定した返答を読んで 2 回改訂しました。
  各版の結果はすべて [レポート](docs/report.md) に載せています。
  計画に乗りつつ軽い保留を添えた返答は、同意と数えます。

## クレジットとライセンス

- 論文と J-lens：Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*（2026）
- [`jacobian-lens`](https://github.com/anthropics/jacobian-lens)（Apache-2.0）はインストール時に固定したコミットで取得し、
  無改変で使います。このリポジトリには含めていません。
- レンズは [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens) が公開したもの（MIT）で、初回の実行時に
  ダウンロードします。
- モデル：[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)、[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、
  [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)（Apache-2.0）、[Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it)
  （Gemma Terms of Use）。`web/data/` には、用意した事例に対する各モデルの返答と読み出しが入っています。
- 書体：[Geist と Geist Mono](https://github.com/vercel/geist-font)（SIL Open Font License 1.1）を、ライセンスとともに
  `web/fonts/` に同梱しています。[NOTICE](NOTICE) も参照してください。

J-Space Demo は [Apache License 2.0](LICENSE) で公開しています。
