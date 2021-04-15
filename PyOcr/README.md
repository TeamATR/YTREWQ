## 環境構築手順(macの場合)

1. pyenv, pyenv-virtualenvのインストール<br>
`brew install pyenv`<br>
`brew install pyenv-virtualenv`<br>

2. python3.7.1のインストール<br>
`pyenv install 3.7.1`<br>

3. pythonのバージョンを変える<br>
`pyenv global 3.7.1`<br>
動作確認したのが3.7.1、というだけです。強いこだわりはありません。

4. cloneもしくはpull<br>
端折ります<br>

5. 仮想環境の作成<br>
`python -m venv env`<br>

6. 仮想環境のアクティベート<br>
`source env/bin/activate`<br> 

7. 関連パッケージのインストール<br>
`pip install -U pip`<br>
`pip install -r requirements.txt`<br>

## サンプルの内訳

1. `example.py`<br>
実行コマンドは`python example.py`<br>
ファイル内で指定している画像を読み込んで、含まれている文字列をコンソール上に出力する。<br>

2. `example_display_area.py`<br>
実行コマンドは`python example_display_area.py`<br>
ファイル内で指定している画像を読み込んで、1の結果＋検出した場所を赤枠で表示した画像ファイルを出力する。出力先もファイル内で指定している。
