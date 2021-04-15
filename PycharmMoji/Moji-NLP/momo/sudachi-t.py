from sudachipy import tokenizer
import json
from sudachipy import config
from sudachipy import dictionary

print(config.DEFAULT_SETTINGFILE)
# /Users/ohke/src/sudachipy/sudachipy/../resources/sudachi.json

with open(config.DEFAULT_SETTINGFILE, 'r', encoding='utf-8') as f:
    settings = json.load(f)
tokenizer_obj = dictionary.Dictionary(settings).create()

#tokenizer_obj = dictionary.Dictionary().create()

print(type(tokenizer_obj))
# <class 'sudachipy.tokenizer.Tokenizer'>

text = '横河電機は1968年に世界で初めて渦流量計を製品化して以来，世界で20万台の販売実績と経験を元にこのたびdigitalYEWFLOを開発いたしました。'
text11 = 'ファイアーエムブレムとは、日本の家庭用ゲーム機・ゲームソフト制作企業任天堂の発売した「愛撫と憎悪とテロの物語」がテーマの、ドロドロの人間ドラマ シミュレーションRPGシリーズである。主に中世風の世界観を用いて人間関係を表す。なお「ファイヤーエンブレム」などと言う奴は営倉行きである。'

tokens = tokenizer_obj.tokenize(tokenizer.Tokenizer.SplitMode.C, text11)
print(type(tokens))
# <class 'sudachipy.morphemelist.MorphemeList'>

for t in tokens:
    print(t.surface(), t.part_of_speech(), t.reading_form(), t.normalized_form())