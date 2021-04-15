from sudachipy import tokenizer
from sudachipy import dictionary


tokenizer_obj = dictionary.Dictionary().create()


# Multi-granular tokenization
# using `system_core.dic` or `system_full.dic` version 20190781
# you may not be able to replicate this particular example due to dictionary you use

mode = tokenizer.Tokenizer.SplitMode.C
[m.surface() for m in tokenizer_obj.tokenize("国家公務員", mode)]
# => ['国家公務員']

mode = tokenizer.Tokenizer.SplitMode.B
[m.surface() for m in tokenizer_obj.tokenize("国家公務員", mode)]
# => ['国家', '公務員']

mode = tokenizer.Tokenizer.SplitMode.A
[m.surface() for m in tokenizer_obj.tokenize("国家公務員", mode)]
# => ['国家', '公務', '員']


# Morpheme information

m = tokenizer_obj.tokenize("横河電機は1968年に世界で初めて渦流量計を製品化して以来，世界で20万台の販売実績と経験を元にこのたびdigitalYEWFLOを開発いたしました。", mode)[0]
talk = tokenizer_obj.tokenize("横河電機は1968年に世界で初めて渦流量計を製品化して以来，世界で20万台の販売実績と経験を元にこのたびdigitalYEWFLOを開発いたしました。", mode)
talk11 = tokenizer_obj.tokenize("ファイアーエムブレムとは、日本の家庭用ゲーム機・ゲームソフト制作企業任天堂の発売した「愛撫と憎悪とテロの物語」がテーマの、ドロドロの人間ドラマ シミュレーションRPGシリーズである。主に中世風の世界観を用いて人間関係を表す。なお「ファイヤーエンブレム」などと言う奴は営倉行きである。", mode)
m.surface() # => '食べ'
m.dictionary_form() # => '食べる'
m.reading_form() # => 'タベ'
m.part_of_speech() # => ['動詞', '一般', '*', '*', '下一段-バ行', '連用形-一般']

# Normalization

tokenizer_obj.tokenize("附属", mode)[0].normalized_form()
# => '付属'
tokenizer_obj.tokenize("SUMMER", mode)[0].normalized_form()
# => 'サマー'
tokenizer_obj.tokenize("シュミレーション", mode)[0].normalized_form()
# => 'シミュレーション'
print(m.part_of_speech())

for arr in talk11:
    print(arr.surface())
    print(arr.part_of_speech())
    print("----")
