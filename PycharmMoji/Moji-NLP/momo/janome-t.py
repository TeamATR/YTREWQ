from janome.tokenizer import Tokenizer

t = Tokenizer('neologd')
text = '横河電機は1968年に世界で初めて渦流量計を製品化して以来，世界で20万台の販売実績と経験を元にこのたびdigitalYEWFLOを開発いたしました。'
text11 = 'ファイアーエムブレムとは、日本の家庭用ゲーム機・ゲームソフト制作企業任天堂の発売した「愛撫と憎悪とテロの物語」がテーマの、ドロドロの人間ドラマ シミュレーションRPGシリーズである。主に中世風の世界観を用いて人間関係を表す。なお「ファイヤーエンブレム」などと言う奴は営倉行きである。'
for token in t.tokenize(text11):
   print(token)