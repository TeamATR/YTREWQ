import spacy
import neologdn

nlp = spacy.load('ja_ginza')

def ginzatest(moji):

    doc = nlp(moji)

    for sent in doc.sents:
        for token in sent:
            info = [
                token.i,  # トークン番号
                token.orth_,  # テキスト
                token._.reading,  # 読みカナ
                token.lemma_,  # 基本形
                token.pos_,  # 品詞
                token.tag_,  # 品詞詳細
                token._.inf  # 活用情報
            ]
            print(info)


def ginzatestS(moji):
    doc = nlp(moji)

    for sent in doc.sents:
        for token in sent:
            info = [
                token.i,  # トークン番号
                token.orth_,  # テキスト
                token._.reading,  # 読みカナ
                token.lemma_,  # 基本形
                token.pos_,  # 品詞
            ]
            print(info)

def extract_words(sentence):
    docs = nlp(sentence)
    words = set(str(w) for w in docs.noun_chunks)
    words.union(str(w) for w in docs.ents)
    return words

def extract_wordsProp(sentence):
    docs = nlp(sentence)
    words = set(str(w) for w in docs.propn_chunks)
    words.union(str(w) for w in docs.ents)
    return words

if __name__ == '__main__':
    s = "SensPlus Noteは紙やペンのように簡単に記録でき、全ての情報をデジタルノート上で瞬時に共有できます。"

    #ginzatest(s)
    result = extract_words(neologdn.normalize(s))
    #result2 = extract_wordsProp(neologdn.normalize(s))
    print(result)
    #print(result2)

    s = '横河電機は1968年に世界で初めて渦流量計を製品化して以来，世界で20万台の販売実績と経験を元にこのたびdigitalYEWFLOを開発いたしました。'

    #ginzatest(s)
    result = extract_words(neologdn.normalize(s))
    print(result)

    s = '私の語彙力まじヤバイ、わかる、ほんとそれな。'

    # ginzatest(s)
    result = extract_words(neologdn.normalize(s))
    print(ginzatestS(s))
    print(result)