# coding: UTF-8
import glob
import random
import sys
import textwrap
from collections import Counter

def read_chars(filename):
    # 文字種ごとの出現回数
    count = Counter()
    with open(filename) as chars:
        for line in chars:
            count[int(line.split(',')[0],base=16)] = 0
    return count

def read_all_words(dir_s):
    words = {}
    files = glob.glob(dir_s + '/*.csv')
    for filename in files:
        with open(filename, encoding='utf-8') as file:
            for line in file:
                word = line.split(',')[0]
                words[word] = True
    return list(words.keys())

def main():
    # training_bs.txt
    text = ''
    count_required = 20
    chars = read_chars('chars.txt')
    words = read_all_words('mecab-ipadic-neologd/seed')
    print("Total words %d" % len(words))
    training = open('training_bs.txt', 'w', encoding='utf-8')
    random.shuffle(words)
    for word in words:
        min_count = 10000
        skip = False
        # wordに含まれる文字の中で出現回数が最少のもの
        for c in word:
            code = ord(c)
            if code not in chars:
                # 文字種リストに含まれない文字がある場合はスキップ
                skip = True
                # スキップの場合は警告表示
                print("skipped %s by %s" % (word, c), file=sys.stderr)
                break
            count = chars[code] + 1
            if count < min_count:
                min_count = count
        # 最少出現回数が20回以下なら、この単語は「使う」
        if not skip and min_count <= count_required:
            text += word
            # 使ったら出現回数をアップデート
            for c in word:
                code = ord(c)
                chars[code] += 1
    # まとめて出力
    training.write("\n".join(textwrap.wrap(text, width=40)))
    training.close()

    # 1回も使われなかった文字
    with open('unused_chars.txt', 'w', encoding='utf-8') as uc:
        for c in chars:
            if chars[c] == 0:
                print('0x%x,%s' % (c, chr(c)), file=uc)

if __name__ == '__main__':
    main()
