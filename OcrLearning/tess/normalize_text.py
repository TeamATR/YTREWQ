import sys
import glob

KATAKANA = ('。「」、・ヲァィゥェォャョラッー'
            'アイウエオカキクケコサシスセソタチツテトナニヌネノ'
            'ハヒフヘホマミムメモヤユヨラリルレロワヲン')
args = sys.argv  # text2imageを実行した先の一時フォルダ名を第1変数として渡す

# 半角カナ→全角カナ, 全角英数記号→半角
# 日本語的におかしい濁点・半濁点などが含まれる場合は厳密には考慮してない（前の文字によって変な結果になる）
# new_linesの項目削除で文字単位のbox座標が崩れるが、LSTM学習用に行ごとに座標をまとめるので結果的に影響しない
def normalize_text(lines):
    new_lines = []
    for i in range(len(lines)):
        code = ord(lines[i][0])
        if ord('！') <= code <= ord('｝') and code != ord('＼'):  # 全角英数
            new_lines.append(chr(code - 0xfee0) + lines[i][1:])
        elif code == ord('ﾞ'):  # 濁点
            if len(new_lines) > 0:
                code_prev = ord(new_lines[-1][0])
                if code_prev == ord('ウ'):
                    del new_lines[-1]
                    new_lines.append('ヴ' + lines[i][1:])
                elif ord('か') <= code_prev <= ord('ホ'):
                    del new_lines[-1]
                    new_lines.append(chr(code_prev + 1) + lines[i][1:])
                else:
                    new_lines.append('゛' + lines[i][1:])
            else:
                new_lines.append('゛' + lines[i][1:])
        elif code == ord('ﾟ'):  # 半濁点
            if len(new_lines) > 0:
                code_prev = ord(new_lines[-1][0])
                if ord('は') <= code_prev <= ord('ホ'):
                    del new_lines[-1]
                    new_lines.append(chr(code_prev + 2) + lines[i][1:])
                else:
                    new_lines.append('゜' + lines[i][1:])
            else:
                new_lines.append('゜' + lines[i][1:])
        elif ord('｡') <= code <= ord('ﾝ'):
            new_lines.append(KATAKANA[code - 0xff61] + lines[i][1:])
        elif code == ord('\\') or code == ord('¥'):
            new_lines.append('￥' + lines[i][1:])
        else:
            new_lines.append(lines[i])
    return new_lines

def main():
    files = glob.glob(args[1] + '/*.box')
    for filename in files:
        with open(filename, 'r+', encoding='utf-8') as file:
            lines = file.readlines()
            file.seek(0)  # 先頭に書き込み位置を移動
            file.write(''.join(normalize_text(lines)))
            file.truncate()  # 書き込んだ位置までで切り詰める

if __name__ == '__main__':
    main()
