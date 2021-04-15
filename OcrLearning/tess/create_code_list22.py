import urllib.request
import re

add_list = [
                0x25b2,  # ▲
                0x25b3,  # △
            ]
chars = {}

with urllib.request.urlopen('http://x0213.org/codetable/sjis-0213-2004-std.txt') as f:
    for line in f.read().decode('ascii').splitlines():
        if line[0] == '#':
            continue
        else:
            m = re.search('U\+([0-9a-f]{4})', line, flags=re.I)
            if m:
                code = int(m.group(1), base=16)
                if code > 0x20:
                    chars[code] = True

del_list = {}
with open('/root/tess/langdata/jpn/forbidden_characters') as f:
    for line in f:
        m = re.search('0x([0-9a-f]{2,4})(-0x([0-9a-f]{2,4}))?\s*$', line, flags=re.I)
        if m:
            if m.group(2):
                range_s = [int(m.group(1), base=16), int(m.group(3), base=16)]
            else:
                range_s = [int(m.group(1), base=16), int(m.group(1), base=16)]
        for c in chars:
            if range_s[0] <= c <= range_s[1]:
                if not (ord('｡') <= c <= ord('ﾟ')  # NOT 半角カタカナ
                        or ord('！') <= c <= ord('｝')):  # NOT 全角英数字
                    print("%s excluded as %x - %x" % (chr(c), range_s[0], range_s[1]))
                    del_list[c] = True

for c in del_list:
    del chars[c]

for c in add_list:
    chars[c] = True

with open('chars.txt', 'w') as wf:
    for code in sorted(chars):
        print("0x%x,%s" % (code, chr(code)), file=wf)
