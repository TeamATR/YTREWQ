import pyocr
import pyocr.builders
import cv2
from PIL import Image
import sys

tools = pyocr.get_available_tools()

filename = "./led_kairo_high.png"
output_filename = "./result/detect_result.png"

if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

tool = tools[0]

res = tool.image_to_string(Image.open(filename), lang="jpn+eng", builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))
#res = tool.image_to_string(Image.open(filename), lang="jpn", builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))

out = cv2.imread(filename)

for d in res:
    print(d.content, d.position, d.confidence)
    cv2.rectangle(out, d.position[0], d.position[1], (0, 0, 255), 2)

#cv2.imshow("img",out)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite(output_filename, out)

