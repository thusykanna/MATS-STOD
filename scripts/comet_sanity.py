from pathlib import Path

from mats_stod.evaluation.comet_metric import CometMetric

read = lambda p: Path(p).read_text(encoding="utf-8")
src = read("data/samples/circular_01/circular_01.si")
ref = read("data/samples/circular_01/circular_01.ta")
wrong = read("data/samples/notice_02/notice_02.ta")  # a different document

m = CometMetric()
print("reference vs itself :", m.score([src], [ref], [ref]))
print("unrelated Tamil doc :", m.score([src], [wrong], [ref]))