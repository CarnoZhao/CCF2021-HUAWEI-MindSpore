from io import BytesIO
import os
import pandas as pd
import glob
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision.c_transforms as vision
from PIL import Image

for name in ("train", "test"):
    MINDRECORD_FILE = f"./data/{name}.msrec"

    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        os.remove(MINDRECORD_FILE + ".db")

    writer = FileWriter(file_name = MINDRECORD_FILE, shard_num = 1)
    cv_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema, "training set")
    writer.add_index(["file_name", "label"])

    images = sorted(glob.glob(f"./data/{name}/*/*.*"))
    df = pd.DataFrame({"file_name": images})
    df["label"] = df.file_name.apply(lambda x: x.split("/")[3])
    labels = {l: i for i, l in enumerate(sorted(os.listdir(f"./data/{name}")))}
    df.label = df.label.apply(lambda x: labels[x])

    data = []
    for i in range(len(df)):
        sample = {}
        bytesio = BytesIO()

        image = Image.open(df.loc[i, "file_name"]).convert(mode = "RGB")
        image.save(bytesio, 'JPEG')
        image_bytes = bytesio.getvalue()
        sample["file_name"] = df.loc[i, "file_name"]
        sample["label"] = int(df.loc[i, "label"])
        sample["data"] = image_bytes

        data.append(sample)
        if (i + 1) % 1000 == 0:
            writer.write_raw_data(data)
            data = []
            print(round(i / len(df) * 100, 2))

    if data:
        writer.write_raw_data(data)

    writer.commit()