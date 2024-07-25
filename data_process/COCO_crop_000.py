import json

from tqdm import tqdm

# read the predict data file —— instance-level
with open('../data/FRCNN_COCO070.json') as f1:
    pre_list = json.load(f1)
# read the truth data file
with open('../data/coco2017/annotations/instances_val2017.json') as f:
    true_list = json.load(f)
print(len(pre_list))
# 给每张图片构建一个空分数列表
id2scores = {x['id']: [] for x in true_list['images']}
print(len(id2scores))
for x in tqdm(pre_list):
    srt = sorted(x['full_score'])
    id2scores[x['image_id']] = id2scores[x['image_id']] + [[srt[-1], srt[-2]]]
results = []
for x in id2scores:
    content_dic = {
        "image_id": x,
        "two_score": id2scores[x]
    }
    results.append(content_dic)
print(len(results))
json_str = json.dumps(results, indent=4)
with open('../data/FRCNN_COCOimage.json', 'w') as json_file:
    json_file.write(json_str)
