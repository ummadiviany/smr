import json, os

def idx2path(dataset_map):
    idx2imgpath = {}
    idx2labelpath = {}
    for dataset in dataset_map:
        print(f"------------{dataset}------------")
        idx2imgpath[dataset] = {}
        idx2labelpath[dataset] = {}
        for idx, (img_path, label_path) in enumerate(zip(dataset_map[dataset]['train']['images'], dataset_map[dataset]['train']['labels'])):
            idx2imgpath[dataset][idx] = img_path
            idx2labelpath[dataset][idx] = label_path
            
        # Sort by idx
        idx2imgpath[dataset] = {k: v for k, v in sorted(idx2imgpath[dataset].items(), key=lambda item: item[0])}
        idx2labelpath[dataset] = {k: v for k, v in sorted(idx2labelpath[dataset].items(), key=lambda item: item[0])}
        
    filename_img = 'idx2imgpath.json'
    filename_label = 'idx2labelpath.json'
    if not os.path.exists(filename_img):
        os.mknod(filename_img)
        os.mknod(filename_label)
    with open(filename_img, 'w') as fp:
        json.dump(idx2imgpath, fp)
    with open(filename_label, 'w') as fp:
        json.dump(idx2labelpath, fp)