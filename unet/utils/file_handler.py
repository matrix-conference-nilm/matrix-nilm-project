from pathlib import Path


def listup_files(_path, type='json'):
    if not isinstance(_path, Path):
        _path = Path(_path)
    return [f for f in _path.glob(f"**/*.{type}")]

def name_mapper(json_list):
    mapper = {}
    for json_file in json_list:
        user_id, channel, _, collected_date, _ = Path(json_file).stem.split("_")
        
        house_id = f"{int(user_id.replace('user','')):03d}"
        channel_id = f"ch{channel}"
        key = f"H{house_id}_{channel_id}_{collected_date}"
        mapper[key] = json_file

    return mapper


def read_labels(csv_path, mapper):

    labeling_path = mapper.get(csv_path.stem, None)
    if not labeling_path: 
        return None
        # raise Exception(f"{csv_path} is has no labeling json {labeling_path}")
    with open(labeling_path, 'r') as j:
        labels = json.load(j)
    return labels['labels']