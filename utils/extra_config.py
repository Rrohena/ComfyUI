import os
import yaml
import folder_paths
import logging

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
            base_path = os.path.expandvars(os.path.expanduser(base_path))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        is_default = False
        if "is_default" in conf:
            is_default = conf.pop("is_default")
        all_model_folders = False
        if "all_model_folders" in conf:
            all_model_folders = conf.pop("all_model_folders")
        if all_model_folders and base_path:
            for folder_name in list(folder_paths.folder_names_and_paths.keys()):
                if folder_name == "custom_nodes":
                    continue
                full_path = os.path.normpath(os.path.join(base_path, folder_name))
                logging.info("Adding extra search path {} {}".format(folder_name, full_path))
                folder_paths.add_model_folder_path(folder_name, full_path, is_default)
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path:
                    full_path = os.path.join(base_path, full_path)
                elif not os.path.isabs(full_path):
                    full_path = os.path.abspath(os.path.join(yaml_dir, y))
                normalized_path = os.path.normpath(full_path)
                logging.info("Adding extra search path {} {}".format(x, normalized_path))
                folder_paths.add_model_folder_path(x, normalized_path, is_default)
