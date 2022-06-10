import minio 
import pathlib 
import logging
logging.basicConfig(level=logging.INFO)
from utils import windows_linux_ambiguity

def samples_loader_from_minio(server, access_key, secret_key, n_annotations):
    """
    Code adapted from https://github.com/fvillena/wl-corpus/blob/master/describe_samples.py. It returns an array 
    with referrals content and its annotation. The files are obtained from a remote object server of the research group.
    """
    referrals = []
    annotations = []
    idx = 0
    logging.info("Establishing connection with minio server...")
    # Connecting to the object server and get annotations objects.
    minio_client = minio.Minio(
        server,
        access_key=access_key,
        secret_key=secret_key,
        region='cl',
        secure=True,
    )
    logging.info("Successful connection, obtaining the data...")
    objects = minio_client.list_objects("brat-data", prefix='wl_ground_truth/')
    for o in objects:
        if o.object_name.endswith(".txt") and idx < n_annotations:
            ann_filepath = f"{o.object_name[:-4]}.ann"
            txt_object = minio_client.get_object("brat-data", o.object_name)
            ann_object = minio_client.get_object("brat-data", ann_filepath)
            txt_name = pathlib.Path(o.object_name).name
            ann_name = pathlib.Path(ann_filepath).name
            txt_content = txt_object.read().decode("utf-8")
            ann_content = ann_object.read().decode("utf-8")
            if windows_linux_ambiguity(txt_content, ann_content, txt_name):
                continue
            referrals.append((txt_name,txt_content))
            annotations.append((ann_name,ann_content))
            idx+=1
            if idx%500==0: logging.info(f'{idx} annotations already downloaded..')
    logging.info(f"{n_annotations} annotations successfully downloaded from the repository.")
    return referrals, annotations