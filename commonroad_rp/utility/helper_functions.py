import os
from datetime import datetime
import subprocess
import zipfile
import ruamel.yaml as yaml
import yaml as yml


def delete_folder(path):
    if os.path.exists(path):
        # shutil.rmtree(path)
        subpr_handle = subprocess.Popen("sudo rm -rf " + path, shell=True)
        wait = subpr_handle.wait()


def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)


def createfolder_if_not_existent(inputpath):
    if not os.path.exists(inputpath):
        os.makedirs(inputpath, mode=0o777)
        name_folder = inputpath.rsplit('/')[-1]
        print("Create " + name_folder + " folder")


def create_time_in_date_folder(inputpath):
    # directory with time stamp to save csv
    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    if not os.path.exists(inputpath):
        os.makedirs(inputpath, mode=0o777)
    if not os.path.exists(os.path.join(inputpath, date)):
        os.makedirs(os.path.join(inputpath, date), mode=0o777)
    os.makedirs(os.path.join(inputpath, date, time), mode=0o777)

    return os.path.join(inputpath, date, time)


def zip_log_files(inputpath):
    filePaths = []
    # Read all directory, subdirectories and file lists
    for root, directories, files in os.walk(inputpath):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)

    # writing files to a zipfile
    zip_file = zipfile.ZipFile(inputpath + '.zip', 'w')
    with zip_file:
        # writing each file one by one
        for file in filePaths:
            zip_file.write(file)

    print(inputpath + '.zip file is created successfully!')

    # Remove Log files
    # shutil.rmtree(inputpath)


def open_config_file(path: str):
    # Load config with the set of tuning parameters
    with open(path) as f:
        config_parameters_ = yml.load(f, Loader=yaml.RoundTripLoader)
    return config_parameters_


def delete_empty_folders(path: str):

    folders = list(os.walk(path))[1:]

    for folder in folders:
        inner_folders = list(os.walk(folder[0]))[1:]
        for inner_folder in inner_folders:
            # folder example: ('FOLDER/3', [], ['file'])
            if not inner_folder[2]:
                os.rmdir(inner_folder[0])
        if not folder[2]:
            try:
                os.rmdir(folder[0])
            except:
                pass

# EOF
