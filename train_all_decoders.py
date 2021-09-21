import os
import traceback

import plb.algorithms.autoencoder.train_autoencoder as trainer

ENCODER_METHODS = {
    "cfm", 
    "inverse",
    "forward", 
    "e2c"
}

DATASETS = {
    "chopsticks",
    "rope",
    "torus",
    "writer"
}

_DRY_RUN = False

def _folder_name_2_dataset(folderName:str) -> str:
    """ Converting the folder name to the corresponding dataset

    e.g. folder name: chopsticks_linear => chopsticks
    e.g. folder name: torus => torus

    :param folderName: the name of the folder storing the encoder model
    :return: the dataset name
    """
    for each in DATASETS:
        if each == folderName.split('_')[0]: return each

    raise NotImplementedError("invalid folderName:" + folderName)

def _get_newest_encoder_path(methodFolder: str, envFolder: str) -> str:
    """ Find the path to the newest encoder

    The directory to be searched for encoder.pth will be pretrain_model/methodFolder/envFolder
    such as: pretrain_model/cfm/chopsticks_linear. Should there exist an model called
    `encoder_new.pth`, path to this one will be returned. Otherwise, path to `encoder.pth`
    will be returned. 

    :param methodFolder: the first level of directory, under pretrain_model folder
    :param envFolder: the second level of directory, under pretrain_model folder
    :return: the path of encoder, relative to pretrain_model
    """
    fullDir = os.path.join(os.path.join("pretrain_model", methodFolder, envFolder))
    if os.path.exists(os.path.join(fullDir, "encoder_new.pth")):
        fullPath2Encoder = os.path.join(methodFolder, envFolder, "encoder_new")
    else:
        fullPath2Encoder = os.path.join(methodFolder, envFolder, "encoder")
    return fullPath2Encoder

def _find_encoder_and_train(firstLevelDir: str, secondLevelDir: str):
    """ Locate the encoder from pretrain_model/{firstLevelDir}/{secondLevelDir}
    and train the autoencoder based on the encoder. 

    :param firstLevelDir: the first level of directory to find the encoder, under pretrain_model folder
    :param secondLevelDir: the second level of directory, under pretrain_model folder
    """
    fullPath2Encoder = _get_newest_encoder_path(firstLevelDir, secondLevelDir)
    if _DRY_RUN:
        print(f"now running auto encoder on {fullPath2Encoder}: \\\n" + \
            f"\t--saved_model = {fullPath2Encoder} \\\n" + \
            f"\t--exp_name    = {firstLevelDir}/{secondLevelDir}/whole \\\n" + \
            f"\t--dataset     = {_folder_name_2_dataset(secondLevelDir)} \n"
        )
    else:
        outFile = f"{firstLevelDir}_{secondLevelDir}.out"
        with open(os.path.join("out", "decoder", outFile), 'w') as f:
            trainer.main(
                loss          = "chamfer", 
                iters         = 100, 
                savedModel    = fullPath2Encoder, 
                expName       = os.path.join(firstLevelDir, secondLevelDir, "whole"),
                dataset       = _folder_name_2_dataset(secondLevelDir),
                freezeEncoder = True,
                loggerFunc    = lambda line: f.write(line+'\n')
            )


if __name__ == '__main__':
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists('out/decoder'):
        os.mkdir('out/decoder')

    methodFolders = os.listdir('pretrain_model')
    for eachMethod in methodFolders:
        if eachMethod not in ENCODER_METHODS:
            print(f"skip pretrain_model/{eachMethod}")
            continue

        envFolders = os.listdir(os.path.join("pretrain_model", eachMethod))
        for eachEnv in envFolders:
            try:
                _find_encoder_and_train(eachMethod, eachEnv)
            except KeyboardInterrupt:
                print("\033[31mKeyboard Interruption[0m") 
                exit(0)
            except Exception:
                print(f"\033[31mFailed to tackle {eachMethod}/{eachEnv} => {traceback.format_exc()}\033[0m") 
