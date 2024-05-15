import os
import numpy as np
from e3nn import o3

ROOT_DIRPATH = '/home/bb596/hdd'
BENCHMARK_DIRNAME = 'ThreeDGenMolBenchmark'
BENCHMARK_DIRPATH = os.path.join(ROOT_DIRPATH,
                                 BENCHMARK_DIRNAME)
if not os.path.exists(BENCHMARK_DIRPATH):
    os.mkdir(BENCHMARK_DIRPATH)

DATA_DIRNAME = 'data'
DATA_DIRPATH = os.path.join(BENCHMARK_DIRPATH, DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.mkdir(DATA_DIRPATH)
    
MBD_PATH = '/home/bb596/hdd/ymir/mbd.p'
    
# LIGBOUNDCONF_FILEPATH = os.path.join(ROOT, 'LigBoundConf/minimized/S2_LigBoundConf_minimized.sdf')
# LIGBOUNDCONF_PDB_DIRPATH = os.path.join(ROOT, 'LigBoundConf/')

# VINA_DIRPATH = '/home/bb596/vina/'
# VINA_URL = 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.4/vina_1.2.4_linux_x86_64'
# VINA_BIN_FILENAME = VINA_URL.split('/')[0]
# VINA_BIN_FILEPATH = os.path.join(VINA_DIRPATH, VINA_BIN_FILENAME)

LIGANDEXPO_FILENAME = 'Components-smiles-stereo-cactvs.smi'
BASE_LIGANDEXPO_URL = 'http://ligand-expo.rcsb.org/dictionaries'
LIGANDEXPO_URL = f"{BASE_LIGANDEXPO_URL}/{LIGANDEXPO_FILENAME}"
LIGANDEXPO_DIRNAME = 'LigandExpo'
LIGANDEXPO_DIRPATH = os.path.join(ROOT_DIRPATH,
                                  LIGANDEXPO_DIRNAME)
if not os.path.exists(LIGANDEXPO_DIRPATH):
    os.mkdir(LIGANDEXPO_DIRPATH)
LIGANDEXPO_FILEPATH = os.path.join(LIGANDEXPO_DIRPATH,
                                   LIGANDEXPO_FILENAME)

# These URL needs to be provided by the user, as PDBbind is under license
# and requires to be logged in. The "Cloud CDN" links are faster than "Local Download" links
# PDBBIND_GENERAL_URL: str = 'PDBbind_v2020_other_PL'
PDBBIND_GENERAL_URL = 'https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_other_PL.tar.gz'
if PDBBIND_GENERAL_URL is None:
    raise Exception("""PDBBIND_GENERAL_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/download.php, 
                    and find the links to the general set""")
    
# PDBBIND_REFINED_URL: str = 'PDBbind_v2020_refined'
PDBBIND_REFINED_URL = 'https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz'
if PDBBIND_REFINED_URL is None:
    raise Exception("""PDBBIND_REFINED_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/download.php, 
                    and find the links to the refined set""")
    
PDBBIND_CORE_URL = 'https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/CASF-2016.tar.gz'
if PDBBIND_CORE_URL is None:
    raise Exception("""PDBBIND_CORE_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/casf.php, 
                    and find the links to the core set""")
    

PDBBIND_DIRNAME = 'PDBbind'
PDBBIND_DIRPATH = os.path.join(ROOT_DIRPATH, 
                               PDBBIND_DIRNAME)
if not os.path.exists(PDBBIND_DIRPATH):
    os.mkdir(PDBBIND_DIRPATH)

PDBBIND_GENERAL_TARGZ_FILENAME = PDBBIND_GENERAL_URL.split('/')[-1]
PDBBIND_GENERAL_TARGZ_FILEPATH = os.path.join(PDBBIND_DIRPATH,
                                              PDBBIND_GENERAL_TARGZ_FILENAME)
PDBBIND_GENERAL_DIRNAME = 'general'
PDBBIND_GENERAL_DIRPATH = os.path.join(PDBBIND_DIRPATH,
                                       PDBBIND_GENERAL_DIRNAME)
# if not os.path.exists(PDBBIND_GENERAL_DIRPATH):
#     os.mkdir(PDBBIND_GENERAL_DIRPATH)

PDBBIND_REFINED_TARGZ_FILENAME = PDBBIND_REFINED_URL.split('/')[-1]
PDBBIND_REFINED_TARGZ_FILEPATH = os.path.join(PDBBIND_DIRPATH,
                                              PDBBIND_REFINED_TARGZ_FILENAME)
PDBBIND_REFINED_DIRNAME = 'refined'
PDBBIND_REFINED_DIRPATH = os.path.join(PDBBIND_DIRPATH,
                                       PDBBIND_REFINED_DIRNAME)
# if not os.path.exists(PDBBIND_REFINED_DIRPATH):
#     os.mkdir(PDBBIND_REFINED_DIRPATH)

PDBBIND_CORE_TARGZ_FILENAME = PDBBIND_CORE_URL.split('/')[-1]
PDBBIND_CORE_TARGZ_FILEPATH = os.path.join(PDBBIND_DIRPATH,
                                           PDBBIND_CORE_TARGZ_FILENAME)
PDBBIND_CORE_DIRNAME = 'core'
PDBBIND_CORE_DIRPATH = os.path.join(PDBBIND_DIRPATH,
                                       PDBBIND_CORE_DIRNAME)

PREPARE_RECEPTOR_BIN_PATH = '/home/bb596/ADFRsuite/bin/prepare_receptor'
SCHRODINGER_PATH = '/usr/local/shared/schrodinger/current/'
GLIDE_OUTPUT_DIRPATH = '/home/bb596/ymir/glide_working_dir/'
if not os.path.exists(GLIDE_OUTPUT_DIRPATH):
    os.mkdir(GLIDE_OUTPUT_DIRPATH)

FEATURES_DIM = 256
NET_ARCH = {'pi': [64, 64],
            'vf': [64, 32]}

TORSION_SPACE_STEP = 10

COMENET_MODEL_NAME = 'ComENet'
COMENET_CONFIG = {"lr":1e-4,
                  'batch_size': 256}

POCKET_RADIUS = 6
CNN_RADIUS = 4
NEIGHBOR_RADIUS = 3
L_MAX = 3
EMBED_HYDROGENS = False
N_HIDDEN_INV_FEATURES = 64
N_HIDDEN_EQUI_FEATURES = 64
# HIDDEN_IRREPS = o3.Irreps(f'{N_HIDDEN_INV_FEATURES}x0e + {N_HIDDEN_EQUI_FEATURES}x1o + {N_HIDDEN_EQUI_FEATURES}x2e + {N_HIDDEN_EQUI_FEATURES}x3o')
# HIDDEN_IRREPS = o3.Irreps(f'{N_HIDDEN_INV_FEATURES}x0e + {N_HIDDEN_EQUI_FEATURES}x1o')
HIDDEN_IRREPS = o3.Irreps(f'{N_HIDDEN_EQUI_FEATURES}x1o + {N_HIDDEN_EQUI_FEATURES}x2e')
# HIDDEN_IRREPS = o3.Irreps(f'{N_HIDDEN_INV_FEATURES}x0e')
IRREPS_OUTPUT = o3.Irreps(f'128x0e')
# HIDDEN_IRREPS = o3.Irreps(f'{N_HIDDEN_INV_FEATURES}x0e')
N_IRREPS_FRAGMENT = 64
IRREPS_FRAGMENTS = o3.Irreps(f'{N_IRREPS_FRAGMENT}x0e')

N_ROTATIONS = 6
TORSION_ANGLES_DEG = np.arange(-180, 180, 360 / N_ROTATIONS)

DEFAULT_SF_NAME = 'vina'
DEFAULT_PREPARATION_METHOD = 'adfr'
DEFAULT_SIZE_BORDER = 25 # Angstrom
DEFAULT_N_CPUS = 8
SEED = 2023

N_INTERACTION_BLOCKS = 2
NUMBER_OF_BASIS = 10
MOL_ID_EMBEDDING_SIZE = 8
NODE_Z_EMBEDDING_SIZE = 64 - MOL_ID_EMBEDDING_SIZE
MIDDLE_LAYER_SIZE = 32
MAX_NUMBER_TYPES = 100

VINA_DIRECTORY = '/home/bb596/ymir/vina/'
VINA_LIGANDS_DIRECTORY = '/home/bb596/ymir/vina/ligands/'
if not os.path.exists(VINA_LIGANDS_DIRECTORY):
    os.mkdir(VINA_LIGANDS_DIRECTORY)
VINA_OUTPUT_DIRECTORY = '/home/bb596/ymir/vina/poses/'
if not os.path.exists(VINA_OUTPUT_DIRECTORY):
    os.mkdir(VINA_OUTPUT_DIRECTORY)
VINA_MAPS_DIRECTORY = '/home/bb596/ymir/vina/maps/'
if not os.path.exists(VINA_MAPS_DIRECTORY):
    os.mkdir(VINA_MAPS_DIRECTORY)
VINA_CPUS = 8
VINA_SEED = SEED
VINA_CONFIG_PATH = '/home/bb596/ymir/vina/config.txt'
VINA_PATH = '/home/bb596/ymir/vina/vina_1.2.5_linux_x86_64'

VINA_DATASET_PATH = '/home/bb596/hdd/ymir/vina_mini_dataset.p'

SCORING_FUNCTION = 'glide'