from dataclasses import dataclass, asdict
import argparse



@dataclass
class Panda():
    tarball_path: str = "/opt/ml/input/data/tarballs/panda.tar"
    output_root: str = "/opt/ml/output/preprocessing/"   #?? should this be the same for all
    remove: str = "/opt/ml/input/data/restrict_files/downstream_panda_test.txt"
    name: str = "panda"
  
@dataclass
class Anglita_he_pten_biopsies():
    tarball_path: str = "/opt/ml/input/data/tarballs/anglita_he_pten_biopsies.tar"
    output_root: str = "/opt/ml/output/preprocessing/"
    remove: str = "/opt/ml/input/data/restrict_files/downstream_panda_test.txt"
    name: str = "anglita_he_pten_biopsies"

@dataclass 
class Tcga_prostate():
    tarball_path: str = "/opt/ml/input/data/tarballs/tcga_prostate.tar"
    output_root: str = "/opt/ml/output/preprocessing/"
    remove: str = "/opt/ml/input/data/restrict_files/downstream_panda_test.txt"
    name: str = "tcga_prostate"

@dataclass
class Ecp():
    tarball_path: str = "/opt/ml/input/data/tarballs/ecp.tar"
    output_root: str = "/opt/ml/output/preprocessing/"
    remove: str = "/opt/ml/input/data/restrict_files/downstream_panda_test.txt"
    name: str = "ecp"
