# Provenance Filtering

This software aims to find the most related images with respect to a given query in a potentially large pool of images. It was developed using the Python 3.5 language and tested on Ubuntu 16.04 LTS.

## How to install this software?

We provide a Dockerfile to build Docker containers capable to install all requirements, to compile, and to install this software. To build a docker container for this software please execute the following command:
>
>     ./extra/build_docker_image.sh


## Download Software and Dataset

To download this software, please execute the following command:
>
>     cd ~
>     git clone https://github.com/allansp84/provenance-filtering.git

To test this software upon the Oxford100k Dataset (http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/flickr100k.html), please execute the following commands:
>
>     cd ~/provenance-filtering
>     mkdir -p datasets/
>     cd datasets
>     wget http://www.recod.ic.unicamp.br/~allansp/files/oxford100k.tar.gz
>     tar -xvzf oxford100k.tar.gz

## Bulding a Docker Container

To build a Docker Image to run this software, please execute the commands below:
>
>     cd ~
>     cd provenance-filtering/extra
>     ./build_docker_image.sh

After build a Docker Image, please execute the following commands to run a docker container:
>
>     docker run --userns=host -it --name provfiltering \
>                -v $(pwd)/datasets:/root/datasets \
>                -v $(pwd)/docker-results:/root/docker-results \
>                provenance-filtering


## Usage

After installing our software, we can use it via command line interface (CLI).
To see how to use this software, execute the following command in any
directory, since it will be already installed in your system:
>
>     filtering_icip17.py --help


## Examples
1. Extract the SURF key-points and SURF descriptors (500 feature vectors per image) from all images in the probe and gallery set:
>
>     filtering_icip17.py --dataset 4 --dataset_path /root/datasets/oxford100k \
>                         --groundtruth_path /root/datasets/oxford100k/rel-oxford5k.txt \
>                         --output_path /root/docker-results \
>                         --detector_kpm SURF --descriptor_kpm SURF --limit_kpm 500 \
>                         --feature_extraction \
>                         --n_jobs 4

2. Index a set of images to perform an approximate nearest neighbors search using KD-Forest with 8 trees
>
>     filtering_icip17.py --dataset 4 --dataset_path /root/datasets/oxford100k \
>                         --groundtruth_path /root/datasets/oxford100k/rel-oxford5k.txt \
>                         --output_path /root/docker-results \
>                         --detector_kpm SURF --descriptor_kpm SURF --limit_kpm 500 \
>                         --n_jobs 4 \
>                         --index_type_kpm KDFOREST --n_kdtree_kpm 8 --distance_kpm L2 \
>                         --search_type_kpm knn --score_type_kpm vote --n_neighbors_kpm 51 \
>                         --compute_distances_kpm \
>                         --matching \
>                         --plot_pr_curves

### Reference

If you use this software, please cite our paper published in *IEEE International Conference on Image Processing* and *IEEE Transactions on Image Processing*:


> **BibTeXs**
>
>     @INPROCEEDINGS{8296532,
>     author={A. {Pinto} and D. {Moreira} and A. {Bharati} and J. {Brogan} and K. {Bowyer} and P. {Flynn} and W. {Scheirer} and A. {Rocha}},
>     booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
>     title={Provenance filtering for multimedia phylogeny},
>     year={2017},
>     volume={},
>     number={},
>     pages={1502-1506},
>     keywords={biological techniques;biology computing;feature extraction;genetics;image filtering;composite/doctored image;query image;traditional digital forensics modeling;potential images;creation process;host images;digital objects;multimedia phylogeny;evolutionary processes;two-tiered provenance filtering approach;Phylogeny;Multimedia communication;Indexing;Task analysis;Forensics;Robustness;Filtering;Provenance Filtering;Multimedia Phylogeny;Phylogeny Graph;Provenance Context Incorporation},
>     doi={10.1109/ICIP.2017.8296532},
>     ISSN={2381-8549},
>     month={Sep.},}

>     @ARTICLE{8438504,
>     author={D. {Moreira} and A. {Bharati} and J. {Brogan} and A. {Pinto} and M. {Parowski} and K. W. {Bowyer} and P. J. {Flynn} and A. {Rocha} and W. J. {Scheirer}},
>     journal={IEEE Transactions on Image Processing},
>     title={Image Provenance Analysis at Scale},
>     year={2018},
>     volume={27},
>     number={12},
>     pages={6109-6123},
>     keywords={feature extraction;graph theory;image filtering;image retrieval;social networking (online);image provenance analysis;image processing;computer vision techniques;individual images;query image;original images;image transformations;cutting-edge image filtering solution;social media site Reddit;public media manipulation;Task analysis;Pipelines;Image retrieval;Cultural differences;Image color analysis;Social network services;Digital image forensics;digital humanities;image retrieval;graphs;image provenance;image phylogeny},
>     doi={10.1109/TIP.2018.2865674},
>     ISSN={1941-0042},
>     month={Dec},}


### License

This software is available under condition of the [AGPL-3.0 Licence](https://github.com/allansp84/provenance-filtering/blob/master/LICENSE).

Copyright (c) 2017, A. Pinto and D. Moreira and A. Bharati and J. Brogan and K. Bowyer and P. Flynn and W. Scheirer and A. Rocha
