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
