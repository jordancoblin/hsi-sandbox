# hsi-sandbox

## Overview

This repo contains models for performing classification on hyperspectral image data. Specifically, it uses the Hytexila dataset (from https://color.univ-lille.fr/datasets/hytexila) as a sandbox for testing out different techniques. Currently, the only classification model implemented is the Local Binary Pattern (LBP) based algorithm as described in [HyTexiLa: High Resolution Visible and Near Infrared Hyperspectral Texture Images](https://www.mdpi.com/1424-8220/18/7/2045).

## Run Instructions

1. Create virtual env and install dependencies from `requirements.txt`.
2. Download dataset from https://color.univ-lille.fr/datasets/hytexila - specifically, we need the `ENVI` and `classification` folders to be loaded into a `hytexila` directory in this repo.
3. Run `python extract_envi_files.py` to extract all zipped files from `hytexila/ENVI`.
4. Run `python lbp.py` to run the LBP-based classification algorithm. Once constructed, the LBP histogram classifier is dumped to a pickle file, and can be loaded in subsequent runs via the `--load-classifier` flag.