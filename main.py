# Made by Isaac Joffe

import analyze
import beam
import number
import relationships
import yolo
import time
import argparse
from pathlib import Path

import os
import sys

# stdout = sys.stdout
# sys.stdout = open(os.devnull, "w")
# sys.stderr = open(os.devnull, "w")


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-name', '--img-name', '--image', '--img', '--i', nargs=1, type=str, required=True, help='image of diagram to analyze')
    parser.add_argument('--features-path', '--features', '--f', nargs=1, default=['models/features/'], type=str, help='location of features YOLO model')
    parser.add_argument('--number-path', '--number', '--n', nargs=1, default=['models/number/'], type=str, help='location of number reading model')
    parser.add_argument('--relationships-path', '--r', nargs=1, default=['models/relationships/'], type=str, help='location of relationships perceptrons')
    options = parser.parse_args()
    return options


def main():
    options = parse_options()
    c_time = time.time()

    print("\nExtracting image features...\n")
    image_features = yolo.run(options.image_name[0], options.features_path[0])
    print(f"\nDone. {time.time() - c_time} seconds.\n")
    c_time = time.time()

    print("\nReading number values...\n")
    number_model = number.create_number_model(options.number_path[0])
    if image_features:
        for i in range(len(image_features)):
            if image_features[i][-1] == 4:
                number_name = number.segment_image(options.image_name[0], image_features[i])
                number_name = number.preprocess_image(number_name)
                if number_name:
                    number_value = number.read_number(number_name, number_model)
                else:
                    number_value = 1
                image_features[i].insert(5, number_value)    # insert read value in second-to-last position
    number.clear_number_model(number_model)
    print(f"\nDone. {time.time() - c_time} seconds.\n")
    c_time = time.time()

    print("\nLoading multilayer perceptrons...\n")
    golden_bs_model = relationships.load_golden_model(options.relationships_path[0] + "Beam-Support")
    golden_bl_model = relationships.load_golden_model(options.relationships_path[0] + "Beam-Load")
    golden_ln_model = relationships.load_golden_model(options.relationships_path[0] + "Load-Number")
    golden_gn_model = relationships.load_golden_model(options.relationships_path[0] + "Length-Number")
    golden_el_model = relationships.load_golden_model(options.relationships_path[0] + "Element-Length")
    golden_ls_model = relationships.load_golden_model(options.relationships_path[0] + "Length-Style")
    print(f"\nDone. {time.time() - c_time} seconds.\n")
    c_time = time.time()

    print("\nConsolidating image features...\n")
    detected_beams = beam.beamify(image_features, golden_bs_model, golden_bl_model, golden_ln_model)
    print(f"\nDone. {time.time() - c_time} seconds.\n")
    c_time = time.time()

    print("\nAnalyzing resultant beam system...\n")
    save_dir = "runs/" + (options.image_name[0].split(".")[0]).split("/")[-1]
    print(image_features)
    for i in range(len(detected_beams)):
        Path(f"{save_dir}/{i}").mkdir(parents=True, exist_ok=True)
        analyze.analyze_beam(detected_beams[i], image_features, golden_gn_model, golden_el_model, golden_ls_model, f"{save_dir}/{i}")
    print(f"\nDone. {time.time() - c_time} seconds.\n")

    print(f"\nAnalysis plots saved to directory runs/{save_dir}.\n")
    return


if __name__ == "__main__":
    main()
