import os, sys

from utils import generate_point_cloud, load_data
from segmentation import compute_point_based_gradient, classify_ground

def main(filename, data_directory="data", debug=False, intermediate_output=False):
    # Perform data load
    fp = os.path.join(os.getcwd(), data_directory, filename)
    data, colors, labels = load_data(fp, intermediate_output=intermediate_output, debug=debug)
    g_mag, g_dir = compute_point_based_gradient(fp, data, intermediate_output=intermediate_output, debug=debug)
    cls = classify_ground(fp, data, g_mag, intermediate_output=intermediate_output, debug=debug)
    print(cls)

if __name__ == "__main__":
    args = {"--file": None, "--debug": False, "--intermediate_output": False, "--datadir": "data"}
    for s in sys.argv[1:]:
        try:
            a, v = s.split("=")
        except Exception as e:
            sys.exit(f"Error: No value provided for argument {s}")
        if a in args.keys():
            if v =='False' or v =='false':
                args[a] = False
            elif v =='True' or v =='true':
                args[a] = True
            else:
                args[a] = str(v)
        else:
            sys.exit(f"Error: argument {a} not accepted")
    if args["--file"] is None:
        sys.exit("Error: A file must be provided")

    main(args["--file"], data_directory=args["--datadir"], debug=args['--debug'], 
        intermediate_output=args["--intermediate_output"])