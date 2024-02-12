import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_ind', required=True, default=7575498, type=int, help="Which prompt index to use for the plot. For example, 5246271 (brown arm chair), 9293003 (blue bird), 9304774 (one office chair), 7539077 (green toaster), 7575498 (red car), 1032122 (bike), 5805831 (computer mouse).")

    cfg = parser.parse_args()

    return cfg