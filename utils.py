from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.absolute()


class PathUtils:
    """
    Utils for file names
    """

    # FOLDERS

    # Input netwroks
    input_networks_folder = get_project_root() / "tntp_networks"
    processed_networks_folder = get_project_root() / "processed_networks"

    # FILES

    # Network files
    anaheim_net_file = input_networks_folder / "Anaheim_net.tntp"
    barcelona_net_file = input_networks_folder / "Barcelona_net.tntp"
    braess_net_file = input_networks_folder / "Braess_net.tntp"
    chicago_net_file = input_networks_folder / "ChicagoSketch_net.tntp"
    eastern_massachusetts_net_file = input_networks_folder / "EMA_net.tntp"
    sioux_falls_net_file = input_networks_folder / "SiouxFalls_net.tntp"
    winnipeg_net_file = input_networks_folder / "Winnipeg_net.tntp"
    friedrichshain_center_net_file = input_networks_folder / "friedrichshain-center_net.tntp"
    sydney_net_file = input_networks_folder / "Sydney_net.tntp"
    berlin_center_net_file = input_networks_folder / "berlin-mitte-center_net.tntp"
    goldcoast_net_file = input_networks_folder / "Goldcoast_net.tntp"
    berlin_renzlauerberg_center_net_file = input_networks_folder / "berlin-prenzlauerberg-center_net.tntp"
    berlin_tiergarten_net_file = input_networks_folder / "berlin-tiergarten_net.tntp"
    berlin_mitte_prenzlauerberg_friedrichshain_center_net_file = input_networks_folder / "berlin-mitte-prenzlauerberg-friedrichshain-center_net.tntp"





