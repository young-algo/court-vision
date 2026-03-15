"""
merge_data.py — Merge Torvik, ESPN BPI, and NET ranking data into a unified teams.json

Inputs (expected in workspace root):
  - torvik_data.json   (scraped from barttorvik.com/trank.php)
  - bpi_data.json      (scraped from espn.com/mens-college-basketball/bpi)
  - net_data.json      (scraped from warrennolan.com/basketball/2026/net)

Output:
  - client/src/data/teams.json  (merged, normalized, sorted by Torvik rank)

Usage:
  python scripts/merge_data.py
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

# -------------------------------------------------------------------
# Team name normalization
# -------------------------------------------------------------------
# The three data sources use different naming conventions:
#   Torvik:  "Iowa St.", "Michigan St.", "N.C. State"
#   BPI:     "Iowa State Cyclones", "Michigan State Spartans"
#   NET:     "Iowa State", "Michigan State", "North Carolina State"
#
# This mapping resolves every team to a single canonical short name.
# -------------------------------------------------------------------

NAME_MAP = {
    # BPI full names → canonical
    'Duke Blue Devils': 'Duke',
    'Michigan Wolverines': 'Michigan',
    'Arizona Wildcats': 'Arizona',
    'Houston Cougars': 'Houston',
    'Florida Gators': 'Florida',
    'Illinois Fighting Illini': 'Illinois',
    'Iowa State Cyclones': 'Iowa State',
    'Purdue Boilermakers': 'Purdue',
    'UConn Huskies': 'UConn',
    'Connecticut Huskies': 'UConn',
    'Connecticut': 'UConn',
    'Gonzaga Bulldogs': 'Gonzaga',
    'Michigan State Spartans': 'Michigan State',
    'Texas Tech Red Raiders': 'Texas Tech',
    'Virginia Cavaliers': 'Virginia',
    'Vanderbilt Commodores': 'Vanderbilt',
    'Tennessee Volunteers': 'Tennessee',
    'Louisville Cardinals': 'Louisville',
    "St. John's Red Storm": "St. John's",
    "St. John's (NY)": "St. John's",
    "Saint John's": "St. John's",
    'Alabama Crimson Tide': 'Alabama',
    'Arkansas Razorbacks': 'Arkansas',
    'Nebraska Cornhuskers': 'Nebraska',
    'Kansas Jayhawks': 'Kansas',
    'Wisconsin Badgers': 'Wisconsin',
    'Ohio State Buckeyes': 'Ohio State',
    'UCLA Bruins': 'UCLA',
    "Saint Mary's Gaels": "Saint Mary's",
    "Saint Mary's College": "Saint Mary's",
    "Saint Mary's": "Saint Mary's",
    'Iowa Hawkeyes': 'Iowa',
    'North Carolina Tar Heels': 'North Carolina',
    'Santa Clara Broncos': 'Santa Clara',
    'Clemson Tigers': 'Clemson',
    'Utah State Aggies': 'Utah State',
    'BYU Cougars': 'BYU',
    'San Diego State Aztecs': 'San Diego State',
    'Indiana Hoosiers': 'Indiana',
    'Kentucky Wildcats': 'Kentucky',
    'Boise State Broncos': 'Boise State',
    'Marquette Golden Eagles': 'Marquette',
    'Providence Friars': 'Providence',
    'Xavier Musketeers': 'Xavier',
    'West Virginia Mountaineers': 'West Virginia',
    'Maryland Terrapins': 'Maryland',
    'Saint Louis Billikens': 'Saint Louis',
    'Oregon Ducks': 'Oregon',
    'Creighton Bluejays': 'Creighton',
    'New Mexico Lobos': 'New Mexico',
    'Auburn Tigers': 'Auburn',
    'Mississippi State Bulldogs': 'Mississippi State',
    'North Carolina State Wolfpack': 'NC State',
    'North Carolina State': 'NC State',
    'NC State Wolfpack': 'NC State',
    'Penn State Nittany Lions': 'Penn State',
    'Pittsburgh Panthers': 'Pittsburgh',
    'Colorado State Rams': 'Colorado State',
    'Drake Bulldogs': 'Drake',
    'Dayton Flyers': 'Dayton',
    'VCU Rams': 'VCU',
    'Memphis Tigers': 'Memphis',
    'Texas Longhorns': 'Texas',
    'Oklahoma Sooners': 'Oklahoma',
    'Oklahoma State Cowboys': 'Oklahoma State',
    'Florida State Seminoles': 'Florida State',
    'Wake Forest Demon Deacons': 'Wake Forest',
    'Wake Forest Demon': 'Wake Forest',
    'Georgia Tech Yellow Jackets': 'Georgia Tech',
    'Syracuse Orange': 'Syracuse',
    'USC Trojans': 'USC',
    'Oregon State Beavers': 'Oregon State',
    'Arizona State Sun Devils': 'Arizona State',
    'South Carolina Gamecocks': 'South Carolina',
    'Mississippi Rebels': 'Mississippi',
    'Ole Miss Rebels': 'Ole Miss',
    'Ole Miss': 'Ole Miss',
    'SMU Mustangs': 'SMU',
    'UCF Knights': 'UCF',
    'TCU Horned Frogs': 'TCU',
    'Miami (OH) RedHawks': 'Miami (OH)',
    'Miami (FL) Hurricanes': 'Miami (FL)',
    'Miami Hurricanes': 'Miami (FL)',
    'Miami (FL)': 'Miami (FL)',
    'Miami (OH)': 'Miami (OH)',
    'South Florida Bulls': 'South Florida',
    'Wichita State Shockers': 'Wichita State',
    'McNeese Cowboys': 'McNeese',
    'Tulsa Golden Hurricane': 'Tulsa',
    'Virginia Tech Hokies': 'Virginia Tech',
    'Minnesota Golden Gophers': 'Minnesota',
    'Georgetown Hoyas': 'Georgetown',
    'Grand Canyon Lopes': 'Grand Canyon',
    'Akron Zips': 'Akron',
    'George Washington Revolutionaries': 'George Washington',
    'George Mason': 'George Mason',
    'Illinois State': 'Illinois State',
    'Kansas State': 'Kansas State',
    'California Baptist': 'California Baptist',
    'UNCW': 'UNCW',

    # Torvik abbreviated names → canonical
    'Michigan St.': 'Michigan State',
    'Iowa St.': 'Iowa State',
    'Utah St.': 'Utah State',
    'San Diego St.': 'San Diego State',
    'Mississippi St.': 'Mississippi State',
    'Ohio St.': 'Ohio State',
    'Boise St.': 'Boise State',
    'Wichita St.': 'Wichita State',
    'Penn St.': 'Penn State',
    'Colorado St.': 'Colorado State',
    'Arizona St.': 'Arizona State',
    'Kansas St.': 'Kansas State',
}


def normalize(name: str) -> str:
    """Resolve a team name from any source to its canonical form."""
    name = name.strip()
    return NAME_MAP.get(name, name)


def load_json(filename: str) -> list:
    path = os.path.join(WORKSPACE_ROOT, filename)
    with open(path) as f:
        return json.load(f)


def main():
    torvik = load_json('torvik_data.json')
    bpi = load_json('bpi_data.json')
    net = load_json('net_data.json')

    # ----- Build base from Torvik (most complete efficiency data) -----
    teams = {}
    for t in torvik:
        name = normalize(t['team'])
        teams[name] = {
            'name': t['team'],
            'conference': t['conference'],
            'record': t['record'],
            'torvik_rank': t['rank'],
            'adjoe': float(t['adjoe']),
            'adjde': float(t['adjde']),
            'adjEM': round(float(t['adjoe']) - float(t['adjde']), 2),
            'barthag': float(t['barthag']),
            'adjt': float(t['adjt']),
            'bpi': None,
            'bpi_off': None,
            'bpi_def': None,
            'bpi_rank': None,
            'net_rank': None,
        }

    # ----- Merge BPI -----
    bpi_matched = 0
    for b in bpi:
        bname = normalize(b.get('team_name', b.get('team', '')))
        if bname in teams:
            teams[bname]['bpi'] = b.get('bpi')
            teams[bname]['bpi_off'] = b.get('bpi_offense')
            teams[bname]['bpi_def'] = b.get('bpi_defense')
            teams[bname]['bpi_rank'] = b.get('rank')
            bpi_matched += 1

    # ----- Merge NET -----
    net_matched = 0
    for n in net:
        nname = normalize(n['team'])
        if nname in teams:
            teams[nname]['net_rank'] = n['net_rank']
            net_matched += 1

    # ----- Sort and save -----
    team_list = sorted(teams.values(), key=lambda x: x['torvik_rank'])

    out_path = os.path.join(PROJECT_ROOT, 'client', 'src', 'data', 'teams.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(team_list, f, indent=2)

    # ----- Report -----
    print(f"Torvik teams:  {len(torvik)}")
    print(f"BPI matched:   {bpi_matched}/{len(bpi)}")
    print(f"NET matched:   {net_matched}/{len(net)}")
    print(f"Output:        {len(team_list)} teams → {out_path}")

    missing_bpi = [t['name'] for t in team_list if t['bpi'] is None]
    missing_net = [t['name'] for t in team_list if t['net_rank'] is None]
    if missing_bpi:
        print(f"\nMissing BPI ({len(missing_bpi)}): {missing_bpi}")
    if missing_net:
        print(f"Missing NET ({len(missing_net)}): {missing_net}")


if __name__ == '__main__':
    main()
