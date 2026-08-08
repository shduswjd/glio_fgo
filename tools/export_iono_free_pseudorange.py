"""Export GPS L1/L2 ionosphere-free pseudoranges from a RINEX 3 file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensors.gnss_raw import carrier_frequency_hz, read_rinex_observations


def export(obs: Path, output: Path, primary: str, secondary: str) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = measurements = 0
    f1 = carrier_frequency_hz("G01", primary)
    f2 = carrier_frequency_hz("G01", secondary)
    a = f1**2 / (f1**2 - f2**2)
    b = -f2**2 / (f1**2 - f2**2)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "gps_week", "gps_sow", "iso_time", "satellite",
            f"C{primary}_m", f"C{secondary}_m", "iono_free_pseudorange_m",
            "primary_coefficient", "secondary_coefficient",
        ])
        for epoch in read_rinex_observations(obs):
            epochs += 1
            for observation in epoch.satellites:
                if not observation.satellite.startswith("G"):
                    continue
                p1 = observation.pseudorange(primary)
                p2 = observation.pseudorange(secondary)
                p_if = observation.ionosphere_free_pseudorange(primary, secondary)
                if p1 is None or p2 is None or p_if is None:
                    continue
                writer.writerow([
                    epoch.gps_week, f"{epoch.gps_seconds_of_week:.3f}",
                    epoch.time.isoformat(), observation.satellite,
                    f"{p1:.3f}", f"{p2:.3f}", f"{p_if:.3f}",
                    f"{a:.12f}", f"{b:.12f}",
                ])
                measurements += 1
    return epochs, measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", default="1C")
    parser.add_argument("--secondary", default="2W")
    args = parser.parse_args()
    epochs, measurements = export(args.obs, args.output, args.primary, args.secondary)
    print(f"epochs={epochs} measurements={measurements} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
