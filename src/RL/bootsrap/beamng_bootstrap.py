from __future__ import annotations

from pathlib import Path

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging


BEAMNG_HOME = Path(r"C:\Users\Jango\BeamNG.tech.v0.38.3.0")
HOST = "localhost"
PORT = 64256

SPAWN_POS = (-406.8015137, 257.4653625, 25.0089035)
# SPAWN_ROT  = (-0.0001524259429, 0.0005056571858, 0.9574455164, 0.2886135897)
SPAWN_ROT = (-0.0001524259429, 0.0005056571858, -0.2886135897, 0.9574455164)


def main() -> None:
    set_up_simple_logging()

    bng = BeamNGpy(HOST, PORT, home=str(BEAMNG_HOME))
    bng.open(launch=True)

    scenario = Scenario(
        "hirochi_raceway",
        "sbr4_bootstrap",
        description="SBR4 Track bootstrap scenario",
    )

    vehicle = Vehicle(
        "ego_vehicle",
        model="sbr",
        part_config="vehicles/sbr/track.pc",
        license="JANGO",
        color="Blue",
    )

    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    scenario.make(bng)

    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()

    print("Ready.")
    input("Press Enter to exit...")
    bng.disconnect()


if __name__ == "__main__":
    main()