from beamngpy import BeamNGpy, set_up_simple_logging

def main():
    print("1. Script started", flush=True)
    set_up_simple_logging()

    bng = BeamNGpy(
        "localhost",
        25252,
        home=r"C:\Users\Jango\BeamNG.tech.v0.38.3.0"
    )

    try:
        print("2. Launching BeamNG.tech via BeamNGpy...", flush=True)
        bng.open()
        print("3. Connected successfully!", flush=True)

        info = bng.system.get_info()
        print("4. System info:", flush=True)
        print(info)

    except Exception as e:
        print("ERROR:", repr(e))

    finally:
        try:
            bng.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()