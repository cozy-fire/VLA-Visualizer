from custom_visualizer.policies.smolvla import SmolVLAAdapter
from custom_visualizer.web import create_app as create_flask_app, run_app


def create_app():
    return create_flask_app(SmolVLAAdapter())


def main() -> None:
    run_app(SmolVLAAdapter())


if __name__ == "__main__":
    main()