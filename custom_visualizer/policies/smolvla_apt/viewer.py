from custom_visualizer.policies.smolvla_apt import SmolVLAAptAdapter
from custom_visualizer.web import create_app as create_flask_app, run_app


def create_app():
    return create_flask_app(SmolVLAAptAdapter())


def main() -> None:
    run_app(SmolVLAAptAdapter())


if __name__ == "__main__":
    main()
