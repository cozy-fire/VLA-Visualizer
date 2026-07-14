from custom_visualizer.gui import AttentionViewer
from custom_visualizer.policies.smolvla import SmolVLAAdapter


def main() -> None:
    import tkinter as tk
    root = tk.Tk()
    AttentionViewer(root, SmolVLAAdapter())
    root.mainloop()


if __name__ == "__main__":
    main()